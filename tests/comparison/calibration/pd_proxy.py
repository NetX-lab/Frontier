#!/usr/bin/env python3
"""Proxy of the 1P1D vLLM ground truth: prefill on one instance, then decode on the other.

Each client request is sent twice under one proxied request id that names the
P2pNccl addresses of both instances: first to the prefill instance with
``max_tokens = min_tokens = 1``, then, unchanged, to the decode instance, whose
response is returned to the client. This is the flow of vLLM's
``disagg_proxy_p2p_nccl_xpyd.py`` with three changes (calibration decision Q4):

- the client's ``X-Request-Id`` replaces the random suffix of the proxied id,
  so both instances name the engine request ``cmpl-<proxied id>-0`` and every
  instance row maps back to one trace row;
- the prefill copy also sets ``min_tokens = 1``, since vLLM rejects
  ``min_tokens > max_tokens`` and the replay sends ``min_tokens = max_tokens``;
- a prefill response other than 200, or a failed connection to either
  instance, is logged as an error and returned to the client as a 502, instead
  of being skipped.

When a request finishes, one line is appended to ``--log-path`` with the proxy
arrival and the prefill and decode response times, each on ``time.time()`` and
``time.monotonic()``. The routes are ``POST /v1/completions`` and
``GET /health``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("pd_proxy")


def proxied_request_id(client_request_id: str, prefill_kv_address: str, decode_kv_address: str) -> str:
    """Return the request id that both instances receive for one client request.

    P2pNcclConnector.parse_request_id finds the decode address with the greedy
    pattern ``___decode_addr_(.*):(\\d+)`` and the prefill address with
    ``___prefill_addr_(.*):(\\d+)___``. A ':' in the client id, or a leading
    '_', would move either match into the client id.
    """
    if ":" in client_request_id or client_request_id.startswith("_"):
        raise ValueError(
            f"client request id {client_request_id!r} contains ':' or starts with '_'"
        )
    return (
        f"___prefill_addr_{prefill_kv_address}___decode_addr_{decode_kv_address}"
        f"_{client_request_id}"
    )


def build_app(prefill_url: str, decode_url: str, prefill_kv_address: str,
              decode_kv_address: str, log_path: Path):
    from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

    session = None
    log = log_path.open("a", buffering=1)

    async def client_session(app):
        nonlocal session
        session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0))
        yield
        await session.close()
        log.close()

    async def forward(url: str, body: bytes, request_id: str) -> tuple[int, bytes, str]:
        async with session.post(
            url, data=body,
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        ) as response:
            return response.status, await response.read(), response.content_type

    async def completions(request):
        arrival_time, arrival_monotonic = time.time(), time.monotonic()
        client_request_id = request.headers["X-Request-Id"]
        request_id = proxied_request_id(client_request_id, prefill_kv_address, decode_kv_address)
        record = {
            "request_id": client_request_id,
            "proxied_request_id": request_id,
            "proxy_arrival_time": arrival_time,
            "proxy_arrival_monotonic": arrival_monotonic,
        }
        body = await request.read()
        prefill_body = json.dumps(json.loads(body) | {"max_tokens": 1, "min_tokens": 1}).encode()
        try:
            status, payload, _ = await forward(f"{prefill_url}/v1/completions", prefill_body, request_id)
            record.update(prefill_status=status, prefill_response_time=time.time(),
                          prefill_response_monotonic=time.monotonic())
            if status != 200:
                record["error"] = payload.decode(errors="replace")
                logger.error("prefill of %s returned %d: %s", client_request_id, status, record["error"])
                return web.json_response(
                    {"error": "prefill failed", "prefill_status": status, "prefill_body": record["error"]},
                    status=502,
                )
            status, payload, content_type = await forward(f"{decode_url}/v1/completions", body, request_id)
            record.update(decode_status=status, decode_response_time=time.time(),
                          decode_response_monotonic=time.monotonic())
            if status != 200:
                logger.error("decode of %s returned %d", client_request_id, status)
            return web.Response(body=payload, status=status, content_type=content_type)
        except ClientError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("request %s failed: %s", client_request_id, record["error"])
            return web.json_response({"error": record["error"]}, status=502)
        finally:
            log.write(json.dumps(record) + "\n")

    async def health(request):
        return web.Response(text="ok")

    app = web.Application()
    app.cleanup_ctx.append(client_session)
    app.add_routes([web.post("/v1/completions", completions), web.get("/health", health)])
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True, help="http://<host>:<port> of the prefill instance")
    parser.add_argument("--decode-url", required=True, help="http://<host>:<port> of the decode instance")
    parser.add_argument("--prefill-kv-address", required=True, help="<kv_ip>:<kv_port> of the prefill instance")
    parser.add_argument("--decode-kv-address", required=True, help="<kv_ip>:<kv_port> of the decode instance")
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args(argv)

    from aiohttp import web

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = build_app(args.prefill_url, args.decode_url, args.prefill_kv_address,
                    args.decode_kv_address, args.log_path)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
