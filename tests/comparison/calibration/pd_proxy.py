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
``GET /health``. The explicit v0.28 adapter follows the pinned NIXL pull toy
proxy: P returns the native ``kv_transfer_params`` passed unchanged to D
under the same client header. Streaming clients receive D's chunks immediately;
P still uses a non-streaming response to supply the handoff metadata. Proxy byte
timestamps are diagnostics, not client token timestamps. These records do not prove native READ
completion or device readiness.
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
              decode_kv_address: str, log_path: Path, *, vllm_version: str = "0.10.2"):
    if vllm_version not in ("0.10.2", "0.28.0"):
        raise ValueError(f"Unsupported proxy vLLM version: {vllm_version!r}")
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
        request_id = (client_request_id if vllm_version == "0.28.0" else
                      proxied_request_id(client_request_id, prefill_kv_address, decode_kv_address))
        record = {
            "request_id": client_request_id,
            "proxied_request_id": request_id,
            "proxy_arrival_time": arrival_time,
            "proxy_arrival_monotonic": arrival_monotonic,
        }
        body = await request.read()
        stream_response = None
        try:
            if vllm_version == "0.28.0":
                record["handoff_protocol"] = "nixl_pull"
                original = json.loads(body)
                if (type(original) is not dict or type(original.get("stream")) is not bool
                        or "kv_transfer_params" in original
                        or type(original.get("prompt")) is not list or not original["prompt"]):
                    raise ValueError("NIXL replay requires token prompts, an explicit stream boolean, and no client transfer metadata")
                prefill = original | {"stream": False, "max_tokens": 1,
                    "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                        "remote_engine_id": None, "remote_block_ids": None,
                        "remote_host": None, "remote_port": None}}
                prefill.pop("min_tokens", None)
                prefill.pop("min_completion_tokens", None)
                prefill.pop("stream_options", None)
                if "max_completion_tokens" in prefill:
                    prefill["max_completion_tokens"] = 1
                prefill_body = json.dumps(prefill).encode()
            else:
                prefill_body = json.dumps(json.loads(body) | {"max_tokens": 1, "min_tokens": 1}).encode()
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
            if vllm_version == "0.28.0":
                response = json.loads(payload)
                params = response.get("kv_transfer_params") if type(response) is dict else None
                if (type(params) is not dict or params.get("transfer_mode") != "pull"
                        or params.get("do_remote_prefill") is not True
                        or params.get("do_remote_decode") is not False
                        or any(type(params.get(key)) is not str or not params[key] for key in
                               ("remote_engine_id", "remote_request_id", "remote_host"))
                        or type(params.get("remote_port")) is not int
                        or not 0 < params["remote_port"] < 65536
                        or type(params.get("remote_block_ids")) is not list
                        or not params["remote_block_ids"]
                        or any(type(group) is not list
                               or any(type(block) is not int or block < 0 for block in group)
                               for group in params["remote_block_ids"])):
                    raise ValueError("Prefill response lacks valid native NIXL pull metadata")
                body = json.dumps(original | {"kv_transfer_params": params}).encode()
                record["kv_transfer_params"] = params
                if original["stream"]:
                    async with session.post(
                        f"{decode_url}/v1/completions", data=body,
                        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
                    ) as response:
                        record["decode_status"] = response.status
                        if response.status != 200:
                            payload = await response.read()
                            record.update(decode_response_time=time.time(),
                                          decode_response_monotonic=time.monotonic())
                            return web.Response(body=payload, status=response.status,
                                                content_type=response.content_type)
                        stream_response = web.StreamResponse(
                            headers={"Content-Type": response.headers["Content-Type"]})
                        await stream_response.prepare(request)
                        async for chunk in response.content.iter_any():
                            if "decode_first_byte_monotonic" not in record:
                                record.update(decode_first_byte_time=time.time(),
                                              decode_first_byte_monotonic=time.monotonic())
                            await stream_response.write(chunk)
                        await stream_response.write_eof()
                        record.update(decode_response_time=time.time(),
                                      decode_response_monotonic=time.monotonic())
                        return stream_response
            status, payload, content_type = await forward(f"{decode_url}/v1/completions", body, request_id)
            record.update(decode_status=status, decode_response_time=time.time(),
                          decode_response_monotonic=time.monotonic())
            if status != 200:
                logger.error("decode of %s returned %d", client_request_id, status)
            return web.Response(body=payload, status=status, content_type=content_type)
        except (ClientError, ValueError, ConnectionError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("request %s failed: %s", client_request_id, record["error"])
            if stream_response is not None and stream_response.prepared:
                stream_response.force_close()
                raise
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
    parser.add_argument("--vllm-version", choices=("0.10.2", "0.28.0"), default="0.10.2")
    parser.add_argument("--prefill-kv-address", default="", help="legacy <kv_ip>:<kv_port>")
    parser.add_argument("--decode-kv-address", default="", help="legacy <kv_ip>:<kv_port>")
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.vllm_version == "0.10.2" and not (args.prefill_kv_address and args.decode_kv_address):
        parser.error("Legacy P2P proxy requires both KV addresses")

    from aiohttp import web

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = build_app(args.prefill_url, args.decode_url, args.prefill_kv_address,
                    args.decode_kv_address, args.log_path, vllm_version=args.vllm_version)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
