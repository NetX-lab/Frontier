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
``GET /health``. The NIXL pull adapters (v0.28 and Step-5) follow the pinned native toy
proxy: P returns the native ``kv_transfer_params`` passed unchanged to D
under the same client header. These HTTP records do not prove native READ
completion or device readiness. An opt-in streaming client still prefills nonstreaming;
the proxy relays D's SSE bytes immediately, without parsing or rewriting them.
Step receipts also include bounded public aiohttp connection-selection events and
redacted exception frames on the proxy monotonic clock. These do not identify a
TCP socket or prove the cause of a native disconnect; capture errors are soft.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from functools import wraps
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parent.parent / "dp_placement_pp")]
from vllm_replay import RUNTIME_ADAPTERS, uses_nixl_pull  # noqa: E402

logger = logging.getLogger("pd_proxy")

HTTP_TRACE_EVENT_LIMIT = 16
HTTP_EXCEPTION_LIMIT = 4
HTTP_TRACEBACK_FRAME_LIMIT = 12
HTTP_SYMBOL_LIMIT = 128
HTTP_CAPTURE_ERROR_LIMIT = 4


def _http_capture_soft_gate(method):
    """Auxiliary evidence cannot change the HTTP operation it describes."""
    @wraps(method)
    def isolated(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.unverified(method.__name__, exc)
    return isolated


def _http_exception_chain(exc):
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < HTTP_EXCEPTION_LIMIT:
        seen.add(id(exc))
        frames, tb = [], exc.__traceback__
        while tb is not None and len(frames) < HTTP_TRACEBACK_FRAME_LIMIT:
            frames.append({"module": tb.tb_frame.f_globals.get("__name__", "<unknown>")[:HTTP_SYMBOL_LIMIT],
                           "function": tb.tb_frame.f_code.co_name[:HTTP_SYMBOL_LIMIT], "line": tb.tb_lineno})
            tb = tb.tb_next
        cause = exc.__cause__
        context = exc.__context__ if not exc.__suppress_context__ else None
        chain.append({"type": type(exc).__name__[:HTTP_SYMBOL_LIMIT], "message": "[REDACTED]",
                      "frames": frames, "frames_truncated": tb is not None,
                      "next": "cause" if cause is not None else "context" if context is not None else None})
        exc = cause if cause is not None else context
    return chain, exc is not None


class HttpCapture:
    """Step-local public trace evidence; no request data or TCP socket claims."""

    def __init__(self, record):
        self.data = {"schema": "step_http_capture_v1", "status": "RECORDED", "clock": "proxy host time.monotonic",
                     "attempts": {}}
        self.phase, self.boundary = "request", "body_json"
        record["http_capture"] = self.data

    def unverified(self, operation, exc):
        # Reporting a broken auxiliary recorder must itself remain non-blocking.
        notify = False
        try:
            notify = self.data.get("status") != "UNVERIFIED"
            errors = self.data.setdefault("capture_errors", [])
            if len(errors) < HTTP_CAPTURE_ERROR_LIMIT:
                errors.append({"operation": operation, "type": type(exc).__name__[:HTTP_SYMBOL_LIMIT]})
            self.data["status"] = "UNVERIFIED"
        except Exception:
            pass
        try:
            if notify:
                logger.warning("Step HTTP capture UNVERIFIED in %s (%s); HTTP outcome remains authoritative",
                               operation, type(exc).__name__[:HTTP_SYMBOL_LIMIT])
        except Exception:
            pass

    @_http_capture_soft_gate
    def at(self, phase, boundary):
        self.phase, self.boundary = phase, boundary

    @_http_capture_soft_gate
    def submit(self, phase):
        self.phase, self.boundary = phase, "response_headers"
        self.data["attempts"][phase] = {"submit_monotonic": time.monotonic(), "connection": "unknown",
            "socket_identity": "UNAVAILABLE_PUBLIC_TRACE", "events": [], "events_truncated": False}

    @_http_capture_soft_gate
    def headers(self, phase):
        self.boundary = "response_body"
        self.data["attempts"][phase]["response_headers_monotonic"] = time.monotonic()

    @_http_capture_soft_gate
    def trace(self, phase, event):
        attempt = self.data["attempts"][phase]
        if event == "connection_create_start":
            attempt["connection"] = "fresh_connecting"
        elif event == "connection_create_end":
            attempt["connection"] = "fresh"
        elif event == "connection_reuseconn":
            attempt["connection"] = "reused"
        if len(attempt["events"]) < HTTP_TRACE_EVENT_LIMIT:
            attempt["events"].append({"event": event, "monotonic": time.monotonic()})
        else:
            attempt["events_truncated"] = True

    @_http_capture_soft_gate
    def failure(self, exc):
        timestamp = time.monotonic()
        chain, truncated = _http_exception_chain(exc)
        self.data["failure"] = {"phase": self.phase, "boundary": self.boundary, "monotonic": timestamp,
                                "exceptions": chain, "chain_truncated": truncated}


def _http_capture(record):
    try:
        return HttpCapture(record)
    except Exception as exc:
        record["http_capture"] = {"schema": "step_http_capture_v1", "status": "UNVERIFIED",
            "capture_errors": [{"operation": "initialization", "type": type(exc).__name__[:HTTP_SYMBOL_LIMIT]}]}
        try:
            logger.warning("Step HTTP capture UNVERIFIED in initialization (%s); HTTP outcome remains authoritative",
                           type(exc).__name__[:HTTP_SYMBOL_LIMIT])
        except Exception:
            pass
        return None


def _http_trace_config():
    from aiohttp import TraceConfig

    config = TraceConfig()
    for event in ("request_start", "connection_create_start", "connection_create_end",
                  "connection_reuseconn", "request_headers_sent", "request_end", "request_exception"):
        async def callback(session, context, params, event=event):
            capture = None
            try:
                capture, phase = context.trace_request_ctx
                capture.trace(phase, event)
            except Exception as exc:
                if capture is not None:
                    capture.unverified("trace_callback", exc)
        getattr(config, "on_" + event).append(callback)
    return config


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
    pull = uses_nixl_pull(vllm_version)
    capture_http = RUNTIME_ADAPTERS[vllm_version] is RUNTIME_ADAPTERS["step5"]
    from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

    session = None
    capture_setup_error = None
    log = log_path.open("a", buffering=1)

    async def client_session(app):
        nonlocal session, capture_setup_error
        trace_configs = []
        if capture_http:
            try:
                trace_configs = [_http_trace_config()]
            except Exception as exc:
                capture_setup_error = exc
        session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0),
                                trace_configs=trace_configs)
        yield
        await session.close()
        log.close()

    async def forward(url: str, body: bytes, request_id: str, capture, phase) -> tuple[int, bytes, str]:
        if capture is not None:
            capture.submit(phase)
        async with session.post(
            url, data=body,
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
            trace_request_ctx=(capture, phase) if capture is not None else None,
        ) as response:
            if capture is not None:
                capture.headers(phase)
            return response.status, await response.read(), response.content_type

    async def completions(request):
        arrival_time, arrival_monotonic = time.time(), time.monotonic()
        client_request_id = request.headers["X-Request-Id"]
        request_id = (client_request_id if pull else
                      proxied_request_id(client_request_id, prefill_kv_address, decode_kv_address))
        record = {
            "request_id": client_request_id,
            "proxied_request_id": request_id,
            "proxy_arrival_time": arrival_time,
            "proxy_arrival_monotonic": arrival_monotonic,
        }
        body = await request.read()
        streamed = None
        capture = _http_capture(record) if capture_http else None
        if capture is not None and capture_setup_error is not None:
            capture.unverified("trace_setup", capture_setup_error)
        try:
            original = json.loads(body)
            if pull:
                record["handoff_protocol"] = "nixl_pull"
                if (type(original) is not dict or type(original.get("stream")) is not bool
                        or "kv_transfer_params" in original
                        or type(original.get("prompt")) is not list or not original["prompt"]):
                    raise ValueError("NIXL replay requires token prompts without client transfer metadata")
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
                prefill = original | {"max_tokens": 1, "min_tokens": 1}
                if original.get("stream"):
                    prefill["stream"] = False
                    prefill.pop("stream_options", None)
                prefill_body = json.dumps(prefill).encode()
            status, payload, _ = await forward(f"{prefill_url}/v1/completions", prefill_body, request_id,
                                               capture, "prefill")
            record.update(prefill_status=status, prefill_response_time=time.time(),
                          prefill_response_monotonic=time.monotonic())
            if status != 200:
                record["error"] = payload.decode(errors="replace")
                logger.error("prefill of %s returned %d: %s", client_request_id, status, record["error"])
                return web.json_response(
                    {"error": "prefill failed", "prefill_status": status, "prefill_body": record["error"]},
                    status=502,
                )
            if pull:
                if capture is not None:
                    capture.at("handoff", "metadata_validation")
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
            cancel_delay = request.headers.get("X-Step5-Cancel-Decode-After-S")
            if cancel_delay is not None:
                if capture is not None:
                    capture.at("decode", "cancellation_control")
                # Diagnostic control only: cancel this proxy's native D client
                # while reading, without fencing or delaying a native READ.
                delay = float(cancel_delay)
                if not pull or not original.get("stream") or not math.isfinite(delay) or delay <= 0:
                    raise ValueError("Decode cancellation requires a positive NIXL streaming deadline")
                record["decode_submit_monotonic"] = time.monotonic()
                try:
                    async with asyncio.timeout(delay):
                        status, payload, content_type = await forward(
                            f"{decode_url}/v1/completions", body, request_id, capture, "decode")
                    record.update(decode_status=status, decode_response_monotonic=time.monotonic(),
                                  decode_response_time=time.time(), decode_cancelled=False)
                    return web.Response(body=payload, status=status, content_type=content_type)
                except TimeoutError as exc:
                    if capture is not None:
                        capture.failure(exc)
                    record.update(decode_cancelled=True, decode_cancel_monotonic=time.monotonic(),
                                  cancel_boundary="native_decode_client_deadline")
                    event = {"step5_decode_cancelled": True}
                    return web.Response(text="data: " + json.dumps(event) + "\n\n",
                                        content_type="text/event-stream")
            if original.get("stream"):
                record["stream"] = True
                if capture is not None:
                    capture.submit("decode")
                async with session.post(
                    f"{decode_url}/v1/completions", data=body,
                    headers={"Content-Type": "application/json", "X-Request-Id": request_id},
                    trace_request_ctx=(capture, "decode") if capture is not None else None,
                ) as response:
                    if capture is not None:
                        capture.headers("decode")
                    record["decode_status"] = response.status
                    streamed = web.StreamResponse(status=response.status, headers={
                        "Content-Type": response.headers["Content-Type"]})
                    if capture is not None:
                        capture.at("client", "response_headers")
                    await streamed.prepare(request)
                    if capture is not None:
                        capture.at("decode", "response_body")
                    async for chunk in response.content.iter_any():
                        if capture is not None:
                            capture.at("client", "stream_write")
                        await streamed.write(chunk)
                        if capture is not None:
                            capture.at("decode", "response_body")
                    record.update(decode_response_time=time.time(),
                                  decode_response_monotonic=time.monotonic())
                    if capture is not None:
                        capture.at("client", "stream_eof")
                    await streamed.write_eof()
                    return streamed
            status, payload, content_type = await forward(f"{decode_url}/v1/completions", body, request_id,
                                                          capture, "decode")
            record.update(decode_status=status, decode_response_time=time.time(),
                          decode_response_monotonic=time.monotonic())
            if status != 200:
                logger.error("decode of %s returned %d", client_request_id, status)
            return web.Response(body=payload, status=status, content_type=content_type)
        except (ClientError, ValueError) as exc:
            if capture is not None:
                capture.failure(exc)
            record["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("request %s failed: %s", client_request_id, record["error"])
            if streamed is not None and streamed.prepared:
                streamed.force_close()
                raise
            return web.json_response({"error": record["error"]}, status=502)
        except BaseException as exc:
            if capture is not None:
                capture.failure(exc)
            raise
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
    parser.add_argument("--vllm-version", choices=tuple(RUNTIME_ADAPTERS), default="0.10.2")
    parser.add_argument("--prefill-kv-address", default="", help="legacy <kv_ip>:<kv_port>")
    parser.add_argument("--decode-kv-address", default="", help="legacy <kv_ip>:<kv_port>")
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if not uses_nixl_pull(args.vllm_version) and not (args.prefill_kv_address and args.decode_kv_address):
        parser.error("Legacy P2P proxy requires both KV addresses")

    from aiohttp import web

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = build_app(args.prefill_url, args.decode_url, args.prefill_kv_address,
                    args.decode_kv_address, args.log_path, vllm_version=args.vllm_version)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
