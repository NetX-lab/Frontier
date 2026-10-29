"""Actual HTTP proxy evidence, without native jobs or transport-policy changes."""

import asyncio
import json

import pytest

from tests.comparison.calibration import pd_proxy


PARAMS = {"transfer_mode": "pull", "do_remote_prefill": True, "do_remote_decode": False,
          "remote_engine_id": "P", "remote_request_id": "native-P", "remote_host": "127.0.0.1",
          "remote_port": 5600, "remote_block_ids": [[1]], "remote_num_tokens": 6128}
BODY = {"model": "dp_pp_case", "prompt": [17, 23, 42], "stream": True,
        "max_tokens": 3, "min_tokens": 3, "return_token_ids": True,
        "stream_options": {"include_usage": True}}
SSE = b'data: {"choices":[{"token_ids":[7]}]}\n\ndata: [DONE]\n\n'


async def exchange(tmp_path, *, eof=False, eof_on_reuse=False, repeat=1, version="step5", stream=True,
                   prefill_eof=False, invalid_metadata=False):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    received = []
    connections = []

    async def prefill(request):
        received.append(("P", request.path, request.headers["X-Request-Id"], await request.json()))
        assert request.content_type == "application/json" and "Authorization" not in request.headers
        if prefill_eof:
            request.transport.close()
        return web.json_response({"kv_transfer_params": None if invalid_metadata else PARAMS})

    async def decode(reader, writer):
        index = len(connections)
        connections.append([])
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    break
                headers = dict(line.decode().split(": ", 1) for line in head.split(b"\r\n")[1:] if b": " in line)
                body = json.loads(await reader.readexactly(int(headers["Content-Length"])))
                path = head.split(b" ")[1].decode()
                received.append(("D", path, headers["X-Request-Id"], body))
                assert headers["Content-Type"] == "application/json" and "Authorization" not in headers
                connections[index].append(headers["X-Request-Id"])
                if eof or (eof_on_reuse and len(connections[index]) > 1):
                    break
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                             + str(len(SSE)).encode() + b"\r\n\r\n" + SSE)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    raw = await asyncio.start_server(decode, "127.0.0.1", 0)
    p = web.Application()
    p.router.add_post("/v1/completions", prefill)
    try:
        async with TestServer(p) as ps:
            app = pd_proxy.build_app(str(ps.make_url("/")).rstrip("/"),
                "http://127.0.0.1:" + str(raw.sockets[0].getsockname()[1]), "127.0.0.1:21000", "127.0.0.1:22000",
                tmp_path / "proxy.jsonl", vllm_version=version)
            async with TestClient(TestServer(app)) as client:
                results = []
                for index in range(repeat):
                    response = await client.post("/v1/completions", json=BODY | {"stream": stream},
                                                 headers={"X-Request-Id": f"local-{index}", "Authorization": SECRET})
                    results.append((response.status, await response.read()))
    finally:
        raw.close()
        await raw.wait_closed()
    records = [json.loads(line) for line in (tmp_path / "proxy.jsonl").read_text().splitlines()]
    assert sum(row[0] == "P" for row in received) == repeat
    no_decode = prefill_eof or invalid_metadata
    assert sum(row[0] == "D" for row in received) == (0 if no_decode else repeat)
    if no_decode:
        return results, records, connections
    for index in range(repeat):
        p_row, d_row = received[index * 2:index * 2 + 2]
        expected_id = (f"local-{index}" if pd_proxy.uses_nixl_pull(version) else
                       pd_proxy.proxied_request_id(f"local-{index}", "127.0.0.1:21000", "127.0.0.1:22000"))
        assert p_row[1:3] == d_row[1:3] == ("/v1/completions", expected_id)
        assert p_row[3]["prompt"] == BODY["prompt"] and p_row[3]["stream"] is False
        assert p_row[3]["max_tokens"] == 1
        expected_body = BODY | {"stream": stream}
        assert d_row[3] == (expected_body | {"kv_transfer_params": PARAMS} if pd_proxy.uses_nixl_pull(version) else expected_body)
        if pd_proxy.uses_nixl_pull(version):
            assert "min_tokens" not in p_row[3]
    return results, records, connections


@pytest.mark.parametrize("stream", [True, False])
def test_actual_proxy_eof_preserves_502_and_records_decode_header_failure(tmp_path, stream):
    results, records, connections = asyncio.run(exchange(tmp_path, eof=True, stream=stream))
    assert results == [(502, b'{"error": "ServerDisconnectedError: Server disconnected"}')]
    assert connections == [["local-0"]]
    row = records[0]
    assert row["prefill_status"] == 200 and "decode_status" not in row
    capture = row["http_capture"]
    assert capture["failure"]["phase"] == "decode"
    assert capture["failure"]["boundary"] == "response_headers"
    decode = capture["attempts"]["decode"]
    assert row["prefill_response_monotonic"] <= decode["submit_monotonic"] <= capture["failure"]["monotonic"]
    assert decode["connection"] == "fresh"
    assert decode["socket_identity"] == "UNAVAILABLE_PUBLIC_TRACE"
    assert capture["failure"]["exceptions"][0]["type"] == "ServerDisconnectedError"
    assert capture["failure"]["exceptions"][0]["frames"]


@pytest.mark.parametrize("fault", ["prefill_eof", "invalid_metadata"])
def test_actual_proxy_separates_prefill_headers_and_metadata_failures(tmp_path, fault):
    results, records, connections = asyncio.run(exchange(tmp_path, **{fault: True}))
    assert results[0][0] == 502 and connections == []
    capture = records[0]["http_capture"]
    failure = capture["failure"]
    assert failure["phase"] == ("prefill" if fault == "prefill_eof" else "handoff")
    assert failure["boundary"] == ("response_headers" if fault == "prefill_eof" else "metadata_validation")
    assert "decode" not in capture["attempts"]


def test_public_trace_does_not_copy_headers_tokens_or_callback_parameters():
    from types import SimpleNamespace

    record = {}
    capture = pd_proxy.HttpCapture(record)
    capture.submit("decode")
    config = pd_proxy._http_trace_config()
    params = SimpleNamespace(headers={"Authorization": SECRET}, url=SECRET, chunk=SECRET,
                             exception=ValueError(SECRET), prompt=BODY["prompt"])
    context = SimpleNamespace(trace_request_ctx=(capture, "decode"))
    asyncio.run(config.on_request_headers_sent[0](None, context, params))
    assert SECRET not in json.dumps(record) and str(BODY["prompt"]) not in json.dumps(record)
    event = record["http_capture"]["attempts"]["decode"]["events"][0]
    assert set(event) == {"event", "monotonic"} and event["event"] == "request_headers_sent"


@pytest.mark.parametrize("stream", [True, False])
def test_actual_proxy_healthy_records_fresh_and_reused_without_policy_change(tmp_path, stream):
    results, records, connections = asyncio.run(exchange(tmp_path, repeat=2, stream=stream))
    assert results == [(200, SSE), (200, SSE)]
    assert connections == [["local-0", "local-1"]]
    assert [row["http_capture"]["attempts"]["decode"]["connection"] for row in records] == ["fresh", "reused"]
    assert [row["http_capture"]["attempts"]["prefill"]["connection"] for row in records] == ["fresh", "reused"]
    assert all("failure" not in row["http_capture"] for row in records)


def test_actual_proxy_reused_connection_eof_records_the_failed_attempt(tmp_path):
    results, records, connections = asyncio.run(exchange(tmp_path, eof_on_reuse=True, repeat=2))
    assert results == [(200, SSE), (502, b'{"error": "ServerDisconnectedError: Server disconnected"}')]
    assert connections == [["local-0", "local-1"]]
    capture = records[1]["http_capture"]
    assert capture["attempts"]["decode"]["connection"] == "reused"
    assert capture["failure"]["phase"] == "decode" and capture["failure"]["boundary"] == "response_headers"
    assert capture["failure"]["exceptions"][0]["type"] == "ServerDisconnectedError"


@pytest.mark.parametrize("version", ["0.10.2", "0.28.0"])
def test_other_adapter_http_receipts_do_not_gain_step_capture(tmp_path, version):
    _, records, _ = asyncio.run(exchange(tmp_path, version=version))
    assert "http_capture" not in records[0]


SECRET = "Bearer PRIVATE_AUTH; tokens=[999999777]; /private/home/secrets; environment=PRIVATE_ENV"


def test_exception_chain_has_symbolic_frames_not_messages_paths_code_or_locals():
    record = {}
    capture = pd_proxy.HttpCapture(record)
    capture.submit("decode")
    try:
        try:
            private_local = SECRET
            raise ValueError(private_local)
        except ValueError as exc:
            raise RuntimeError(SECRET) from exc
    except RuntimeError as exc:
        capture.failure(exc)
    failure = record["http_capture"]["failure"]
    assert [row["type"] for row in failure["exceptions"]] == ["RuntimeError", "ValueError"]
    assert [row["next"] for row in failure["exceptions"]] == ["cause", None]
    assert all(row["message"] == "[REDACTED]" for row in failure["exceptions"])
    assert SECRET not in json.dumps(record) and "PRIVATE_" not in json.dumps(record)
    for row in failure["exceptions"]:
        assert row["frames"]
        assert all(set(frame) == {"module", "function", "line"} for frame in row["frames"])


def test_trace_and_exception_evidence_limits():
    record = {}
    capture = pd_proxy.HttpCapture(record)
    capture.submit("decode")
    for _ in range(pd_proxy.HTTP_TRACE_EVENT_LIMIT + 2):
        capture.trace("decode", "request_start")
    assert len(record["http_capture"]["attempts"]["decode"]["events"]) == pd_proxy.HTTP_TRACE_EVENT_LIMIT
    assert record["http_capture"]["attempts"]["decode"]["events_truncated"]
    error = ValueError(SECRET)
    for _ in range(pd_proxy.HTTP_EXCEPTION_LIMIT + 2):
        parent = RuntimeError(SECRET)
        parent.__cause__ = error
        error = parent
    capture.failure(error)
    failure = record["http_capture"]["failure"]
    assert len(failure["exceptions"]) == pd_proxy.HTTP_EXCEPTION_LIMIT and failure["chain_truncated"]

    def nested(depth):
        if depth:
            nested(depth - 1)
        else:
            raise RuntimeError(SECRET)
    try:
        nested(pd_proxy.HTTP_TRACEBACK_FRAME_LIMIT + 2)
    except RuntimeError as exc:
        capture.failure(exc)
    row = record["http_capture"]["failure"]["exceptions"][0]
    assert len(row["frames"]) == pd_proxy.HTTP_TRACEBACK_FRAME_LIMIT and row["frames_truncated"]


@pytest.mark.parametrize("eof", [False, True])
def test_trace_callback_failure_is_soft_for_healthy_and_native_failure(tmp_path, monkeypatch, caplog, eof):
    def broken_capture(*args):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(pd_proxy.HttpCapture, "trace", broken_capture)
    results, records, _ = asyncio.run(exchange(tmp_path, eof=eof))
    assert results == ([(502, b'{"error": "ServerDisconnectedError: Server disconnected"}')] if eof else [(200, SSE)])
    capture = records[0]["http_capture"]
    assert capture["status"] == "UNVERIFIED" and capture["capture_errors"]
    assert len(capture["capture_errors"]) <= pd_proxy.HTTP_CAPTURE_ERROR_LIMIT
    assert all(row["type"] == "RuntimeError" for row in capture["capture_errors"])
    assert "PRIVATE_" not in json.dumps(capture) and "PRIVATE_" not in caplog.text
    if eof:
        assert records[0]["error"] == "ServerDisconnectedError: Server disconnected"
        assert capture["failure"]["exceptions"][0]["type"] == "ServerDisconnectedError"


def test_exception_capture_failure_does_not_replace_native_error(tmp_path, monkeypatch):
    def broken_chain(exc):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(pd_proxy, "_http_exception_chain", broken_chain)
    results, records, _ = asyncio.run(exchange(tmp_path, eof=True))
    assert results == [(502, b'{"error": "ServerDisconnectedError: Server disconnected"}')]
    assert records[0]["error"] == "ServerDisconnectedError: Server disconnected"
    capture = records[0]["http_capture"]
    assert capture["status"] == "UNVERIFIED" and "failure" not in capture
    assert capture["capture_errors"][0] == {"operation": "failure", "type": "RuntimeError"}


def test_trace_setup_failure_does_not_prevent_healthy_request(tmp_path, monkeypatch):
    def broken_setup():
        raise RuntimeError(SECRET)
    monkeypatch.setattr(pd_proxy, "_http_trace_config", broken_setup)
    results, records, _ = asyncio.run(exchange(tmp_path))
    assert results == [(200, SSE)]
    capture = records[0]["http_capture"]
    assert capture["status"] == "UNVERIFIED"
    assert capture["capture_errors"][0] == {"operation": "trace_setup", "type": "RuntimeError"}


def test_capture_initialization_failure_does_not_prevent_healthy_request(tmp_path, monkeypatch):
    def broken_init(self, record):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(pd_proxy.HttpCapture, "__init__", broken_init)
    results, records, _ = asyncio.run(exchange(tmp_path))
    assert results == [(200, SSE)]
    capture = records[0]["http_capture"]
    assert capture["status"] == "UNVERIFIED"
    assert capture["capture_errors"][0] == {"operation": "initialization", "type": "RuntimeError"}
