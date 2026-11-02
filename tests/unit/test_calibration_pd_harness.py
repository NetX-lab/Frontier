"""Inputs and joins the 1P1D ground-truth harness builds.

A request id the connector cannot parse, a wrong connector switch or a wrong
join is only visible after a two-GPU run, so the pieces that decide them are
checked here.
"""

from __future__ import annotations

import json
import re
import asyncio
from collections import Counter
from pathlib import Path

import pytest

from tests.comparison.calibration.e2e_metrics_gap import NORMALIZED_COLUMNS
from tests.comparison.calibration.pd_metrics import engine_request_id, join_pd_rows, write_pd_rows
from tests.comparison.calibration.pd_proxy import proxied_request_id, build_app
from tests.comparison.calibration.pd_replay import (
    connector_check,
    instance_devices,
    instance_env,
    kv_transfer_config,
)
from tests.comparison.calibration import pd_replay
from tests.unit.test_dp_placement_harness import native_engine, native_observer, native_replay_inputs

# P2pNcclConnector.parse_request_id, vLLM-BS ff328f82a
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:514-530.
DECODE_ADDRESS_OF_PREFILL = re.compile(r"___decode_addr_(.*):(\d+)")
PREFILL_ADDRESS_OF_DECODE = re.compile(r"___prefill_addr_(.*):(\d+)___")


def nixl_engine() -> dict:
    return native_engine() | {
        "role_devices": {"prefill": [str(i) for i in range(16)],
                         "decode": [str(i) for i in range(16, 32)]},
        "kv_transfer": {"kv_buffer_device": "cuda", "side_channel_host": "127.0.0.1",
                        "prefill_side_channel_port": 5557, "decode_side_channel_port": 5657},
    }


def test_v028_pd_selects_pull_roles_explicit_devices_and_distinct_side_channels():
    engine = nixl_engine()
    assert kv_transfer_config(engine, "prefill", 8100) == {
        "kv_connector": "NixlConnector", "kv_role": "kv_producer",
        "kv_buffer_device": "cuda", "kv_load_failure_policy": "fail",
    }
    devices = instance_devices({}, engine=engine)
    assert devices == {"prefill": ",".join(map(str, range(16))),
                       "decode": ",".join(map(str, range(16, 32)))}
    observer = native_observer("DECODE")
    env, changed = instance_env("light", engine, Path("/run/decode"), {},
                                devices["decode"], observer=observer, role="decode")
    assert json.loads(env["VLLM_FRONTIER_OBSERVER"]) == observer
    assert changed["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "5657"
    assert changed["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "127.0.0.1"
    assert changed["CUDA_VISIBLE_DEVICES"] == devices["decode"]
    assert "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH" not in env


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_v028_pd_node_uses_its_local_lane_and_reachable_side_channel(role):
    engine = nixl_engine() | {
        "host": "192.0.2.10", "data_parallel_size_local": 1,
        "data_parallel_address": "192.0.2.10", "data_parallel_rpc_port": 24461,
        "role_devices": {role: [str(i) for i in range(8)]},
    }
    engine["kv_transfer"]["side_channel_host"] = "192.0.2.10"
    inherited = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NCCL_IB_HCA": "platform-value"}
    devices = instance_devices(inherited, engine=engine, role=role)
    assert devices == {role: "0,1,2,3,4,5,6,7"}
    env, changed = instance_env("light", engine, Path("/run") / role, inherited,
                                devices[role], observer=native_observer(role.upper()), role=role)
    assert env["NCCL_IB_HCA"] == "platform-value"
    assert changed["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.0.2.10"
    assert changed["VLLM_NIXL_SIDE_CHANNEL_PORT"] == ("5557" if role == "prefill" else "5657")
    assert json.loads(env["VLLM_FRONTIER_OBSERVER"])["expected_ranks"] == list(range(16))


@pytest.mark.parametrize("fault", ["loopback", "unspecified", "multicast", "rpc_window", "http_window"])
def test_v028_pd_node_rejects_unusable_side_channels(fault):
    engine = nixl_engine() | {
        "host": "192.0.2.10", "data_parallel_size_local": 1,
        "data_parallel_address": "192.0.2.10", "data_parallel_rpc_port": 24461,
    }
    engine["kv_transfer"]["side_channel_host"] = {
        "loopback": "127.0.0.1", "unspecified": "0.0.0.0", "multicast": "224.0.0.1",
    }.get(fault, "192.0.2.10")
    if fault == "rpc_window":
        engine["data_parallel_rpc_port"] = 5558
    with pytest.raises(ValueError):
        kv_transfer_config(engine, "prefill", 5558 if fault == "http_window" else 8100)


@pytest.mark.parametrize("fault", [None, "missing", "push", "identity", "blocks", "response_list"])
def test_v028_proxy_passes_only_native_pull_response_metadata_to_decode(tmp_path, fault):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    params = {
        "do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
        "remote_engine_id": "native-p-engine", "remote_request_id": "native-p-request",
        "remote_host": "127.0.0.1", "remote_port": 5557,
        "remote_block_ids": [[7, 9], []], "remote_num_tokens": 65, "tp_size": 8,
    }
    if fault == "push":
        params["transfer_mode"] = "push"
    if fault == "identity":
        params["remote_request_id"] = None
    if fault == "blocks":
        params["remote_block_ids"] = [[True, -1], "invalid-group"]
    observed = []

    async def exercise():
        async def prefill(request):
            observed.append(("P", request.headers["X-Request-Id"], await request.json()))
            payload = ([] if fault == "response_list" else
                       {} if fault == "missing" else {"kv_transfer_params": params})
            return web.json_response(payload)

        async def decode(request):
            observed.append(("D", request.headers["X-Request-Id"], await request.json()))
            return web.json_response({"choices": [{"text": "native response"}]})

        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", prefill)
        d.router.add_post("/v1/completions", decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = build_app(str(ps.make_url("")).rstrip("/"), str(ds.make_url("")).rstrip("/"),
                              "", "", tmp_path / "proxy.jsonl", vllm_version="0.28.0")
            async with TestClient(TestServer(proxy)) as client:
                body = {"model": "dp_pp_case", "prompt": [1, 2, 3], "stream": False,
                        "max_tokens": 3, "min_tokens": 3, "ignore_eos": True}
                response = await client.post("/v1/completions", json=body,
                                             headers={"X-Request-Id": "client-7"})
                assert response.status == (200 if fault is None else 502)
                await response.read()
    asyncio.run(exercise())
    assert observed[0][1] == "client-7"
    pbody = observed[0][2]
    assert pbody["max_tokens"] == 1 and "min_tokens" not in pbody
    assert pbody["kv_transfer_params"] == {
        "do_remote_decode": True, "do_remote_prefill": False,
        "remote_engine_id": None, "remote_block_ids": None,
        "remote_host": None, "remote_port": None,
    }
    if fault is None:
        assert observed[1][0:2] == ("D", "client-7")
        assert observed[1][2]["kv_transfer_params"] == params
        assert observed[1][2]["max_tokens"] == observed[1][2]["min_tokens"] == 3
    else:
        assert len(observed) == 1
    record = json.loads((tmp_path / "proxy.jsonl").read_text())
    assert record["handoff_protocol"] == "nixl_pull"
    assert "kv_ready" not in record and "device_completion" not in record


@pytest.mark.parametrize(("decode_status", "disconnect"), [(200, False), (503, False), (200, True)])
def test_v028_proxy_streams_before_decode_finishes_and_preserves_pull_metadata(tmp_path, decode_status, disconnect):
    from aiohttp import ClientPayloadError, web
    from aiohttp.test_utils import TestClient, TestServer

    params = {
        "do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
        "remote_engine_id": "p-engine", "remote_request_id": "p-request",
        "remote_host": "127.0.0.1", "remote_port": 5557, "remote_block_ids": [[7]],
    }
    observed = []

    async def exercise():
        client_received = asyncio.Event()

        async def prefill(request):
            observed.append(("P", request.headers["X-Request-Id"], await request.json()))
            return web.json_response({"kv_transfer_params": params})

        async def decode(request):
            observed.append(("D", request.headers["X-Request-Id"], await request.json()))
            if decode_status != 200:
                return web.json_response({"error": "native decode failed"}, status=decode_status)
            stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await stream.prepare(request)
            await stream.write(b'data: {"choices":[{"token_ids":[42],"text":""}]}\n\n')
            # The upstream cannot finish until the client has received the first event.
            await asyncio.wait_for(client_received.wait(), timeout=5)
            if disconnect:
                request.transport.close()
                return stream
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            return stream

        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", prefill)
        d.router.add_post("/v1/completions", decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = build_app(str(ps.make_url("")).rstrip("/"), str(ds.make_url("")).rstrip("/"),
                              "", "", tmp_path / "proxy.jsonl", vllm_version="0.28.0")
            async with TestClient(TestServer(proxy)) as client:
                body = {"prompt": [1, 2, 3], "stream": True, "max_tokens": 2, "min_tokens": 2,
                        "stream_options": {"include_usage": True}, "return_token_ids": True}
                response = await client.post("/v1/completions", json=body,
                                             headers={"X-Request-Id": "repeat-1-formal-0"})
                assert response.status == decode_status
                if decode_status == 200:
                    assert response.content_type == "text/event-stream"
                    first = await asyncio.wait_for(response.content.readline(), timeout=2)
                    assert json.loads(first.removeprefix(b"data: "))["choices"][0]["token_ids"] == [42]
                    client_received.set()
                    if disconnect:
                        with pytest.raises(ClientPayloadError):
                            await response.read()
                    else:
                        assert (await response.read()).endswith(b"data: [DONE]\n\n")
                else:
                    assert (await response.json())["error"] == "native decode failed"
                assert observed[1][2] == body | {"kv_transfer_params": params}

    asyncio.run(exercise())
    assert [item[1] for item in observed] == ["repeat-1-formal-0"] * 2
    assert observed[0][2]["stream"] is False
    assert observed[0][2]["max_tokens"] == 1
    assert "min_tokens" not in observed[0][2] and "stream_options" not in observed[0][2]
    record = json.loads((tmp_path / "proxy.jsonl").read_text())
    assert record["decode_status"] == decode_status
    if decode_status == 200 and not disconnect:
        assert record["prefill_response_monotonic"] <= record["decode_first_byte_monotonic"]
        assert record["decode_first_byte_monotonic"] <= record["decode_response_monotonic"]
    if disconnect:
        assert "ClientPayloadError" in record["error"]
        assert "decode_response_monotonic" not in record
    assert "client_ttft" not in record and "device_completion" not in record


def test_legacy_proxy_preserves_address_id_and_unchanged_decode_body(tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    observed = []

    async def exercise():
        async def endpoint(request):
            observed.append((request.path, request.headers["X-Request-Id"], await request.json()))
            return web.json_response({"choices": [{"text": "fixture"}]})

        app = web.Application()
        app.router.add_post("/P/v1/completions", endpoint)
        app.router.add_post("/D/v1/completions", endpoint)
        async with TestServer(app) as server:
            proxy = build_app(str(server.make_url("/P")), str(server.make_url("/D")),
                              "127.0.0.1:21001", "127.0.0.1:22001", tmp_path / "proxy.jsonl")
            async with TestClient(TestServer(proxy)) as client:
                body = {"prompt": [1, 2, 3], "stream": False, "max_tokens": 3, "min_tokens": 3}
                response = await client.post("/v1/completions", json=body,
                                             headers={"X-Request-Id": "client-7"})
                assert response.status == 200
                await response.read()
                assert observed[1][2] == body

    asyncio.run(exercise())
    expected = "___prefill_addr_127.0.0.1:21001___decode_addr_127.0.0.1:22001_client-7"
    assert [entry[1] for entry in observed] == [expected, expected]
    assert observed[0][2]["max_tokens"] == observed[0][2]["min_tokens"] == 1
    assert "kv_transfer_params" not in observed[0][2]


@pytest.mark.parametrize(("vllm_version", "stream"), [
    ("0.10.2", False), ("0.28.0", False), ("0.28.0", True),
])
def test_proxy_uses_independent_upstream_connections_for_consecutive_posts(tmp_path, vllm_version, stream):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    params = {
        "do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
        "remote_engine_id": "p-engine", "remote_request_id": "p-request",
        "remote_host": "127.0.0.1", "remote_port": 5557, "remote_block_ids": [[7]],
    }
    token = b'data: {"choices":[{"token_ids":[42],"text":""}]}\n\n'
    done = b"data: [DONE]\n\n"
    observed, transports = [], []

    async def exercise():
        received = {f"client-{index}": asyncio.Event() for index in range(2)}

        def endpoint(role):
            async def complete(request):
                body = await request.json()
                identity = request.headers["X-Request-Id"]
                # Retain transports so object identities cannot be recycled after close.
                transports.append(request.transport)
                observed.append({"role": role, "request_id": identity, "body": body,
                                 "connection_id": id(request.transport),
                                 "peer": request.transport.get_extra_info("peername")})
                if role == "P":
                    return web.json_response({"kv_transfer_params": params})
                if stream:
                    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                    await response.prepare(request)
                    await response.write(token)
                    await asyncio.wait_for(received[identity].wait(), timeout=5)
                    await response.write(done)
                    await response.write_eof()
                    return response
                return web.json_response({"choices": [{"token_ids": [42], "text": ""}]})
            return complete

        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", endpoint("P"))
        d.router.add_post("/v1/completions", endpoint("D"))
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = build_app(str(ps.make_url("")).rstrip("/"), str(ds.make_url("")).rstrip("/"),
                              "127.0.0.1:21001", "127.0.0.1:22001", tmp_path / "proxy.jsonl",
                              vllm_version=vllm_version)
            async with TestClient(TestServer(proxy)) as client:
                for index in range(2):
                    identity = f"client-{index}"
                    body = {"prompt": [1, 2, 3], "stream": stream, "max_tokens": 2,
                            "min_tokens": 2, "return_token_ids": True}
                    if stream:
                        body["stream_options"] = {"include_usage": True}
                    response = await client.post("/v1/completions", json=body,
                                                 headers={"X-Request-Id": identity})
                    assert response.status == 200
                    if stream:
                        first = await asyncio.wait_for(response.content.readexactly(len(token)), timeout=2)
                        assert first == token
                        received[identity].set()
                        assert first + await response.read() == token + done
                    else:
                        assert (await response.json())["choices"][0]["token_ids"] == [42]
                    assert observed[-1]["body"] == (body | {"kv_transfer_params": params}
                                                   if vllm_version == "0.28.0" else body)

    asyncio.run(exercise())
    (tmp_path / "connections.json").write_text(json.dumps(observed, indent=2) + "\n")
    identities = [f"client-{index}" if vllm_version == "0.28.0" else
                  proxied_request_id(f"client-{index}", "127.0.0.1:21001", "127.0.0.1:22001")
                  for index in range(2)]
    assert [(row["role"], row["request_id"]) for row in observed] == [
        (role, identity) for identity in identities for role in ("P", "D")]
    assert len({id(transport) for transport in transports}) == 4
    records = [json.loads(line) for line in (tmp_path / "proxy.jsonl").read_text().splitlines()]
    assert [record["request_id"] for record in records] == ["client-0", "client-1"]
    for record in records:
        assert record["prefill_status"] == record["decode_status"] == 200
        assert record["proxy_arrival_monotonic"] <= record["prefill_response_monotonic"]
        assert record["prefill_response_monotonic"] <= record["decode_response_monotonic"]
        if stream:
            assert record["prefill_response_monotonic"] <= record["decode_first_byte_monotonic"]
            assert record["decode_first_byte_monotonic"] <= record["decode_response_monotonic"]
        assert "client_ttft" not in record and "device_completion" not in record


@pytest.mark.parametrize("disconnect_at", ["prefill", "decode_headers", "decode_stream"])
def test_proxy_does_not_retry_disconnected_posts_and_next_request_is_independent(tmp_path, disconnect_at):
    from aiohttp import ClientPayloadError, web
    from aiohttp.test_utils import TestClient, TestServer

    params = {
        "do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
        "remote_engine_id": "p-engine", "remote_request_id": "p-request",
        "remote_host": "127.0.0.1", "remote_port": 5557, "remote_block_ids": [[7]],
    }
    token = b'data: {"choices":[{"token_ids":[42],"text":""}]}\n\n'
    observed, transports = [], []

    async def exercise():
        received = asyncio.Event()

        def endpoint(role):
            async def complete(request):
                await request.read()
                identity = request.headers["X-Request-Id"]
                transports.append(request.transport)
                observed.append({"role": role, "request_id": identity,
                                 "connection_id": id(request.transport)})
                if identity == "bad" and ((role == "P" and disconnect_at == "prefill") or
                                          (role == "D" and disconnect_at == "decode_headers")):
                    request.transport.close()
                    return web.Response()
                if role == "P":
                    return web.json_response({"kv_transfer_params": params})
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(token)
                if identity == "bad" and disconnect_at == "decode_stream":
                    await asyncio.wait_for(received.wait(), timeout=5)
                    request.transport.close()
                    return response
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            return complete

        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", endpoint("P"))
        d.router.add_post("/v1/completions", endpoint("D"))
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = build_app(str(ps.make_url("")).rstrip("/"), str(ds.make_url("")).rstrip("/"),
                              "", "", tmp_path / "proxy.jsonl", vllm_version="0.28.0")
            async with TestClient(TestServer(proxy)) as client:
                for identity in ("bad", "good"):
                    response = await client.post("/v1/completions",
                        json={"prompt": [1, 2, 3], "stream": True, "max_tokens": 2, "min_tokens": 2,
                              "return_token_ids": True, "stream_options": {"include_usage": True}},
                        headers={"X-Request-Id": identity})
                    if identity == "bad" and disconnect_at != "decode_stream":
                        assert response.status == 502
                        assert "ServerDisconnectedError" in (await response.json())["error"]
                    elif identity == "bad":
                        assert response.status == 200
                        assert await response.content.readexactly(len(token)) == token
                        received.set()
                        with pytest.raises(ClientPayloadError):
                            await response.read()
                    else:
                        assert response.status == 200
                        assert await response.read() == token + b"data: [DONE]\n\n"

    asyncio.run(exercise())
    (tmp_path / "connections.json").write_text(json.dumps(observed, indent=2) + "\n")
    expected = [("P", "bad")] + ([] if disconnect_at == "prefill" else [("D", "bad")])
    expected += [("P", "good"), ("D", "good")]
    assert Counter((row["role"], row["request_id"]) for row in observed) == Counter(expected)
    assert len({id(transport) for transport in transports}) == len(expected)
    bad, good = [json.loads(line) for line in (tmp_path / "proxy.jsonl").read_text().splitlines()]
    assert [bad["request_id"], good["request_id"]] == ["bad", "good"]
    assert ("ClientPayloadError" if disconnect_at == "decode_stream" else
            "ServerDisconnectedError") in bad["error"]
    assert "decode_response_monotonic" not in bad
    assert good["prefill_status"] == good["decode_status"] == 200 and "error" not in good


@pytest.mark.parametrize("fault", ["role", "nested", "same_output", "alias_output",
                                  "nested_output", "parent_output", "run", "devices", "pp",
                                  "missing_kv", "overlap", "side_http"])
def test_v028_pd_preflights_both_roles_before_any_launch(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(pd_replay.os, "environ", {"PATH": "/usr/bin"})
    engine = nixl_engine()
    observers = {"prefill": native_observer("PREFILL"), "decode": native_observer("DECODE")}
    if fault == "role":
        observers["decode"]["role"] = "PREFILL"
    elif fault == "nested":
        observers["decode"] = None
    elif fault == "same_output":
        observers["decode"]["output_dir"] = observers["prefill"]["output_dir"]
    elif fault == "alias_output":
        observers["decode"]["output_dir"] = observers["prefill"]["output_dir"] + "/child/.."
    elif fault == "nested_output":
        observers["decode"]["output_dir"] = observers["prefill"]["output_dir"] + "/decode"
    elif fault == "parent_output":
        observers["prefill"]["output_dir"] = observers["decode"]["output_dir"] + "/prefill"
    elif fault == "run":
        observers["decode"]["run_id"] = "different-run"
    elif fault == "devices":
        engine["role_devices"]["decode"][0] = "0"
    elif fault == "pp":
        engine["pipeline_parallel_size"] = 2
    elif fault == "missing_kv":
        del engine["kv_transfer"]
    elif fault == "overlap":
        engine["kv_transfer"]["decode_side_channel_port"] = 5558
    elif fault == "side_http":
        engine["kv_transfer"]["decode_side_channel_port"] = 8100
    argv, output = native_replay_inputs(tmp_path, engine, observers)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launches = []
    monkeypatch.setattr(pd_replay.subprocess, "Popen", lambda *args, **kwargs: launches.append(True))
    with pytest.raises(ValueError):
        pd_replay.main(argv)
    assert launches == []
    assert not output.exists()


@pytest.mark.parametrize("failure", [None, "prefill_spawn", "decode_spawn"])
def test_v028_pd_main_retains_plans_and_terminal_state_without_legacy_admission(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(pd_replay.os, "environ", {
        "PATH": "/usr/bin", "VLLM_FRONTIER_OBSERVER": '{"mode":"inherited"}',
        "VLLM_MOE_UNIFORM_ROUTING": "1",
    })
    engine = nixl_engine()
    observers = {"prefill": native_observer("PREFILL"), "decode": native_observer("DECODE")}
    argv, output = native_replay_inputs(tmp_path, engine, observers)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launched, stopped = [], []

    class Server:
        returncode = None

        def __init__(self, name):
            self.name = name

        def poll(self):
            return self.returncode

        def send_signal(self, sent):
            stopped.append(self.name)
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def spawn(command, **kwargs):
        name = ("prefill", "decode", "proxy")[len(launched)]
        launched.append((name, command, kwargs.get("env")))
        if failure == f"{name}_spawn":
            raise OSError(f"fixture {name} spawn failed")
        return Server(name)

    async def http_replay(rows, port, vocab_size, origin_lead_s, *, vllm_version):
        assert vllm_version == "0.28.0"
        return {"origin_monotonic": 100.0, "origin_wall": 1000.0}, [
            {"request_id": "request-7", "dispatch_monotonic": 100.0, "http_status": 200},
        ]

    monkeypatch.setattr(pd_replay.subprocess, "Popen", spawn)
    monkeypatch.setattr(pd_replay, "wait_until_ready", lambda *args: 0.0)
    monkeypatch.setattr(pd_replay, "replay", http_replay)
    if failure is not None:
        with pytest.raises(OSError, match=f"fixture {failure.split('_')[0]} spawn failed"):
            pd_replay.main(argv)
        assert stopped == ([] if failure == "prefill_spawn" else ["prefill"])
    else:
        assert pd_replay.main(argv) == 0
        assert stopped == ["proxy", "decode", "prefill"]
    summary = json.loads((output / "replay_summary.json").read_text())
    assert summary["qualification_verified"] is False
    assert "pd_join" not in summary and "connector_check_failed" not in summary
    for role in ("prefill", "decode"):
        instance = summary["instances"][role]
        assert "VLLM_MOE_UNIFORM_ROUTING" in instance["server_env_removed"]
        assert json.loads(instance["server_env_set"]["VLLM_FRONTIER_OBSERVER"]) == observers[role]
        assert instance["observer_evidence"]["installed_runtime_verified"] is False
        assert instance["observer_evidence"]["process_expectations"]["inventory_verified"] is False
        command = instance["server_command"]
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        assert transfer["kv_connector"] == "NixlConnector"
        assert transfer["kv_role"] == {"prefill": "kv_producer", "decode": "kv_consumer"}[role]
        assert "engine_id" not in transfer
    if failure is None:
        assert summary["native_transfer_verification"] == {
            "status": "UNVERIFIED", "expected_protocol": "NixlConnector pull READ",
            "native_done": "unobserved", "device_completion": "unobserved",
            "decode_readiness": "unobserved", "metric_normalization": "not_applied",
        }
        record = json.loads((output / "client_requests.jsonl").read_text())
        assert "engine_request_id" not in record
        assert "--vllm-version" in launched[2][1]
        assert "--prefill-kv-address" not in launched[2][1]
    else:
        assert summary["error"]["message"] == f"fixture {failure.split('_')[0]} spawn failed"


@pytest.mark.parametrize("decode_status,health_status", [(200, 200), (503, 200), (200, 503)])
def test_v028_external_pd_replays_http_without_owning_native_servers(
    tmp_path, monkeypatch, decode_status, health_status,
):
    import socket
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received, launched = [], []
    params = {
        "do_remote_prefill": True, "do_remote_decode": False, "transfer_mode": "pull",
        "remote_engine_id": "native-p-engine", "remote_request_id": "native-p-request",
        "remote_host": "192.0.2.10", "remote_port": 5557, "remote_block_ids": [[7, 9]],
    }

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/health"
            self.send_response(health_status)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.server.role, self.headers["X-Request-Id"], body))
            status = 200 if self.server.role == "prefill" else decode_status
            payload = ({"kv_transfer_params": params} if self.server.role == "prefill" else
                       {"choices": [{"text": "native response", "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 65, "completion_tokens": 3}})
            streaming = self.server.role == "decode" and status == 200 and body["stream"]
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.end_headers()
            if streaming:
                events = [
                    {"choices": [{"index": 0, "text": "", "token_ids": [7, 8, 9], "finish_reason": "length"}]},
                    {"choices": [], "usage": {"prompt_tokens": 65, "completion_tokens": 3, "total_tokens": 68}},
                ]
                self.wfile.write(("".join("data: " + json.dumps(event) + "\n\n" for event in events)
                                 + "data: [DONE]\n\n").encode())
            else:
                self.wfile.write(json.dumps(payload).encode())

    servers, threads = [], []
    try:
        urls = []
        for role in ("prefill", "decode"):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
            server.role = role
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
            urls.append(f"http://127.0.0.1:{server.server_port}")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            proxy_port = listener.getsockname()[1]
        engine = nixl_engine()
        del engine["role_devices"]
        argv, output = native_replay_inputs(tmp_path, engine)
        argv += ["--prefill-url", urls[0], "--decode-url", urls[1],
                 "--proxy-port", str(proxy_port), "--origin-lead-s", "0",
                 "--startup-timeout-s", "0.01" if health_status != 200 else "10"]
        popen = pd_replay.subprocess.Popen

        def spawn(command, **kwargs):
            assert command[1].endswith("/pd_proxy.py")
            process = popen(command, **kwargs)
            launched.append(process)
            return process

        monkeypatch.setattr(pd_replay.subprocess, "Popen", spawn)
        if health_status == 200:
            assert pd_replay.main(argv) == (0 if decode_status == 200 else 4)
        else:
            with pytest.raises(RuntimeError, match="not ready"):
                pd_replay.main(argv)
        summary = json.loads((output / "replay_summary.json").read_text())
        assert set(summary["process_stop"]) == ({"proxy"} if health_status == 200 else set())
        assert summary["devices"] == {}
        assert len(launched) == (1 if health_status == 200 else 0)
        assert all(process.poll() == 0 for process in launched)
        assert summary["qualification_verified"] is False
        for role, url in zip(("prefill", "decode"), urls):
            assert summary["instances"][role]["lifecycle_owner"] == "external"
            assert summary["instances"][role]["http_url"] == url
            assert "server_command" not in summary["instances"][role]
            assert "ready_after_launch_s" not in summary["instances"][role]
            if health_status == 200:
                with urllib.request.urlopen(url + "/health") as response:
                    assert response.status == 200
            else:
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(url + "/health")
                assert error.value.code == 503
                error.value.close()
        assert all(thread.is_alive() for thread in threads)
        if health_status != 200:
            assert received == [] and summary["error"]["type"] == "RuntimeError"
            return
        assert [(role, request) for role, request, body in received] == [
            ("prefill", "request-7"), ("decode", "request-7")]
        assert received[1][2]["kv_transfer_params"] == params
        assert received[1][2]["max_tokens"] == 3
        client = json.loads((output / "client_requests.jsonl").read_text())
        assert client["http_status"] == decode_status
        assert client["status"] == ("completed" if decode_status == 200 else "failed")
        if decode_status == 200:
            assert client["output_token_count"] == 3 and client["stream_done"]
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


@pytest.mark.parametrize("prefill,decode", [
    ("http://192.0.2.10:8100", None), (None, "http://192.0.2.12:8200"),
    ("https://192.0.2.10:8100", "http://192.0.2.12:8200"),
    ("http://user:secret@192.0.2.10:8100", "http://192.0.2.12:8200"),
    ("http://192.0.2.10:8100/path", "http://192.0.2.12:8200"),
    ("http://192.0.2.10:8100?query=1", "http://192.0.2.12:8200"),
    ("http://192.0.2.10", "http://192.0.2.12:8200"),
    ("http://192.0.2.10:65536", "http://192.0.2.12:8200"),
    ("http://0.0.0.0:8100", "http://192.0.2.12:8200"),
    ("http://127.0.0.1:8000", "http://192.0.2.12:8200"),
    ("http://192.0.2.10:8100", "http://192.0.2.10:8100"),
])
def test_v028_external_pd_rejects_invalid_endpoint_pairs_before_launch(tmp_path, monkeypatch, prefill, decode):
    argv, output = native_replay_inputs(tmp_path, nixl_engine())
    for role, url in (("prefill", prefill), ("decode", decode)):
        if url is not None:
            argv += [f"--{role}-url", url]
    launches = []
    monkeypatch.setattr(pd_replay.subprocess, "Popen", lambda *args, **kwargs: launches.append(True))
    with pytest.raises(ValueError):
        pd_replay.main(argv)
    assert launches == [] and not output.exists()


def test_v028_local_pair_rejects_an_unmanaged_partial_dp_deployment(tmp_path, monkeypatch):
    engine = nixl_engine() | {
        "host": "192.0.2.10", "data_parallel_size_local": 1,
        "data_parallel_address": "192.0.2.10", "data_parallel_rpc_port": 24461,
    }
    argv, output = native_replay_inputs(tmp_path, engine)
    launches = []
    monkeypatch.setattr(pd_replay.subprocess, "Popen", lambda *args, **kwargs: launches.append(True))
    with pytest.raises(ValueError, match="externally managed"):
        pd_replay.main(argv)
    assert launches == [] and not output.exists()


@pytest.mark.parametrize("client_request_id", ["g3-p000", "g3-batch-w15", "7-1", "a_b"])
def test_both_instances_parse_the_other_address_from_the_proxied_id(client_request_id):
    proxied = proxied_request_id(client_request_id, "127.0.0.1:21001", "127.0.0.1:22001")
    engine_id = engine_request_id(proxied)

    assert DECODE_ADDRESS_OF_PREFILL.search(engine_id).groups() == ("127.0.0.1", "22001")
    assert PREFILL_ADDRESS_OF_DECODE.search(engine_id).groups() == ("127.0.0.1", "21001")
    assert engine_id == f"cmpl-{proxied}-0"
    assert proxied.endswith(f"_{client_request_id}")


@pytest.mark.parametrize("client_request_id", ["g3:p000", "_g3-p000"])
def test_a_client_id_that_would_move_an_address_match_is_rejected(client_request_id):
    with pytest.raises(ValueError):
        proxied_request_id(client_request_id, "127.0.0.1:21001", "127.0.0.1:22001")


def proxy_record(request_id: str, arrival: float, prefill_status: int = 200) -> dict:
    record = {
        "request_id": request_id,
        "proxied_request_id": proxied_request_id(request_id, "127.0.0.1:21001", "127.0.0.1:22001"),
        "proxy_arrival_time": arrival,
        "prefill_status": prefill_status,
    }
    if prefill_status == 200:
        record["decode_status"] = 200
    return record


def instance_row(request_id: str, arrival: float, ttft_ms: float, completion: float,
                 decode_tokens: int, tpot_ms: float = 0.0) -> dict:
    return {
        "request_id": engine_request_id(
            proxied_request_id(request_id, "127.0.0.1:21001", "127.0.0.1:22001")
        ),
        "arrival_time": arrival,
        "completion_time": completion,
        "ttft": ttft_ms,
        "tpot": tpot_ms,
        "request_e2e_time": (completion - arrival) * 1000.0,
        "request_num_prefill_tokens": 2048,
        "request_num_decode_tokens": decode_tokens,
    }


def synthetic_run() -> tuple[list[dict], list[dict], list[dict]]:
    """a joins; b lacks its decode row; c has two decode rows; e failed at prefill."""

    proxy = [proxy_record("a", 100.0), proxy_record("b", 101.0), proxy_record("c", 102.0),
             proxy_record("e", 103.0, prefill_status=500)]
    prefill = [instance_row("a", 100.25, 250.0, 100.5, 1), instance_row("b", 101.25, 250.0, 101.5, 1),
               instance_row("c", 102.25, 250.0, 102.5, 1)]
    decode = [instance_row("a", 100.75, 125.0, 102.0, 256, tpot_ms=5.0),
              instance_row("c", 102.75, 125.0, 104.0, 256, tpot_ms=5.0),
              instance_row("c", 103.75, 125.0, 105.0, 256, tpot_ms=5.0),
              instance_row("x", 104.0, 125.0, 105.0, 256, tpot_ms=5.0)]
    return proxy, prefill, decode


def test_the_join_applies_the_pd_metric_contract_and_reports_every_other_request():
    rows, gaps = join_pd_rows(*synthetic_run())

    assert rows == [{
        "request_id": "a",
        "arrival_s": 100.0,
        "completion_s": 102.0,
        "ttft_ms": 500.0,             # prefill first token 100.25 + 0.25 s, minus proxy arrival
        "tpot_ms": 5.0,
        "request_e2e_time_ms": 2000.0,
        "request_num_prefill_tokens": 2048,
        "request_num_decode_tokens": 256,
        "decode_ttft_ms": 125.0,
        "handoff_ms": 250.0,          # decode arrival 100.75 - prefill completion 100.5
        "prefill_arrival_s": 100.25,
        "prefill_first_token_s": 100.5,
        "prefill_completion_s": 100.5,
        "decode_arrival_s": 100.75,
        "prefill_num_prefill_tokens": 2048,
        "prefill_num_decode_tokens": 1,
    }]
    assert set(NORMALIZED_COLUMNS) <= set(rows[0])
    assert {kind: ids for kind, ids in gaps.items() if ids} == {
        "missing_decode_rows": ["b"],
        "duplicate_decode_rows": ["c"],
        "proxy_failures": ["e"],
        "unmatched_decode_rows": [instance_row("x", 0.0, 0.0, 0.0, 0)["request_id"]],
    }


def test_a_prefill_with_more_than_one_output_token_is_reported_but_kept():
    proxy, prefill, decode = synthetic_run()
    prefill[0]["request_num_decode_tokens"] = 2

    rows, gaps = join_pd_rows(proxy[:1], prefill[:1], decode[:1])

    assert [row["request_id"] for row in rows] == ["a"]
    assert gaps["prefill_output_token_mismatches"] == ["a"]


def test_the_joined_files_are_written_once(tmp_path: Path):
    proxy, prefill, decode = synthetic_run()
    for path, rows in ((tmp_path / "proxy_requests.jsonl", proxy),
                       (tmp_path / "prefill" / "request_metrics.jsonl", prefill),
                       (tmp_path / "decode" / "request_metrics.jsonl", decode)):
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = write_pd_rows(tmp_path)

    assert (report["joined"], report["complete"]) == (1, False)
    assert [json.loads(line)["request_id"]
            for line in (tmp_path / "pd_request_metrics.jsonl").read_text().splitlines()] == ["a"]
    assert json.loads((tmp_path / "pd_join_report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        write_pd_rows(tmp_path)


INIT_LINE = (
    "\x1b[1;36m(EngineCore_DP0 pid=7)\x1b[0;0m INFO 09-27 12:00:00 [p2p_nccl_engine.py:168] "
    "💯P2pNcclEngine init, rank:0, local_rank:0, http_address:127.0.0.1:8200, "
    "zmq_address:127.0.0.1:22001, proxy_address:, send_type:PUT_ASYNC, "
    "buffer_size_threshold:8000000000.00, nccl_num_channels:16"
)
FALLBACK_LINE = (
    "\x1b[1;36m(EngineCore_DP0 pid=7)\x1b[0;0m WARNING 09-27 12:00:05 [p2p_nccl_engine.py:347] "
    "🔴[PUT]Recv Tensor, Out Of Threshold, 127.0.0.1:22001👈127.0.0.1:21001, data:{}, addr:0"
)


@pytest.mark.parametrize(("lines", "faults", "passed"), [
    ([INIT_LINE, "INFO serving"], {}, True),
    ([INIT_LINE, FALLBACK_LINE, FALLBACK_LINE], {"[PUT]Recv Tensor, Out Of Threshold": 2}, False),
    (["INFO serving"], {}, False),
    ([INIT_LINE.replace("send_type:PUT_ASYNC", "send_type:PUT")], {}, False),
])
def test_the_connector_check_needs_its_put_async_init_line_and_no_fault(tmp_path, lines, faults, passed):
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(lines) + "\n")

    check = connector_check(server_log)

    assert (check["faults"], check["passed"]) == (faults, passed)


def test_each_instance_gets_its_role_gpu_and_connector_values_from_the_engine_file():
    engine = {"attention_backend": "FLASHINFER", "enable_chunked_prefill": False, "kv_transfer": {
        "kv_buffer_size": 8e9, "mem_pool_size_gb": 1, "prefill_kv_port": 21001,
        "decode_kv_port": 22001, "nccl_num_channels": 16,
    }}
    inherited = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "3,5",
                 "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH": "/old/kv.jsonl"}
    devices = instance_devices(inherited)

    env, set_env = instance_env("clean", engine, Path("/run/decode"), inherited, devices["decode"])

    assert devices == {"prefill": "3", "decode": "5"}
    assert set_env == {
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/decode/dp_placement",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/run/decode/request_metrics.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL": "1",
        "CUDA_VISIBLE_DEVICES": "5",
        "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH": "/run/decode/kv_transfer.jsonl",
    }
    assert env == {"PATH": "/usr/bin", **set_env}
    assert kv_transfer_config(engine, "decode", 8200) == {
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_buffer_size": 8e9,
        "kv_ip": "127.0.0.1",
        "kv_port": 22001,
        "kv_connector_extra_config": {
            "http_port": "8200", "send_type": "PUT_ASYNC",
            "nccl_num_channels": "16", "mem_pool_size_gb": "1",
        },
    }
    with pytest.raises(SystemExit):
        instance_devices({"CUDA_VISIBLE_DEVICES": "4"})
