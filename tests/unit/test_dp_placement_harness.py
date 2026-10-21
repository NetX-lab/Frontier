"""Inputs the DP-placement harness builds for a ground-truth run.

A wrong trace row or server switch is only visible after a multi-GPU run, so
the pieces that decide them are checked here.
"""

from __future__ import annotations

import json
import asyncio
import hashlib
import random
from pathlib import Path

import pytest

from tests.comparison.dp_placement_pp.make_trace import build_rows
from tests.comparison.dp_placement_pp.vllm_replay import OP_TIMING_SCOPES, server_command, server_env
from tests.comparison.dp_placement_pp import vllm_replay


def native_engine() -> dict:
    return {
        "vllm_version": "0.28.0",
        "native_reference": {
            "source_sha": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
            "instrumentation_sha": "6c523e85331b6fcaa1016bc97da8fdb3c1b3b9d2",
            "image": "hub.i.basemind.com/vllm-openai/vllm-openai@sha256:"
                     "2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635",
        },
        "load_format": "dummy", "skip_tokenizer_init": True,
        "tensor_parallel_size": 8, "pipeline_parallel_size": 1,
        "data_parallel_size": 2, "enable_expert_parallel": True,
        "enable_chunked_prefill": True, "enable_prefix_caching": False,
        "enforce_eager": False, "dtype": "bfloat16", "seed": 0,
        "max_num_batched_tokens": 64, "max_num_seqs": 8, "block_size": 64,
        "max_model_len": 4096, "gpu_memory_utilization": 0.8,
    }


def native_observer(role="MONOLITHIC") -> dict:
    return {
        "run_id": "glm-observer-01", **{
            key: value for key, value in native_engine()["native_reference"].items()
            if key != "image"
        },
        "config_id": "glm-full", "replica_id": "pod", "role": role,
        "expected_ranks": list(range(16)),
        "output_dir": f"/mnt/codesign-exp/ycfeng/glm-observer-01/{role}/trace",
        "mode": "light", "capacity_bytes": 4194304,
    }


def test_v028_installs_only_explicit_observer_after_sanitizing_inherited_switches():
    inherited = {"PATH": "/usr/bin", "VLLM_FRONTIER_OBSERVER": '{"mode":"light"}',
                 "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH": "/old/routing",
                 "VLLM_MOE_UNIFORM_ROUTING": "1", "VLLM_ATTENTION_BACKEND": "OLD"}
    clean, changed = server_env("clean", native_engine(), Path("/run"), inherited)
    assert clean == {"PATH": "/usr/bin"}
    assert changed == {}
    observer = native_observer()
    light, changed = server_env("light", native_engine(), Path("/run"), inherited,
                                observer=observer)
    assert json.loads(changed["VLLM_FRONTIER_OBSERVER"]) == observer
    assert light == {"PATH": "/usr/bin", **changed}
    assert set(changed) == {"VLLM_FRONTIER_OBSERVER"}
    assert observer["mode"] == "light"
    explicit_clean = observer | {"mode": "clean"}
    clean, changed = server_env("clean", native_engine(), Path("/run"), inherited,
                                observer=explicit_clean)
    assert json.loads(changed["VLLM_FRONTIER_OBSERVER"]) == explicit_clean


def test_v028_command_uses_local_dummy_metadata_and_native_attention_args(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 154880,
    }))
    engine = native_engine() | {"attention_backend": "FLASHMLA_SPARSE",
                                 "async_scheduling": True, "quantization": "fp8",
                                 "shutdown_timeout": 30,
                                 "distributed_executor_backend": "mp",
                                 "worker_extension_cls": "native_full_structure.FullModelInventory"}
    command = server_command(engine, tmp_path, 8100)
    assert command[command.index("--load-format") + 1] == "dummy"
    assert command[command.index("--attention-backend") + 1] == "FLASHMLA_SPARSE"
    assert command[command.index("--quantization") + 1] == "fp8"
    assert command[command.index("--shutdown-timeout") + 1] == "30"
    assert command[command.index("--distributed-executor-backend") + 1] == "mp"
    assert command[command.index("--worker-extension-cls") + 1] == "native_full_structure.FullModelInventory"
    assert "--async-scheduling" in command
    assert "--enable-return-routed-experts" not in command
    assert "--trust-remote-code" not in command
    with pytest.raises(ValueError, match="local"):
        server_command(engine, tmp_path / "absent", 8100)


@pytest.mark.parametrize("headless", [False, True])
def test_v028_command_preserves_internal_dp_head_and_secondary_node(tmp_path, headless):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 154880,
    }))
    engine = native_engine() | {
        "host": "0.0.0.0", "data_parallel_size_local": 1,
        "data_parallel_address": "192.0.2.10", "data_parallel_rpc_port": 24461,
    }
    if headless:
        engine.update(headless=True, data_parallel_start_rank=1)
    command = server_command(engine, tmp_path, 8100)
    for flag, value in {
        "--host": "0.0.0.0", "--tensor-parallel-size": "8",
        "--data-parallel-size": "2", "--data-parallel-size-local": "1",
        "--data-parallel-address": "192.0.2.10", "--data-parallel-rpc-port": "24461",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert ("--headless" in command) is headless
    if headless:
        assert command[command.index("--data-parallel-start-rank") + 1] == "1"
    else:
        assert "--data-parallel-start-rank" not in command
    assert "--data-parallel-rank" not in command
    assert "--data-parallel-hybrid-lb" not in command
    assert "--data-parallel-external-lb" not in command


@pytest.mark.parametrize("changes", [
    {"data_parallel_start_rank": 0}, {"headless": True},
    {"data_parallel_size_local": 0}, {"data_parallel_size_local": 3},
    {"data_parallel_size_local": True}, {"data_parallel_address": "127.0.0.1"},
    {"data_parallel_address": "0.0.0.0"}, {"data_parallel_address": "224.0.0.1"},
    {"data_parallel_address": None}, {"data_parallel_rpc_port": 65536},
    {"data_parallel_rpc_port": 8100}, {"host": "127.0.0.1"}, {"host": "not-an-ip"},
    {"headless": "true"}, {"headless": True, "data_parallel_start_rank": 2},
    {"headless": True, "data_parallel_start_rank": -1}, {"pipeline_parallel_size": 2},
    {"data_parallel_rank": 0}, {"data_parallel_hybrid_lb": True},
    {"data_parallel_external_lb": True},
])
def test_v028_rejects_invalid_or_external_dp_node_options(tmp_path, changes):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 154880,
    }))
    engine = native_engine() | {
        "host": "0.0.0.0", "data_parallel_size_local": 1,
        "data_parallel_address": "192.0.2.10", "data_parallel_rpc_port": 24461,
    } | changes
    with pytest.raises(ValueError):
        server_command(engine, tmp_path, 8100)


@pytest.mark.parametrize("blocks", [0, -1, True, None, "17", 17.5])
def test_v028_rejects_invalid_native_gpu_block_override(tmp_path, blocks):
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        server_command(native_engine() | {"num_gpu_blocks_override": blocks}, tmp_path, 8100)


@pytest.mark.parametrize("block_counts", [{}, {"decode": 17}, {"prefill": 19, "decode": 17},
                                          {"other": 17}, {"decode": True}, {"decode": 0}, None])
def test_native_pd_node_capacity_reaches_only_its_selected_role(tmp_path, block_counts):
    from tests.comparison.calibration.native_pd_worker import node_plan

    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 154880,
    }))
    engine = native_engine() | {"kv_transfer": {}}
    config = {"engine": engine, "data_parallel_size_local": 1,
              "dp_rpc_ports": {"prefill": 24471, "decode": 24571},
              "http_ports": {"prefill": 8100, "decode": 8200},
              "role_num_gpu_blocks_override": block_counts}
    peers = [{"host": f"node{index}", "ip": f"192.0.2.{10 + index}"} for index in range(4)]
    before = json.dumps(config, sort_keys=True)
    for index, peer in enumerate(peers):
        if block_counts in ({"other": 17}, {"decode": True}, {"decode": 0}, None):
            with pytest.raises(ValueError, match="Role GPU block overrides"):
                node_plan(config, peers, peer["host"], [str(n) for n in range(8)])
            continue
        plan = node_plan(config, peers, peer["host"], [str(n) for n in range(8)])
        command = server_command(plan["engine"], tmp_path, config["http_ports"][plan["role"]])
        flag = "--num-gpu-blocks-override"
        if plan["role"] in block_counts:
            assert command[command.index(flag) + 1] == str(block_counts[plan["role"]])
        else:
            assert flag not in command
        assert ("--headless" in command) is bool(index % 2)
    assert json.dumps(config, sort_keys=True) == before


@pytest.mark.parametrize("fault", ["version", "source", "image", "mode", "no_observer",
                                  "clean_light", "capacity", "ranks", "role", "output",
                                  "observer_source", "startup", "routing", "weights",
                                  "boolean", "rank_count"])
def test_v028_rejects_incompatible_inputs_before_launch(fault):
    engine, observer, mode = native_engine(), native_observer(), "light"
    if fault == "version":
        engine["vllm_version"] = "0.29.0"
    elif fault in ("source", "image"):
        engine["native_reference"]["source_sha" if fault == "source" else "image"] = "wrong"
    elif fault == "mode":
        mode = "instrumented"
    elif fault == "no_observer":
        observer = None
    elif fault == "clean_light":
        mode = "clean"
    elif fault == "capacity":
        observer["capacity_bytes"] = True
    elif fault == "ranks":
        observer["expected_ranks"] = [0, True]
    elif fault == "role":
        observer["role"] = "DECODE_ATTN"
    elif fault == "output":
        observer["output_dir"] = "/mnt/codesign-exp/other/trace"
    elif fault == "observer_source":
        observer["source_sha"] = "a" * 40
    elif fault == "startup":
        observer["startup_memory"] = {"records": 12, "bytes": 65536}
    elif fault == "routing":
        engine["enable_return_routed_experts"] = True
    elif fault == "weights":
        engine["load_format"] = "auto"
    elif fault == "boolean":
        engine["enforce_eager"] = "false"
    elif fault == "rank_count":
        observer["expected_ranks"] = [0]
    with pytest.raises(ValueError):
        server_env(mode, engine, Path("/run"), {}, observer=observer)


@pytest.mark.parametrize("version", ["0.10.2", "0.28.0"])
def test_replay_keeps_client_identity_and_only_legacy_internal_id_contract(version):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def exercise():
        async def complete(request):
            assert request.headers["X-Request-Id"] == "request-7"
            body = await request.json()
            if version == "0.28.0":
                assert body["stream"] and body["return_token_ids"]
                payloads = [{"choices": [{"index": 0, "text": "", "token_ids": [1], "finish_reason": "length"}]},
                            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}]
                return web.Response(text="".join("data: " + json.dumps(payload) + "\n\n" for payload in payloads)
                                    + "data: [DONE]\n\n", content_type="text/event-stream")
            return web.json_response({"choices": [{"finish_reason": "length"}], "usage": {}})

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        async with TestServer(app) as server:
            _, records = await vllm_replay.replay([
                {"request_id": "request-7", "segment": "formal", "role": "formal",
                 "arrived_at": 0.0, "num_prefill_tokens": 3, "num_decode_tokens": 1},
            ], server.port, 1024, 0.0, vllm_version=version)
        return records

    records = asyncio.run(exercise())
    assert records[0]["request_id"] == "request-7"
    assert records[0]["http_status"] == 200
    assert records[0]["status"] == "completed"
    if version == "0.10.2":
        assert records[0]["engine_request_id"] == "cmpl-request-7-0"
    else:
        assert "engine_request_id" not in records[0]


def native_replay_inputs(tmp_path, engine, observer=None):
    engine_path = tmp_path / "engine.json"
    engine_path.write_text(json.dumps(engine))
    model = tmp_path / "metadata"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 154880,
    }))
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "request_ids.json").write_text(json.dumps({"rows": [
        {"request_id": "request-7", "arrived_at": 0.0, "role": "formal", "segment": "formal",
         "num_prefill_tokens": 65, "num_decode_tokens": 3},
    ]}))
    output = tmp_path / "output"
    argv = ["--engine-config", str(engine_path), "--trace-dir", str(trace),
            "--output-dir", str(output), "--model-dir", str(model),
            "--mode", "clean" if observer is None else "light", "--drain-s", "0"]
    if observer is not None:
        observer_path = tmp_path / "observer.json"
        observer_path.write_text(json.dumps(observer))
        argv += ["--observer-config", str(observer_path)]
    return argv, output


@pytest.mark.parametrize("version", ["0.10.2", "0.28.0"])
@pytest.mark.parametrize("phase", ["completed", "cancelled"])
def test_main_posts_and_retains_every_planned_request_over_http(tmp_path, monkeypatch, version, phase):
    import threading
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    engine = native_engine() | {"vllm_version": version, "model_name": "cpu-entry-fixture"}
    argv, output = native_replay_inputs(tmp_path, engine)
    if version == "0.10.2":
        index = argv.index("--model-dir")
        del argv[index:index + 2]

    def local_metadata(source, destination):
        destination.mkdir()
        return {"vocab_size": 1024}

    monkeypatch.setattr(vllm_replay, "write_model_dir", local_metadata)
    monkeypatch.setattr(vllm_replay.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(vllm_replay, "wait_until_ready", lambda *args: 0.)
    monkeypatch.setattr(vllm_replay, "stop_server", lambda *args: {"signal": "SIGINT", "returncode": 0})
    planned = [{"request_id": f"main-{index}", "arrived_at": 0., "role": "formal", "segment": "formal",
                "num_prefill_tokens": 4, "num_decode_tokens": 3} for index in range(2)]
    if phase == "cancelled":
        planned[1]["arrived_at"] = 10000.
    trace = Path(argv[argv.index("--trace-dir") + 1])
    (trace / "request_ids.json").write_text(json.dumps({"rows": planned}))
    posted = []
    started = threading.Event()
    actual_replay = vllm_replay.replay

    if phase == "cancelled":
        async def cancel_replay(*args, **kwargs):
            task = asyncio.create_task(actual_replay(*args, **kwargs))
            try:
                assert await asyncio.to_thread(started.wait, 5), "main never submitted its planned request"
                task.cancel()
                return await task
            finally:
                if not task.done():
                    task.cancel()
        monkeypatch.setattr(vllm_replay, "replay", cancel_replay)

    async def exercise():
        release = asyncio.Event()

        async def complete(request):
            body = await request.json()
            posted.append(request.headers["X-Request-Id"])
            assert body["stream"] is (version == "0.28.0")
            started.set()
            if phase == "cancelled":
                await release.wait()
            usage = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            if version == "0.28.0":
                payloads = [{"choices": [{"index": 0, "text": "", "token_ids": [1, 2, 3], "finish_reason": "length"}]},
                            {"choices": [], "usage": usage}]
                return web.Response(text="".join("data: " + json.dumps(payload) + "\n\n" for payload in payloads)
                                    + "data: [DONE]\n\n", content_type="text/event-stream")
            return web.json_response({"choices": [{"finish_reason": "length"}], "usage": usage})

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        async with TestServer(app) as server:
            args = argv + ["--port", str(server.port), "--origin-lead-s", "0"]
            try:
                if phase == "cancelled":
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.to_thread(vllm_replay.main, args)
                else:
                    assert await asyncio.to_thread(vllm_replay.main, args) == 0
            finally:
                release.set()

    asyncio.run(exercise())
    records = [json.loads(line) for line in (output / "client_requests.jsonl").read_text().splitlines()]
    assert len(posted) == (2 if phase == "completed" else 1)
    assert len(records) == len(planned) == 2
    assert {record["request_id"] for record in records} == {entry["request_id"] for entry in planned}
    assert len({record["request_id"] for record in records}) == len(records)
    assert all(record["status"] == ("completed" if phase == "completed" else "unfinished") for record in records)
    summary = json.loads((output / "replay_summary.json").read_text())
    if phase == "cancelled":
        assert records[0]["dispatch_monotonic"] is not None
        assert records[1]["dispatch_monotonic"] is None
        assert summary["error"]["type"] == "CancelledError"
    else:
        assert summary["num_requests"] == 2 and summary["failed_request_ids"] == []
    print("MAIN_HTTP_EVIDENCE", json.dumps({"version": version, "phase": phase, "posts": len(posted),
                                           "records": len(records), "ids": sorted(record["request_id"] for record in records)}))


@pytest.mark.parametrize("failure", ["spawn", "startup"])
def test_v028_main_retains_terminal_evidence_on_launch_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(vllm_replay.os, "environ", {
        "PATH": "/usr/bin", "VLLM_FRONTIER_OBSERVER": '{"mode":"inherited"}',
        "VLLM_MOE_UNIFORM_ROUTING": "1",
    })
    argv, output = native_replay_inputs(tmp_path, native_engine(), native_observer())
    error = OSError("fixture spawn failed") if failure == "spawn" else RuntimeError("fixture startup failed")

    class ExitedServer:
        returncode = 23

        def poll(self):
            return self.returncode

    def spawn(*args, **kwargs):
        if failure == "spawn":
            raise error
        return ExitedServer()

    def ready(*args):
        raise error

    monkeypatch.setattr(vllm_replay.subprocess, "Popen", spawn)
    monkeypatch.setattr(vllm_replay, "wait_until_ready", ready)
    with pytest.raises(type(error)) as raised:
        vllm_replay.main(argv)
    assert raised.value is error
    summary = json.loads((output / "replay_summary.json").read_text())
    assert summary["error"] == {"type": type(error).__name__, "message": str(error)}
    assert summary["qualification_verified"] is False
    assert "VLLM_MOE_UNIFORM_ROUTING" in summary["server_env_removed"]
    assert summary["observer_evidence"]["process_expectations"] == {
        "declared_dp_size": 2, "declared_worker_ranks": list(range(16)),
        "engine_owner_inventory": None, "worker_inventory": None, "inventory_verified": False,
    }
    assert summary["observer_evidence"]["installed_runtime_verified"] is False
    assert summary["observer_evidence"]["native_close_and_device_completion"] == "unverified"
    if failure == "startup":
        assert summary["server_stop"] == {"signal": None, "returncode": 23}
    else:
        assert "server_stop" not in summary


def test_v028_retains_nested_trace_file_identities_without_admitting_contents(tmp_path):
    trace = tmp_path / "worker" / "rank0.jsonl"
    trace.parent.mkdir()
    payload = b'{"fixture":"not native qualification"}\n'
    trace.write_bytes(payload)
    observer = native_observer() | {"output_dir": str(tmp_path)}
    evidence = vllm_replay.observer_evidence(native_engine(), observer)
    assert evidence["trace_files"] == [{
        "path": str(trace), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }]
    assert evidence["collection_errors"] == []
    assert evidence["qualification_verified"] is False
    assert evidence["process_expectations"]["inventory_verified"] is False


def test_v028_collector_selects_only_the_configured_root_and_retains_partial_bytes(tmp_path):
    trace = tmp_path / "persisted" / "engine" / "EngineCore.jsonl"
    trace.parent.mkdir(parents=True)
    payload = b'{"kind":"manifest"}\n{"kind":"incomplete'
    trace.write_bytes(payload)
    observer = native_observer() | {"output_dir": str(tmp_path / "wrong-root")}
    excluded = vllm_replay.observer_evidence(native_engine(), observer)
    assert excluded["trace_files"] == []
    assert excluded["trace_directory_exists"] is False
    assert excluded["collection_errors"] == []

    included = vllm_replay.observer_evidence(
        native_engine(), observer | {"output_dir": str(trace.parents[1])})
    assert included["trace_files"] == [{
        "path": str(trace), "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }]
    assert included["qualification_verified"] is False
    assert included["process_expectations"]["inventory_verified"] is False
    assert included["native_close_and_device_completion"] == "unverified"


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_v028_collector_read_failure_preserves_only_preceding_file_identities(
        tmp_path, monkeypatch, failed_index):
    traces = [tmp_path / name for name in ("a_engine.jsonl", "b_worker.jsonl", "c.jsonl")]
    payloads = [b"engine fixture\n", b"worker fixture\n", b"unclassified fixture\n"]
    for trace, payload in zip(traces, payloads):
        trace.write_bytes(payload)
    original_open, reads = Path.open, []

    def open_trace(path, mode="r", *args, **kwargs):
        if path in traces and mode == "rb":
            reads.append(path)
            if path == traces[failed_index]:
                raise OSError("injected collector read failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_trace)
    observer = native_observer() | {"output_dir": str(tmp_path)}
    evidence = vllm_replay.observer_evidence(native_engine(), observer)
    assert reads == traces[:failed_index + 1]
    assert evidence["trace_files"] == [{
        "path": str(trace), "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    } for trace, payload in zip(traces[:failed_index], payloads)]
    assert evidence["collection_errors"] == ["OSError: injected collector read failure"]
    assert evidence["trace_directory_exists"] is True
    assert evidence["qualification_verified"] is False
    assert evidence["process_expectations"]["inventory_verified"] is False


@pytest.mark.parametrize("config", [[], {"architectures": ["DeepseekV4ForCausalLM"]},
                                    {"architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": True}])
def test_v028_main_rejects_invalid_local_metadata_before_launch(tmp_path, monkeypatch, config):
    monkeypatch.setattr(vllm_replay.os, "environ", {"PATH": "/usr/bin"})
    argv, output = native_replay_inputs(tmp_path, native_engine())
    (tmp_path / "metadata" / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(vllm_replay.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(ValueError):
        vllm_replay.main(argv)
    assert not output.exists()


def workload(bursts: list[dict]) -> dict:
    return {
        "request_id_namespace": "t",
        "warmups": {"count": 1, "interval_s": 1.0, "num_prefill_tokens": 8, "num_decode_tokens": 2},
        "idle_gap_s": 20.0,
        "bursts": bursts,
        "steady": {
            "gap_after_burst_s": 5.0, "count": 1, "interval_s": 0.1,
            "num_prefill_tokens": [16], "num_decode_tokens": [4],
        },
    }


def test_a_burst_has_a_probe_only_when_it_names_an_offset_and_may_set_its_own_gap():
    rows = build_rows(workload([
        {"name": "q", "order": ["short", "short"], "short_prefill_tokens": 32,
         "num_decode_tokens": 4, "idle_gap_s": 3.0, "probe_offset_s": 0.0005,
         "probe_prefill_tokens": 32, "probe_decode_tokens": 4},
        {"name": "l", "order": ["long"], "long_prefill_tokens": 4096, "num_decode_tokens": 1},
    ]))

    burst = [
        (row["request_id"], row["arrived_at"], row["num_prefill_tokens"], row["probe"])
        for row in rows if row["segment"] == "burst"
    ]
    assert burst == [
        ("t-q-b1", 3.0, 32, False),
        ("t-q-b2", 3.0, 32, False),
        ("t-q-b3", 3.0005, 32, True),
        ("t-l-b1", 23.0005, 4096, False),
    ]
    assert rows[-1]["arrived_at"] == 28.0005


def test_warmup_groups_precede_seeded_poisson_arrivals():
    workload = {
        "request_id_namespace": "t",
        "warmups": {"groups": [
            {"name": "single", "count": 2, "interval_s": 1.0,
             "num_prefill_tokens": 2048, "num_decode_tokens": 16},
            {"name": "batch", "gap_before_s": 3.0, "count": 2, "interval_s": 0.03,
             "num_prefill_tokens": 2048, "num_decode_tokens": 64},
        ]},
        "poisson": {"gap_before_s": 10.0, "count": 3, "qps": 2.0, "seed": 0,
                    "num_prefill_tokens": 2048, "num_decode_tokens": 256},
    }
    rows = build_rows(workload)

    assert [(row["request_id"], row["arrived_at"]) for row in rows if row["role"] == "warmup"] == [
        ("t-single-w0", 0.0), ("t-single-w1", 1.0), ("t-batch-w0", 4.0), ("t-batch-w1", 4.03),
    ]
    formal = [row for row in rows if row["role"] == "formal"]
    assert [row["request_id"] for row in formal] == ["t-p000", "t-p001", "t-p002"]
    assert {row["segment"] for row in formal} == {"poisson"}
    assert {(row["num_prefill_tokens"], row["num_decode_tokens"]) for row in formal} == {(2048, 256)}
    intervals = random.Random(0)
    expected = [14.03]
    expected.append(expected[-1] + intervals.expovariate(2.0))
    expected.append(expected[-1] + intervals.expovariate(2.0))
    assert [row["arrived_at"] for row in formal] == [round(value, 6) for value in expected]
    assert build_rows(workload) == rows


def test_the_mode_alone_sets_the_frontier_switches_of_the_server():
    inherited = {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/tmp/sched.jsonl",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/tmp/old.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL": "1",
        "VLLM_MOE_UNIFORM_ROUTING": "1",
        "VLLM_TORCH_PROFILER_DIR": "/tmp/old_traces",
        "VLLM_TORCH_PROFILER_RECORD_SHAPES": "1",
        "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
    }
    out = Path("/run")

    clean, clean_set = server_env("clean", {"enable_chunked_prefill": True}, out, inherited)
    instrumented, _ = server_env(
        "instrumented",
        {"attention_backend": "FLASHINFER", "enable_chunked_prefill": False,
         "moe_uniform_routing": True},
        out,
        inherited,
    )
    op_timing, _ = server_env("op_timing", {"enable_chunked_prefill": True}, out, inherited)
    kernel_timing, _ = server_env("kernel_timing", {"enable_chunked_prefill": True}, out, inherited)
    schedule_timing, _ = server_env("schedule_timing", {"enable_chunked_prefill": True}, out, inherited)
    cpu, _ = server_env("cpu", {"enable_chunked_prefill": True}, out, inherited)
    kv_save_timing, _ = server_env("kv_save_timing", {"enable_chunked_prefill": True}, out, inherited)
    kv_save_device_ids, _ = server_env("kv_save_device_ids", {"enable_chunked_prefill": True}, out, inherited)
    device_timeline, _ = server_env("device_timeline", {"enable_chunked_prefill": True}, out, inherited,
                                    Path("/traces"))

    assert clean == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/run/request_metrics.jsonl",
    }
    assert clean_set == {key: value for key, value in clean.items() if key != "PATH"}
    assert instrumented == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_INSTRUMENTATION": "1",
        "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH": "/run/moe_routing.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "VLLM_MOE_UNIFORM_ROUTING": "1",
        "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL": "1",
    }
    assert op_timing == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_INSTRUMENTATION": "1",
        "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH": "/run/op_timing.jsonl",
        "VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES": ",".join(OP_TIMING_SCOPES),
        "VLLM_FRONTIER_OP_TIMING_MODE": "cuda_event",
        "VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE": "default",
        "VLLM_FRONTIER_OP_AGG_MODE": "per_scope",
        "VLLM_FRONTIER_BATCH_LOG_PATH": "/run/batch_log.jsonl",
        "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH": "/run/pp_boundary.jsonl",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/run/schedule.jsonl",
    }
    assert kernel_timing == op_timing | {"VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE": "kernel_only"}
    assert schedule_timing == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/run/schedule.jsonl",
    }
    assert cpu == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_CPU_PROBE_LOG_PATH": "/run/cpu_probe.jsonl",
    }
    assert kv_save_timing == cpu | {"VLLM_FRONTIER_KV_SAVE_PROBE_LOG_PATH": "/run/kv_save_probe.jsonl"}
    assert kv_save_device_ids == kv_save_timing | {"VLLM_FRONTIER_KV_SAVE_DEVICE_BLOCK_IDS": "1"}
    assert device_timeline == cpu | {
        "VLLM_TORCH_PROFILER_DIR": "/traces",
        "VLLM_TORCH_PROFILER_WITH_STACK": "0",
        "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
    }
    # The fork's default scope list has no dense MLP scope; the timing modes name them.
    assert {"mlp_up_proj", "mlp_act", "mlp_down_proj"} <= set(OP_TIMING_SCOPES)


def test_the_engine_file_alone_sets_the_cuda_graph_setup_of_the_server():
    engine = {
        "load_format": "dummy", "dtype": "float16", "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1, "data_parallel_size": 1, "max_num_batched_tokens": 16384,
        "max_num_seqs": 64, "block_size": 16, "max_model_len": 4096,
        "gpu_memory_utilization": 0.9, "seed": 0, "skip_tokenizer_init": True,
        "enable_expert_parallel": False, "enforce_eager": True,
        "enable_chunked_prefill": False, "enable_prefix_caching": False,
    }
    graph = engine | {
        "enforce_eager": False,
        "compilation_config": {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
    }

    eager_command = server_command(engine, Path("/m"), 8000)
    graph_command = server_command(graph, Path("/m"), 8000)

    assert "--enforce-eager" in eager_command
    assert "--compilation-config" not in eager_command
    assert "--enforce-eager" not in graph_command
    config = graph_command[graph_command.index("--compilation-config") + 1]
    assert json.loads(config) == {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
