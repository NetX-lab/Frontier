"""Check diagnostic-only staging, opt-in activation and a real HTTP client cancellation."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.comparison.calibration import native_pd_worker
from tests.comparison.dp_placement_pp import vllm_replay
from tests.integration.step5_pd import build_stage, r2_stage, verify_r2_stage
from tests.unit.test_native_pd_worker import exercise
from tests.unit.test_step5_native_pd import stage_inputs


def definition():
    return json.loads((r2_stage.REPO / "tests/integration/step5_pd/r2_run.json").read_text())


def test_r2_workload_has_boundaries_concurrency_cancel_reuse_and_no_clean_window():
    run = definition()
    traces = build_stage.wave_traces(run, "r2test")
    assert len(traces) == 15 and sum(map(len, traces.values())) == 20
    lengths = [row["num_prefill_tokens"] for rows in traces.values() for row in rows]
    for boundary in (8, 16):
        assert {boundary - 1, 0, 1} <= {length % boundary for length in lengths}
    assert len(traces["concurrent6"]) == 6
    assert all(row["arrived_at"] == 0 and row["num_decode_tokens"] == 128
               for row in traces["concurrent6"])
    assert traces["cancel1"][0]["cancel_after_first_content"] is True
    assert traces["reuse1537"][0]["num_prefill_tokens"] == 1537
    assert all(row["stream"] and row["num_prefill_tokens"] + row["num_decode_tokens"] <= 2048
               for rows in traces.values() for row in rows)
    for rows in traces.values():
        for row in rows:
            n, out = row["num_prefill_tokens"], row["num_decode_tokens"]
            assert n // 8 < (n + out - 2) // 8
    assert run["role_num_gpu_blocks_override"] == {"prefill": 1024, "decode": 512}
    assert run["engine"]["max_num_seqs"] == 4 and run["engine"]["enable_prefix_caching"] is False
    assert run["resources"]["replica"] * run["resources"]["gpu"] == 32


def test_state_activation_is_server_only_and_off_by_default(tmp_path):
    inherited = {"PYTHONPATH": "existing", "NCCL_TEST": "preserve"}
    assert native_pd_worker.pd_state_environment({}, "decode", tmp_path, tmp_path, inherited) == (inherited, {})
    config = {"pd_state_observation": {"selected_client_request_ids": ["r2-1"]}}
    original = copy.deepcopy(config)
    env, changed = native_pd_worker.pd_state_environment(config, "decode", tmp_path / "scratch", tmp_path, inherited)
    settings = json.loads(env["STEP5_PD_STATE_OBSERVATION"])
    assert settings["role"] == "decode" and settings["output_dir"] == str(tmp_path / "scratch/pd_state")
    assert env["PYTHONPATH"] == f"{tmp_path}/r2_bootstrap:{tmp_path}:existing"
    assert env["NCCL_TEST"] == "preserve" and config == original
    assert "STEP5_PD_STATE_OBSERVATION" in changed
    assert "STEP5_PD_STATE_OBSERVATION" not in inherited


def test_native_worker_archives_local_state_after_native_stop(tmp_path):
    def settings(directory):
        output = directory / "scratch/pd_state"
        output.mkdir()
        (output / "100.jsonl").write_text('{"event":"source_free"}\n')
        return {"serve_observers": None, "pd_state_observation": {"region_size": 8}}
    result = exercise(native_pd_worker, tmp_path, "r2_state", overrides=settings)
    node = result["directory"] / "output/nodes/host0"
    assert result["worker_returncode"] == 0 and not result["failed_sentinel"]
    assert (node / "pd_state/100.jsonl").read_text() == '{"event":"source_free"}\n'
    recorded = json.loads((node / "result.json").read_text())
    assert json.loads(recorded["server_env_set"]["STEP5_PD_STATE_OBSERVATION"])["role"] == "prefill"
    assert "--observer-config" not in recorded["replay_command"]


def test_r2_stage_freezes_runtime_and_unchanged_library_bytes(tmp_path, monkeypatch):
    run = definition()
    source = tmp_path / "native"
    for module in r2_stage.RUNTIME_MODULES + r2_stage.SCHEMA_MODULES:
        path = source / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{module}"""\n')
    run["pd_state_observation"]["source_root"] = str(source)
    repo = tmp_path / "repo"
    for relative in r2_stage.CAPTURE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""Frozen bytes: {relative}"""\n')
    monkeypatch.setattr(r2_stage, "REPO", repo)
    stage = tmp_path / "stage"
    trace = stage / "trace/seq255/request_ids.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"rows": [{"request_id": "r2test-seq255-0"}]}))
    model = tmp_path / "metadata"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 2, "layer_types": ["full_attention", "sliding_attention", "full_attention"],
        "sparse_config": {"apply_to_layer_types": ["full_attention"]}}}))
    config = {}
    r2_stage.stage_observation(config, run, stage, model)
    assert config["diagnostic_only"] and config["clean_timing_eligible"] is False
    observation = config["pd_state_observation"]
    assert observation["expected_layer_count"] == 2 and observation["expected_summary_layer_count"] == 1
    assert observation["selected_client_request_ids"] == ["r2test-seq255-0"]
    assert observation["concurrent_waves"] == ["concurrent6"]
    assert "source_root" not in observation
    for source, destination in r2_stage.CAPTURE_FILES.items():
        assert (repo / source).read_bytes() == (stage / destination).read_bytes()
    for name, digest in observation["frozen_environment_hashes"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('gpu_type,charged_group', [('H800', 'codesign'), ('H200', 'step_main')])
def test_offline_verifier_uses_the_frozen_resource_definition(tmp_path, gpu_type, charged_group):
    inputs = stage_inputs(tmp_path)
    run = definition()
    run['probe_helpers'] = inputs['definition']['probe_helpers']
    run['resources'].update(gpu_type=gpu_type, charged_group=charged_group)
    source_root = tmp_path / 'native'
    for module in r2_stage.RUNTIME_MODULES + r2_stage.SCHEMA_MODULES:
        path = source_root / (module.replace('.', '/')+'.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{module}"""\n')
    run['pd_state_observation']['source_root'] = str(source_root)
    model_config = inputs['model_dir'] / 'config.json'
    model = json.loads(model_config.read_text())
    model['text_config'].update(num_hidden_layers=2, layer_types=['full_attention', 'sliding_attention'],
                                sparse_config={'apply_to_layer_types': ['full_attention']})
    model_config.write_text(json.dumps(model))
    workspace = tmp_path / 'workspace'
    stage = workspace / 'stage'
    config = build_stage.build(run, stage=stage, workspace=workspace,
        output_dir=Path('/mnt/codesign-exp/ycfeng/step5-preview/r2_resource_test'),
        model_dir=inputs['model_dir'], image_source_hashes=inputs['image_source_hashes'],
        probe_dir=inputs['probe_dir'], job_tag='r2test',
        source={'repository': str(r2_stage.REPO), 'commit': 'unit'})
    receipt = verify_r2_stage.verify(stage)
    assert receipt['status'] == 'PASS_OFFLINE_STAGE'
    assert receipt['resources']['gpu_type'] == gpu_type
    config['charged_group'] = 'not_the_frozen_group'
    (stage / 'config.json').write_text(json.dumps(config))
    with pytest.raises(AssertionError):
        verify_r2_stage.verify(stage)


def test_state_archive_error_does_not_stop_healthy_native_work(tmp_path, monkeypatch):
    def settings(directory):
        (directory / "scratch/pd_state").mkdir()
        return {"serve_observers": None, "pd_state_observation": {"region_size": 8}}
    def failed_copy(*args, **kwargs):
        raise OSError("Injected state archive write failure")
    monkeypatch.setattr(native_pd_worker.shutil, "copytree", failed_copy)
    result = exercise(native_pd_worker, tmp_path, "r2_archive_error", overrides=settings)
    assert result["worker_returncode"] == 0 and not result["failed_sentinel"]
    assert result["native_server_returncode"] == 0
    assert any("Injected state archive write failure" in error for error in result["observation_errors"])


@pytest.mark.parametrize("case", ["control", "native_exit_at_replay_completion", "native_shutdown_error"])
def test_r2_worker_marks_actual_native_lifecycle_errors_only(tmp_path, case):
    result = exercise(native_pd_worker, tmp_path, case,
                      overrides=lambda directory: {"pd_state_observation": {"region_size": 8}})
    path = result["directory"] / "output/nodes/host0/pd_state/native_lifecycle.jsonl"
    assert path.exists() == case.startswith("native_")
    if path.exists():
        event = json.loads(path.read_text())
        assert event["event"] == "native_error" and event["server_stop"]["returncode"] == 23


def test_r2_cancellation_closes_after_first_nonempty_content_without_token_estimate():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def run():
        finish = asyncio.Event()
        bodies = []
        async def serve(request):
            bodies.append(await request.json())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":""}]}\n\n')
            await response.write(b'data: {"choices":[{"text":"first"}]}\n\n')
            await finish.wait()
            return response
        app = web.Application()
        app.router.add_post("/v1/completions", serve)
        async with TestServer(app) as server:
            _, records = await vllm_replay.replay([{
                "request_id": "cancel-1", "segment": "cancel", "role": "recorded", "arrived_at": 0,
                "num_prefill_tokens": 255, "num_decode_tokens": 256, "stream": True,
                "cancel_after_first_content": True,
            }], server.port, 1024, 0, vllm_version="step5")
            finish.set()
        return records[0], bodies[0]
    row, body = asyncio.run(run())
    assert row["client_cancelled"] is True and row["first_token_monotonic"] > row["dispatch_monotonic"]
    assert row["stream_completed"] is False and row["usage"] is None
    assert "cancel_after_first_content" not in body and body["stream_options"] == {"include_usage": True}
