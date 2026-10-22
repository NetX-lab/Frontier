"""Case-aligned R2 caller/consumer controls; no native/reference qualification claims."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.integration.step5_pd import build_stage, canonical_replay, r2_canonical, r2_stage, r2_verdict, verify_r2_stage
from tests.unit.test_step5_native_pd import stage_inputs


def stage_capture(tmp_path, role_overrides=None):
    inputs = stage_inputs(tmp_path)
    source = tmp_path / "native"
    for module in r2_stage.RUNTIME_MODULES + r2_stage.SCHEMA_MODULES + r2_stage.CANONICAL_SCHEMA_MODULES:
        path = source / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{module}"""\n')
    run = r2_canonical.definition(source)
    if role_overrides is not None:
        run["role_num_gpu_blocks_override"] = role_overrides
    run["probe_helpers"] = inputs["definition"]["probe_helpers"]
    model_path = inputs["model_dir"] / "config.json"
    model = json.loads(model_path.read_text())
    model["text_config"].update(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"],
                               sparse_config={"apply_to_layer_types": ["full_attention"]})
    model_path.write_text(json.dumps(model))
    workspace = tmp_path / "workspace"
    stage = workspace / "stage"
    config = build_stage.build(run, stage=stage, workspace=workspace,
        output_dir=Path("/mnt/codesign-exp/ycfeng/step5-preview/canonical_g3_cpu_test"),
        model_dir=inputs["model_dir"], image_source_hashes=inputs["image_source_hashes"],
        probe_dir=inputs["probe_dir"], job_tag="g3cpu",
        source={"repository": str(r2_stage.REPO), "commit": "unit"})
    return run, config, stage


def test_d28_decode_only_control_preserves_golden_prefill_and_base_engine(tmp_path):
    (tmp_path / "baseline").mkdir()
    (tmp_path / "d28").mkdir()
    baseline_run, baseline, baseline_stage = stage_capture(tmp_path / "baseline")
    run, config, stage = stage_capture(tmp_path / "d28", {"decode": 32000})
    assert verify_r2_stage.verify(stage)["status"] == "PASS_OFFLINE_STAGE"
    assert config["engine"] == baseline["engine"]
    assert run["resources"] == baseline_run["resources"]
    assert run["waves"] == baseline_run["waves"]
    assert config["role_num_gpu_blocks_override"] == {"decode": 32000}
    assert "role_num_gpu_blocks_override" not in baseline
    golden = json.loads((baseline_stage / "preflight.json").read_text())
    plans = json.loads((stage / "preflight.json").read_text())
    for original, plan in zip(golden, plans):
        command = plan["server_command"][:]
        if plan["role"] == "decode":
            index = command.index("--num-gpu-blocks-override")
            assert command[index + 1] == "32000"
            del command[index:index + 2]
        else:
            assert "--num-gpu-blocks-override" not in command
        assert [value.replace(config["model_dir"], baseline["model_dir"])
                for value in command] == original["server_command"]
        assert (plan["role"], plan["lane"], plan["headless"], plan["urls"]) == (
            original["role"], original["lane"], original["headless"], original["urls"])
    assert config["pd_state_observation"]["local_reference_client_request_ids"] == (
        baseline["pd_state_observation"]["local_reference_client_request_ids"])


@pytest.mark.parametrize("overrides", [{"prefill": 32000}, {"decode": 31999},
    {"prefill": 32000, "decode": 32000}, {"decode": True}, {"decode": 32000.0}, {}, None])
def test_d28_definition_rejects_every_unapproved_pool_control(overrides):
    run = r2_canonical.definition(Path("native"))
    run["role_num_gpu_blocks_override"] = overrides
    with pytest.raises(ValueError, match="D28 decode-only"):
        r2_canonical.validate_definition(run)


@pytest.mark.parametrize("fault", ["prefill_flag", "missing_decode_flag", "wrong_count", "definition_map"])
def test_d28_offline_verifier_rejects_role_or_definition_drift(tmp_path, fault):
    _, config, stage = stage_capture(tmp_path, {"decode": 32000})
    plans = json.loads((stage / "preflight.json").read_text())
    if fault == "prefill_flag":
        plans[0]["server_command"] += ["--num-gpu-blocks-override", "32000"]
    elif fault == "missing_decode_flag":
        command = plans[2]["server_command"]
        index = command.index("--num-gpu-blocks-override")
        del command[index:index + 2]
    elif fault == "wrong_count":
        command = plans[3]["server_command"]
        command[command.index("--num-gpu-blocks-override") + 1] = "31999"
    else:
        config["role_num_gpu_blocks_override"] = {"decode": 31999}
    (stage / "preflight.json").write_text(json.dumps(plans))
    config["tool_hashes"]["preflight.json"] = build_stage.sha256(stage / "preflight.json")
    (stage / "config.json").write_text(json.dumps(config))
    with pytest.raises(AssertionError):
        verify_r2_stage.verify(stage)


def test_actual_canonical_capture_stage_preserves_native_case_and_original_prompts(tmp_path):
    run, config, stage = stage_capture(tmp_path)
    receipt = verify_r2_stage.verify(stage)
    assert receipt["requests"] == 163 and receipt["content_requests"] == 8
    assert receipt["native_readiness_only_requests"] == 155
    assert receipt["g3_qualified"] is False and receipt["clean_timing_eligible"] is False
    assert "role_num_gpu_blocks_override" not in config
    assert config["engine"]["api_server_count"] == 1
    assert len(config["pd_state_observation"]["source_hashes"]) == 13
    assert "vllm.v1.engine.core" in config["pd_state_observation"]["source_hashes"]
    assert config["pd_state_observation"]["max_capture_bytes_per_request"] > 139887104
    abort = json.loads((stage / "trace/abort_read6144/request_ids.json").read_text())["rows"][0]
    assert abort["cancel_after_decode_submit_s"] == 0.01 and abort["return_token_ids"] is True
    _, formal, warmup = canonical_replay.canonical_workload(stage / "canonical_workload")
    assert not {row['request_id'] for row in warmup} & {row['request_id'] for row in formal}
    for wave, templates in (("warmup16", warmup[:16]), ("concurrent8", formal[:8])):
        rows = json.loads((stage / f"trace/{wave}/request_ids.json").read_text())["rows"]
        assert [row["prompt_id"] for row in rows] == [row["request_id"] for row in templates]
        assert all(row["return_token_ids"] for row in rows)
    original = json.loads((stage / "trace/canonical6144/request_ids.json").read_text())["rows"][0]
    reused = json.loads((stage / "trace/reuse6144/request_ids.json").read_text())["rows"][0]
    assert original["request_id"] != reused["request_id"] and original["prompt_id"] == reused["prompt_id"]
    assert all(row["num_prefill_tokens"] == 6144 and row["num_decode_tokens"] == 1024
               for row in json.loads((stage / "trace/canonical6144/request_ids.json").read_text())["rows"])


@pytest.mark.parametrize("key,value", [
    ("api_server_count", 2), ("max_model_len", 2048), ("max_num_batched_tokens", 512),
    ("max_num_seqs", 4), ("enable_chunked_prefill", True), ("enforce_eager", True),
    ("enable_prefix_caching", True), ("tensor_parallel_size", 16), ("data_parallel_size", 4),
])
def test_canonical_capture_rejects_old_or_changed_engine(key, value):
    run = r2_canonical.definition(Path("native"))
    run["engine"][key] = value
    with pytest.raises(ValueError, match="frozen P0"):
        r2_canonical.validate_definition(run)


@pytest.mark.parametrize("fault", ["short_content", "short_output", "no_content", "context", "concurrent_content", "cancel", "duplicate", "old_blocks"])
def test_capture_definition_rejects_unsupported_wave_substitutions(fault):
    run = r2_canonical.definition(Path("native"))
    waves = {wave["name"]: wave for wave in run["waves"]}
    if fault == "short_content":
        waves["canonical6144"]["prompt_lengths"] = [1537]
    elif fault == "short_output":
        waves["canonical6144"]["num_decode_tokens"] = 17
    elif fault == "no_content":
        waves["canonical6144"]["capture_content"] = False
    elif fault == "context":
        waves["boundary6145"]["num_decode_tokens"] = 1024
    elif fault == "concurrent_content":
        waves["concurrent8"]["capture_content"] = True
    elif fault == "cancel":
        waves["cancel6144"].pop("cancel_after_first_content")
    elif fault == "duplicate":
        run["waves"].append(copy.deepcopy(run["waves"][0]))
    else:
        run["role_num_gpu_blocks_override"] = {"prefill": 1024, "decode": 512}
    with pytest.raises(ValueError):
        r2_canonical.validate_definition(run)


def client_receipt():
    return {"http_status": 200, "accepted_prompt_token_count": 6144,
        "prompt_sha256": "a" * 64, "accepted_prompt_sha256": "a" * 64,
        "generated_token_count": 1024, "token_id_deltas": [{"token_ids": [2] * 1024}],
        "usage": {"prompt_tokens": 6144, "completion_tokens": 1024},
        "stream_completed": True, "finish_reason": "length"}


@pytest.mark.parametrize("fault", ["none", "missing", "duplicate", "prompt", "hash", "count", "chunks", "stream", "reason", "usage", "delta", "token"])
def test_case_aligned_consumer_requires_actual_complete_native_token_population(fault):
    row = client_receipt()
    records = [row]
    if fault == "missing":
        records = []
    elif fault == "duplicate":
        records.append(copy.deepcopy(row))
    elif fault == "prompt":
        row["accepted_prompt_token_count"] = 6128
    elif fault == "hash":
        row["accepted_prompt_sha256"] = "b" * 64
    elif fault == "count":
        row["generated_token_count"] = 1023
    elif fault == "chunks":
        row["token_id_deltas"] = [{"token_ids": [2]}]
    elif fault == "stream":
        row["stream_completed"] = False
    elif fault == "reason":
        row["finish_reason"] = "stop"
    elif fault == "usage":
        row["usage"] = "corrupt"
    elif fault == "delta":
        row["token_id_deltas"] = [{"token_ids": None}]
    elif fault == "token":
        row["token_id_deltas"][0]["token_ids"][0] = True
    check = r2_verdict.Check()
    r2_verdict.check_canonical_requests({"c": {"wave": "canonical6144", "num_prefill_tokens": 6144,
        "num_decode_tokens": 1024}}, {("canonical6144", "c"): records}, check)
    assert (check.result()["status"] == "PASS") == (fault == "none"), check.result()


def test_exact_canonical_http_caller_preserves_proxy_handoff_and_native_token_counts(tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from tests.comparison.calibration import pd_proxy
    from tests.comparison.dp_placement_pp import vllm_replay

    async def run():
        bodies = {"prefill": [], "decode": []}
        params = {"transfer_mode": "pull", "do_remote_prefill": True, "do_remote_decode": False,
            "remote_engine_id": "P", "remote_request_id": "p", "remote_host": "127.0.0.1",
            "remote_port": 5600, "remote_block_ids": [[1]], "remote_num_tokens": 6128}
        async def prefill(request):
            bodies["prefill"].append(await request.json())
            return web.json_response({"kv_transfer_params": params})
        async def decode(request):
            body = await request.json()
            bodies["decode"].append(body)
            payload = {"choices": [{"index": 0, "prompt_token_ids": body["prompt"],
                "token_ids": [2] * body["max_tokens"], "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"]}}
            return web.Response(text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")
        p, d = web.Application(), web.Application()
        p.router.add_post("/v1/completions", prefill)
        d.router.add_post("/v1/completions", decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            proxy = pd_proxy.build_app(str(ps.make_url("/")).rstrip("/"), str(ds.make_url("/")).rstrip("/"),
                "", "", tmp_path / "proxy.jsonl", vllm_version="step5")
            async with TestServer(proxy) as server:
                rows = [{"request_id": f"c{i}", "prompt_id": f"template{i}", "segment": "concurrent8",
                    "role": "recorded", "arrived_at": 0, "num_prefill_tokens": 6144,
                    "num_decode_tokens": 1024, "stream": True, "return_token_ids": True} for i in range(8)]
                _, records = await vllm_replay.replay(rows, server.port, 32000, 0, vllm_version="step5")
        return bodies, records
    bodies, records = asyncio.run(run())
    assert len(records) == len(bodies["decode"]) == len(bodies["prefill"]) == 8
    assert all(not canonical_replay.validate_record(row) for row in records)
    assert all(body["max_tokens"] == 1 and body["stream"] is False for body in bodies["prefill"])
    assert all(body["kv_transfer_params"]["remote_num_tokens"] == 6128
               and body["max_tokens"] == body["min_tokens"] == 1024 and body["return_token_ids"]
               and len(body["prompt"]) == 6144 for body in bodies["decode"])


def test_preparation_command_never_claims_reference_or_launch_eligibility():
    report = r2_canonical.describe(Path("native"))
    assert report["sealed"] is report["submitted"] is report["ready_for_formal"] is False
    assert report["definition"]["canonical_g3"]["independent_reference"] == "EXISTING_D_FULL_LOCAL_PREFILL_NO_KV_TRANSFER_PARAMS"
    assert len(report["required_before_sealing"]) == 1


@pytest.mark.parametrize("prompt", [6143, 6144, 6145])
@pytest.mark.parametrize("source_delta", [0, -16, 16])
def test_canonical_native_truncated_prefix_is_not_backed_off_twice(prompt, source_delta):
    external = 16 * ((prompt - 16) // 16)
    source = [{"event": "source_hold", "engine_id": "P", "request_id": "p",
        "num_computed_tokens": external + source_delta,
        "kv_xfer_params": {"remote_num_tokens": external, "remote_engine_id": "P", "remote_request_id": "p"}}]
    reserve = {"event": "target_reserve", "engine_id": "D", "request_id": "d",
        "num_prompt_tokens": prompt, "num_external_tokens": external}
    target = [{"event": "read_metadata", "engine_id": "D", "request_id": "d", "tp_rank": rank,
        "remote_num_tokens": external} for rank in range(8)]
    check = r2_verdict.Check()
    counts = r2_verdict.token_attribution(source, target, reserve, {"engine": {"block_size": 16}}, check,
                                          frozen_prompt=prompt)
    assert (check.result()["status"] == "PASS") == (source_delta == 0), check.result()
    if prompt == 6144 and not source_delta:
        assert counts["source_computed_tokens"] == counts["external_tokens"] == 6128
        assert counts["prompt_tokens"] - counts["external_tokens"] == 16


@pytest.mark.parametrize('fault', ['none', 'deadline_only', 'already_ready', 'compute', 'early_free', 'missing_rank', 'pool_leak', 'cross_host', 'fence', 'not_aborted'])
def test_pending_read_abort_requires_actual_native_interval_and_conservation(fault):
    from tests.unit.test_step5_native_pd_r2_verdict import observations, configuration
    rows, _ = observations((6144,))
    source = [row for row in rows if row['role'] == 'prefill' and row['event'] != 'state_registration'
              and row.get('observer_fence') is not True]
    target = [row for row in rows if row['role'] == 'decode'
              and row['event'] not in ('state_registration', 'frontend_select', 'target_compute')
              and row.get('observer_fence') is not True]
    abort = next(row.copy() for row in target if row['event'] == 'target_reserve')
    abort.update(event='abort', monotonic_ns=abort['monotonic_ns'] + 15, status='FINISHED_ABORTED', before={
        'request_status': 'WAITING_FOR_REMOTE_KVS', 'native_receive_completed': False})
    target.append(abort)
    if fault == 'deadline_only':
        abort['monotonic_ns'] = 5
    elif fault == 'already_ready':
        abort['before']['native_receive_completed'] = True
    elif fault == 'compute':
        row = abort.copy()
        row['event'] = 'target_compute'
        target.append(row)
    elif fault == 'early_free':
        next(row for row in target if row['event'] == 'target_free')['monotonic_ns'] = 22
    elif fault == 'missing_rank':
        target = [row for row in target if not (row['event'] == 'read_network_done' and row.get('tp_rank') == 7)]
    elif fault == 'pool_leak':
        next(row for row in target if row['event'] == 'target_free')['pool_free_after'] -= 1
    elif fault == 'cross_host':
        abort['host'] = 'different_clock_domain'
    elif fault == 'fence':
        abort['observer_fence'] = True
    elif fault == 'not_aborted':
        abort['status'] = 'FINISHED_LENGTH_CAPPED'
    check = r2_verdict.Check()
    r2_verdict.check_pending_read_abort(source, target, configuration(), [], check)
    assert (check.result()['status'] == 'PASS') is (fault == 'none'), check.result()


def test_proxy_decode_deadline_cancels_native_client_before_first_content(tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from tests.comparison.calibration import pd_proxy
    from tests.comparison.dp_placement_pp import vllm_replay

    async def run():
        received = []
        disconnected = asyncio.Event()
        async def prefill(request):
            return web.json_response({'kv_transfer_params': {'transfer_mode': 'pull',
                'do_remote_prefill': True, 'do_remote_decode': False, 'remote_engine_id': 'P',
                'remote_request_id': 'p', 'remote_host': '127.0.0.1', 'remote_port': 5600,
                'remote_block_ids': [[1]], 'remote_num_tokens': 6128}})
        async def decode(request):
            try:
                received.append(await request.json())
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                disconnected.set()
                raise
            if request.transport is None or request.transport.is_closing():
                disconnected.set()
            return web.Response(text='unreached')
        p, d = web.Application(), web.Application()
        p.router.add_post('/v1/completions', prefill)
        d.router.add_post('/v1/completions', decode)
        async with TestServer(p) as ps, TestServer(d) as ds:
            app = pd_proxy.build_app(str(ps.make_url('/')).rstrip('/'), str(ds.make_url('/')).rstrip('/'),
                '', '', tmp_path / 'abort_proxy.jsonl', vllm_version='step5')
            async with TestServer(app) as server:
                _, records = await vllm_replay.replay([{'request_id': 'abort', 'segment': 'abort',
                    'role': 'recorded', 'arrived_at': 0, 'num_prefill_tokens': 6144,
                    'num_decode_tokens': 1024, 'stream': True, 'return_token_ids': True,
                    'cancel_after_decode_submit_s': 0.02}], server.port, 32000, 0, vllm_version='step5')
                await asyncio.wait_for(disconnected.wait(), 1)
        return records[0], received
    row, received = asyncio.run(run())
    assert row['client_cancelled'] is True and row['cancel_origin'] == 'proxy_native_decode_client'
    assert row['first_token_monotonic'] is None and row['stream_completed'] is False
    assert row['generated_token_count'] == 0 and row['usage'] is None
    assert len(received) == 1 and 'cancel_after_decode_submit_s' not in received[0]
    proof = json.loads((tmp_path / 'abort_proxy.jsonl').read_text())
    assert proof['decode_cancelled'] and proof['decode_cancel_monotonic'] > proof['decode_submit_monotonic']


def test_abort_observation_preserves_native_request_id_iterable(tmp_path):
    from tests.integration.step5_pd import r2_observation
    from tests.unit.test_step5_native_pd_r2_observation import configuration
    observer = r2_observation.Observation(configuration(tmp_path))
    request_ids = iter(['case'])
    scheduler = NS(requests={'case': NS(status=NS(name='WAITING_FOR_REMOTE_KVS'))},
                   finished_recving_kv_req_ids=set())
    snapshot = observer.abort_before(scheduler, {'request_ids': request_ids})
    assert list(request_ids) == ['case']
    assert snapshot == {'case': {'request_status': 'WAITING_FOR_REMOTE_KVS', 'native_receive_completed': False}}
