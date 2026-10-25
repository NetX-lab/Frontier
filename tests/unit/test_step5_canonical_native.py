"""D18 pin, HTTP token transport, phase population and soft-observation regressions."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from tests.integration.step5_pd import build_stage, canonical_replay as cr
from tests.unit.test_step5_native_pd import stage_inputs

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
DEFINITION = REPO / "tests/integration/step5_pd/canonical_run_h200.json"


def metrics(count=0, dp=4):
    return "\n".join(
        f'vllm:{name}{{engine="{lane}",model_name="dp_pp_case"}} {value}'
        for lane in range(dp)
        for name, value in [(key, 0) for key in cr.GAUGES] + [
            ("request_success_total", count if lane == 0 else 0)]) + "\n"


def test_common_pins_and_disjoint_stable_template_repeat_ids():
    manifest, formal, warmup = cr.canonical_workload(WORKLOAD)
    rows = [row for repeat in (1, 2, 3) for phase, templates in (("warmup", warmup), ("formal", formal))
            for row in cr.phase_rows(manifest, templates, repeat, phase)]
    assert len(rows) == len({row["request_id"] for row in rows}) == 432
    assert sum(row["role"] == "formal" for row in rows) == 384
    assert sum(row["role"] == "warmup" for row in rows) == 48
    assert len({row["prompt_id"] for row in rows}) == 144
    assert all(row["prompt_id"] == row["template_request_id"] for row in rows)
    assert all(row["num_prefill_tokens"] == 6144 and row["num_decode_tokens"] == 1024 for row in rows)
    assert [row["arrived_at"] for row in rows if row["repeat"] == 2 and row["role"] == "formal"] == list(range(128))


def test_formal_diagnostic_throughputs_share_the_last_output_window():
    records = [{"status": "completed", "dispatch_monotonic": 100.0 + (index > 0),
                "last_token_monotonic": 200.0, "response_monotonic": 900.0,
                "num_prefill_tokens": 6144, "num_decode_tokens": 1024,
                "ttft_ms": 1000.0, "tpot_ms": 1.0, "e2e_ms": 99000.0}
               for index in range(128)]
    result = cr.formal_metrics(records)
    assert result["window_s"] == 100.0
    assert result["total_thpt_tokens_s"] == 9175.04
    assert result["output_tokens"] == 131072
    assert result["output_thpt_tokens_s"] == 1310.72
    assert result["request_thpt_requests_s"] == 1.28
    records[-1]["status"] = "failed"
    assert cr.formal_metrics(records) is None


def test_native_workload_p90_uses_nearest_rank_and_keeps_mean():
    records = [{"status": "completed", "dispatch_monotonic": 100.0,
                "last_token_monotonic": 200.0, "num_prefill_tokens": 6144,
                "num_decode_tokens": 1024, "ttft_ms": index, "tpot_ms": index * 2,
                "e2e_ms": index * 3} for index in range(1, 129)]
    result = cr.formal_metrics(records)
    assert result["ttft_ms"]["mean"] == 64.5
    assert result["ttft_ms"]["p90"] == 116
    assert result["ttft_ms"]["p95"] == 122
    assert result["tpot_ms"]["p90"] == 232
    assert result["e2e_ms"]["p90"] == 348


@pytest.mark.parametrize("member", ["manifest.json", "SHA256SUMS", "trace.csv", "request_ids.json", "warmup_pool.csv"])
def test_modified_common_member_is_rejected(tmp_path, member):
    directory = tmp_path / "workload"
    shutil.copytree(WORKLOAD, directory)
    with (directory / member).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="mismatch"):
        cr.canonical_workload(directory)


@pytest.mark.parametrize("kind,fault", [("canonical", fault) for fault in (
    None, "one_http_failure", "idle_failure", "checkpoint_failure", "bad_usage", "text_only")]
    + [("qualification", None), ("qualification", "startup_missing"),
       ("workload", None), ("workload", "one_http_failure")])
@pytest.mark.parametrize("dp", [2, 4])
def test_all_repetitions_use_existing_client_and_real_http_token_deltas(tmp_path, monkeypatch, kind, fault, dp):
    workload = cr.canonical_workload(WORKLOAD)
    failure_id = workload[1][0]["request_id"] + "-repeat01"
    bodies = {role: {} for role in cr.ROLES}
    phase_calls = []
    checkpoint_counts = []
    qualification_idle_roles = set()
    warmup_busy_reads = {role: 0 for role in cr.ROLES}
    qualification_sends = []
    shared_replay = cr.replay

    async def replay(rows, *args, **kwargs):
        phase_calls.append(copy.deepcopy(rows))
        # Put the clock origin in the past to test all exact HTTP bodies without a
        # 426-second wall-clock wait. Production scheduling remains untouched.
        return await shared_replay(rows, args[0], args[1], -128, **kwargs)

    monkeypatch.setattr(cr, "replay", replay)
    if fault == "idle_failure":
        async def unavailable(*args, **kwargs):
            raise RuntimeError("metrics unavailable")
        monkeypatch.setattr(cr, "observe_idle", unavailable)

    async def exercise():
        concurrent_arrivals = 0
        all_concurrent_arrived = asyncio.Event()

        def native(role):
            async def complete(request):
                nonlocal concurrent_arrivals
                rid = request.headers["X-Request-Id"]
                body = await request.json()
                bodies[role][rid] = body
                if kind == "qualification":
                    is_concurrent = rid.removeprefix("qualification-").removesuffix("-repeat01") in {
                        row["request_id"] for row in workload[1][:8]}
                    qualification_sends.append((role, is_concurrent))
                    if is_concurrent:
                        assert qualification_idle_roles == set(cr.ROLES)
                        if role == "decode":
                            concurrent_arrivals += 1
                            if concurrent_arrivals == 8:
                                all_concurrent_arrived.set()
                            await asyncio.wait_for(all_concurrent_arrived.wait(), timeout=2)
                if role == "prefill":
                    return web.json_response({"choices": [{"finish_reason": "length"}],
                        "kv_transfer_params": {"transfer_mode": "pull", "do_remote_prefill": True,
                            "do_remote_decode": False, "remote_engine_id": "p0", "remote_request_id": rid,
                            "remote_host": "10.0.0.1", "remote_port": 5600, "remote_block_ids": [[0, 1]]}})
                if fault == "one_http_failure" and rid == failure_id:
                    return web.json_response({"error": "native failure"}, status=500)
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                for index in range(2):
                    choice = {"index": 0, "text": "", "token_ids": [index + 200] * 512,
                              "finish_reason": "length" if index else None}
                    if fault == "text_only" and rid == failure_id:
                        choice.update(text="not a token-ID delta", token_ids=[])
                    if index == 0:
                        choice["prompt_token_ids"] = body["prompt"]
                    await response.write(("data: " + json.dumps({"choices": [choice]}) + "\n\n").encode())
                await asyncio.sleep(.001)
                await response.write(("data: " + json.dumps({"choices": [], "usage": {
                    "prompt_tokens": 6144, "completion_tokens": 1023 if fault == "bad_usage" and rid == failure_id else 1024}})
                    + "\n\ndata: [DONE]\n\n").encode())
                await response.write_eof()
                return response
            return complete

        apps = {}
        for role in cr.ROLES:
            apps[role] = web.Application()
            apps[role].router.add_post("/v1/completions", native(role))

            async def page(request, role=role):
                text = metrics(len(bodies[role]), dp)
                if fault == "startup_missing" and not bodies[role]:
                    text = "\n".join(line for line in text.splitlines() if "request_success_total" in line)
                if kind == "qualification" and len(bodies[role]) == 16:
                    warmup_busy_reads[role] += 1
                    if warmup_busy_reads[role] == 1:
                        text = text.replace('num_requests_running{engine="0",model_name="dp_pp_case"} 0',
                                            'num_requests_running{engine="0",model_name="dp_pp_case"} 1')
                    else:
                        qualification_idle_roles.add(role)
                return web.Response(text=text)

            apps[role].router.add_get("/metrics", page)
        async with TestServer(apps["prefill"]) as p, TestServer(apps["decode"]) as d:
            urls = {"prefill": str(p.make_url("")).rstrip("/"), "decode": str(d.make_url("")).rstrip("/")}
            app = cr.build_app(urls["prefill"], urls["decode"], "", "", tmp_path / "proxy.jsonl",
                               vllm_version="step5")
            async with TestServer(app) as proxy:
                def checkpoint(result, records):
                    checkpoint_counts.append(len(records))
                    if fault == "checkpoint_failure":
                        raise OSError("injected auxiliary write failure")

                return await cr.replay_repetitions(workload, proxy.port, 128896, 0, urls,
                    {"dp_size": dp, "kind": kind, "idle_timeout_s": .1,
                     "idle_poll_s": .001, "concurrent_requests": 8}, checkpoint=checkpoint)

    result, records = asyncio.run(exercise())
    assert result["replay_kind"] == kind
    expected_count = 24 if kind == "qualification" else 432
    assert len(records) == len(bodies["prefill"]) == len(bodies["decode"]) == expected_count
    assert [len(rows) for rows in phase_calls] == ([16, 8] if kind == "qualification" else [16, 128] * 3)
    for rows in phase_calls:
        expected = [0.0] * 8 if rows[0]["segment"] == "concurrent" else list(range(len(rows)))
        assert [row["arrived_at"] for row in rows] == expected
    assert len(result["idle_boundaries"]) == (2 if kind == "qualification" else 6)
    assert result["startup_observation"]["diagnostic_only"]
    if kind == "qualification":
        assert len({row["request_id"] for row in records}) == 24
        assert {row["role"] for row in records} == {"diagnostic"}
        assert all(row["request_id"].startswith("qualification-") for row in records)
        assert qualification_idle_roles == set(cr.ROLES)
        assert all(count >= 2 for count in warmup_busy_reads.values())
        warmup_boundary = result["idle_boundaries"][0]
        assert warmup_boundary["boundary"] == "after_repeat01_warmup" and warmup_boundary["verified"]
        assert len(warmup_boundary["observations"]) >= 2
        assert all(row["completed_requests"] == 16
                   for row in warmup_boundary["observations"][-1]["roles"].values())
        assert sum(is_concurrent for _, is_concurrent in qualification_sends) == 16
        assert "metrics" not in result["repetitions"][0]
        assert result["repeat_metric_means"] is None
    assert result["reference_admission"] == result["runtime_qualification"] == "UNVERIFIED"
    failures = [record for record in records if record["status"] != "completed"]
    if fault in ("one_http_failure", "bad_usage", "text_only"):
        assert failures and len(failures) < 432
        assert result["status"] == "REFERENCE_UNSTABLE"
        assert result["repetitions"][-1]["metrics"]["formal_requests"] == 128
    else:
        assert not failures
        assert result["client_workload_verified"] is (fault in (None, "startup_missing"))
    prompts = {}
    for record in records:
        body = bodies["decode"][record["request_id"]]
        assert body["stream"] and body["return_token_ids"] and not body["add_special_tokens"] and not body["echo"]
        assert body["ignore_eos"] and body["min_tokens"] == body["max_tokens"] == 1024
        assert len(body["prompt"]) == 6144
        template = record["template_request_id"]
        if template in prompts:
            assert body["prompt"] == prompts[template]
        prompts[template] = body["prompt"]
        if record["status"] == "completed":
            assert record["generated_token_count"] == 1024
            assert len(record["token_id_deltas"]) == 2  # Not 1024 SSE chunks.
            assert record["dispatch_monotonic"] <= record["first_token_monotonic"] <= record["last_token_monotonic"]
            assert record["last_token_monotonic"] < record["response_monotonic"]
            assert record["tpot_ms"] == pytest.approx((record["e2e_ms"] - record["ttft_ms"]) / 1023)
            assert record["accepted_prompt_sha256"] == record["prompt_sha256"]
    for repeat in result["repetitions"]:
        if kind == "qualification":
            continue
        population = [record for record in records if record["repeat"] == repeat["repeat"] and record["role"] == "formal"]
        actual = repeat["metrics"]
        if all(record["status"] == "completed" for record in population):
            assert actual["total_tokens"] == 917504
            assert actual["window_s"] == max(row["last_token_monotonic"] for row in population) - min(
                row["dispatch_monotonic"] for row in population)
            assert actual["total_thpt_tokens_s"] == 917504 / actual["window_s"]
            assert actual["output_tokens"] == 131072
            assert actual["output_thpt_tokens_s"] == 131072 / actual["window_s"]
            assert actual["request_thpt_requests_s"] == 128 / actual["window_s"]
        else:
            assert actual is None
    assert max(checkpoint_counts) == expected_count


def test_actual_qualification02_startup_snapshot_is_diagnostic_but_warmed_idle_is_strict():
    evidence = json.loads((REPO / "tests/fixtures/step5_pd/startup_metrics_qualification02.json").read_text())
    snapshots = evidence["boundaries"]
    for page in snapshots["before_repeat01_warmup"].values():
        with pytest.raises(ValueError, match="Missing native"):
            cr.idle_metrics(page, 4)
    for boundary, expected in (("after_repeat01_warmup", 16), ("after_repeat01_concurrent", 24)):
        for page in snapshots[boundary].values():
            row = cr.idle_metrics(page, 4)
            assert row["idle"] and row["completed_requests"] == expected

    async def exercise():
        async def page(request):
            return web.Response(text=snapshots["before_repeat01_warmup"]["prefill"])
        app = web.Application()
        app.router.add_get("/metrics", page)
        async with TestServer(app) as server:
            urls = dict.fromkeys(cr.ROLES, str(server.make_url("")).rstrip("/"))
            diagnostic = await cr.observe_idle(urls, 4, .1, .001, diagnostic_only=True)
            assert not diagnostic["verified"] and len(diagnostic["observations"]) == 1
            assert diagnostic["observations"][0]["errors"]
            strict = await cr.observe_idle(urls, 4, .02, .001, dict.fromkeys(cr.ROLES, 16))
            assert not strict["verified"]
    asyncio.run(exercise())


def test_existing_client_dispatch_is_open_loop_and_isolates_a_broken_sse_response():
    from tests.comparison.dp_placement_pp import vllm_replay

    received = {}

    async def exercise():
        async def complete(request):
            rid = request.headers["X-Request-Id"]
            received[rid] = asyncio.get_running_loop().time()
            if rid == "slow":
                await asyncio.sleep(.08)
            body = await request.json()
            if rid == "broken":
                return web.Response(text="data: not-json\n\n", content_type="text/event-stream")
            choice = {"text": "", "token_ids": [200, 201], "prompt_token_ids": body["prompt"],
                      "finish_reason": "length"}
            events = [{"choices": [choice]}, {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}]
            return web.Response(text="".join("data: " + json.dumps(event) + "\n\n" for event in events)
                                + "data: [DONE]\n\n", content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        rows = [{"request_id": rid, "segment": "formal", "role": "formal", "arrived_at": index * .01,
                 "num_prefill_tokens": 3, "num_decode_tokens": 2, "stream": True, "return_token_ids": True}
                for index, rid in enumerate(("slow", "broken", "healthy"))]
        async with TestServer(app) as server:
            return await vllm_replay.replay(rows, server.port, 128896, .01, vllm_version="step5")

    clock, records = asyncio.run(exercise())
    by_id = {record["request_id"]: record for record in records}
    assert set(received) == {"slow", "broken", "healthy"}
    assert by_id["broken"]["status"] == "failed" and "JSONDecodeError" in by_id["broken"]["error"]
    assert by_id["healthy"]["status"] == "received"
    assert by_id["healthy"]["dispatch_monotonic"] < by_id["slow"]["response_monotonic"]
    assert all(row["dispatch_monotonic"] >= clock["origin_monotonic"] + row["arrived_at"] for row in records)
    assert by_id["slow"]["dispatch_lateness_s"] >= 0


@pytest.mark.parametrize("fault", ["missing_lane", "busy", "stale", "nan"])
def test_idle_does_not_accept_missing_busy_stale_or_nonfinite_native_metrics(fault):
    page = metrics(144)
    if fault == "missing_lane":
        page = "\n".join(line for line in page.splitlines() if 'engine="3"' not in line)
    if fault == "busy":
        page = page.replace('num_requests_running{engine="0",model_name="dp_pp_case"} 0',
                            'num_requests_running{engine="0",model_name="dp_pp_case"} 1')
    if fault == "nan":
        page = page.replace(" 0\n", " nan\n", 1)
    if fault in ("missing_lane", "nan"):
        with pytest.raises(ValueError):
            cr.idle_metrics(page, 4)
    else:
        parsed = cr.idle_metrics(page, 4)
        assert parsed["idle"] is (fault != "busy")
        if fault == "stale":
            async def exercise():
                async def respond(request):
                    return web.Response(text=page)
                app = web.Application()
                app.router.add_get("/metrics", respond)
                async with TestServer(app) as server:
                    url = str(server.make_url("")).rstrip("/")
                    return await cr.observe_idle({role: url for role in cr.ROLES}, 4, .02, .001,
                                                 {role: 145 for role in cr.ROLES})
            assert not asyncio.run(exercise())["verified"]


def test_canonical_stage_is_full_h200_topology_without_r2_hooks(tmp_path):
    inputs = stage_inputs(tmp_path)
    definition = json.loads(DEFINITION.read_text())
    definition["probe_helpers"] = inputs["definition"]["probe_helpers"]
    stage = tmp_path / "workspace/stage"
    config = build_stage.build(definition, stage=stage, workspace=stage.parent,
        output_dir=Path("/mnt/codesign-exp/ycfeng/step5_d18_test"), job_tag="d18test", source={"commit": "test"},
        **{key: value for key, value in inputs.items() if key != "definition"})
    cr.canonical_workload(stage / "trace/workload")
    assert not (stage / "r2_bootstrap").exists()
    assert "pd_state_observation" not in config and "role_num_gpu_blocks_override" not in config
    assert config["mode"] == "clean" and "serve_observers" not in config
    assert Path(config["replay_script"]).name == "canonical_replay.py"
    rows = json.loads((stage / "preflight.json").read_text())
    assert [(row["role"], row["lane"]) for row in rows] == [(role, lane) for role in cr.ROLES for lane in range(4)]
    for row in rows:
        command = row["server_command"]
        for flag, value in (("--max-model-len", "7168"), ("--max-num-batched-tokens", "8192"),
                            ("--max-num-seqs", "128"), ("--tensor-parallel-size", "8"),
                            ("--data-parallel-size", "4")):
            assert command[command.index(flag) + 1] == value
        assert "--no-enable-chunked-prefill" in command and "--no-enable-prefix-caching" in command
        assert "--speculative-config" not in command and "--enforce-eager" not in command
        assert "--enable-expert-parallel" in command
        assert "--cudagraph-metrics" in command
        compilation = json.loads(command[command.index("--compilation-config") + 1])
        assert compilation == {"cudagraph_mode": "FULL_AND_PIECEWISE", "max_cudagraph_capture_size": 256}
        assert not any("PD_STATE" in key for key in row["server_env_set"])
    for name, expected in config["tool_hashes"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("label, error", [("step5", FileNotFoundError),
                                          ("step5_position_cyclic", FileNotFoundError),
                                          ("0.10.2", ValueError)])
def test_workload_consumer_accepts_both_step_runtime_labels(tmp_path, monkeypatch, label, error):
    """The routing image must pass the adapter guard and reach the replay plan read."""
    from types import SimpleNamespace

    monkeypatch.setattr(cr, "runtime_version", lambda engine: label)
    engine = tmp_path / "engine.json"
    engine.write_text("{}")
    args = SimpleNamespace(engine_config=engine, mode="clean", trace_dir=tmp_path / "missing_trace")
    with pytest.raises(error):
        asyncio.run(cr.run(args))
