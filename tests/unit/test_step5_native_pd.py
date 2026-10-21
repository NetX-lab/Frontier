"""CPU checks of the Step-5 native P/D harness: metrics, staging, wave driver, worker options, verdict."""

from __future__ import annotations

import hashlib
import asyncio
import json
from pathlib import Path
import re

import pytest

from tests.comparison.calibration import serving_metrics
from tests.integration.step5_pd import build_stage, run_waves, submit_native_pd, verdict
from tests.unit.test_native_pd_worker import exercise
from tests.comparison.calibration import native_pd_worker
from tests.comparison.dp_placement_pp import vllm_replay

REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "tests/integration/step5_pd/r1_run.json"
PACKAGE = "/usr/local/lib/python3.12/dist-packages/vllm"
SOURCES = {"local_compute": 25, "local_cache_hit": 0, "external_kv_transfer": 4080}


def page(*, reads=(0.0, 0.0), descriptors=(0.0, 0.0), sources=None, failures=0.0, usage=(0.0, 0.0),
         requests=0.0) -> str:
    """One /metrics page of a two-engine role, with cumulative values."""
    lines = ["# HELP vllm:nixl_bytes_transferred bytes", "# TYPE vllm:nixl_bytes_transferred histogram"]
    for engine in ("0", "1"):
        labels = f'engine="{engine}",model_name="step5"'
        index = int(engine)
        lines += [
            f"vllm:nixl_bytes_transferred_sum{{{labels}}} {reads[index]}",
            f"vllm:nixl_bytes_transferred_count{{{labels}}} {1.0 if reads[index] else 0.0}",
            f"vllm:nixl_num_descriptors_sum{{{labels}}} {descriptors[index]}",
            f"vllm:nixl_num_descriptors_count{{{labels}}} {1.0 if descriptors[index] else 0.0}",
            f"vllm:kv_cache_usage_perc{{{labels}}} {usage[index]}",
            f"vllm:e2e_request_latency_seconds_count{{{labels}}} {requests if index == 0 else 0.0}",
            f"vllm:e2e_request_latency_seconds_sum{{{labels}}} {requests * 0.5 if index == 0 else 0.0}",
        ]
        lines += [f"{name}_total{{{labels}}} {failures if index == 0 else 0.0}"
                  for name in serving_metrics.NIXL_FAILURE_COUNTERS]
        for source in serving_metrics.PROMPT_TOKEN_SOURCES:
            value = (sources or {}).get(source, 0) if index == 0 else 0
            lines.append(f'vllm:prompt_tokens_by_source_total{{{labels},source="{source}"}} {float(value)}')
    return "\n".join(lines) + "\n"


def test_prometheus_text_parses_labels_escapes_and_rejects_garbage():
    samples = serving_metrics.parse_prometheus(
        '# comment\nvllm:x_total{engine="0",path="a\\"b\\\\c\\nd"} 3.5 1700000000\nup 1\n')
    assert samples == [("vllm:x_total", {"engine": "0", "path": 'a"b\\c\nd'}, 3.5), ("up", {}, 1.0)]
    with pytest.raises(ValueError):
        serving_metrics.parse_prometheus("not a sample line at all {")


def test_transfer_delta_reports_per_engine_changes_and_absent_series():
    before = serving_metrics.parse_prometheus(page())
    after = serving_metrics.parse_prometheus(page(reads=(0.0, 4096.0), descriptors=(0.0, 30.0),
                                                  sources=SOURCES, usage=(0.0, 0.25), requests=1))
    delta = serving_metrics.transfer_delta(before, after)
    bytes_sum = delta["nixl_transfers"]["vllm:nixl_bytes_transferred"]["sum"]
    assert bytes_sum == {"present": True, "per_engine": {"0": 0.0, "1": 4096.0}, "total": 4096.0}
    assert delta["prompt_tokens_by_source"]["external_kv_transfer"]["total"] == 4080
    assert delta["kv_cache_usage_perc"] == {"0": 0.0, "1": 0.25}
    assert delta["requests"]["vllm:e2e_request_latency_seconds"]["count"]["total"] == 1
    assert delta["prompt_tokens"] == {"present": False, "per_engine": {}, "total": 0}
    assert delta["nixl_transfers"]["vllm:nixl_xfer_time_seconds"]["count"]["present"] is False


@pytest.mark.parametrize("length,expected", [
    (4105, {"local_compute": 25, "local_cache_hit": 0, "external_kv_transfer": 4080}),
    (2064, {"local_compute": 16, "local_cache_hit": 0, "external_kv_transfer": 2048}),
    (32, {"local_compute": 16, "local_cache_hit": 0, "external_kv_transfer": 16}),
])
def test_pull_prompt_split_leaves_one_block_and_the_partial_tail_to_decode(length, expected):
    assert serving_metrics.expected_pull_prompt_sources(length, 16) == expected


@pytest.mark.parametrize("fault", [None, "no_read", "failure", "split", "absent_failure"])
def test_transfer_checks_need_reads_no_failure_change_and_the_expected_split(fault):
    pages = {"before": page(), "prefill": page(failures=1.0 if fault == "failure" else 0.0),
             "decode": page(reads=(0.0, 0.0) if fault == "no_read" else (4096.0, 0.0),
                            descriptors=(30.0, 0.0),
                            sources=SOURCES | ({"local_compute": 26} if fault == "split" else {}))}
    samples = {name: serving_metrics.parse_prometheus(text) for name, text in pages.items()}
    if fault == "absent_failure":
        samples["decode"] = [row for row in samples["decode"] if row[0] != "vllm:nixl_num_failed_transfers_total"]
        samples["before"] = [row for row in samples["before"] if row[0] != "vllm:nixl_num_failed_transfers_total"]
    prefill = serving_metrics.transfer_delta(samples["before"], samples["prefill"])
    decode = serving_metrics.transfer_delta(samples["before"], samples["decode"])
    checks = serving_metrics.transfer_checks(prefill, decode, SOURCES)
    assert checks["nixl_counters"]["passed"] is (fault in (None, "split"))
    assert checks["decode_prompt_sources"]["passed"] is (fault != "split")
    if fault == "absent_failure":
        assert checks["nixl_counters"]["failure_changes"]["decode"]["vllm:nixl_num_failed_transfers"] is None


def stage_inputs(tmp_path: Path) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Step5ForConditionalGeneration"], "text_config": {"vocab_size": 128896}}))
    (model / "tokenizer_config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    probes = tmp_path / "probes"
    probes.mkdir()
    definition = json.loads(R1.read_text())
    for name in definition["probe_helpers"]:
        (probes / name).write_text(f"# {name}\n")
        definition["probe_helpers"][name] = hashlib.sha256((probes / name).read_bytes()).hexdigest()
    listing = tmp_path / "sha256_in_image.txt"
    listing.write_text(f"{'a' * 64}  {PACKAGE}/v1/engine/core.py\n{'b' * 64}  /usr/bin/other\n")
    return dict(definition=definition, model_dir=model, probe_dir=probes, image_source_hashes=listing)


@pytest.fixture
def staged(tmp_path):
    inputs = stage_inputs(tmp_path)
    stage = tmp_path / "workspace" / "stage"
    config = build_stage.build(
        inputs["definition"], stage=stage, workspace=tmp_path / "workspace",
        output_dir=Path("/mnt/codesign-exp/ycfeng/step5-preview/native_pd_test"),
        model_dir=inputs["model_dir"], image_source_hashes=inputs["image_source_hashes"],
        probe_dir=inputs["probe_dir"], job_tag="r1test", source={"commit": "test"})
    return stage, config, inputs


def test_h200_run_definition_changes_only_the_hardware():
    h800 = json.loads(R1.read_text())
    h200 = json.loads(R1.with_name("r1_run_h200.json").read_text())
    assert h200["resources"].pop("charged_group") == "step_main"
    assert h200["resources"].pop("gpu_type") == "H200"
    for definition in (h800, h200):
        definition.pop("description")
    h800["resources"].pop("charged_group")
    h800["resources"].pop("gpu_type")
    assert h200 == h800


def test_stage_holds_every_worker_config_key_and_the_harness_layout(staged):
    stage, config, _ = staged
    worker_source = (REPO / "tests/comparison/calibration/native_pd_worker.py").read_text()
    required = set(re.findall(r"config\['(\w+)'\]", worker_source))
    assert required <= set(config), required - set(config)
    submit_source = (REPO / "tests/integration/step5_pd/submit_native_pd.py").read_text()
    assert set(re.findall(r'config\["(\w+)"\]', submit_source)) <= set(config)
    assert config["expected_source_hashes"] == {"vllm/v1/engine/core.py": "a" * 64}
    assert config["mode"] == "clean" and "serve_observers" not in config
    assert Path(config["replay_script"]).is_file() and Path(config["script"]).is_file()
    assert (Path(config["harness_dir"]) / "pd_replay.py").is_file()
    for name in ("probe_nixl_read.py", "probe_runtime_inventory.py", "native_pd_worker.py"):
        assert (stage / name).is_file() and name in config["tool_hashes"]
    engine = json.loads((stage / "engine.json").read_text())
    assert engine == config["engine"]
    assert engine["vllm_version"] == "step5" and engine["skip_tokenizer_init"] is False


def test_stage_traces_and_expected_decode_split_follow_the_waves(staged):
    stage, _, _ = staged
    waves = json.loads((stage / "trace/waves.json").read_text())
    assert waves["waves"] == ["w1", "w2"] and waves["gating_wave"] == "w1"
    assert waves["release_wait_s"] == 35
    assert waves["expected_decode_prompt_sources"] == {
        "w1": SOURCES, "w2": {"local_compute": 41, "local_cache_hit": 0, "external_kv_transfer": 6128}}
    rows = json.loads((stage / "trace/w2/request_ids.json").read_text())["rows"]
    assert [(row["request_id"], row["num_prefill_tokens"], row["arrived_at"], row["role"]) for row in rows] == [
        ("r1test-w2-0", 4105, 0.0, "recorded"), ("r1test-w2-1", 2064, 0.01, "recorded")]


def test_preflight_maps_two_dp_lanes_per_role_with_headless_peers_and_nixl_ports(staged):
    stage, _, _ = staged
    rows = json.loads((stage / "preflight.json").read_text())
    assert [(row["role"], row["lane"], row["headless"]) for row in rows] == [
        ("prefill", 0, False), ("prefill", 1, True), ("decode", 0, False), ("decode", 1, True)]
    for index, row in enumerate(rows):
        command = row["server_command"]
        assert command[command.index("--host") + 1] == f"10.255.0.{index + 1}"
        assert ("--headless" in command) is row["headless"]
        assert command[command.index("--data-parallel-size") + 1] == "2"
        assert command[command.index("--data-parallel-size-local") + 1] == "1"
        if row["headless"]:
            assert command[command.index("--data-parallel-start-rank") + 1] == "1"
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        assert transfer["kv_connector"] == "NixlConnector"
        assert transfer["kv_connector_extra_config"]["kv_lease_duration"] == 30
        env = row["server_env_set"]
        assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == f"10.255.0.{index + 1}"
        assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == ("5600" if row["role"] == "prefill" else "5700")
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert row["urls"] == {"prefill": "http://10.255.0.1:8100", "decode": "http://10.255.0.3:8200"}


def test_stage_rejects_reuse_and_outputs_outside_the_cloud_root(tmp_path, staged):
    stage, _, inputs = staged
    arguments = dict(workspace=stage.parent, model_dir=inputs["model_dir"],
                     image_source_hashes=inputs["image_source_hashes"], probe_dir=inputs["probe_dir"],
                     job_tag="r1test", source={})
    with pytest.raises(FileExistsError):
        build_stage.build(inputs["definition"], stage=stage,
                          output_dir=Path("/mnt/codesign-exp/ycfeng/x"), **arguments)
    with pytest.raises(ValueError, match="Cloud output"):
        build_stage.build(inputs["definition"], stage=stage.parent / "other",
                          output_dir=Path("/mnt/codesign-exp/other"), **arguments)
    with pytest.raises(ValueError, match="inside the workspace"):
        build_stage.build(inputs["definition"], stage=tmp_path / "outside",
                          output_dir=Path("/mnt/codesign-exp/ycfeng/x"), **arguments)


def test_wave_driver_records_each_wave_and_returns_the_gating_status(tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "waves.json").write_text(json.dumps({
        "waves": ["w1", "w2"], "gating_wave": "w1", "release_wait_s": 35, "scrape_timeout_s": 10}))
    calls, sleeps = [], []
    pages = {"prefill": iter([page(), page(), page(), page()]),
             "decode": iter([page(), page(reads=(10.0, 0.0), sources=SOURCES), page(), page()])}

    def replay(argv):
        calls.append(argv)
        if Path(argv[1]).name == "w2":
            raise SystemExit("w2 failed")
        return 0

    def scrape(url, timeout):
        text = next(pages["prefill" if "8100" in url else "decode"])
        return {"url": url + "/metrics", "text": text, "samples": serving_metrics.parse_prometheus(text),
                "error": None}

    monkeypatch.setattr(run_waves.pd_replay, "main", replay)
    monkeypatch.setattr(run_waves, "scrape", scrape)
    monkeypatch.setattr(run_waves.time, "sleep", sleeps.append)
    output = tmp_path / "out"
    status = run_waves.main(["--trace-dir", str(trace), "--output-dir", str(output),
                             "--prefill-url", "http://p:8100", "--decode-url", "http://d:8200",
                             "--mode", "clean", "--drain-s", "3"])
    assert status == 0 and sleeps == [35]
    assert calls[0][:4] == ["--trace-dir", str(trace / "w1"), "--output-dir", str(output / "w1")]
    assert calls[0][-4:] == ["--mode", "clean", "--drain-s", "3"]
    report = json.loads((output / "native_transfer_metrics.json").read_text())
    assert report["statuses"] == {"w1": 0, "w2": None} and "w2 failed" in report["errors"]["w2"]
    assert sorted(path.name for path in (output / "metrics").iterdir()) == sorted(
        f"{point}_{role}.prom" for point in ("baseline", "w1", "w2", "released")
        for role in ("prefill", "decode"))
    w1 = report["changes"]["w1"]["decode"]
    assert w1["prompt_tokens_by_source"]["external_kv_transfer"]["total"] == 4080


def test_worker_clean_mode_uses_the_staged_driver_and_mirrors_evidence(tmp_path):
    def overrides(directory):
        (directory / "output/replay").mkdir(parents=True)
        (directory / "output/replay/marker.json").write_text("{}")
        return {"mode": "clean", "serve_observers": None,
                "replay_script": str(directory / "run_waves.py"),
                "evidence_dir": str(directory / "evidence")}

    result = exercise(native_pd_worker, tmp_path, "control", overrides)
    assert result["worker_returncode"] == 0 and not result["observation_errors"]
    command = result["replay_command"]
    assert command[1] == str(result["directory"] / "run_waves.py")
    assert command[command.index("--mode") + 1] == "clean" and "--observer-config" not in command
    evidence = result["directory"] / "evidence"
    assert json.loads((evidence / "nodes/host0/result.json").read_text())["stage"] == "finished"
    assert (evidence / "replay/marker.json").is_file()


def write_evidence(stage: Path, config: dict, *, fault: str | None = None) -> None:
    evidence = stage / "evidence"
    replay = evidence / "replay"
    hosts = [f"gpu-h800-{index:04d}" for index in range(4)]
    tp = config["engine"]["tensor_parallel_size"]
    for index, host in enumerate(hosts):
        node = evidence / "nodes" / host
        node.mkdir(parents=True)
        role = ("prefill", "decode")[index // 2]
        (node / "ready.json").write_text(json.dumps({"host": host, "ip": f"10.0.0.{index + 1}"}))
        (node / "result.json").write_text(json.dumps({
            "plan": {"role": role, "lane": index % 2}, "ucx_devices": ["mlx5_0:1"],
            "nvidia_smi": "".join(f"GPU {gpu}: NVIDIA H800 (UUID: x)\n" for gpu in range(8)),
            "workload_returncode": 0, "server_stop": {"returncode": 0}, "stage": "finished",
            "observation_errors": []}))
        log = [f"(EngineCore_DP{index % 2} pid=1) Initializing NIXL Scheduler e{index}",
               f"(EngineCore_DP{index % 2} pid=1) Using LBNHC KV cache layout.",
               f"(EngineCore_DP{index % 2} pid=1) GPU KV cache size: 1,036,282 tokens"]
        log += [f"(Worker_TP{rank} pid=2) Detected kv cache layout LBNHC" for rank in range(tp)]
        if fault == "startup" and index == 3:
            log = log[:-1]
        (node / "server.log").write_text("\n".join(log) + "\n")
    base = 1000.0
    params = {"do_remote_prefill": True, "do_remote_decode": False, "remote_block_ids": [[1, 2, 3]],
              "remote_engine_id": "e0", "remote_request_id": "rr", "remote_host": "10.0.0.1",
              "remote_port": 5601, "tp_size": tp, "transfer_mode": "pull"}
    waves = json.loads((stage / "trace/waves.json").read_text())
    for wave in waves["waves"]:
        rows = json.loads((stage / "trace" / wave / "request_ids.json").read_text())["rows"]
        directory = replay / wave
        directory.mkdir(parents=True)
        client, proxy = [], []
        for offset, row in enumerate(rows):
            start = base + offset
            client.append({"request_id": row["request_id"], "num_prefill_tokens": row["num_prefill_tokens"],
                           "http_status": 200, "dispatch_monotonic": start, "response_monotonic": start + 3,
                           "usage": {"completion_tokens": 9}, "finish_reason": "length"})
            proxy.append({"request_id": row["request_id"], "proxy_arrival_monotonic": start + 0.1,
                          "prefill_status": 200, "prefill_response_monotonic": start + 1,
                          "decode_status": 200, "decode_response_monotonic": start + 2.9,
                          "kv_transfer_params": params})
        (directory / "client_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in client))
        (directory / "proxy_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in proxy))
    w2 = {"local_compute": 41, "local_cache_hit": 0, "external_kv_transfer": 6128}
    cumulative = {"baseline": {}, "w1": SOURCES,
                  "w2": {key: SOURCES[key] + w2[key] for key in SOURCES}}
    cumulative["released"] = cumulative["w2"]
    if fault == "split":
        cumulative["w1"] = SOURCES | {"local_compute": 41}
    metrics = replay / "metrics"
    metrics.mkdir()
    for point, sources in cumulative.items():
        moved = 0.0 if point == "baseline" else 4096.0
        (metrics / f"{point}_prefill.prom").write_text(page(usage=(0.0, 0.0)))
        (metrics / f"{point}_decode.prom").write_text(page(
            reads=(moved, 0.0), descriptors=(moved and 30.0, 0.0), sources=sources,
            usage=(0.3, 0.0) if fault == "stale_gauge" and point == "released" else (0.0, 0.0)))
        if fault == "absent_counter":
            for path in metrics.glob("*_decode.prom"):
                path.write_text("".join(line + "\n" for line in path.read_text().splitlines()
                                        if "nixl_num_failed_transfers_total" not in line))
    (replay / "native_transfer_metrics.json").write_text(json.dumps({
        "statuses": {"w1": 0, "w2": 0}, "errors": {}}))


@pytest.mark.parametrize("fault", [None, "split", "startup", "stale_gauge", "missing_log",
                                   "absent_counter"])
def test_verdict_judges_each_condition_from_the_mirrored_evidence(staged, fault):
    stage, config, _ = staged
    write_evidence(stage, config, fault=fault)
    if fault == "missing_log":
        (stage / "evidence/nodes/gpu-h800-0000/server.log").unlink()
    result = verdict.judge(stage, "Succeeded")
    statuses = {name: row["status"] for name, row in result["checks"].items()}
    expected = {name: "PASS" for name in (
        "1_placement", "2_startup", "3_prefill_response", "4_decode_output", "5_nixl_counters",
        "6_decode_prompt_sources", "7_release", "8_ordering", "9_exit")}
    expected.update({None: {}, "split": {"6_decode_prompt_sources": "FAIL"},
                     "startup": {"2_startup": "FAIL"}, "stale_gauge": {"7_release": "UNVERIFIED"},
                     "missing_log": {"2_startup": "UNVERIFIED"},
                     "absent_counter": {"5_nixl_counters": "UNVERIFIED"}}[fault])
    assert statuses == expected
    assert result["all_pass"] is (fault is None)
    request = result["observations"]["requests"]["r1test-w1-0"]
    assert request["prefill_lane"] == 1 and request["prefill_phase_s"] == pytest.approx(0.9)
    assert result["observations"]["wave_changes"]["w2"]["decode_prompt_sources"]["external_kv_transfer"][
        "total"] == 6128


def test_submit_command_runs_the_staged_worker_under_the_timeout_with_cuda_compat(staged):
    stage, config, _ = staged
    import shlex
    command = submit_native_pd.job_command(config, stage / "config.json")
    words = shlex.split(command)
    assert words[:2] == ["bash", "-c"] and len(words) == 3
    assert f"export LD_LIBRARY_PATH={config['cuda_compat_path']}" in words[2]
    assert words[2].endswith(shlex.join(["timeout", "3600", "python3", "-B", config["script"],
                                         "--config", str(stage / "config.json")]))
    arguments = submit_native_pd.resource_arguments(config)
    assert arguments["preemptible"] is False and "custom_resources" not in arguments
    assert (arguments["replica"], arguments["gpu"], arguments["positive_tags"]) == (4, 8, ["H800"])
    assert arguments["envs"]["VLLM_ENABLE_CUDA_COMPATIBILITY"] == "1"
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in arguments["envs"]


def test_warm_definition_keeps_cold_w1_engine_and_one_request_per_wave():
    r1 = json.loads(R1.read_text())
    warm = json.loads(R1.with_name("r1w_run.json").read_text())
    assert warm["waves"][0] == r1["waves"][0] | {"prompt_id": "r1h20001-w1-0"}
    assert warm["drain_seconds"] == 10
    for field in ("description", "waves", "drain_seconds"):
        r1.pop(field)
        warm.pop(field)
    assert warm == r1
    traces = build_stage.wave_traces(json.loads(R1.with_name("r1w_run.json").read_text()), "warmtest")
    assert len(traces) == 17 and all(len(rows) == 1 for rows in traces.values())
    assert "stream" not in traces["w1"][0]
    assert all(rows[0]["stream"] is True for name, rows in traces.items() if name != "w1")
    assert len({rows[0]["request_id"] for rows in traces.values()}) == 17
    baseline = [rows[0] for rows in traces.values() if rows[0]["num_prefill_tokens"] == 4105]
    assert len(baseline) == 7 and {row["prompt_id"] for row in baseline} == {"r1h20001-w1-0"}
    assert [row["role"] for row in baseline] == ["gating"] + ["recorded"] * 6


@pytest.mark.parametrize("version", ["0.10.2", "0.28.0", "step5"])
def test_replay_repeats_token_ids_without_reusing_request_identity(version):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    bodies = {}

    async def exercise():
        async def complete(request):
            bodies[request.headers["X-Request-Id"]] = await request.json()
            return web.json_response({"choices": [{"finish_reason": "length"}], "usage": {}})

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        rows = [{"request_id": rid, "segment": "w1", "role": "recorded", "arrived_at": 0.0,
                 "num_prefill_tokens": 4105, "num_decode_tokens": 9,
                 **({"prompt_id": "r1h20001-w1-0"} if rid != "r1h20001-w1-0" else {})}
                for rid in ("r1h20001-w1-0", "new-w1-0", "new-warm-0")]
        async with TestServer(app) as server:
            return await vllm_replay.replay(rows, server.port, 128896, 0.0, vllm_version=version)

    _, records = asyncio.run(exercise())
    assert len(records) == len(bodies) == 3
    assert bodies["r1h20001-w1-0"] == bodies["new-w1-0"] == bodies["new-warm-0"]
    assert bodies["new-w1-0"]["stream"] is False


def test_warm_stage_freezes_sequential_boundary_traces_and_expected_splits(tmp_path):
    inputs = stage_inputs(tmp_path)
    definition = json.loads(R1.with_name("r1w_run.json").read_text())
    definition["probe_helpers"] = inputs["definition"]["probe_helpers"]
    stage = tmp_path / "workspace/stage"
    config = build_stage.build(
        definition, stage=stage, workspace=stage.parent,
        output_dir=Path("/mnt/codesign-exp/ycfeng/step5-preview/native_pd_warmtest"),
        model_dir=inputs["model_dir"], image_source_hashes=inputs["image_source_hashes"],
        probe_dir=inputs["probe_dir"], job_tag="warmtest", source={"commit": "test"})
    plan = json.loads((stage / "trace/waves.json").read_text())
    assert plan["waves"][0] == plan["gating_wave"] == "w1"
    assert config["drain_seconds"] == 10 and config["replay_timeout_s"] == 900
    expected = {"short128": (112, 16), "block4095": (4064, 31), "block4096": (4080, 16),
                "block4097": (4080, 17), "region4103": (4080, 23), "region4104": (4080, 24),
                "block4111": (4080, 31), "block4112": (4096, 16), "block4113": (4096, 17),
                "long7681": (7664, 17)}
    for wave, (external, computed) in expected.items():
        assert plan["expected_decode_prompt_sources"][wave] == {
            "external_kv_transfer": external, "local_compute": computed, "local_cache_hit": 0}
    for relative, digest in config["tool_hashes"].items():
        assert hashlib.sha256((stage / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("fault", [None, "missing_sources", "multiple_requests", "missing_ttft"])
@pytest.mark.parametrize("stream", [False, True])
def test_warm_table_attributes_single_request_windows_without_inventing_client_ttft(staged, fault, stream):
    stage, config, _ = staged
    write_evidence(stage, config)
    trace = stage / "trace/w2/request_ids.json"
    trace.write_text(json.dumps({"rows": json.loads(trace.read_text())["rows"][:1]}))
    replay = stage / "evidence/replay"
    for name in ("client_requests.jsonl", "proxy_requests.jsonl"):
        path = replay / "w2" / name
        path.write_text(path.read_text().splitlines()[0] + "\n")
    if stream:
        path = replay / "w2/client_requests.jsonl"
        record = json.loads(path.read_text())
        record.update(stream=True, first_token_monotonic=1001.5, stream_completed=True)
        path.write_text(json.dumps(record) + "\n")
    cumulative = {name: 2 * value for name, value in SOURCES.items()}
    text = page(sources=cumulative, requests=2 if fault != "multiple_requests" else 3)
    text += ('vllm:time_to_first_token_seconds_count{engine="0"} 2\n'
             'vllm:time_to_first_token_seconds_sum{engine="0"} 1.75\n')
    if fault == "missing_ttft":
        text = "\n".join(line for line in text.splitlines() if "time_to_first_token" not in line)
    if fault == "missing_sources":
        text = "\n".join(line for line in text.splitlines() if "prompt_tokens_by_source" not in line)
    (replay / "metrics/w1_decode.prom").write_text(page(
        reads=(4096.0, 0.0), descriptors=(30.0, 0.0), sources=SOURCES, requests=1)
        + 'vllm:time_to_first_token_seconds_count{engine="0"} 1\n'
          'vllm:time_to_first_token_seconds_sum{engine="0"} 1.25\n')
    (replay / "metrics/w2_decode.prom").write_text(text)
    result = verdict.judge(stage, "Succeeded")
    assert result["all_pass"]
    assert len(result["checks"]) == 9
    warm, = result["observations"]["warm_requests"]
    assert warm["client_send_monotonic"] == 1000
    assert warm["proxy_prefill_response_monotonic"] == 1001
    assert warm["client_completion_monotonic"] == 1003
    assert warm["client_first_token_monotonic"] == (1001.5 if stream else None)
    assert warm["client_first_token_status"] == ("PASS" if stream else "UNVERIFIED_NONSTREAMING")
    assert warm["client_ttft_s"] == (1.5 if stream else None)
    assert warm["post_prefill_first_token_s"] == (0.5 if stream else None)
    assert warm["decode_engine_ttft_s"] == (0.5 if fault is None else None)
    assert warm["output_tokens"] == 9
    attributed = fault in (None, "missing_ttft")
    assert warm["decode_prompt_attribution"] == ("single_request_window" if attributed else "UNVERIFIED")
    assert warm["decode_external_tokens"] == (4080 if attributed else None)
    assert warm["decode_computed_prompt_tokens"] == (25 if attributed else None)
    assert "not client TTFT" in result["observations"]["client_first_token_note"]


@pytest.mark.parametrize("fault", ["missing_first", "missing_done", "before_prefill", "stream_error", "missing_proxy"])
def test_streaming_timing_gaps_are_unverified_without_changing_w1(staged, fault):
    stage, config, _ = staged
    write_evidence(stage, config)
    path = stage / "evidence/replay/w2/client_requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(stream=True, first_token_monotonic=1001.5, stream_completed=True)
    if fault == "missing_first":
        rows[0]["first_token_monotonic"] = None
    if fault == "missing_done":
        rows[0]["stream_completed"] = False
    if fault == "before_prefill":
        rows[0]["first_token_monotonic"] = 1000.5
    if fault == "stream_error":
        rows[0]["stream_error"] = {"message": "generation failed"}
    if fault == "missing_proxy":
        (path.parent / "proxy_requests.jsonl").write_text("")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = verdict.judge(stage, "Succeeded")
    assert result["all_pass"] and len(result["checks"]) == 9
    row = result["observations"]["requests"][rows[0]["request_id"]]
    assert row["client_first_token_status"] == "UNVERIFIED_STREAM"
    assert row["client_ttft_s"] is None and row["post_prefill_first_token_s"] is None
