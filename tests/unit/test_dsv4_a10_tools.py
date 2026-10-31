"""CPU checks of A10 config/receipts, gate reuse and A9 exact comparisons."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config import global_vars

TOOLS = (Path(__file__).resolve().parents[2] /
         "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10")


@pytest.fixture(autouse=True)
def reset_simulation_globals():
    # Building the PD case config sets the process-wide sys_arch and graph mode.
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    import a10_driver
    import a10_gate
    import compare_cache_runs
    return SimpleNamespace(driver=a10_driver, gate=a10_gate, compare=compare_cache_runs)


@pytest.mark.parametrize("case", ["p2_32", "p2_64"])
def test_operator_config_reuses_case_and_resolved_inputs(tools, tmp_path, case):
    driver = tools.driver
    retained = dict(driver.CASES[case])
    config, changes = driver.build_config(case, tmp_path)
    assert driver.CASES[case] == retained
    flat = config.__flat_config__
    assert flat.metrics_config_run_id == f"a10_{case}_operator_path"
    assert flat.decode_cuda_graph_mode == "full_and_piecewise"
    assert flat.replica_config_network_device == "h200_dgx"
    assert flat.analytical_kv_cache_transfer_config_nixl_read_completion == "engine_step"
    assert flat.cluster_config_decode_replica_scheduler_config_max_concurrent_batches == 2
    assert flat.vllm_v1_scheduler_config_max_forwards_in_flight == 19
    assert {role.name: cluster.replica_scheduler_config.max_forwards_in_flight
            for role, cluster in config.get_clusters().items()} == {"PREFILL": 19, "DECODE": 19}
    assert changes["added_fields"]["vllm_v1_scheduler_config_max_forwards_in_flight"] is None
    assert changes["override_diff"]["vllm_v1_scheduler_config_max_forwards_in_flight"]["h4a4"] == 19
    assert flat.cudagraph_capture_sizes == json.loads(
        (driver.REPO / "tests/fixtures/dsv4_native_dp_sync_rows.json").read_text())["capture_sizes"]
    records, facts = driver.resolved_inputs(config)
    assert len(records) == 7
    assert {Path(record["path"]).name for record in records} == {
        "attention_graph_replay_kernel_only.csv", "linear_op_kernel_only.csv", "moe_kernel_only.csv",
        "model_runner_block_kernel_only.csv", "model_runner_replay_kernel_only.csv",
        "kernel_gap.csv", "summary.csv",
    }
    assert all(record["rows"] > 0 and len(record["sha256"]) == 64 for record in records)
    assert all(len(record["consumers"]) == 2 for record in records)
    assert facts["enable_dummy_mode"] == {"PREFILL": False, "DECODE": False}
    assert facts["no_step_cost_file_beside_inputs"]
    assert facts["step_cost_files_beside_inputs"] == []
    assert driver.h4a4.ATTENTION_FIELD not in changes["override_diff"]
    assert changes["override_diff"]["metrics_config_run_id"]["h4a4"] == f"a10_{case}_operator_path"


def test_a10_matrix_changes_only_label_and_routing(tools, tmp_path):
    gate = tools.gate
    receipt = tmp_path / "native_receipt.json"
    receipt.write_text("{}")
    case = gate.CASES["p2_32"]
    cell_id = "same-cell"
    before = json.loads(gate.h4a4.matrix(case, cell_id, tmp_path, receipt).read_text())
    after = json.loads(gate.matrix(case, cell_id, tmp_path, receipt).read_text())
    before["cells"][0]["label"] = "A10 operator-path client reading"
    before["cases"][case["case_id"]]["routing"] = {
        "status": "MATCH", "reason": gate.ROUTING_REASON,
    }
    assert before == after
    assert "R0" in gate.ROUTING_REASON and "A13 (b)" in gate.ROUTING_REASON


def four_metrics():
    return {"metrics": [{"metric": name, "vllm": 10.0, "frontier": 12.5, "relative_error": 0.25}
                        for name in ("ttft_ms", "tpot_ms", "request_e2e_time_ms", "output_throughput_tps")]}


def test_gate_uses_existing_normalize_and_gate_cli(tools, monkeypatch, tmp_path, capsys):
    gate = tools.gate
    calls = []

    def native_layout(name, case, out):
        receipt = out / "native_receipt.json"
        receipt.write_text("{}")
        return out / "native", receipt

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "gate" in command:
            Path(command[command.index("--output") + 1]).write_text(json.dumps(four_metrics()))

    monkeypatch.setattr(gate.h4a4, "native_layout", native_layout)
    monkeypatch.setattr(gate.subprocess, "run", run)
    gate.main("p2_32", tmp_path / "frontier", tmp_path / "reading")
    assert [command[2] for command, _ in calls] == ["normalize", "gate"]
    assert all(command[1] == str(gate.REPO / "tests/comparison/e2e_gate/run_gate.py")
               and kwargs["check"] and kwargs["timeout"] == gate.GATE_TIMEOUT_S
               for command, kwargs in calls)
    text = capsys.readouterr().out
    assert text.count("absolute_error=2.5 relative_error=0.25") == 4


def small_run(root, common):
    metrics, cache = root / "metrics/model/workload/a10", root / "cache"
    metrics.mkdir(parents=True)
    cache.mkdir()
    (metrics / "request_metrics.csv").write_text("id,ttft\n1,10\n")
    (metrics / "config.json").write_text(json.dumps({"output": str(metrics), "cache": str(cache)}))
    (cache / "model_abcdef123456.pkl").write_bytes(b"same model")
    (cache / "abcdef123456_model_lock.file").write_bytes(b"")
    receipt = {"status": "FINISHED", "run_root": str(root), "metrics_dir": str(metrics),
               "cache_dir": str(cache), "cache_files": common.cache_inventory(cache)}
    (root / "receipt.json").write_text(json.dumps(receipt))
    return metrics, cache


def test_a9_equal_and_differences_are_named(tools, tmp_path, capsys):
    compare = tools.compare
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    small_run(cold, compare)
    metrics, cache = small_run(warm, compare)
    assert compare.main(cold, warm) == 0
    assert "EQUAL" in capsys.readouterr().out
    (metrics / "request_metrics.csv").write_text("id,ttft\n1,11\n")
    assert compare.main(cold, warm) == 1
    assert "metrics/request_metrics.csv" in capsys.readouterr().out
    (metrics / "request_metrics.csv").write_text("id,ttft\n1,10\n")
    (cache / "extra_model_123456abcdef.pkl").write_bytes(b"newly trained")
    assert compare.main(cold, warm) == 1
    assert "extra_model_123456abcdef.pkl" in capsys.readouterr().out


def test_a9_checks_hashes_and_ignores_lock_contents(tools, tmp_path, capsys):
    compare = tools.compare
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    small_run(cold, compare)
    _, cache = small_run(warm, compare)
    (cache / "abcdef123456_model_lock.file").write_bytes(b"transient lock state")
    assert compare.main(cold, warm) == 0
    (cache / "model_abcdef123456.pkl").write_bytes(b"different model under same key")
    assert compare.main(cold, warm) == 1
    text = capsys.readouterr().out
    assert "model_abcdef123456.pkl" in text and "SHA-256 differs" in text


def test_a9_rebases_scratch_copies(tools, tmp_path):
    compare = tools.compare
    original = tmp_path / "original"
    small_run(original, compare)
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    shutil.copytree(original, cold)
    shutil.copytree(original, warm)
    assert compare.differences(cold, warm) == []


def test_a9_requires_finished_runs(tools, tmp_path):
    compare = tools.compare
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    small_run(cold, compare)
    small_run(warm, compare)
    receipt = json.loads((warm / "receipt.json").read_text())
    receipt["status"] = "FAILED"
    (warm / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="finished run"):
        compare.differences(cold, warm)


@pytest.mark.parametrize("run_error", [False, True])
@pytest.mark.parametrize("observe", [False, True])
def test_driver_writes_receipts_at_each_phase(tools, monkeypatch, tmp_path, run_error, observe):
    import frontier.simulator
    import a10_observer
    from frontier.types import ClusterType

    driver = tools.driver
    root = tmp_path / "run"
    phases = []
    trace = tmp_path / "trace.csv"
    trace.write_text("arrived_at\n0.0\n")
    config = SimpleNamespace(seed=0, metrics_config=SimpleNamespace(output_dir=str(root / "metrics"),
                                                                  cache_dir=str(root / "cache")),
                             __flat_config__=SimpleNamespace(trace_request_generator_config_trace_file=str(trace)))

    def build(name, directory, fields=None):
        Path(config.metrics_config.output_dir).mkdir(parents=True)
        Path(config.metrics_config.cache_dir).mkdir()
        return config, {"added_fields": {}, "override_diff": {}}

    class Simulator:
        def __init__(self, config):
            phases.append(json.loads((root / "receipt.json").read_text())["status"])
            self._predictors = {ClusterType.DECODE: SimpleNamespace(_dsv4_step_costs=None)}

        def run(self):
            phases.append(json.loads((root / "receipt.json").read_text())["status"])
            if run_error:
                raise RuntimeError("missing native coordinate")

    class Observer:
        def __init__(self, root):
            self.root = root

        def __enter__(self):
            return self

        def __exit__(self, *args):
            (self.root / "collectives.json").write_text("{}")

        def receipt(self):
            return {"forwards": {"rows": 0}, "collectives": {"calls": 0}}

    monkeypatch.setattr(a10_observer, "ForwardObserver", Observer)
    monkeypatch.setattr(driver, "build_config", build)
    monkeypatch.setattr(driver, "resolved_inputs", lambda config: ([], {
        "enable_dummy_mode": {"DECODE": False}, "no_step_cost_file_beside_inputs": True,
    }))
    monkeypatch.setattr(driver, "git_output", lambda *args: "fake-head" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(driver.h4a4, "kv_admission", lambda *args: {"status": "MATCH"})
    monkeypatch.setattr(frontier.simulator, "Simulator", Simulator)
    if run_error:
        with pytest.raises(RuntimeError, match="missing native coordinate"):
            driver.main("p2_32", root, observe=observe)
    else:
        driver.main("p2_32", root, observe=observe)
    receipt = json.loads((root / "receipt.json").read_text())
    assert phases == ["STARTED", "INITIALIZED"]
    assert receipt["status"] == ("FAILED" if run_error else "FINISHED")
    assert receipt["label"] == "A10"
    assert receipt["a1"]["step_cost_table_bound_by_predictor"] == {"DECODE": False}
    assert receipt["init_wall_s"] >= 0 and receipt["run_wall_s"] >= 0
    assert receipt["cache_files"] == []
    assert receipt["observer_enabled"] is observe
    assert receipt["trace"] == {"path": str(trace), "sha256": tools.driver.sha256(trace)}
    if observe:
        assert receipt["a11"]["forwards"]["rows"] == 0
    else:
        assert "a11" not in receipt


@pytest.mark.parametrize("case", ["p2_32", "p2_64"])
@pytest.mark.parametrize("without_ingress", [False, True])
def test_serving_delay_options_and_receipt_diff(tools, tmp_path, case, without_ingress):
    driver = tools.driver
    args = driver.parse_args([case, str(tmp_path)] + (["--without-ingress"] if without_ingress else []))
    config, changes = driver.build_config(case, tmp_path, args.fields)
    ingress = 0.0 if without_ingress else driver.REQUEST_INGRESS_DELAY_MS
    handoff = 0.0 if without_ingress else driver.REQUEST_HANDOFF_DELAY_MS
    assert config.request_ingress_delay_ms == ingress
    assert config.kv_cache_transfer_config.request_handoff_delay_ms == handoff
    for name, value in (("request_ingress_delay_ms", ingress),
                        ("analytical_kv_cache_transfer_config_request_handoff_delay_ms", handoff)):
        assert changes["added_fields"][name] == 0.0
        if without_ingress:
            assert name not in changes["override_diff"]
        else:
            assert changes["override_diff"][name]["h4a4"] == value


@pytest.mark.parametrize("options", [
    ["--field", "seed=1", "--field", "seed=2"],
    ["--trace", "/tmp/a.csv", "--field", "trace_request_generator_config_trace_file=b.csv"],
    ["--without-run-ahead", "--field", "vllm_v1_scheduler_config_max_forwards_in_flight=1"],
    ["--without-ingress", "--field", "request_ingress_delay_ms=1"],
    ["--without-ingress", "--field", "analytical_kv_cache_transfer_config_request_handoff_delay_ms=1"],
])
def test_field_duplicates_and_option_conflicts_fail(tools, tmp_path, capsys, options):
    with pytest.raises(SystemExit) as error:
        tools.driver.parse_args(["p2_32", str(tmp_path), *options])
    assert error.value.code == 2
    assert "given twice or also set through another option" in capsys.readouterr().err


def test_field_variants_record_backend_and_override_diff(tools, tmp_path):
    driver = tools.driver
    args = driver.parse_args(["p2_32", str(tmp_path), "--field", "cc_backend_config_type=astra_sim_analytical",
                              "--field", "seed=7", "--field", "request_ingress_delay_ms=12.5"])
    config, changes = driver.build_config("p2_32", tmp_path, args.fields)
    for name, value in args.fields.items():
        assert changes["override_diff"][name]["h4a4"] == value
    records, facts = driver.resolved_inputs(config)
    assert facts["cc_backend_types"] == {"PREFILL": "astra_sim_analytical", "DECODE": "astra_sim_analytical"}
    assert len(records) == 6
    assert "summary.csv" not in {Path(record["path"]).name for record in records}
    assert all(not hasattr(cluster.cc_backend_config, "summary_csv_path")
               for cluster in config.get_clusters().values())



def test_field_override_is_written_in_receipt(tools, monkeypatch, tmp_path):
    import frontier.simulator
    from frontier.types import ClusterType

    class Simulator:
        def __init__(self, config):
            self._predictors = {role: SimpleNamespace(_dsv4_step_costs=None)
                                for role in (ClusterType.PREFILL, ClusterType.DECODE)}

        def run(self):
            pass

    driver = tools.driver
    monkeypatch.setattr(frontier.simulator, "Simulator", Simulator)
    monkeypatch.setattr(driver.h4a4, "kv_admission", lambda *args: {})
    args = driver.parse_args(["p2_32", str(tmp_path), "--field", "seed=7",
                              "--field", "cc_backend_config_type=astra_sim_analytical"])
    driver.main("p2_32", tmp_path, observe=False, fields=args.fields)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "FINISHED"
    assert receipt["override_diff"]["seed"]["h4a4"] == 7
    assert receipt["override_diff"]["cc_backend_config_type"]["h4a4"] == "astra_sim_analytical"
    assert receipt["a1"]["cc_backend_types"] == {"PREFILL": "astra_sim_analytical", "DECODE": "astra_sim_analytical"}
