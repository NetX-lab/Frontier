"""CPU checks for bootstrap mode invocations and saved bridge verdicts."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.profiling.attention.graph_capture import CaptureRecord, GraphCapture, NodeClone, ScopeEntry
from frontier.profiling.attention.dsv4_graph_replay_runner import CUDAGRAPH_DISPATCH_KEY, LAYER_SCOPE
from test_graph_replay_trace import _activity, _launch, _node, _scope, _step

_SCRIPT = Path(__file__).resolve().parents[1] / "comparison/calibration/dsv4_graph_bridge_check.py"
_SPEC = importlib.util.spec_from_file_location("dsv4_graph_bridge_check", _SCRIPT)
bridge = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bridge
_SPEC.loader.exec_module(bridge)


@pytest.fixture
def config(tmp_path):
    return bridge.BridgeConfig(
        bootstrap={"script": "accepted/profile_dsv4_prefill.py", "source_snapshot": "snapshot",
                   "expected_source_hashes": {"vllm/file.py": "unchanged"}, "tool_hashes": {"tool": "unchanged"},
                   "profile_args": ["old"], "worker_wheel_site": "wheel", "moe_gating_routing_runtime_path": "balanced_topk",
                   "native_package": "native", "overlay_manifest": "overlay"},
        plan_args=("--model_name", "model", "--chunk_budgets", "8"),
        run_args=("--model_dir", "model_dir", "--expected_attention_backend", "FLASHMLA_SPARSE_DSV4",
                  "--gpu_memory_utilization", "0.85", "--seed", "1234"),
        engine_index=2, case_ids=("prefill", "decode"),
        modes=("FULL_AND_PIECEWISE", "PIECEWISE", "NONE"), cudagraph_capture_sizes=(1, 2, 4, 8), output_root=tmp_path,
    )


@pytest.mark.parametrize("mode", ["FULL_AND_PIECEWISE", "PIECEWISE", "NONE"])
def test_profile_args_match_mode_and_keep_shared_flags(config, mode):
    root = config.output_root / mode
    sizes = [] if mode == "NONE" else ["--cudagraph_capture_sizes", "1", "2", "4", "8"]
    assert bridge.profile_args(config, mode) == [
        "run", *config.plan_args, "--engine_policy", "graph_replay", "--cudagraph_mode", mode,
        *sizes, "--engine_index", "2", *config.run_args, "--raw_dir", str(root / "raw"),
        "--trace_dir", str(root / "traces"), "--keep_traces", "--case_ids", "prefill", "decode",
    ]


@pytest.mark.parametrize("failed", [None, "PIECEWISE", "OSError"])
def test_run_uses_bootstrap_environment_logs_and_continues_after_failure(config, monkeypatch, failed):
    calls = []
    monkeypatch.setenv("BRIDGE_PARENT_SENTINEL", "retained")

    def run(command, **kwargs):
        payload = json.loads(Path(command[-1]).read_text())
        mode = payload["profile_args"][payload["profile_args"].index("--cudagraph_mode") + 1]
        calls.append((mode, command, kwargs, payload))
        kwargs["stdout"].write(f"fake {mode}\n")
        if mode == "PIECEWISE" and failed == "OSError":
            raise OSError("fake unavailable interpreter")
        return SimpleNamespace(returncode=3 if mode == failed else 0)

    monkeypatch.setattr(bridge.subprocess, "run", run)
    assert bridge.run_bridge(config) == int(failed is not None)
    assert [mode for mode, *_ in calls] == list(config.modes)
    runs = json.loads((config.output_root / "bridge_run.json").read_text())["modes"]
    for mode, command, kwargs, payload in calls:
        assert command == [sys.executable, config.bootstrap["script"], "--config", str(config.output_root / mode / "bootstrap_config.json")]
        assert {key: payload[key] for key in config.bootstrap if key != "profile_args"} == {
            key: value for key, value in config.bootstrap.items() if key != "profile_args"
        }
        assert kwargs["env"]["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
        assert kwargs["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        assert kwargs["env"]["BRIDGE_PARENT_SENTINEL"] == "retained"
        assert kwargs["stderr"] is bridge.subprocess.STDOUT and kwargs["check"] is False
        assert runs[mode]["duration_seconds"] >= 0 and runs[mode]["start"] <= runs[mode]["end"]
        assert (config.output_root / mode / "run.log").read_text().startswith(f"fake {mode}")
    if failed == "OSError":
        assert runs["PIECEWISE"]["exit_code"] is None
        assert "unavailable interpreter" in runs["PIECEWISE"]["error"]


def _artifacts(config, *, full_steps=True, attribution_error=False, exit_code=0):
    owner = (4, "dsv4_q_projection")
    record = CaptureRecord(
        captures=(GraphCapture(0, "graph", None, 7),), nodes=(_node(10, "kernel", owner=owner),),
        scope_entries=(ScopeEntry(0, (4, LAYER_SCOPE), None), ScopeEntry(0, owner, (4, LAYER_SCOPE))),
        clones=(NodeClone(1010, 10),),
    )
    runs = {}
    for mode in config.modes:
        root = config.output_root / mode
        (root / "raw").mkdir(parents=True)
        (root / "traces").mkdir()
        capture_summary = {
            "captures_by_kind": {"graph": 1, "breakable_segment": 0},
            "node_count": 1, "kernel_node_count": 1, "clone_count": 1,
            "eager_break_count": 0, "eager_break_owner_scopes": [],
        }
        saved_record = CaptureRecord() if mode == "NONE" else record
        (root / "raw" / "dsv4_capture_e2.json").write_text(json.dumps({"record": saved_record.to_json(), "summary": capture_summary}))
        cg_mode = ("FULL" if full_steps else "PIECEWISE") if mode == "FULL_AND_PIECEWISE" else mode
        (root / "raw" / "dsv4_raw_e2.jsonl").write_text(json.dumps({
            "kind": "step", CUDAGRAPH_DISPATCH_KEY: {"cg_mode": cg_mode, "num_tokens": 1, "num_reqs": 1},
        }) + "\n")
        if mode == "NONE":
            events = [_step(), _scope(owner, 10, 50), _launch(1, 20, graph=False), _activity(0, "eager", 100, 20)]
        else:
            name = "mismatch" if attribution_error and mode == "FULL_AND_PIECEWISE" else "kernel"
            events = [_step(), _launch(1, 10), _activity(1010, name, 100, 20)]
        # Outside eager kernels are reported but are valid in the bridge contract.
        events.extend([_launch(2, 200, graph=False), _activity(0, "outside", 210, 5, correlation=2)])
        (root / "traces" / "case.json").write_text(json.dumps({"traceEvents": events}))
        runs[mode] = {"exit_code": exit_code}
    (config.output_root / "bridge_run.json").write_text(json.dumps({"modes": runs}))


def test_report_passes_graph_and_eager_attribution(config, capsys):
    _artifacts(config)
    assert bridge.report_bridge(config) == 0
    report = json.loads((config.output_root / "bridge_report.json").read_text())
    assert report["verdict"] == "PASS" and report["reasons"] == []
    assert "PASS" in capsys.readouterr().out
    for mode, result in report["modes"].items():
        trace = result["traces"][0]
        assert result["run_exit_code"] == 0 and result["capture_summary"]["clone_count"] == 1
        assert trace["error"] is None and trace["outside_kernel_count"] == 1
        assert trace["graph_kernel_interval_count"] == (0 if mode == "NONE" else 1)
        assert trace["eager_kernel_interval_count"] == (2 if mode == "NONE" else 1)
        assert trace["unattributed_launch_count"] == 0


def test_report_records_attribution_error_and_continues(config):
    _artifacts(config, attribution_error=True)
    assert bridge.report_bridge(config) == 1
    report = json.loads((config.output_root / "bridge_report.json").read_text())
    assert report["verdict"] == "FAIL"
    assert "Kernel name mismatch" in report["modes"]["FULL_AND_PIECEWISE"]["traces"][0]["error"]
    assert report["modes"]["NONE"]["traces"][0]["error"] is None
    assert any("attribution failed" in reason for reason in report["reasons"])


def test_report_fails_without_full_steps(config):
    _artifacts(config, full_steps=False)
    assert bridge.report_bridge(config) == 1
    report = json.loads((config.output_root / "bridge_report.json").read_text())
    assert report["verdict"] == "FAIL" and report["reasons"] == ["No FULL steps observed"]


def test_report_fails_unsuccessful_mode_runs(config):
    _artifacts(config, exit_code=3)
    assert bridge.report_bridge(config) == 1
    report = json.loads((config.output_root / "bridge_report.json").read_text())
    assert all("run exit code 3" in reason for reason in report["reasons"])


def test_missing_artifacts_produce_fail_report(config):
    assert bridge.report_bridge(config) == 1
    report = json.loads((config.output_root / "bridge_report.json").read_text())
    assert report["verdict"] == "FAIL"
    assert all(mode["artifact_errors"] for mode in report["modes"].values())


@pytest.mark.parametrize("modes", [[], ["FULL"], ["NONE", "NONE"]])
def test_config_rejects_invalid_mode_set(config, tmp_path, modes):
    payload = dict(config.__dict__, modes=modes, output_root=str(config.output_root))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-empty ordered subset"):
        bridge.load_config(path)


def test_report_cli_loads_config_and_writes_verdict(config, tmp_path):
    _artifacts(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(config.__dict__, output_root=str(config.output_root))))
    assert bridge.main(["--config", str(path), "--report"]) == 0
    assert json.loads((config.output_root / "bridge_report.json").read_text())["verdict"] == "PASS"


def test_bootstrap_cli_runs_every_mode(config, tmp_path, monkeypatch):
    modes = []

    def run(command, **kwargs):
        payload = json.loads(Path(command[-1]).read_text())
        modes.append(payload["profile_args"][payload["profile_args"].index("--cudagraph_mode") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", run)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(config.__dict__, output_root=str(config.output_root))))
    assert bridge.main(["--config", str(path)]) == 0
    assert modes == list(config.modes)
    assert not (config.output_root / "bridge_report.json").exists()
