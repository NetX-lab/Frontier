"""CPU A4 grouping, host-ahead admission, planner-role and report contracts."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_dsv4_graph_bridge_check import bridge
from test_dsv4_replay_gaps import complete_records

_SCRIPT = Path(__file__).resolve().parents[1] / "comparison/calibration/dsv4_composition_check.py"
_SPEC = importlib.util.spec_from_file_location("dsv4_composition_check", _SCRIPT)
check = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check
_SPEC.loader.exec_module(check)


def roles(saved):
    return {(record.engine["cudagraph_capture"]["cudagraph_mode"], 0): True for record in saved}


def test_composed_and_call_gap_diagnostic_equal_joined_target():
    saved = complete_records()
    report = check.check_composition(saved, roles(saved))
    assert report["verdict"] == "PASS"
    assert report["gap_values_ms"] == dict.fromkeys(("cuda_graph", "cuda_graph_piecewise", "eager", "eager_preparation"), 0.001)
    assert len(report["groups"]) == 3
    for group in report["groups"]:
        assert group["status"] == "pass" and group["host_ahead_count"] == group["call_count"] == 1
        assert group["median_error"] == pytest.approx(0, abs=1e-15)
        assert group["p90_error"] == pytest.approx(0, abs=1e-15)
        assert group["signed_median_error"] == pytest.approx(0, abs=1e-15)
        assert group["diagnostic"]["median_error"] == pytest.approx(0, abs=1e-15)


def test_cost_mismatch_above_error_limits_fails_group():
    saved = list(complete_records())
    saved[0] = replace(saved[0], calls=tuple(dict(row, time_ms=row["time_ms"] + 0.01) for row in saved[0].calls))
    report = check.check_composition(saved, roles(saved))
    full = next(group for group in report["groups"] if group["step_mode"] == "FULL")
    assert report["verdict"] == "FAIL" and full["status"] == "fail"
    assert full["median_error"] > check.MEDIAN_ERROR_LIMIT
    assert full["p90_error"] > check.P90_ERROR_LIMIT
    assert full["signed_median_error"] > 0


def test_host_bound_call_excluded_while_its_class_value_remains_available():
    saved = list(complete_records())
    saved[0] = replace(saved[0], intervals=tuple(dict(row, launch_end_us=5) if row["scope"] == "dsv4_q_projection" else row for row in saved[0].intervals))
    report = check.check_composition(saved, roles(saved))
    full = next(group for group in report["groups"] if group["step_mode"] == "FULL")
    assert full["status"] == "no_host_ahead_call"
    assert full["host_ahead_count"] == 0 and full["call_count"] == 1
    assert full["exclusion_counts"] == {"host_bound": 1}
    assert full["median_error"] is None


def test_missing_class_value_is_a_distinct_group_status():
    full, _, eager = complete_records()
    eager = replace(eager, intervals=tuple(dict(row, launch_end_us=100) if row["scope"] is not None else row for row in eager.intervals))
    saved = [full, eager]
    report = check.check_composition(saved, roles(saved))
    group = next(group for group in report["groups"] if group["step_mode"] == "NONE")
    assert group["status"] == "missing_gap_value" and group["missing_gap_values"] == ["eager"]
    assert report["verdict"] == "FAIL"


def test_groups_keep_planner_variant_case_threshold_and_short_context_coordinates():
    saved = complete_records()
    full = saved[0]
    other = replace(full, source="second", engine=dict(full.engine, multi_stream_gemm_token_threshold=0))
    report = check.check_composition([full, other, *saved[1:]], roles(saved))
    full_groups = [group for group in report["groups"] if group["step_mode"] == "FULL"]
    assert len(full_groups) == 2
    assert {group["rows_at_or_below_threshold"] for group in full_groups} == {True, False}
    assert all(group["call_count"] == 1 for group in full_groups)


def _config(tmp_path, saved):
    return bridge.BridgeConfig({}, (), (), 0, ("case",), tuple(record.engine["cudagraph_capture"]["cudagraph_mode"] for record in saved), (1,), tmp_path)


def test_planner_role_comes_from_plan_and_ordered_inventory_is_checked(tmp_path, monkeypatch):
    saved = complete_records()
    config = _config(tmp_path, saved)
    invocations = []
    def build(args):
        invocations.append(args)
        return SimpleNamespace(shape=SimpleNamespace(target_tensor_parallel_size=8, layers=(SimpleNamespace(
            global_layer_id=0, attention_variant="uncompressed", compress_ratio=1, plans_tile_schedule=False),)))
    monkeypatch.setattr(check.dsv4_attention_main, "build_parser", lambda: SimpleNamespace(parse_args=lambda argv: argv))
    monkeypatch.setattr(check.dsv4_attention_main, "build_plan", build)
    assert set(check.planner_roles(config, saved).values()) == {False}
    assert invocations == [["plan", *bridge.graph_mode_flags(config, mode)] for mode in config.modes]
    engine = dict(saved[0].engine, layers=[dict(saved[0].engine["layers"][0], compress_ratio=0)])
    with pytest.raises(ValueError, match="engine layers do not match.*recorded.*planned"):
        check.planner_roles(config, [replace(saved[0], engine=engine), *saved[1:]])
    with pytest.raises(ValueError, match="target TP does not match"):
        check.planner_roles(config, [replace(saved[0], engine=dict(saved[0].engine, target_tensor_parallel_size=4)), *saved[1:]])


def test_real_planner_uncompressed_ratio_zero_matches_engine_ratio_one(tmp_path):
    from test_dsv4_attention_profiling import PLAN_ARGS
    saved = complete_records()
    config = replace(_config(tmp_path, saved), plan_args=tuple(PLAN_ARGS))
    plan = check.dsv4_attention_main.build_plan(check.dsv4_attention_main.build_parser().parse_args(
        ["plan", *config.plan_args, *bridge.graph_mode_flags(config, "NONE")]))
    assert any(layer.attention_variant == "uncompressed" and layer.compress_ratio == 0 for layer in plan.shape.layers)
    layers = [{"global_layer_id": layer.global_layer_id, "attention_variant": layer.attention_variant,
               "compress_ratio": max(1, layer.compress_ratio)} for layer in plan.shape.layers]
    saved = tuple(replace(record, engine=dict(record.engine, layers=layers)) for record in saved)
    actual = check.planner_roles(config, saved)
    for mode in config.modes:
        assert {layer.global_layer_id: actual[(mode, layer.global_layer_id)] for layer in plan.shape.layers} == {
            layer.global_layer_id: layer.plans_tile_schedule for layer in plan.shape.layers}


def test_report_cli_writes_report_and_exit_verdict(tmp_path, monkeypatch):
    saved = complete_records()
    config = _config(tmp_path, saved)
    for mode, record in zip(config.modes, saved):
        path = tmp_path / mode / "raw" / "dsv4_raw_e0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in (record.engine, *record.steps, *record.calls, *record.intervals)))
    monkeypatch.setattr(check, "planner_roles", lambda config, records: roles(records))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(config.__dict__, output_root=str(tmp_path))))
    assert check.main(["--config", str(path)]) == 0
    report = json.loads((tmp_path / "composition_report.json").read_text())
    assert report["verdict"] == "PASS" and len(report["groups"]) == 3
    assert report["limits"] == {"median_error": 0.05, "p90_error": 0.10}


def test_planner_failure_writes_explicit_failure_report(tmp_path, monkeypatch):
    config = _config(tmp_path, complete_records())
    monkeypatch.setattr(check, "read_replay_records", lambda paths: complete_records())
    def reject(*args):
        raise ValueError("planner inventory mismatch")
    monkeypatch.setattr(check, "planner_roles", reject)
    assert check.report_composition(config) == 1
    report = json.loads((tmp_path / "composition_report.json").read_text())
    assert report["verdict"] == "FAIL" and report["reasons"] == ["planner inventory mismatch"]
