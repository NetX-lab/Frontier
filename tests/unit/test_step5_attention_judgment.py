"""Synthetic CPU faults for the local attention qualification precheck."""

import copy
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "integration/step5_profiling/judge_attention_smoke.py"
SPEC = importlib.util.spec_from_file_location("judge_attention_smoke", SCRIPT)
judgment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judgment)
OP = "step5_qkv_projection"
ACTIVE = [OP, "step5_output_projection"]


def make_row(kind="CUDA_EVENT", phase="decode"):
    means = {name: .3 if kind == "KERNEL_ONLY" else .45 for name in ACTIVE}
    means[judgment.LAYER] = .6
    row = {
        "measurement_type": kind, "global_layer_id": "0", "phase": phase,
        "has_attention_metadata": "1",
        "batch_size": "1", "batch_num_tokens": "1" if phase == "decode" else "16",
        "query_lens": "[1]" if phase == "decode" else "[16]", "context_lens": "[512]",
        "native_max_num_seqs": "1", "native_cudagraph_capture_sizes": "[1, 2]",
        "graph_mode": "full" if phase == "decode" else "none",
        "output_state_validation": "PASS", "prefix_continuation_validation": "PASS",
        "active_operators": json.dumps(ACTIVE), "profile_iterations": "2",
        "window_replays": "[2, 2]", "window_durations_ms": "[1.2, 1.2]",
        "min_timing_window_ms": "0.5", "stability_pass": "True",
        "operator_sum_to_layer_ratio": "1" if kind == "KERNEL_ONLY" else ".8",
        "scoped_layer_time_stats": json.dumps({"mean": 1.125, "std": .01, "count": 2}),
        "scoped_outer_time_stats": json.dumps({"mean": 1.2, "std": .01, "count": 2}),
        "instrumentation_overhead_fraction": "1",
        "kernel_count": "4", "kernel_time_ms": "2.4", "memory_copy_count": "0",
        "memory_copy_time_ms": "0", "skipped_kernel_count": "4",
        "skipped_memory_copy_count": "10", "skipped_uncorrelated_activity_count": "0",
        "per_iteration_time_stats": json.dumps({
            name: {"median": mean, "count": 2 if kind == "CUDA_EVENT" and name == judgment.LAYER else 4}
            for name, mean in means.items()}),
    }
    for name, mean in means.items():
        row.update({f"time_stats.{name}.mean": str(mean), f"time_stats.{name}.std": str(mean * .01),
                    f"time_stats.{name}.count": "2"})
    if kind == "CUDA_EVENT":
        raw = json.loads(row["per_iteration_time_stats"])
        raw.pop(judgment.LAYER)
        row.update({
            "per_iteration_time_stats": json.dumps(raw),
            "layer_reference_time_stats": json.dumps({"mean": .6, "median": .6, "count": 2}),
            "reference_window_replays": "[2, 2]", "reference_window_durations_ms": "[1.2, 1.2]",
            "layer_reference_times_ms": "[0.6, 0.6]",
            "layer_reference_timing_basis": "independent_outer_only_normalized_clean_replay_windows",
            "boundary_single_replay_role": "instrumentation_diagnostics_only",
            "boundary_single_replay_times_ms": "[0.9, 0.9]",
            "boundary_single_replay_time_stats": json.dumps({"median": .9, "count": 2}),
        })
    return row


def test_request_free_judgment_keeps_positive_work_and_distinct_identity():
    idle = make_row("KERNEL_ONLY")
    idle.update(has_attention_metadata="0", context_lens="[0]", graph_mode="none",
                prefix_continuation_validation="not_applicable")
    result = judgment.inspect_row(idle, "KERNEL_ONLY", .05)
    assert result["candidate"] and all(result["checks"].values())
    ordinary = {**idle, "has_attention_metadata": "1", "graph_mode": "full"}
    assert judgment.inspect_row(ordinary, "KERNEL_ONLY", .05)["cell"] != result["cell"]
    idle[f"time_stats.{OP}.std"] = ".02"
    assert not judgment.inspect_row(idle, "KERNEL_ONLY", .05)["checks"]["cv"]
    idle["context_lens"] = "[8]"
    with pytest.raises(ValueError, match="no history"):
        judgment.inspect_row(idle, "KERNEL_ONLY", .05)


def test_independent_eager_profile_and_piecewise_graph_are_separate_cells():
    eager = make_row("KERNEL_ONLY", "prefill")
    eager.update(native_cudagraph_capture_sizes="[1, 2, 4, 8, 16]",
                 graph_dispatch_basis="independent_eager_profile")
    graph = make_row("CUDA_EVENT", "prefill")
    graph.update(native_cudagraph_capture_sizes="[1, 2, 4, 8, 16]", graph_mode="piecewise")
    a = judgment.inspect_row(eager, "KERNEL_ONLY", .05)
    b = judgment.inspect_row(graph, "CUDA_EVENT", .05)
    assert a["checks"]["graph_dispatch"] and b["checks"]["graph_dispatch"]
    assert a["candidate"] and b["candidate"] and a["cell"] != b["cell"]


@pytest.fixture
def fixture_rows():
    return [make_row(), make_row(phase="prefill")], [make_row("KERNEL_ONLY", "prefill")]


def write_fixture(directory, events, gpu):
    for filename, rows in (("step5_attention.csv", events), ("step5_attention_kernel_only.csv", gpu)):
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps({
        "expected_rows": 2, "actual_rows": len(events), "expected_gpu_active_rows": 1,
        "gpu_active_rows": len(gpu), "stable_rows": 99, "complete": True,
    }))


def run_judge(directory, rows):
    write_fixture(directory, *rows)
    return judgment.judge(directory, expected_event_rows=2, expected_gpu_active_rows=1)


def test_valid_qualification_separates_native_cells_from_measurement_rows(tmp_path, fixture_rows):
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "PASS" and not result["admitted"]
    assert all(result["checks"].values())
    assert result["counts"] == {
        "raw_rows": 3, "event_rows": 2, "gpu_active_rows": 1, "native_cells": 2,
        "latency_diagnostic_rows": 1, "calibration_candidates": 2, "qualified_candidates": 2,
        "full_event_candidates": 1,
        "cv_pass_candidates": 2, "output_state_pass_rows": 3,
        "prefix_continuation_rows": 3, "prefix_continuation_pass_rows": 3,
    }
    assert result["bounds"]["event_scope_ratio"]["min"] == pytest.approx(.8)
    assert result["rows"][-1]["cupti"]["skipped_kernel_count"] == 4
    assert result["rows"][-1]["cupti"]["skipped_memory_copy_count"] == 10


@pytest.mark.parametrize("column,value", [
    ("kernel_count", "0"), ("kernel_time_ms", "0"), ("kernel_time_ms", "NaN"),
    ("skipped_uncorrelated_activity_count", "1"), ("memory_copy_count", "-1"),
    (f"time_stats.{OP}.count", "1"), (f"time_stats.{OP}.count", "2.5"),
    (f"time_stats.{OP}.mean", "nan"), (f"time_stats.{OP}.std", "inf"),
    (f"time_stats.{OP}.std", "-0.01"), ("window_replays", "[0, 1]"),
    ("window_durations_ms", "[0.49, 0.6]"), ("min_timing_window_ms", "1.3"),
    ("window_replays", "[1]"), ("profile_iterations", "0"),
])
def test_bad_counters_finite_values_and_windows_fail(tmp_path, fixture_rows, column, value):
    fixture_rows[1][0][column] = value
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "FAIL" and result["failed_cells"]
    assert result["counts"]["qualified_candidates"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("column", [f"time_stats.{OP}.std", "active_operators", "per_iteration_time_stats",
                                   "skipped_uncorrelated_activity_count"])
def test_missing_required_columns_are_explicit_failures(tmp_path, fixture_rows, column):
    fixture_rows[1][0].pop(column)
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "FAIL"
    assert any(column in error and "KeyError" in error for error in result["errors"])


def test_cv_is_recomputed_despite_claimed_stability_and_summary(tmp_path, fixture_rows):
    fixture_rows[1][0][f"time_stats.{OP}.std"] = ".03"
    fixture_rows[1][0]["max_operator_time_cv"] = "0"
    result = run_judge(tmp_path, fixture_rows)
    assert not result["checks"]["cv"]
    failure = result["failed_cells"][0]["targets"][0]
    assert failure["target"] == OP and failure["cv"] == pytest.approx(.1)


def test_median_error_uses_raw_median_denominator(tmp_path, fixture_rows):
    row = fixture_rows[0][0]
    reference = .6 / 1.051
    row["layer_reference_time_stats"] = json.dumps({"mean": reference, "median": reference, "count": 2})
    row["layer_reference_times_ms"] = json.dumps([reference] * 2)
    row["reference_window_durations_ms"] = json.dumps([2 * reference] * 2)
    result = run_judge(tmp_path, fixture_rows)
    assert result["checks"]["cv"] and not result["checks"]["median_agreement"]
    assert result["counts"]["qualified_candidates"] == 1
    failure = result["failed_cells"][0]["targets"][0]
    assert failure["target"] == judgment.LAYER
    assert failure["median_relative_error"] == pytest.approx(.051)


def test_boundary_event_diagnostics_cannot_reject_independent_reference(tmp_path, fixture_rows):
    row = fixture_rows[0][0]
    row["boundary_single_replay_times_ms"] = "[0.9, 9.0]"
    row["boundary_single_replay_time_stats"] = json.dumps({"median": 4.95, "count": 2})
    assert run_judge(tmp_path, fixture_rows)["verdict"] == "PASS"


@pytest.mark.parametrize("column,value", [
    ("reference_window_replays", "[1, 2]"),
    ("reference_window_durations_ms", "[0.49, 1.2]"),
    ("layer_reference_times_ms", "[0.3, 0.6]"),
    ("layer_reference_timing_basis", "boundary_single_replays"),
    ("boundary_single_replay_role", "agreement_reference"),
    ("boundary_single_replay_times_ms", "[0.9]"),
])
def test_independent_reference_protocol_cannot_be_mislabelled_or_shortened(tmp_path, fixture_rows, column, value):
    fixture_rows[0][0][column] = value
    assert run_judge(tmp_path, fixture_rows)["verdict"] == "FAIL"


def test_false_stability_flags_do_not_reject_recomputed_good_measurements(tmp_path, fixture_rows):
    for row in (*fixture_rows[0], *fixture_rows[1]):
        row["stability_pass"] = "False"
    assert run_judge(tmp_path, fixture_rows)["verdict"] == "PASS"


def test_replayed_operator_counts_and_isolated_event_layer_counts_qualify(tmp_path, fixture_rows):
    result = run_judge(tmp_path, fixture_rows)
    assert result["checks"]["counts"]
    assert all(row["checks"]["counts"] for row in result["rows"])
    assert result["counts"]["qualified_candidates"] == 2
    assert result["verdict"] == "PASS"


@pytest.mark.parametrize("kind,target,wrong_count", [
    ("CUDA_EVENT", OP, 2), ("CUDA_EVENT", judgment.LAYER, 4),
    ("KERNEL_ONLY", OP, 2), ("KERNEL_ONLY", judgment.LAYER, 2),
])
def test_raw_operator_and_layer_counts_must_keep_their_own_units(
        tmp_path, fixture_rows, kind, target, wrong_count):
    row = fixture_rows[0 if kind == "CUDA_EVENT" else 1][0]
    if kind == "CUDA_EVENT" and target == judgment.LAYER:
        reference = json.loads(row["layer_reference_time_stats"])
        reference["count"] = wrong_count
        row["layer_reference_time_stats"] = json.dumps(reference)
    else:
        singles = json.loads(row["per_iteration_time_stats"])
        singles[target]["count"] = wrong_count
        row["per_iteration_time_stats"] = json.dumps(singles)
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "FAIL" and not result["checks"]["counts"]
    assert result["counts"]["qualified_candidates"] == 1
    assert result["checks"]["cv"] and result["checks"]["median_agreement"]
    assert result["failed_cells"][0]["checks"] == ["counts"]


def test_none_latency_diagnostics_cannot_reject_calibration(tmp_path, fixture_rows):
    diagnostic = fixture_rows[0][1]
    diagnostic[f"time_stats.{OP}.std"] = "1"
    raw = json.loads(diagnostic["per_iteration_time_stats"])
    raw[OP]["median"] = .1
    diagnostic["per_iteration_time_stats"] = json.dumps(raw)
    diagnostic["calibration_candidate"] = "True"
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "PASS" and not result["failed_cells"]
    assert result["counts"]["qualified_candidates"] == 2
    assert not result["rows"][1]["checks"]["cv"]


def test_reconciliation_uses_scoped_layer_not_clean_layer(tmp_path, fixture_rows):
    # Exclusive operator means sum to .9, clean layer is .6, scoped layer is 1.125.
    assert run_judge(tmp_path, fixture_rows)["checks"]["reconciliation"]
    fixture_rows[0][0]["operator_sum_to_layer_ratio"] = "1.5"
    result = run_judge(tmp_path, fixture_rows)
    assert not result["checks"]["reconciliation"]
    assert result["rows"][0]["operator_sum_to_layer_ratio"] == pytest.approx(.8)


def test_gpu_active_ratio_must_equal_one_not_merely_fit_zero_to_one(tmp_path, fixture_rows):
    row = fixture_rows[1][0]
    row[f"time_stats.{OP}.mean"] = ".15"
    raw = json.loads(row["per_iteration_time_stats"])
    raw[OP]["median"] = .15
    row["per_iteration_time_stats"] = json.dumps(raw)
    row["operator_sum_to_layer_ratio"] = ".75"
    assert not run_judge(tmp_path, fixture_rows)["checks"]["reconciliation"]


def test_roundoff_and_large_instrumentation_overhead_are_not_arbitrary_gates(tmp_path, fixture_rows):
    fixture_rows[1][0][f"time_stats.{OP}.mean"] = ".3000000000000001"
    fixture_rows[1][0]["operator_sum_to_layer_ratio"] = str(1 + 2e-16)
    fixture_rows[0][0]["scoped_outer_time_stats"] = json.dumps({"mean": 600.6, "std": .01, "count": 2})
    fixture_rows[0][0]["instrumentation_overhead_fraction"] = "1000"
    result = run_judge(tmp_path, fixture_rows)
    assert result["verdict"] == "PASS"
    assert result["bounds"]["gpu_active_scope_ratio"]["max"] > 1
    assert result["bounds"]["instrumentation_overhead_fraction"]["max"] == pytest.approx(1000)


@pytest.mark.parametrize("column,value", [
    ("output_state_validation", "FAIL"), ("prefix_continuation_validation", "FAIL"),
    ("native_cudagraph_capture_sizes", "[1, 2, 16]"), ("native_max_num_seqs", "0"),
    ("graph_mode", "piecewise"),
])
def test_functional_and_native_dispatch_failures(tmp_path, fixture_rows, column, value):
    fixture_rows[1][0][column] = value
    assert run_judge(tmp_path, fixture_rows)["verdict"] == "FAIL"


def test_native_decode_query_lengths_and_max_sequences_are_checked(tmp_path, fixture_rows):
    row = fixture_rows[0][0]
    row.update(query_lens="[1, 1]", context_lens="[512, 512]", batch_size="2", batch_num_tokens="2")
    assert not run_judge(tmp_path, fixture_rows)["checks"]["graph_dispatch"]
    row["native_max_num_seqs"] = "2"
    assert run_judge(tmp_path, fixture_rows)["verdict"] == "PASS"
    row.update(query_lens="[2]", context_lens="[512]", batch_size="1")
    assert not run_judge(tmp_path, fixture_rows)["checks"]["graph_dispatch"]


def test_count_and_cell_coverage_fail_without_trusting_complete_summary(tmp_path, fixture_rows):
    fixture_rows[1].append(copy.deepcopy(fixture_rows[1][0]))
    result = run_judge(tmp_path, fixture_rows)
    assert not result["checks"]["row_counts"] and not result["checks"]["cell_coverage"]
    assert result["verdict"] == "FAIL"


@pytest.mark.parametrize("column,value", [("actual_rows", float("nan")), ("gpu_active_rows", None)])
def test_missing_or_nonfinite_summary_counts_fail_clearly(tmp_path, fixture_rows, column, value):
    write_fixture(tmp_path, *fixture_rows)
    path = tmp_path / "summary.json"
    summary = json.loads(path.read_text())
    if value is None:
        summary.pop(column)
    else:
        summary[column] = value
    path.write_text(json.dumps(summary))
    result = judgment.judge(tmp_path, 2, 1)
    assert result["verdict"] == "FAIL" and not result["checks"]["summary_counts"]
    assert column in result["errors"][0]


def test_cli_is_stdlib_only_and_never_overwrites_report(tmp_path, fixture_rows):
    write_fixture(tmp_path, *fixture_rows)
    report = tmp_path / "judgment.json"
    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'vllm', 'pandas', 'numpy'}:
        raise AssertionError('Unexpected import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    args = [sys.executable, "-B", "-c", script, str(SCRIPT), str(tmp_path),
            "--expected-event-rows", "2", "--expected-gpu-active-rows", "1", "--report", str(report)]
    env = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[3])}
    result = subprocess.run(args, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(report.read_text())
    assert subprocess.run(args, text=True, capture_output=True, env=env).returncode != 0
    rejected = subprocess.run(args[:-2] + ["--expected-event-rows", "3"], text=True,
                              capture_output=True, env=env)
    assert rejected.returncode == 1 and json.loads(rejected.stdout)["verdict"] == "FAIL"
