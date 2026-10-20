"""CPU regression checks for detached Step communication statistics."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "integration/step5_comm/analyze_result.py"
SPEC = importlib.util.spec_from_file_location("step5_comm_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def record(timings, held, ranks=None):
    return {
        "participating_ranks": list(range(len(timings))) if ranks is None else ranks,
        "per_rank_call_us": timings,
        "per_rank_launch_call_us": timings,
        "per_rank_queue_held": held,
    }


def test_intersection_not_mean_rank_fraction():
    item = record([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]],
                  [[False, True, True, True, True], [True, False, True, True, True]])
    result = ANALYSIS.queue_held_analysis(item, "device", 0.01)
    assert result["per_rank_mean_held_fraction"] == 0.8
    assert result["all_rank_held_fraction"] == 0.6
    assert result["retained_repeat_indices"] == [2, 3, 4]
    assert result["median_us"] == 30
    assert result["filtered_median_us"] == 40
    assert result["queue_gate"] == "FAIL"


def test_critical_path_before_median_and_active_ranks_only():
    item = record([[1, 20, 1, 20, 1], None, [20, 1, 20, 1, 20]],
                  [[True] * 5, None, [True] * 5], [0, 2])
    result = ANALYSIS.queue_held_analysis(item, "device", 0.01)
    assert result["critical_path_repeats_us"] == [20] * 5
    assert result["median_us"] == 20
    assert result["rank_repeat_count"] == 10
    assert result["queue_gate"] == "PASS"


def test_threshold_is_strict():
    item = record([[90, 100, 100, 102, 120]], [[False, False, True, True, False]])
    result = ANALYSIS.queue_held_analysis(item, "device", 0.01)
    assert result["filtered_median_us"] == 101
    assert result["absolute_fractional_median_delta"] == 0.01
    assert result["queue_gate"] == "FAIL"


@pytest.mark.parametrize("held", [[[False] * 5], [None]])
def test_no_held_evidence_is_unverified(held):
    result = ANALYSIS.queue_held_analysis(record([[1] * 5], held), "device", 0.01)
    assert result["queue_gate"] == "UNVERIFIED"


@pytest.mark.parametrize("verdict", ["FAIL", "UNVERIFIED"])
def test_main_returns_success_for_observed_quality_rejection(tmp_path, monkeypatch, verdict):
    result = {"queue_gate": verdict, "counts": {}, "timings": [], "label_noise": [],
              "comparisons": [], "native_layers": [], "native_collectives": [], "queue_failures": []}
    monkeypatch.setattr(ANALYSIS, "analyze_result", lambda directory, threshold: result)
    monkeypatch.setattr(ANALYSIS, "markdown_report", lambda result: result["queue_gate"])
    assert ANALYSIS.main([str(tmp_path)]) == 0
    assert (tmp_path / "analysis/analysis.json").is_file()
    assert (tmp_path / "analysis/report.md").read_text() == verdict


def test_host_is_not_device_quality_failure():
    item = record([[1] * 5], [None])
    item["per_rank_launch_call_us"] = [None]
    result = ANALYSIS.queue_held_analysis(item, "host", 0.01)
    assert result["queue_gate"] == "NOT_APPLICABLE"
    assert result["launch_critical_path_median_us"] is None


@pytest.mark.parametrize("op,expected_input,expected_output", [
    ("all_gatherv", {0: 64, 1: 8}, {0: 72, 1: 72}),
    ("reduce_scatterv", {0: 72, 1: 72}, {0: 64, 1: 8}),
    ("all_reduce", {0: 64, 1: 8}, {0: 64, 1: 8}),
])
def test_tensor_endpoint_bytes(op, expected_input, expected_output):
    item = {"op": op, "dtypes": ["bfloat16"], "columns": [4],
            "shard_rows": [8, 1], "participating_ranks": [0, 1],
            "group_ranks": [[0, 1]], "max_input_bytes": max(expected_input.values()),
            "max_output_bytes": max(expected_output.values())}
    result = ANALYSIS.payload_bytes(item)
    assert result["per_rank_input_bytes"] == expected_input
    assert result["per_rank_output_bytes"] == expected_output
    assert result["payload_maxima_match"]


def test_sample_cv_uses_five_observed_repeats():
    result = ANALYSIS.timing_stats([1, 2, 3, 4, 5])
    assert result["repeat_count"] == 5
    assert result["sample_std_us"] == pytest.approx(2.5 ** 0.5)
    assert result["sample_cv"] == pytest.approx((2.5 ** 0.5) / 3)
