"""CPU regression checks for detached Step communication statistics."""

import importlib.util
import copy
import gzip
import json
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


def test_gzip_jsonl_is_streamed_without_whole_file_reads(tmp_path, monkeypatch):
    path = tmp_path / "measurements.jsonl"
    with gzip.open(path.with_suffix(".jsonl.gz"), "wt") as handle:
        handle.write('{"pass": 0}\n\nnot-json-yet\n')
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("Whole-file read"))
    records = ANALYSIS.iter_jsonl(path)
    assert next(records) == {"pass": 0}
    with pytest.raises(json.JSONDecodeError):
        next(records)


def test_plain_jsonl_list_api_is_preserved(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "complete"}\n')
    assert ANALYSIS.read_jsonl(path) == [{"event": "complete"}]


def label(pass_id, name, count, mean):
    return {"pass": pass_id, "scenario": "graph_decode_b8", "call": name,
            "op": "all_reduce", "group": "tp", "calls_per_forward": count,
            "per_rank_input_bytes": {0: 8}, "per_rank_output_bytes": {0: 8},
            "communicator": "nccl", "kernels": ["observed_kernel"],
            **ANALYSIS.timing_stats([mean] * 30)}


def test_label_multiplicity_and_statistics_are_separate_for_each_pass():
    labels = [("embedding", 1), ("attention", 92), ("dense", 4),
              ("shared", 88), ("routed", 88)]
    rows = [label(p, name, count, mean) for p, mean in [(0, 10), (1, 100), (2, 1000)]
            for name, count in labels]
    result = ANALYSIS.label_noise(rows)
    assert [r["pass"] for r in result] == [0, 1, 2]
    assert [r["label_count"] for r in result] == [5, 5, 5]
    assert [r["calls_per_forward_sum"] for r in result] == [273, 273, 273]
    assert [r["median_of_label_medians_us"] for r in result] == [10, 100, 1000]
    assert all(r["same_payload_and_dispatch"] for r in result)


def test_comparisons_pair_same_pass_and_skip_missing_pairs():
    def row(p, scenario, median):
        return {"pass": p, "scenario": scenario, "call": "dispatch", "group": "dp",
                "concurrent_dp_rails": 1 if "one_rail" in scenario else 8,
                "median_us": median, "filtered_median_us": median, "queue_gate": "PASS",
                "max_input_bytes": 8, "max_output_bytes": 16,
                "per_rank_input_bytes": {0: 8}, "per_rank_output_bytes": {0: 16}}
    rows = [row(p, scenario, median) for p, scenario, median in [
        (0, "graph_decode_b8", 10), (0, "graph_one_rail_b8", 20),
        (1, "graph_decode_b8", 100), (1, "graph_one_rail_b8", 110),
        (2, "graph_decode_b8", 1000), (2, "graph_idle_b64", 2000)]]
    result = ANALYSIS.comparison_rows(rows)
    assert [r["pass"] for r in result] == [0, 1]
    assert [r["comparison"] for r in result] == ["rails_1_vs_8"] * 2
    assert [r["fractional_change"] for r in result] == pytest.approx([1, 0.1])


def test_pass_spread_uses_three_medians_not_ninety_repeats():
    rows = [{"pass": p, "scenario": "size", "call": "dispatch",
             "timing_domain": "device", "median_us": value}
            for p, value in [(2, 30), (0, 10), (1, 20)]]
    result, = ANALYSIS.pass_spreads(rows)
    assert result["passes"] == [0, 1, 2]
    assert result["pass_medians_us"] == [10, 20, 30]
    assert result["pass_count"] == 3
    assert "repeat_count" not in result
    assert result["range_over_median"] == 1


def test_queue_denominators_include_padded_idle_participants_not_absent_ranks():
    timing = ANALYSIS.queue_held_analysis(
        record([[1] * 5, None, [1] * 5, None], [[True] * 5, None, [True] * 5, None], [0, 2]),
        "device", 0.01)
    timing.update(timing_domain="device", world_rank_repeat_count=20,
                  nonparticipating_rank_repeat_count=10, useful_idle_rank_repeat_count=5)
    result = ANALYSIS.queue_counts([timing, {"timing_domain": "host"}])
    assert result["device_rank_repeat_observations"] == 10
    assert result["device_useful_idle_participating_rank_repeat_observations"] == 5
    assert result["device_nonparticipating_rank_repeat_positions"] == 10
    assert result["device_world_rank_repeat_positions"] == 20
    assert result["per_rank_mean_held_fraction"] == 1
    assert result["host_records"] == 1


def native_probe():
    def tensor(rows, columns, dtype):
        element_size = ANALYSIS.DTYPE_BYTES[dtype]
        return {"shape": [rows, columns], "stride": [columns, 1], "dtype": dtype,
                "numel": rows * columns, "element_size": element_size,
                "nbytes": rows * columns * element_size, "device": "cuda:0"}
    topology = ANALYSIS.expected_topology(2, 2, 4)
    ranks = []
    for rank in topology["ranks"]:
        rows = [2, 1][rank["dp_rank"]]
        tp, dp = rank["group_ranks"]["tp"], rank["group_ranks"]["dp"]
        ar = {"op": "all_reduce", "group_ranks": tp, "completed": True,
              "inputs": [tensor(rows, 4, "bfloat16")], "outputs": [tensor(rows, 4, "bfloat16")]}
        dispatch = {"op": "all_gatherv", "group_ranks": dp, "completed": True,
                    "dim": 0, "sizes": [2, 1],
                    "inputs": [tensor(rows, c, d) for c, d in [(4, "bfloat16"), (1, "float32"), (1, "int32")]],
                    "outputs": [tensor(3, c, d) for c, d in [(4, "bfloat16"), (1, "float32"), (1, "int32")]]}
        combine = {"op": "reduce_scatterv_into_output", "group_ranks": dp,
                   "completed": True, "dim": 0, "sizes": [2, 1],
                   "inputs": [tensor(3, 4, "bfloat16")], "outputs": [tensor(rows, 4, "bfloat16")]}
        layer = {"layer_id": 88, "quant_method": "UnquantizedFusedMoEMethod",
                 "oracle_backend": "TRITON", "experts_class": "TritonExperts",
                 "executed_experts_classes": ["TritonExperts"],
                 "prepare_finalize_class": "MoEPrepareAndFinalizeNaiveDPEPModular",
                 "comparison": {"status": "match"},
                 "collectives": [ar, dispatch, combine, copy.deepcopy(ar)]}
        ranks.append({**rank, "host": "node", "status": "match", "layers": [layer]})
    return {"parallelism": {"tp": 2, "dp": 2, "ep": 4}, "tokens_per_dp_rank": [2, 1],
            "cases": [{"layer_id": 88, "expected_columns_dtypes": [[4, "bfloat16"], [1, "float32"], [1, "int32"]],
                       "design_row_bytes": 16}], "ranks": ranks}


def test_native_completeness_and_tensor_bytes_are_recomputed():
    result = ANALYSIS.native_evidence(native_probe(), 4)
    assert result["rank_layer_keys_complete"]
    assert result["rank_layer_shape_matches"] == 4
    assert result["expected_collective_records"] == len(result["collectives"]) == 16
    assert result["tensor_descriptor_matches"] == result["tensor_descriptors"] == 48
    assert result["rank_group_matches"] == 4
    assert result["placement"]["status"] == "match"


def test_native_source_match_does_not_hide_malformed_observed_shape():
    probe = native_probe()
    probe["ranks"][2]["layers"][0]["collectives"][1]["outputs"][0]["shape"] = [4, 4]
    result = ANALYSIS.native_evidence(probe, 4)
    assert result["rank_layer_source_matches"] == 4
    assert result["rank_layer_shape_matches"] == 3
    assert result["tensor_descriptor_matches"] == 47


def test_native_missing_rank_is_not_complete():
    probe = native_probe()
    probe["ranks"].pop()
    result = ANALYSIS.native_evidence(probe, 4)
    assert not result["rank_layer_keys_complete"]
    assert result["placement"]["status"] == "discrepancy"
