from dataclasses import replace
import json

import pytest

from frontier.validation.graph_event_report import (
    expected_scopes, summarize_decoder_graph_events, summarize_graph_events,
)
from frontier.validation.report import summarize_capture, summarize_profile_perturbation
from frontier.validation.sglang_graph_events import DEFAULT_COMPONENTS
from tests.unit.test_operator_trace import timing


def fixture():
    baseline = [timing(batch_id="base", rank=r, forward_gpu_ms=10 + r) for r in range(2)]
    records = [replace(r, batch_id="events", profiled=True) for r in baseline]
    quality = summarize_profile_perturbation(baseline + records, tensor_parallel_size=2)
    samples = [{"batch_id": "events", "rank": rank, "layer_id": 0, "component": "gdn",
                "measurement_type": "HIP_GRAPH_EVENT", "parent_id": None,
                "physical_size": 1, "capture_id": 10 + rank, "inclusive_gpu_ms": 1 + rank}
               for rank in range(2)]
    return records, samples, quality


def summarize(records, samples, quality):
    return summarize_graph_events(records, samples, quality=quality,
                                  tensor_parallel_size=2, expected={(0, "gdn")})


def test_model_schedule_resolves_disjoint_default_scopes():
    expected = expected_scopes(["gdn", "gdn", "gdn", "attention"], [0, 3], DEFAULT_COMPONENTS)
    assert len(expected) == 8
    assert (0, "gdn") in expected and (3, "full_attention") in expected
    assert (3, "gdn") not in expected
    with pytest.raises(ValueError, match="Nested"):
        expected_scopes(["gdn"], [0], ["gdn", "gdn_attn"])
    with pytest.raises(ValueError, match="exist"):
        expected_scopes(["gdn"], [1], DEFAULT_COMPONENTS)
    with pytest.raises(ValueError, match="Unknown"):
        expected_scopes(["gdn"], [0], ["unknown"])


def test_report_preserves_rank_identity_and_never_exports_native_profiles():
    records, samples, quality = fixture()
    result = summarize(records, samples, quality)
    assert result["scope_samples"] == 2
    assert result["decode_batches"] == 1
    assert result["all_decode_batches_pass_latency_check"]
    assert not result["native_profile_export_admitted"]
    assert [r["gpu_median_of_repeat_medians_ms"] for r in result["summaries"]] == [1, 2]
    quality["batches"][0]["passes_latency_check"] = False
    rejected = summarize(records, samples, quality)
    assert not rejected["all_decode_batches_pass_latency_check"]
    assert rejected["scope_samples"] == 2  # retain, do not silently trim rejected samples
    assert all(r["samples_failing_forward_latency_check"] == 1 for r in rejected["summaries"])


@pytest.mark.parametrize("change", [
    {"batch_id": "unrelated"}, {"rank": 3}, {"layer_id": 1}, {"layer_id": False},
    {"component": "full_attention"}, {"parent_id": 1}, {"capture_id": 0},
    {"physical_size": 4}, {"measurement_type": "KERNEL_ONLY"},
    {"inclusive_gpu_ms": -1}, {"inclusive_gpu_ms": float("nan")}, {"inclusive_gpu_ms": 12},
])
def test_malformed_or_misattributed_graph_scope_fails(change):
    records, samples, quality = fixture()
    samples[0].update(change)
    with pytest.raises(ValueError):
        summarize(records, samples, quality)


def test_scope_coverage_and_quality_are_required():
    records, samples, quality = fixture()
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(records, samples[:1], quality)
    with pytest.raises(ValueError, match="duplicate"):
        summarize(records, samples + samples[:1], quality)
    with pytest.raises(ValueError, match="perturbation"):
        summarize(records, samples, {"batches": []})
    with pytest.raises(ValueError, match="full-decode"):
        summarize([replace(r, profiled=False) for r in records], samples, quality)


def test_disjoint_scope_sum_cannot_exceed_same_rank_forward():
    records, samples, quality = fixture()
    samples += [{**row, "component": "input_layernorm", "inclusive_gpu_ms": 10}
                for row in samples]
    with pytest.raises(ValueError, match="exceed"):
        summarize_graph_events(records, samples, quality=quality, tensor_parallel_size=2,
                               expected={(0, "gdn"), (0, "input_layernorm")})


def test_whole_decoder_summary_preserves_rank_and_forward_boundary_gap():
    records, _, quality = fixture()
    samples = [{
        "batch_id": "events", "rank": rank, "component": "decoder",
        "first_layer_id": 0, "last_layer_id": 3,
        "measurement_type": "HIP_GRAPH_EVENT", "physical_size": 1,
        "capture_id": 10 + rank, "inclusive_gpu_ms": 9 + rank,
    } for rank in range(2)]
    result = summarize_decoder_graph_events(
        records, samples, tensor_parallel_size=2, num_layers=4, quality=quality)
    assert result["decoder_scope_samples"] == 2
    assert result["all_decode_batches_pass_latency_check"]
    assert [row["forward_minus_decoder_median_ms"]
            for row in result["summaries"]] == [1, 1]
    assert not result["native_profile_export_admitted"]


def test_whole_decoder_summary_rejects_wrong_boundary_and_missing_rank():
    records, _, quality = fixture()
    sample = {
        "batch_id": "events", "rank": 0, "component": "decoder",
        "first_layer_id": 0, "last_layer_id": 2,
        "measurement_type": "HIP_GRAPH_EVENT", "physical_size": 1,
        "capture_id": 10, "inclusive_gpu_ms": 9,
    }
    with pytest.raises(ValueError, match="boundary"):
        summarize_decoder_graph_events(
            records, [sample], tensor_parallel_size=2, num_layers=4, quality=quality)
    sample["last_layer_id"] = 3
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_decoder_graph_events(
            records, [sample], tensor_parallel_size=2, num_layers=4, quality=quality)


def test_capture_report_includes_separate_graph_output_diagnostics(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "topology": {"tp": 1},
        "options": {"freeze_decode_inputs": True}}))
    baseline = timing(step_wall_ms=11, decode_input_sha256="a" * 64)
    graph = replace(baseline, batch_id="graph", profiled=True)
    for name, row in (("batches", baseline), ("graph-batches", graph)):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(row.to_dict()) + "\n")
    check = {"batch_size": 1, "input_len": 1024, "output_tokens": 16,
             "different_from_first_repeat": 0, "tp_output_agreement": True, "output_sha256": "first"}
    (tmp_path / "output-checks-rank0.json").write_text(json.dumps([check]))
    (tmp_path / "graph-output-checks-rank0.json").write_text(json.dumps([
        {**check, "output_sha256": "second", "different_from_first_repeat": 1}]))
    result = summarize_capture(tmp_path, skip_decode_steps=0)
    assert result["graph_event_perturbation"]["all_pass_latency_check"]
    assert result["graph_output_repeatability"]["fixed_decode_inputs"]
    assert result["graph_output_repeatability"]["max_changed_output_fraction"] == 1 / 16
    assert result["graph_output_repeatability"]["repetitions_matching_any_baseline_output_digest"] == 0
