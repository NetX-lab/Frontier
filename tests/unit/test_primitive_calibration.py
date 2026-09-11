from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from frontier.profiling.runtime.sglang_primitives import attach_kernel_evidence, validate_plan
from frontier.runtime_cost.primitives import PrimitiveCalibration, aggregate_profiles
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload, RuntimeIdentity


IDENTITY = RuntimeIdentity("a" * 64, "synthetic-unit-device", 8, "synthetic-unit-stack")
NAMES = ("gemma_norm", "gemma_residual_norm", "shared_gate", "attention_post", "tp_allreduce")


def profiles(split="calibration", sizes=(16, 32)):
    # Invented arithmetic costs exercise mechanics only, never performance fixtures.
    return [{"schema_version": 1, "identity": asdict(IDENTITY), "split": split,
        "rank": rank, "world_size": 8, "versions": {"synthetic": True},
        "hardware": {"name": "synthetic-unit-device"}, "extra_collective_environment": {},
        "producer_source_sha256": "b" * 64, "method": "synthetic-unit-test-only",
        "rows": [{"primitive": name, "physical_size": size, "invocations_per_graph": count,
            "rank": rank, "backend": "ca" if name == "tp_allreduce" else "synthetic",
            "samples_ms": [size * .01 + .1 + rank * .001] * 20,
            "measurement_type": "HIP_GRAPH_REPLAY", "correctness_checked": True,
            "kernel_trace_kind": "representative_graph", "kernel_trace_invocations": 2, "kernel_count": 2,
            "kernel_names": ["synthetic-unit-kernel"] if rank == 0 else []}
            for name in NAMES for size in sizes for count in (32, 128)]} for rank in range(8)]


def test_rank_alignment_uses_per_repetition_max_not_max_rank_median():
    data = profiles()
    for rank in range(8):
        data[rank]["rows"][0]["samples_ms"] = [1. if i % 8 == rank else .1 for i in range(24)]
    rows, source = aggregate_profiles(data, split="calibration", identity=IDENTITY)
    row = next(r for r in rows if r["primitive"] == "gemma_norm" and r["physical_size"] == 16 and r["invocations_per_graph"] == 32)
    assert row["median_ms"] == 1. and max(row["rank_medians_ms"]) == .1
    assert source.startswith("sha256:")


@pytest.mark.parametrize("mutation", ["missing_rank", "duplicate_rank", "split", "identity", "row_rank",
    "missing_row", "duplicate_row", "backend", "nan", "zero", "few_repetitions", "alignment",
    "correctness", "measurement", "trace", "provenance", "environment", "boolean_size"])
def test_incomplete_or_untrustworthy_profiles_fail(mutation):
    data = profiles()
    row = data[0]["rows"][0]
    if mutation == "missing_rank": data.pop()
    elif mutation == "duplicate_rank": data[-1]["rank"] = 0
    elif mutation == "split": data[0]["split"] = "validation"
    elif mutation == "identity": data[0]["identity"]["device"] = "other"
    elif mutation == "row_rank": row["rank"] = 1
    elif mutation == "missing_row": data[0]["rows"].pop()
    elif mutation == "duplicate_row": data[0]["rows"].append(deepcopy(row))
    elif mutation == "backend": data[0]["rows"][-1]["backend"] = "rccl"
    elif mutation == "nan": row["samples_ms"][0] = float("nan")
    elif mutation == "zero": row["samples_ms"][0] = 0
    elif mutation == "few_repetitions": row["samples_ms"] = [1.] * 19
    elif mutation == "alignment": row["samples_ms"].append(1.)
    elif mutation == "correctness": row["correctness_checked"] = False
    elif mutation == "measurement": row["measurement_type"] = "KERNEL_ONLY"
    elif mutation == "trace": row["kernel_names"] = []
    elif mutation == "provenance": data[0]["producer_source_sha256"] = "c" * 64
    elif mutation == "environment": data[0]["extra_collective_environment"] = {"NCCL_TEST": "1"}
    elif mutation == "boolean_size": row["physical_size"] = True
    with pytest.raises(ValueError):
        aggregate_profiles(data, split="calibration", identity=IDENTITY)


def test_fit_is_bounded_and_validation_never_changes_predictions():
    fit = PrimitiveCalibration(profiles(), identity=IDENTITY)
    prediction = fit.primitive_ms("gemma_norm", 24)
    assert prediction == pytest.approx(.347)
    report = fit.validate(profiles("validation", (24,)))
    assert report["all_primitives_passed"]
    assert not report["full_forward_parity_admitted"]
    changed = profiles("validation", (24,))
    for payload in changed:
        for row in payload["rows"]: row["samples_ms"] = [10.] * 20
    assert not fit.validate(changed)["all_primitives_passed"]
    assert fit.primitive_ms("gemma_norm", 24) == prediction
    for size in (15, 33, True, 24.5):
        with pytest.raises(ValueError): fit.primitive_ms("gemma_norm", size)
    with pytest.raises(ValueError, match="overlaps"):
        fit.validate(profiles("validation", (16,)))
    changed = profiles("validation", (24,))
    for payload in changed: payload["producer_source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="provenance"):
        fit.validate(changed)


def query(component, layer=0):
    return CostQuery(IDENTITY, layer, "attention", component, DecodeWorkload(20, (1029,) * 20 + (1,) * 4),
        residual_from_layer=layer - 1 if component == "input_layernorm" and layer else
        layer if component == "post_attention_layernorm" else None)


def test_partial_provider_preserves_fusion_ownership_and_does_not_fill_other_costs():
    fit = PrimitiveCalibration(profiles(), identity=IDENTITY)
    for component, layer, primitive in (("input_layernorm", 0, "gemma_norm"),
        ("input_layernorm", 1, "gemma_residual_norm"), ("post_attention_layernorm", 0, "gemma_residual_norm"),
        ("attn_post_proj_gate", 0, "attention_post"), ("shared_gate_sigmoid_mul_add", 0, "shared_gate"),
        ("attention_tp_allreduce", 0, "tp_allreduce"), ("mlp_tp_allreduce", 0, "tp_allreduce")):
        q = query(component, layer)
        estimate = fit(q)
        assert estimate.source.endswith("/" + primitive) and estimate.query_key == q.key
        assert estimate.measurement_type == "HIP_GRAPH_REPLAY"
        assert estimate.basis == ("calibrated_collective" if q.is_collective else "isolated_compute")
    with pytest.raises(ValueError, match="Missing"):
        fit(query("attn_decode"))
    with pytest.raises(ValueError, match="another runtime"):
        fit(replace(query("input_layernorm"), identity=replace(IDENTITY, device="other")))


@pytest.mark.parametrize("mode", ["sensitivity", "spread"])
def test_unstable_calibration_rejected(mode):
    data = profiles()
    for payload in data:
        for row in payload["rows"]:
            if row["primitive"] == "gemma_norm" and row["invocations_per_graph"] == 32:
                row["samples_ms"] = [1., 2.] * 10 if mode == "spread" else [2.] * 20
    fit = PrimitiveCalibration(data, identity=IDENTITY)
    assert not fit.quality["gemma_norm"]["admitted"]
    with pytest.raises(ValueError, match="quality-rejected"):
        fit(query("input_layernorm"))


def test_kernel_evidence_is_separate_and_rejects_empty_partial_ranges():
    records = [{"primitive": "gemma_norm", "physical_size": 16, "invocations_per_graph": 32, "kernel_trace_invocations": 2}]
    events = [{"name": "primitive:gemma_norm:b16:n32", "cat": "user_annotation", "ph": "X", "ts": 10, "dur": 10},
        {"name": "primitive:gemma_norm:b16:n32", "cat": "gpu_user_annotation", "ph": "X", "ts": 10, "dur": 10}]
    with pytest.raises(ValueError, match="incomplete"):
        attach_kernel_evidence(records, events)
    events.extend({"name": "norm", "cat": "kernel", "ph": "X", "ts": ts, "dur": 1} for ts in (11, 15))
    attach_kernel_evidence(records, events)
    assert records[0]["kernel_count"] == 2
    events[-1]["ts"] = 100
    with pytest.raises(ValueError, match="incomplete"):
        attach_kernel_evidence(records, events)


@pytest.mark.parametrize("sizes,counts,reps,split", [([], [2], 20, "calibration"),
    ([16, 16], [2], 20, "calibration"), ([16], [1], 20, "calibration"),
    ([16], [2], 4, "calibration"), ([16], [2], 20, "unknown")])
def test_profile_plan_validation(sizes, counts, reps, split):
    with pytest.raises(ValueError): validate_plan(sizes, counts, reps, split)
