from copy import deepcopy

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.runtime.sglang_moe import (
    MOE_ROUTING_PRIMITIVES,
    moe_routed_spec,
    moe_routing_spec,
    reconstruct_topk_ids,
    validate_moe_route_workload,
    validate_moe_routed_spec,
    validate_moe_routing_spec,
)
from frontier.profiling.runtime.sglang_primitives import DEFAULT_PRIMITIVES
from frontier.runtime_cost.primitives import PrimitiveCalibration
from frontier.runtime_cost.routed_primitives import ExactRoutedCalibration
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload
from tests.unit.test_primitive_calibration import IDENTITY, profiles


@pytest.fixture
def model():
    return ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")


def test_qwen38_moe_topk_contract(model):
    spec = moe_routing_spec(model)
    assert spec == {
        "hidden_size": 8192,
        "num_experts": 512,
        "top_k": 10,
        "logits_dtype": "bfloat16",
        "weights_dtype": "float32",
        "ids_dtype": "int32",
        "scoring": "softmax",
        "renormalize": False,
        "backend": "aiter.fused_moe.fused_topk/topk_softmax",
        "includes_router_linear": False,
        "includes_padding_mask": False,
    }
    validate_moe_routing_spec(spec)
    assert not set(MOE_ROUTING_PRIMITIVES) & set(DEFAULT_PRIMITIVES)


def test_qwen38_tp8_routed_mxfp4_contract(model):
    spec = moe_routed_spec(model, 8)
    assert spec["local_intermediate_size"] == 256
    assert spec["w13_packed_shape"] == (512, 512, 4096)
    assert spec["w2_packed_shape"] == (512, 8192, 128)
    assert spec["w13_scale_shape"] == (512, 512, 256)
    assert spec["w2_scale_shape"] == (512, 8192, 8)
    assert spec["block_size_m"] == 32
    assert not spec["includes_sorting"]
    assert not spec["includes_tp_allreduce"]
    validate_moe_routed_spec(spec, 8)


def test_route_histogram_reconstruction_is_deterministic_and_unique(model):
    spec = moe_routed_spec(model, 8)
    counts = (4,) * 20 + (0,) * (spec["num_experts"] - 20)
    workload = validate_moe_route_workload(8, counts, spec)
    assert workload == {
        "physical_size": 8,
        "active_experts": 20,
        "maximum_expert_load": 4,
        "sorted_token_blocks": 20,
    }
    rows = reconstruct_topk_ids(8, counts, spec)
    assert rows == reconstruct_topk_ids(8, counts, spec)
    assert all(len(row) == spec["top_k"] and len(set(row)) == len(row) for row in rows)
    rebuilt = [0] * spec["num_experts"]
    for row in rows:
        for expert in row:
            rebuilt[expert] += 1
    assert tuple(rebuilt) == counts


def test_invalid_route_histograms_fail_closed(model):
    spec = moe_routed_spec(model, 8)
    with pytest.raises(ValueError):
        validate_moe_route_workload(8, (80,) + (0,) * 511, spec)
    with pytest.raises(ValueError):
        validate_moe_route_workload(8, (1,) * 80, spec)


def moe_profiles(model, split="calibration", sizes=(16, 32)):
    result = profiles(split, sizes)
    spec = moe_routing_spec(model)
    for payload in result:
        base = [row for row in payload["rows"] if row["primitive"] == "gemma_norm"]
        payload["rows"] = []
        for row in base:
            row = deepcopy(row)
            row.update(
                primitive="moe_routing_topk",
                backend=spec["backend"],
                moe_routing_spec=deepcopy(spec),
            )
            payload["rows"].append(row)
    return result


def test_moe_topk_fits_holdout_and_prices_physical_work(model):
    fit = PrimitiveCalibration(moe_profiles(model), identity=IDENTITY)
    assert fit.validate(moe_profiles(model, "validation", (24,)))[
        "all_primitives_passed"]
    query = CostQuery(
        IDENTITY, 3, "attention", "moe_routing_topk",
        DecodeWorkload(20, (1029,) * 20 + (1,) * 4))
    assert fit(query).basis == "isolated_compute"


@pytest.mark.parametrize("field,value", [
    ("num_experts", 256),
    ("top_k", 8),
    ("renormalize", True),
    ("backend", "torch.topk"),
])
def test_changed_moe_topk_contract_fails_closed(model, field, value):
    data = moe_profiles(model)
    data[0]["rows"][0]["moe_routing_spec"][field] = value
    with pytest.raises(ValueError):
        PrimitiveCalibration(data, identity=IDENTITY)


def routed_profiles(model, primitive="moe_sorting"):
    spec = moe_routed_spec(model, 8)
    counts = (10,) * 24 + (0,) * 488
    query = CostQuery(
        IDENTITY, 0, "gdn", primitive,
        DecodeWorkload(20, (1029,) * 20 + (1,) * 4),
        physical_expert_counts=counts)
    workload = validate_moe_route_workload(24, counts, spec)
    payloads = []
    for rank in range(8):
        rows = []
        for graph_length, value in ((32, .0101), (128, .0100)):
            rows.append({
                "primitive": primitive,
                "query_key": query.key,
                "layer_id": 0,
                "physical_size": 24,
                "physical_expert_counts": list(counts),
                "invocations_per_graph": graph_length,
                "rank": rank,
                "backend": (spec["sorting_backend"] if primitive == "moe_sorting"
                            else spec["experts_backend"]),
                "samples_ms": [value + rank * 1e-6] * 20,
                "measurement_type": "HIP_GRAPH_REPLAY",
                "correctness_checked": True,
                "moe_routed_spec": deepcopy(spec),
                "moe_route_workload": deepcopy(workload),
                "kernel_names": (["aiter::opus_moe_sorting_entry<P0>"]
                                 if primitive == "moe_sorting"
                                 else ["fused_mx_quant_moe_sort_kernel",
                                       "mfma_moe1_test", "mfma_moe2_test"])
                                if rank == 0 else [],
                "kernel_count": (2 if primitive == "moe_sorting" else 6)
                                if rank == 0 else None,
                "kernel_trace_kind": "representative_graph",
                "kernel_trace_invocations": 2,
                "trace_scope": f"routed:{primitive}:l0:b24:n{graph_length}",
            })
        payloads.append({
            "schema_version": 1,
            "profile_kind": "exact_routed_query",
            "identity": IDENTITY.__dict__,
            "rank": rank,
            "world_size": 8,
            "versions": {"torch": "x", "rocm": "y"},
            "rows": rows,
            "producer_source_sha256": "f" * 64,
            "hardware": {"name": "MI355X", "arch": "gfx950"},
            "extra_collective_environment": {},
            "method": "unit",
        })
    return query, payloads


@pytest.mark.parametrize("primitive", [
    "moe_sorting",
    "moe_experts_quant_gemm_combine",
])
def test_exact_route_profiles_price_only_the_exact_query(model, primitive):
    query, payloads = routed_profiles(model, primitive)
    fit = ExactRoutedCalibration(payloads, identity=IDENTITY)
    assert fit.report()["all_queries_passed"]
    assert fit(query).basis == "isolated_compute"
    changed = CostQuery(
        query.identity, 1, query.layer_kind, query.component, query.workload,
        physical_expert_counts=query.physical_expert_counts)
    with pytest.raises(ValueError, match="exact routed"):
        fit(changed)


def test_exact_route_profiles_require_every_aligned_rank(model):
    _, payloads = routed_profiles(model)
    with pytest.raises(ValueError, match="Complete"):
        ExactRoutedCalibration(payloads[:-1], identity=IDENTITY)
