"""Selected Step physical contracts; CPU fixtures do not qualify GPU prices."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor, CollectivePlanCost
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.step5_cost import step5_operator_binding
from frontier.attention.step5 import describe_step5_layer
from frontier.operators.families import get_moe_profiling_operators


def context(quant_dtype=None):
    return dict(model_architecture_profile="step5", routing_runtime_path="position_cyclic",
                quant_dtype=quant_dtype, defer_input_quant=False)


@pytest.mark.parametrize("quant,names", [
    (None, ("moe_gating_linear", "moe_gating_routing_topk", "moe_grouped_gemm")),
    ("float8_e4m3fn", ("moe_gating_linear", "moe_gating_routing_topk", "moe_input_quantize", "moe_grouped_gemm")),
])
def test_selected_registry_and_real_routed_binding(quant, names):
    assert tuple(op.name for op in get_moe_profiling_operators(context(quant))) == names
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    profile = model.get_model_architecture_profile()
    prepare = next(op for op in get_moe_profiling_operators(context("float8_e4m3fn"))
                   if op.name == "moe_input_quantize")
    assert prepare.ep_agnostic
    contract = profile.resolve_layer_contract(model, layer_id=3, operator_name=prepare.name,
        attention_tp_size=8, ffn_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    assert contract.operator_family_id == "moe"
    metadata = describe_step5_layer(model, 3, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    predictor = SimpleNamespace(_model_config=model, _get_model_architecture_profile=lambda: profile,
        _get_ffn_tp_key_for_linear_op=lambda: 8,
        _replica_config=SimpleNamespace(attn_tensor_parallel_size=8, moe_tensor_parallel_size=1,
                                      moe_expert_parallel_size=16))
    assert step5_operator_binding(predictor, metadata, prepare.name, context=context("float8_e4m3fn"))[0] is PrecisionType.FP8
    assert step5_operator_binding(predictor, metadata, "moe_gating_linear", context=context(quant))[0] is PrecisionType.FP32


def test_legacy_and_deferred_prepare_target_selection():
    assert tuple(op.name for op in get_moe_profiling_operators()) == (
        "moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm")
    assert "moe_input_quantize" not in [op.name for op in get_moe_profiling_operators(
        {**context("float8_e4m3fn"), "defer_input_quant": True})]
    with pytest.raises(ValueError, match="quant_dtype"):
        get_moe_profiling_operators(context("int8"))


def plan(sizes=(16, 1), phase="dispatch"):
    return CollectivePlan(phase, sizes, (CollectiveTensor("activation", "bfloat16", 2, 4096, 4096),),
        tuple((t, 8 + t) for t in range(8)), "none", json.dumps(context()), '{"supplied_output": true}')


def test_lossless_plan_order_identity_and_count_separation():
    p = plan()
    assert p.stages == (("Broadcast", 0, 0), ("Broadcast", 0, 1))
    assert p.endpoint_bytes(8) == (8192, 17 * 8192)
    assert p.cache_key() != plan((1, 16)).cache_key()
    assert plan((1, 1)).stages == (("AllGather", 0, None),)
    with pytest.raises(ValueError, match="rank order"):
        replace(p, groups=((0, 8),))
    with pytest.raises(ValueError, match="uniform"):
        replace(p, graph_mode="full")
    cost = CollectivePlanCost(6., 1., (2.,) * 16, 3., (7,) * 16,
                             '{"evidence_kind":"cpu_fixture"}', p.cache_key(), "cpu_fixture")
    assert cost.kernel_counts[0] == 7 and len(p.stages) == 2
    with pytest.raises(ValueError, match="composition"):
        replace(cost, duration_ms=5.)
