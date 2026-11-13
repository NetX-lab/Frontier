"""Step layer representation and physical ownership from the pinned native source.

These tests establish contracts, not native execution or latency accuracy.
"""

from collections import Counter
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from frontier.attention.gdn.config import LayerAttentionSpec
from frontier.attention.step5 import (
    Step5AttentionWorkload,
    describe_step5_layer,
    validate_step5_config,
)
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.model_architectures import LayerKind
from frontier.operators.binding import build_operator_manifest
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")


def layer(config, layer_id, *, tp=8, moe_tp=1, ep=32):
    return describe_step5_layer(
        config, layer_id, attention_tp_size=tp,
        moe_tp_size=moe_tp, expert_parallel_size=ep,
    )


def workload(config, *, queries=(1,), lengths=(4097,), rows=None, mode="none", phase="decode"):
    return Step5AttentionWorkload.from_lengths(
        config, phase=phase, query_lengths=queries, sequence_lengths=lengths,
        physical_query_rows=sum(queries) if rows is None else rows,
        block_size_tokens=64, block_table_columns=128,
        max_model_len_tokens=8192, graph_mode=mode,
    )


def projections(metadata):
    return {projection.module_prefix: projection for projection in metadata.projections}


def test_tp8_sparse_packed_shapes_and_ep32_expert_placement(model):
    metadata = layer(model, 3)
    assert metadata.attention_spec == LayerAttentionSpec(3, "step5_attention", "sparse_gqa")
    shapes = projections(metadata)
    packed = shapes["model.layers.3.self_attn.qkv_indexer_proj"]
    assert packed.logical_weight_shape == (18496, 4096)
    assert packed.rank_weight_shape == (3464, 4096)
    assert packed.weight_precision is packed.input_precision is packed.output_precision is PrecisionType.BF16
    index_weight = shapes["model.layers.3.self_attn.sparse_indexer_w"]
    assert index_weight.logical_weight_shape == (16, 4096)
    assert index_weight.rank_weight_shape == (4, 4096)
    assert index_weight.output_precision is PrecisionType.BF16
    output = shapes["model.layers.3.self_attn.o_proj"]
    assert output.logical_weight_shape == (4096, 12288)
    assert output.rank_weight_shape == (4096, 1536)
    w13 = shapes["model.layers.3.moe.experts.w13_weight"]
    w2 = shapes["model.layers.3.moe.experts.w2_weight"]
    assert w13.logical_weight_shape == (352, 3072, 4096)
    assert w13.rank_weight_shape == (11, 3072, 4096)
    assert w2.logical_weight_shape == (352, 4096, 1536)
    assert w2.rank_weight_shape == (11, 4096, 1536)
    assert w13.weight_precision is w2.weight_precision is PrecisionType.FP8
    assert w13.input_precision is w13.output_precision is PrecisionType.BF16
    assert [contract.layer_kind for contract in metadata.ffn_contracts] == [LayerKind.ROUTED, LayerKind.SHARED]
    assert [contract.tensor_parallel_size for contract in metadata.ffn_contracts] == [1, 8]
    shared_down = shapes["model.layers.3.moe.share_expert.down_proj"]
    assert shared_down.rank_weight_shape == (4096, 192)
    assert shared_down.output_precision is shared_down.weight_precision is PrecisionType.BF16


def test_tp8_swa_qkv_and_dense_layers_keep_the_final_dense_ffn(model):
    for layer_id in (0, 1, 2, 91):
        metadata = layer(model, layer_id)
        shapes = projections(metadata)
        if layer_id != 91:
            qkv = shapes[f"model.layers.{layer_id}.self_attn.qkv_proj"]
            assert qkv.rank_weight_shape == (1920, 4096)
        else:
            assert metadata.attention_spec.variant_id == "sparse_gqa"
            assert shapes["model.layers.91.self_attn.qkv_indexer_proj"].rank_weight_shape == (3464, 4096)
        assert [contract.layer_kind for contract in metadata.ffn_contracts] == [LayerKind.DENSE]
        up = shapes[f"model.layers.{layer_id}.mlp.gate_up_proj"]
        down = shapes[f"model.layers.{layer_id}.mlp.down_proj"]
        assert up.logical_weight_shape == (27648, 4096)
        assert up.rank_weight_shape == (3456, 4096)
        assert down.rank_weight_shape == (4096, 1728)
        assert up.weight_precision is down.weight_precision is PrecisionType.BF16
        assert metadata.routed_swiglu_limit is metadata.router_precision is None


def test_all_92_layers_use_existing_operator_and_ffn_owners(model):
    descriptions = [layer(model, layer_id) for layer_id in range(92)]
    assert Counter(item.attention_spec.variant_id for item in descriptions) == {"swa": 69, "sparse_gqa": 23}
    assert Counter(item.ffn_contracts[0].layer_kind for item in descriptions) == {LayerKind.DENSE: 4, LayerKind.ROUTED: 88}
    registered = {operator.name for operator in build_operator_manifest(model).operators()}
    routed_precisions = []
    for item in descriptions:
        for projection in item.projections:
            assert projection.operator_name in registered
            if projection.module_prefix.endswith("experts.w13_weight"):
                routed_precisions.append(projection.weight_precision)
        assert item.attention_spec.global_layer_id < 92
        if item.ffn_contracts[0].layer_kind is LayerKind.ROUTED:
            assert item.ffn_merge.operator_name in registered
            assert item.ffn_merge.module_prefix == f"model.layers.{item.attention_spec.global_layer_id}.moe"
            assert item.ffn_merge.input_precisions == (PrecisionType.BF16, PrecisionType.BF16)
            assert item.ffn_merge.output_precision is PrecisionType.FP32
            assert item.ffn_merge.hidden_dim == model.embedding_dim
            for projection in item.projections:
                if ".moe.share_expert." in projection.module_prefix:
                    assert projection.weight_precision is PrecisionType.BF16
        else:
            assert item.ffn_merge is None
    assert Counter(routed_precisions) == {PrecisionType.FP8: 85, PrecisionType.BF16: 3}


def test_main_norm_residual_and_indexer_dtypes_have_separate_owners(model):
    sparse = layer(model, 3)
    assert model.uses_fused_add_norm is sparse.uses_fused_add_norm is True
    assert sparse.model_precision is PrecisionType.BF16
    assert sparse.norm_precision is sparse.residual_precision is PrecisionType.FP32
    assert sparse.indexer_norm_precision is sparse.indexer_key_precision is PrecisionType.BF16
    assert sparse.indexer_query_precision is PrecisionType.FP8
    assert sparse.indexer_weight_precision is PrecisionType.FP32
    router = projections(sparse)["model.layers.3.moe.gate"]
    assert router.input_precision is PrecisionType.BF16
    assert router.weight_precision is router.output_precision is PrecisionType.FP32
    assert sparse.router_scoring == "sigmoid"
    assert sparse.router_bias is sparse.router_renormalize is True
    assert sparse.router_scaling_factor == model.moe_router_scaling_factor
    assert layer(model, 0).indexer_norm_precision is None


def test_metadata_preserves_per_layer_rope_and_separate_clamp_arrays(model):
    routed_limits = list(model.swiglu_limits)
    shared_limits = list(model.swiglu_limits_shared)
    routed_limits[3] = 0
    shared_limits[3] = 5
    shared_limits[0] = 3
    changed = replace(model, swiglu_limits=tuple(routed_limits), swiglu_limits_shared=tuple(shared_limits))
    assert layer(changed, 3).routed_swiglu_limit == 0
    assert layer(changed, 3).dense_swiglu_limit == 5
    assert layer(changed, 0).dense_swiglu_limit == 3
    for layer_id in (0, 3, 91):
        description = layer(changed, layer_id)
        assert description.rope_theta == model.layer_rope_thetas[layer_id]
        assert description.rotary_dim == int(192 * model.partial_rotary_factors[layer_id])


def test_runtime_and_profiling_layer_metadata_agree(model):
    profiling = ProfilingModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    for layer_id in (0, 3, 88, 90, 91):
        assert layer(model, layer_id) == layer(profiling, layer_id)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_unquantized_partitioned_swa_gate_uses_actual_and_quantization_prefixes(model, tp):
    metadata = layer(model, 0, tp=tp)
    assert metadata.fused_qkv_gate is True
    qkv = projections(metadata)["model.layers.0.self_attn.qkvg_proj"]
    assert qkv.quantization_prefix == "model.layers.0.self_attn.qkv_proj"
    assert qkv.logical_weight_shape == (13888, 4096)
    assert qkv.rank_weight_shape == (13888 // tp, 4096)
    scopes = metadata.attention_scopes(workload(model))
    assert scopes[0] == ("step5_qkv_projection", ("q_projection", "k_projection", "v_projection", "gate_projection"))
    assert "step5_gate_projection" not in dict(scopes)
    assert dict(scopes)["step5_headwise_gate"] == ("headwise_sigmoid_gate",)


def test_replicated_swa_gate_remains_separate(model):
    metadata = layer(model, 0)
    assert metadata.fused_qkv_gate is False
    scopes = dict(metadata.attention_scopes(workload(model)))
    assert scopes["step5_gate_projection"] == ("gate_projection",)
    assert scopes["step5_headwise_gate"] == ("headwise_sigmoid_gate",)


@pytest.mark.parametrize(("phase", "queries", "lengths", "rows", "mode", "fused"), [
    ("decode", (1,), (512,), 128, "none", True),
    ("decode", (1,), (512,), 129, "none", False),
    ("decode", (1, 1), (512, 513), 16, "full", True),
    ("decode", (1,), (512,), 8, "piecewise", False),
    ("prefill", (1,), (1,), 1, "none", True),
    ("prefill", (8,), (8,), 8, "none", False),
    ("mixed", (1, 8), (512, 520), 16, "none", False),
])
def test_sparse_preparation_owns_summary_update_exactly_once(model, phase, queries, lengths, rows, mode, fused):
    metadata = layer(model, 3)
    item = workload(model, phase=phase, queries=queries, lengths=lengths, rows=rows, mode=mode)
    scopes = metadata.attention_scopes(item)
    owners = dict(scopes)
    assert ("step5_indexer_prepare" not in owners) is fused
    assert scopes[0][0] == "step5_qkv_projection"
    order = list(owners)
    if fused:
        assert order.index("step5_indexer_weights") < order.index("step5_attention_prepare")
    else:
        assert order.index("step5_attention_prepare") < order.index("step5_indexer_weights")
    census = Counter(work for _, covered in scopes for work in covered)
    assert census["summary_update"] == 1
    assert census["index_q_fp8_conversion"] == 1
    assert all(count == 1 for count in census.values())
    assert scopes[-1] == ("step5_output_projection", ("output_projection",))


@pytest.mark.parametrize(("tokens", "scored"), [(8, 0), (4096, 0), (4097, 513), (8192, 1024)])
def test_short_prefill_bypasses_summary_pack_score_and_selection(model, tokens, scored):
    item = workload(model, phase="prefill", queries=(tokens,), lengths=(tokens,))
    assert item.score_regions == scored
    attention = dict(layer(model, 3).attention_scopes(item))["step5_sparse_attention"]
    for name in ("summary_packing", "selector_scoring", "historical_region_selection"):
        assert (name in attention) is bool(scored)
    assert "summary_update" in attention
    assert "kv_staging_or_gather" in attention
    assert "headwise_sigmoid_gate" in attention


def test_single_token_prefill_uses_decode_scores_and_paged_history(model):
    item = workload(model, phase="prefill", queries=(1,), lengths=(1,), rows=8)
    assert item.is_single_token_step is True
    assert item.score_regions == 1024
    attention = dict(layer(model, 3).attention_scopes(item))["step5_sparse_attention"]
    assert "summary_packing" not in attention
    assert "selector_scoring" in attention
    assert "historical_region_selection" in attention
    assert "paged_kv_read" in attention
    assert "kv_staging_or_gather" not in attention


def test_decode_physical_extent_and_padding_do_not_credit_live_progress(model):
    item = workload(model, queries=(1, 1), lengths=(9, 4097), rows=128)
    assert item.valid_tokens == 2
    assert item.padded_tokens == 126
    assert item.score_regions == 1024
    assert item.selected_region_bounds == (2, 513)
    narrower = replace(item, block_table_columns=65)
    assert narrower.score_regions == 520
    assert narrower.valid_tokens == item.valid_tokens
    assert narrower.sequence_lengths == item.sequence_lengths
    assert narrower.selected_region_bounds == item.selected_region_bounds


@pytest.mark.parametrize(("changes", "message"), [
    ({"query_lengths": (2,), "sequence_lengths": (1,)}, "context limit"),
    ({"query_lengths": (2,), "sequence_lengths": (4,), "physical_query_rows": 2}, "ordinary decode"),
    ({"physical_query_rows": 0}, "physical_query_rows"),
    ({"max_model_len_tokens": 4096}, "context limit"),
    ({"block_table_columns": 64}, "physical block-table extent"),
    ({"block_size_tokens": 63}, "integer number of regions"),
    ({"graph_mode": "invalid"}, "graph_mode"),
])
def test_workload_rejects_invalid_physical_or_live_units(model, changes, message):
    item = workload(model)
    with pytest.raises(ValueError, match=message):
        replace(item, **changes)


def test_wrong_per_layer_binding_and_mtp_execution_are_rejected(model):
    specs = list(model.get_layer_attention_specs())
    specs[3] = LayerAttentionSpec(3, "step5_attention", "swa")
    model._layer_attention_specs_cache = tuple(specs)
    with pytest.raises(ValueError, match="binding"):
        layer(model, 3)
    with pytest.raises(ValueError, match="out of range"):
        layer(model, 92)


def test_zero_centered_and_residual_fusion_conflicts_fail_closed(model):
    with pytest.raises(NotImplementedError, match="zero_centered"):
        validate_step5_config(replace(model, zero_centered=False))
    conflicting = copy(model)
    conflicting.fused_add_norm_capability = False
    with pytest.raises(ValueError, match="uses_fused_add_norm"):
        validate_step5_config(conflicting)
    ordinary = replace(model, fp32_residual_connection=False, fused_add_norm_capability=False)
    assert layer(ordinary, 3).residual_precision is PrecisionType.BF16
    assert layer(ordinary, 3).uses_fused_add_norm is False


@pytest.mark.parametrize(("tp", "moe_tp", "ep", "message"), [
    (3, 1, 32, "Q heads"), (8, 5, 32, "width"), (8, 1, 64, "divisible by EP"),
])
def test_inexact_parallel_geometry_never_silently_floors_shapes(model, tp, moe_tp, ep, message):
    with pytest.raises(ValueError, match=message):
        layer(model, 3, tp=tp, moe_tp=moe_tp, ep=ep)
