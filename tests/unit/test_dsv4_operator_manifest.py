"""Source-derived DeepSeek-V4 scopes, without enabling execution or profiling."""

from collections import Counter
from pathlib import Path

import pytest

from frontier.attention.families import get_attention_family
from frontier.config.model_config import BaseModelConfig
from frontier.operators.binding import bind_operator_query, build_operator_manifest
from frontier.operators.families import get_operator_family
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan


ROOT = Path(__file__).resolve().parents[2]
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
BASE_ATTENTION = {
    "dsv4_fused_qkv_projection", "dsv4_fused_qkv_rmsnorm",
    "dsv4_q_projection", "dsv4_fused_q_kv_prepare", "dsv4_sparse_attention",
    "dsv4_output_inv_rope_quant", "dsv4_grouped_output_projection",
    "dsv4_output_projection",
}
COMPRESSOR = {
    "dsv4_compressor_projection", "dsv4_compressor_save_partial_states",
    "dsv4_compressor_fused_compress_store",
}
INDEXER = {
    "dsv4_index_weights_projection", "dsv4_index_compressor_projection",
    "dsv4_index_compressor_save_partial_states",
    "dsv4_index_compressor_fused_compress_store", "dsv4_index_query_projection",
    "dsv4_index_fused_query_prepare", "dsv4_index_scores", "dsv4_index_topk",
    "dsv4_index_short_context_fill",
}


@pytest.fixture(params=["simulation", "profiling"])
def model(request, monkeypatch):
    monkeypatch.chdir(ROOT)
    if request.param == "profiling":
        return ProfilingModelConfig.from_model_name(MODEL)
    return BaseModelConfig.create_from_name(MODEL)


@pytest.mark.parametrize(
    ("layer", "variant", "expected"),
    [(0, "uncompressed", BASE_ATTENTION),
     (1, "uncompressed", BASE_ATTENTION),
     (2, "csa", BASE_ATTENTION | COMPRESSOR | INDEXER),
     (3, "hca", BASE_ATTENTION | COMPRESSOR),
     (42, "csa", BASE_ATTENTION | COMPRESSOR | INDEXER)],
)
def test_layer_manifest_uses_exact_compression_scope_inventory(
    model, layer, variant, expected,
):
    manifest = build_operator_manifest(model, global_layer_id=layer)
    attention, mhc, routed, shared = manifest.family_bindings
    assert attention.family_id == "deepseek_v4_attention"
    assert attention.variant_id == variant
    assert {operator.name for operator in attention.family.operators} == expected
    assert mhc.family_id == "deepseek_v4_mhc"
    assert routed.family_id == "moe"
    assert routed.variant_id == ("hash" if layer < 3 else "score")
    assert shared.family_id == "share_expert"
    assert all(binding.frozen for binding in manifest.family_bindings)
    assert all(not operator.predictor_target and not operator.profiling_target
               and not operator.e2e_trace_target for operator in manifest.operators())
    for binding in manifest.family_bindings:
        assert binding.variant_id in binding.family.supported_variants
        with pytest.raises(NotImplementedError, match="frozen"):
            binding.require_enabled_for_execution()


def test_43_layer_inventory_counts_mhc_and_endpoints_once(model):
    manifests = [build_operator_manifest(model, global_layer_id=i) for i in range(43)]
    counts = Counter(operator.name for m in manifests for operator in m.operators())
    assert counts["dsv4_embedding"] == 1
    assert counts["dsv4_mhc_first_pre"] == 1
    assert counts["dsv4_mhc_ffn_post_attn_pre"] == 42
    assert counts["dsv4_mhc_attn_post_ffn_pre"] == 43
    for endpoint in (
        "dsv4_mhc_final_post", "dsv4_mhc_head", "dsv4_final_norm", "dsv4_lm_head",
    ):
        assert counts[endpoint] == 1
    assert counts["moe_grouped_gemm"] == 43
    assert counts["share_expert_up_proj"] == 43
    assert counts["dsv4_compressor_projection"] == 41
    assert counts["dsv4_index_query_projection"] == 21
    assert counts["dsv4_index_short_context_fill"] == 21
    assert not {"input_layernorm", "post_attention_layernorm", "add_attn_residual",
                "add_ffn_residual", "emb"}.intersection(counts)
    routes = Counter(m.family_bindings[2].variant_id for m in manifests)
    assert routes == {"hash": 3, "score": 40}


def test_new_scope_inventory_is_not_a_predictor_or_profiling_target():
    attention = get_attention_family("deepseek_v4_attention")
    assert get_operator_family("deepseek_v4_attention") is attention
    mhc = get_operator_family("deepseek_v4_mhc")
    for family in (attention, mhc):
        assert family.operators
        assert family.execution_enabled is False
        assert family.profiling_ops() == ()
        for operator in family.operators:
            assert operator.predictor_target is False
            assert operator.profiling_target is False
            assert operator.e2e_trace_target is False
            assert operator.execution_time_attr is None
            with pytest.raises(NotImplementedError, match="not enabled"):
                bind_operator_query(operator.name)


def test_variant_views_reuse_registered_scope_objects(model):
    canonical = get_attention_family("deepseek_v4_attention")
    names = {op.name: op for op in canonical.operators}
    for layer in (0, 2, 3):
        family = build_operator_manifest(model, global_layer_id=layer).families()[0]
        assert family.family_id == canonical.family_id
        assert all(operator is names[operator.name] for operator in family.operators)


def test_static_moe_views_leave_canonical_generic_targets_unchanged(model):
    manifest = build_operator_manifest(model, global_layer_id=2)
    for binding in manifest.family_bindings[2:]:
        canonical = get_operator_family(binding.family_id)
        assert canonical.execution_enabled is True
        assert canonical.profiling_ops()
        assert {operator.name for operator in canonical.operators} == {
            operator.name for operator in binding.family.operators
        }
        assert all(operator.predictor_target and operator.profiling_target
                   and operator.e2e_trace_target for operator in canonical.operators)
        assert binding.family.profiling_ops() == ()


def test_whole_mixed_model_still_requires_explicit_layer(model):
    with pytest.raises(ValueError, match="explicit global layer id"):
        build_operator_manifest(model)


@pytest.mark.parametrize("layer", [-1, 43, True, 0.0, "0"])
def test_layer_manifest_rejects_invalid_global_layer_id(model, layer):
    with pytest.raises((TypeError, ValueError), match="layer|int"):
        build_operator_manifest(model, global_layer_id=layer)


def test_generic_linear_producer_cannot_profile_v4_scope_inventory(model):
    with pytest.raises(NotImplementedError, match="DeepSeek-V4.*linear profiling"):
        build_profiling_plan(model, tp_size=8, attn_tp=[8], ffn_tp=[1], is_moe=True)


def test_csa_conditional_branches_remain_inventory_only(model):
    # The native short-context fill replaces index scoring/top-k, while HCA
    # compression/store can skip a non-boundary decode step. A static manifest
    # lists reachable scopes; it does not schedule every listed scope.
    csa = build_operator_manifest(model, global_layer_id=2).family_bindings[0]
    assert {"dsv4_index_scores", "dsv4_index_topk", "dsv4_index_short_context_fill"} <= {
        operator.name for operator in csa.family.operators
    }
    hca = build_operator_manifest(model, global_layer_id=3).family_bindings[0]
    assert "dsv4_compressor_fused_compress_store" in {
        operator.name for operator in hca.family.operators
    }
    assert csa.frozen and hca.frozen


@pytest.mark.parametrize("config_type", [BaseModelConfig, ProfilingModelConfig])
@pytest.mark.parametrize(
    ("name", "layer", "expected_ffn"),
    [("zai-org/GLM-5.3", 0, {"ffn"}),
     ("zai-org/GLM-5.3", 2, {"ffn"}),
     ("zai-org/GLM-5.3", 3, {"moe", "share_expert"}),
     ("zai-org/GLM-5.3", 77, {"moe", "share_expert"}),
     ("llama3.1-8b", 0, {"ffn"})],
)
def test_explicit_layer_manifest_selects_its_own_ffn_schedule(
    monkeypatch, config_type, name, layer, expected_ffn,
):
    monkeypatch.chdir(ROOT)
    config = (config_type.create_from_name(name) if config_type is BaseModelConfig
              else config_type.from_model_name(name))
    manifest = build_operator_manifest(config, global_layer_id=layer)
    assert {binding.family_id for binding in manifest.family_bindings[2:]} == expected_ffn


@pytest.mark.parametrize("config_type", [BaseModelConfig, ProfilingModelConfig])
def test_legacy_whole_model_manifest_preserves_model_wide_ffn_selection(
    monkeypatch, config_type,
):
    monkeypatch.chdir(ROOT)
    name = "zai-org/GLM-5.3"
    config = (config_type.create_from_name(name) if config_type is BaseModelConfig
              else config_type.from_model_name(name))
    assert {binding.family_id for binding in build_operator_manifest(config).family_bindings[2:]} == {
        "moe", "share_expert",
    }
