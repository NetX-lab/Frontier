from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    get_attention_family,
    iter_attention_families,
    iter_execution_enabled_families,
)
from frontier.config.model_config import BaseModelConfig
from frontier.operators.binding import FamilyBinding, build_operator_manifest
from frontier.operators.families import OPERATOR_REGISTRY
from frontier.operators.registry import OperatorRegistry
from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorPhase,
    OperatorRole,
    OperatorSpec,
    ResourceClass,
)
from frontier.types import ActivationType, NormType


def test_operator_registry_preserves_order_and_rejects_duplicate_family_ids() -> None:
    registry = OperatorRegistry()
    registry.register(DENSE_ATTENTION_FAMILY)
    registry.register(LATENT_MLA_ATTENTION_FAMILY)

    assert registry.get_family("dense_attention") is DENSE_ATTENTION_FAMILY
    assert tuple(registry.iter_families()) == (
        DENSE_ATTENTION_FAMILY,
        LATENT_MLA_ATTENTION_FAMILY,
    )
    with pytest.raises(ValueError, match="Duplicate operator family"):
        registry.register(DENSE_ATTENTION_FAMILY)


def test_operator_registry_execution_enabled_view_excludes_frozen_families() -> None:
    registry = OperatorRegistry()
    registry.register(DENSE_ATTENTION_FAMILY)
    registry.register(DSA_ATTENTION_FAMILY)

    assert tuple(registry.iter_execution_enabled_families()) == (DENSE_ATTENTION_FAMILY,)


def test_operator_registry_uses_generic_execution_enabled_contract() -> None:
    enabled_family = OperatorFamilySpec(
        family_id="enabled_unit_family",
        display_name="Enabled Unit Family",
        supported_variants=("unit",),
        operators=(),
    )
    disabled_family = OperatorFamilySpec(
        family_id="disabled_unit_family",
        display_name="Disabled Unit Family",
        supported_variants=("unit",),
        operators=(),
        execution_enabled=False,
        disabled_reason="unit test disabled family",
    )
    registry = OperatorRegistry()
    registry.register(enabled_family)
    registry.register(disabled_family)

    assert tuple(registry.iter_execution_enabled_families()) == (enabled_family,)
    with pytest.raises(NotImplementedError, match="unit test disabled family"):
        disabled_family.require_enabled_for_execution()


def test_family_binding_frozen_state_fails_closed_without_family_override() -> None:
    family = OperatorFamilySpec(
        family_id="binding_disabled_unit_family",
        display_name="Binding Disabled Unit Family",
        supported_variants=("unit",),
        operators=(
            OperatorSpec(
                name="unit_op",
                role=OperatorRole.PREFILL_KERNEL,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="unit_time",
                resource_class=ResourceClass.COMP,
            ),
        ),
    )
    binding = FamilyBinding(
        family_id=family.family_id,
        variant_id="unit",
        family=family,
        frozen=True,
        reason="unit binding freeze",
    )

    with pytest.raises(NotImplementedError, match="unit binding freeze"):
        binding.require_enabled_for_execution()


def test_attention_family_accessors_are_thin_registry_views() -> None:
    assert get_attention_family("dense_attention") is DENSE_ATTENTION_FAMILY
    assert tuple(iter_attention_families()) == (
        DENSE_ATTENTION_FAMILY,
        LATENT_MLA_ATTENTION_FAMILY,
        DSA_ATTENTION_FAMILY,
    )
    assert tuple(iter_execution_enabled_families()) == (
        DENSE_ATTENTION_FAMILY,
        LATENT_MLA_ATTENTION_FAMILY,
    )


def _base_model_config() -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=2,
        num_q_heads=32,
        num_kv_heads=8,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        model_type="unit_dense_model",
    )


def test_build_operator_manifest_returns_wave_b_dense_families() -> None:
    manifest = build_operator_manifest(_base_model_config())

    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "ffn",
    ]
    assert [family.family_id for family in manifest.families()] == [
        "dense_attention",
        "memory",
        "ffn",
    ]
    assert [operator.name for operator in manifest.operators()] == [
        "attn_kv_cache_save",
        "attn_prefill",
        "attn_decode",
        "input_layernorm",
        "post_attention_layernorm",
        "add_attn_residual",
        "add_ffn_residual",
        "emb",
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
    ]


def test_build_operator_manifest_validates_architecture_structural_requirements() -> None:
    with pytest.raises(ValueError, match="Step3Text MFA.*use_mfa=True"):
        build_operator_manifest(
            SimpleNamespace(
                model_type="step3_text",
                model_arch="generic",
                model_architecture_profile=None,
                num_q_heads=8,
                num_kv_heads=1,
                embedding_dim=128,
                head_dim=16,
                is_moe=True,
                num_experts=16,
                share_expert_dim=64,
                use_mla=False,
                use_mfa=False,
                supports_share_expert=lambda: True,
            )
        )


def test_global_operator_registry_assigns_resource_class_to_every_operator() -> None:
    invalid_resource_class = [
        (family.family_id, operator.name, operator.resource_class)
        for family in OPERATOR_REGISTRY.iter_families()
        for operator in family.operators
        if not isinstance(operator.resource_class, ResourceClass)
    ]

    assert invalid_resource_class == []


def test_global_operator_registry_resource_class_matches_ratified_assignments() -> None:
    resource_by_operator = {
        (family.family_id, operator.name): operator.resource_class
        for family in OPERATOR_REGISTRY.iter_families()
        for operator in family.operators
    }

    assert resource_by_operator[("dense_attention", "attn_kv_cache_save")] is ResourceClass.MEMORY
    assert resource_by_operator[("dense_attention", "attn_prefill")] is ResourceClass.COMP
    assert resource_by_operator[("dense_attention", "attn_decode")] is ResourceClass.COMP
    assert resource_by_operator[("latent_mla_attention", "attn_mla_kv_cache_save")] is ResourceClass.MEMORY
    assert resource_by_operator[("memory", "input_layernorm")] is ResourceClass.MEMORY
    assert resource_by_operator[("memory", "add_attn_residual")] is ResourceClass.MEMORY
    assert resource_by_operator[("ffn", "mlp_up_proj")] is ResourceClass.COMP
    assert resource_by_operator[("moe", "moe_gating_routing_topk")] is ResourceClass.MEMORY
    assert resource_by_operator[("comm", "expert_parallel_allreduce")] is ResourceClass.COMM
