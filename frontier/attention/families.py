from __future__ import annotations

from typing import cast

from frontier.operators.registry import OperatorRegistry
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
    AttentionRuntimeMetaContract,
    ProjectionOwnership,
)
from frontier.operators.spec import ResourceClass


_PREFILL_MIXED = (AttentionPhase.PREFILL, AttentionPhase.MIXED)
_DECODE_MIXED = (AttentionPhase.DECODE, AttentionPhase.MIXED)
_ALL_PHASES = (AttentionPhase.PREFILL, AttentionPhase.DECODE, AttentionPhase.MIXED)


def _required_int_attr(config, attr_name: str) -> int:
    value = getattr(config, attr_name, None)
    if value is None:
        raise ValueError(f"Attention runtime resolver requires {attr_name}")
    return int(value)


def _dense_runtime_num_kv_heads(config) -> int:
    return _required_int_attr(config, "num_kv_heads")


def _dense_runtime_head_size(config) -> int:
    get_head_dim = getattr(config, "get_head_dim", None)
    if get_head_dim is None:
        raise ValueError("Dense runtime head size resolver requires get_head_dim()")
    return int(get_head_dim())


def _latent_mla_runtime_num_kv_heads(_config) -> int:
    return 1


def _latent_mla_runtime_head_size(config) -> int:
    return _required_int_attr(config, "kv_lora_rank") + _required_int_attr(
        config, "qk_rope_head_dim"
    )


DENSE_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="dense_attention",
    display_name="Dense-KV Attention",
    supported_variants=("gqa", "mha", "mqa"),
    operators=(
        AttentionOperatorSpec(
            name="attn_kv_cache_save",
            role=AttentionOperatorRole.CACHE_WRITE,
            phases=_ALL_PHASES,
            execution_time_attr="attention_kv_cache_save_execution_time",
            resource_class=ResourceClass.MEMORY,
        ),
        AttentionOperatorSpec(
            name="attn_prefill",
            role=AttentionOperatorRole.PREFILL_KERNEL,
            phases=_PREFILL_MIXED,
            execution_time_attr="attention_prefill_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        AttentionOperatorSpec(
            name="attn_decode",
            role=AttentionOperatorRole.DECODE_KERNEL,
            phases=_DECODE_MIXED,
            execution_time_attr="attention_decode_execution_time",
            resource_class=ResourceClass.COMP,
        ),
    ),
    memory_layout=AttentionMemoryLayout.DENSE_KV,
    dense_compatible=True,
    requires_runtime_kv_helpers=False,
    kv_factor=2,
    required_profiling_feature_columns=(
        "measurement_type",
        "attention_backend",
        "n_q_head",
        "n_kv_head",
        "block_size",
        "num_tensor_parallel_workers",
        "max_model_len",
        "batch_size",
        "prefill_chunk_size",
        "kv_cache_size",
        "is_prefill",
    ),
    runtime_num_kv_heads_resolver=_dense_runtime_num_kv_heads,
    runtime_head_size_resolver=_dense_runtime_head_size,
)


LATENT_MLA_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="latent_mla_attention",
    display_name="Latent MLA Attention",
    supported_variants=("mla",),
    operators=(
        AttentionOperatorSpec(
            name="attn_mla_kv_cache_save",
            role=AttentionOperatorRole.CACHE_WRITE,
            phases=_ALL_PHASES,
            execution_time_attr="attn_mla_kv_cache_save_time",
            resource_class=ResourceClass.MEMORY,
        ),
        AttentionOperatorSpec(
            name="attn_mla_prefill_kv_up_proj",
            role=AttentionOperatorRole.PROJECTION,
            phases=_PREFILL_MIXED,
            execution_time_attr="attn_mla_prefill_kv_up_proj_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="attn_mla_prefill",
            role=AttentionOperatorRole.PREFILL_KERNEL,
            phases=_PREFILL_MIXED,
            execution_time_attr="attn_mla_prefill_time",
            resource_class=ResourceClass.COMP,
        ),
        AttentionOperatorSpec(
            name="attn_mla_decode_q_latent_proj",
            role=AttentionOperatorRole.PROJECTION,
            phases=_DECODE_MIXED,
            execution_time_attr="attn_mla_decode_q_latent_proj_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="attn_mla_decode",
            role=AttentionOperatorRole.DECODE_KERNEL,
            phases=_DECODE_MIXED,
            execution_time_attr="attn_mla_decode_time",
            resource_class=ResourceClass.COMP,
        ),
        AttentionOperatorSpec(
            name="attn_mla_v_up_proj",
            role=AttentionOperatorRole.PROJECTION,
            phases=_DECODE_MIXED,
            execution_time_attr="attn_mla_v_up_proj_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
    ),
    memory_layout=AttentionMemoryLayout.LATENT_MLA,
    dense_compatible=False,
    requires_runtime_kv_helpers=True,
    # vLLM keeps model pre/post projections outside self.mla_attn(...), while
    # the six attn_mla_* scopes live inside MLACommonImpl.
    disjoint_model_projection_attrs=(
        "attention_layer_pre_proj_execution_time",
        "attention_layer_post_proj_execution_time",
    ),
    kv_factor=1,
    required_profiling_feature_columns=(
        "measurement_type",
        "attention_backend",
        "n_q_head",
        "n_kv_head",
        "head_size",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "qk_head_dim",
        "kv_lora_rank",
        "v_head_dim",
        "block_size",
        "num_tensor_parallel_workers",
        "max_model_len",
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_seqlen_q",
        "max_seqlen_k",
        "num_actual_tokens",
        "is_prefill",
        "max_seq_len",
        "is_mla_profile_import",
    ),
    imported_predictor_excluded_feature_columns=(
        "measurement_type",
        "attention_backend",
        "max_model_len",
        "is_mla_profile_import",
    ),
    runtime_num_kv_heads_resolver=_latent_mla_runtime_num_kv_heads,
    runtime_head_size_resolver=_latent_mla_runtime_head_size,
    runtime_meta_contract=AttentionRuntimeMetaContract(
        expected_runtime_num_kv_heads=1,
        runtime_head_size_formula="kv_lora_rank + qk_rope_head_dim",
        supported_block_sizes=(32, 64),
        expected_n_q_head=128,
    ),
)


DSA_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="dsa_attention",
    display_name="Frozen DSA Attention",
    supported_variants=("dsa",),
    operators=(),
    memory_layout=AttentionMemoryLayout.FROZEN_DSA,
    dense_compatible=False,
    requires_runtime_kv_helpers=True,
    dsa_frozen=True,
    execution_enabled=False,
    disabled_reason=(
        "DSA attention is frozen until a real vLLM/FlashInfer truth backend "
        "and Frontier mapping are approved."
    ),
)


_ATTENTION_OPERATOR_REGISTRY = OperatorRegistry()
for _family in (
    DENSE_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
):
    _ATTENTION_OPERATOR_REGISTRY.register(_family)

_FAMILIES_BY_ID: dict[str, AttentionFamilySpec] = {
    family.family_id: cast(AttentionFamilySpec, family)
    for family in _ATTENTION_OPERATOR_REGISTRY.iter_families()
}


def get_attention_family(family_id: str) -> AttentionFamilySpec:
    try:
        return _FAMILIES_BY_ID[family_id]
    except KeyError as exc:
        raise ValueError(f"Unknown attention family: {family_id}") from exc


def iter_attention_families() -> tuple[AttentionFamilySpec, ...]:
    return tuple(
        cast(AttentionFamilySpec, family)
        for family in _ATTENTION_OPERATOR_REGISTRY.iter_families()
    )


def iter_execution_enabled_families() -> tuple[AttentionFamilySpec, ...]:
    """Catalog families that participate in execution/profiling/training.

    Single source of truth for family execution state: catalog consumers
    iterate this instead of re-implementing per-family skip logic.
    """
    return tuple(
        cast(AttentionFamilySpec, family)
        for family in _ATTENTION_OPERATOR_REGISTRY.iter_execution_enabled_families()
    )
