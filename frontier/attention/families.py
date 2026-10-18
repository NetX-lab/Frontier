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
from frontier.operators.spec import ResourceClass, TensorParallelMode


_PREFILL_MIXED = (AttentionPhase.PREFILL, AttentionPhase.MIXED)
_DECODE_MIXED = (AttentionPhase.DECODE, AttentionPhase.MIXED)
_ALL_PHASES = (AttentionPhase.PREFILL, AttentionPhase.DECODE, AttentionPhase.MIXED)
_GDN_PHASES = (AttentionPhase.PREFILL, AttentionPhase.DECODE)


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


def _dsa_kv_cache_capacity(config, block_size: int):
    from frontier.attention.memory import (
        AttentionKVCacheCapacity,
        AttentionKVCacheComponent,
    )
    from frontier.config.precision_type import PrecisionType

    if block_size != 64:
        raise NotImplementedError(
            "DSA indexer capacity currently requires the CUDA kernel block size 64"
        )
    dimensions = {}
    for name in ("kv_lora_rank", "qk_rope_head_dim", "index_head_dim"):
        value = getattr(config, name, None)
        if type(value) is not int or value <= 0:
            raise ValueError(f"DSA cache capacity requires positive int {name}")
        dimensions[name] = value

    # Native Indexer packs FP8 K and one FP32 scale per 128 values.
    quant_block_size = 128
    index_head_dim = dimensions["index_head_dim"]
    if index_head_dim % quant_block_size:
        raise NotImplementedError(
            "DSA indexer capacity requires complete 128-element quantization blocks"
        )
    specs = tuple(config.get_layer_attention_specs())
    if tuple(spec.global_layer_id for spec in specs) != tuple(range(config.num_layers)):
        raise ValueError("DSA cache capacity requires every backbone layer in order")
    full_index_layers = []
    producer = None
    for spec in specs:
        if spec.family_id != "dsa_attention":
            raise ValueError("DSA cache capacity requires a DSA backbone")
        if spec.variant_id == "full_index":
            producer = spec.global_layer_id
            full_index_layers.append(producer)
        elif spec.variant_id != "shared_index":
            raise ValueError("DSA cache capacity requires full/shared index bindings")
        if producer is None or spec.index_producer_layer_id != producer:
            raise ValueError("DSA cache capacity has an invalid index producer binding")

    cache_precision = PrecisionType.from_torch_dtype(config.torch_dtype)
    main_width = dimensions["kv_lora_rank"] + dimensions["qk_rope_head_dim"]
    main_page_bytes = int(block_size * main_width * cache_precision.bytes_per_element)
    index_bytes_per_token = index_head_dim + (
        index_head_dim // quant_block_size * int(PrecisionType.FP32.bytes_per_element)
    )
    return AttentionKVCacheCapacity(
        family_id="dsa_attention",
        block_size=block_size,
        num_layers=config.num_layers,
        components=(
            AttentionKVCacheComponent(
                component_id="main_mla",
                owner_layer_ids=tuple(spec.global_layer_id for spec in specs),
                page_size_bytes=main_page_bytes,
            ),
            AttentionKVCacheComponent(
                component_id="indexer_k_and_scale",
                owner_layer_ids=tuple(full_index_layers),
                page_size_bytes=block_size * index_bytes_per_token,
            ),
        ),
    )


# vLLM's P2P NCCL connector gathers a request's KV blocks after each layer's
# attention (P2pNcclConnector.save_kv_layer: layer[block_ids]) in the PDD prefill
# step that completes its prompt. Only runs with that connector price it, and
# attention tables profiled before it carry no column for it, so it is not a
# profiling, predictor or E2E trace target of the family.
DENSE_ATTENTION_KV_CACHE_EXTRACT = AttentionOperatorSpec(
    name="attn_kv_cache_extract",
    role=AttentionOperatorRole.CACHE_EXTRACT,
    phases=(AttentionPhase.PREFILL,),
    predictor_target=False,
    profiling_target=False,
    e2e_trace_target=False,
    execution_time_attr="attention_kv_cache_extract_execution_time",
    resource_class=ResourceClass.MEMORY,
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
        DENSE_ATTENTION_KV_CACHE_EXTRACT,
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
    display_name="DSA Attention",
    supported_variants=("dsa", "full_index", "shared_index"),
    operators=(
        AttentionOperatorSpec(
            name="attn_dsa_indexer",
            role=AttentionOperatorRole.INDEXING,
            phases=_ALL_PHASES,
            execution_time_attr="attn_dsa_indexer_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="attn_dsa_mla",
            role=AttentionOperatorRole.ATTENTION,
            phases=_ALL_PHASES,
            execution_time_attr="attn_dsa_mla_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
    ),
    memory_layout=AttentionMemoryLayout.FROZEN_DSA,
    dense_compatible=False,
    requires_runtime_kv_helpers=True,
    dsa_frozen=True,
    structural_execution_enabled=True,
    execution_enabled=False,
    disabled_reason=(
        "DSA attention is frozen until a real vLLM/FlashInfer truth backend "
        "and Frontier mapping are approved."
    ),
    kv_cache_capacity_resolver=_dsa_kv_cache_capacity,
    disjoint_model_projection_attrs=(
        "attention_layer_pre_proj_execution_time",
        "attention_layer_post_proj_execution_time",
    ),
)


GATED_DELTA_NET_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="gated_delta_net",
    display_name="Gated Delta Network",
    supported_variants=("qwen3_5",),
    operators=(
        AttentionOperatorSpec(
            name="gdn_input_projections",
            role=AttentionOperatorRole.PROJECTION,
            phases=_GDN_PHASES,
            execution_time_attr="attention_layer_pre_proj_execution_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
        ),
        AttentionOperatorSpec(
            name="gdn_core_prefill",
            role=AttentionOperatorRole.PREFILL_KERNEL,
            phases=(AttentionPhase.PREFILL,),
            execution_time_attr="attention_prefill_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        AttentionOperatorSpec(
            name="gdn_core_decode",
            role=AttentionOperatorRole.DECODE_KERNEL,
            phases=(AttentionPhase.DECODE,),
            execution_time_attr="attention_decode_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        AttentionOperatorSpec(
            name="gdn_output_projection",
            role=AttentionOperatorRole.PROJECTION,
            phases=_GDN_PHASES,
            execution_time_attr="attention_layer_post_proj_execution_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
        ),
    ),
    memory_layout=AttentionMemoryLayout.FIXED_STATE,
    dense_compatible=False,
    requires_runtime_kv_helpers=False,
    kv_factor=None,
    profiling_order=(
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_core_decode",
        "gdn_output_projection",
    ),
    required_profiling_feature_columns=(
        "measurement_type",
        "model_architecture_profile",
        "quant_signature",
        "device",
        "runtime_stack_signature",
        "gdn_runtime_backend",
        "gdn_rank_aggregation",
        "gdn_prefill_backend",
        "gdn_decode_backend",
        "gqa_interleaved_layout",
        "packed_recurrent_decode",
        "model_dtype",
        "conv_state_dtype",
        "recurrent_state_dtype",
        "num_tensor_parallel_workers",
        "hidden_size",
        "conv_kernel_size",
        "key_head_dim",
        "value_head_dim",
        "num_key_heads",
        "num_value_heads",
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_query_len",
        "has_initial_state",
    ),
)

STEP5_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="step5_attention",
    display_name="Step text SWA and sparse GQA",
    supported_variants=("swa", "sparse_gqa"),
    operators=(
        AttentionOperatorSpec(
            name="step5_qkv_projection", role=AttentionOperatorRole.PROJECTION,
            phases=_ALL_PHASES, execution_time_attr="step5_qkv_projection_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="step5_attention_prepare", role=AttentionOperatorRole.NORMALIZATION,
            phases=_ALL_PHASES, execution_time_attr="step5_attention_prepare_time",
            resource_class=ResourceClass.MEMORY, tp_mode=TensorParallelMode.ATTENTION_TP,
        ),
        AttentionOperatorSpec(
            name="step5_indexer_weights", role=AttentionOperatorRole.PROJECTION,
            phases=_ALL_PHASES, execution_time_attr="step5_indexer_weights_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="step5_indexer_prepare", role=AttentionOperatorRole.NORMALIZATION,
            phases=_ALL_PHASES, execution_time_attr="step5_indexer_prepare_time",
            resource_class=ResourceClass.MEMORY, tp_mode=TensorParallelMode.ATTENTION_TP,
        ),
        AttentionOperatorSpec(
            name="step5_sparse_attention", role=AttentionOperatorRole.ATTENTION,
            phases=_ALL_PHASES, execution_time_attr="step5_sparse_attention_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
        ),
        AttentionOperatorSpec(
            name="step5_swa_attention", role=AttentionOperatorRole.ATTENTION,
            phases=_ALL_PHASES, execution_time_attr="step5_swa_attention_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
        ),
        AttentionOperatorSpec(
            name="step5_gate_projection", role=AttentionOperatorRole.PROJECTION,
            phases=_ALL_PHASES, execution_time_attr="step5_gate_projection_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
        AttentionOperatorSpec(
            name="step5_headwise_gate", role=AttentionOperatorRole.ACTIVATION,
            phases=_ALL_PHASES, execution_time_attr="step5_headwise_gate_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
        ),
        AttentionOperatorSpec(
            name="step5_output_projection", role=AttentionOperatorRole.PROJECTION,
            phases=_ALL_PHASES, execution_time_attr="step5_output_projection_time",
            resource_class=ResourceClass.COMP, tp_mode=TensorParallelMode.ATTENTION_TP,
            projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
        ),
    ),
    memory_layout=AttentionMemoryLayout.DENSE_KV,
    dense_compatible=False,
    requires_runtime_kv_helpers=False,
    kv_factor=2,
    runtime_num_kv_heads_resolver=_dense_runtime_num_kv_heads,
    runtime_head_size_resolver=_dense_runtime_head_size,
    execution_enabled=False,
    disabled_reason=(
        "Step attention requires complete physical-scope timing, mixed cache allocation, "
        "per-layer precision and serving qualification before execution admission."
    ),
    required_profiling_feature_columns=(
        "measurement_type", "attention_backend", "model_architecture_profile",
        "attention_variant", "global_layer_id", "quant_signature", "model_dtype",
        "norm_dtype", "residual_dtype", "runtime_stack_signature", "device",
        "num_tensor_parallel_workers", "n_q_head", "n_kv_head", "head_size",
        "local_q_heads", "local_kv_heads", "kv_replication", "local_provider_groups",
        "provider_replication", "local_index_q_heads", "local_index_k_heads",
        "index_k_replication", "proxy_dim", "region_block_size", "topk_regions",
        "sliding_window", "rope_theta", "rotary_dim", "block_size", "cache_dtype",
        "batch_size", "batch_num_tokens", "batch_num_prefill_tokens", "batch_num_decode_tokens",
        "physical_scope", "fusion", "graph_mode", "valid_tokens", "padded_tokens",
    ),
)


DEEPSEEK_V4_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="deepseek_v4_attention",
    display_name="DeepSeek-V4 Compressed Attention",
    supported_variants=("uncompressed", "csa", "hca"),
    operators=(),
    memory_layout=AttentionMemoryLayout.COMPRESSED_KV,
    dense_compatible=False,
    requires_runtime_kv_helpers=True,
    execution_enabled=False,
    disabled_reason=(
        "DeepSeek-V4 execution requires compressed cache, compressor, indexer, "
        "mHC and mixed-precision operator coverage."
    ),
)


# Short aliases make the family discoverable without creating a second
# registry entry or a second semantic identity.
GDN_ATTENTION_FAMILY = GATED_DELTA_NET_ATTENTION_FAMILY
GATED_DELTA_NET_FAMILY = GATED_DELTA_NET_ATTENTION_FAMILY


_ATTENTION_OPERATOR_REGISTRY = OperatorRegistry()
for _family in (
    DENSE_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    GATED_DELTA_NET_ATTENTION_FAMILY,
    DEEPSEEK_V4_ATTENTION_FAMILY,
    STEP5_ATTENTION_FAMILY,
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


def iter_execution_enabled_families(
    *, include_structural: bool = False,
) -> tuple[AttentionFamilySpec, ...]:
    """Catalog families that participate in execution/profiling/training.

    Single source of truth for family execution state: catalog consumers
    iterate this instead of re-implementing per-family skip logic.
    """
    return tuple(
        cast(AttentionFamilySpec, family)
        for family in _ATTENTION_OPERATOR_REGISTRY.iter_families()
        if family.execution_enabled
        or (include_structural and family.structural_execution_enabled)
    )
