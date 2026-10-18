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


def _dsa_kv_cache_capacity(config, block_size: int, selection=None):
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


def _deepseek_v4_kv_cache_capacity(config, block_size: int, selection=None):
    """Pinned FlashMLA packed pages, independent of execution admission."""
    from frontier.attention.memory import (
        AttentionKVCacheCapacity,
        AttentionKVCacheComponent,
        AttentionKVCacheGroup,
        AttentionKVCacheSelection,
    )
    from frontier.types import DeviceSKUType

    if not isinstance(selection, AttentionKVCacheSelection):
        raise NotImplementedError("DeepSeek-V4 cache requires explicit runtime cache selection")
    if selection.device != str(DeviceSKUType.H800) or selection.gpu_platform != "cuda":
        raise NotImplementedError("DeepSeek-V4 grouped cache currently requires H800/SM90 CUDA")
    if selection.attention_backend != "FLASHMLA_SPARSE_DSV4":
        raise NotImplementedError("DeepSeek-V4 cache requires FLASHMLA_SPARSE_DSV4")
    # vLLM DeepseekV4SparseMLABackend publishes fp8 as an alias of this
    # UE8M0-paged format. No automatic or alternate backend selection is made.
    main_dtype = {"fp8": "fp8_ds_mla"}.get(selection.kv_cache_dtype, selection.kv_cache_dtype)
    if main_dtype != "fp8_ds_mla":
        raise NotImplementedError("DeepSeek-V4 cache requires explicit fp8/fp8_ds_mla main KV dtype")
    if selection.indexer_kv_cache_dtype != "fp8":
        raise NotImplementedError("DeepSeek-V4 H800 indexer cache requires fp8, not the SM100 mxfp4 path")
    if block_size != 256 or selection.decode_context_parallel_size != 1:
        raise NotImplementedError("DeepSeek-V4 FlashMLA cache requires block_size=256 and DCP1")

    specs = tuple(config.get_layer_attention_specs())
    if tuple(spec.global_layer_id for spec in specs) != tuple(range(config.num_layers)):
        raise ValueError("DeepSeek-V4 cache requires every backbone layer in order")
    if any(spec.family_id != "deepseek_v4_attention" for spec in specs):
        raise ValueError("DeepSeek-V4 cache requires compressed-attention layer bindings")
    if any(spec.variant_id not in {"uncompressed", "csa", "hca"} for spec in specs):
        raise ValueError("DeepSeek-V4 cache has an invalid compression schedule")
    csa = tuple(spec.global_layer_id for spec in specs if spec.variant_id == "csa")
    hca = tuple(spec.global_layer_id for spec in specs if spec.variant_id == "hca")
    all_layers = tuple(spec.global_layer_id for spec in specs)
    if not csa or not hca:
        raise NotImplementedError("DeepSeek-V4 packed layout requires both CSA and HCA owners")
    head_dim = config.get_head_dim()
    dimensions = {}
    for name in ("qk_rope_head_dim", "index_head_dim", "sliding_window"):
        value = getattr(config, name, None)
        if type(value) is not int or value <= 0:
            raise ValueError(f"DeepSeek-V4 cache requires positive int {name}")
        dimensions[name] = value
    rope_dim = dimensions["qk_rope_head_dim"]
    index_dim = dimensions["index_head_dim"]
    window = dimensions["sliding_window"]
    # This backend publishes a fixed 448B NoPE / 128B BF16 RoPE / 8B scale
    # format. Reject unsupported geometry rather than generalize its kernel.
    if head_dim != 512 or rope_dim != 64:
        raise NotImplementedError("DeepSeek-V4 FlashMLA requires head_dim=512 and rope_dim=64")
    if index_dim <= 0 or index_dim % 128 or window <= 0:
        raise ValueError("DeepSeek-V4 cache requires positive 128-aligned index_head_dim and sliding_window")
    if config.torch_dtype != "bfloat16":
        raise NotImplementedError("DeepSeek-V4 FlashMLA model dtype must be bfloat16")

    alignment = 576  # Pinned fp8_ds_mla backend page alignment.
    main_row_bytes = (head_dim - rope_dim) + 2 * rope_dim + 8
    index_row_bytes = index_dim + index_dim // 128 * 4
    components = []

    def component(component_id, owners, logical_block, ratio, row_bytes, precision):
        rows = logical_block // ratio
        content = rows * row_bytes
        value = AttentionKVCacheComponent(
            component_id=component_id, owner_layer_ids=owners,
            page_size_bytes=(content + alignment - 1) // alignment * alignment,
            stored_rows=rows, compression_ratio=ratio, precision=precision,
            content_size_bytes=content,
        )
        components.append(value)
        return tuple((component_id, layer_id) for layer_id in owners)

    full = (
        *component("main_csa", csa, block_size, 4, main_row_bytes, main_dtype),
        *component("main_hca", hca, block_size, 128, main_row_bytes, main_dtype),
        *component("indexer_csa", csa, block_size, 4, index_row_bytes, "fp8_fp32_scale"),
    )
    swa = component("swa", all_layers, 64, 1, main_row_bytes, main_dtype)
    csa_state = (
        *component("main_csa_state", csa, 4, 1, 2 * 2 * head_dim * 4, "float32"),
        *component("indexer_csa_state", csa, 4, 1, 2 * 2 * index_dim * 4, "float32"),
    )
    hca_state = component("main_hca_state", hca, 8, 1, 2 * head_dim * 4, "float32")

    # Native grouping minimizes tuple padding with a larger-size tie break.
    # Counts are source component multiplicities, not padded/fake layer IDs.
    by_identity = {
        (item.component_id, owner): item
        for item in components for owner in item.owner_layer_ids
    }

    def tuples_by_page_size(owners):
        by_size = {}
        for owner in owners:
            size = by_identity[owner].page_size_bytes
            by_size.setdefault(size, []).append(owner)
        return tuple(tuple(value) for value in by_size.values())

    domains = (full, swa, csa_state, hca_state)
    tuple_counts = tuple(max(map(len, tuples_by_page_size(owners))) for owners in domains)
    lower_bound = tuple_counts[0]
    tuple_size = min(
        range(lower_bound, max(tuple_counts) + 1),
        key=lambda size: (sum((-count) % size for count in tuple_counts), -size),
    )
    groups = [AttentionKVCacheGroup("compressed", block_size, full)]
    for name, owners, logical_block, group_window in (
        ("swa", swa, 64, window),
        ("csa_state", csa_state, 4, 2 * 4),
        ("hca_state", hca_state, 8, 128),
    ):
        rows_by_size = tuples_by_page_size(owners)
        if len({len(row) for row in rows_by_size}) != 1:
            raise NotImplementedError("DeepSeek-V4 sliding tuple page classes must have equal owner counts")
        owner_tuples = tuple(zip(*rows_by_size))
        count = (len(owner_tuples) + tuple_size - 1) // tuple_size
        for index in range(count):
            slices = tuple(pair for row in owner_tuples[index::count] for pair in row)
            groups.append(AttentionKVCacheGroup(
                group_id=f"{name}_{index}", logical_block_size=logical_block,
                component_owners=slices, retention="sliding_window",
                window_tokens=group_window, extra_retained_tokens=0,
            ))
    return AttentionKVCacheCapacity(
        family_id="deepseek_v4_attention", block_size=block_size,
        num_layers=config.num_layers, components=tuple(components),
        groups=tuple(groups), reserved_blocks=1, zero_new_blocks=True,
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

def _deepseek_v4_attention_scopes(
    scopes: tuple[
        tuple[str, AttentionOperatorRole, TensorParallelMode, ResourceClass], ...
    ],
) -> tuple[AttentionOperatorSpec, ...]:
    return tuple(
        AttentionOperatorSpec(
            name=name,
            role=role,
            phases=_ALL_PHASES,
            predictor_target=False,
            profiling_target=False,
            e2e_trace_target=False,
            resource_class=resource,
            tp_mode=tp_mode,
            projection_ownership=(
                ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
                if role is AttentionOperatorRole.PROJECTION
                else ProjectionOwnership.NOT_PROJECTION
            ),
        )
        for name, role, tp_mode, resource in scopes
    )


_DEEPSEEK_V4_COMMON_OPERATORS = _deepseek_v4_attention_scopes((
    ("dsv4_fused_qkv_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_fused_qkv_rmsnorm", AttentionOperatorRole.NORMALIZATION,
     TensorParallelMode.REPLICATED, ResourceClass.MEMORY),
    ("dsv4_q_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.ATTENTION_TP, ResourceClass.COMP),
    # One native call owns Q norm/RoPE/padding and KV RoPE/quant/insert.
    ("dsv4_fused_q_kv_prepare", AttentionOperatorRole.CACHE_WRITE,
     TensorParallelMode.ATTENTION_TP, ResourceClass.MEMORY),
    ("dsv4_sparse_attention", AttentionOperatorRole.ATTENTION,
     TensorParallelMode.ATTENTION_TP, ResourceClass.COMP),
    ("dsv4_output_inv_rope_quant", AttentionOperatorRole.POSITION_ENCODING,
     TensorParallelMode.ATTENTION_TP, ResourceClass.MEMORY),
    ("dsv4_grouped_output_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.ATTENTION_TP, ResourceClass.COMP),
    ("dsv4_output_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.ATTENTION_TP, ResourceClass.COMP),
))


_DEEPSEEK_V4_COMPRESSOR_OPERATORS = _deepseek_v4_attention_scopes((
    ("dsv4_compressor_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_compressor_save_partial_states", AttentionOperatorRole.CACHE_WRITE,
     TensorParallelMode.REPLICATED, ResourceClass.MEMORY),
    ("dsv4_compressor_fused_compress_store", AttentionOperatorRole.CACHE_WRITE,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
))


_DEEPSEEK_V4_INDEXER_OPERATORS = _deepseek_v4_attention_scopes((
    ("dsv4_index_weights_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_compressor_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_compressor_save_partial_states", AttentionOperatorRole.CACHE_WRITE,
     TensorParallelMode.REPLICATED, ResourceClass.MEMORY),
    ("dsv4_index_compressor_fused_compress_store", AttentionOperatorRole.CACHE_WRITE,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_query_projection", AttentionOperatorRole.PROJECTION,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_fused_query_prepare", AttentionOperatorRole.POSITION_ENCODING,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_scores", AttentionOperatorRole.INDEXING,
     TensorParallelMode.REPLICATED, ResourceClass.COMP),
    ("dsv4_index_topk", AttentionOperatorRole.INDEXING,
     TensorParallelMode.REPLICATED, ResourceClass.MEMORY),
    ("dsv4_index_short_context_fill", AttentionOperatorRole.INDEXING,
     TensorParallelMode.REPLICATED, ResourceClass.MEMORY),
))


DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT = {
    "uncompressed": _DEEPSEEK_V4_COMMON_OPERATORS,
    "hca": (*_DEEPSEEK_V4_COMMON_OPERATORS, *_DEEPSEEK_V4_COMPRESSOR_OPERATORS),
    "csa": (*_DEEPSEEK_V4_COMMON_OPERATORS, *_DEEPSEEK_V4_COMPRESSOR_OPERATORS,
            *_DEEPSEEK_V4_INDEXER_OPERATORS),
}


DEEPSEEK_V4_ATTENTION_FAMILY = AttentionFamilySpec(
    family_id="deepseek_v4_attention",
    display_name="DeepSeek-V4 Compressed Attention",
    supported_variants=("uncompressed", "csa", "hca"),
    # Reachable physical scopes in the pinned CUDA path, not an execution
    # schedule. CSA short-context fill replaces scoring/top-k; HCA compression
    # and store can skip a non-boundary decode step after projection/state save.
    operators=DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT["csa"],
    memory_layout=AttentionMemoryLayout.COMPRESSED_KV,
    dense_compatible=False,
    requires_runtime_kv_helpers=True,
    execution_enabled=False,
    kv_cache_capacity_resolver=_deepseek_v4_kv_cache_capacity,
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
