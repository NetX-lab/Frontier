"""Step text attention geometry derived from configuration, without native imports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

from frontier.attention.model_binding import resolve_attention_head_partition
from frontier.config.precision_type import PrecisionType

if TYPE_CHECKING:
    from frontier.attention.gdn.config import LayerAttentionSpec
    from frontier.model_architectures import ResolvedLayerContract


STEP5_ATTENTION_VARIANTS = {
    "sliding_attention": "swa",
    "full_attention": "sparse_gqa",
}
# The pinned native sparse kernels and side storage support this geometry.
STEP5_SPARSE_DIMENSIONS = {
    "proxy_dim": 256,
    "region_block_size": 8,
    "sparse_indexer_num_k_heads": 1,
}
STEP5_PACKED_PROJECTIONS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "qkvg_proj": ("q_proj", "k_proj", "v_proj", "g_proj"),
    "qkv_indexer_proj": ("q_proj", "k_proj", "v_proj", "sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z", "g_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

# Native Step5ForCausalLM applies these renames to exclusion prefixes once.
STEP5_MODULE_PREFIX_RENAMES = ((".share_expert.", ".moe.share_expert."),)

STEP5_SPARSE_OPTIONS = {
    "enabled": True,
    "sparse_indexer_use_rope": True,
    "sparse_indexer_q_norm_type": "rmsnorm",
    "sparse_indexer_k_norm_type": "layernorm",
    "sparse_indexer_csa_z_norm_type": "none",
    "compression_method": "csa_block_compress",
    "attention_impl": "sparse_gqa",
}


def describe_step5_stage_operator_tensors(
    config: Any, operator_name: str, *, physical_rows: int, attention_tp_size: int,
) -> dict[str, Any]:
    """Describe once-only model work from the ordinary text forward source."""
    rows = _positive_int(physical_rows, "physical_rows")
    tp = _positive_int(attention_tp_size, "attention_tp_size")
    hidden = _positive_int(config.embedding_dim, "embedding_dim")
    precision = PrecisionType.from_torch_dtype(config.torch_dtype)
    residual = PrecisionType.FP32 if config.fp32_residual_connection else precision
    if operator_name == "emb":
        vocabulary = _positive_int(config.vocab_size, "vocab_size")
        if vocabulary % tp:
            raise ValueError("Step stage embedding requires even vocabulary shards")
        tensors = {"input_ids": [rows], "weight": [vocabulary // tp, hidden],
                   "output": [rows, hidden]}
        dtypes = {"input_ids": "INT32", "weight": precision.name, "output": precision.name}
    elif operator_name == "input_cast":
        tensors = {"input": [rows, hidden], "output": [rows, hidden]}
        dtypes = {"input": precision.name, "output": residual.name}
    elif operator_name == "final_residual":
        tensors = {"input": [rows, hidden], "residual": [rows, hidden], "output": [rows, hidden]}
        dtypes = {"input": precision.name, "residual": residual.name, "output": residual.name}
    elif operator_name == "embedding_tensor_parallel_allreduce":
        elements = rows * hidden
        return {
            "precision_op": "emb", "dtype": precision.name,
            "dtype_bytes": precision.bytes_per_element,
            "tensor_shape": {"data": [rows, hidden]}, "tensor_dtype": {"data": precision.name},
            "element_count": elements, "base_size_bytes": elements * precision.bytes_per_element,
            "data_size_bytes": int(elements * precision.bytes_per_element),
            "evidence_kind": "source_structural_metadata", "scope": "stage",
        }
    else:
        raise ValueError(f"Unknown Step stage operator {operator_name}")
    element_bytes = {precision.name: precision.bytes_per_element,
                     residual.name: residual.bytes_per_element, "INT32": 4}
    return {
        "tensor_shape": tensors, "tensor_dtype": dtypes,
        "tensor_size_bytes": {
            name: int(math.prod(shape) * element_bytes[dtypes[name]])
            for name, shape in tensors.items()
        },
        "precision_op": operator_name, "dtype": precision.name,
        "dtype_bytes": precision.bytes_per_element,
        "evidence_kind": "source_structural_metadata", "scope": "stage",
    }


def describe_step5_sampler_scope(config: Any, *, actual_requests: int, attention_tp_size: int):
    """Preserve logits and sampling inside the existing inclusive time owner."""
    requests = _positive_int(actual_requests, "actual_requests")
    tp = _positive_int(attention_tp_size, "attention_tp_size")
    precision = PrecisionType.from_torch_dtype(config.torch_dtype)
    covered = ["select_hidden_states"]
    if config.fp32_residual_connection:
        covered.append("logits_input_cast")
    covered += ["final_norm", "lm_head_linear"]
    if tp > 1:
        covered.append("logits_gather")
    covered.append("sampling")
    return {
        "scope": "stage", "evidence_kind": "source_structural_metadata",
        "time_ownership": "inclusive_sampler_e2e", "covered_scopes": covered,
        "logits_rows": requests, "model_precision": precision.name,
        "norm_weight_precision": PrecisionType.from_torch_dtype(config.norm_dtype).name,
        "tensor_shape": {"selected_hidden": [requests, config.embedding_dim],
                         "local_logits": [requests, config.vocab_size // tp]},
    }


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Step {name} must be a positive int")
    return value


def _number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Step {name} must be finite numeric metadata")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Step {name} must be {'non-negative' if allow_zero else 'positive'}")
    return float(value)


def step5_fused_add_norm_capability(embedding_dim: int, fp32_residual_connection: bool) -> bool:
    """Describe native CUDA eligibility; actual dispatch still needs observation."""
    _positive_int(embedding_dim, "embedding_dim")
    if type(fp32_residual_connection) is not bool:
        raise ValueError("Step fp32_residual_connection must be bool")
    # supports_rms_norm accepts aligned widths up to 16384 with FP32 residuals.
    return fp32_residual_connection and embedding_dim <= 16384 and embedding_dim % 32 == 0


def resolve_step5_model_precision(config: Any) -> PrecisionType:
    """Read the existing runtime or profiling dtype contract without torch imports."""
    getter = getattr(config, "get_default_precision", None)
    if callable(getter):
        return getter()
    return PrecisionType.from_torch_dtype(str(config.dtype))


def validate_step5_config(config: Any) -> None:
    """Reject incomplete text metadata before it can select a generic path."""
    if getattr(config, "model_type", None) != "step5":
        raise ValueError("Step profile requires model_type='step5'")
    if getattr(config, "att_impl_type", None) != "GQA":
        raise ValueError("Step profile requires att_impl_type='GQA'")
    if getattr(config, "use_mla", False) or getattr(config, "use_mfa", False):
        raise ValueError("Step sparse GQA cannot use MLA or MFA bindings")
    if resolve_step5_model_precision(config) is not PrecisionType.BF16:
        raise NotImplementedError("Step native sparse attention requires BF16 model dtype")
    if config.use_qkv_bias or config.use_bias:
        raise NotImplementedError("Step native attention and FFN require bias-free projections")
    for name in ("num_layers", "num_q_heads", "num_kv_heads", "embedding_dim", "sliding_window"):
        _positive_int(getattr(config, name, None), name)
    head_dim = getattr(config, "head_dim", getattr(config, "_head_dim", None))
    _positive_int(head_dim, "explicit head_dim")
    if not config.is_moe or not config.attn_output_gate or not config.use_qk_norm:
        raise ValueError("Step requires MoE, headwise attention gate and QK normalization")
    for name in ("need_fp32_gate", "use_moe_router_bias"):
        if getattr(config, name, None) is not True:
            raise ValueError(f"Step requires {name}=true")
    for name in ("fp32_residual_connection", "zero_centered", "tie_word_embeddings", "norm_expert_weight"):
        if type(getattr(config, name, None)) is not bool:
            raise ValueError(f"Step {name} must be bool")
    if not config.zero_centered:
        raise NotImplementedError("Step native RMSNorm requires zero_centered=true")
    if config.uses_fused_add_norm is not step5_fused_add_norm_capability(
        config.embedding_dim, config.fp32_residual_connection,
    ):
        raise ValueError("Step uses_fused_add_norm disagrees with the native residual/norm path")
    if getattr(config, "moe_router_activation", None) != "sigmoid":
        raise NotImplementedError("Step native router requires sigmoid scores")
    _number(getattr(config, "moe_router_scaling_factor", None), "moe_router_scaling_factor")
    _number(getattr(config, "rms_norm_eps", None), "rms_norm_eps")
    if getattr(config, "norm_dtype", None) not in ("float32", "fp32", "bfloat16", "bf16"):
        raise ValueError("Step norm_dtype must name FP32 or BF16")
    num_layers = config.num_layers
    schedule = getattr(config, "layer_types", None)
    if not isinstance(schedule, (tuple, list)) or len(schedule) != num_layers:
        raise ValueError("Step layer_types must cover exactly the backbone layers")
    if any(layer_type not in STEP5_ATTENTION_VARIANTS for layer_type in schedule):
        raise ValueError("Step layer_types supports only sliding_attention/full_attention")
    if not any(layer == "full_attention" for layer in schedule) or not any(
        layer == "sliding_attention" for layer in schedule
    ):
        raise ValueError("Step representation requires both SWA and sparse GQA layers")
    mtp_count = getattr(config, "num_nextn_predict_layers", None)
    if type(mtp_count) is not int or mtp_count < 0:
        raise ValueError("Step num_nextn_predict_layers must be a non-negative int")
    mtp_types = getattr(config, "mtp_layer_types", None)
    if not isinstance(mtp_types, (tuple, list)) or len(mtp_types) != mtp_count:
        raise ValueError("Step mtp_layer_types must retain the declared MTP inventory")
    if any(value not in STEP5_ATTENTION_VARIANTS for value in mtp_types):
        raise ValueError("Step MTP inventory contains an unsupported layer type")
    total_layers = num_layers + mtp_count
    for name in ("layer_rope_thetas", "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared"):
        values = getattr(config, name, None)
        if not isinstance(values, (tuple, list)) or len(values) != total_layers:
            raise ValueError(f"Step {name} must cover the backbone and declared MTP layers")
        for value in values:
            _number(value, name, allow_zero=name.startswith("swiglu"))
    use_rope = getattr(config, "use_rope_layers", None)
    if not isinstance(use_rope, (tuple, list)) or len(use_rope) != total_layers or any(
        type(value) is not bool for value in use_rope
    ):
        raise ValueError("Step use_rope_layers must contain one bool per backbone and MTP layer")
    yarn_types = getattr(config, "yarn_only_types", None)
    if not isinstance(yarn_types, (tuple, list)) or any(
        value not in STEP5_ATTENTION_VARIANTS for value in yarn_types
    ):
        raise ValueError("Step yarn_only_types must contain supported attention types")
    other = getattr(config, "attention_other_setting", None)
    if other is not None:
        if not isinstance(other, dict) or other.get("attention_type") not in STEP5_ATTENTION_VARIANTS:
            raise ValueError("Step attention_other_setting requires a supported attention_type")
        for name in ("num_attention_heads", "num_attention_groups", "head_dim"):
            _positive_int(other.get(name), f"attention_other_setting.{name}")
    sparse = getattr(config, "sparse_config", None)
    if not isinstance(sparse, dict):
        raise ValueError("Step requires sparse_config; dense-only configurations are unsupported")
    for name, expected in {**STEP5_SPARSE_DIMENSIONS, **STEP5_SPARSE_OPTIONS}.items():
        value = sparse.get(name)
        if value != expected or type(value) is not type(expected):
            raise NotImplementedError(f"Step native sparse kernels require {name}={expected!r}")
    providers = _positive_int(sparse.get("num_provider_groups"), "num_provider_groups")
    if "index_tp_size" in sparse and sparse["index_tp_size"] != providers:
        raise ValueError("Step index_tp_size and num_provider_groups disagree")
    index_heads = _positive_int(sparse.get("sparse_indexer_num_heads"), "sparse_indexer_num_heads")
    if index_heads % providers or index_heads // providers not in (2, 4):
        raise ValueError("Step indexer requires 2 or 4 Q heads per provider group")
    topk = _positive_int(sparse.get("topk"), "topk regions")
    if topk > 1024:
        raise NotImplementedError("Step native selector supports at most 1024 historical regions")
    rope_dim = _positive_int(sparse.get("sparse_indexer_rope_dim"), "sparse_indexer_rope_dim")
    if rope_dim > sparse["proxy_dim"] or rope_dim % 2:
        raise ValueError("Step indexer rotary dimension must be even and no larger than proxy_dim")
    if sparse.get("sparse_indexer_softmax_variant") not in ("softmax", "standard", "ssmax"):
        raise ValueError("Step sparse_indexer_softmax_variant is unsupported")
    if sparse.get("sparse_indexer_ssmax_s_granularity") != "q_head":
        raise ValueError("Step sparse_indexer_ssmax_s_granularity is unsupported")
    apply_to = sparse.get("apply_to_layer_types")
    if not isinstance(apply_to, (tuple, list)) or tuple(apply_to) != ("full_attention",):
        raise ValueError("Step sparse_config must apply to full_attention layers")
    # Check every layer rather than substituting the first layer's geometry.
    for layer_id in range(num_layers):
        resolve_step5_attention_geometry(config, layer_id, tensor_parallel_size=1)
    from frontier.model_architectures import parse_moe_layer_ids

    routed = parse_moe_layer_ids(config.moe_layers_enum, num_layers)
    _positive_int(config.num_experts, "num_experts")
    _positive_int(config.num_experts_per_tok, "num_experts_per_tok")
    if config.num_experts_per_tok > config.num_experts:
        raise ValueError("Step top-k experts cannot exceed num_experts")
    for name in ("dense_mlp_hidden_dim", "routed_mlp_hidden_dim", "share_expert_dim"):
        _positive_int(getattr(config, name, None), name)
    for layer_id in routed:
        if config.swiglu_limits[layer_id] not in (0, 7):
            raise NotImplementedError("Step native routed SwiGLU supports clamp 7 or an unclamped activation")


@dataclass(frozen=True)
class Step5AttentionGeometry:
    """Logical and per-rank attention/provider dimensions for one layer."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    tensor_parallel_size: int
    local_q_heads: int
    local_kv_heads: int
    kv_replication: int
    local_provider_groups: int = 0
    provider_replication: int = 0
    local_index_q_heads: int = 0
    local_index_k_heads: int = 0
    index_k_replication: int = 0
    proxy_dim: int = 0

    @property
    def q_width(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def local_qkv_width(self) -> int:
        return (self.local_q_heads + 2 * self.local_kv_heads) * self.head_dim

    @property
    def local_fused_indexer_width(self) -> int:
        if not self.local_index_q_heads:
            return 0
        # Native packs Q/K/Z plus one gate value per local main-attention Q head.
        return (
            self.local_index_q_heads * self.proxy_dim
            + 2 * self.local_index_k_heads * self.proxy_dim
            + self.local_q_heads
        )


def resolve_step5_attention_geometry(
    config: Any, global_layer_id: int, *, tensor_parallel_size: int,
) -> Step5AttentionGeometry:
    """Resolve native GQA/provider partitioning without importing the runtime."""
    if type(global_layer_id) is not int or not 0 <= global_layer_id < config.num_layers:
        raise ValueError("Step global_layer_id must be inside the backbone")
    tp_size = _positive_int(tensor_parallel_size, "tensor_parallel_size")
    layer_type = config.layer_types[global_layer_id]
    if layer_type not in STEP5_ATTENTION_VARIANTS:
        raise ValueError("Unsupported Step layer attention type")
    q_heads, kv_heads = config.num_q_heads, config.num_kv_heads
    head_dim = getattr(config, "head_dim", getattr(config, "_head_dim", None))
    other = config.attention_other_setting
    if other is not None and layer_type == other["attention_type"]:
        q_heads = other["num_attention_heads"]
        kv_heads = other["num_attention_groups"]
        head_dim = other["head_dim"]
    if layer_type == "sliding_attention" and config.swa_num_attention_heads is not None:
        q_heads = config.swa_num_attention_heads
    _positive_int(q_heads, "num_q_heads")
    _positive_int(head_dim, "explicit head_dim")
    if q_heads % tp_size:
        raise ValueError("Step Q heads must divide tensor_parallel_size")
    local_q = q_heads // tp_size
    local_kv, kv_replication = resolve_attention_head_partition(kv_heads, tp_size, name="Step KV groups")
    if local_q % local_kv:
        raise ValueError("Step local Q heads must divide into local KV groups")
    partial = config.partial_rotary_factors[global_layer_id]
    _number(partial, "partial_rotary_factor")
    rotary_dim = int(head_dim * partial)
    if partial > 1 or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("Step partial_rotary_factor must produce a positive even rotary dimension")
    kwargs: dict[str, int] = {}
    if layer_type == "full_attention":
        sparse = config.sparse_config
        providers = sparse["num_provider_groups"]
        if providers != kv_heads:
            raise ValueError("Step provider groups must match attention KV groups")
        if head_dim not in (128, 192):
            raise NotImplementedError("Step sparse GQA kernels require head_dim 128 or 192")
        if local_q // local_kv not in (4, 8, 16):
            raise ValueError("Step sparse GQA kernels require 4, 8 or 16 local Q heads per KV group")
        local_providers, provider_replication = resolve_attention_head_partition(providers, tp_size, name="Step provider groups")
        index_heads = sparse["sparse_indexer_num_heads"]
        if index_heads % providers:
            raise ValueError("Step index Q heads must divide provider groups")
        local_index_k, index_replication = resolve_attention_head_partition(sparse["sparse_indexer_num_k_heads"], tp_size, name="Step index KV heads")
        kwargs = {
            "local_provider_groups": local_providers,
            "provider_replication": provider_replication,
            "local_index_q_heads": index_heads // providers * local_providers,
            "local_index_k_heads": local_index_k,
            "index_k_replication": index_replication,
            "proxy_dim": sparse["proxy_dim"],
        }
    return Step5AttentionGeometry(
        q_heads, kv_heads, head_dim, tp_size, local_q, local_kv, kv_replication, **kwargs,
    )


@dataclass(frozen=True)
class Step5Projection:
    """One native module's logical and per-rank weight shapes.

    The operator is owned by the existing registry. Packed projections and
    the two expert matrices can share an operator without becoming separate
    timing targets. Precision names describe inputs, weights and outputs;
    they do not imply a particular backend or measured duration.
    """

    operator_name: str
    module_prefix: str
    quantization_prefix: str
    logical_weight_shape: tuple[int, ...]
    rank_weight_shape: tuple[int, ...]
    weight_precision: PrecisionType
    input_precision: PrecisionType
    output_precision: PrecisionType


@dataclass(frozen=True)
class Step5FFNMerge:
    """Precision and shape of the independently owned routed/shared branch sum."""

    operator_name: str
    module_prefix: str
    input_precisions: tuple[PrecisionType, PrecisionType]
    output_precision: PrecisionType
    hidden_dim: int


@dataclass(frozen=True)
class Step5LayerMetadata:
    """Source-derived shapes and precision for the model-owned layer binding."""

    attention_spec: LayerAttentionSpec
    geometry: Step5AttentionGeometry
    ffn_contracts: tuple[ResolvedLayerContract, ...]
    projections: tuple[Step5Projection, ...]
    ffn_merge: Step5FFNMerge | None
    model_precision: PrecisionType
    norm_precision: PrecisionType
    residual_precision: PrecisionType
    uses_fused_add_norm: bool
    indexer_norm_precision: PrecisionType | None
    indexer_query_precision: PrecisionType | None
    indexer_key_precision: PrecisionType | None
    indexer_weight_precision: PrecisionType | None
    rope_theta: float
    rotary_dim: int
    use_rope: bool
    routed_swiglu_limit: float | None
    dense_swiglu_limit: float
    router_precision: PrecisionType | None
    router_scoring: str | None
    router_bias: bool
    router_renormalize: bool
    router_scaling_factor: float | None
    fused_qkv_gate: bool
    input_norm_precision: PrecisionType

    def attention_scopes(
        self, workload: Step5AttentionWorkload,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return exclusive physical owners and their semantic work in order.

        Summary, selector, packing and staging stay inside the native attention
        scope, except a summary update covered by decode preparation. The
        selected scope is evidence about the retained source, not a measured
        kernel inventory or a claim of native graph replay.
        """
        if not workload.has_attention_metadata:
            return self.request_free_attention_scopes()
        sparse = self.attention_spec.variant_id == "sparse_gqa"
        qkv_work = ("q_projection", "k_projection", "v_projection")
        preparation = ("qk_norm", "main_rope", "raw_kv_write")
        if not self.use_rope:
            preparation = ("qk_norm", "raw_kv_write")
        if sparse:
            qkv_work += ("index_q_projection", "index_k_projection", "index_z_projection", "gate_projection")
            index_work = (
                "index_q_norm", "index_k_norm", "index_rope",
                "index_q_fp8_conversion", "index_weight_scaling",
            )
            fused_prepare = (
                self.geometry.head_dim == 192
                and workload.is_single_token_step
                and workload.graph_mode != "piecewise"
                and workload.physical_query_rows <= 128
            )
            # The separate indexer preparation can itself update decode CSA.
            summary_prepared = workload.is_single_token_step and workload.graph_mode != "piecewise"
            if summary_prepared:
                index_work += ("summary_update",)
            scopes = [("step5_qkv_projection", qkv_work)]
            if fused_prepare:
                # The weight GEMM is called before the fused preparation.
                scopes += [
                    ("step5_indexer_weights", ("index_weight_projection",)),
                    ("step5_attention_prepare", preparation + index_work),
                ]
            else:
                scopes += [
                    ("step5_attention_prepare", preparation),
                    ("step5_indexer_weights", ("index_weight_projection",)),
                    ("step5_indexer_prepare", index_work),
                ]
            attention_work = () if summary_prepared else ("summary_update",)
            if workload.score_regions:
                if not workload.is_single_token_step:
                    attention_work += ("summary_packing",)
                attention_work += ("selector_scoring", "historical_region_selection")
            attention_work += (
                "paged_kv_read" if workload.is_single_token_step else "kv_staging_or_gather",
                "sparse_gqa", "headwise_sigmoid_gate",
            )
            scopes.append(("step5_sparse_attention", attention_work))
        else:
            if self.fused_qkv_gate:
                qkv_work += ("gate_projection",)
            scopes = [
                ("step5_qkv_projection", qkv_work),
                ("step5_attention_prepare", preparation),
                ("step5_swa_attention", ("sliding_window_attention",)),
            ]
            # The H800/H200 Hopper path does not use SM103's fused SWA gate.
            if not self.fused_qkv_gate:
                scopes.append(("step5_gate_projection", ("gate_projection",)))
            scopes.append(("step5_headwise_gate", ("headwise_sigmoid_gate",)))
        scopes.append(("step5_output_projection", ("output_projection",)))
        return tuple(scopes)

    def request_free_attention_scopes(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the scopes of vLLM's attention-DP dummy forward.

        Without attention metadata the slot mapping is absent, so the source
        takes the unfused norm/RoPE path, writes no raw KV or summary, and the
        attention backends return zero-filled outputs (model.py
        Step5Attention.forward; sparse_attention.py forward;
        attention.py unified_kv_cache_update). Projections and the SWA gate
        still run.
        """
        qkv_work = ("q_projection", "k_projection", "v_projection")
        preparation = ("qk_norm", "main_rope") if self.use_rope else ("qk_norm",)
        if self.attention_spec.variant_id == "sparse_gqa":
            qkv_work += ("index_q_projection", "index_k_projection", "index_z_projection", "gate_projection")
            scopes = [
                ("step5_qkv_projection", qkv_work),
                ("step5_attention_prepare", preparation),
                ("step5_indexer_weights", ("index_weight_projection",)),
                ("step5_indexer_prepare", (
                    "index_q_norm", "index_k_norm", "index_rope",
                    "index_q_fp8_conversion", "index_weight_scaling",
                )),
                ("step5_sparse_attention", ("output_zero_fill",)),
            ]
        else:
            if self.fused_qkv_gate:
                qkv_work += ("gate_projection",)
            scopes = [
                ("step5_qkv_projection", qkv_work),
                ("step5_attention_prepare", preparation),
                ("step5_swa_attention", ("output_zero_fill",)),
            ]
            if not self.fused_qkv_gate:
                scopes.append(("step5_gate_projection", ("gate_projection",)))
            scopes.append(("step5_headwise_gate", ("headwise_sigmoid_gate",)))
        scopes.append(("step5_output_projection", ("output_projection",)))
        return tuple(scopes)


def describe_step5_layer(
    config: Any, global_layer_id: int, *, attention_tp_size: int,
    moe_tp_size: int, expert_parallel_size: int,
) -> Step5LayerMetadata:
    """Resolve one backbone layer using the existing binding and FFN contracts.

    This describes the ordinary TP/EP path. MegaMoE sequence parallelism, MTP,
    multimodal execution and native serving qualification are outside it.
    """
    from frontier.attention.model_binding import bind_layer_attention
    from frontier.model_architectures import LayerKind
    from frontier.operators.binding import build_operator_manifest

    validate_step5_config(config)
    for name, value in (("moe_tp_size", moe_tp_size), ("expert_parallel_size", expert_parallel_size)):
        _positive_int(value, name)
    spec = bind_layer_attention(config, global_layer_id)
    expected_variant = STEP5_ATTENTION_VARIANTS[config.layer_types[global_layer_id]]
    if spec.family_id != "step5_attention" or spec.variant_id != expected_variant:
        raise ValueError("Step layer binding disagrees with the configured attention schedule")
    geometry = resolve_step5_attention_geometry(
        config, global_layer_id, tensor_parallel_size=attention_tp_size,
    )
    model_precision = resolve_step5_model_precision(config)
    quantization = config.quantization_config

    def module_precision(prefix: str, packed: str | None = None, *, experts: bool = False) -> PrecisionType:
        if quantization is None:
            return model_precision
        return quantization.resolve_module_precision(
            prefix, model_precision,
            packed_modules=STEP5_PACKED_PROJECTIONS[packed] if packed else (),
            routed_experts=experts, module_prefix_renames=STEP5_MODULE_PREFIX_RENAMES,
        )

    registered_operators = {
        operator.name for operator in build_operator_manifest(config).operators()
    }
    projections: list[Step5Projection] = []

    def projection(
        operator_name: str, prefix: str, logical: tuple[int, ...], rank: tuple[int, ...],
        precision: PrecisionType, *, output_precision: PrecisionType = model_precision,
        input_precision: PrecisionType | None = None, quantization_prefix: str | None = None,
    ) -> None:
        # The existing operator registry owns classification and name validation.
        if operator_name not in registered_operators:
            raise ValueError(f"Unknown Step projection operator {operator_name!r}")
        projections.append(Step5Projection(
            operator_name, prefix, quantization_prefix or prefix, logical, rank, precision,
            input_precision or model_precision, output_precision,
        ))

    attention_prefix = f"model.layers.{global_layer_id}.self_attn"
    sparse = spec.variant_id == "sparse_gqa"
    qkv_name = "qkv_indexer_proj" if sparse else "qkv_proj"
    qkv_precision = module_precision(f"{attention_prefix}.{qkv_name}", qkv_name)
    gate_precision = module_precision(f"{attention_prefix}.g_proj")
    fused_qkv_gate = (
        not sparse and geometry.num_kv_heads >= attention_tp_size
        and qkv_precision is model_precision and gate_precision is model_precision
    )
    logical_qkv = (geometry.num_q_heads + 2 * geometry.num_kv_heads) * geometry.head_dim
    rank_qkv = geometry.local_qkv_width
    if sparse:
        if qkv_precision is not PrecisionType.BF16:
            raise ValueError("Step packed sparse QKV/indexer/gate projection must be excluded from FP8")
        sparse_config = config.sparse_config
        logical_qkv += (
            (sparse_config["sparse_indexer_num_heads"] + 2 * sparse_config["sparse_indexer_num_k_heads"])
            * geometry.proxy_dim + geometry.num_q_heads
        )
        rank_qkv += geometry.local_fused_indexer_width
    elif fused_qkv_gate:
        logical_qkv += geometry.num_q_heads
        rank_qkv += geometry.local_q_heads
    projection(
        "step5_qkv_projection", f"{attention_prefix}.{'qkvg_proj' if fused_qkv_gate else qkv_name}",
        (logical_qkv, config.embedding_dim), (rank_qkv, config.embedding_dim), qkv_precision,
        quantization_prefix=f"{attention_prefix}.{qkv_name}",
    )
    if sparse:
        projection(
            "step5_indexer_weights", f"{attention_prefix}.sparse_indexer_w",
            (config.sparse_config["sparse_indexer_num_heads"], config.embedding_dim),
            (geometry.local_index_q_heads, config.embedding_dim), model_precision,
        )
    elif not fused_qkv_gate:
        projection(
            "step5_gate_projection", f"{attention_prefix}.g_proj",
            (geometry.num_q_heads, config.embedding_dim),
            (geometry.local_q_heads, config.embedding_dim), gate_precision,
        )
    projection(
        "step5_output_projection", f"{attention_prefix}.o_proj",
        (config.embedding_dim, geometry.q_width),
        (config.embedding_dim, geometry.local_q_heads * geometry.head_dim),
        module_precision(f"{attention_prefix}.o_proj"),
    )

    profile = config.get_model_architecture_profile()
    main_ffn = profile.resolve_layer_contract(
        config, layer_id=global_layer_id, attention_tp_size=attention_tp_size,
        moe_tp_size=moe_tp_size, expert_parallel_size=expert_parallel_size,
    )
    ffn_contracts = [main_ffn]
    if main_ffn.layer_kind is LayerKind.ROUTED:
        ffn_contracts.append(profile.resolve_layer_contract(
            config, layer_id=global_layer_id, layer_kind=LayerKind.SHARED,
            attention_tp_size=attention_tp_size, moe_tp_size=moe_tp_size,
            expert_parallel_size=expert_parallel_size,
        ))
    for contract in ffn_contracts:
        width = contract.width
        local_width = width // contract.tensor_parallel_size
        if contract.layer_kind is LayerKind.ROUTED:
            local_experts = config.num_experts // expert_parallel_size
            prefix = f"model.layers.{global_layer_id}.moe.experts"
            precision = module_precision(prefix, experts=True)
            for name, logical, rank in (
                ("w13_weight", (config.num_experts, 2 * width, config.embedding_dim),
                 (local_experts, 2 * local_width, config.embedding_dim)),
                ("w2_weight", (config.num_experts, config.embedding_dim, width),
                 (local_experts, config.embedding_dim, local_width)),
            ):
                projection("moe_grouped_gemm", f"{prefix}.{name}", logical, rank, precision)
            projection(
                "moe_gating_linear", f"model.layers.{global_layer_id}.moe.gate",
                (config.num_experts, config.embedding_dim), (config.num_experts, config.embedding_dim),
                PrecisionType.FP32, input_precision=model_precision, output_precision=PrecisionType.FP32,
            )
        else:
            shared = contract.layer_kind is LayerKind.SHARED
            prefix = f"model.layers.{global_layer_id}." + ("moe.share_expert" if shared else "mlp")
            for name, operator_name, logical, rank in (
                ("gate_up_proj", "share_expert_up_proj" if shared else "mlp_up_proj",
                 (2 * width, config.embedding_dim), (2 * local_width, config.embedding_dim)),
                ("down_proj", "share_expert_down_proj" if shared else "mlp_down_proj",
                 (config.embedding_dim, width), (config.embedding_dim, local_width)),
            ):
                projection(
                    operator_name, f"{prefix}.{name}", logical, rank,
                    module_precision(f"{prefix}.{name}", "gate_up_proj" if name == "gate_up_proj" else None),
                )
    routed = main_ffn.layer_kind is LayerKind.ROUTED
    merge = None
    if routed:
        if "step5_ffn_merge" not in registered_operators:
            raise ValueError("Step routed/shared FFN merge has no registered physical owner")
        merge = Step5FFNMerge(
            operator_name="step5_ffn_merge",
            module_prefix=f"model.layers.{global_layer_id}.moe",
            input_precisions=(model_precision, model_precision),
            output_precision=PrecisionType.FP32 if config.fp32_residual_connection else model_precision,
            hidden_dim=config.embedding_dim,
        )
    return Step5LayerMetadata(
        attention_spec=spec, geometry=geometry, ffn_contracts=tuple(ffn_contracts),
        projections=tuple(projections), ffn_merge=merge, model_precision=model_precision,
        norm_precision=PrecisionType.from_torch_dtype(config.norm_dtype),
        residual_precision=PrecisionType.FP32 if config.fp32_residual_connection else model_precision,
        uses_fused_add_norm=config.uses_fused_add_norm,
        indexer_norm_precision=model_precision if sparse else None,
        indexer_query_precision=PrecisionType.FP8 if sparse else None,
        indexer_key_precision=model_precision if sparse else None,
        indexer_weight_precision=PrecisionType.FP32 if sparse else None,
        rope_theta=float(config.layer_rope_thetas[global_layer_id]),
        rotary_dim=int(geometry.head_dim * config.partial_rotary_factors[global_layer_id]),
        use_rope=config.use_rope_layers[global_layer_id],
        routed_swiglu_limit=float(config.swiglu_limits[global_layer_id]) if routed else None,
        dense_swiglu_limit=float(config.swiglu_limits_shared[global_layer_id]),
        router_precision=PrecisionType.FP32 if routed else None,
        router_scoring=config.moe_router_activation if routed else None,
        router_bias=routed and config.use_moe_router_bias,
        router_renormalize=routed and config.norm_expert_weight,
        router_scaling_factor=float(config.moe_router_scaling_factor) if routed else None,
        fused_qkv_gate=fused_qkv_gate,
        input_norm_precision=(
            PrecisionType.FP32
            if config.fp32_residual_connection and (
                global_layer_id == 0 or global_layer_id - 1 in config.get_moe_layer_ids()
            )
            else model_precision
        ),
    )


@dataclass(frozen=True)
class Step5AttentionWorkload:
    """Physical token/region extents for cache-backed or request-free attention.

    Block-table width is an explicit physical extent. Decode scoring uses it
    independently of live lengths; neither padded query rows nor allocated
    history receives effective-token or state-progress credit.
    The request-free slot is one physical eager token without a block table,
    request-owned attention metadata, or scored history.
    """

    phase: str
    query_lengths: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    physical_query_rows: int
    block_size_tokens: int
    block_table_columns: int
    max_model_len_tokens: int
    region_tokens: int
    topk_regions: int
    graph_mode: str
    has_attention_metadata: bool = True

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode", "mixed"):
            raise ValueError("Step workload phase must be prefill, decode or mixed")
        if self.graph_mode not in ("none", "piecewise", "full"):
            raise ValueError("Step graph_mode must be none, piecewise or full")
        if type(self.has_attention_metadata) is not bool:
            raise ValueError("Step has_attention_metadata must be boolean")
        for name in (
            "physical_query_rows", "block_size_tokens",
            "max_model_len_tokens", "region_tokens", "topk_regions",
        ):
            _positive_int(getattr(self, name), name)
        if self.has_attention_metadata:
            _positive_int(self.block_table_columns, "block_table_columns")
        elif not (
            self.phase == "decode" and self.graph_mode == "none"
            and tuple(self.query_lengths) == tuple(self.sequence_lengths) == (1,)
            and self.physical_query_rows == 1
            and type(self.block_table_columns) is int and self.block_table_columns == 0
        ):
            raise ValueError("Step request-free workload requires one eager token and no block table")
        if self.block_size_tokens % self.region_tokens:
            raise ValueError("Step cache blocks must contain an integer number of regions")
        queries, lengths = tuple(self.query_lengths), tuple(self.sequence_lengths)
        if not queries or len(queries) != len(lengths):
            raise ValueError("Step workload requires one live sequence length per query")
        for query, length in zip(queries, lengths):
            _positive_int(query, "query_tokens")
            _positive_int(length, "sequence_tokens")
            if query > length or length > self.max_model_len_tokens:
                raise ValueError("Step query/sequence tokens must fit the serving context limit")
            if self.has_attention_metadata and length > self.block_table_columns * self.block_size_tokens:
                raise ValueError("Step live tokens exceed the supplied physical block-table extent")
        if sum(queries) > self.physical_query_rows:
            raise ValueError("Step physical query rows cannot be smaller than valid tokens")
        if self.phase == "decode" and any(query != 1 for query in queries):
            raise ValueError("Step ordinary decode requires one query token per request")
        object.__setattr__(self, "query_lengths", queries)
        object.__setattr__(self, "sequence_lengths", lengths)

    @property
    def valid_tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def padded_tokens(self) -> int:
        return self.physical_query_rows - self.valid_tokens

    @property
    def is_single_token_step(self) -> bool:
        return all(query == 1 for query in self.query_lengths)

    @property
    def score_regions(self) -> int:
        if not self.has_attention_metadata:
            return 0
        if self.is_single_token_step:
            return min(
                (self.max_model_len_tokens + self.region_tokens - 1) // self.region_tokens,
                self.block_table_columns * self.block_size_tokens // self.region_tokens,
            )
        regions = max(
            (length + self.region_tokens - 1) // self.region_tokens
            for length in self.sequence_lengths
        )
        # Native prefill reuses preplanned region indices until history exceeds top-k.
        return regions if regions > self.topk_regions else 0

    @property
    def selected_region_bounds(self) -> tuple[int, ...]:
        """Per-request maximum historical top-k plus the current region."""
        if not self.has_attention_metadata:
            return ()
        return tuple(
            min((length - 1) // self.region_tokens, self.topk_regions) + 1
            for length in self.sequence_lengths
        )

    @classmethod
    def from_lengths(
        cls, config: Any, *, phase: str, query_lengths: tuple[int, ...],
        sequence_lengths: tuple[int, ...], physical_query_rows: int,
        block_size_tokens: int, block_table_columns: int,
        max_model_len_tokens: int, graph_mode: str,
        has_attention_metadata: bool = True,
    ) -> Step5AttentionWorkload:
        validate_step5_config(config)
        if type(max_model_len_tokens) is not int or not 0 < max_model_len_tokens <= config.max_position_embeddings:
            raise ValueError("Step serving limit must fit configured position tokens")
        return cls(
            phase, query_lengths, sequence_lengths, physical_query_rows,
            block_size_tokens, block_table_columns, max_model_len_tokens,
            config.sparse_config["region_block_size"], config.sparse_config["topk"], graph_mode,
            has_attention_metadata,
        )


def describe_step5_operator_tensors(
    metadata: Step5LayerMetadata, operator_name: str, *, tokens: int,
    routed_tokens: int, router_topk: int, attention_operator_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe ordinary physical scope boundaries, not measured native tensors.

    Packed projections retain their per-rank weights. Fused expert internals
    are shapes inside one owner rather than extra separately charged work.
    """
    geometry = metadata.geometry
    hidden = metadata.projections[0].rank_weight_shape[-1]
    model_precision = metadata.model_precision
    shapes: dict[str, list[int]] = {}
    precisions: dict[str, PrecisionType | str] = {}
    internal_shapes: dict[str, list[int]] = {}
    internal_precisions: dict[str, PrecisionType | str] = {}
    projections = tuple(p for p in metadata.projections if p.operator_name == operator_name)

    def tensor(name: str, shape: list[int], precision=model_precision):
        shapes[name] = shape
        precisions[name] = precision

    def internal(name: str, shape: list[int], precision=model_precision):
        internal_shapes[name] = shape
        internal_precisions[name] = precision

    def qkv():
        tensor("q", [tokens, geometry.local_q_heads, geometry.head_dim])
        tensor("k", [tokens, geometry.local_kv_heads, geometry.head_dim])
        tensor("v", [tokens, geometry.local_kv_heads, geometry.head_dim])
        tensor("positions", [tokens], "INT64")

    def indexer():
        tensor("index_q", [tokens, geometry.local_index_q_heads, geometry.proxy_dim])
        tensor("index_k", [tokens, geometry.local_index_k_heads, geometry.proxy_dim])
        tensor("index_z", [tokens, geometry.local_index_k_heads, geometry.proxy_dim])
        tensor("weights", [tokens, geometry.local_index_q_heads])
        internal("quantized_index_q", [tokens, geometry.local_index_q_heads, geometry.proxy_dim],
                 metadata.indexer_query_precision)
        internal("scaled_weights", [tokens, geometry.local_index_q_heads],
                 metadata.indexer_weight_precision)

    if projections:
        if operator_name == "moe_grouped_gemm":
            tensor("input", [tokens, hidden])
            tensor("output", [tokens, hidden])
            local_width = projections[0].rank_weight_shape[1] // 2
            internal("gate_up", [tokens, router_topk, 2 * local_width])
            internal("activation", [routed_tokens, local_width])
            internal("down", [tokens, router_topk, hidden])
        else:
            projection = projections[0]
            tensor("input", [tokens, projection.rank_weight_shape[1]], projection.input_precision)
            tensor("output", [tokens, projection.rank_weight_shape[0]], projection.output_precision)
    elif operator_name in ("input_layernorm", "post_attention_layernorm"):
        tensor("input", [tokens, hidden], metadata.input_norm_precision if operator_name == "input_layernorm"
               else model_precision)
        if metadata.uses_fused_add_norm and (operator_name == "post_attention_layernorm"
                                            or metadata.attention_spec.global_layer_id > 0):
            tensor("residual", [tokens, hidden], metadata.residual_precision)
            tensor("residual_output", [tokens, hidden], metadata.residual_precision)
        tensor("output", [tokens, hidden])
        internal("norm_weight", [hidden], metadata.norm_precision)
    elif operator_name == "step5_attention_prepare":
        qkv()
        if metadata.attention_spec.variant_id == "sparse_gqa" and "step5_indexer_prepare" not in attention_operator_names:
            indexer()
    elif operator_name == "step5_indexer_prepare":
        if metadata.attention_spec.variant_id != "sparse_gqa":
            raise ValueError("Step indexer scope requires a sparse layer")
        indexer()
    elif operator_name in ("step5_sparse_attention", "step5_swa_attention"):
        qkv()
        tensor("output", [tokens, geometry.local_q_heads, geometry.head_dim])
        if metadata.attention_spec.variant_id == "sparse_gqa":
            internal("index_query", [tokens, geometry.local_index_q_heads, geometry.proxy_dim],
                     metadata.indexer_query_precision)
            internal("selector_weights", [tokens, geometry.local_index_q_heads],
                     metadata.indexer_weight_precision)
    elif operator_name == "step5_headwise_gate":
        tensor("input", [tokens, geometry.local_q_heads, geometry.head_dim])
        tensor("gate", [tokens, geometry.local_q_heads])
        tensor("output", [tokens, geometry.local_q_heads, geometry.head_dim])
    elif operator_name == "step5_ffn_merge":
        merge = metadata.ffn_merge
        if merge is None:
            raise ValueError("Step branch merge requires a routed/shared FFN layer")
        tensor("routed", [tokens, merge.hidden_dim], merge.input_precisions[0])
        tensor("shared", [tokens, merge.hidden_dim], merge.input_precisions[1])
        tensor("output", [tokens, merge.hidden_dim], merge.output_precision)
    elif operator_name in ("mlp_act", "share_expert_act"):
        up_name = "mlp_up_proj" if operator_name == "mlp_act" else "share_expert_up_proj"
        up = next(p for p in metadata.projections if p.operator_name == up_name)
        tensor("input", [tokens, up.rank_weight_shape[0]])
        tensor("output", [tokens, up.rank_weight_shape[0] // 2])
    elif operator_name == "moe_gating_routing_topk":
        router = next(p for p in metadata.projections if p.operator_name == "moe_gating_linear")
        tensor("input", [tokens, router.rank_weight_shape[0]], metadata.router_precision)
        tensor("expert_ids", [tokens, router_topk], "INT32")
        tensor("expert_weights", [tokens, router_topk], metadata.router_precision)
    elif operator_name == "moe_shuffling":
        tensor("expert_ids", [tokens, router_topk], "INT32")
    else:
        raise ValueError(f"Unsupported Step physical operator for trace metadata: {operator_name}")

    def precision_name(value):
        return value.name if isinstance(value, PrecisionType) else value

    def tensor_bytes(shape, precision):
        element_bytes = precision.bytes_per_element if isinstance(precision, PrecisionType) else {
            "INT64": 8, "INT32": 4,
        }[precision]
        return int(math.prod(shape) * element_bytes)

    result = {
        "precision_op": operator_name, "dtype": model_precision.name,
        "dtype_bytes": model_precision.bytes_per_element,
        "tensor_shape": shapes,
        "tensor_dtype": {name: precision_name(precisions[name]) for name in shapes},
        "tensor_size_bytes": {name: tensor_bytes(shape, precisions[name]) for name, shape in shapes.items()},
        "source_geometry": {
            "global_layer_id": metadata.attention_spec.global_layer_id,
            "head_dim": geometry.head_dim, "local_q_heads": geometry.local_q_heads,
            "local_kv_heads": geometry.local_kv_heads, "kv_replication": geometry.kv_replication,
        },
        "evidence_kind": "source_structural_metadata",
    }
    if operator_name == "moe_grouped_gemm":
        result["covered_scopes"] = ["gate_up_gemm", "swiglu", "down_gemm", "topk_reduce"]
    elif operator_name == "moe_shuffling":
        result["covered_scopes"] = ["expert_assignment_alignment"]
        result["alignment_outputs"] = ["sorted_token_ids", "expert_ids", "num_tokens_post_padded"]
        result["alignment_output_shapes"] = "unresolved native dispatch and alignment padding"
        result["sorted_token_ids_nullable"] = True
    if projections:
        result["projection_modules"] = [
            {"module_prefix": p.module_prefix, "weight_shape": list(p.rank_weight_shape),
             "weight_dtype": p.weight_precision.name, "input_dtype": p.input_precision.name,
             "output_dtype": p.output_precision.name} for p in projections
        ]
    if internal_shapes:
        result["internal_tensor_shape"] = internal_shapes
        result["internal_tensor_dtype"] = {name: precision_name(internal_precisions[name]) for name in internal_shapes}
    return result
