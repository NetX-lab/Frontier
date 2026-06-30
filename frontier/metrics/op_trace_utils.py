from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.memory import get_attention_runtime_kv_layout
from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionOperatorRole
from frontier.config import get_quantization_manager
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.types import ClusterType


@dataclass(frozen=True)
class OpTraceContext:
    cluster_type: ClusterType
    model_config: BaseModelConfig
    replica_config: Any
    total_tokens: int
    effective_tokens_compute: int
    effective_tokens_transfer: int
    effective_tokens_rounded: int
    tokens_are_post_routing: bool

    @property
    def hidden_size(self) -> int:
        return self.model_config.embedding_dim

    @property
    def num_q_heads(self) -> int:
        return self.model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self.model_config.num_kv_heads

    @property
    def head_dim(self) -> int:
        # Use model_config.get_head_dim() to prioritize explicit head_dim from JSON config
        # This ensures consistency with the profiling module's ModelConfig.get_head_size()
        return self.model_config.get_head_dim()

    @property
    def uses_mla(self) -> bool:
        uses_mla = getattr(self.model_config, "uses_mla", None)
        if callable(uses_mla):
            return bool(uses_mla())
        return bool(getattr(self.model_config, "use_mla", False))

    @property
    def runtime_num_kv_heads(self) -> int:
        if not self.uses_mla:
            return self.num_kv_heads
        getter = getattr(self.model_config, "get_runtime_num_kv_heads", None)
        if not callable(getter):
            raise ValueError(
                "MLA trace metadata requires model_config.get_runtime_num_kv_heads()"
            )
        return int(getter())

    @property
    def runtime_head_size(self) -> int:
        if not self.uses_mla:
            return self.head_dim
        getter = getattr(self.model_config, "get_runtime_head_size", None)
        if not callable(getter):
            raise ValueError(
                "MLA trace metadata requires model_config.get_runtime_head_size()"
            )
        return int(getter())

    def require_mla_dim(self, attr_name: str) -> int:
        value = getattr(self.model_config, attr_name, None)
        if value is None:
            raise ValueError(f"MLA trace metadata requires {attr_name}")
        return int(value)

    @property
    def intermediate_size(self) -> int:
        return self.model_config.mlp_hidden_dim

    @property
    def share_expert_dim(self) -> int:
        if self.model_config.share_expert_dim is None:
            raise ValueError("share_expert_dim must be set for share_expert ops")
        return int(self.model_config.share_expert_dim)

    @property
    def num_experts(self) -> int:
        return self.model_config.num_experts

    @property
    def attn_tp(self) -> int:
        return int(self.replica_config.attn_tensor_parallel_size)

    @property
    def attn_dp(self) -> int:
        return int(self.replica_config.attn_data_parallel_size)

    @property
    def moe_tp(self) -> int:
        return int(self.replica_config.moe_tensor_parallel_size)

    @property
    def moe_ep(self) -> int:
        return int(self.replica_config.moe_expert_parallel_size)

    @property
    def pp(self) -> int:
        return int(self.replica_config.num_pipeline_stages)

    @property
    def router_topk(self) -> int:
        if not self.model_config.is_moe:
            return 1
        router_topk = int(getattr(self.replica_config, "router_topk", 0) or 0)
        if router_topk > 0:
            return router_topk
        fallback = int(self.model_config.num_experts_per_tok or 0)
        if fallback > 0:
            return fallback
        raise ValueError("router_topk is not set for MoE operations")


def build_parallel_context(ctx: OpTraceContext) -> Dict[str, int]:
    return {
        "PP": ctx.pp,
        "ATTN_TP": ctx.attn_tp,
        "ATTN_DP": ctx.attn_dp,
        "MOE_TP": ctx.moe_tp,
        "MOE_EP": ctx.moe_ep,
        "ROUTER_TOPK": ctx.router_topk if ctx.model_config.is_moe else 1,
    }


def map_trace_op_to_precision_op(op_name: str) -> str:
    mapping = {
        "add_attn_residual": "add",
        "add_ffn_residual": "add",
        "moe_gating_linear": "moe_gating",
        "moe_gating_routing_topk": "moe_gating",
        "attn_tensor_parallel_allreduce": "allreduce",
        "mlp_tensor_parallel_allreduce": "allreduce",
        "moe_tensor_parallel_allreduce": "allreduce",
        "moe_tensor_parallel_allgather": "allgather",
        "share_expert_tensor_parallel_allreduce": "allreduce",
        "expert_parallel_allreduce": "allreduce",
        "pipeline_parallel_send_recv": "send_recv",
        "expert_parallel_alltoall": "expert_parallel_communication",
        "expert_parallel_alltoall_dispatch": "expert_parallel_communication",
        "expert_parallel_alltoall_combine": "expert_parallel_communication",
        "m2n_transfer_attn_to_ffn": "m2n_transfer",
        "m2n_transfer_ffn_to_attn": "m2n_transfer",
        "m2n_transfer_ffn_to_attn_recv": "m2n_transfer",
    }
    return mapping.get(op_name, op_name)


def _validate_divisible(value: int, divisor: int, label: str) -> None:
    if divisor <= 0:
        raise ValueError(f"{label} divisor must be > 0")
    if value % divisor != 0:
        raise ValueError(f"{label} must be divisible by {divisor}")


def _elements_from_shape(shape: List[int]) -> int:
    if not shape:
        return 0
    result = 1
    for dim in shape:
        if dim < 0:
            raise ValueError("Shape dimensions must be >= 0")
        result *= dim
    return result


def _bytes_for_elements(num_elements: int, dtype_bytes: float) -> int:
    if num_elements <= 0:
        return 0
    return int(math.ceil(num_elements * dtype_bytes))


def _get_dense_attention_op_name_by_role(role: AttentionOperatorRole) -> str:
    matches = tuple(
        operator.name
        for operator in DENSE_ATTENTION_FAMILY.e2e_trace_ops()
        if operator.role is role
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one E2E trace operator for role "
            f"{role.value!r} in attention family "
            f"{DENSE_ATTENTION_FAMILY.family_id!r}; found {len(matches)}: "
            f"{list(matches)}"
        )
    return matches[0]


def _get_pre_routing_tokens(tokens: int, ctx: OpTraceContext) -> int:
    if not ctx.model_config.is_moe:
        return tokens
    if ctx.tokens_are_post_routing:
        if tokens % ctx.router_topk != 0:
            # EP sharding can produce imbalanced post-routing tokens that are not divisible by router_topk.
            # Use ceil division to recover a stable pre-routing token estimate for trace metadata.
            return (tokens + ctx.router_topk - 1) // ctx.router_topk
        return tokens // ctx.router_topk
    return tokens


def _get_routed_tokens(tokens: int, ctx: OpTraceContext) -> int:
    if not ctx.model_config.is_moe:
        return tokens
    if ctx.tokens_are_post_routing:
        return tokens
    return tokens * ctx.router_topk


def _precision_for_op(op_name: str, cluster_type: ClusterType) -> PrecisionType:
    quant_manager = get_quantization_manager()
    precision_op = map_trace_op_to_precision_op(op_name)
    return quant_manager.get_precision(precision_op, cluster_type)


def _comm_size_bytes(
    op_name: str,
    element_count: int,
    cluster_type: ClusterType,
) -> Tuple[int, int]:
    quant_manager = get_quantization_manager()
    precision_op = map_trace_op_to_precision_op(op_name)
    base_size_bytes = element_count * 2
    adjusted = quant_manager.adjust_tensor_size(precision_op, base_size_bytes, cluster_type)
    return base_size_bytes, adjusted


def compute_op_trace_meta(
    op_name: str,
    op_type: str,
    ctx: OpTraceContext,
) -> Dict[str, Any]:
    precision_op = map_trace_op_to_precision_op(op_name)
    precision = _precision_for_op(op_name, ctx.cluster_type)
    dtype_bytes = precision.bytes_per_element

    tokens = ctx.effective_tokens_rounded
    hidden_size = ctx.hidden_size
    intermediate_size = ctx.intermediate_size

    attention_meta: Optional[Tuple[int, int, int, int]] = None
    mla_attention_meta: Optional[
        Tuple[int, int, int, int, int, int, int, int, int]
    ] = None

    def _get_attention_meta() -> Tuple[int, int, int, int]:
        nonlocal attention_meta
        if attention_meta is None:
            _validate_divisible(ctx.num_q_heads, ctx.attn_tp, "num_q_heads")
            _validate_divisible(hidden_size, ctx.attn_tp, "hidden_size")

            # Match vLLM KV-head semantics:
            # - Partition when kv_heads >= tp and divisible.
            # - Replicate when kv_heads < tp and tp % kv_heads == 0.
            if ctx.num_kv_heads >= ctx.attn_tp:
                _validate_divisible(ctx.num_kv_heads, ctx.attn_tp, "num_kv_heads")
                kv_heads_per_tp = ctx.num_kv_heads // ctx.attn_tp
            else:
                if ctx.attn_tp % ctx.num_kv_heads != 0:
                    raise ValueError(
                        "num_kv_heads replication requires attn_tp to be divisible by num_kv_heads"
                    )
                kv_heads_per_tp = 1

            head_dim = ctx.head_dim
            q_heads_per_tp = ctx.num_q_heads // ctx.attn_tp
            hidden_size_per_tp = hidden_size // ctx.attn_tp
            attention_meta = (head_dim, q_heads_per_tp, kv_heads_per_tp, hidden_size_per_tp)
        return attention_meta

    def _get_mla_attention_meta() -> Tuple[int, int, int, int, int, int, int, int, int]:
        nonlocal mla_attention_meta
        if mla_attention_meta is None:
            if not ctx.uses_mla:
                raise ValueError("MLA trace metadata requires use_mla=True")
            _validate_divisible(ctx.num_q_heads, ctx.attn_tp, "num_q_heads")
            _validate_divisible(hidden_size, ctx.attn_tp, "hidden_size")

            runtime_num_kv_heads = ctx.runtime_num_kv_heads
            if runtime_num_kv_heads <= 0:
                raise ValueError(
                    f"runtime_num_kv_heads must be positive: {runtime_num_kv_heads}"
                )
            if runtime_num_kv_heads >= ctx.attn_tp:
                _validate_divisible(
                    runtime_num_kv_heads,
                    ctx.attn_tp,
                    "runtime_num_kv_heads",
                )
                runtime_kv_heads_per_tp = runtime_num_kv_heads // ctx.attn_tp
            else:
                if ctx.attn_tp % runtime_num_kv_heads != 0:
                    raise ValueError(
                        "MLA runtime_num_kv_heads replication requires attn_tp "
                        "to be divisible by runtime_num_kv_heads"
                    )
                runtime_kv_heads_per_tp = 1

            kv_lora_rank = ctx.require_mla_dim("kv_lora_rank")
            qk_nope_head_dim = ctx.require_mla_dim("qk_nope_head_dim")
            qk_rope_head_dim = ctx.require_mla_dim("qk_rope_head_dim")
            qk_head_dim = ctx.require_mla_dim("qk_head_dim")
            v_head_dim = ctx.require_mla_dim("v_head_dim")
            runtime_head_size = ctx.runtime_head_size

            if qk_head_dim != qk_nope_head_dim + qk_rope_head_dim:
                raise ValueError(
                    "MLA qk_head_dim must equal qk_nope_head_dim + "
                    f"qk_rope_head_dim: qk_head_dim={qk_head_dim}, "
                    f"qk_nope_head_dim={qk_nope_head_dim}, "
                    f"qk_rope_head_dim={qk_rope_head_dim}"
                )
            if runtime_head_size != kv_lora_rank + qk_rope_head_dim:
                raise ValueError(
                    "MLA runtime_head_size must equal kv_lora_rank + "
                    f"qk_rope_head_dim: runtime_head_size={runtime_head_size}, "
                    f"kv_lora_rank={kv_lora_rank}, "
                    f"qk_rope_head_dim={qk_rope_head_dim}"
                )

            q_heads_per_tp = ctx.num_q_heads // ctx.attn_tp
            hidden_size_per_tp = hidden_size // ctx.attn_tp
            mla_attention_meta = (
                q_heads_per_tp,
                runtime_kv_heads_per_tp,
                hidden_size_per_tp,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                qk_head_dim,
                v_head_dim,
                runtime_head_size,
            )
        return mla_attention_meta

    tensor_shape: Dict[str, Any] = {}
    tensor_size_bytes: Dict[str, int] = {}

    if op_type == "COMPUTE":
        if op_name == "input_layernorm":
            tensor_shape = {"input": [tokens, hidden_size], "output": [tokens, hidden_size]}
        elif op_name == "attn_pre_proj":
            head_dim, q_heads_per_tp, kv_heads_per_tp, _ = _get_attention_meta()
            qkv_out_dim_per_tp = (q_heads_per_tp + 2 * kv_heads_per_tp) * head_dim
            tensor_shape = {
                "input": [tokens, hidden_size],
                "output": [tokens, qkv_out_dim_per_tp],
            }
        elif op_name == "attn_rope":
            head_dim, q_heads_per_tp, kv_heads_per_tp, _ = _get_attention_meta()
            tensor_shape = {
                "q": [tokens, q_heads_per_tp, head_dim],
                "k": [tokens, kv_heads_per_tp, head_dim],
                "q_out": [tokens, q_heads_per_tp, head_dim],
                "k_out": [tokens, kv_heads_per_tp, head_dim],
            }
        elif op_name in (
            _get_dense_attention_op_name_by_role(AttentionOperatorRole.PREFILL_KERNEL),
            _get_dense_attention_op_name_by_role(AttentionOperatorRole.DECODE_KERNEL),
        ):
            head_dim, q_heads_per_tp, kv_heads_per_tp, hidden_size_per_tp = _get_attention_meta()
            tensor_shape = {
                "q": [tokens, q_heads_per_tp, head_dim],
                "k": [tokens, kv_heads_per_tp, head_dim],
                "v": [tokens, kv_heads_per_tp, head_dim],
                "output": [tokens, hidden_size_per_tp],
            }
        elif op_name == _get_dense_attention_op_name_by_role(
            AttentionOperatorRole.CACHE_WRITE
        ):
            head_dim, _, kv_heads_per_tp, _ = _get_attention_meta()
            tensor_shape = {
                "k": [tokens, kv_heads_per_tp, head_dim],
                "v": [tokens, kv_heads_per_tp, head_dim],
            }
        elif op_name == "attn_mla_kv_cache_save":
            (
                _q_heads_per_tp,
                runtime_kv_heads_per_tp,
                _hidden_size_per_tp,
                _kv_lora_rank,
                _qk_nope_head_dim,
                _qk_rope_head_dim,
                _qk_head_dim,
                _v_head_dim,
                runtime_head_size,
            ) = _get_mla_attention_meta()
            tensor_shape = {
                "kv": [tokens, runtime_kv_heads_per_tp, runtime_head_size],
            }
        elif op_name == "attn_mla_prefill_kv_up_proj":
            (
                q_heads_per_tp,
                runtime_kv_heads_per_tp,
                _hidden_size_per_tp,
                kv_lora_rank,
                qk_nope_head_dim,
                _qk_rope_head_dim,
                _qk_head_dim,
                v_head_dim,
                _runtime_head_size,
            ) = _get_mla_attention_meta()
            tensor_shape = {
                "latent_kv": [tokens, runtime_kv_heads_per_tp, kv_lora_rank],
                "k_nope_v": [tokens, q_heads_per_tp, qk_nope_head_dim + v_head_dim],
            }
        elif op_name in ("attn_mla_prefill", "attn_mla_decode"):
            (
                q_heads_per_tp,
                runtime_kv_heads_per_tp,
                hidden_size_per_tp,
                _kv_lora_rank,
                _qk_nope_head_dim,
                _qk_rope_head_dim,
                qk_head_dim,
                _v_head_dim,
                runtime_head_size,
            ) = _get_mla_attention_meta()
            tensor_shape = {
                "q": [tokens, q_heads_per_tp, qk_head_dim],
                "latent_kv": [tokens, runtime_kv_heads_per_tp, runtime_head_size],
                "output": [tokens, hidden_size_per_tp],
            }
        elif op_name == "attn_mla_decode_q_latent_proj":
            (
                q_heads_per_tp,
                _runtime_kv_heads_per_tp,
                _hidden_size_per_tp,
                _kv_lora_rank,
                qk_nope_head_dim,
                _qk_rope_head_dim,
                _qk_head_dim,
                _v_head_dim,
                _runtime_head_size,
            ) = _get_mla_attention_meta()
            tensor_shape = {
                "input": [tokens, hidden_size],
                "q_nope": [tokens, q_heads_per_tp, qk_nope_head_dim],
            }
        elif op_name == "attn_mla_v_up_proj":
            (
                q_heads_per_tp,
                _runtime_kv_heads_per_tp,
                hidden_size_per_tp,
                _kv_lora_rank,
                _qk_nope_head_dim,
                _qk_rope_head_dim,
                _qk_head_dim,
                v_head_dim,
                _runtime_head_size,
            ) = _get_mla_attention_meta()
            tensor_shape = {
                "input": [tokens, q_heads_per_tp, v_head_dim],
                "output": [tokens, hidden_size_per_tp],
            }
        elif op_name == "attn_post_proj":
            _, _, _, hidden_size_per_tp = _get_attention_meta()
            tensor_shape = {
                "input": [tokens, hidden_size_per_tp],
                "output": [tokens, hidden_size],
            }
        elif op_name == "post_attention_layernorm":
            tensor_shape = {"input": [tokens, hidden_size], "output": [tokens, hidden_size]}
        elif op_name in ("add_attn_residual", "add_ffn_residual", "add"):
            tensor_shape = {
                "input_a": [tokens, hidden_size],
                "input_b": [tokens, hidden_size],
                "output": [tokens, hidden_size],
            }
        elif op_name == "mlp_up_proj":
            _validate_divisible(intermediate_size, ctx.attn_tp, "intermediate_size")
            tensor_shape = {
                "input": [tokens, hidden_size],
                "output": [tokens, intermediate_size // ctx.attn_tp],
            }
        elif op_name == "mlp_act":
            _validate_divisible(intermediate_size, ctx.attn_tp, "intermediate_size")
            tensor_shape = {
                "input": [tokens, intermediate_size // ctx.attn_tp],
                "output": [tokens, intermediate_size // ctx.attn_tp],
            }
        elif op_name == "mlp_down_proj":
            _validate_divisible(intermediate_size, ctx.attn_tp, "intermediate_size")
            tensor_shape = {
                "input": [tokens, intermediate_size // ctx.attn_tp],
                "output": [tokens, hidden_size],
            }
        elif op_name == "share_expert_up_proj":
            share_expert_dim = ctx.share_expert_dim
            _validate_divisible(share_expert_dim, ctx.moe_tp, "share_expert_dim")
            tensor_shape = {
                "input": [tokens, hidden_size],
                "output": [tokens, share_expert_dim // ctx.moe_tp],
            }
        elif op_name == "share_expert_act":
            share_expert_dim = ctx.share_expert_dim
            _validate_divisible(share_expert_dim, ctx.moe_tp, "share_expert_dim")
            tensor_shape = {
                "input": [tokens, share_expert_dim // ctx.moe_tp],
                "output": [tokens, share_expert_dim // ctx.moe_tp],
            }
        elif op_name == "share_expert_down_proj":
            share_expert_dim = ctx.share_expert_dim
            _validate_divisible(share_expert_dim, ctx.moe_tp, "share_expert_dim")
            tensor_shape = {
                "input": [tokens, share_expert_dim // ctx.moe_tp],
                "output": [tokens, hidden_size],
            }
        elif op_name == "moe_gating_linear":
            if ctx.num_experts <= 0:
                raise ValueError("num_experts must be > 0 for MoE gating")
            tensor_shape = {
                "input": [tokens, hidden_size],
                "output": [tokens, ctx.num_experts],
            }
        elif op_name == "moe_gating_routing_topk":
            if ctx.num_experts <= 0:
                raise ValueError("num_experts must be > 0 for MoE gating")
            tensor_shape = {
                "input": [tokens, ctx.num_experts],
                "output": [tokens, ctx.router_topk],
            }
        elif op_name == "moe_shuffling":
            pre_routing_tokens = _get_pre_routing_tokens(tokens, ctx)
            tensor_shape = {
                "input": [pre_routing_tokens, ctx.router_topk, hidden_size],
                "output": [pre_routing_tokens, ctx.router_topk, hidden_size],
            }
        elif op_name == "moe_grouped_gemm":
            _validate_divisible(intermediate_size, ctx.moe_tp, "moe_intermediate_size")
            routed_tokens = _get_routed_tokens(tokens, ctx)
            tensor_shape = {
                "input": [routed_tokens, hidden_size],
                "output": [routed_tokens, intermediate_size // ctx.moe_tp],
            }
        else:
            raise ValueError(f"Unsupported compute op for tracing: {op_name}")

        for key, shape in tensor_shape.items():
            elements = _elements_from_shape(shape)
            tensor_size_bytes[key] = _bytes_for_elements(elements, dtype_bytes)

        return {
            "precision_op": precision_op,
            "dtype": precision.name,
            "dtype_bytes": dtype_bytes,
            "tensor_shape": tensor_shape,
            "tensor_size_bytes": tensor_size_bytes,
        }

    if op_type == "COMM":
        if op_name in (
            "attn_tensor_parallel_allreduce",
            "mlp_tensor_parallel_allreduce",
            "moe_tensor_parallel_allreduce",
            "moe_tensor_parallel_allgather",
            "share_expert_tensor_parallel_allreduce",
            "expert_parallel_allreduce",
            "pipeline_parallel_send_recv",
        ):
            tensor_shape = {"data": [tokens, hidden_size]}
            element_count = _elements_from_shape(tensor_shape["data"])
        elif op_name in (
            "expert_parallel_alltoall",
            "expert_parallel_alltoall_dispatch",
            "expert_parallel_alltoall_combine",
        ):
            pre_routing_tokens = _get_pre_routing_tokens(tokens, ctx)
            tensor_shape = {"data": [pre_routing_tokens, ctx.router_topk, hidden_size]}
            element_count = _elements_from_shape(tensor_shape["data"])
        else:
            raise ValueError(f"Unsupported communication op for tracing: {op_name}")

        base_size_bytes, data_size_bytes = _comm_size_bytes(
            op_name, element_count, ctx.cluster_type
        )

        return {
            "precision_op": precision_op,
            "dtype": precision.name,
            "dtype_bytes": dtype_bytes,
            "tensor_shape": tensor_shape,
            "element_count": element_count,
            "base_size_bytes": base_size_bytes,
            "data_size_bytes": data_size_bytes,
        }

    raise ValueError(f"Unsupported op_type for tracing: {op_type}")


def build_kv_cache_transfer_meta(
    batch: Any,
    replica_config: Any,
    cluster_type: ClusterType,
    transfer_size_bytes: int,
) -> Dict[str, Any]:
    model_config = replica_config.model_config
    num_layers = model_config.num_layers
    num_q_heads = model_config.num_q_heads
    # Family-aware runtime KV layout: dense keeps (num_kv_heads, get_head_dim(), kv_factor=2);
    # latent MLA collapses to (1, kv_lora_rank + qk_rope_head_dim, kv_factor=1). Mirrors the
    # analytical transfer predictor so capacity and transfer agree on the same MLA cache.
    family = bind_attention_family(model_config).family
    layout = get_attention_runtime_kv_layout(
        family,
        runtime_num_kv_heads_per_worker=model_config.get_runtime_num_kv_heads(),
        runtime_head_size=model_config.get_runtime_head_size(),
    )
    num_kv_heads = layout.runtime_num_kv_heads_per_worker
    head_dim = layout.runtime_head_size

    total_tokens = sum(req.num_prefill_tokens for req in batch.requests)
    precision = _precision_for_op("kv_cache_transfer", cluster_type)
    dtype_bytes = precision.bytes_per_element
    tensor_shape = {
        "kv": [total_tokens, num_layers, num_kv_heads, head_dim, layout.kv_factor],
    }
    element_count = _elements_from_shape(tensor_shape["kv"])
    tensor_size_bytes = {
        "kv": _bytes_for_elements(element_count, dtype_bytes),
    }

    return {
        "precision_op": "kv_cache_transfer",
        "dtype": precision.name,
        "dtype_bytes": dtype_bytes,
        "tensor_shape": tensor_shape,
        "tensor_size_bytes": tensor_size_bytes,
        "transfer_size_bytes": transfer_size_bytes,
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "num_heads": num_kv_heads,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def build_m2n_transfer_meta(
    batch: Any,
    replica_config: Any,
    cluster_type: ClusterType,
    activation_size_bytes: int,
) -> Dict[str, Any]:
    model_config = replica_config.model_config
    hidden_size = model_config.embedding_dim
    total_tokens = batch.get_effective_total_tokens_for_transfer(cluster_type)
    precision = _precision_for_op("m2n_transfer", cluster_type)
    dtype_bytes = precision.bytes_per_element
    tensor_shape = {"activation": [total_tokens, hidden_size]}
    element_count = _elements_from_shape(tensor_shape["activation"])
    tensor_size_bytes = {
        "activation": _bytes_for_elements(element_count, dtype_bytes),
    }

    return {
        "precision_op": "m2n_transfer",
        "dtype": precision.name,
        "dtype_bytes": dtype_bytes,
        "tensor_shape": tensor_shape,
        "tensor_size_bytes": tensor_size_bytes,
        "activation_size_bytes": activation_size_bytes,
        "total_tokens": total_tokens,
        "hidden_size": hidden_size,
    }
