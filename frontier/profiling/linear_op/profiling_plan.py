"""Profiling plan construction for decoupled attention/FFN TP profiling."""

from __future__ import annotations

from typing import Dict, List, Sequence

from frontier.operators.families import (
    FFN_FAMILY,
    MEMORY_FAMILY,
    SHARE_EXPERT_FAMILY,
    get_family_profiling_names,
)
from frontier.model_architectures import get_model_architecture_profile
from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_linear_ops,
    get_target_embedded_mtp_same_tp_linear_ops,
)


ATTN_BASE_OPS = [
    "attn_pre_proj",
    "attn_rope",
    "attn_post_proj",
]
ATTN_STEP2MINI_OPS = [
    "attn_inter_norm",
    "attn_wq_proj",
]
ATTN_STEP3TEXT_REPLICATED_OPS = [
    "attn_pre_proj_qkv",
    "attn_pre_proj_q_norm",
]
ATTN_STEP3TEXT_SHARDED_OPS = [
    "attn_pre_proj_wq",
]
ATTN_REPLICATED_OPS = [
    "input_layernorm",
]
TARGET_EMBEDDED_MTP_OPS = list(get_target_embedded_mtp_linear_ops())


def _dedupe_preserving_order(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _memory_profiling_names(model_config) -> List[str]:
    operators = []
    for operator in MEMORY_FAMILY.profiling_ops():
        if (
            operator.name == "post_attention_layernorm"
            and not getattr(model_config, "post_attn_norm", False)
        ):
            continue
        operators.append(operator)
    return _dedupe_preserving_order(
        [operator.profiling_name() for operator in operators]
    )


def _ffn_profiling_names() -> List[str]:
    return list(get_family_profiling_names(FFN_FAMILY))


def _share_expert_profiling_names() -> List[str]:
    return list(get_family_profiling_names(SHARE_EXPERT_FAMILY))


def memory_operator_enabled(
    enabled_ops: Sequence[str] | set[str] | None,
    operator_name: str,
) -> bool:
    if enabled_ops is None:
        return True
    enabled_op_set = set(enabled_ops)
    for operator in MEMORY_FAMILY.profiling_ops():
        if operator.name == operator_name:
            return operator.profiling_name() in enabled_op_set
    raise ValueError(f"Unknown MEMORY profiling operator: {operator_name}")


def _bool_config_value(model_config, name: str) -> bool:
    value = getattr(model_config, name, False)
    if callable(value):
        value = value()
    return bool(value)


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _supports_share_expert(model_config) -> bool:
    if not hasattr(model_config, "supports_share_expert"):
        raise TypeError("linear-op profiling requires model_config.supports_share_expert()")
    return model_config.supports_share_expert()


def build_profiling_plan(
    model_config,
    tp_size: int,
    attn_tp: Sequence[int],
    ffn_tp: Sequence[int],
    disable_replicated: bool = False,
    is_moe: bool = False,
    include_target_embedded_mtp: bool = False,
) -> Dict[str, object]:
    attn_tp_set = set(attn_tp)
    ffn_tp_set = set(ffn_tp)

    skip_reasons: List[str] = []

    if getattr(model_config, "no_tensor_parallel", False) and tp_size > 1:
        skip_reasons.append("no_tensor_parallel")
        attn_sharded_enabled = False
        ffn_sharded_enabled = False
    else:
        attn_sharded_enabled = tp_size in attn_tp_set
        if attn_sharded_enabled:
            if model_config.embedding_dim % tp_size != 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"embedding_dim={model_config.embedding_dim} not divisible by TP={tp_size}"
                )
            if model_config.num_q_heads % tp_size != 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"num_q_heads={model_config.num_q_heads} not divisible by TP={tp_size}"
                )
            if model_config.num_kv_heads <= 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"num_kv_heads must be positive, got {model_config.num_kv_heads}"
                )
            elif model_config.num_kv_heads >= tp_size:
                if model_config.num_kv_heads % tp_size != 0:
                    attn_sharded_enabled = False
                    skip_reasons.append(
                        f"num_kv_heads={model_config.num_kv_heads} not divisible by TP={tp_size}"
                    )
            else:
                if tp_size % model_config.num_kv_heads != 0:
                    attn_sharded_enabled = False
                    skip_reasons.append(
                        f"TP={tp_size} must be divisible by num_kv_heads={model_config.num_kv_heads} for KV-head replication"
                    )

        ffn_sharded_enabled = tp_size in ffn_tp_set

    padded_n_embd = model_config.embedding_dim
    padded_n_expanded_embd = model_config.mlp_hidden_dim
    if ffn_sharded_enabled:
        padded_n_embd = _pad_to_multiple(model_config.embedding_dim, tp_size)
        padded_n_expanded_embd = _pad_to_multiple(model_config.mlp_hidden_dim, tp_size)

    replicated_enabled = not disable_replicated
    attn_enabled = attn_sharded_enabled or replicated_enabled
    ffn_enabled = ffn_sharded_enabled or replicated_enabled

    architecture_profile = get_model_architecture_profile(model_config)
    linear_attention = architecture_profile.linear_attention
    memory_ops = _memory_profiling_names(model_config)

    replicated_ops: List[str] = []
    replicated_ops.extend(
        [op_name for op_name in ATTN_REPLICATED_OPS if op_name in memory_ops]
    )
    replicated_ops.extend(linear_attention.replicated_ops)
    replicated_ops.extend(
        [op_name for op_name in memory_ops if op_name not in ATTN_REPLICATED_OPS]
    )
    target_embedded_same_tp_ops: List[str] = []
    if include_target_embedded_mtp:
        target_embedded_same_tp_ops.extend(
            [
                op_name
                for op_name in get_target_embedded_mtp_same_tp_linear_ops()
                if op_name != "post_attention_layernorm"
                or getattr(model_config, "post_attn_norm", False)
            ]
        )
        same_tp_ops_set = set(target_embedded_same_tp_ops)
        replicated_ops = [
            op_name for op_name in replicated_ops if op_name not in same_tp_ops_set
        ]

    enabled_ops: List[str] = []
    if replicated_enabled:
        enabled_ops.extend(replicated_ops)
    if include_target_embedded_mtp and (tp_size == 1 or attn_sharded_enabled):
        enabled_ops.extend(target_embedded_same_tp_ops)

    if attn_sharded_enabled:
        enabled_ops.extend(linear_attention.sharded_ops)
        if include_target_embedded_mtp:
            enabled_ops.extend(TARGET_EMBEDDED_MTP_OPS)

    if ffn_sharded_enabled:
        if not is_moe:
            enabled_ops.extend(_ffn_profiling_names())
        if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
            enabled_ops.extend(_share_expert_profiling_names())

    all_ops: List[str] = []
    all_ops.extend(replicated_ops)
    all_ops.extend(linear_attention.sharded_ops)
    if include_target_embedded_mtp:
        all_ops.extend(TARGET_EMBEDDED_MTP_OPS)
    if not is_moe:
        all_ops.extend(_ffn_profiling_names())
    if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
        all_ops.extend(_share_expert_profiling_names())
    # Remove duplicates while preserving order.
    all_ops = list(dict.fromkeys(all_ops))
    enabled_ops = list(dict.fromkeys(enabled_ops))
    disabled_ops = [op for op in all_ops if op not in set(enabled_ops)]

    return {
        "tp_size": tp_size,
        "attn_enabled": attn_enabled,
        "ffn_enabled": ffn_enabled,
        "attn_sharded_enabled": attn_sharded_enabled,
        "ffn_sharded_enabled": ffn_sharded_enabled,
        "replicated_enabled": replicated_enabled,
        "disable_replicated": disable_replicated,
        "enabled_ops": enabled_ops,
        "disabled_ops": disabled_ops,
        "replicated_ops": replicated_ops if replicated_enabled else [],
        "padded_n_embd": padded_n_embd,
        "padded_n_expanded_embd": padded_n_expanded_embd,
        "skip_reasons": skip_reasons,
    }
