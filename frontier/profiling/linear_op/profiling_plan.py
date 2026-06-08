"""Profiling plan construction for decoupled attention/FFN TP profiling."""

from __future__ import annotations

from typing import Dict, List, Sequence

from frontier.spec_decode.runtime import TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS


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
FFN_OPS = [
    "mlp_up_proj",
    "mlp_down_proj",
    "mlp_act",
]
SHARE_EXPERT_OPS = [
    "share_expert_up_proj",
    "share_expert_down_proj",
    "share_expert_act",
]
FFN_REPLICATED_OPS = [
    "post_attention_layernorm",
]
COMMON_REPLICATED_OPS = [
    "add",
    "emb",
]
TARGET_EMBEDDED_MTP_OPS = [
    "mtp_fusion_proj",
    "lm_head_linear",
]


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _supports_share_expert(model_config) -> bool:
    if hasattr(model_config, "supports_share_expert"):
        return model_config.supports_share_expert()
    return getattr(model_config, "model_type", None) in {"step2_mini", "step3_text"}


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

    replicated_ops: List[str] = []
    replicated_ops.extend(ATTN_REPLICATED_OPS)
    if getattr(model_config, "model_type", None) == "step3_text":
        replicated_ops.extend(ATTN_STEP3TEXT_REPLICATED_OPS)
    if getattr(model_config, "post_attn_norm", False):
        replicated_ops.extend(FFN_REPLICATED_OPS)
    replicated_ops.extend(COMMON_REPLICATED_OPS)
    target_embedded_same_tp_ops: List[str] = []
    if include_target_embedded_mtp:
        target_embedded_same_tp_ops.extend(
            [
                op_name
                for op_name in TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS
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
        enabled_ops.extend(ATTN_BASE_OPS)
        if getattr(model_config, "is_step2_mini", False):
            enabled_ops.extend(ATTN_STEP2MINI_OPS)
        if getattr(model_config, "model_type", None) == "step3_text":
            enabled_ops.extend(ATTN_STEP3TEXT_SHARDED_OPS)
        if include_target_embedded_mtp:
            enabled_ops.extend(TARGET_EMBEDDED_MTP_OPS)

    if ffn_sharded_enabled:
        if not is_moe:
            enabled_ops.extend(FFN_OPS)
        if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
            enabled_ops.extend(SHARE_EXPERT_OPS)

    all_ops: List[str] = []
    all_ops.extend(replicated_ops)
    all_ops.extend(ATTN_BASE_OPS)
    if getattr(model_config, "is_step2_mini", False):
        all_ops.extend(ATTN_STEP2MINI_OPS)
    if getattr(model_config, "model_type", None) == "step3_text":
        all_ops.extend(ATTN_STEP3TEXT_REPLICATED_OPS)
        all_ops.extend(ATTN_STEP3TEXT_SHARDED_OPS)
    if include_target_embedded_mtp:
        all_ops.extend(TARGET_EMBEDDED_MTP_OPS)
    if not is_moe:
        all_ops.extend(FFN_OPS)
    if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
        all_ops.extend(SHARE_EXPERT_OPS)
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
