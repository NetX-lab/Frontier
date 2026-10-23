from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, TypedDict, cast


class MTPRuntimePolicy(TypedDict):
    fusion_op_name: str
    fusion_is_tp_sharded: bool
    fusion_requires_allgather: bool
    norm_op_name: str
    num_pre_fusion_norms: int
    num_post_decoder_norms: int
    embedding_requires_allreduce: bool
    lm_head_op_name: str
    lm_head_requires_allgather: bool


class TargetEmbeddedMTPMethodContract(TypedDict):
    mtp_family: str
    uses_lookahead_slots: bool
    requires_prefix_matching_disabled: bool
    same_tp_linear_ops: tuple[str, ...]
    linear_ops: tuple[str, ...]
    runtime_policy: MTPRuntimePolicy


_TARGET_EMBEDDED_MTP_POLICIES = {
    "norm_policy": {
        "embedding": "rms_norm",
        "hidden_states": "rms_norm",
    },
    "decoder_layer_policy": "reuse_target_decoder_layer",
    "lm_head_policy": "tie_word_embeddings_based",
    "tp_policy": "reuse_target_attn_tp",
}

_TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS = (
    "emb",
    "input_layernorm",
    "post_attention_layernorm",
)

_TARGET_EMBEDDED_MTP_LINEAR_OPS = (
    "mtp_fusion_proj",
    "lm_head_linear",
)

_TARGET_EMBEDDED_MTP_RUNTIME_POLICY: MTPRuntimePolicy = {
    "fusion_op_name": "mtp_fusion_proj",
    "fusion_is_tp_sharded": True,
    "fusion_requires_allgather": True,
    "norm_op_name": "input_layernorm",
    "num_pre_fusion_norms": 2,
    "num_post_decoder_norms": 1,
    "embedding_requires_allreduce": True,
    "lm_head_op_name": "lm_head_linear",
    "lm_head_requires_allgather": True,
}

TARGET_EMBEDDED_MTP_METHOD_REGISTRY: dict[str, TargetEmbeddedMTPMethodContract] = {
    "qwen3_moe_mtp": {
        "mtp_family": "target_embedded_mtp",
        "uses_lookahead_slots": True,
        "requires_prefix_matching_disabled": False,
        "same_tp_linear_ops": _TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS,
        "linear_ops": _TARGET_EMBEDDED_MTP_LINEAR_OPS,
        "runtime_policy": _TARGET_EMBEDDED_MTP_RUNTIME_POLICY,
    },
    "qwen3_next_mtp": {
        "mtp_family": "target_embedded_mtp",
        "uses_lookahead_slots": True,
        "requires_prefix_matching_disabled": True,
        "same_tp_linear_ops": _TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS,
        "linear_ops": _TARGET_EMBEDDED_MTP_LINEAR_OPS,
        "runtime_policy": _TARGET_EMBEDDED_MTP_RUNTIME_POLICY,
    },
}

MTP_MODEL_REGISTRY = {
    "qwen3-a3b-30b-moe": {
        "mtp_family": "target_embedded_mtp",
        **_TARGET_EMBEDDED_MTP_POLICIES,
    },
    "tiny-random/qwen3-next-moe": {
        "mtp_family": "target_embedded_mtp",
        **_TARGET_EMBEDDED_MTP_POLICIES,
    },
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "mtp_family": "target_embedded_mtp",
        **_TARGET_EMBEDDED_MTP_POLICIES,
    },
    "qwen3-next-80b-a3b-instruct-reduced-l2": {
        "mtp_family": "target_embedded_mtp",
        **_TARGET_EMBEDDED_MTP_POLICIES,
    },
    "qwen3-next-80b-a3b-instruct-reduced-l20": {
        "mtp_family": "target_embedded_mtp",
        **_TARGET_EMBEDDED_MTP_POLICIES,
    },
}

MTP_MODEL_ALIAS_MAP = {
    "qwen3-a3b-30b-moe": "qwen3-a3b-30b-moe",
    "Qwen/Qwen3-30B-A3B": "qwen3-a3b-30b-moe",
    "Qwen3-30B-A3B": "qwen3-a3b-30b-moe",
    "tiny-random/qwen3-next-moe": "tiny-random/qwen3-next-moe",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "qwen3-next-80b-a3b-instruct-reduced-l2": "qwen3-next-80b-a3b-instruct-reduced-l2",
    "qwen3-next-80b-a3b-instruct-reduced-l20": "qwen3-next-80b-a3b-instruct-reduced-l20",
    "/tmp/qwen3_next_80b_a3b_instruct_reduced_l2": "qwen3-next-80b-a3b-instruct-reduced-l2",
    "/tmp/qwen3_next_80b_a3b_instruct_reduced_l20": "qwen3-next-80b-a3b-instruct-reduced-l20",
}


def get_mtp_model_alias_map() -> Dict[str, str]:
    return dict(MTP_MODEL_ALIAS_MAP)


def get_registered_mtp_model_contract(
    canonical_model_key: str,
    *,
    mtp_family: str,
) -> Optional[dict[str, object]]:
    entry = MTP_MODEL_REGISTRY.get(str(canonical_model_key))
    if entry is None:
        return None
    if str(entry.get("mtp_family")) != str(mtp_family):
        return None
    return dict(entry)


def get_target_embedded_mtp_methods() -> tuple[str, ...]:
    return tuple(TARGET_EMBEDDED_MTP_METHOD_REGISTRY)


def is_target_embedded_mtp_method(method: str) -> bool:
    return str(method).strip() in TARGET_EMBEDDED_MTP_METHOD_REGISTRY


def get_target_embedded_mtp_method_contract(
    method: str,
) -> TargetEmbeddedMTPMethodContract:
    normalized_method = str(method).strip()
    try:
        return cast(
            TargetEmbeddedMTPMethodContract,
            deepcopy(TARGET_EMBEDDED_MTP_METHOD_REGISTRY[normalized_method]),
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown target-embedded MTP method: {normalized_method!r}"
        ) from exc


def get_target_embedded_mtp_prefix_matching_disabled_methods() -> tuple[str, ...]:
    return tuple(
        method
        for method, contract in TARGET_EMBEDDED_MTP_METHOD_REGISTRY.items()
        if bool(contract["requires_prefix_matching_disabled"])
    )


def get_target_embedded_mtp_same_tp_linear_ops() -> tuple[str, ...]:
    return _TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS


def get_target_embedded_mtp_linear_ops() -> tuple[str, ...]:
    return _TARGET_EMBEDDED_MTP_LINEAR_OPS


def is_target_embedded_mtp_same_tp_linear_op(op_name: str) -> bool:
    return str(op_name) in _TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS


def get_target_embedded_mtp_runtime_policy(method: str) -> MTPRuntimePolicy:
    contract = get_target_embedded_mtp_method_contract(method)
    runtime_policy = contract.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        raise TypeError(
            f"Target-embedded MTP method {method!r} has invalid runtime_policy"
        )
    return cast(MTPRuntimePolicy, deepcopy(runtime_policy))
