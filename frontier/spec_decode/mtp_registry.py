from __future__ import annotations

from typing import Dict, Optional


_TARGET_EMBEDDED_MTP_POLICIES = {
    "norm_policy": {
        "embedding": "rms_norm",
        "hidden_states": "rms_norm",
    },
    "decoder_layer_policy": "reuse_target_decoder_layer",
    "lm_head_policy": "tie_word_embeddings_based",
    "tp_policy": "reuse_target_attn_tp",
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
