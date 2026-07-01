from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.attention.families import get_attention_family


_DSA_MODEL_TYPE_MARKERS = (
    "deepseek_v32",
    "deepseek_v3_2",
    "deepseek_v3.2",
)
_DSA_FIELD_MARKERS = (
    "dsa_topk",
    "dsa_top_k",
    "dsa_index_topk",
    "dsa_indexer",
)
_UNKNOWN_EXOTIC_FIELD_MARKERS = (
    "sliding_window_pattern",
    "dual_chunk_attention",
    "attention_chunk_size",
)
_REQUIRED_MLA_FIELDS = (
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


@dataclass(frozen=True)
class AttentionFamilyBinding:
    """Concrete attention family selected for one model configuration."""

    family_id: str
    variant_id: str
    frozen: bool
    reason: str

    @property
    def family(self):
        return get_attention_family(self.family_id)

    def require_enabled_for_execution(self) -> None:
        self.family.require_enabled_for_execution()


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _has_truthy_attr(config: Any, name: str) -> bool:
    return bool(_get_attr(config, name, None))


def _model_type(config: Any) -> str:
    raw_model_type = _get_attr(config, "model_type", None)
    return str(raw_model_type or "").lower()


def _has_dsa_marker(config: Any) -> bool:
    model_type = _model_type(config)
    if any(marker in model_type for marker in _DSA_MODEL_TYPE_MARKERS):
        return True
    return any(_has_truthy_attr(config, marker) for marker in _DSA_FIELD_MARKERS)


def _unknown_exotic_fields(config: Any) -> list[str]:
    return [
        marker
        for marker in _UNKNOWN_EXOTIC_FIELD_MARKERS
        if _has_truthy_attr(config, marker)
    ]


def _validate_mla_fields(config: Any) -> None:
    missing = [
        field_name
        for field_name in _REQUIRED_MLA_FIELDS
        if _get_attr(config, field_name, None) is None
    ]
    if missing:
        raise ValueError(f"MLA attention binding requires fields: {missing}")


def bind_attention_family(config: Any) -> AttentionFamilyBinding:
    """Bind a runtime or profiling model config to an attention family.

    Dense-FFN and MoE-FFN are intentionally ignored here. This rule engine only
    classifies the attention topology/cache family.
    """
    if _has_dsa_marker(config):
        return AttentionFamilyBinding(
            family_id="dsa_attention",
            variant_id="dsa",
            frozen=True,
            reason="DSA marker detected; truth backend remains frozen",
        )

    if bool(_get_attr(config, "use_mla", False)):
        _validate_mla_fields(config)
        return AttentionFamilyBinding(
            family_id="latent_mla_attention",
            variant_id="mla",
            frozen=False,
            reason="use_mla=True with required latent cache fields",
        )

    exotic_fields = _unknown_exotic_fields(config)
    if exotic_fields:
        raise ValueError(
            "Unrecognized exotic attention fields require an explicit family "
            f"binding before execution: {exotic_fields}"
        )

    num_q_heads = int(_get_attr(config, "num_q_heads"))
    num_kv_heads = int(_get_attr(config, "num_kv_heads"))
    reason = f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"

    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "Attention head counts must be positive: "
            f"{reason}"
        )
    if num_kv_heads == num_q_heads:
        variant_id = "mha"
    elif num_kv_heads == 1:
        variant_id = "mqa"
    elif 1 < num_kv_heads < num_q_heads:
        variant_id = "gqa"
    else:
        raise ValueError(
            "Unsupported attention head topology: "
            f"{reason}"
        )

    return AttentionFamilyBinding(
        family_id="dense_attention",
        variant_id=variant_id,
        frozen=False,
        reason=reason,
    )
