from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_MOE_GATING_RUNTIME_CONTEXT = "standalone_legacy"
PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT = "prefill_hot"
# Empirically aligns qwen3_moe prefill-hot gating profiling with live vLLM uniform_topk.
PREFILL_HOT_MOE_GATING_PREFIX_REPEATS = 20

MOE_GATING_RUNTIME_CONTEXT_COLUMN = "gating_runtime_context"
MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN = "gating_runtime_context_impl"

PREFILL_HOT_MOE_GATING_RUNTIME_IMPL = (
    f"ffn_like_prefix_{PREFILL_HOT_MOE_GATING_PREFIX_REPEATS}x"
)
PREFILL_HOT_MOE_GATING_MODEL_SUFFIX = "__prefill_hot"

_SUPPORTED_MOE_GATING_RUNTIME_CONTEXTS = {
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
}


def should_enable_prefill_hot_moe_gating_contract(
    *,
    model_config: Any | None = None,
    model_arch: str | None = None,
    model_type: str | None = None,
    model_name: str | None = None,
) -> bool:
    if model_arch is None and model_config is not None:
        get_model_arch = getattr(model_config, "get_model_arch", None)
        if callable(get_model_arch):
            model_arch = get_model_arch()
        else:
            model_arch = getattr(model_config, "model_arch", None)

    if model_type is None and model_config is not None:
        model_type = getattr(model_config, "model_type", None)

    if model_name is None and model_config is not None:
        get_name = getattr(model_config, "get_name", None)
        if callable(get_name):
            model_name = get_name()
        else:
            model_name = getattr(model_config, "name", None)

    normalized_model_arch = str(model_arch or "").strip().lower()
    if normalized_model_arch == "qwen3_moe":
        return True

    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type == "qwen3_moe":
        return True

    normalized_model_name = str(model_name or "").strip().lower()
    return normalized_model_name == "qwen3-a3b-30b-moe"


def validate_moe_gating_runtime_context(requested_context: str) -> str:
    normalized_context = str(requested_context).strip()
    if normalized_context not in _SUPPORTED_MOE_GATING_RUNTIME_CONTEXTS:
        raise ValueError(
            f"Unsupported gating_runtime_context={requested_context!r}. "
            f"Expected one of {sorted(_SUPPORTED_MOE_GATING_RUNTIME_CONTEXTS)}."
        )
    return normalized_context


def get_moe_gating_runtime_context_metadata(
    requested_context: str,
) -> dict[str, str]:
    normalized_context = validate_moe_gating_runtime_context(requested_context)
    if normalized_context == DEFAULT_MOE_GATING_RUNTIME_CONTEXT:
        return {
            MOE_GATING_RUNTIME_CONTEXT_COLUMN: DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
            MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN: "none",
        }
    return {
        MOE_GATING_RUNTIME_CONTEXT_COLUMN: PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
        MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN: PREFILL_HOT_MOE_GATING_RUNTIME_IMPL,
    }


def has_prefill_hot_moe_gating_rows(df: pd.DataFrame) -> bool:
    """Return whether dataset contains usable prefill-hot gating rows."""
    if MOE_GATING_RUNTIME_CONTEXT_COLUMN not in df.columns:
        return False
    if MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN not in df.columns:
        return False

    context_mask = (
        df[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str)
        == PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
    )
    if not bool(context_mask.any()):
        return False

    impl_mask = (
        df.loc[context_mask, MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN].astype(str)
        == PREFILL_HOT_MOE_GATING_RUNTIME_IMPL
    )
    return bool(impl_mask.any())


def filter_moe_gating_rows_by_runtime_context(
    df: pd.DataFrame,
    *,
    requested_context: str,
    source_name: str,
) -> pd.DataFrame:
    normalized_context = validate_moe_gating_runtime_context(requested_context)
    requested_metadata = get_moe_gating_runtime_context_metadata(normalized_context)

    if MOE_GATING_RUNTIME_CONTEXT_COLUMN not in df.columns:
        raise ValueError(
            "MoE gating runtime-context metadata is missing, "
            f"but runtime requires {MOE_GATING_RUNTIME_CONTEXT_COLUMN}="
            f"{normalized_context!r}. Source: {source_name}"
        )

    filtered_df = df[
        df[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str) == normalized_context
    ].copy()
    if normalized_context != DEFAULT_MOE_GATING_RUNTIME_CONTEXT:
        if MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN not in filtered_df.columns:
            raise ValueError(
                "MoE gating runtime-context impl metadata is missing, "
                f"but runtime requires {MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN}="
                f"{requested_metadata[MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN]!r}. "
                f"Source: {source_name}"
            )
        filtered_df = filtered_df[
            filtered_df[MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN].astype(str)
            == requested_metadata[MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN]
        ].copy()
    if len(filtered_df) == 0:
        available_contexts = sorted(
            df[MOE_GATING_RUNTIME_CONTEXT_COLUMN]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "No MoE gating profiling rows match the requested "
            f"{MOE_GATING_RUNTIME_CONTEXT_COLUMN}={normalized_context!r}. "
            f"Available contexts: {available_contexts}. Source: {source_name}"
        )
    return filtered_df


def get_moe_gating_prediction_model_name(
    base_model_name: str,
    *,
    requested_context: str,
) -> str:
    normalized_context = validate_moe_gating_runtime_context(requested_context)
    if normalized_context == PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT:
        return f"{base_model_name}{PREFILL_HOT_MOE_GATING_MODEL_SUFFIX}"
    return base_model_name


def get_moe_gating_base_model_name(model_name: str) -> str:
    if model_name.endswith(PREFILL_HOT_MOE_GATING_MODEL_SUFFIX):
        return model_name[: -len(PREFILL_HOT_MOE_GATING_MODEL_SUFFIX)]
    return model_name


def should_use_prefill_hot_moe_gating_context(
    *,
    model_arch: str | None = None,
    model_config: Any | None = None,
    model_name: str | None = None,
    batch: Any,
) -> bool:
    if not should_enable_prefill_hot_moe_gating_contract(
        model_config=model_config,
        model_arch=model_arch,
        model_name=model_name,
    ):
        return False

    if bool(getattr(batch, "is_pure_decode_batch", False)):
        return False

    return int(getattr(batch, "num_prefill_tokens", 0)) > 0
