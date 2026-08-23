from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

DIRECT_MOE_GATING_RUNTIME_CONTEXT = "direct"
PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT = "prefill_warmed"
LEGACY_STANDALONE_MOE_GATING_RUNTIME_CONTEXT = "standalone_legacy"
LEGACY_PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT = "prefill_hot"

# Empirically aligns qwen3_moe prefill-warmed gating profiling with live vLLM
# uniform_topk.
PREFILL_WARMED_MOE_GATING_PREFIX_REPEATS = 20

MOE_GATING_RUNTIME_CONTEXT_COLUMN = "gating_runtime_context"
MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN = "gating_runtime_context_impl"

PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL = (
    f"ffn_like_prefix_{PREFILL_WARMED_MOE_GATING_PREFIX_REPEATS}x"
)
PREFILL_WARMED_MOE_GATING_MODEL_SUFFIX = "__prefill_warmed"
LEGACY_PREFILL_HOT_MOE_GATING_MODEL_SUFFIX = "__prefill_hot"


@dataclass(frozen=True)
class _MoeGatingRuntimeContextSpec:
    value: str
    implementation: str
    prediction_model_suffix: str
    legacy_values: tuple[str, ...] = ()
    legacy_prediction_model_suffixes: tuple[str, ...] = ()


_MOE_GATING_RUNTIME_CONTEXT_REGISTRY = {
    DIRECT_MOE_GATING_RUNTIME_CONTEXT: _MoeGatingRuntimeContextSpec(
        value=DIRECT_MOE_GATING_RUNTIME_CONTEXT,
        implementation="none",
        prediction_model_suffix="",
        legacy_values=(LEGACY_STANDALONE_MOE_GATING_RUNTIME_CONTEXT,),
    ),
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT: _MoeGatingRuntimeContextSpec(
        value=PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
        implementation=PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL,
        prediction_model_suffix=PREFILL_WARMED_MOE_GATING_MODEL_SUFFIX,
        legacy_values=(LEGACY_PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,),
        legacy_prediction_model_suffixes=(
            LEGACY_PREFILL_HOT_MOE_GATING_MODEL_SUFFIX,
        ),
    ),
}

_MOE_GATING_RUNTIME_CONTEXT_ALIASES = {
    legacy_value: spec.value
    for spec in _MOE_GATING_RUNTIME_CONTEXT_REGISTRY.values()
    for legacy_value in spec.legacy_values
}

# A later release will remove both legacy values: standalone_legacy and
# prefill_hot.
DEFAULT_MOE_GATING_RUNTIME_CONTEXT = DIRECT_MOE_GATING_RUNTIME_CONTEXT
PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT = (
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT
)
PREFILL_HOT_MOE_GATING_PREFIX_REPEATS = (
    PREFILL_WARMED_MOE_GATING_PREFIX_REPEATS
)
PREFILL_HOT_MOE_GATING_RUNTIME_IMPL = PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL
PREFILL_HOT_MOE_GATING_MODEL_SUFFIX = PREFILL_WARMED_MOE_GATING_MODEL_SUFFIX


def should_enable_prefill_warmed_moe_gating_contract(
    *,
    model_config: Any | None = None,
    model_arch: str | None = None,
    model_type: str | None = None,
    model_name: str | None = None,
) -> bool:
    resolved_model_arch: Any = model_arch
    resolved_model_type: Any = model_type
    resolved_model_name: Any = model_name

    if resolved_model_arch is None and model_config is not None:
        get_model_arch = getattr(model_config, "get_model_arch", None)
        if callable(get_model_arch):
            resolved_model_arch = get_model_arch()
        else:
            resolved_model_arch = getattr(model_config, "model_arch", None)

    if resolved_model_type is None and model_config is not None:
        resolved_model_type = getattr(model_config, "model_type", None)

    if resolved_model_name is None and model_config is not None:
        get_name = getattr(model_config, "get_name", None)
        if callable(get_name):
            resolved_model_name = get_name()
        else:
            resolved_model_name = getattr(model_config, "name", None)

    normalized_model_arch = str(resolved_model_arch or "").strip().lower()
    if normalized_model_arch == "qwen3_moe":
        return True

    normalized_model_type = str(resolved_model_type or "").strip().lower()
    if normalized_model_type == "qwen3_moe":
        return True

    normalized_model_name = str(resolved_model_name or "").strip().lower()
    return normalized_model_name == "qwen3-a3b-30b-moe"


def get_supported_moe_gating_runtime_context_values() -> tuple[str, ...]:
    return (
        *tuple(_MOE_GATING_RUNTIME_CONTEXT_REGISTRY),
        *tuple(_MOE_GATING_RUNTIME_CONTEXT_ALIASES),
    )


def normalize_moe_gating_runtime_context(requested_context: str) -> str:
    normalized_context = str(requested_context).strip()
    if normalized_context in _MOE_GATING_RUNTIME_CONTEXT_REGISTRY:
        return normalized_context

    canonical_context = _MOE_GATING_RUNTIME_CONTEXT_ALIASES.get(normalized_context)
    if canonical_context is not None:
        warnings.warn(
            f"gating_runtime_context={normalized_context!r} is a legacy alias "
            "and will be removed in a future release; "
            f"use {canonical_context!r} instead.",
            FutureWarning,
            stacklevel=2,
        )
        return canonical_context

    raise ValueError(
        f"Unsupported gating_runtime_context={requested_context!r}. "
        "Expected one of "
        f"{sorted(get_supported_moe_gating_runtime_context_values())}."
    )


def validate_moe_gating_runtime_context(requested_context: str) -> str:
    return normalize_moe_gating_runtime_context(requested_context)


def get_moe_gating_runtime_context_metadata(
    requested_context: str,
) -> dict[str, str]:
    normalized_context = normalize_moe_gating_runtime_context(requested_context)
    context_spec = _MOE_GATING_RUNTIME_CONTEXT_REGISTRY[normalized_context]
    return {
        MOE_GATING_RUNTIME_CONTEXT_COLUMN: context_spec.value,
        MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN: context_spec.implementation,
    }


def _normalize_moe_gating_runtime_context_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_values: dict[str, str] = {}
    for raw_value in (
        normalized_df[MOE_GATING_RUNTIME_CONTEXT_COLUMN]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ):
        normalized_values[raw_value] = normalize_moe_gating_runtime_context(raw_value)
    normalized_df[MOE_GATING_RUNTIME_CONTEXT_COLUMN] = normalized_df[
        MOE_GATING_RUNTIME_CONTEXT_COLUMN
    ].map(
        lambda value: (
            value
            if pd.isna(value)
            else normalized_values[str(value)]
        )
    )
    return normalized_df


def has_prefill_warmed_moe_gating_rows(df: pd.DataFrame) -> bool:
    """Return whether a dataset contains usable prefill-warmed gating rows."""
    if MOE_GATING_RUNTIME_CONTEXT_COLUMN not in df.columns:
        return False
    if MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN not in df.columns:
        return False

    normalized_df = _normalize_moe_gating_runtime_context_column(df)
    context_mask = (
        normalized_df[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str)
        == PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT
    )
    if not bool(context_mask.any()):
        return False

    impl_mask = (
        normalized_df.loc[
            context_mask,
            MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN,
        ].astype(str)
        == PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL
    )
    return bool(impl_mask.any())


def filter_moe_gating_rows_by_runtime_context(
    df: pd.DataFrame,
    *,
    requested_context: str,
    source_name: str,
) -> pd.DataFrame:
    normalized_context = normalize_moe_gating_runtime_context(requested_context)
    requested_metadata = get_moe_gating_runtime_context_metadata(normalized_context)

    if MOE_GATING_RUNTIME_CONTEXT_COLUMN not in df.columns:
        raise ValueError(
            "MoE gating runtime-context metadata is missing, "
            f"but runtime requires {MOE_GATING_RUNTIME_CONTEXT_COLUMN}="
            f"{normalized_context!r}. Source: {source_name}"
        )

    normalized_df = _normalize_moe_gating_runtime_context_column(df)
    filtered_df = normalized_df[
        normalized_df[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str)
        == normalized_context
    ].copy()
    if normalized_context != DIRECT_MOE_GATING_RUNTIME_CONTEXT:
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
    normalized_context = normalize_moe_gating_runtime_context(requested_context)
    context_spec = _MOE_GATING_RUNTIME_CONTEXT_REGISTRY[normalized_context]
    return f"{base_model_name}{context_spec.prediction_model_suffix}"


def get_moe_gating_base_model_name(model_name: str) -> str:
    for context_spec in _MOE_GATING_RUNTIME_CONTEXT_REGISTRY.values():
        if (
            context_spec.prediction_model_suffix
            and model_name.endswith(context_spec.prediction_model_suffix)
        ):
            return model_name[: -len(context_spec.prediction_model_suffix)]
        for legacy_suffix in context_spec.legacy_prediction_model_suffixes:
            if model_name.endswith(legacy_suffix):
                warnings.warn(
                    f"MoE gating prediction-model suffix {legacy_suffix!r} is "
                    "a legacy alias and will be removed in a future release; "
                    f"use {context_spec.prediction_model_suffix!r} instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                return model_name[: -len(legacy_suffix)]
    return model_name


def get_moe_gating_prediction_model_context(model_name: str) -> str:
    for context_spec in _MOE_GATING_RUNTIME_CONTEXT_REGISTRY.values():
        if (
            context_spec.prediction_model_suffix
            and model_name.endswith(context_spec.prediction_model_suffix)
        ):
            return context_spec.value
        for legacy_suffix in context_spec.legacy_prediction_model_suffixes:
            if model_name.endswith(legacy_suffix):
                warnings.warn(
                    f"MoE gating prediction-model suffix {legacy_suffix!r} is "
                    "a legacy alias and will be removed in a future release; "
                    f"use {context_spec.prediction_model_suffix!r} instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                return context_spec.value
    return DIRECT_MOE_GATING_RUNTIME_CONTEXT


def should_use_prefill_warmed_moe_gating_context(
    *,
    model_arch: str | None = None,
    model_config: Any | None = None,
    model_name: str | None = None,
    batch: Any,
) -> bool:
    if not should_enable_prefill_warmed_moe_gating_contract(
        model_config=model_config,
        model_arch=model_arch,
        model_name=model_name,
    ):
        return False

    if bool(getattr(batch, "is_pure_decode_batch", False)):
        return False

    return int(getattr(batch, "num_prefill_tokens", 0)) > 0


# Temporary source-level aliases keep existing imports working while all new
# call sites use the canonical prefill-warmed names.
should_enable_prefill_hot_moe_gating_contract = (
    should_enable_prefill_warmed_moe_gating_contract
)
has_prefill_hot_moe_gating_rows = has_prefill_warmed_moe_gating_rows
should_use_prefill_hot_moe_gating_context = (
    should_use_prefill_warmed_moe_gating_context
)
