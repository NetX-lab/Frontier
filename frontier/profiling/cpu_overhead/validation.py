"""Validation helpers for CPU overhead profiling outputs."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from frontier.logger import init_logger
from frontier.profiling.cpu_overhead.schema import (
    CPU_OVERHEAD_IDENTITY_COLUMNS,
    CPU_OVERHEAD_NUMERIC_COLUMNS,
    CPU_OVERHEAD_REQUIRED_COLUMNS,
    DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR,
    DEFAULT_NUM_PREFILL_TOKENS,
    DEFAULT_SCHEDULING_MODE,
    VALID_SCHEDULING_MODES,
)

logger = init_logger(__name__)


def _default_warn_fn(message: str) -> None:
    logger.warning(message)


def _legacy_decode_tokens_from_batch_size(df: pd.DataFrame) -> pd.Series:
    if "batch_size" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    batch_size = pd.to_numeric(df["batch_size"], errors="coerce")
    return batch_size * DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR


def apply_cpu_overhead_schema_v2_defaults(
    df: pd.DataFrame,
    *,
    warn_fn: Callable[[str], None] | None = None,
    context: str | None = None,
) -> pd.DataFrame:
    """Apply schema-v2 compatibility defaults for legacy schema-v1 CSV rows."""
    normalized = df.copy()
    applied_defaults: list[str] = []

    if "num_prefill_tokens" not in normalized.columns:
        normalized["num_prefill_tokens"] = DEFAULT_NUM_PREFILL_TOKENS
        applied_defaults.append("num_prefill_tokens")
    else:
        missing_prefill = normalized["num_prefill_tokens"].isna()
        if missing_prefill.any():
            normalized.loc[missing_prefill, "num_prefill_tokens"] = (
                DEFAULT_NUM_PREFILL_TOKENS
            )
            applied_defaults.append("num_prefill_tokens")

    if "num_decode_tokens" not in normalized.columns:
        normalized["num_decode_tokens"] = _legacy_decode_tokens_from_batch_size(normalized)
        applied_defaults.append("num_decode_tokens")
    else:
        decode_series = normalized["num_decode_tokens"]
        missing_decode = decode_series.isna()
        if decode_series.dtype == object:
            missing_decode = missing_decode | decode_series.astype(str).str.strip().eq("")
        if missing_decode.any():
            normalized.loc[missing_decode, "num_decode_tokens"] = (
                _legacy_decode_tokens_from_batch_size(normalized).loc[missing_decode]
            )
            applied_defaults.append("num_decode_tokens")

    if "scheduling_mode" not in normalized.columns:
        normalized["scheduling_mode"] = DEFAULT_SCHEDULING_MODE
        applied_defaults.append("scheduling_mode")
    else:
        scheduling = normalized["scheduling_mode"].astype(str).str.strip()
        missing_mode = scheduling.eq("") | normalized["scheduling_mode"].isna()
        if missing_mode.any():
            normalized.loc[missing_mode, "scheduling_mode"] = DEFAULT_SCHEDULING_MODE
            applied_defaults.append("scheduling_mode")

    if applied_defaults:
        emit_warning = warn_fn or _default_warn_fn
        context_suffix = f" ({context})" if context else ""
        emit_warning(
            "Applying schema v1 compatibility defaults for CPU overhead dataframe"
            f"{context_suffix}: "
            f"num_prefill_tokens={DEFAULT_NUM_PREFILL_TOKENS}, "
            "num_decode_tokens=batch_size*"
            f"{DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR}, "
            f"scheduling_mode={DEFAULT_SCHEDULING_MODE}."
        )

    return normalized


def validate_cpu_overhead_dataframe(
    df: pd.DataFrame, expected_precision: str | None = None
) -> pd.DataFrame:
    """Validate CPU overhead profiling dataframe and return normalized copy."""
    if df.empty:
        raise ValueError("CPU overhead profiling dataframe must not be empty.")

    validated = apply_cpu_overhead_schema_v2_defaults(df)

    missing_columns = [
        column for column in CPU_OVERHEAD_REQUIRED_COLUMNS if column not in validated.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required CPU overhead columns: {missing_columns}. "
            f"Required columns: {list(CPU_OVERHEAD_REQUIRED_COLUMNS)}"
        )

    numeric_frame = validated.loc[:, CPU_OVERHEAD_NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_numeric_columns = [
        column for column in CPU_OVERHEAD_NUMERIC_COLUMNS if numeric_frame[column].isna().any()
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "CPU overhead profiling dataframe has non-numeric values in columns: "
            f"{invalid_numeric_columns}"
        )

    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError("CPU overhead profiling dataframe contains non-finite numeric values.")

    validated.loc[:, CPU_OVERHEAD_NUMERIC_COLUMNS] = numeric_frame

    if (validated["batch_size"] <= 0).any():
        raise ValueError("batch_size must be positive for all CPU overhead rows.")
    if (validated["tensor_parallel_degree"] <= 0).any():
        raise ValueError(
            "tensor_parallel_degree must be positive for all CPU overhead rows."
        )

    if (validated["num_prefill_tokens"] < 0).any():
        raise ValueError("num_prefill_tokens must be non-negative for all CPU overhead rows.")
    if (validated["num_decode_tokens"] < 0).any():
        raise ValueError("num_decode_tokens must be non-negative for all CPU overhead rows.")
    if ((validated["num_prefill_tokens"] + validated["num_decode_tokens"]) <= 0).any():
        raise ValueError(
            "num_prefill_tokens + num_decode_tokens must be > 0 for all CPU overhead rows."
        )

    int_columns = (
        "batch_size",
        "tensor_parallel_degree",
        "num_prefill_tokens",
        "num_decode_tokens",
    )
    for column in int_columns:
        if not (validated[column] == validated[column].astype(int)).all():
            raise ValueError(f"{column} must be an integer for all CPU overhead rows.")
        validated[column] = validated[column].astype(int)

    model_name_series = validated["model_name"].astype(str).str.strip()
    if model_name_series.eq("").any():
        raise ValueError("model_name must be non-empty for all CPU overhead rows.")
    validated["model_name"] = model_name_series

    scheduling_mode_series = validated["scheduling_mode"].astype(str).str.strip().str.lower()
    if scheduling_mode_series.eq("").any():
        raise ValueError("scheduling_mode must be non-empty for all CPU overhead rows.")
    if not scheduling_mode_series.isin(VALID_SCHEDULING_MODES).all():
        invalid_modes = sorted(
            set(scheduling_mode_series[~scheduling_mode_series.isin(VALID_SCHEDULING_MODES)].tolist())
        )
        raise ValueError(
            f"scheduling_mode must be one of {VALID_SCHEDULING_MODES}, got {invalid_modes}."
        )
    validated["scheduling_mode"] = scheduling_mode_series

    precision_series = validated["profiling_precision"].astype(str).str.strip().str.upper()
    if precision_series.eq("").any():
        raise ValueError("profiling_precision must be non-empty for all CPU overhead rows.")

    unique_precisions = precision_series.unique().tolist()
    if len(unique_precisions) != 1:
        raise ValueError(
            "profiling_precision mismatch inside dataframe: "
            f"found multiple values {unique_precisions}"
        )

    actual_precision = unique_precisions[0]
    if expected_precision is not None and actual_precision != expected_precision.upper():
        raise ValueError(
            f"profiling_precision mismatch: expected {expected_precision.upper()} "
            f"but got {actual_precision}"
        )
    validated["profiling_precision"] = actual_precision

    duplicate_rows = validated.duplicated(
        subset=list(CPU_OVERHEAD_IDENTITY_COLUMNS), keep=False
    )
    if duplicate_rows.any():
        duplicated_entries = validated.loc[
            duplicate_rows, list(CPU_OVERHEAD_IDENTITY_COLUMNS)
        ].to_dict("records")
        raise ValueError(
            "Duplicate CPU overhead rows found for identity columns "
            f"{CPU_OVERHEAD_IDENTITY_COLUMNS}: {duplicated_entries}"
        )

    return validated
