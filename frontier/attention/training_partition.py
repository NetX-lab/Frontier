"""Shared dense-attention training row partitioning.

Standalone training and the E2E model manager must select identical rows for
the same attention model.  Keeping this policy in one small, dependency-light
module prevents cache identity drift between producers.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from frontier.attention.string_coercion import coerce_truthy_bool


CACHE_WRITE_FEATURE_COLUMNS = (
    "total_tokens",
    "kv_cache_size",
    "batch_size",
)


def validate_cache_write_target_consistency(
    df: pd.DataFrame,
    *,
    target_col: str,
    tolerance: float = 1e-9,
    allow_repeated_measurements: bool = False,
) -> None:
    """Validate targets after cache-write feature normalization.

    A strict caller rejects multiple targets for one normalized key.  Real
    profiling CSVs may intentionally contain repeated timing samples for the
    same shape (for example thousands of one-token prefill samples); those
    rows are valid training observations and may opt in explicitly through
    ``allow_repeated_measurements``.  The opt-in does not aggregate, clamp, or
    rewrite any target value.
    """

    missing = [column for column in (*CACHE_WRITE_FEATURE_COLUMNS, target_col) if column not in df.columns]
    if missing:
        raise ValueError(
            "Cache-write target consistency requires columns "
            f"{missing!r}."
        )
    numeric = df.loc[:, list(CACHE_WRITE_FEATURE_COLUMNS) + [target_col]].copy()
    numeric[target_col] = pd.to_numeric(numeric[target_col], errors="raise")
    if isinstance(allow_repeated_measurements, bool) is False:
        raise ValueError(
            "allow_repeated_measurements must be a boolean, got "
            f"{allow_repeated_measurements!r}."
        )
    if allow_repeated_measurements:
        return
    grouped = numeric.groupby(list(CACHE_WRITE_FEATURE_COLUMNS), dropna=False)[target_col]
    conflicts = grouped.agg(lambda values: float(values.max()) - float(values.min()))
    conflicts = conflicts[conflicts > float(tolerance)]
    if not conflicts.empty:
        examples = [tuple(index) if isinstance(index, tuple) else (index,) for index in conflicts.index[:3]]
        raise ValueError(
            "Cache-write training rows contain conflicting targets for normalized "
            f"keys; examples={examples!r}."
        )

DENSE_MIXED_PREFILL_FEATURES = (
    "avg_seq_len",
    "batch_cv_interaction",
    "batch_size",
    "batch_variance_interaction",
    "kv_cache_size",
    "max_seq_len",
    "min_seq_len",
    "seq_len_cv",
    "seq_len_range",
    "seq_len_variance",
    "total_tokens",
    "total_tokens_squared",
)

_TRUE_MIXED_PREFILL_FEATURE_MAP = {
    feature: f"prefill_mixed_{feature}"
    for feature in DENSE_MIXED_PREFILL_FEATURES
}


def prepare_dense_cache_write_training_rows(
    df: pd.DataFrame,
    *,
    dataset_path: str | None = None,
) -> pd.DataFrame:
    """Normalize cache-write features to the runtime batch contract.

    Dense true-mixed profiler rows retain compatibility values
    ``kv_cache_size=0`` and ``batch_size=total_batch_size``.  The runtime cache
    write operator instead uses the average KV context of decode sequences when
    a decode side exists.  Training must replace the compatibility context with
    that runtime feature before model fitting, hashing, or domain construction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Dense cache-write training rows must be a DataFrame, "
            f"got {type(df).__name__}."
        )
    missing = [column for column in CACHE_WRITE_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Dense cache-write training rows are missing required columns "
            f"{missing}; dataset={dataset_path!r}."
        )

    output = df.copy()
    if "is_true_mixed_batch" in output.columns:
        true_mixed = coerce_truthy_bool(output["is_true_mixed_batch"])
    else:
        true_mixed = pd.Series(False, index=output.index)

    if bool(true_mixed.any()):
        source_columns = ("decode_avg_kv_cache_size", "total_batch_size")
        missing_source = [column for column in source_columns if column not in output.columns]
        if missing_source:
            raise ValueError(
                "Dense cache-write true-mixed rows require runtime context columns "
                f"{list(source_columns)}; missing {missing_source}; "
                f"dataset={dataset_path!r}."
            )
        source_values = output.loc[true_mixed, list(source_columns)]
        if source_values.isna().any(axis=None):
            raise ValueError(
                "Dense cache-write true-mixed rows contain missing "
                "decode_avg_kv_cache_size or total_batch_size values; "
                f"dataset={dataset_path!r}."
            )
        decode_kv = pd.to_numeric(
            source_values["decode_avg_kv_cache_size"], errors="raise"
        )
        total_batch = pd.to_numeric(
            source_values["total_batch_size"], errors="raise"
        )
        if not np.isfinite(decode_kv.to_numpy(dtype=float)).all() or (decode_kv < 0).any():
            raise ValueError(
                "Dense cache-write true-mixed decode_avg_kv_cache_size must be "
                "finite and non-negative."
            )
        if not np.isfinite(total_batch.to_numpy(dtype=float)).all() or (total_batch <= 0).any():
            raise ValueError(
                "Dense cache-write true-mixed total_batch_size must be finite and positive."
            )
        if not np.equal(total_batch.to_numpy(dtype=float), np.floor(total_batch.to_numpy(dtype=float))).all():
            raise ValueError(
                "Dense cache-write true-mixed total_batch_size must contain integers."
            )
        output.loc[true_mixed, "kv_cache_size"] = decode_kv.to_numpy()
        output.loc[true_mixed, "batch_size"] = total_batch.to_numpy()

    # Runtime cache-write features use decode-side KV context only.  A pure
    # prefill batch has no decode sequences, so its key is always context zero
    # even when the profiled request itself resumes from a non-zero prefix.
    # Normalize the training rows at the same semantic boundary.
    if "is_prefill" in output.columns:
        is_prefill = coerce_truthy_bool(output["is_prefill"])
        pure_prefill = is_prefill & ~true_mixed
        output.loc[pure_prefill, "kv_cache_size"] = 0

    numeric = output.loc[:, list(CACHE_WRITE_FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Dense cache-write training features must be finite and non-missing; "
            f"dataset={dataset_path!r}."
        )
    if (numeric["kv_cache_size"] < 0).any():
        raise ValueError("Dense cache-write kv_cache_size must be non-negative.")
    for column in ("total_tokens", "kv_cache_size"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"Dense cache-write {column} must contain integers."
            )
    if (numeric["batch_size"] <= 0).any():
        raise ValueError("Dense cache-write batch_size must be positive.")
    if not np.equal(
        numeric["batch_size"].to_numpy(dtype=float),
        np.floor(numeric["batch_size"].to_numpy(dtype=float)),
    ).all():
        raise ValueError("Dense cache-write batch_size must contain integers.")
    if (numeric["total_tokens"] < numeric["batch_size"]).any():
        raise ValueError(
            "Dense cache-write total_tokens must be at least batch_size for every row."
        )
    for column in CACHE_WRITE_FEATURE_COLUMNS:
        output[column] = numeric[column]
    return output


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a profile cell is missing without coercing arrays to bool."""

    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        # List-valued cells are handled by the explicit list-column parser.  They
        # are not valid scalar feature values and must not be passed to
        # ``pd.isna`` as that returns an array for them.
        return len(value) == 0
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _coerce_feature_value(value: Any, *, feature_name: str) -> float | int:
    """Normalize one scalar mixed-prefill feature value."""

    if _is_missing_scalar(value):
        raise ValueError(f"Missing mixed-prefill feature {feature_name!r}.")
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError(
            f"Mixed-prefill feature {feature_name!r} must be scalar, got {value!r}."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Mixed-prefill feature {feature_name!r} must be numeric, got {value!r}."
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"Mixed-prefill feature {feature_name!r} must be finite, got {value!r}."
        )
    return int(numeric) if numeric.is_integer() else numeric


def partition_dense_attention_rows(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Partition dense attention rows into the canonical training subsets.

    ``standard`` excludes mixed-prefill and true-mixed rows.  The dedicated
    mixed subsets remain available for their explicit on-demand models, while
    ``all`` is used for cache-write training because that operator occurs for
    every batch composition.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Dense attention training rows must be a DataFrame, got {type(df).__name__}."
        )
    required = {"is_decode"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Dense attention training rows are missing required columns: "
            f"{missing!r}"
        )

    is_decode = df["is_decode"].fillna(False).astype(bool)
    if "is_true_mixed_batch" in df.columns:
        true_mixed = df["is_true_mixed_batch"].fillna(False).astype(bool)
    else:
        true_mixed = pd.Series(False, index=df.index)
    if "is_mixed_batch" in df.columns:
        mixed_batch = df["is_mixed_batch"].fillna(False).astype(bool)
    else:
        if "batch_size" not in df.columns:
            raise ValueError(
                "Dense attention training rows require batch_size when "
                "is_mixed_batch metadata is absent."
            )
        mixed_batch = (~is_decode) & (df["batch_size"] > 1)

    mixed_prefill = mixed_batch & ~true_mixed
    standard = df[~mixed_prefill & ~true_mixed].copy()
    mixed_prefill_df = df[mixed_prefill].copy()
    true_mixed_df = df[true_mixed].copy()

    return {
        "all": df.copy(),
        "standard": standard,
        "mixed_prefill": mixed_prefill_df,
        "true_mixed": true_mixed_df,
        "prefill": standard[~is_decode.loc[standard.index]].copy(),
        "decode": standard[is_decode.loc[standard.index]].copy(),
    }


def _parse_numeric_list(value: Any, *, column_name: str) -> list[float]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "<na>"}:
            return []
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid true-mixed attention list column {column_name}: {value!r}."
            ) from exc
    if isinstance(value, pd.Series):
        value = value.tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value if not pd.isna(item)]
    if pd.isna(value):
        return []
    return [float(value)]


def _derive_true_mixed_prefill_features(
    row: pd.Series,
    *,
    kv_cache_prediction_granularity: int,
) -> dict[str, float | int] | None:
    # Raw list columns are the profiler's source of truth.  Prefer them when a
    # row contains either list, even if a stale/generated prefixed overlay is
    # also present.  This keeps standalone and E2E producers deterministic.
    has_raw_seq = "prefill_seq_lens" in row.index and not _is_missing_scalar(
        row["prefill_seq_lens"]
    )
    has_raw_kv = "prefill_kv_cache_sizes" in row.index and not _is_missing_scalar(
        row["prefill_kv_cache_sizes"]
    )
    if has_raw_seq or has_raw_kv:
        if not (has_raw_seq and has_raw_kv):
            raise ValueError(
                "True-mixed attention row has only one of prefill_seq_lens and "
                "prefill_kv_cache_sizes."
            )
        seq_lens = _parse_numeric_list(
            row["prefill_seq_lens"], column_name="prefill_seq_lens"
        )
        kv_sizes = _parse_numeric_list(
            row["prefill_kv_cache_sizes"], column_name="prefill_kv_cache_sizes"
        )
        if not seq_lens:
            raise ValueError("True-mixed attention row has no prefill_seq_lens.")
        if len(seq_lens) != len(kv_sizes):
            raise ValueError(
                "True-mixed attention row has mismatched prefill_seq_lens and "
                f"prefill_kv_cache_sizes lengths: {len(seq_lens)} vs {len(kv_sizes)}."
            )
        seq_array = np.asarray(seq_lens, dtype=np.float64)
        batch_size = len(seq_lens)
        total_tokens = int(seq_array.sum())
        avg_seq_len = float(seq_array.mean())
        variance = float(seq_array.var()) if batch_size > 1 else 0.0
        std = math.sqrt(variance)
        cv = std / avg_seq_len if avg_seq_len > 0 else 0.0
        min_seq_len = int(seq_array.min())
        max_seq_len = int(seq_array.max())
        avg_kv = int(np.mean(kv_sizes))
        rounded_kv = (
            (avg_kv + kv_cache_prediction_granularity - 1)
            // kv_cache_prediction_granularity
        ) * kv_cache_prediction_granularity
        return {
            "batch_size": batch_size,
            "kv_cache_size": rounded_kv,
            "total_tokens": total_tokens,
            "avg_seq_len": avg_seq_len,
            "min_seq_len": min_seq_len,
            "max_seq_len": max_seq_len,
            "total_tokens_squared": total_tokens**2,
            "seq_len_variance": variance,
            "seq_len_cv": cv,
            "seq_len_range": max_seq_len - min_seq_len,
            "batch_variance_interaction": batch_size * variance,
            "batch_cv_interaction": batch_size * cv,
        }

    prefixed_sources = tuple(_TRUE_MIXED_PREFILL_FEATURE_MAP.values())
    prefixed_values = {
        feature: row.get(source)
        for feature, source in _TRUE_MIXED_PREFILL_FEATURE_MAP.items()
    }
    prefixed_present = [
        source
        for source in prefixed_sources
        if source in row.index and not _is_missing_scalar(row[source])
    ]
    if prefixed_present and len(prefixed_present) != len(prefixed_sources):
        missing = [
            source for source in prefixed_sources if source not in prefixed_present
        ]
        raise ValueError(
            "True-mixed attention row has an incomplete prefill-side feature "
            f"contract; missing {missing!r}."
        )
    if len(prefixed_present) == len(prefixed_sources):
        return {
            feature: _coerce_feature_value(value, feature_name=feature)
            for feature, value in prefixed_values.items()
        }

    direct = {feature: row.get(feature) for feature in DENSE_MIXED_PREFILL_FEATURES}
    if all(not _is_missing_scalar(value) for value in direct.values()):
        return {
            feature: _coerce_feature_value(value, feature_name=feature)
            for feature, value in direct.items()
        }
    # Legacy true-mixed rows often carry only generic batch columns (and
    # derived overlays such as ``total_tokens_squared``).  Those columns do
    # not prove that the prefill-side representation is complete.  Skip this
    # optional prefill training row and retain the row for decode-in-mixed
    # training instead.  Explicit raw/prefixed metadata remains fail-fast.
    return None


def build_dense_mixed_prefill_training_rows(
    partitions: Mapping[str, pd.DataFrame],
    *,
    kv_cache_prediction_granularity: int,
    target_col: str,
    dataset_path: str | None = None,
) -> pd.DataFrame:
    """Build the canonical mixed-prefill training frame for every producer."""

    if (
        isinstance(kv_cache_prediction_granularity, bool)
        or not isinstance(kv_cache_prediction_granularity, (int, np.integer))
        or int(kv_cache_prediction_granularity) <= 0
    ):
        raise ValueError(
            "kv_cache_prediction_granularity must be a positive integer, got "
            f"{kv_cache_prediction_granularity}."
        )
    kv_cache_prediction_granularity = int(kv_cache_prediction_granularity)
    for name in ("mixed_prefill", "true_mixed"):
        if name not in partitions or not isinstance(partitions[name], pd.DataFrame):
            raise ValueError(f"Dense attention partition {name!r} is missing.")

    sources: list[pd.DataFrame] = []
    mixed = partitions["mixed_prefill"].copy()
    if not mixed.empty:
        required = [*DENSE_MIXED_PREFILL_FEATURES, target_col]
        missing = [column for column in required if column not in mixed.columns]
        if missing:
            raise ValueError(
                "Model attn_prefill_mixed has profiling rows but is missing "
                f"required columns {missing}; dataset={dataset_path!r}. Re-run "
                "attention profiling with the current mixed-prefill schema."
            )
        valid = mixed[required].notna().all(axis=1)
        if not bool(valid.any()):
            raise ValueError(
                "Model attn_prefill_mixed has profiling rows but no complete "
                f"feature/target rows; dataset={dataset_path!r}. Re-run attention "
                "profiling with the current mixed-prefill schema."
            )
        sources.append(mixed.loc[valid].copy())

    true_mixed_rows: list[pd.Series] = []
    true_mixed = partitions["true_mixed"].copy()
    if not true_mixed.empty and target_col in true_mixed.columns:
        for _, row in true_mixed[true_mixed[target_col].notna()].iterrows():
            features = _derive_true_mixed_prefill_features(
                row,
                kv_cache_prediction_granularity=kv_cache_prediction_granularity,
            )
            if features is None:
                continue
            output = row.copy()
            for feature, value in features.items():
                output[feature] = value
            true_mixed_rows.append(output)
    if true_mixed_rows:
        sources.append(pd.DataFrame(true_mixed_rows))

    if not sources:
        return pd.DataFrame(columns=[*DENSE_MIXED_PREFILL_FEATURES, target_col])
    combined = pd.concat(sources, ignore_index=True, sort=False)
    required = [*DENSE_MIXED_PREFILL_FEATURES, target_col]
    if combined[required].isna().any(axis=None):
        raise ValueError("Mixed-prefill training rows contain incomplete features.")
    return combined
