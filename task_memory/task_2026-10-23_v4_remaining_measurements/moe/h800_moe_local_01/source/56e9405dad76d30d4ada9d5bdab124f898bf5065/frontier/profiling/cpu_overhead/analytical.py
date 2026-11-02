"""Analytical TP scaling helpers for CPU overhead profiling."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from frontier.profiling.cpu_overhead.schema import CPU_OVERHEAD_IDENTITY_COLUMNS
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe


def _get_identity_without_tp() -> tuple[str, ...]:
    return tuple(
        column
        for column in CPU_OVERHEAD_IDENTITY_COLUMNS
        if column != "tensor_parallel_degree"
    )


def _pick_base_row_for_target(group_df: pd.DataFrame, target_tp: int) -> pd.Series:
    lower_or_equal = group_df[group_df["tensor_parallel_degree"] <= target_tp]
    if not lower_or_equal.empty:
        return lower_or_equal.sort_values("tensor_parallel_degree").iloc[-1]
    # Fallback to the smallest available TP when all measured TP values are above target.
    return group_df.sort_values("tensor_parallel_degree").iloc[0]


def extrapolate_cpu_overhead_for_missing_tp(
    measured_df: pd.DataFrame,
    target_tp_degrees: Iterable[int],
) -> pd.DataFrame:
    """Generate analytical TP rows for missing degrees and merge with measured rows."""
    if measured_df.empty:
        raise ValueError("measured_df must not be empty for analytical TP extrapolation.")

    validated_measured = validate_cpu_overhead_dataframe(measured_df)
    measured = validated_measured.copy()

    if "cpu_overhead_source" not in measured.columns:
        measured["cpu_overhead_source"] = "measured"
    else:
        measured["cpu_overhead_source"] = measured["cpu_overhead_source"].fillna(
            "measured"
        )

    requested_tp = sorted(set(int(tp) for tp in target_tp_degrees))
    if not requested_tp:
        raise ValueError("target_tp_degrees must contain at least one TP degree.")

    invalid_tp = [tp for tp in requested_tp if tp <= 0]
    if invalid_tp:
        raise ValueError(
            f"All target TP degrees must be positive integers, got {invalid_tp}."
        )

    identity_without_tp = list(_get_identity_without_tp())
    analytical_rows: list[dict] = []

    for _, group_df in measured.groupby(identity_without_tp, sort=False):
        group_df = group_df.sort_values("tensor_parallel_degree")
        existing_tp = {int(value) for value in group_df["tensor_parallel_degree"].tolist()}

        for target_tp in requested_tp:
            if target_tp in existing_tp:
                continue

            base_row = _pick_base_row_for_target(group_df, target_tp)
            base_tp = int(base_row["tensor_parallel_degree"])
            if base_tp <= 0:
                continue

            scale_factor = float(target_tp) / float(base_tp)
            analytical_row = base_row.to_dict()
            analytical_row["tensor_parallel_degree"] = target_tp
            analytical_row["ray_comm_time_mean"] = (
                float(base_row["ray_comm_time_mean"]) * scale_factor
            )
            analytical_row["cpu_overhead_source"] = "analytical_tp_scaling"
            analytical_row["analytical_tp_base_degree"] = base_tp
            analytical_row["analytical_tp_scale_factor"] = scale_factor
            analytical_rows.append(analytical_row)

    if analytical_rows:
        merged = pd.concat([measured, pd.DataFrame(analytical_rows)], ignore_index=True)
    else:
        merged = measured

    merged = validate_cpu_overhead_dataframe(merged)
    return merged.sort_values(list(CPU_OVERHEAD_IDENTITY_COLUMNS)).reset_index(drop=True)
