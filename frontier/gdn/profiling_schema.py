"""Stable CSV schema helpers for GDN profiling and training."""

from __future__ import annotations

from frontier.gdn.family import GATED_DELTA_NET_FAMILY, GatedDeltaNetFamilySpec
from frontier.types import MeasurementType


_TIME_STAT_SUFFIXES = ("min", "max", "mean", "median", "std", "count")


def get_gdn_profiling_metric_names(
    family: GatedDeltaNetFamilySpec = GATED_DELTA_NET_FAMILY,
) -> tuple[str, ...]:
    return tuple(operator.profiling_name() for operator in family.profiling_ops())


def get_gdn_profiling_time_stat_columns(
    family: GatedDeltaNetFamilySpec = GATED_DELTA_NET_FAMILY,
) -> tuple[str, ...]:
    return tuple(
        f"time_stats.{operator_name}.{suffix}"
        for operator_name in get_gdn_profiling_metric_names(family)
        for suffix in _TIME_STAT_SUFFIXES
    )


def get_required_gdn_profiling_columns(
    family: GatedDeltaNetFamilySpec = GATED_DELTA_NET_FAMILY,
) -> tuple[str, ...]:
    return (
        *family.required_profiling_feature_columns,
        *get_gdn_profiling_time_stat_columns(family),
        # Validation-only direct module timing. It is deliberately absent
        # from the predictor operator family to prevent double counting.
        "time_stats.gdn_layer_e2e.min",
        "time_stats.gdn_layer_e2e.max",
        "time_stats.gdn_layer_e2e.mean",
        "time_stats.gdn_layer_e2e.median",
        "time_stats.gdn_layer_e2e.std",
        "time_stats.gdn_layer_e2e.count",
    )


def validate_gdn_profiling_dataframe(
    dataframe,
    family: GatedDeltaNetFamilySpec = GATED_DELTA_NET_FAMILY,
) -> None:
    missing = [
        column
        for column in get_required_gdn_profiling_columns(family)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "GDN profiling dataframe is missing required columns: "
            f"{missing}"
        )

    for value in dataframe["measurement_type"]:
        MeasurementType.from_string(value)
