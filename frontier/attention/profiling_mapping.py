from __future__ import annotations

from dataclasses import dataclass

from frontier.attention.families import (
    LATENT_MLA_ATTENTION_FAMILY,
    iter_attention_families,
)
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
)
from frontier.types import MeasurementType


_TIME_STAT_SUFFIXES = ("min", "max", "mean", "median", "std", "count")


@dataclass(frozen=True)
class AttentionProfilingFeatureSchema:
    """Feature-column schema for one attention family and measurement source.

    CUDA-event and kernel-only profiling currently share the same required
    columns; measurement_type records the validated timing source.
    """

    family_id: str
    memory_layout: AttentionMemoryLayout
    measurement_type: MeasurementType
    required_columns: tuple[str, ...]
    predictor_feature_columns: tuple[str, ...] | None = None


def _normalize_measurement_type(
    measurement_type: str | MeasurementType,
) -> MeasurementType:
    if isinstance(measurement_type, MeasurementType):
        return measurement_type
    return MeasurementType.from_string(measurement_type)


def get_imported_mla_predictor_feature_columns() -> tuple[str, ...]:
    excluded = set(
        LATENT_MLA_ATTENTION_FAMILY.imported_predictor_excluded_feature_columns
    )
    return tuple(
        column
        for column in LATENT_MLA_ATTENTION_FAMILY.required_profiling_feature_columns
        if column not in excluded
    )


def get_attention_profiling_feature_schema(
    family: AttentionFamilySpec,
    *,
    measurement_type: str | MeasurementType = MeasurementType.CUDA_EVENT,
) -> AttentionProfilingFeatureSchema:
    family.require_enabled_for_execution()
    normalized_measurement_type = _normalize_measurement_type(measurement_type)
    required_columns = family.required_profiling_feature_columns
    predictor_feature_columns = (
        get_imported_mla_predictor_feature_columns()
        if family.memory_layout is AttentionMemoryLayout.LATENT_MLA
        else None
    )
    return AttentionProfilingFeatureSchema(
        family_id=family.family_id,
        memory_layout=family.memory_layout,
        measurement_type=normalized_measurement_type,
        required_columns=required_columns,
        predictor_feature_columns=predictor_feature_columns,
    )


def get_profiling_metric_names(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    return tuple(operator.name for operator in family.profiling_ops())


def get_profiling_metric_name_by_role(
    family: AttentionFamilySpec,
    role: AttentionOperatorRole,
) -> str:
    family.require_enabled_for_execution()
    matches = tuple(
        operator.name for operator in family.profiling_ops() if operator.role is role
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one profiling operator for role "
            f"{role.value!r} in attention family {family.family_id!r}; "
            f"found {len(matches)}: {list(matches)}"
        )
    return matches[0]


def get_profiling_time_stat_columns(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    family.require_enabled_for_execution()
    return tuple(
        f"time_stats.{operator_name}.{suffix}"
        for operator_name in get_profiling_metric_names(family)
        for suffix in _TIME_STAT_SUFFIXES
    )


def get_required_profiling_feature_columns(
    family: AttentionFamilySpec,
    *,
    measurement_type: str | MeasurementType = MeasurementType.CUDA_EVENT,
) -> tuple[str, ...]:
    return get_attention_profiling_feature_schema(
        family,
        measurement_type=measurement_type,
    ).required_columns


def get_required_profiling_columns(
    family: AttentionFamilySpec,
    *,
    measurement_type: str | MeasurementType = MeasurementType.CUDA_EVENT,
) -> tuple[str, ...]:
    return (
        *get_required_profiling_feature_columns(
            family,
            measurement_type=measurement_type,
        ),
        *get_profiling_time_stat_columns(family),
    )


def validate_attention_profiling_dataframe(
    df,
    family: AttentionFamilySpec,
    *,
    measurement_type: str | MeasurementType | None = None,
) -> None:
    family.require_enabled_for_execution()
    required_columns = get_required_profiling_columns(
        family,
        measurement_type=(
            measurement_type
            if measurement_type is not None
            else MeasurementType.CUDA_EVENT
        ),
    )
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Attention profiling dataframe is missing required attention "
            f"profiling columns for {family.family_id!r}: {missing_columns}"
        )
    if "measurement_type" not in df.columns:
        raise ValueError("Attention profiling dataframe is missing measurement_type")

    observed_measurement_types = {
        MeasurementType.from_string(value).value for value in df["measurement_type"]
    }
    if measurement_type is not None:
        expected = (
            measurement_type
            if isinstance(measurement_type, MeasurementType)
            else MeasurementType.from_string(measurement_type)
        )
        if observed_measurement_types != {expected.value}:
            raise ValueError(
                "Attention profiling dataframe measurement_type mismatch: "
                f"expected {expected.value}, got {sorted(observed_measurement_types)}"
            )


def get_e2e_metric_names(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    return tuple(operator.name for operator in family.e2e_trace_ops())


def get_enabled_predictor_metric_names(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    family.require_enabled_for_execution()
    return tuple(operator.name for operator in family.predictor_ops())


def get_enabled_predictor_metric_name_by_role(
    family: AttentionFamilySpec,
    role: AttentionOperatorRole,
) -> str:
    family.require_enabled_for_execution()
    matches = tuple(
        operator.name for operator in family.predictor_ops() if operator.role is role
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one predictor operator for role "
            f"{role.value!r} in attention family {family.family_id!r}; "
            f"found {len(matches)}: {list(matches)}"
        )
    return matches[0]


def get_enabled_predictor_median_column_by_role(
    family: AttentionFamilySpec,
    role: AttentionOperatorRole,
) -> str:
    operator_name = get_enabled_predictor_metric_name_by_role(family, role)
    return f"time_stats.{operator_name}.median"


def get_enabled_predictor_median_columns(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    return tuple(
        f"time_stats.{name}.median"
        for name in get_enabled_predictor_metric_names(family)
    )


def get_enabled_predictor_feature_columns(
    family: AttentionFamilySpec,
) -> dict[str, tuple[str, ...]]:
    family.require_enabled_for_execution()
    if family.memory_layout is AttentionMemoryLayout.DENSE_KV:
        feature_columns_by_role = {
            AttentionOperatorRole.CACHE_WRITE: ("num_tokens",),
            AttentionOperatorRole.PREFILL_KERNEL: (
                "kv_cache_size",
                "prefill_chunk_size_squared",
            ),
            AttentionOperatorRole.DECODE_KERNEL: ("batch_size", "kv_cache_size"),
        }
        feature_columns = {
            operator.name: feature_columns_by_role[operator.role]
            for operator in family.predictor_ops()
            if operator.role in feature_columns_by_role
        }
    elif family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
        imported_predictor_feature_columns = get_imported_mla_predictor_feature_columns()
        feature_columns = {
            operator.name: imported_predictor_feature_columns
            for operator in family.predictor_ops()
        }
    else:
        raise ValueError(
            f"Unsupported attention predictor memory layout: "
            f"{family.memory_layout.value}"
        )
    predictor_names = get_enabled_predictor_metric_names(family)
    missing_features = sorted(set(predictor_names) - feature_columns.keys())
    if missing_features:
        raise ValueError(
            "Predictor feature columns are missing attention operators: "
            f"{missing_features}"
        )
    return {name: feature_columns[name] for name in predictor_names}


def get_enabled_shared_predictor_feature_columns(
    family: AttentionFamilySpec,
) -> dict[str, tuple[str, ...]]:
    family.require_enabled_for_execution()
    if family.memory_layout is AttentionMemoryLayout.DENSE_KV:
        feature_columns_by_role = {
            AttentionOperatorRole.CACHE_WRITE: (
                "total_tokens",
                "kv_cache_size",
                "batch_size",
            ),
            AttentionOperatorRole.PREFILL_KERNEL: (
                "kv_cache_size",
                "prefill_chunk_size_squared",
            ),
            AttentionOperatorRole.DECODE_KERNEL: ("batch_size", "kv_cache_size"),
        }
        feature_columns = {
            operator.name: feature_columns_by_role[operator.role]
            for operator in family.predictor_ops()
            if operator.role in feature_columns_by_role
        }
    elif family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
        imported_predictor_feature_columns = get_imported_mla_predictor_feature_columns()
        feature_columns = {
            operator.name: imported_predictor_feature_columns
            for operator in family.predictor_ops()
        }
    else:
        raise ValueError(
            f"Unsupported shared attention predictor memory layout: "
            f"{family.memory_layout.value}"
        )
    predictor_names = get_enabled_predictor_metric_names(family)
    missing_features = sorted(set(predictor_names) - feature_columns.keys())
    if missing_features:
        raise ValueError(
            "Shared predictor feature columns are missing attention operators: "
            f"{missing_features}"
        )
    return {name: feature_columns[name] for name in predictor_names}


def get_enabled_predictor_required_feature_columns(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    feature_columns = get_enabled_predictor_feature_columns(family)
    required_columns: dict[str, None] = {}
    for columns in feature_columns.values():
        for column in columns:
            required_columns.setdefault(column, None)
    return tuple(required_columns)


def get_enabled_shared_predictor_required_feature_columns(
    family: AttentionFamilySpec,
) -> tuple[str, ...]:
    feature_columns = get_enabled_shared_predictor_feature_columns(family)
    required_columns: dict[str, None] = {}
    for columns in feature_columns.values():
        for column in columns:
            required_columns.setdefault(column, None)
    return tuple(required_columns)


def validate_attention_catalog_alignment(
    *,
    profiling_metric_values: set[str],
    e2e_metric_values: set[str],
) -> None:
    """Validate that existing metric enums cover all enabled family operators."""
    profiling_required: set[str] = set()
    e2e_required: set[str] = set()
    for family in iter_attention_families():
        profiling_required.update(get_profiling_metric_names(family))
        e2e_required.update(get_e2e_metric_names(family))

    missing_profiling = sorted(profiling_required - profiling_metric_values)
    if missing_profiling:
        raise ValueError(
            "Profiling metrics catalog is missing attention operators: "
            f"{missing_profiling}"
        )

    missing_e2e = sorted(e2e_required - e2e_metric_values)
    if missing_e2e:
        raise ValueError(
            "E2E metrics catalog is missing attention operators: "
            f"{missing_e2e}"
        )
