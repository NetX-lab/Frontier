from __future__ import annotations

from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionOperatorSpec,
    ProjectionOwnership,
)
from frontier.entities.execution_time import ExecutionTime


_MODEL_PROJECTION_EXECUTION_TIME_ATTRS = (
    "attention_layer_pre_proj_execution_time",
    "attention_layer_post_proj_execution_time",
)


def _get_attention_time_component_attr(
    execution_time: ExecutionTime,
    attr_name: str,
) -> float:
    try:
        return float(getattr(execution_time.attention_time_component, attr_name))
    except AttributeError as exc:
        raise ValueError(
            "ExecutionTime attention component is missing model-level "
            f"projection attr: {attr_name}"
        ) from exc


def _get_attention_operator_time(
    execution_time: ExecutionTime,
    operator: AttentionOperatorSpec,
    *,
    require_structured: bool,
) -> float:
    if (
        execution_time.attention_operator_times is not None
        and operator.name in execution_time.attention_operator_times.op_times
    ):
        return (
            execution_time.attention_operator_times.get_required_time(operator.name)
            * execution_time.num_layers
        )
    if require_structured:
        return (
            execution_time.attention_operator_times.get_required_time(operator.name)
            * execution_time.num_layers
        )
    attr_name = operator.execution_time_attr
    if not attr_name:
        raise ValueError(
            "No ExecutionTime attribute registered for attention operator "
            f"{operator.name}"
        )
    try:
        return float(getattr(execution_time, attr_name))
    except AttributeError as exc:
        raise ValueError(
            "ExecutionTime is missing attribute for attention operator "
            f"{operator.name}: {attr_name}"
        ) from exc


def _get_optional_attention_operator_time(
    execution_time: ExecutionTime,
    operator: AttentionOperatorSpec,
) -> float | None:
    if execution_time.attention_operator_times is not None:
        if operator.name not in execution_time.attention_operator_times.op_times:
            return None
        return _get_attention_operator_time(
            execution_time,
            operator,
            require_structured=True,
        )
    if not operator.execution_time_attr:
        return None
    return _get_attention_operator_time(
        execution_time,
        operator,
        require_structured=False,
    )


def _validate_projection_ownership(
    execution_time: ExecutionTime,
    family: AttentionFamilySpec,
) -> None:
    disjoint_attrs = set(family.disjoint_model_projection_attrs)
    unknown_disjoint_attrs = sorted(
        disjoint_attrs.difference(_MODEL_PROJECTION_EXECUTION_TIME_ATTRS)
    )
    if unknown_disjoint_attrs:
        raise ValueError(
            "Unknown disjoint model projection attr(s) for attention family "
            f"{family.family_id}: {unknown_disjoint_attrs}"
        )

    overlapping_projection_ops = tuple(
        operator
        for operator in family.projection_ops()
        if (
            operator.projection_ownership
            is ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
            and (
                _get_optional_attention_operator_time(execution_time, operator)
                or 0.0
            )
            > 0.0
        )
    )
    if not overlapping_projection_ops:
        return

    # This is an ownership existence check only. Top-level ExecutionTime op
    # values can be layer-aggregated, while model projection component fields
    # are per-layer, so this guard must not compare or sum their magnitudes.
    overlapping_model_attrs = tuple(
        attr_name
        for attr_name in _MODEL_PROJECTION_EXECUTION_TIME_ATTRS
        if attr_name not in disjoint_attrs
        and _get_attention_time_component_attr(execution_time, attr_name) > 0.0
    )
    if not overlapping_model_attrs:
        return

    raise ValueError(
        "Invalid attention projection ownership: nonzero model-level "
        f"projection attrs {list(overlapping_model_attrs)} overlap with "
        "nonzero attention physical projection ops "
        f"{[operator.name for operator in overlapping_projection_ops]}. "
        "Declare disjoint_model_projection_attrs on the attention family only "
        "after vLLM/source evidence proves the work is disjoint."
    )


def get_attention_trace_op_times(
    execution_time: ExecutionTime,
    family: AttentionFamilySpec,
    *,
    per_layer_count: int | None = None,
    skip_zero: bool = True,
) -> tuple[tuple[AttentionOperatorSpec, float], ...]:
    """Return family-ordered E2E trace timings for attention physical ops."""
    family.require_enabled_for_execution()
    if per_layer_count is not None and per_layer_count <= 0:
        raise ValueError(f"per_layer_count must be positive: {per_layer_count}")
    _validate_projection_ownership(execution_time, family)

    require_structured = (
        execution_time.attention_operator_times is not None
        and any(
            operator.name in execution_time.attention_operator_times.op_times
            for operator in family.e2e_trace_ops()
        )
    )
    op_times: list[tuple[AttentionOperatorSpec, float]] = []
    for operator in family.e2e_trace_ops():
        duration_ms = _get_attention_operator_time(
            execution_time,
            operator,
            require_structured=require_structured,
        )
        if duration_ms < 0.0:
            raise ValueError(
                "Negative attention trace timing is invalid: "
                f"{operator.name}={duration_ms}"
            )
        if per_layer_count is not None:
            duration_ms /= per_layer_count
        if skip_zero and duration_ms == 0.0:
            continue
        op_times.append((operator, duration_ms))

    return tuple(op_times)
