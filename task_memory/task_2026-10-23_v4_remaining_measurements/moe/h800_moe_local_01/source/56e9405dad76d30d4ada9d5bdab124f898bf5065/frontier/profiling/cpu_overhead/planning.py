"""Planning helpers for single-node CPU overhead profiling."""

from __future__ import annotations

from typing import Iterable


def resolve_single_node_tp_plan(
    requested_tp_degrees: Iterable[int],
    single_node_gpu_capacity: int | None,
) -> tuple[list[int], list[int]]:
    """Split requested TP degrees into measurable vs missing-on-single-node lists."""
    normalized_tp_degrees = sorted(set(int(tp) for tp in requested_tp_degrees))
    if not normalized_tp_degrees:
        raise ValueError("requested_tp_degrees must contain at least one TP degree.")

    invalid_tp = [tp for tp in normalized_tp_degrees if tp <= 0]
    if invalid_tp:
        raise ValueError(
            f"All tensor_parallel_degree values must be positive, got {invalid_tp}."
        )

    if single_node_gpu_capacity is None:
        return normalized_tp_degrees, []

    capacity = int(single_node_gpu_capacity)
    if capacity <= 0:
        raise ValueError(
            "single_node_gpu_capacity must be a positive integer when provided. "
            f"Got {single_node_gpu_capacity!r}."
        )

    measurable_tp = [tp for tp in normalized_tp_degrees if tp <= capacity]
    missing_tp = [tp for tp in normalized_tp_degrees if tp > capacity]
    return measurable_tp, missing_tp
