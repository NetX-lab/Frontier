"""Preserve numerical ownership when preparing metrics timing records."""

from copy import copy

from frontier.entities.execution_time import ExecutionTime
from frontier.entities.stage_execution_time import StageExecutionTime


def build_metrics_execution_time(
    original: ExecutionTime | StageExecutionTime,
) -> ExecutionTime | StageExecutionTime:
    """Copy a timing record for annotations without changing its numerical scope.

    Published stages already own finalized layer and owner snapshots. A shallow
    record copy therefore isolates diagnostic annotations while preserving every
    numerical field and avoiding additional simulator entity IDs.
    """
    if isinstance(original, StageExecutionTime):
        return copy(original)
    if isinstance(original, ExecutionTime):
        return copy(original.finalized_copy())
    raise TypeError("Metrics timing requires ExecutionTime or StageExecutionTime")


def build_single_layer_metrics_execution_time(
    original: ExecutionTime | StageExecutionTime,
) -> ExecutionTime:
    """Copy one explicitly identified layer without inventing a stage aggregate."""
    if isinstance(original, StageExecutionTime):
        if original.num_layers != 1:
            raise ValueError("Single-layer metrics require exactly one stage layer")
        original = original.layer_execution_times[0]
    if not isinstance(original, ExecutionTime):
        raise TypeError("Single-layer metrics require an ExecutionTime layer")
    return copy(original.finalized_copy())
