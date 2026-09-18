"""Dense-layer completion and physical-layer timing for mixed-MoE metrics."""

from __future__ import annotations

from typing import Any

from frontier.config.model_config import BaseModelConfig
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.stage_execution_time import StageExecutionTime

from frontier.scheduler.utils.execution_time_metrics import (
    build_metrics_execution_time,
    build_single_layer_metrics_execution_time,
)


def complete_dense_layer(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    layer_id: int,
    phase: str,
    metrics_store: Any,
) -> list:
    """Advance a dense layer through the scheduler's existing phase handler."""
    if phase == "prefill":
        return scheduler.on_prefill_sync_collective(
            time,
            replica_id,
            stage_id,
            int(batch.global_id),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    if phase == "decode":
        return scheduler.on_decode_sync_collective(
            time,
            replica_id,
            stage_id,
            scheduler._get_decode_sync_wait_key(batch),
            "post_moe",
            layer_id,
            metrics_store,
            direct_batch=batch,
        )
    raise ValueError(f"Unsupported dense layer completion phase: {phase!r}")


def build_prefill_metrics_execution_time(
    *,
    original_execution_time: ExecutionTime | StageExecutionTime,
    sample_batch: Any,
    predictor: Any,
    stage_id: int,
    cluster_type: Any,
    model_config: BaseModelConfig,
) -> ExecutionTime | StageExecutionTime:
    """Preserve stage scope and include each executed dense FFN's own timing.

    EP wave records own routed MoE timing. Stage records own attention and
    dense FFNs, whose timings must be predicted at their actual layer IDs.
    """
    if not isinstance(original_execution_time, StageExecutionTime) or not (
        0 < model_config.get_num_moe_layers() < model_config.num_layers
    ):
        return build_metrics_execution_time(original_execution_time)
    layers = []
    for layer in original_execution_time.layer_execution_times:
        if model_config.is_moe_layer(layer.global_layer_id):
            layers.append(layer)
            continue
        dense_stage = predictor.predict_stage_execution_time(
            sample_batch, stage_id, cluster_type=cluster_type,
            num_layers=1, layer_id=layer.global_layer_id,
        )
        layers.append(build_single_layer_metrics_execution_time(dense_stage))
    return StageExecutionTime(
        layers, stage_execution_time=original_execution_time.stage_execution_time,
    )
