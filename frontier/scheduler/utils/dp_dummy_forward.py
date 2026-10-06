"""One pipeline stage's part of vLLM's attention-DP dummy forward.

The part walks the stage's layers through the same pre_moe rooms as a real
forward, so every MoE layer of the other lanes finds this lane's partner. It
carries no request: no layer is credited and no metric is recorded.
"""

from typing import Any

from frontier.entities import BatchStage, DummyForwardBatch
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
)
from frontier.scheduler.utils.collective_timing import attention_delay_seconds
from frontier.types import ClusterType


def _enter_layer(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    dummy_forward: DummyForwardBatch,
    layer_id: int,
) -> list:
    """Run the layer's attention-free projections, then enter its pre_moe room."""

    from frontier.events.decode_sync_event import DecodeSyncEvent
    from frontier.events.prefill_sync_event import PrefillSyncEvent

    execution_time = scheduler._predictor.predict_stage_execution_time(
        dummy_forward,
        stage_id,
        scheduler._cluster_type,
        num_layers=1,
        layer_id=layer_id,
        include_ffn=False,
    )
    delay = attention_delay_seconds(execution_time)
    # A PREFILL cluster admits lanes to its rooms through the prefill entry.
    event_cls = (
        PrefillSyncEvent
        if scheduler._cluster_type == ClusterType.PREFILL
        else DecodeSyncEvent
    )
    return [
        event_cls(
            time + delay,
            replica_id,
            stage_id,
            dummy_forward,
            dummy_forward._stage_owner_replica_local_id,
            "pre_moe",
            layer_id,
            delay,
            cluster_type=scheduler._cluster_type,
        )
    ]


def start_dummy_forward(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    dummy_forward: DummyForwardBatch,
) -> list:
    """Start the part on its stage's first layer."""

    num_layers = scheduler._predictor._num_layers_per_pipeline_stage
    first_layer_id, _ = scheduler.get_pipeline_stage_layer_bounds(stage_id, num_layers)
    dummy_forward.stage_start_time = time
    return _enter_layer(scheduler, time, replica_id, stage_id, dummy_forward, first_layer_id)


def advance_dummy_forward(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    dummy_forward: DummyForwardBatch,
    next_layer_id: int,
    stage_layer_end: int,
    owners_restored: bool,
) -> list:
    """Continue the part after a layer, or end it after the stage's last layer."""

    from frontier.events.batch_stage_end_event import BatchStageEndEvent

    if next_layer_id < stage_layer_end:
        if not owners_restored:
            scheduler.transition_stage_admission_for_layer(
                dummy_forward,
                stage_id=stage_id,
                layer_id=next_layer_id,
                operation_kind="attention",
                scope=FULL_STAGE_WORLD,
            )
        return _enter_layer(
            scheduler, time, replica_id, stage_id, dummy_forward, next_layer_id
        )

    predictor = scheduler._predictor
    num_layers = predictor._num_layers_per_pipeline_stage
    stage_execution = predictor.predict_stage_execution_time(
        dummy_forward,
        stage_id,
        scheduler._cluster_type,
        num_layers=num_layers,
        layer_id=stage_layer_end - num_layers,
        include_ffn=False,
    )
    start_time = dummy_forward.stage_start_time
    # A dummy run samples nothing and sends nothing to the next rank; only a
    # kernel launch slower than the device work it overlaps extends it.
    end_time = time + stage_execution.forward_launch_stall_time(time - start_time)
    lane_id = dummy_forward._stage_owner_replica_local_id
    stage_scheduler = scheduler.get_replica_stage_scheduler(replica_id, lane_id, stage_id)
    batch_stage = BatchStage(
        dummy_forward.id,
        replica_id,
        stage_id,
        end_time - start_time,
        time - start_time,
        dummy_forward.requests,
        dummy_forward.num_tokens,
        dummy_forward.request_is_decoding,
        scheduler._cluster_type,
    )
    batch_stage.on_schedule(start_time)
    return [
        BatchStageEndEvent(
            end_time,
            replica_id,
            stage_id,
            stage_scheduler.is_last_stage,
            dummy_forward,
            batch_stage,
            scheduler._cluster_type,
            lane_id,
        )
    ]
