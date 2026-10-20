"""One pipeline stage's part of vLLM's attention-DP dummy forward.

The part walks the stage's layers through the same pre_moe rooms as a real
forward, so every MoE layer of the other lanes finds this lane's partner. It
carries no request: no layer is credited and no metric is recorded.
"""

from dataclasses import replace
from typing import Any

from frontier.config import global_vars
from frontier.entities import BatchStage, DummyForwardBatch
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
)
from frontier.scheduler.utils.collective_timing import attention_delay_seconds
from frontier.types import ClusterType


def _predict_layer(
    scheduler: Any, stage_id: int, dummy_forward: DummyForwardBatch, layer_id: int
) -> Any:
    return scheduler._predictor.predict_stage_execution_time(
        dummy_forward,
        stage_id,
        scheduler._cluster_type,
        num_layers=1,
        layer_id=layer_id,
        include_ffn=False,
    )


def _enter_layer(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    dummy_forward: DummyForwardBatch,
    layer_id: int,
    execution_time: Any,
) -> list:
    """Run the layer's attention-free projections, then enter its pre_moe room."""

    from frontier.events.decode_sync_event import DecodeSyncEvent
    from frontier.events.prefill_sync_event import PrefillSyncEvent

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
    dummy_forward._forward_launch_start_time = time
    if (
        scheduler._cluster_type in (ClusterType.MONOLITHIC, ClusterType.DECODE)
        and global_vars.get_decode_cuda_graph_mode() != "none"
    ):
        # The first pre_moe room already collects exactly this forward's lanes.
        # Defer the dummy's first prediction until its peers' dispatch descriptors
        # are known, instead of charging eager host overhead before DP negotiation.
        dummy_forward._dummy_dispatch_pending = True
        dummy_forward._forward_launch_time = 0.0
        from frontier.events.decode_sync_event import DecodeSyncEvent

        return [DecodeSyncEvent(
            time, replica_id, stage_id, dummy_forward,
            dummy_forward._stage_owner_replica_local_id,
            "pre_moe", first_layer_id, 0.0, cluster_type=scheduler._cluster_type,
        )]
    execution_time = _predict_layer(scheduler, stage_id, dummy_forward, first_layer_id)
    dummy_forward._forward_launch_time = execution_time.forward_launch_time * 1e-3
    return _enter_layer(
        scheduler, time, replica_id, stage_id, dummy_forward, first_layer_id, execution_time
    )


def resolve_dummy_forward_inputs(scheduler: Any, stage_id: int, layer_id: int, room: dict) -> None:
    """Price deferred first layers with vLLM's DP-min mode and DP-max width."""
    batches = room["batches"]
    pending = [
        (lane_id, batch) for lane_id, batch in batches.items()
        if isinstance(batch, DummyForwardBatch)
        and getattr(batch, "_dummy_dispatch_pending", False)
    ]
    if not pending:
        return
    modes = [batch.get_decode_cuda_graph_runtime_mode() for batch in batches.values()]
    graph_mode = "NONE" if "NONE" in modes else (
        "PIECEWISE" if "PIECEWISE" in modes else "FULL"
    )
    width = max(batch.get_effective_total_tokens_for_compute(scheduler._cluster_type)
                for batch in batches.values())
    for lane_id, dummy in pending:
        if graph_mode == "NONE":
            dummy.decode_cuda_graph_metadata = None
        else:
            dummy.decode_cuda_graph_metadata = replace(
                dummy.decode_cuda_graph_metadata,
                runtime_mode=graph_mode, padded_total_tokens=width,
                padded_decode_batch_size=width,
            )
        execution_time = _predict_layer(scheduler, stage_id, dummy, layer_id)
        dummy._forward_launch_time = execution_time.forward_launch_time * 1e-3
        room["arrival_times"][lane_id] = (
            dummy.stage_start_time + attention_delay_seconds(execution_time)
        )
        dummy._dummy_dispatch_pending = False


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
            scheduler, time, replica_id, stage_id, dummy_forward, next_layer_id,
            _predict_layer(scheduler, stage_id, dummy_forward, next_layer_id),
        )

    start_time = dummy_forward.stage_start_time
    # A dummy run samples nothing and sends nothing to the next rank. Its last
    # layer's room opened only once every lane had launched the whole stage
    # (sync_entry), so no launch is left exposed and the part ends here.
    lane_id = dummy_forward._stage_owner_replica_local_id
    stage_scheduler = scheduler.get_replica_stage_scheduler(replica_id, lane_id, stage_id)
    batch_stage = BatchStage(
        dummy_forward.id,
        replica_id,
        stage_id,
        time - start_time,
        time - start_time,
        dummy_forward.requests,
        dummy_forward.num_tokens,
        dummy_forward.request_is_decoding,
        scheduler._cluster_type,
    )
    batch_stage.on_schedule(start_time)
    return [
        BatchStageEndEvent(
            time,
            replica_id,
            stage_id,
            stage_scheduler.is_last_stage,
            dummy_forward,
            batch_stage,
            scheduler._cluster_type,
            lane_id,
        )
    ]
