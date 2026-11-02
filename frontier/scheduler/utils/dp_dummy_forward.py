"""One pipeline stage's part of vLLM's attention-DP dummy forward, and the
first-layer pricing that waits for a forward's DP peers.

The dummy part walks the stage's layers through the same pre_moe rooms as a
real forward, so every MoE layer of the other lanes finds this lane's partner.
It carries no request: no layer is credited and no metric is recorded.

A lane whose first layer depends on its peers enters the first pre_moe room at
its stage start and is priced there once the room is full. For DeepSeek V4,
served by the vLLM V2 runner, that room is the DP rendezvous of every lane:
the runner synchronizes the lanes' CUDA graph descriptors before it prepares
inputs (v1/worker/gpu/dp_utils.py, sync_cudagraph_and_dp_padding).
"""

from dataclasses import replace
from typing import Any

from frontier.config import global_vars
from frontier.entities import BatchStage, DummyForwardBatch
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
)
from frontier.scheduler.utils.collective_timing import attention_delay_seconds
from frontier.scheduler.utils.ep_wave_schedule import _source_mode
from frontier.types import ClusterType

# CUDAGraphMode values: the DP group runs the lowest mode any lane wants.
_GRAPH_MODE_ORDER = {"NONE": 0, "PIECEWISE": 1, "FULL": 2}


def runs_dp_rendezvous(scheduler: Any) -> bool:
    """Whether every lane's first layer waits for the forward's DP rendezvous.

    DeepSeek V4 is served by the vLLM V2 runner, which synchronizes the DP
    group's descriptor before any lane prepares its inputs.
    """

    model_config = scheduler._config.replica_config.model_config
    return model_config.get_model_architecture_profile().profile_id == "deepseek_v4"


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


def defer_first_layer(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    replica_local_id: int | None,
    *,
    prefill: bool,
) -> list:
    """Enter the first pre_moe room at stage start; the room prices the layer.

    `prefill` selects the room's prefill entry. The first pre_moe room already
    collects exactly this forward's lanes, so the layer is priced there once
    its peers' descriptors are known (resolve_first_layer_inputs).
    """

    from frontier.events.decode_sync_event import DecodeSyncEvent
    from frontier.events.prefill_sync_event import PrefillSyncEvent

    num_layers = scheduler._predictor._num_layers_per_pipeline_stage
    first_layer_id, _ = scheduler.get_pipeline_stage_layer_bounds(stage_id, num_layers)
    batch._first_layer_pricing_deferred = True
    batch._forward_launch_start_time = time
    batch._forward_launch_time = 0.0
    event_cls = PrefillSyncEvent if prefill else DecodeSyncEvent
    return [event_cls(
        time, replica_id, stage_id, batch, replica_local_id,
        "pre_moe", first_layer_id, 0.0, cluster_type=scheduler._cluster_type,
    )]


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
    if runs_dp_rendezvous(scheduler):
        return defer_first_layer(
            scheduler, time, replica_id, stage_id, dummy_forward,
            dummy_forward._stage_owner_replica_local_id,
            prefill=scheduler._cluster_type == ClusterType.PREFILL,
        )
    if (
        scheduler._cluster_type in (ClusterType.MONOLITHIC, ClusterType.DECODE)
        and global_vars.get_decode_cuda_graph_mode() != "none"
    ):
        # Charge no eager host overhead before the DP negotiation.
        return defer_first_layer(
            scheduler, time, replica_id, stage_id, dummy_forward,
            dummy_forward._stage_owner_replica_local_id, prefill=False,
        )
    execution_time = _predict_layer(scheduler, stage_id, dummy_forward, first_layer_id)
    dummy_forward._forward_launch_time = execution_time.forward_launch_time * 1e-3
    return _enter_layer(
        scheduler, time, replica_id, stage_id, dummy_forward, first_layer_id, execution_time
    )


def resolve_first_layer_inputs(
    scheduler: Any, replica_id: int, stage_id: int, layer_id: int, room: dict
) -> None:
    """Price the first layers deferred to this full room."""

    batches = room["batches"]
    if not any(getattr(batch, "_first_layer_pricing_deferred", False)
               for batch in batches.values()):
        return
    if global_vars.get_decode_cuda_graph_mode() == "full_and_piecewise":
        _synchronize_dp_descriptors(scheduler, replica_id, batches)
    if runs_dp_rendezvous(scheduler):
        _price_dp_rendezvous(scheduler, stage_id, layer_id, room)
    else:
        _price_deferred_dummies(scheduler, stage_id, layer_id, room)


def _synchronize_dp_descriptors(scheduler: Any, replica_id: int, batches: dict) -> None:
    """Apply the V2 runner's DP descriptor synchronization to every lane.

    The group runs the lowest mode any lane wants. Eager lanes keep their own
    width. Otherwise each lane dispatches again on the widest lane's tokens,
    with the uniform token count only when every lane agrees on it, and its
    own request count, so PIECEWISE pads tokens and not requests
    (dp_utils.py, sync_cudagraph_and_dp_padding). A lane without graph
    metadata runs eager.
    """

    metadata = {
        lane_id: batch.decode_cuda_graph_metadata for lane_id, batch in batches.items()
    }
    modes = [batch.get_decode_cuda_graph_runtime_mode() for batch in batches.values()]
    if min(_GRAPH_MODE_ORDER[mode] for mode in modes) == _GRAPH_MODE_ORDER["NONE"]:
        for lane_id, batch in batches.items():
            if metadata[lane_id] is not None:
                batch.decode_cuda_graph_metadata = replace(
                    metadata[lane_id],
                    runtime_mode="NONE",
                    capture_hit=False,
                    padded_total_tokens=metadata[lane_id].original_total_tokens,
                    padded_decode_batch_size=metadata[lane_id].original_decode_batch_size,
                )
        return
    num_tokens = max(lane.original_total_tokens for lane in metadata.values())
    uniform_token_count = (
        None if any(lane.is_mixed_batch for lane in metadata.values()) else 1
    )
    for lane_id, batch in batches.items():
        lane_scheduler = scheduler.get_replica_scheduler(replica_id, lane_id)
        runtime_mode, padded_total_tokens, padded_num_reqs = (
            lane_scheduler.dispatch_full_and_piecewise(
                batch.size, num_tokens, uniform_token_count
            )
        )
        batch.decode_cuda_graph_metadata = replace(
            metadata[lane_id],
            runtime_mode=runtime_mode,
            capture_hit=True,
            padded_total_tokens=padded_total_tokens,
            padded_decode_batch_size=(
                metadata[lane_id].original_decode_batch_size + padded_num_reqs - batch.size
            ),
        )


def _price_dp_rendezvous(scheduler: Any, stage_id: int, layer_id: int, room: dict) -> None:
    """Place every lane's first layer on the V2 runner's DP rendezvous timeline.

    A lane entered at its stage start: V4 host time is an explicit zero, so
    its pre-DP host share is zero, and its pre-DP device work (cache-block
    zeroing) starts there. The rendezvous is the latest entry; device work
    does not delay it. Each lane's preparation and first layer start at the
    later of the rendezvous plus its preparation and its pre-DP device end.
    """

    entry_times = dict(room["arrival_times"])
    rendezvous_time = max(entry_times.values())
    for lane_id, batch in room["batches"].items():
        execution_time = _predict_layer(scheduler, stage_id, batch, layer_id)
        pre_dp_time_ms = execution_time.get_single_layer_pre_dp_time()
        attention_time_ms = execution_time.get_single_layer_attention_scope_time()
        launch_start_time = (
            rendezvous_time + execution_time.forward_preparation_time * 1e-3
        )
        device_start_time = max(
            launch_start_time, entry_times[lane_id] + pre_dp_time_ms * 1e-3
        )
        batch._forward_launch_start_time = launch_start_time
        batch._forward_launch_time = execution_time.forward_launch_time * 1e-3
        if _source_mode(scheduler._sync_kind, batch) == "prefill":
            batch._prefill_model_execution_components_ms_by_stage[stage_id].extend(
                (pre_dp_time_ms, attention_time_ms)
            )
        room["arrival_times"][lane_id] = device_start_time + attention_time_ms * 1e-3
        batch._first_layer_pricing_deferred = False


def _price_deferred_dummies(scheduler: Any, stage_id: int, layer_id: int, room: dict) -> None:
    """Price deferred dummy first layers with vLLM's DP-min mode and DP-max width."""
    batches = room["batches"]
    pending = [
        (lane_id, batch) for lane_id, batch in batches.items()
        if getattr(batch, "_first_layer_pricing_deferred", False)
    ]
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
        dummy._first_layer_pricing_deferred = False


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
