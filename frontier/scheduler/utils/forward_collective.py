"""Completion of one shared monolithic forward across mixed source lanes."""

from __future__ import annotations

from typing import Any

from frontier.scheduler.utils.collective_timing import validate_decode_layer_advance
from frontier.scheduler.utils.forward_sync_state import source_forward_mode
from frontier.scheduler.utils.request_selection import collect_active_requests


def handle_forward_sync_collective(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch_global_id: int,
    sync_stage: str,
    layer_id: int,
    metrics_store: Any,
):
    """Complete one shared forward once, then continue each source locally.

    Two things belong to the forward as a whole and must happen exactly once:
    the decode-phase layer counters advance, and the full-stage owners the EP
    wave took over are restored. Everything after that is source-local — each
    lane predicts its own next attention, keeps its own stage tail and reaches
    its own completion — so it is delegated to the same per-phase helper the
    disaggregated roles use, entered once per source.
    """

    if sync_stage != "post_moe":
        raise ValueError(
            "Forward collective completion accepts only post_moe; the canonical "
            "EP_WAVE enters this method at post_moe"
        )

    replica_rooms = scheduler._forward_sync_waiting_room.get(replica_id)
    stage_rooms = replica_rooms.get(stage_id) if replica_rooms is not None else None
    step_rooms = stage_rooms.get(batch_global_id) if stage_rooms is not None else None
    layer_rooms = step_rooms.get(layer_id) if step_rooms is not None else None
    if layer_rooms is None or sync_stage not in layer_rooms:
        raise RuntimeError(
            "Forward collective event has no matching waiting room: "
            f"replica={replica_id}, stage={stage_id}, "
            f"batch_global_id={batch_global_id}, layer={layer_id}, "
            f"sync_stage={sync_stage}"
        )
    source_batches = layer_rooms.pop(sync_stage)["batches"]

    live_batches = [batch for batch in source_batches.values() if not batch.is_idle]
    if not live_batches:
        raise RuntimeError(
            "Forward collective completion requires a non-idle participant batch: "
            f"replica={replica_id}, stage={stage_id}, "
            f"batch_global_id={batch_global_id}, layer={layer_id}"
        )
    # Each per-phase helper refuses a legacy aggregate synchronization by
    # checking that its batch carries the wave's lane timings, but only when it
    # pops the room itself. Delegation passes `direct_batch`, so that check is
    # skipped there and belongs here instead -- once per source, against the
    # marker the wave writes for that source's own phase.
    for source_batch in live_batches:
        marker = (
            "_prefill_ep_wave_lane_times_ms"
            if source_forward_mode(source_batch) == "prefill"
            else "_decode_ep_wave_lane_times_ms"
        )
        if not hasattr(source_batch, marker):
            raise RuntimeError(
                "Legacy aggregate synchronization is removed; a shared forward "
                "source must carry its own EP wave lane timings: "
                f"replica={replica_id}, stage={stage_id}, "
                f"batch_global_id={batch_global_id}, layer={layer_id}, "
                f"batch={source_batch.id}, expected={marker}"
            )

    # One completed layer advances a request's decode counter once, and only if
    # that request is decoding. A prefill chunk has no decode layer to credit,
    # and a request carried in a prefill batch after its own prefill finished
    # does, which is the case the phase-specific paths could not express.
    decoding_requests = [
        request
        for request in collect_active_requests(source_batches.values())
        if request.is_prefill_complete
    ]
    num_layers = scheduler._config.replica_config.model_config.num_layers
    validate_decode_layer_advance(decoding_requests, num_layers)
    for request in decoding_requests:
        request.mb_on_step_layer_count_increment(num_layers_completed=1)

    stage_layer_end = _stage_layer_end(scheduler, stage_id)
    next_layer_id = layer_id + 1
    owners_restored = scheduler._restore_forward_step_full_stage_owners(
        source_batches=source_batches,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=next_layer_id,
        cohort_id=batch_global_id,
        operation_kind="attention" if next_layer_id < stage_layer_end else "final",
    )

    from frontier.scheduler.utils.decode_collective import handle_decode_sync_collective
    from frontier.scheduler.utils.prefill_collective import handle_prefill_sync_collective

    events = []
    for source_batch in live_batches:
        is_prefill_source = source_forward_mode(source_batch) == "prefill"
        handler = (
            handle_prefill_sync_collective
            if is_prefill_source
            else handle_decode_sync_collective
        )
        # Only the decode helper advances layer counters, so only it needs to be
        # told they are already advanced.
        already_done = {} if is_prefill_source else {"layer_advance_done": True}
        events.extend(
            handler(
                scheduler,
                time,
                replica_id,
                stage_id,
                batch_global_id,
                sync_stage,
                layer_id,
                metrics_store,
                direct_batch=source_batch,
                owners_restored=owners_restored,
                **already_done,
            )
        )
    return events


def _stage_layer_end(scheduler: Any, stage_id: int) -> int:
    """Return the half-open last layer this pipeline stage owns."""

    num_layers = scheduler._predictor._num_layers_per_pipeline_stage
    _, stage_layer_end = scheduler.get_pipeline_stage_layer_bounds(stage_id, num_layers)
    return stage_layer_end
