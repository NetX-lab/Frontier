"""Per-layer forward-step synchronization entry for PREFILL, DECODE, MONOLITHIC."""

from typing import Any

from frontier.entities import Batch
from frontier.types import ClusterType


def _can_supply_idle_lane(scheduler, sibling_stage, replica_id, stage_id):
    """Allow queued next-forward work to wait while the active group advances."""
    if sibling_stage.is_busy:
        return False
    if sibling_stage.is_empty():
        return True
    return scheduler.get_stage_execution_context(replica_id, stage_id).forward_group_sealed


def uses_shared_forward_room(scheduler: Any) -> bool:
    """Return whether this cluster keeps one room for both local phases.

    A monolithic Replica runs prefill and decode on the same lanes, so one
    forward can hold a prefill batch on one lane and a decode batch on another.
    Those lanes must wait in one room and resolve to one step id.
    """

    # Checked room-first so a lightweight scheduler fixture that never sets up
    # a shared room is answered without requiring a cluster type.
    return (
        getattr(scheduler, "_forward_sync_waiting_room", None) is not None
        and getattr(scheduler, "_cluster_type", None) is ClusterType.MONOLITHIC
    )


def _load_sync_event(mode: str):
    if mode == "prefill":
        from frontier.events.prefill_sync_event import PrefillSyncEvent

        return PrefillSyncEvent
    from frontier.events.decode_sync_event import DecodeSyncEvent

    return DecodeSyncEvent


def enter_layer_sync(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Batch,
    replica_local_id: int | None,
    sync_stage: str,
    layer_id: int,
    stage_execution_time: float,
    *,
    mode: str,
    metrics_store: Any = None,
) -> list:
    """Admit one lane into its forward's pre_moe room, and dispatch when full.

    `mode` is the entering batch's own local phase. It selects the layer-path
    check, the event class used to fill an idle lane, and, for a disaggregated
    role, which room is used. On a monolithic cluster the room and the step-id
    namespace are shared, so a cohort whose lanes disagree about their phase
    still resolves to one forward.
    """

    del stage_execution_time
    if mode not in ("prefill", "decode"):
        raise ValueError(f"unsupported layer synchronization mode: {mode!r}")
    mode_name = mode.upper()

    shared_room = uses_shared_forward_room(scheduler)
    if shared_room:
        waiting_room = scheduler._forward_sync_waiting_room
        sync_kind = "forward"
    elif mode == "prefill":
        waiting_room = scheduler._prefill_sync_waiting_room
        sync_kind = "prefill"
    else:
        waiting_room = scheduler._decode_sync_waiting_room
        sync_kind = "decode"

    if waiting_room is None:
        raise ValueError(
            f"{mode_name} synchronization is unavailable for a dense model; "
            "dense execution must use the full-stage protocol"
        )
    if sync_stage != "pre_moe":
        raise ValueError(
            f"{mode_name} synchronization entry must start at pre_moe; post_moe "
            "completion is handled by the collective event"
        )
    layer_path_ok = (
        scheduler._uses_shared_prefill_layer_path(batch, layer_id)
        if mode == "prefill"
        else scheduler._uses_shared_decode_layer_path(batch, layer_id)
    )
    if not layer_path_ok:
        raise RuntimeError(
            f"Legacy {mode_name} DP synchronization is removed; the current "
            "layer must use the canonical per-layer protocol"
        )
    if replica_local_id is None:
        lane_id = 0
    elif type(replica_local_id) is not int or replica_local_id < 0:
        raise ValueError(
            f"{mode_name} replica_local_id must be an exact non-negative int or None"
        )
    else:
        lane_id = replica_local_id

    step_id = scheduler._resolve_forward_step(
        sync_kind=sync_kind,
        waiting_room=waiting_room,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        lane_id=lane_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
    )
    if step_id is None:
        return []
    sync_room = waiting_room[replica_id][stage_id][step_id][layer_id][sync_stage]
    # resolve_step retains this binding identity while step_id advances per layer.
    provisional_id = batch._forward_cohort_provisional_id
    sync_room.setdefault("provisional_cohort_id", provisional_id)
    existing_batch = sync_room["batches"].get(lane_id)
    if batch.is_idle and existing_batch is not None and not existing_batch.is_idle:
        return []
    sync_room["batches"][lane_id] = batch
    sync_room["arrival_times"][lane_id] = float(time)

    expected_lanes = scheduler._replica_dp_size
    if type(expected_lanes) is not int or expected_lanes <= 0:
        raise ValueError(
            f"{mode_name} attention-DP lane count must be positive, got {expected_lanes}"
        )
    if len(sync_room["batches"]) < expected_lanes and not batch.is_idle:
        idle_events = []
        event_cls = _load_sync_event(mode)
        replica_schedulers = scheduler._replica_schedulers
        for missing_lane in range(expected_lanes):
            if missing_lane in sync_room["batches"]:
                continue
            sibling = replica_schedulers.get((replica_id, missing_lane))
            if sibling is None:
                raise RuntimeError(
                    "Missing Replica scheduler for expected attention-DP lane: "
                    f"replica_id={replica_id}, replica_local_id={missing_lane}"
                )
            sibling_stage = sibling.get_replica_stage_scheduler(stage_id)
            if not _can_supply_idle_lane(scheduler, sibling_stage, replica_id, stage_id):
                continue
            idle_batch = Batch(
                replica_id=replica_id,
                requests=[],
                num_tokens=[],
                is_idle=True,
                is_moe=batch.is_moe,
            )
            idle_batch.set_global_id(expected_lanes * step_id + missing_lane)
            idle_batch._forward_cohort_id = step_id
            idle_batch._forward_cohort_provisional_id = provisional_id
            idle_batch._stage_owner_replica_local_id = missing_lane
            sync_room["batches"][missing_lane] = idle_batch
            sync_room["arrival_times"][missing_lane] = float(time)
            idle_events.append(
                event_cls(
                    time=float(time),
                    replica_id=replica_id,
                    stage_id=stage_id,
                    batch=idle_batch,
                    replica_local_id=missing_lane,
                    sync_stage=sync_stage,
                    layer_id=layer_id,
                    stage_execution_time=0.0,
                    cluster_type=scheduler._cluster_type,
                )
            )
        if idle_events:
            return idle_events

    if len(sync_room["batches"]) != expected_lanes:
        return []
    sync_time = max(sync_room["arrival_times"].values())
    step_batches = dict(sync_room["batches"])
    provisional_id = sync_room["provisional_cohort_id"]
    if type(provisional_id) is not int or provisional_id < 0:
        raise RuntimeError(
            f"{mode_name} synchronization room has an invalid provisional step id: "
            f"{provisional_id!r}"
        )
    scheduler._close_forward_step(
        sync_kind=sync_kind,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
        provisional_id=provisional_id,
    )
    sync_room.pop("batches", None)
    sync_room.pop("arrival_times", None)
    sync_room.pop("provisional_cohort_id", None)

    if shared_room:
        return scheduler._on_forward_ep_wave_ready(
            time=sync_time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
            replica_local_id=replica_local_id,
            cohort_batches=step_batches,
            metrics_store=metrics_store,
        )
    ready = (
        scheduler._on_prefill_ep_wave_ready
        if mode == "prefill"
        else scheduler._on_decode_ep_wave_ready
    )
    return ready(
        time=sync_time,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        layer_id=layer_id,
        replica_local_id=replica_local_id,
        cohort_batches=step_batches,
        metrics_store=metrics_store,
    )


def enter_prefill_sync(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Batch,
    replica_local_id: int | None,
    sync_stage: str,
    layer_id: int,
    stage_execution_time: float,
    *,
    metrics_store: Any = None,
) -> list:
    """Admit a lane whose local phase is prefill."""

    return enter_layer_sync(
        scheduler, time, replica_id, stage_id, batch, replica_local_id,
        sync_stage, layer_id, stage_execution_time,
        mode="prefill", metrics_store=metrics_store,
    )


def enter_decode_sync(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Batch,
    replica_local_id: int | None,
    sync_stage: str,
    layer_id: int,
    stage_execution_time: float,
    *,
    metrics_store: Any = None,
) -> list:
    """Admit a lane whose local phase is decode."""

    return enter_layer_sync(
        scheduler, time, replica_id, stage_id, batch, replica_local_id,
        sync_stage, layer_id, stage_execution_time,
        mode="decode", metrics_store=metrics_store,
    )
