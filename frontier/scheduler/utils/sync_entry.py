"""PREFILL and DECODE forward-step synchronization entry handlers."""

from typing import Any

from frontier.entities import Batch


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
) -> list:
    del stage_execution_time
    if scheduler._prefill_sync_waiting_room is None:
        raise ValueError(
            "PREFILL synchronization is unavailable for a dense model; dense execution must use the full-stage protocol"
        )
    if sync_stage != "pre_moe":
        raise ValueError(
            "PREFILL synchronization entry must start at pre_moe; post_moe completion is handled by PrefillSyncCollectiveEvent"
        )
    if not scheduler._uses_shared_prefill_layer_path(batch, layer_id):
        raise RuntimeError(
            "Legacy PREFILL DP synchronization is removed; the current layer must use the canonical per-layer protocol"
        )
    lane_id = 0 if replica_local_id is None else int(replica_local_id)
    requested_step_id = scheduler._get_forward_step_id(batch)
    step_id, already_completed = scheduler._resolve_forward_step(
        sync_kind="prefill",
        waiting_room=scheduler._prefill_sync_waiting_room,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        lane_id=lane_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
    )
    if already_completed:
        return []
    sync_room = scheduler._prefill_sync_waiting_room[replica_id][stage_id][step_id][layer_id][sync_stage]
    sync_room.setdefault("provisional_cohort_id", requested_step_id)
    existing_batch = sync_room["batches"].get(lane_id)
    if batch.is_idle and existing_batch is not None and not existing_batch.is_idle:
        return []
    sync_room["batches"][lane_id] = batch
    sync_room["arrival_times"][lane_id] = float(time)

    expected_lanes = int(getattr(scheduler, "_replica_dp_size", 1) or 1)
    if expected_lanes <= 0:
        raise ValueError(f"PREFILL attention-DP lane count must be positive, got {expected_lanes}")
    if len(sync_room["batches"]) < expected_lanes and not batch.is_idle:
        idle_events = []
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
            if sibling_stage.is_busy or not sibling_stage.is_empty():
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
            idle_batch._forward_cohort_provisional_id = requested_step_id
            idle_batch._stage_owner_replica_local_id = missing_lane
            sync_room["batches"][missing_lane] = idle_batch
            sync_room["arrival_times"][missing_lane] = float(time)
            from frontier.events.prefill_sync_event import PrefillSyncEvent

            idle_events.append(
                PrefillSyncEvent(
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
    scheduler._close_forward_step(
        sync_kind="prefill",
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
        provisional_id=int(sync_room.get("provisional_cohort_id", requested_step_id)),
        cohort_id=step_id,
        cohort_batches=step_batches,
    )
    sync_room.pop("batches", None)
    sync_room.pop("arrival_times", None)
    sync_room.pop("provisional_cohort_id", None)
    return scheduler._on_prefill_ep_wave_ready(
        time=sync_time,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        layer_id=layer_id,
        replica_local_id=replica_local_id,
        cohort_batches=step_batches,
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
) -> list:
    del stage_execution_time
    if scheduler._decode_sync_waiting_room is None:
        raise ValueError(
            "DECODE synchronization is unavailable for a dense model; dense execution must use the full-stage protocol"
        )
    if sync_stage != "pre_moe":
        raise ValueError(
            "DECODE synchronization entry must start at pre_moe; post_moe completion is handled by DecodeSyncCollectiveEvent"
        )
    if not scheduler._uses_shared_decode_layer_path(batch, layer_id):
        raise RuntimeError(
            "Legacy DECODE DP synchronization is removed; the current layer must use the canonical per-layer protocol"
        )
    lane_id = 0 if replica_local_id is None else int(replica_local_id)
    requested_step_id = scheduler._get_forward_step_id(batch)
    step_id, already_completed = scheduler._resolve_forward_step(
        sync_kind="decode",
        waiting_room=scheduler._decode_sync_waiting_room,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        lane_id=lane_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
    )
    sync_room = scheduler._decode_sync_waiting_room[replica_id][stage_id][step_id][layer_id][sync_stage]
    if already_completed:
        return []
    sync_room.setdefault("provisional_cohort_id", requested_step_id)
    existing_batch = sync_room["batches"].get(lane_id)
    if batch.is_idle and existing_batch is not None and not existing_batch.is_idle:
        return []
    sync_room["batches"][lane_id] = batch
    sync_room["arrival_times"][lane_id] = float(time)

    expected_lanes = int(getattr(scheduler, "_replica_dp_size", 1) or 1)
    if expected_lanes <= 0:
        raise ValueError(f"DECODE attention-DP lane count must be positive, got {expected_lanes}")
    if len(sync_room["batches"]) < expected_lanes and not batch.is_idle:
        idle_events = []
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
            if sibling_stage.is_busy or not sibling_stage.is_empty():
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
            idle_batch._forward_cohort_provisional_id = requested_step_id
            idle_batch._stage_owner_replica_local_id = missing_lane
            sync_room["batches"][missing_lane] = idle_batch
            sync_room["arrival_times"][missing_lane] = float(time)
            from frontier.events.decode_sync_event import DecodeSyncEvent

            idle_events.append(
                DecodeSyncEvent(
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
    scheduler._close_forward_step(
        sync_kind="decode",
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
        provisional_id=int(sync_room.get("provisional_cohort_id", requested_step_id)),
        cohort_id=step_id,
        cohort_batches=step_batches,
    )
    sync_room.pop("batches", None)
    sync_room.pop("arrival_times", None)
    sync_room.pop("provisional_cohort_id", None)
    return scheduler._on_decode_ep_wave_ready(
        time=sync_time,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        layer_id=layer_id,
        replica_local_id=replica_local_id,
        cohort_batches=step_batches,
    )
