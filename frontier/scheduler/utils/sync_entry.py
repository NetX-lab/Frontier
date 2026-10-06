"""Per-layer forward-step synchronization entry for PREFILL, DECODE, MONOLITHIC."""

from typing import Any

from frontier.entities import Batch


def _wake_idle_engines(scheduler: Any, time: float, replica_id: int, sync_room: dict) -> list:
    """Wake the engines of missing lanes that have nothing in flight.

    vLLM starts a DP wave on every engine of the group once one engine has a
    request, so an engine with no work of its own joins with a dummy forward.
    An engine that is still busy reaches this room through its own forward.
    """

    from frontier.events.replica_schedule_event import ReplicaScheduleEvent

    return [
        ReplicaScheduleEvent(time, replica_id, scheduler._cluster_type, lane_id)
        for lane_id in range(scheduler._replica_dp_size)
        if lane_id not in sync_room["batches"]
        and scheduler.get_replica_scheduler(replica_id, lane_id).engine_loop_idle
    ]


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
    check. Every lane of the cluster waits in its one room, and the cluster's
    sync kind selects the wave handler, so on a monolithic cluster a cohort
    whose lanes disagree about their phase still resolves to one forward.
    """

    del stage_execution_time
    mode_name = mode.upper()

    waiting_room = scheduler._sync_waiting_room
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
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        lane_id=lane_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
    )
    sync_room = waiting_room[replica_id][stage_id][step_id][layer_id][sync_stage]
    # resolve_step retains this binding identity while step_id advances per layer.
    sync_room.setdefault("provisional_cohort_id", batch._forward_cohort_provisional_id)
    sync_room["batches"][lane_id] = batch
    sync_room["arrival_times"][lane_id] = float(time)

    expected_lanes = scheduler._replica_dp_size
    if type(expected_lanes) is not int or expected_lanes <= 0:
        raise ValueError(
            f"{mode_name} attention-DP lane count must be positive, got {expected_lanes}"
        )
    if len(sync_room["batches"]) < expected_lanes:
        return _wake_idle_engines(scheduler, time, replica_id, sync_room)
    sync_time = max(sync_room["arrival_times"].values())
    step_batches = dict(sync_room["batches"])
    # vLLM all-reduces the token count over the DP group as each stage forward
    # starts, and an eager forward launches its kernels only after it. Every
    # lane of this forward therefore launches from the latest lane's start.
    launch_start_time = max(
        lane_batch._forward_launch_start_time for lane_batch in step_batches.values()
    )
    for lane_batch in step_batches.values():
        lane_batch._forward_launch_start_time = launch_start_time
    provisional_id = sync_room["provisional_cohort_id"]
    if type(provisional_id) is not int or provisional_id < 0:
        raise RuntimeError(
            f"{mode_name} synchronization room has an invalid provisional step id: "
            f"{provisional_id!r}"
        )
    scheduler._close_forward_step(
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
        sync_stage=sync_stage,
        provisional_id=provisional_id,
    )
    sync_room.pop("batches", None)
    sync_room.pop("arrival_times", None)
    sync_room.pop("provisional_cohort_id", None)

    if scheduler._sync_kind == "forward":
        ready = scheduler._on_forward_ep_wave_ready
    elif scheduler._sync_kind == "prefill":
        ready = scheduler._on_prefill_ep_wave_ready
    else:
        ready = scheduler._on_decode_ep_wave_ready
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
