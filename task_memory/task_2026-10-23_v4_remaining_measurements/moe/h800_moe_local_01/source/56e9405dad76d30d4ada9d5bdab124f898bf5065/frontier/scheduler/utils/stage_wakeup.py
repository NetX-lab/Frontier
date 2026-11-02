"""Wake queued Replica-local stage schedulers after an owner release."""

import math
from numbers import Real
from typing import Any


def build_stage_wakeup_events(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    exclude_replica_local_id: int | None,
) -> list:
    """Create wakeup events for non-empty, idle sibling lanes."""

    if not isinstance(time, Real) or not math.isfinite(float(time)):
        raise ValueError(f"stage wakeup time must be finite, got {time!r}")
    if type(replica_id) is not int or replica_id < 0:
        raise ValueError("replica_id must be an exact non-negative int")
    if type(stage_id) is not int or stage_id < 0:
        raise ValueError("stage_id must be an exact non-negative int")
    from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

    replica_schedulers = getattr(scheduler, "_replica_schedulers", {})
    if not isinstance(replica_schedulers, dict):
        raise RuntimeError("Replica scheduler registry was not initialized")
    events = []
    for (candidate_replica_id, candidate_local_id), replica_scheduler in sorted(
        replica_schedulers.items(), key=lambda item: str(item[0])
    ):
        if candidate_replica_id != replica_id or candidate_local_id == exclude_replica_local_id:
            continue
        stage_scheduler = replica_scheduler.get_replica_stage_scheduler(stage_id)
        if stage_scheduler.is_busy or stage_scheduler.is_empty():
            continue
        events.append(
            ReplicaStageScheduleEvent(
                float(time), replica_id, stage_id, scheduler._cluster_type, candidate_local_id
            )
        )
    return events
