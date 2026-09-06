"""Periodic scheduling setup and initial event construction."""

from typing import Any, List

from frontier.types import ClusterType


def configure_periodic_scheduling(scheduler: Any, config: Any, logger) -> None:
    """Set scheduler fields and validate the supported periodic path."""

    scheduler._is_periodic_scheduling_enabled = (
        scheduler._cluster_type in config.periodic_scheduling_clusters
    )
    scheduler._periodic_scheduling_interval_ms = config.periodic_scheduling_interval_ms
    if not scheduler._is_periodic_scheduling_enabled:
        return
    if scheduler._cluster_type is not ClusterType.DECODE_ATTN:
        raise NotImplementedError(
            "Periodic scheduling is not implemented for cluster type "
            f"{scheduler._cluster_type.name}. Currently only DECODE_ATTN is supported."
        )
    logger.info(
        f"Periodic scheduling enabled for {scheduler._cluster_type.name} cluster "
        f"with interval {scheduler._periodic_scheduling_interval_ms}ms"
    )


def build_initial_periodic_events(
    scheduler: Any, start_time: float, logger
) -> List[Any]:
    """Build the first periodic event when scheduling is enabled."""

    if not scheduler._is_periodic_scheduling_enabled:
        return []
    from frontier.events.periodic_schedule_event import PeriodicScheduleEvent

    interval_ms = scheduler._periodic_scheduling_interval_ms
    first_schedule_time = start_time + interval_ms / 1000.0
    logger.info(
        f"Initializing periodic scheduling for {scheduler._cluster_type.name} cluster: "
        f"first event at {first_schedule_time:.3f}s, interval={interval_ms}ms"
    )
    return [PeriodicScheduleEvent(first_schedule_time, scheduler._cluster_type, interval_ms)]
