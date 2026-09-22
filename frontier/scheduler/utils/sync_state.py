"""Layer synchronization waiting-room initialization."""

from collections import defaultdict
from typing import Any

from frontier.types import ClusterType


def _new_sync_waiting_room():
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                )
            )
        )
    )


def initialize_sync_waiting_rooms(scheduler: Any) -> None:
    """Initialize layer-sync rooms for PREFILL, MONOLITHIC, or DECODE.

    A monolithic Replica runs prefill and decode on the same lanes, so one
    forward may hold a prefill batch on one lane and a decode batch on another.
    Those lanes have to wait in the *same* room or neither ever reaches the
    expected lane count. The monolithic cluster therefore gets one room, bound
    to `_forward_sync_waiting_room` and to both phase names that existing call
    sites select by mode. A disaggregated role runs one phase and keeps its own
    single room.
    """
    cluster_type = scheduler._cluster_type
    model_config = scheduler._config.replica_config.model_config
    model_is_moe = model_config is not None and model_config.is_moe

    if cluster_type is ClusterType.MONOLITHIC:
        shared_room = _new_sync_waiting_room() if model_is_moe else None
        scheduler._forward_sync_waiting_room = shared_room
        scheduler._prefill_sync_waiting_room = shared_room
        scheduler._decode_sync_waiting_room = shared_room
        return

    scheduler._forward_sync_waiting_room = None

    if cluster_type is ClusterType.PREFILL:
        scheduler._prefill_sync_waiting_room = (
            _new_sync_waiting_room() if model_is_moe else None
        )
        scheduler._decode_sync_waiting_room = None
        return

    if cluster_type is ClusterType.DECODE:
        scheduler._prefill_sync_waiting_room = None
        scheduler._decode_sync_waiting_room = (
            _new_sync_waiting_room() if model_is_moe else None
        )
