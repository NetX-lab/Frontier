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
    """Initialize layer-sync rooms for PREFILL, MONOLITHIC, or DECODE."""
    cluster_type = scheduler._cluster_type
    model_config = scheduler._config.replica_config.model_config
    model_is_moe = model_config is not None and model_config.is_moe

    if cluster_type in (ClusterType.PREFILL, ClusterType.MONOLITHIC):
        if model_is_moe:
            scheduler._prefill_sync_waiting_room = _new_sync_waiting_room()
            scheduler._decode_sync_waiting_room = (
                _new_sync_waiting_room()
                if cluster_type is ClusterType.MONOLITHIC
                else None
            )
        else:
            scheduler._prefill_sync_waiting_room = None
            scheduler._decode_sync_waiting_room = None
        return

    if cluster_type is ClusterType.DECODE:
        scheduler._prefill_sync_waiting_room = None
        scheduler._decode_sync_waiting_room = (
            _new_sync_waiting_room() if model_is_moe else None
        )
