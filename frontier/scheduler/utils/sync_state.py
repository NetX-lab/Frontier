"""Layer synchronization waiting-room initialization."""

from collections import defaultdict
from typing import Any

from frontier.types import ClusterType


#: Synchronization kind of each cluster role whose lanes step MoE layers
#: together. A monolithic Replica runs prefill and decode on the same lanes, so
#: one forward may hold a prefill batch on one lane and a decode batch on
#: another: its lanes synchronize as one "forward". A disaggregated role runs
#: one phase.
SYNC_KIND_BY_CLUSTER_TYPE = {
    ClusterType.MONOLITHIC: "forward",
    ClusterType.PREFILL: "prefill",
    ClusterType.DECODE: "decode",
}


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
    """Give a layer-synchronizing cluster its waiting room and its sync kind.

    Every lane of the cluster waits in `_sync_waiting_room`, whatever its local
    phase: two lanes of one monolithic forward that wait in different rooms
    never reach the expected lane count. `_sync_kind` selects the wave and
    completion handlers. A dense model has no MoE layer to synchronize and gets
    no room.
    """
    model_config = scheduler._config.replica_config.model_config
    scheduler._sync_kind = SYNC_KIND_BY_CLUSTER_TYPE[scheduler._cluster_type]
    scheduler._sync_waiting_room = (
        _new_sync_waiting_room() if model_config.is_moe else None
    )
