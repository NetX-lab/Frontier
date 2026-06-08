from typing import List, TYPE_CHECKING

from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import EventType, ClusterType

if TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler


class EPAllToAllCombineCollectiveEvent(BaseEvent):
    """
    Event to handle EP AllToAll combine collective synchronization in decode-ffn cluster.

    This event is triggered when all EP replicas within a replica have completed
    their expert computation and are ready for AllToAll combine aggregation.

    Similar to PrefillSyncCollectiveEvent, this handles collective synchronization
    completion in EP processing.
    """

    def __init__(self, time: float, replica_id: int, stage_id: int, batch_global_id: int):
        super().__init__(time, EventType.EP_ALLTOALL_COMBINE_COLLECTIVE)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch_global_id = batch_global_id
        self._cluster_type = ClusterType.DECODE_FFN

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        logger.debug(f"EPAllToAllCombineCollectiveEvent triggered at {self.time:.3f}s: "
                     f"replica_id={self._replica_id}, stage_id={self._stage_id}, batch_global_id={self._batch_global_id}")

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)

        result_events = cluster_scheduler.on_ep_alltoall_combine_collective_schedule(
            self.time, self._replica_id, self._stage_id, self._batch_global_id, metrics_store
        )

        logger.debug(f"EPAllToAllCombineCollectiveEvent generated {len(result_events)} events: "
                     f"{[event.event_type.name if event and hasattr(event, 'event_type') and event.event_type else 'Unknown' for event in result_events]}")

        return result_events

    def get_target_cluster(self) -> ClusterType:
        """
        EP AllToAll combine collective events are always processed by the decode-ffn cluster.

        Returns:
            DECODE_FFN cluster type
        """
        return ClusterType.DECODE_FFN

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_global_id": self._batch_global_id,
            "cluster_type": self._cluster_type.name,
        }
