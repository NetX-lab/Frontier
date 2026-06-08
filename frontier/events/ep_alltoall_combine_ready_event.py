from typing import List, TYPE_CHECKING

from frontier.entities import Batch
from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import EventType, ClusterType

if TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler


class EPAllToAllCombineReadyEvent(BaseEvent):
    """
    Event triggered when an EP replica completes expert computation and is ready for AllToAll combine.

    This event is used in decode-ffn cluster to signal that an EP replica has finished
    processing its assigned experts and is ready to participate in AllToAll combine
    communication to aggregate results from all EP replicas.

    Similar to PrefillSyncEvent, this handles synchronization points in EP processing.
    """

    def __init__(self, time: float, replica_id: int, stage_id: int, batch: Batch, ep_id: int):
        super().__init__(time, EventType.EP_ALLTOALL_COMBINE_READY)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._ep_id = ep_id  # Expert parallel replica ID
        self._cluster_type = ClusterType.DECODE_FFN

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # DIAGNOSTIC: Log EP AllToAll combine ready with global_id
        logger.info(f"[EP-ALLTOALL-COMBINE-READY] time={self.time:.3f}s, "
                   f"batch_id={self._batch.id}, global_id={self._batch.global_id}, "
                   f"replica={self._replica_id}, stage={self._stage_id}, ep_id={self._ep_id}")

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)

        # Call the cluster scheduler method and log the result
        result_events = cluster_scheduler.on_ep_alltoall_combine_ready(
            self.time, self._replica_id, self._stage_id, self._batch, self._ep_id
        )

        logger.info(f"[EP-ALLTOALL-COMBINE-READY] Generated {len(result_events)} events: "
                   f"{[event.event_type.name if event and hasattr(event, 'event_type') and event.event_type else 'Unknown' for event in result_events]}")

        return result_events

    def get_target_cluster(self) -> ClusterType:
        """
        EP AllToAll combine ready events are always processed by the decode-ffn cluster.

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
            "batch_id": self._batch.id,
            "ep_id": self._ep_id,
            "cluster_type": self._cluster_type.name,
        }
