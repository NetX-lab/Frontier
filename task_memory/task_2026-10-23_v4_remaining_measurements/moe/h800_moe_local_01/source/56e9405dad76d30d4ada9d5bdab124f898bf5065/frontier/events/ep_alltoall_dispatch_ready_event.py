from typing import List, TYPE_CHECKING

from frontier.entities import Batch
from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler


class EPAllToAllDispatchReadyEvent(BaseEvent):
    """Event emitted when an EP lane is ready to enter dispatch collective."""

    def __init__(
        self, time: float, replica_id: int, stage_id: int, batch: Batch, ep_id: int
    ) -> None:
        super().__init__(time, EventType.EP_ALLTOALL_DISPATCH_READY)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._ep_id = ep_id
        self._cluster_type = ClusterType.DECODE_FFN

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(
            self._cluster_type
        )
        return cluster_scheduler.on_ep_alltoall_dispatch_ready(
            self.time, self._replica_id, self._stage_id, self._batch, self._ep_id
        )

    def get_target_cluster(self) -> ClusterType:
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
