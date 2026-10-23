from typing import List

from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import EventType, ClusterType


class DecodeSyncCollectiveEvent(BaseEvent):
    """Event to handle MoE decode collective synchronization for a target cluster."""

    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
        sync_stage: str,
        layer_id: int,
        cluster_type: ClusterType = ClusterType.DECODE,
    ):
        super().__init__(time, EventType.DECODE_SYNC_COLLECTIVE)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch_global_id = batch_global_id
        self._sync_stage = sync_stage
        self._layer_id = layer_id
        self._cluster_type = cluster_type

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(
            self._cluster_type
        )
        return cluster_scheduler.on_decode_sync_collective(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_global_id,
            self._sync_stage,
            self._layer_id,
            metrics_store,
        )

    def get_target_cluster(self) -> ClusterType:
        """Return the cluster that should process this collective event."""
        return self._cluster_type

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "cluster_type": self._cluster_type.name,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_global_id": self._batch_global_id,
            "sync_stage": self._sync_stage,
            "layer_id": self._layer_id,
        }

