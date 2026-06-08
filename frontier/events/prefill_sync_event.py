from typing import List

from frontier.entities import Batch
from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import EventType, ClusterType


class PrefillSyncEvent(BaseEvent):
    """Event to handle MoE prefill synchronization points for a target cluster."""

    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        dp_id: int,
        sync_stage: str,
        layer_id: int,
        stage_execution_time: float,
        cluster_type: ClusterType = ClusterType.PREFILL,
    ):
        super().__init__(time, EventType.PREFILL_SYNC)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._dp_id = dp_id
        self._sync_stage = sync_stage  # "pre_moe" or "post_moe"
        self._layer_id = layer_id
        self._stage_execution_time = stage_execution_time
        self._cluster_type = cluster_type

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        # Use cluster logger to ensure visibility in per-cluster event logs
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        logger.info(
            f"[PREFILL_SYNC][{self._sync_stage}] t={self.time:.6f}s, batch_id={self._batch.id}, "
            f"replica={self._replica_id}, stage={self._stage_id}, layer={self._layer_id}, dp={self._dp_id}, "
            f"stage_exec_time={self._stage_execution_time}"
        )

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(
            self._cluster_type
        )
        return cluster_scheduler.on_prefill_sync(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch,
            self._dp_id,
            self._sync_stage,
            self._layer_id,
            self._stage_execution_time,
        )

    def get_target_cluster(self) -> ClusterType:
        """Return the cluster that should process this sync event."""
        return self._cluster_type

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "cluster_type": self._cluster_type.name,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "dp_id": self._dp_id,
            "sync_stage": self._sync_stage,
            "layer_id": self._layer_id,
            "stage_execution_time": self._stage_execution_time,
        }
