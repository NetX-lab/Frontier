from typing import List

from frontier.entities.batch import Batch
from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.types import ClusterType, EventType

logger = init_logger(__name__)


class BatchStageArrivalEvent(BaseEvent):
    def __init__(
        self, time: float, replica_id: int, stage_id: int, batch: Batch, cluster_type: ClusterType, dp_id: int
    ):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._cluster_type = cluster_type
        self._dp_id = dp_id
        self._batch_schedule_epoch = batch.schedule_epoch

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        stage_scheduler = cluster_scheduler.get_dp_replica_stage_scheduler(
            self._replica_id, self._dp_id, self._stage_id
        )
        if self._batch.schedule_epoch != self._batch_schedule_epoch:
            logger.warning(
                "[STALE-BATCH-STAGE-ARRIVAL] Skipping batch %s for stage %s: "
                "expected_schedule_epoch=%s current_schedule_epoch=%s",
                self._batch.id,
                self._stage_id,
                self._batch_schedule_epoch,
                self._batch.schedule_epoch,
            )
            return []
        logger.info(f"BatchStageArrivalEvent: adding batch {self._batch.id} to stage {self._stage_id}, (replica_id, dp_id) = ({self._replica_id}, {self._dp_id})")
        stage_scheduler.add_batch(self._batch)

        if stage_scheduler.is_busy:
            logger.info(
                "BatchStageArrivalEvent: stage already busy for batch %s at stage %s, "
                "skip redundant ReplicaStageScheduleEvent",
                self._batch.id,
                self._stage_id,
            )
            return []

        return [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
                self._cluster_type,
                self._dp_id,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "cluster_type": self._cluster_type.name,
            "dp_id": self._dp_id,
        }
