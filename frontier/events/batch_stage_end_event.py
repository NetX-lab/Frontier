from typing import List

from frontier.entities.batch import Batch
from frontier.entities.batch_stage import BatchStage
from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseGlobalScheduler
from frontier.types import EventType, ClusterType

logger = init_logger(__name__)


class BatchStageEndEvent(BaseEvent):
    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        batch: Batch,
        batch_stage: BatchStage,
        cluster_type: ClusterType,
        dp_id: int,
    ):
        super().__init__(time, EventType.BATCH_STAGE_END)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._cluster_type = cluster_type
        self._dp_id = dp_id

        self._batch = batch
        self._batch_stage = batch_stage
        self._batch_schedule_epoch = batch.schedule_epoch
        self._request_execution_signatures = batch.request_execution_signatures
        self._thinking_round_start_times = batch.thinking_round_start_times

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
        from frontier.events.batch_stage_arrival_event import BatchStageArrivalEvent
        from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

        if self._batch.schedule_epoch != self._batch_schedule_epoch:
            logger.warning(
                "[STALE-BATCH-STAGE-END] Skipping batch %s: expected_schedule_epoch=%s "
                "current_schedule_epoch=%s",
                self._batch.id,
                self._batch_schedule_epoch,
                self._batch.schedule_epoch,
            )
            return []

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        stage_scheduler = cluster_scheduler.get_dp_replica_stage_scheduler(
            self._replica_id, self._dp_id, self._stage_id
        )
        stage_scheduler.on_stage_end() # update status: _is_busy

        self._batch_stage.on_stage_end(self.time)
        metrics_store.on_batch_stage_end(
            self._batch_stage,
            self.time,
            self._replica_id,
            self._stage_id,
            self._cluster_type,
            self._dp_id,
        )

        next_events = []

        # Only trigger a same-stage follow-up schedule when queued work remains.
        # Otherwise this event would be a guaranteed no-op.
        if not stage_scheduler.is_empty():
            next_events.append(ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
                self._cluster_type,
                self._dp_id,
            ))

        if self._is_last_stage:
            return next_events + [
                ClusterBatchEndEvent(
                    self.time,
                    self._replica_id,
                    self._batch,
                    self._cluster_type,
                    self._dp_id,
                    batch_schedule_epoch=self._batch_schedule_epoch,
                    request_execution_signatures=self._request_execution_signatures,
                    thinking_round_start_times=self._thinking_round_start_times,
                )
            ]

        return next_events + [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                self._stage_id + 1,
                self._batch,
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
            "batch_stage_id": self._batch_stage.id,
            "is_last_stage": self._is_last_stage,
            "cluster_type": self._cluster_type.name,
            "dp_id": self._dp_id,
        }

    def to_chrome_trace(self) -> dict:
        return self._batch_stage.to_chrome_trace(self.time)
