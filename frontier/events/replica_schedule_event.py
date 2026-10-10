from typing import List

from frontier.entities import DummyForwardBatch
from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler, BaseReplicaScheduler
from frontier.types import EventType, ClusterType

logger = init_logger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(
        self,
        time: float,
        replica_id: int,
        cluster_type: ClusterType,
        replica_local_id: int | None,
    ):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id
        self._cluster_type = cluster_type
        self._replica_local_id = replica_local_id

        self._batches = []

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        replica_scheduler: BaseReplicaScheduler = cluster_scheduler.get_replica_scheduler(self._replica_id, self._replica_local_id)

        # Log replica scheduling details
        pending_requests = replica_scheduler.num_pending_requests
        logger.info(f"Replica scheduling started at {self.time:.3f}s: "
                   f"{self._cluster_type.name} cluster, replica {self._replica_id}, replica_local_id {self._replica_local_id}, "
                   f"pending_requests={pending_requests}")

        waiting_requests = []
        if hasattr(replica_scheduler, "peek_waiting_requests"):
            waiting_requests = list(replica_scheduler.peek_waiting_requests())
        for request in waiting_requests:
            latest_arrival = request.get_cluster_arrival_time(self._cluster_type)
            if latest_arrival > self.time + 1e-9:
                logger.warning(
                    "[STALE-REPLICA-SCHEDULE] Skipping schedule event at %.6fs for "
                    "replica=%s replica_local_id=%s cluster=%s because request %s has a newer "
                    "arrival %.6fs",
                    self.time,
                    self._replica_id,
                    self._replica_local_id,
                    self._cluster_type.name,
                    request.id,
                    latest_arrival,
                )
                return []
            thinking_home_cluster_type = getattr(
                request,
                "thinking_home_cluster_type",
                None,
            )
            if (
                thinking_home_cluster_type is not None
                and self._cluster_type in (ClusterType.DECODE, ClusterType.MONOLITHIC)
            ):
                thinking_home_arrival = request.get_cluster_arrival_time(
                    thinking_home_cluster_type
                )
                if thinking_home_arrival > self.time + 1e-9:
                    logger.warning(
                        "[STALE-REPLICA-SCHEDULE-HOME-QUEUE] Skipping schedule event "
                        "at %.6fs for replica=%s replica_local_id=%s cluster=%s because request %s "
                        "has a newer thinking-home arrival %.6fs in %s",
                        self.time,
                        self._replica_id,
                        self._replica_local_id,
                        self._cluster_type.name,
                        request.id,
                        thinking_home_arrival,
                        thinking_home_cluster_type.name,
                    )
                    return []
        
        # bachting operation based on the replica scheduler (internal engine like orca/vllm/..., )
        # also consider current running batch in pipeline stage
        self._batches = replica_scheduler.on_schedule(self.time)
        events = self._events_for_schedule_pass(replica_scheduler, metrics_store, logger)
        if hasattr(replica_scheduler, "consume_dp_sync_release"):
            events.extend(replica_scheduler.consume_popped_output_events())
            # A pass that ended the DP sync's all-reduce resumes the other
            # engines of the group, whose hosts waited in it.
            events.extend(
                ReplicaScheduleEvent(self.time, self._replica_id, self._cluster_type, lane_id)
                for lane_id in replica_scheduler.consume_dp_sync_release()
            )
        return events

    def _events_for_schedule_pass(
        self, replica_scheduler: BaseReplicaScheduler, metrics_store: MetricsStore, logger
    ) -> List[BaseEvent]:
        from frontier.events.batch_stage_arrival_event import BatchStageArrivalEvent

        # if there are no batches, we return an empty list
        if not self._batches:
            logger.info(f"Replica scheduling completed: no batches formed for replica {self._replica_id}")
            if (
                hasattr(replica_scheduler, "consume_preemption_followup_poll")
                and replica_scheduler.consume_preemption_followup_poll()
            ):
                logger.info(
                    "Replica scheduling completed: emitting one follow-up "
                    "schedule poll after an empty pass that preempted"
                )
                return [
                    ReplicaScheduleEvent(
                        self.time,
                        self._replica_id,
                        self._cluster_type,
                        self._replica_local_id,
                    )
                ]
            if (
                hasattr(
                    replica_scheduler,
                    "consume_monolithic_pp_terminal_release_followup_poll",
                )
                and replica_scheduler.consume_monolithic_pp_terminal_release_followup_poll()
            ):
                logger.info(
                    "Replica scheduling completed: emitting one empty follow-up "
                    "schedule poll after MONOLITHIC+PP terminal release"
                )
                return [
                    ReplicaScheduleEvent(
                        self.time,
                        self._replica_id,
                        self._cluster_type,
                        self._replica_local_id,
                    )
                ]
            if (
                hasattr(
                    replica_scheduler,
                    "consume_monolithic_pp_mtp_output_wait_followup_poll",
                )
                and replica_scheduler.consume_monolithic_pp_mtp_output_wait_followup_poll()
            ):
                logger.info(
                    "Replica scheduling completed: emitting one empty follow-up "
                    "schedule poll after MONOLITHIC+PP MTP output wait"
                )
                return [
                    ReplicaScheduleEvent(
                        self.time,
                        self._replica_id,
                        self._cluster_type,
                        self._replica_local_id,
                    )
                ]
            return []

        # A dummy forward runs on every pipeline stage at once, with no transfer
        # between its parts. It may follow batches of the same pass.
        dummy_forward_arrivals = [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                dummy_forward.pipeline_stage_id,
                dummy_forward,
                self._cluster_type,
                self._replica_local_id,
            )
            for dummy_forward in self._batches
            if isinstance(dummy_forward, DummyForwardBatch)
        ]
        batches = [
            batch for batch in self._batches if not isinstance(batch, DummyForwardBatch)
        ]
        if not batches:
            return dummy_forward_arrivals

        # Log batching results
        total_requests = sum(len(batch.requests) for batch in batches)
        batch_info = []
        for i, batch in enumerate(batches):
            request_ids = [req.id for req in batch.requests]
            batch_info.append(f"batch_{i}(requests={request_ids})")
            
        logger.info(f"Replica scheduling completed: {len(batches)} batches formed with {total_requests} total requests")
        logger.info(f"Batch details: {', '.join(batch_info)}")

        # we get the memory usage percent from the replica scheduler
        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent, self._cluster_type
        )

        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
                self._cluster_type,
                self._replica_local_id,
            )
            for batch in batches
        ] + dummy_forward_arrivals

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "cluster_type": self._cluster_type.name,
            "batch_ids": [batch.id for batch in self._batches],
            "replica_local_id": self._replica_local_id,
        }
