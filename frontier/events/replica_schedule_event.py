from typing import List

from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler, BaseReplicaScheduler
from frontier.types import EventType, ClusterType

logger = init_logger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, cluster_type: ClusterType, dp_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id
        self._cluster_type = cluster_type
        self._dp_id = dp_id

        self._batches = []

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.batch_stage_arrival_event import BatchStageArrivalEvent
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        replica_scheduler: BaseReplicaScheduler = cluster_scheduler.get_dp_replica_scheduler(self._replica_id, self._dp_id)

        # Log replica scheduling details
        pending_requests = replica_scheduler.num_pending_requests
        logger.info(f"Replica scheduling started at {self.time:.3f}s: "
                   f"{self._cluster_type.name} cluster, replica {self._replica_id}, dp_id {self._dp_id}, "
                   f"pending_requests={pending_requests}")

        waiting_requests = []
        if hasattr(replica_scheduler, "peek_waiting_requests"):
            waiting_requests = list(replica_scheduler.peek_waiting_requests())
        for request in waiting_requests:
            latest_arrival = request.get_cluster_arrival_time(self._cluster_type)
            if latest_arrival > self.time + 1e-9:
                logger.warning(
                    "[STALE-REPLICA-SCHEDULE] Skipping schedule event at %.6fs for "
                    "replica=%s dp=%s cluster=%s because request %s has a newer "
                    "arrival %.6fs",
                    self.time,
                    self._replica_id,
                    self._dp_id,
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
                        "at %.6fs for replica=%s dp=%s cluster=%s because request %s "
                        "has a newer thinking-home arrival %.6fs in %s",
                        self.time,
                        self._replica_id,
                        self._dp_id,
                        self._cluster_type.name,
                        request.id,
                        thinking_home_arrival,
                        thinking_home_cluster_type.name,
                    )
                    return []
        
        # bachting operation based on the replica scheduler (internal engine like orca/vllm/..., )
        # also consider current running batch in pipeline stage
        self._batches = replica_scheduler.on_schedule(self.time)

        # if there are no batches, we return an empty list
        if not self._batches:
            logger.info(f"Replica scheduling completed: no batches formed for replica {self._replica_id}")
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
                        self._dp_id,
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
                        self._dp_id,
                    )
                ]
            return []

        # Log batching results
        total_requests = sum(len(batch.requests) for batch in self._batches)
        batch_info = []
        for i, batch in enumerate(self._batches):
            request_ids = [req.id for req in batch.requests]
            batch_info.append(f"batch_{i}(requests={request_ids})")
            
        logger.info(f"Replica scheduling completed: {len(self._batches)} batches formed with {total_requests} total requests")
        logger.info(f"Batch details: {', '.join(batch_info)}")

        # we get the memory usage percent from the replica scheduler
        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent, self._cluster_type
        )

        # record schedule time and status of batch and requests in batch
        try:
            for i, batch in enumerate(self._batches):
                logger.info(f"Processing batch {i}: batch_id={batch.id}, num_requests={len(batch.requests)}")
                batch.on_schedule(self.time, self._cluster_type)

                # Log individual batch scheduling
                request_ids = [req.id for req in batch.requests]
                logger.info(f"Batch {batch.id} scheduled at {self.time:.3f}s in {self._cluster_type.name} cluster, "
                           f"requests={request_ids}, scheduled_at={self.time:.3f}s")
        except Exception as e:
            logger.error(f"Error in batch scheduling loop: {e}")
            logger.error(f"Batch details: num_batches={len(self._batches)}")
            for i, batch in enumerate(self._batches):
                logger.error(f"Batch {i}: id={batch.id}, num_requests={len(batch.requests)}, "
                            f"requests={[req.id for req in batch.requests]}")
            raise

        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
                self._cluster_type,
                self._dp_id,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "cluster_type": self._cluster_type.name,
            "batch_ids": [batch.id for batch in self._batches],
            "dp_id": self._dp_id,
        }
