from typing import List

from frontier.events.base_event import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from frontier.types import ClusterType, EventType

logger = init_logger(__name__)


class ClusterScheduleEvent(BaseEvent):
    def __init__(self, time: float, cluster_type: ClusterType):
        super().__init__(time, EventType.CLUSTER_SCHEDULE)
        self._cluster_type = cluster_type

    def __repr__(self):
        return f"ClusterScheduleEvent(time={self.time}, cluster_type={self._cluster_type})"

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        self._dp_replica_set = set()
        self._ep_replica_set = set()
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)

        # For DECODE_FFN, use EP-aware MoE routing inside the cluster scheduler; log M2N queue size
        if self._cluster_type == ClusterType.DECODE_FFN:
            m2n_count = len(getattr(cluster_scheduler, "_m2n_immediate_batches", []))
            logger.info(f"DECODE_FFN cluster scheduling: processing {m2n_count} M2N immediate batches")

        # Log cluster scheduling details
        queue_size = len(cluster_scheduler._request_queue)
        logger.info(f"Cluster scheduling started at {self.time:.3f}s: "
                   f"{self._cluster_type.name} cluster with {queue_size} requests in queue")

        self._request_mapping = cluster_scheduler.schedule()

        # DEBUG: Log request mapping
        mapping_summary = {}
        for replica_id, dp_id, request in self._request_mapping:
            key = (replica_id, dp_id)
            if key not in mapping_summary:
                mapping_summary[key] = []
            if request is not None:
                mapping_summary[key].append(request.id)

        debug_msg = f"[CLUSTER_SCHEDULE] cluster={self._cluster_type.name}, request_mapping={mapping_summary}"
        logger.debug(debug_msg)

        # replica[num_replica][dp_size]
        for replica_id, dp_id, request in self._request_mapping:
            self._dp_replica_set.add((replica_id, dp_id))
            # Only add individual requests to replica scheduler
            # Batch-level assignments (request=None) are already handled by the cluster scheduler
            if request is not None:
                if self._cluster_type in [ClusterType.MONOLITHIC, ClusterType.PREFILL]:
                    request.bind_thinking_home_queue(
                        self._cluster_type, replica_id, dp_id
                    )
                cluster_scheduler.get_dp_replica_scheduler(replica_id, dp_id).add_request(request)

        # For each (replica_id, dp_id) that has been assigned a request, trigger a replica schedule event.
        return [
            ReplicaScheduleEvent(self.time, replica_id, self._cluster_type, dp_id)
            for replica_id, dp_id in self._dp_replica_set
        ]


    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "cluster_type": str(self._cluster_type),
            "replica_dp_set": list(self._dp_replica_set),
            "replica_ep_set": list(self._ep_replica_set),
            "request_mapping": [
                (replica_id, dp_id, (request.id if request is not None else None))
                for replica_id, dp_id, request in self._request_mapping
            ],
        }
