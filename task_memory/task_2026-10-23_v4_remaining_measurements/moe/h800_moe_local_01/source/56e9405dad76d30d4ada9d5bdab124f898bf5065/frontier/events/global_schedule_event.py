from typing import Dict, List

from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseGlobalScheduler
from frontier.types import EventType

logger = init_logger(__name__)


class GlobalScheduleEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        self._cluster_set = set()
        self._request_mapping = scheduler.schedule()

        # Route queued requests to their target cluster.
        for cluster_type, requests in self._request_mapping.items():
            self._cluster_set.add(cluster_type)
            for request in requests:
                scheduler.get_cluster_scheduler(cluster_type).add_request(request)
        # clean queue
        scheduler.clear_queues()

        # Launch cluster scheduling for each target cluster with new work.
        return [
            ClusterScheduleEvent(self.time, cluster_type)
            for cluster_type in self._cluster_set
        ]

    def to_dict(self):
        request_mapping = []
        for cluster_type, requests in self._request_mapping.items():
            for request in requests:
                request_mapping.append(
                    {"cluster_type": cluster_type.name, "request_id": request.id}
                )
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "cluster_set": [str(cluster_type) for cluster_type in self._cluster_set],
            "request_mapping": request_mapping,
        }
