from typing import List

from frontier.entities import Request
from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseGlobalScheduler
from frontier.types import ClusterType, EventType


class ThinkingRoundRequeueEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        if request.thinking_home_cluster_type is None:
            raise ValueError(
                "ThinkingRoundRequeueEvent requires a bound home cluster."
            )
        super().__init__(time, EventType.THINKING_ROUND_REQUEUE)
        self._request = request
        self._cluster_type = request.thinking_home_cluster_type

    def handle_event(
        self,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        del metrics_store

        cluster_type = self._request.thinking_home_cluster_type
        replica_id = self._request.thinking_home_replica_id
        dp_id = self._request.thinking_home_dp_id
        if cluster_type is None or replica_id is None or dp_id is None:
            raise ValueError(
                "ThinkingRoundRequeueEvent cannot run without full home-lane affinity."
            )
        if cluster_type not in [ClusterType.MONOLITHIC, ClusterType.PREFILL]:
            raise ValueError(
                "ThinkingRoundRequeueEvent only supports MONOLITHIC or PREFILL "
                f"home queues, got={cluster_type.name}"
            )

        cluster_scheduler = scheduler.get_cluster_scheduler(cluster_type)
        replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(replica_id, dp_id)

        self._request.finish_thinking_tool_wait_and_requeue(self.time)
        replica_scheduler.add_request(self._request)

        return [ReplicaScheduleEvent(self.time, replica_id, cluster_type, dp_id)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
            "cluster_type": self._cluster_type.name,
            "replica_id": self._request.thinking_home_replica_id,
            "dp_id": self._request.thinking_home_dp_id,
        }
