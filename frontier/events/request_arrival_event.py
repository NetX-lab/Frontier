import typing
from typing import List

from frontier.entities import Request
from frontier.events.base_event import BaseEvent
from frontier.logger import get_cluster_logger
from frontier.metrics import MetricsStore
from frontier.types import ClusterType

if typing.TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler

from frontier.types import EventType



class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request, cluster_type: ClusterType) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request
        self._cluster_type = cluster_type

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.global_schedule_event import GlobalScheduleEvent
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        logger.debug(
            f"Request {self._request.id} arrived at {self.time:.3f}s → {self._cluster_type.name} cluster "
            f"(prefill_tokens={self._request.num_prefill_tokens}, decode_tokens={self._request.num_decode_tokens})"
        )

        self._request.on_arrival(self.time, self._cluster_type)
        scheduler.add_request(self._request, self._cluster_type)
        metrics_store.on_request_arrival(self.time, self._request, self._cluster_type)

        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
            "cluster_type": self._cluster_type.name,
        }
