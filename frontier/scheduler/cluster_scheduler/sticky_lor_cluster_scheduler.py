from __future__ import annotations

from typing import List, Tuple

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import (
    LORClusterScheduler,
)
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.types import ClusterType


class StickyLORClusterScheduler(LORClusterScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_to_target_map: dict[int, tuple[int, int]] = {}

    def _get_current_pending_by_target(self) -> dict[tuple[int, int], int]:
        pending_by_target: dict[tuple[int, int], int] = {}
        for target, scheduler in self._dp_replica_schedulers.items():
            pending_by_target[target] = int(scheduler.num_pending_requests)
        return pending_by_target

    def _pick_least_loaded_target(
        self, pending_by_target: dict[tuple[int, int], int]
    ) -> tuple[int, int]:
        return min(
            pending_by_target.items(),
            key=lambda item: (item[1], item[0][0], item[0][1]),
        )[0]

    def _schedule_lor(self) -> List[Tuple[int, int, Request]]:
        request_mapping: List[Tuple[int, int, Request]] = []
        pending_by_target = self._get_current_pending_by_target()

        while self._request_queue:
            request = self._request_queue.pop(0)
            if request.session_id is None:
                raise ValueError(
                    "session_id is required for sticky_lor cluster scheduler"
                )
            if request.session_id not in self._session_to_target_map:
                self._session_to_target_map[request.session_id] = (
                    self._pick_least_loaded_target(pending_by_target)
                )
            target = self._session_to_target_map[request.session_id]
            pending_by_target[target] += 1
            request_mapping.append((target[0], target[1], request))

        return request_mapping

    def schedule(self) -> List[Tuple[int, int, Request]]:
        self.sort_requests()
        cluster_type = getattr(self, "_cluster_type", None)
        if cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        return self._schedule_lor()
