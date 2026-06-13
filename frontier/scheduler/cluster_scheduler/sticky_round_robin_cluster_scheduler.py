from __future__ import annotations

from typing import List, Tuple

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType


class StickyRoundRobinClusterScheduler(RoundRobinClusterScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_counter = 0
        self._session_to_target_map: dict[int, tuple[int, int]] = {}

    def _get_ordered_targets(self) -> list[tuple[int, int]]:
        replica_ids = list(self._cluster.replicas.keys())
        return [
            (replica_id, dp_id)
            for replica_id in replica_ids
            for dp_id in range(self._replica_dp_size)
        ]

    def _get_target_for_request(self, request: Request) -> tuple[int, int]:
        if request.session_id is None:
            raise ValueError(
                "session_id is required for sticky_round_robin cluster scheduler"
            )
        if request.session_id not in self._session_to_target_map:
            targets = self._get_ordered_targets()
            target = targets[self._session_counter % len(targets)]
            self._session_counter += 1
            self._session_to_target_map[request.session_id] = target
        return self._session_to_target_map[request.session_id]

    def _schedule_batch_mode(self) -> List[Tuple[int, int, Request]]:
        request_mapping: List[Tuple[int, int, Request]] = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id, dp_id = self._get_target_for_request(request)
            request_mapping.append((replica_id, dp_id, request))
        return request_mapping

    def schedule(self) -> List[Tuple[int, int, Request]]:
        self.sort_requests()
        cluster_type = getattr(self, "_cluster_type", None)
        if cluster_type == ClusterType.DECODE:
            return self._schedule_decode_with_priority()
        if cluster_type == ClusterType.DECODE_ATTN:
            initial_mapping = self._try_initial_request_allocation()
            if initial_mapping is not None:
                return initial_mapping
            return self._schedule_dynamic_with_af_priority()
        if cluster_type == ClusterType.DECODE_FFN:
            affected = self.schedule_ffn_with_m2n_immediate()
            return [(replica_id, ep_id, None) for (replica_id, ep_id) in affected]
        return self._schedule_batch_mode()
