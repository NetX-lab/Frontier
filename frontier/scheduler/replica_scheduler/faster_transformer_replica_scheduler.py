from typing import Sequence

from frontier.entities.batch import Batch
from frontier.entities.request import Request
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.types import ClusterType


class FasterTransformerReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_batches = []
        self._num_running_batches = 0
        self._pending_free_map = {}
        self._pending_kv_transfer_requests = set()

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        if batch.all_requests_completed:
            # free memory for all requests at once
            self.free_batch(batch)
            self.free(*self._pending_free_map.pop(batch.id, []))
            return

        if self._cluster_type == ClusterType.PREFILL:
            for request in batch.requests:
                if (
                    request.is_prefill_complete
                    and request.num_decode_tokens > 0
                    and not request.completed
                ):
                    # Retain PREFILL-side KV until the disaggregated release path frees it.
                    self._pending_kv_transfer_requests.add(request.id)

        if any(
            not request.completed
            and request.id not in self._pending_kv_transfer_requests
            for request in batch.requests
        ):
            self._preempted_batches.append(batch)

    def _generate_next_batch_from_preempted(self, preempted_batch: Batch) -> Batch:
        requests = []
        num_tokens = []

        for request in preempted_batch.requests:
            if (
                request.completed
                or request.id in self._pending_kv_transfer_requests
                or (
                    self._cluster_type == ClusterType.PREFILL
                    and request.is_prefill_complete
                    and request.num_decode_tokens > 0
                )
            ):
                continue
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return self._create_batch(requests, num_tokens)

    def _free_request_resources(self, request: Request) -> None:
        self.free(request.id)

    def complete_kv_transfer_for_requests(
        self, requests: Sequence[Request]
    ) -> None:
        for request in requests:
            if request.id not in self._pending_kv_transfer_requests:
                raise ValueError(
                    "KV transfer completion for request without pending transfer state: "
                    f"request_id={request.id}, "
                    f"source_cluster={self._cluster_type.name}, "
                    f"source_replica={self._replica_id}, "
                    f"source_dp={self._dp_id}"
                )

            if request.id in self._allocation_map:
                self._free_request_resources(request)
            self._pending_kv_transfer_requests.discard(request.id)

    @property
    def num_pending_requests(self) -> int:
        preempted_request_count = 0
        for batch in self._preempted_batches:
            for request in batch.requests:
                if (
                    request.completed
                    or request.id in self._pending_kv_transfer_requests
                ):
                    continue
                preempted_request_count += 1
        return len(self._request_queue) + preempted_request_count

    def _get_next_batch(self, is_micro_batch: bool = False) -> Batch:
        if self._preempted_batches:
            preempted_batch = self._preempted_batches.pop(0)
            return self._generate_next_batch_from_preempted(preempted_batch)

        requests = []
        num_tokens = []

        while self._request_queue:
            if len(requests) == self._max_batch_size:
                break

            if not self.can_allocate(self._max_blocks_per_sequence):
                break

            request = self._request_queue.pop(0)
            self.allocate(request.id, self._max_blocks_per_sequence)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return self._create_batch(requests, num_tokens)
