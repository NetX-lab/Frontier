from typing import List, Tuple

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.types import ClusterType


class LORClusterScheduler(BaseClusterScheduler):
    """
    Least outstanding requests (LOR) cluster scheduler.
    """

    def schedule(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests with the release-supported monolithic LOR strategy.
        """
        self.sort_requests()

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        return self._schedule_lor()

    def _schedule_lor(self) -> List[Tuple[int, int, Request]]:
        """Original LOR scheduling logic."""
        # First, distribute requests to replicas using LOR strategy
        replica_requests = [[] for _ in range(self._num_replicas)]
        replica_ids = list(self._cluster.replicas.keys())
        
        # Keep a map of replica_id -> total pending requests across all dp_ids
        pending_requests_map = {}
        for replica_id in replica_ids:
            total_pending = 0
            for dp_id in range(self._replica_dp_size):
                scheduler_key = (replica_id, dp_id)
                total_pending += self._dp_replica_schedulers[scheduler_key].num_pending_requests
            pending_requests_map[replica_id] = total_pending

        # Distribute requests using LOR strategy
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            replica_idx = replica_ids.index(replica_id)
            replica_requests[replica_idx].append(request)
            pending_requests_map[replica_id] += 1

        # Then, distribute requests within each replica to dp_ids evenly
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue
                
            replica_id = replica_ids[replica_idx]
            num_requests = len(requests)
            
            # Assert that requests can be evenly distributed among dp_ids
            assert num_requests % self._replica_dp_size == 0, \
                f"Cannot evenly distribute {num_requests} requests among {self._replica_dp_size} dp_ids for replica {replica_id}"
            
            requests_per_dp = num_requests // self._replica_dp_size
            
            for dp_id in range(self._replica_dp_size):
                start_idx = dp_id * requests_per_dp
                end_idx = start_idx + requests_per_dp
                for request in requests[start_idx:end_idx]:
                    request_mapping.append((replica_id, dp_id, request))

        return request_mapping

    def _schedule_with_m2n_immediate(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests for decode-ffn cluster with M2N immediate processing using LOR.

        Priority order:
        1. M2N immediate batches (from decode-attn cluster)
        2. Regular request queue
        """
        request_mapping = []

        # First, process M2N immediate batches with highest priority
        if hasattr(self, '_m2n_immediate_batches') and self._m2n_immediate_batches:
            m2n_batches = self._m2n_immediate_batches[:]
            self._m2n_immediate_batches.clear()

            # Use LOR strategy for M2N immediate batches
            replica_ids = list(self._cluster.replicas.keys())
            pending_requests_map = {}
            for replica_id in replica_ids:
                total_pending = 0
                for dp_id in range(self._replica_dp_size):
                    scheduler_key = (replica_id, dp_id)
                    total_pending += self._dp_replica_schedulers[scheduler_key].num_pending_requests
                pending_requests_map[replica_id] = total_pending

            for batch in m2n_batches:
                for request in batch.requests:
                    # Find replica with least outstanding requests
                    replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
                    # Find least loaded dp_id within that replica
                    dp_id = 0
                    min_pending = float('inf')
                    for dp in range(self._replica_dp_size):
                        scheduler_key = (replica_id, dp)
                        pending = self._dp_replica_schedulers[scheduler_key].num_pending_requests
                        if pending < min_pending:
                            min_pending = pending
                            dp_id = dp

                    request_mapping.append((replica_id, dp_id, request))
                    pending_requests_map[replica_id] += 1

        # Then, process regular request queue using LOR
        regular_mapping = self._schedule_lor()
        request_mapping.extend(regular_mapping)

        return request_mapping

    def _schedule_with_af_priority(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests for decode-attn cluster with A→F priority processing using LOR.

        Priority order:
        1. A→F batch queue (batches returning from decode-ffn cluster) - maintain original replica/DP assignment
        2. Regular request queue (new requests from prefill cluster) - use LOR strategy

        This method implements batch-level scheduling to preserve batch integrity and maintain
        the original replica ID and DP ID mapping for batches returning from decode-ffn cluster,
        following the same pattern as the round-robin scheduler.
        """
        request_mapping = []

        # Process A→F batch queue with highest priority
        if len(self._af_batch_queue) > 0:
            af_batches = self._af_batch_queue[:]
            self._af_batch_queue.clear()

            # Process each batch returning from decode-ffn cluster
            for batch in af_batches:
                # Preserve batch integrity by scheduling the entire batch to its original replica/DP assignment
                if (
                    batch.decode_attn_original_replica_id is None
                    or batch.decode_attn_original_dp_id is None
                ):
                    raise ValueError(
                        f"Batch {batch.id} returning to DECODE_ATTN cluster without original assignment."
                    )
                
                original_replica_id = batch.decode_attn_original_replica_id
                target_dp_id = batch.decode_attn_original_dp_id

                # Schedule the entire batch to the selected replica and DP
                scheduler_key = (original_replica_id, target_dp_id)

                # Add the complete batch to the replica scheduler's immediate queue
                if scheduler_key in self._dp_replica_schedulers:
                    replica_scheduler = self._dp_replica_schedulers[scheduler_key]
                    replica_scheduler.add_batch_to_immediate_queue(batch)

                    # Track the affected replica for event scheduling
                    request_mapping.append((original_replica_id, target_dp_id, None))  # None indicates batch-level scheduling
                else:
                    # Fallback: if original replica is not available, use LOR for individual requests
                    for request in batch.requests:
                        replica_id, dp_id = self._get_least_loaded_replica_lor()
                        request_mapping.append((replica_id, dp_id, request))

        # Process regular request queue using LOR strategy
        regular_mapping = self._schedule_lor()
        request_mapping.extend(regular_mapping)

        return request_mapping

    def _get_least_loaded_replica_lor(self) -> Tuple[int, int]:
        """
        Find the (replica_id, dp_id) combination with least outstanding requests using LOR strategy.

        Returns:
            Tuple[int, int]: (replica_id, dp_id) with minimum pending requests
        """
        min_pending = float('inf')
        best_replica_id, best_dp_id = 0, 0

        for (replica_id, dp_id), scheduler in self._dp_replica_schedulers.items():
            pending = scheduler.num_pending_requests
            if pending < min_pending:
                min_pending = pending
                best_replica_id, best_dp_id = replica_id, dp_id

        return best_replica_id, best_dp_id
