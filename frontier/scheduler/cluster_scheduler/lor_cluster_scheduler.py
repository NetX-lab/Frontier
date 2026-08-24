from typing import List, Optional, Tuple

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

    def schedule(self) -> List[Tuple[int, Optional[int], Request]]:
        """
        Schedule requests with the release-supported monolithic LOR strategy.
        """
        self.sort_requests()

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        return self._schedule_lor()

    def _schedule_lor(self) -> List[Tuple[int, Optional[int], Request]]:
        """Original LOR scheduling logic."""
        # First, distribute requests to replicas using LOR strategy
        replica_requests = [[] for _ in range(self._num_replicas)]
        replica_ids = list(self._cluster.replicas.keys())
        
        # Keep a map of Replica -> pending requests in its full-stage child.
        pending_requests_map = {}
        for replica_id in replica_ids:
            scheduler_key = (replica_id, None)
            pending_requests_map[replica_id] = self._replica_schedulers[
                scheduler_key
            ].num_pending_requests

        # Distribute requests using LOR strategy
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            replica_idx = replica_ids.index(replica_id)
            replica_requests[replica_idx].append(request)
            pending_requests_map[replica_id] += 1

        # A non-FFN Replica has one full-stage child; no local DP lane exists.
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue
                
            replica_id = replica_ids[replica_idx]
            request_mapping.extend((replica_id, None, request) for request in requests)

        return request_mapping

    def _schedule_with_m2n_immediate(self) -> List[Tuple[int, Optional[int], Request]]:
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

            # Use LOR strategy for M2N immediate batches.
            replica_ids = list(self._cluster.replicas.keys())
            pending_requests_map = {}
            for replica_id in replica_ids:
                scheduler_key = (replica_id, None)
                pending_requests_map[replica_id] = self._replica_schedulers[
                    scheduler_key
                ].num_pending_requests

            for batch in m2n_batches:
                for request in batch.requests:
                    # Find replica with least outstanding requests
                    replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
                    request_mapping.append((replica_id, None, request))
                    pending_requests_map[replica_id] += 1

        # Then, process regular request queue using LOR
        regular_mapping = self._schedule_lor()
        request_mapping.extend(regular_mapping)

        return request_mapping

    def _schedule_with_af_priority(self) -> List[Tuple[int, Optional[int], Request]]:
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
                if batch.decode_attn_original_replica_id is None:
                    raise ValueError(
                        f"Batch {batch.id} returning to DECODE_ATTN cluster without original Replica assignment."
                    )
                if batch.decode_attn_original_replica_local_id is not None:
                    raise ValueError(
                        "DECODE_ATTN A-to-F uses full-stage identity; "
                        f"expected original local identity None, got {batch.decode_attn_original_replica_local_id!r}"
                    )
                
                original_replica_id = batch.decode_attn_original_replica_id

                # Schedule the entire batch to the selected replica and DP
                scheduler_key = (original_replica_id, None)

                # Add the complete batch to the replica scheduler's immediate queue
                if scheduler_key in self._replica_schedulers:
                    replica_scheduler = self._replica_schedulers[scheduler_key]
                    replica_scheduler.add_batch_to_immediate_queue(batch)

                    # Track the affected replica for event scheduling
                    request_mapping.append((original_replica_id, None, None))  # None request indicates batch-level scheduling
                else:
                    raise ValueError(
                        "A-to-F batch references a Replica outside the DECODE_ATTN cluster"
                    )

        # Process regular request queue using LOR strategy
        regular_mapping = self._schedule_lor()
        request_mapping.extend(regular_mapping)

        return request_mapping

    def _get_least_loaded_replica_lor(self) -> Tuple[int, Optional[int]]:
        """
        Find the least-loaded serving Replica using LOR strategy.

        Returns:
            Tuple[int, Optional[int]]: (replica_id, None) with minimum pending requests
        """
        min_pending = float('inf')
        best_replica_id = None

        for (replica_id, replica_local_id), scheduler in self._replica_schedulers.items():
            if replica_local_id is not None:
                continue
            pending = scheduler.num_pending_requests
            if pending < min_pending:
                min_pending = pending
                best_replica_id = replica_id

        if best_replica_id is None:
            raise ValueError("No full-stage Replica scheduler is available")
        return best_replica_id, None
