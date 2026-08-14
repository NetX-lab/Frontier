from random import randint
from typing import List, Optional, Tuple

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.types import ClusterType


class RandomClusterScheduler(BaseClusterScheduler):
    def schedule(self) -> List[Tuple[int, Optional[int], Request]]:
        """
        Schedule requests with the release-supported monolithic random strategy.
        """
        self.sort_requests()

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        return self._schedule_random()

    def _schedule_random(self) -> List[Tuple[int, Optional[int], Request]]:
        """Original random scheduling logic."""
        # First, distribute requests to replicas randomly
        replica_requests = [[] for _ in range(self._num_replicas)]
        replica_ids = list(self._cluster.replicas.keys())

        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_idx = randint(0, self._num_replicas - 1)
            replica_requests[replica_idx].append(request)

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
        Schedule requests for decode-ffn cluster with M2N immediate processing using random strategy.

        Priority order:
        1. M2N immediate batches (from decode-attn cluster)
        2. Regular request queue
        """
        request_mapping = []

        # First, process M2N immediate batches with highest priority
        if hasattr(self, '_m2n_immediate_batches') and self._m2n_immediate_batches:
            m2n_batches = self._m2n_immediate_batches[:]
            self._m2n_immediate_batches.clear()

            replica_ids = list(self._cluster.replicas.keys())

            for batch in m2n_batches:
                for request in batch.requests:
                    # Randomly select a serving Replica; its target is full-stage.
                    replica_idx = randint(0, self._num_replicas - 1)
                    replica_id = replica_ids[replica_idx]
                    request_mapping.append((replica_id, None, request))

        # Then, process regular request queue using random strategy
        regular_mapping = self._schedule_random()
        request_mapping.extend(regular_mapping)

        return request_mapping

    def _schedule_with_af_priority(self) -> List[Tuple[int, Optional[int], Request]]:
        """
        Schedule requests for decode-attn cluster with A→F priority processing using random strategy.

        Priority order:
        1. A→F batch queue (batches returning from decode-ffn cluster) - maintain original replica/DP assignment
        2. Regular request queue (new requests from prefill cluster) - use random strategy

        This method implements batch-level scheduling to preserve batch integrity and maintain
        the original replica ID and DP ID mapping for batches returning from decode-ffn cluster,
        following the same pattern as other schedulers.
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
                if batch.decode_attn_original_dp_id is not None:
                    raise ValueError(
                        "DECODE_ATTN A-to-F uses full-stage identity; "
                        f"expected original local identity None, got {batch.decode_attn_original_dp_id!r}"
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

        # Process regular request queue using random strategy
        regular_mapping = self._schedule_random()
        request_mapping.extend(regular_mapping)

        return request_mapping
