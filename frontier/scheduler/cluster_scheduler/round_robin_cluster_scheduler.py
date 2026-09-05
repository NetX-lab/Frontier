from collections import deque
from typing import Any, List, NamedTuple, Tuple, Optional
from frontier.entities import Batch, Request, EPBatchGroup
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class DecodeFFNReadyGroupPreflight(NamedTuple):
    """Validated, side-effect-free facts for one DECODE_FFN ready group."""

    group: List[Tuple[Batch, Any]]
    source_batches: Tuple[Batch, ...]
    source_batch_ids: Tuple[int, ...]
    group_activation_bytes: int
    requests: Tuple[Request, ...]
    num_tokens: Tuple[int, ...]
    layer_global_id: int
    afd_stage_idx: int
    target_replica_id: int


class RoundRobinClusterScheduler(BaseClusterScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

        # DECODE_ATTN load is tracked per serving Replica.  There is no
        # intra-Replica attention-DP lane dimension.
        self._replica_load_tracker = {}
        if self._cluster_type == ClusterType.DECODE_ATTN:
            replica_ids = list(self._cluster.replicas.keys())
            for replica_id in replica_ids:
                self._replica_load_tracker[replica_id] = 0

        # Decode-attn initial request allocation setup state
        self._decode_attn_initial_allocation_done = False
        self._decode_attn_initial_allocation_allocated_requests = 0
        expected_total_requests = getattr(
            self._request_generator_config, "num_requests", None
        )
        self._decode_attn_expected_total_requests = (
            int(expected_total_requests)
            if expected_total_requests is not None
            else None
        )
        self._decode_attn_request_allocation_threshold = None  # total requests required cluster-wide
        self._initial_allocation_enabled = self._cluster_type == ClusterType.DECODE_ATTN
        if self._initial_allocation_enabled:
            # Use explicit decode_attn_request_allocation_threshold if provided
            explicit_threshold = getattr(self._config, 'decode_attn_request_allocation_threshold', None)
            if explicit_threshold is not None:
                self._decode_attn_request_allocation_threshold = explicit_threshold
            else:
                self._initial_allocation_enabled = False
                self._decode_attn_initial_allocation_done = True

        # Internal buffer for initial wait (decode-attn)
        self._initial_allocation_buffer = []  # type: List[Request]
        self._decode_attn_wave_release_pending_request_ids: set[int] = set()
        self._decode_attn_wave_release_completed_request_ids: set[int] = set()

    def schedule(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests using cluster-type-aware round-robin strategy.

        - PREFILL cluster: Batch processing mode (offline-style)
        - DECODE cluster (PD mode): Priority-based scheduling with batch backfilling
        - DECODE_ATTN cluster (PD+AF mode): Optional initial request allocation with threshold, then A↔F priority dynamic routing
        - DECODE_FFN cluster (PD+AF mode): Batch processing mode with M2N immediate processing
        - Other clusters: Default batch processing mode
        """
        self.sort_requests()

        if self._cluster_type == ClusterType.DECODE:
            return self._schedule_decode_with_priority()
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            initial_mapping = self._try_initial_request_allocation()
            if initial_mapping is not None:
                return initial_mapping
            return self._schedule_dynamic_with_af_priority()
        elif self._cluster_type == ClusterType.DECODE_FFN:
            affected = self.schedule_ffn_with_m2n_immediate()
            return [(replica_id, ep_id, None) for (replica_id, ep_id) in affected]
        else:
            return self._schedule_batch_mode()

    def _try_initial_request_allocation(self) -> Optional[List[Tuple[int, int, Request]]]:
        """
        Optional initial request allocation for DECODE_ATTN with threshold-based batching.

        Returns:
            - [] when still waiting for enough requests (defers scheduling)
            - list[(replica_id, replica_local_id, request)] when requests are allocated
            - None when feature is disabled or already completed (fall back to normal flow)
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        if not self._initial_allocation_enabled:
            return None
        if self._decode_attn_initial_allocation_done:
            return None
        if self._decode_attn_request_allocation_threshold is None:
            # No threshold configured; disable initial allocation path
            self._decode_attn_initial_allocation_done = True
            return None

        if self._should_hold_decode_attn_buffered_wave_release():
            logger.info(
                "[initial allocation] holding buffered later wave until the active wave reaches a global batch end boundary"
            )
            return None

        # Move current queued requests into the initial buffer
        while self._request_queue:
            self._initial_allocation_buffer.append(self._request_queue.pop(0))
            logger.info("[initial allocation] pop 1 req from request queue")

        threshold = self._decode_attn_request_allocation_threshold
        total_expected_requests = getattr(
            self._request_generator_config, "num_requests", None
        )
        if total_expected_requests is None:
            total_expected_requests = getattr(
                self,
                "_decode_attn_expected_total_requests",
                None,
            )
        if total_expected_requests is not None:
            total_expected_requests = int(total_expected_requests)
            self._decode_attn_expected_total_requests = total_expected_requests
        if len(self._initial_allocation_buffer) < threshold:
            allocated_requests = self._decode_attn_initial_allocation_allocated_requests
            should_release_partial_final_wave = (
                total_expected_requests is not None
                and allocated_requests + len(self._initial_allocation_buffer)
                >= total_expected_requests
                and len(self._initial_allocation_buffer) > 0
            )
            if not should_release_partial_final_wave:
                logger.info("[initial allocation] not enough requests, keep buffering")
                return None

            logger.info(
                "[initial allocation] releasing final partial wave: buffered=%s expected_total=%s allocated=%s",
                len(self._initial_allocation_buffer),
                total_expected_requests,
                allocated_requests,
            )
            return self._perform_initial_request_allocation(allocate_all_buffered=True)
        
        logger.info(f"[initial allocation] enough requests:{len(self._initial_allocation_buffer)}, perform initial request allocation next.")
        # Perform the initial request allocation now
        return self._perform_initial_request_allocation()

    def _should_hold_decode_attn_buffered_wave_release(self) -> bool:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return False
        if not self._initial_allocation_enabled:
            return False
        pending_request_ids = getattr(
            self,
            "_decode_attn_wave_release_pending_request_ids",
            set(),
        )
        if not pending_request_ids:
            return False
        return len(self._initial_allocation_buffer) > 0

    def _arm_decode_attn_wave_release_guard(
        self,
        requests_to_allocate: List[Request],
    ) -> None:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return

        if self._initial_allocation_buffer:
            self._decode_attn_wave_release_pending_request_ids = {
                int(request.id) for request in requests_to_allocate
            }
            self._decode_attn_wave_release_completed_request_ids = set()
            return

        self._decode_attn_wave_release_pending_request_ids = set()
        self._decode_attn_wave_release_completed_request_ids = set()

    def _has_ready_decode_attn_buffered_wave(self) -> bool:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return False
        if not self._initial_allocation_enabled:
            return False
        if self._decode_attn_initial_allocation_done:
            return False

        threshold = self._decode_attn_request_allocation_threshold
        if threshold is None:
            return False
        if len(self._initial_allocation_buffer) >= threshold:
            return True

        total_expected_requests = getattr(
            self._request_generator_config,
            "num_requests",
            None,
        )
        if total_expected_requests is None:
            total_expected_requests = getattr(
                self,
                "_decode_attn_expected_total_requests",
                None,
            )
        if total_expected_requests is None:
            return False

        total_expected_requests = int(total_expected_requests)
        allocated_requests = self._decode_attn_initial_allocation_allocated_requests
        return (
            len(self._initial_allocation_buffer) > 0
            and allocated_requests + len(self._initial_allocation_buffer)
            >= total_expected_requests
        )

    def on_decode_attn_global_batch_end(self, time: float, batch) -> List[object]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return []

        pending_request_ids = getattr(
            self,
            "_decode_attn_wave_release_pending_request_ids",
            set(),
        )
        if not pending_request_ids:
            return []

        completed_request_ids = getattr(
            self,
            "_decode_attn_wave_release_completed_request_ids",
            set(),
        )
        batch_request_ids = {
            int(request.id)
            for request in getattr(batch, "requests", [])
            if int(request.id) in pending_request_ids
        }
        if not batch_request_ids:
            return []

        completed_request_ids.update(batch_request_ids)
        if completed_request_ids != pending_request_ids:
            return []

        self._decode_attn_wave_release_pending_request_ids = set()
        self._decode_attn_wave_release_completed_request_ids = set()

        if not self._has_ready_decode_attn_buffered_wave():
            return []

        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        return [ClusterScheduleEvent(time, self._cluster_type)]

    def _perform_initial_request_allocation(
        self,
        *,
        allocate_all_buffered: bool = False,
    ) -> List[Tuple[int, int, Request]]:
        """
        Perform initial request allocation for DECODE_ATTN using two-level distribution strategy.

        Two-level distribution:
        - Level 1: Distribute requests to replica schedulers (round-robin)
        - Level 2: no intra-Replica lane distribution; each request uses the full-stage scheduler

        Returns:
            List of (replica_id, replica_local_id, request) tuples representing request-level assignments
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the accumulated requests up to the threshold
        threshold = self._decode_attn_request_allocation_threshold
        if threshold is None:
            raise ValueError("decode_attn_request_allocation_threshold must be configured")

        allocation_size = (
            len(self._initial_allocation_buffer)
            if allocate_all_buffered
            else threshold
        )
        if allocation_size <= 0:
            return []

        if len(self._initial_allocation_buffer) < allocation_size:
            # Defensive check; should not happen because caller guards threshold
            raise ValueError(f"Insufficient requests in buffer for initial allocation: "
                             f"buffer_size={len(self._initial_allocation_buffer)}, allocation_size={allocation_size}")

        # Take requests for the next cluster-wide wave.
        requests_to_allocate = self._initial_allocation_buffer[:allocation_size]
        self._initial_allocation_buffer = self._initial_allocation_buffer[allocation_size:]

        replica_ids = list(self._cluster.replicas.keys())
        num_replicas = len(replica_ids)

        logger.info(
            f"[INITIAL_ALLOCATION] Starting two-level request allocation: "
            f"total_requests={len(requests_to_allocate)}, num_replicas={num_replicas}"
        )

        # Level 1: Distribute requests to replicas using round-robin
        replica_requests = [[] for _ in range(num_replicas)]
        for idx, request in enumerate(requests_to_allocate):
            replica_idx = idx % num_replicas
            replica_requests[replica_idx].append(request)

        # Each serving Replica has one full-stage scheduler.  The old
        # second-level attention-DP split is retired.
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue

            replica_id = replica_ids[replica_idx]
            request_mapping.extend(
                (replica_id, None, request) for request in requests
            )

        logger.info(
            f"[INITIAL_ALLOCATION] Completed: allocated {len(request_mapping)} requests, "
            f"remaining in buffer={len(self._initial_allocation_buffer)}"
        )

        self._arm_decode_attn_wave_release_guard(requests_to_allocate)

        self._decode_attn_initial_allocation_allocated_requests += len(
            requests_to_allocate
        )
        total_expected_requests = getattr(
            self._request_generator_config, "num_requests", None
        )
        if total_expected_requests is None:
            total_expected_requests = getattr(
                self,
                "_decode_attn_expected_total_requests",
                None,
            )
        if total_expected_requests is not None:
            total_expected_requests = int(total_expected_requests)
            self._decode_attn_expected_total_requests = total_expected_requests
        self._decode_attn_initial_allocation_done = (
            total_expected_requests is not None
            and self._decode_attn_initial_allocation_allocated_requests
            >= total_expected_requests
            and len(self._initial_allocation_buffer) == 0
        )
        return request_mapping

    def _schedule_batch_mode(self) -> List[Tuple[int, int, Request]]:
        """
        Original batch processing logic for prefill cluster and other cluster types.
        Processes all requests in the queue at once using traditional round-robin.
        """

        # First, distribute requests to replicas using round-robin
        replica_requests = [[] for _ in range(self._num_replicas)]
        replica_ids = list(self._cluster.replicas.keys())

        request_idx = 0
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_idx = (self._request_counter + request_idx) % self._num_replicas
            replica_requests[replica_idx].append(request)
            request_idx += 1

        self._request_counter += request_idx

        # Distribute requests across logical DP lanes inside each Replica.
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue

            replica_id = replica_ids[replica_idx]
            for local_idx, request in enumerate(requests):
                dp_id = local_idx % self._replica_dp_size
                request_mapping.append((replica_id, dp_id, request))

        return request_mapping

    def _schedule_decode_with_priority(self) -> List[Tuple[int, int, Request]]:
        """
        Priority-based scheduling for unified DECODE cluster (PD-disaggregation mode).

        This method implements the following priority scheme:
        1. In-progress batches (already being processed) have highest priority
        2. Newly arrived requests from prefill cluster have lower priority

        The scheduler will:
        - Continue processing existing batches until completion
        - Backfill new requests into batches when there's capacity
        - Form new batches from the request queue when replicas are idle

        Returns:
            List of (replica_id, replica_local_id, request) tuples for scheduling
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # For PD-disaggregation mode, we use batch processing similar to PREFILL
        # The priority is implicitly handled by the ReplicaScheduler:
        # - Existing batches continue processing (they're already in the replica scheduler)
        # - New requests are added to the replica scheduler's queue
        # - The replica scheduler will backfill when appropriate

        # KV-cache arrivals in online PD mode are often emitted per request.
        # If each scheduling cycle contains one request, the generic batch-mode
        # split still selects the serving Replica deterministically. There is no
        # retired intra-Replica attention-DP allocation to flatten here.
        request_mapping = self._schedule_decode_lane_round_robin()

        logger.debug(f"[DECODE-PRIORITY] Scheduled {len(request_mapping)} requests across replicas")

        return request_mapping

    def _schedule_decode_lane_round_robin(self) -> List[Tuple[int, int, Request]]:
        """Schedule unified PD decode requests across Replica-local DP lanes."""
        replica_ids = list(self._cluster.replicas.keys())
        if not replica_ids:
            return []

        num_replicas = len(replica_ids)
        request_mapping: List[Tuple[int, int, Request]] = []

        request_idx = 0
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_idx = (self._request_counter + request_idx) % num_replicas
            dp_id = (self._request_counter + request_idx) // num_replicas % self._replica_dp_size
            request_mapping.append((replica_ids[replica_idx], dp_id, request))
            request_idx += 1

        self._request_counter += request_idx
        return request_mapping

    def _schedule_dynamic(self) -> List[Tuple[int, int, Request]]:
        """
        Dynamic load-aware round-robin scheduling for decode-attn cluster.

        This method handles the case where requests arrive incrementally from
        prefill cluster via KV cache transfer. It maintains load awareness
        across serving Replicas and assigns requests to the least loaded Replica
        in a round-robin fashion.
        """
        request_mapping = []

        # Update load tracker with current pending requests from replica-local DP schedulers.
        self._update_load_tracker()

        # Process each request in the queue
        while self._request_queue:
            request = self._request_queue.pop(0)

            # Find the Replica-local DP lane with minimum load.
            # In case of ties, use round-robin to break ties
            min_load = min(self._replica_load_tracker.values())
            candidates = [
                lane
                for lane, load in self._replica_load_tracker.items()
                if load == min_load
            ]

            # Use round-robin among candidates with minimum load
            selected_idx = self._request_counter % len(candidates)
            selected_replica_id, selected_dp_id = candidates[selected_idx]

            # Assign request to selected replica
            request_mapping.append((selected_replica_id, selected_dp_id, request))

            # Update load tracker for the selected replica
            self._replica_load_tracker[(selected_replica_id, selected_dp_id)] += 1

            # Increment request counter for round-robin tie-breaking
            self._request_counter += 1

        return request_mapping

    def _update_load_tracker(self) -> None:
        """
        Update the load tracker with current pending requests from replica schedulers.

        This ensures that the load tracker reflects the actual current load
        including requests that may have been processed or completed since
        the last scheduling round.
        """
        for replica_id, dp_id in self._replica_load_tracker.keys():
            scheduler_key = (replica_id, dp_id)
            current_pending = self._replica_schedulers[scheduler_key].num_pending_requests
            self._replica_load_tracker[(replica_id, dp_id)] = current_pending

    def _schedule_dynamic_with_af_priority(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests for decode-attn cluster with A <-> F priority processing.

        Priority order:
        1. A <-> F batch queue (batches returning from decode-ffn cluster) - maintain original replica/DP assignment
        2. Regular request queue (new requests from prefill cluster) - load-aware dynamic scheduling

        This method implements batch-level scheduling to preserve batch integrity and maintain
        the original replica ID and DP ID mapping for batches returning from decode-ffn cluster,
        following the pattern established in schedule_ffn_with_m2n_immediate().
        """
        request_mapping = []

        # Process A <-> F batch queue with highest priority
        if len(self._af_batch_queue) > 0:
            af_batches = self._af_batch_queue[:]
            self._af_batch_queue.clear()

            # Process each batch returning from decode-ffn cluster
            for batch in af_batches:
                # Preserve batch integrity by returning to the original
                # attention-serving Replica.  There is no local DP lane to
                # restore; the target uses its full-stage scheduler.
                if batch.decode_attn_original_replica_id is None:
                    raise ValueError(
                        f"Batch {batch.id} returning to DECODE_ATTN cluster without original assignment."
                    )

                target_replica_id = batch.decode_attn_original_replica_id
                target_dp_id = batch.decode_attn_original_dp_id
                if target_dp_id is None:
                    raise ValueError(f"Batch {batch.id} returning without original DP lane")

                # Schedule the entire batch to the original Replica.
                scheduler_key = (target_replica_id, target_dp_id)

                # Add the complete batch to the replica scheduler's immediate queue
                # This preserves batch integrity and avoids re-batching overhead
                replica_scheduler = self._replica_schedulers[scheduler_key]

                # Add batch directly to replica scheduler for immediate processing
                # This follows the pattern from schedule_ffn_with_m2n_immediate()
                replica_scheduler.add_batch_to_immediate_queue(batch)

                # Track the affected replica for event scheduling
                # Return the scheduler key as a tuple for ReplicaScheduleEvent creation
                request_mapping.append((target_replica_id, target_dp_id, None))

        # Schedule newly arrived requests from PREFILL using the dynamic load-aware policy.
        if self._request_queue:
            request_mapping.extend(self._schedule_dynamic())

        return request_mapping

    def _get_least_loaded_replica(self) -> Tuple[int, None]:
        """
        Find the serving Replica with the least load.

        Returns:
            Tuple[int, None]: (replica_id, None) for the full-stage scheduler
        """
        # Initialize load tracker if not exists
        if not hasattr(self, '_replica_load_tracker') or not self._replica_load_tracker:
            self._replica_load_tracker = {}
            replica_ids = list(self._cluster.replicas.keys())
            for replica_id in replica_ids:
                self._replica_load_tracker[replica_id] = 0

        # Update load tracker with current pending requests from replica schedulers
        self._update_load_tracker()

        # Find the serving Replica with minimum load.
        min_load = min(self._replica_load_tracker.values())
        candidates = [
            replica_id
            for replica_id, load in self._replica_load_tracker.items()
            if load == min_load
        ]

        # Use round-robin among candidates with minimum load
        selected_idx = self._request_counter % len(candidates)
        return candidates[selected_idx], None

    def _commit_decode_ffn_m2n_queue_operations(self, operations) -> None:
        """Commit prepared DECODE_FFN queue writes as one validated batch."""

        grouped_operations = []
        grouped_by_identity = {}
        for replica_scheduler, batch in operations:
            key = id(replica_scheduler)
            group_index = grouped_by_identity.get(key)
            if group_index is None:
                group_index = len(grouped_operations)
                grouped_by_identity[key] = group_index
                grouped_operations.append((replica_scheduler, []))
            grouped_operations[group_index][1].append(batch)

        prepared_commits = []
        for replica_scheduler, batches in grouped_operations:
            immediate_queue = getattr(
                replica_scheduler,
                "_m2n_immediate_batch_queue",
                None,
            )
            activation_memory = getattr(
                replica_scheduler,
                "_activation_bytes_allocated",
                None,
            )
            if type(immediate_queue) in {list, deque}:
                if type(activation_memory) is not int or activation_memory < 0:
                    raise RuntimeError(
                        "DECODE_FFN replica activation memory must be an exact "
                        f"non-negative int, got {activation_memory!r}"
                    )
                activation_delta = 0
                for batch in batches:
                    activation_bytes = getattr(batch, "activation_bytes", None)
                    if type(activation_bytes) is not int or activation_bytes < 0:
                        raise ValueError(
                            "DECODE_FFN prepared batch activation_bytes must be "
                            f"an exact non-negative int, got {activation_bytes!r}"
                        )
                    activation_delta += activation_bytes
                prepared_commits.append(
                    (replica_scheduler, batches, activation_delta)
                )
                continue

            raise RuntimeError(
                "DECODE_FFN replica scheduler must expose an exact immediate "
                "batch queue for atomic queue commit"
            )

        for target, batches, activation_delta in prepared_commits:
            target._m2n_immediate_batch_queue.extend(batches)
            target._activation_bytes_allocated += activation_delta

    def _preflight_decode_ffn_ready_group(
        self,
        ready_groups,
    ) -> Optional[DecodeFFNReadyGroupPreflight]:
        """Validate the next DECODE_FFN group without constructing entities."""

        if type(ready_groups) is not deque:
            raise RuntimeError(
                "DECODE_FFN ready-group inventory must be an exact deque"
            )

        group_counter = getattr(self, "_batch_group_creation_counter", None)
        if type(group_counter) is not int or group_counter < 0:
            raise ValueError(
                "DECODE_FFN batch-group creation counter must be an exact "
                f"non-negative int, got {group_counter!r}"
            )

        raw_batches = getattr(self, "_raw_batch_waiting_for_m2n_back", None)
        if type(raw_batches) is not dict:
            raise RuntimeError(
                "DECODE_FFN raw waiting-room inventory must be an exact dict"
            )

        if not ready_groups:
            return None

        group = ready_groups[0]
        if type(group) is not list or not group:
            raise ValueError(
                "DECODE_FFN ready group must be an exact non-empty list"
            )

        for entry in group:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(
                    "DECODE_FFN ready-group entries must be exact "
                    "(batch, transfer_info) tuples"
                )

        layer_global_id = self._get_ffn_layer_id_from_group(group)
        source_batches = []
        source_batch_ids = []
        all_requests = []
        all_num_tokens = []
        seen_source_batch_ids = set()
        group_activation_bytes = 0
        afd_stage_idx = None
        target_replica_id = None
        for entry in group:
            source_batch, transfer_info = entry
            if type(source_batch) is not Batch:
                raise ValueError(
                    "DECODE_FFN source batch must be an exact Batch, "
                    f"got {type(source_batch).__name__}"
                )

            requests = getattr(source_batch, "requests", None)
            if type(requests) is not list:
                raise ValueError(
                    "DECODE_FFN source batch requests must be an exact list"
                )
            if not requests:
                raise ValueError(
                    "DECODE_FFN source batch requests must not be empty"
                )
            for request in requests:
                if type(request) is not Request:
                    raise ValueError(
                        "DECODE_FFN source batch requests must contain exact "
                        f"Request objects, got {request!r}"
                    )

            num_tokens = getattr(source_batch, "num_tokens", None)
            if type(num_tokens) is not list:
                raise ValueError(
                    "DECODE_FFN source batch num_tokens must be an exact list"
                )
            if len(requests) != len(num_tokens):
                raise ValueError(
                    "DECODE_FFN source batch request/token length mismatch: "
                    f"requests={len(requests)}, num_tokens={len(num_tokens)}"
                )
            for token_count in num_tokens:
                if type(token_count) is not int or token_count < 0:
                    raise ValueError(
                        "DECODE_FFN source batch num_tokens must contain exact "
                        f"non-negative ints, got {token_count!r}"
                    )

            total_num_tokens = getattr(
                source_batch,
                "total_num_tokens",
                None,
            )
            if (
                type(total_num_tokens) is not int
                or total_num_tokens < 0
                or total_num_tokens != sum(num_tokens)
            ):
                raise ValueError(
                    "DECODE_FFN source batch total_num_tokens must be an exact "
                    "non-negative int equal to the sum of num_tokens, "
                    f"got total={total_num_tokens!r}, num_tokens={num_tokens!r}"
                )

            source_batch_id = getattr(source_batch, "id", None)
            if type(source_batch_id) is not int or source_batch_id < 0:
                raise ValueError(
                    "DECODE_FFN source batch id must be an exact non-negative "
                    f"int, got {source_batch_id!r}"
                )
            if source_batch_id in seen_source_batch_ids:
                raise ValueError(
                    "DECODE_FFN ready group contains duplicate source batch IDs: "
                    f"{source_batch_id!r}"
                )
            if source_batch_id in raw_batches:
                raise ValueError(
                    "DECODE_FFN source batch is already registered in the raw "
                    f"waiting-room inventory: batch_id={source_batch_id}"
                )

            source_stage_idx = getattr(source_batch, "afd_stage_idx", None)
            if source_stage_idx is None:
                raise ValueError("DECODE_FFN source batch afd_stage_idx missing")
            if type(source_stage_idx) is not int or source_stage_idx < 0:
                raise ValueError(
                    "DECODE_FFN source batch afd_stage_idx must be an exact "
                    f"non-negative int, got {source_stage_idx!r}"
                )
            transfer_stage_idx = getattr(
                transfer_info,
                "afd_stage_idx",
                None,
            )
            if type(transfer_stage_idx) is not int or transfer_stage_idx < 0:
                raise ValueError(
                    "DECODE_FFN transfer afd_stage_idx must be an exact "
                    f"non-negative int, got {transfer_stage_idx!r}"
                )
            if transfer_stage_idx != source_stage_idx:
                raise ValueError(
                    "DECODE_FFN source/transfer afd_stage_idx mismatch: "
                    f"source={source_stage_idx}, transfer={transfer_stage_idx}"
                )
            if afd_stage_idx is None:
                afd_stage_idx = source_stage_idx
            elif source_stage_idx != afd_stage_idx:
                raise ValueError(
                    "DECODE_FFN ready group afd_stage_idx mismatch: "
                    f"expected={afd_stage_idx}, got={source_stage_idx}"
                )

            entry_target_replica_id = getattr(
                transfer_info,
                "target_ffn_replica_id",
                None,
            )
            if (
                type(entry_target_replica_id) is not int
                or entry_target_replica_id < 0
            ):
                raise ValueError(
                    "DECODE_FFN target_ffn_replica_id must be an exact "
                    f"non-negative int, got {entry_target_replica_id!r}"
                )
            if target_replica_id is None:
                target_replica_id = entry_target_replica_id
            elif entry_target_replica_id != target_replica_id:
                raise ValueError(
                    "DECODE_FFN ready group target replica mismatch: "
                    f"expected={target_replica_id}, "
                    f"got={entry_target_replica_id}"
                )

            activation_size_bytes = getattr(
                transfer_info,
                "activation_size_bytes",
                None,
            )
            if (
                type(activation_size_bytes) is not int
                or activation_size_bytes < 0
            ):
                raise ValueError(
                    "DECODE_FFN activation_size_bytes must be an exact "
                    f"non-negative int, got {activation_size_bytes!r}"
                )

            seen_source_batch_ids.add(source_batch_id)
            source_batches.append(source_batch)
            source_batch_ids.append(source_batch_id)
            all_requests.extend(requests)
            all_num_tokens.extend(num_tokens)
            group_activation_bytes += activation_size_bytes

        replica_ids = list(self._cluster.replicas.keys())
        if target_replica_id not in replica_ids:
            raise ValueError(
                "DECODE_FFN ready group target replica is not available: "
                f"target={target_replica_id}, available={replica_ids}"
            )

        return DecodeFFNReadyGroupPreflight(
            group=group,
            source_batches=tuple(source_batches),
            source_batch_ids=tuple(source_batch_ids),
            group_activation_bytes=group_activation_bytes,
            requests=tuple(all_requests),
            num_tokens=tuple(all_num_tokens),
            layer_global_id=layer_global_id,
            afd_stage_idx=afd_stage_idx,
            target_replica_id=target_replica_id,
        )

    @staticmethod
    def _preflight_decode_ffn_queue_target(replica_scheduler) -> None:
        """Validate one resolved DECODE_FFN queue target without writing to it."""

        immediate_queue = getattr(
            replica_scheduler,
            "_m2n_immediate_batch_queue",
            None,
        )
        activation_memory = getattr(
            replica_scheduler,
            "_activation_bytes_allocated",
            None,
        )
        if type(immediate_queue) in {list, deque}:
            if type(activation_memory) is not int or activation_memory < 0:
                raise RuntimeError(
                    "DECODE_FFN replica activation memory must be an exact "
                    f"non-negative int, got {activation_memory!r}"
                )
            return

        raise RuntimeError(
            "DECODE_FFN replica scheduler must expose an exact immediate "
            "batch queue for atomic queue commit"
        )

    def schedule_ffn_with_m2n_immediate(self) -> List[Tuple[int, int]]:
        """
        Schedule decode-ffn micro-batches with corrected group aggregation and two-level MoE routing.

        Implements logical aggregation for grouped GEMM without creating new Batch objects:
        - Dry-run per-batch EP/expert allocation using routing_details
        - Aggregate per-expert allocations per EP within the group
        - Annotate EP sub-batches with group metadata and a single representative per EP for billing

        Returns:
            List[(replica_id, ep_id)]: EP lanes affected; outer handler will emit ReplicaScheduleEvents
        """
        from collections import defaultdict
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        affected_ep_pairs: set[Tuple[int, int]] = set()

        ready_groups = getattr(self, "_m2n_ready_groups", None)
        group_preflight = self._preflight_decode_ffn_ready_group(ready_groups)
        if group_preflight is None:
            return []
        group = group_preflight.group
        raw_batches = group_preflight.source_batches
        source_batch_ids = group_preflight.source_batch_ids
        group_activation_bytes = group_preflight.group_activation_bytes
        layer_global_id = group_preflight.layer_global_id
        afd_stage_idx = group_preflight.afd_stage_idx
        target_replica_id = group_preflight.target_replica_id

        model_config = self._config.replica_config.model_config
        if model_config is None:
            raise ValueError("Missing model_config for DECODE_FFN layer classification")

        # Dense layers do not use expert routing or EP collectives, including
        # dense layers inside a mixed dense/MoE model.
        if not model_config.is_moe_layer(layer_global_id):
            return self._schedule_dense_ffn_from_m2n_group(ready_groups, logger)

        ep_size = getattr(self, '_replica_ep_size', self._config.replica_config.moe_expert_parallel_size)
        # total_experts = self._config.replica_config.total_expert_num
        experts_per_ep = self._config.replica_config.local_expert_num


        routing_details = getattr(self._predictor, "_decode_ffn_routing_details", None)
        if routing_details is None:
            raise ValueError("Missing _decode_ffn_routing_details on predictor for DECODE_FFN")


        if ready_groups and len(ready_groups) > 0:
            replica_ids = list(self._cluster.replicas.keys())

            # Extract source batch IDs for diagnostic logging
            logger.info(f"[FFN-GROUP][DEBUG] replica_ids={replica_ids}")

            rd_replicas = list(getattr(routing_details, 'keys', lambda: [])())
            logger.info(
                f"[FFN-GROUP] Consuming group size={len(group)} -> target_replica={target_replica_id}, "
                f"routing_detail_replicas={rd_replicas}, layer_global_id={layer_global_id}"
            )

            # Materialize the global routing vector exactly once for the complete
            # EP wave, then reuse its ownership split for every lane.
            shared_layer_workload = self._materialize_ep_wave_workload(
                group,
                target_replica_id,
                layer_global_id,
                routing_details,
            )

            # Prepare a shared group_global_id so all EP sub-batches share the same global_id.
            # The counter is committed only after all queue writes succeed.
            shared_group_id = self._batch_group_creation_counter

            # DIAGNOSTIC: Log the shared_group_id assignment
            logger.info(f"[EP-GLOBAL-ID][ASSIGN] shared_group_id={shared_group_id} assigned to group with source_batch_ids={source_batch_ids}, layer={layer_global_id}, target_replica={target_replica_id}")

            # Level 2: workload -> ep (corresponding replica)
            # 总体目标：将group中的所有batches按照routing信息，分配到target_replica_id的各个EP中
            # 分配的单位是EPBatchGroup，该实体包含关键metadata (it's a logic batch)

            # 为每个EP 构建一个EPBatchGroup
            created_ep_batches = []  # Track created EP batches for diagnostic logging
            ep_batch_groups = []  # (ep_id, EPBatchGroup)
            queue_operations = []
            replica_schedulers = []
            for ep_id in range(ep_size):
                try:
                    replica_scheduler = self.get_replica_scheduler(
                        target_replica_id,
                        ep_id,
                    )
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        "DECODE_FFN target replica scheduler is unavailable: "
                        f"replica_id={target_replica_id}, ep_id={ep_id}"
                    ) from exc
                self._preflight_decode_ffn_queue_target(replica_scheduler)
                replica_schedulers.append(replica_scheduler)

            for ep_id in range(ep_size):
                # 计算当前ep_id对应experts的global_id (based on offset)
                expert_global_ids = list(range(ep_id * experts_per_ep, ep_id * experts_per_ep + experts_per_ep))
                ep_batch_group: EPBatchGroup = self._distribute_tokens_within_ep_replica(
                    group,
                    target_replica_id,
                    ep_id,
                    expert_global_ids,
                    layer_global_id,
                    routing_details,
                    layer_workload=shared_layer_workload,
                )
                # Ensure all EP sub-batches share the same global_id for AllGather synchronization
                ep_batch_group.set_global_id(shared_group_id)

                # DIAGNOSTIC: Log EP batch creation and global_id assignment
                logger.info(f"[EP-GLOBAL-ID][SET] EPBatchGroup created: batch_id={ep_batch_group.id}, ep_id={ep_id}, global_id={ep_batch_group.global_id}, replica={target_replica_id}, layer={layer_global_id}, source_batches={source_batch_ids}")
                created_ep_batches.append((ep_batch_group.id, ep_id, ep_batch_group.global_id))
                ep_batch_groups.append((ep_id, ep_batch_group))

            # Allocate activation memory proportionally per EP batch group
            group_total_tokens = sum(
                getattr(ep_batch, "total_num_tokens", 0)
                for (_, ep_batch) in ep_batch_groups
            )
            remaining_bytes = group_activation_bytes
            for idx, (ep_id, ep_batch_group) in enumerate(ep_batch_groups):
                if group_total_tokens > 0:
                    if idx == len(ep_batch_groups) - 1:
                        activation_bytes = remaining_bytes
                    else:
                        activation_bytes = int(
                            group_activation_bytes
                            * (ep_batch_group.total_num_tokens / group_total_tokens)
                        )
                        remaining_bytes -= activation_bytes
                else:
                    activation_bytes = 0
                ep_batch_group.activation_bytes = activation_bytes

                replica_scheduler = replica_schedulers[ep_id]
                queue_operations.append((replica_scheduler, ep_batch_group))
                affected_ep_pairs.add((target_replica_id, ep_id))

            # DIAGNOSTIC: Verify all EP batches have the same global_id
            global_ids = [gid for (_, _, gid) in created_ep_batches]
            if len(set(global_ids)) != 1:
                logger.error(f"[EP-GLOBAL-ID][ERROR] EP batches have different global_ids! created_batches={created_ep_batches}")
                raise ValueError(f"EP batches from the same group have different global_ids: {created_ep_batches}")
            else:
                logger.info(f"[EP-GLOBAL-ID][VERIFY] All {len(created_ep_batches)} EP batches share global_id={shared_group_id}")

            logger.info(
                "[FFN-GROUP] Affected EP lanes prepared for commit: "
                f"{sorted(affected_ep_pairs)}"
            )
            stage_context = self.get_stage_execution_context(
                target_replica_id,
                afd_stage_idx,
            )
            stage_ticket = stage_context.enqueue_ep_wave(
                operation_id=shared_group_id,
                participant_ep_ids=tuple(range(ep_size)),
            )
            for _, ep_batch_group in ep_batch_groups:
                ep_batch_group._stage_admission_ticket = stage_ticket
            try:
                self._commit_decode_ffn_m2n_queue_operations(queue_operations)
            except Exception:
                for _, ep_batch_group in ep_batch_groups:
                    ep_batch_group.__dict__.pop("_stage_admission_ticket", None)
                stage_context.cancel(stage_ticket)
                raise
            for source_batch in raw_batches:
                self._raw_batch_waiting_for_m2n_back[source_batch.id] = source_batch
            self._batch_group_creation_counter = shared_group_id + 1
            ready_groups.popleft()

            return sorted(list(affected_ep_pairs))

        return []

    def _schedule_dense_ffn_from_m2n_group(
        self, ready_groups, logger
    ) -> List[Tuple[int, int]]:
        """Schedule a dense (non-MoE) FFN batch from the M2N ready group.

        For dense models there is no expert routing, no EP dispatch/combine,
        and no all-to-all. We aggregate the group into one full-stage batch
        and queue it on the target Replica's dedicated full-stage scheduler.
        """
        from frontier.entities.batch import DenseFFNBatchGroup

        group_preflight = self._preflight_decode_ffn_ready_group(ready_groups)
        if group_preflight is None:
            return []
        raw_batches = group_preflight.source_batches
        source_batch_ids = group_preflight.source_batch_ids
        group_activation_bytes = group_preflight.group_activation_bytes
        layer_global_id = group_preflight.layer_global_id
        afd_stage_idx = group_preflight.afd_stage_idx
        target_replica_id = group_preflight.target_replica_id
        all_requests = list(group_preflight.requests)
        all_num_tokens = list(group_preflight.num_tokens)

        shared_group_id = self._batch_group_creation_counter

        try:
            replica_scheduler = self.get_full_stage_replica_scheduler(
                target_replica_id
            )
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                "DECODE_FFN target replica scheduler is unavailable: "
                f"replica_id={target_replica_id}"
            ) from exc
        self._preflight_decode_ffn_queue_target(replica_scheduler)

        dense_batch = DenseFFNBatchGroup(
            requests=all_requests,
            num_tokens=all_num_tokens,
            replica_id=target_replica_id,
            time=0.0,
            source_batch_ids=source_batch_ids,
            cluster_type=self._cluster_type,
        )
        dense_batch.set_global_id(shared_group_id)
        dense_batch.decode_ffn_layer_id = layer_global_id
        dense_batch.afd_stage_idx = afd_stage_idx
        dense_batch.source_batches = list(raw_batches)
        (
            dense_batch.afd_stage_metadata,
            dense_batch.afd_stage_represents_all_stages,
        ) = self._aggregate_decode_ffn_afd_metadata(raw_batches)
        dense_batch.activation_bytes = group_activation_bytes

        logger.info(
            f"[FFN-GROUP][DENSE] Prepared DenseFFNBatchGroup id={dense_batch.id} "
            f"global_id={shared_group_id} target_replica={target_replica_id} "
            f"layer={layer_global_id} source_batches={source_batch_ids} "
            f"total_tokens={sum(all_num_tokens)}"
        )
        stage_context = self.get_stage_execution_context(
            target_replica_id,
            afd_stage_idx,
        )
        stage_ticket = stage_context.enqueue_full_stage(
            operation_id=shared_group_id,
        )
        dense_batch._stage_admission_ticket = stage_ticket
        try:
            self._commit_decode_ffn_m2n_queue_operations(
                [(replica_scheduler, dense_batch)]
            )
        except Exception:
            dense_batch.__dict__.pop("_stage_admission_ticket", None)
            stage_context.cancel(stage_ticket)
            raise
        for batch in raw_batches:
            self._raw_batch_waiting_for_m2n_back[batch.id] = batch
        self._batch_group_creation_counter = shared_group_id + 1
        ready_groups.popleft()

        return [(target_replica_id, None)]

    @staticmethod
    def _get_ffn_layer_id_from_group(group: List[Tuple["Batch", Any]]) -> int:
        if type(group) is not list or not group:
            raise ValueError(
                "DECODE_FFN layer lookup requires an exact non-empty group list"
            )

        layer_id = None
        for entry in group:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(
                    "DECODE_FFN layer lookup requires exact "
                    "(batch, transfer_info) tuples"
                )
            transfer_info = entry[1]
            entry_layer_id = getattr(transfer_info, "layer_id", None)
            if type(entry_layer_id) is not int or entry_layer_id < 0:
                raise ValueError(
                    "DECODE_FFN layer_id must be an exact non-negative int, "
                    f"got {entry_layer_id!r}"
                )
            if layer_id is None:
                layer_id = entry_layer_id
            elif entry_layer_id != layer_id:
                raise ValueError(
                    "M2N transfer_info layer_id mismatch within DECODE_FFN group: "
                    f"expected={layer_id}, got={entry_layer_id}"
                )
        return layer_id
