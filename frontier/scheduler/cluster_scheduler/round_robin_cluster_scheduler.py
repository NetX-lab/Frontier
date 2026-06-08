from typing import Any, List, Tuple, Optional
from frontier.entities import Request, EPBatchGroup
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.types import ClusterType


class RoundRobinClusterScheduler(BaseClusterScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

        # For decode-attn cluster: track load per (replica_id, dp_id) for dynamic scheduling
        # This maintains the current load state for load-aware round-robin scheduling
        self._replica_dp_load_tracker = {}
        if self._cluster_type == ClusterType.DECODE_ATTN:
            replica_ids = list(self._cluster.replicas.keys())
            for replica_id in replica_ids:
                for dp_id in range(self._replica_dp_size):
                    self._replica_dp_load_tracker[(replica_id, dp_id)] = 0

        # Decode-attn initial request allocation setup state
        self._decode_attn_initial_allocation_done = False
        self._decode_attn_request_allocation_threshold = None  # total requests required cluster-wide
        self._initial_allocation_enabled = self._cluster_type == ClusterType.DECODE_ATTN
        if self._initial_allocation_enabled:
            # Use explicit decode_attn_request_allocation_threshold if provided
            explicit_threshold = getattr(self._config, 'decode_attn_request_allocation_threshold', None)
            if explicit_threshold is not None:
                self._decode_attn_request_allocation_threshold = explicit_threshold
            else:
                raise ValueError("decode_attn_request_allocation_threshold not configured")

        # Internal buffer for initial wait (decode-attn)
        self._initial_allocation_buffer = []  # type: List[Request]

    def schedule(self) -> List[Tuple[int, int, Request]]:
        """
        Schedule requests with the release-supported monolithic round-robin strategy.
        """
        self.sort_requests()

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        return self._schedule_batch_mode()

    def _try_initial_request_allocation(self) -> Optional[List[Tuple[int, int, Request]]]:
        """
        Optional initial request allocation for DECODE_ATTN with threshold-based batching.

        Returns:
            - [] when still waiting for enough requests (defers scheduling)
            - list[(replica_id, dp_id, request)] when requests are allocated
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

        # Move current queued requests into the initial buffer
        while self._request_queue:
            self._initial_allocation_buffer.append(self._request_queue.pop(0))
            logger.info("[initial allocation] pop 1 req from request queue")

        if len(self._initial_allocation_buffer) < self._decode_attn_request_allocation_threshold:
            # Keep waiting until threshold is reached
            logger.info("[initial allocation] not enough requests, keep waiting")
            return []
        
        logger.info(f"[initial allocation] enough requests:{len(self._initial_allocation_buffer)}, perform initial request allocation next.")
        # Perform the initial request allocation now
        return self._perform_initial_request_allocation()

    def _perform_initial_request_allocation(self) -> List[Tuple[int, int, Request]]:
        """
        Perform initial request allocation for DECODE_ATTN using two-level distribution strategy.

        Two-level distribution:
        - Level 1: Distribute requests to replica schedulers (round-robin)
        - Level 2: Within each replica, distribute to DP replicas (round-robin)

        Returns:
            List of (replica_id, dp_id, request) tuples representing request-level assignments
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the accumulated requests up to the threshold
        threshold = self._decode_attn_request_allocation_threshold
        if len(self._initial_allocation_buffer) < threshold:
            # Defensive check; should not happen because caller guards threshold
            raise ValueError(f"Insufficient requests in buffer for initial allocation: "
                             f"buffer_size={len(self._initial_allocation_buffer)}, threshold={threshold}")

        # Take requests up to the threshold
        requests_to_allocate = self._initial_allocation_buffer[:threshold]
        remaining_buffer = self._initial_allocation_buffer[threshold:]
        self._initial_allocation_buffer = []
        if remaining_buffer:
            self._request_queue = remaining_buffer + self._request_queue

        replica_ids = list(self._cluster.replicas.keys())
        dp_size = self._replica_dp_size
        num_replicas = len(replica_ids)

        logger.info(
            f"[INITIAL_ALLOCATION] Starting two-level request allocation: "
            f"total_requests={len(requests_to_allocate)}, num_replicas={num_replicas}, dp_size={dp_size}"
        )

        # Level 1: Distribute requests to replicas using round-robin
        replica_requests = [[] for _ in range(num_replicas)]
        for idx, request in enumerate(requests_to_allocate):
            replica_idx = idx % num_replicas
            replica_requests[replica_idx].append(request)

        # Level 2: Within each replica, distribute to DP replicas
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue

            replica_id = replica_ids[replica_idx]
            num_requests = len(requests)

            # Calculate requests per DP replica
            requests_per_dp = num_requests // dp_size
            extra_requests = num_requests % dp_size

            logger.info(
                f"[INITIAL_ALLOCATION] Replica {replica_id}: "
                f"total_requests={num_requests}, requests_per_dp={requests_per_dp}, extra={extra_requests}"
            )

            # Distribute requests to DP replicas
            current_idx = 0
            for dp_id in range(dp_size):
                # First 'extra_requests' dp_ids get one extra request
                num_requests_for_this_dp = requests_per_dp + (1 if dp_id < extra_requests else 0)

                for _ in range(num_requests_for_this_dp):
                    if current_idx < len(requests):
                        request_mapping.append((replica_id, dp_id, requests[current_idx]))
                        current_idx += 1

        logger.info(
            f"[INITIAL_ALLOCATION] Completed: allocated {len(request_mapping)} requests, "
            f"remaining in buffer={len(self._initial_allocation_buffer)}"
        )

        # Mark initial allocation as complete
        self._decode_attn_initial_allocation_done = True
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

        # Then, distribute requests within each replica to dp_ids as evenly as possible
        request_mapping = []
        for replica_idx, requests in enumerate(replica_requests):
            if not requests:
                continue

            replica_id = replica_ids[replica_idx]
            num_requests = len(requests)

            # Distribute requests as evenly as possible among dp_ids.
            # For unified DECODE with MoE, missing sync participants are created by
            # decode-sync idle batches; a real request must not be replicated across lanes.
            requests_per_dp = num_requests // self._replica_dp_size
            extra_requests = num_requests % self._replica_dp_size

            current_idx = 0
            for dp_id in range(self._replica_dp_size):
                # First 'extra_requests' dp_ids get one extra request
                num_requests_for_this_dp = requests_per_dp + (1 if dp_id < extra_requests else 0)

                for _ in range(num_requests_for_this_dp):
                    if current_idx < len(requests):
                        request_mapping.append((replica_id, dp_id, requests[current_idx]))
                        current_idx += 1

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
            List of (replica_id, dp_id, request) tuples for scheduling
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
        # split restarts its intra-replica DP allocation from dp_id=0 every time,
        # leaving decode DP lanes 1..N idle. Use a flattened (replica, dp) round
        # robin for DECODE so single-arrival cycles still exercise all DP lanes.
        request_mapping = self._schedule_decode_lane_round_robin()

        logger.debug(f"[DECODE-PRIORITY] Scheduled {len(request_mapping)} requests across replicas")

        return request_mapping

    def _schedule_decode_lane_round_robin(self) -> List[Tuple[int, int, Request]]:
        """Schedule unified PD decode requests across flattened replica-DP lanes.

        The lane order preserves the historical replica-level round-robin first:
        (replica0, dp0), (replica1, dp0), ..., then (replica0, dp1), ...
        Thus ``dp_size == 1`` keeps the previous behavior, while ``dp_size > 1``
        avoids pinning repeated one-request scheduling cycles to ``dp_id == 0``.
        """
        replica_ids = list(self._cluster.replicas.keys())
        if not replica_ids:
            return []

        num_replicas = len(replica_ids)
        total_lanes = num_replicas * self._replica_dp_size
        request_mapping: List[Tuple[int, int, Request]] = []

        request_idx = 0
        while self._request_queue:
            request = self._request_queue.pop(0)
            lane_idx = (self._request_counter + request_idx) % total_lanes
            replica_idx = lane_idx % num_replicas
            dp_id = lane_idx // num_replicas
            request_mapping.append((replica_ids[replica_idx], dp_id, request))
            request_idx += 1

        self._request_counter += request_idx
        return request_mapping

    def _schedule_dynamic(self) -> List[Tuple[int, int, Request]]:
        """
        Dynamic load-aware round-robin scheduling for decode-attn cluster.

        This method handles the case where requests arrive incrementally from
        prefill cluster via KV cache transfer. It maintains load awareness
        across (replica_id, dp_id) combinations and assigns requests to the
        least loaded replicas in a round-robin fashion.
        """
        request_mapping = []

        # Update load tracker with current pending requests from replica schedulers
        self._update_load_tracker()

        # Process each request in the queue
        while self._request_queue:
            request = self._request_queue.pop(0)

            # Find the (replica_id, dp_id) combination with minimum load
            # In case of ties, use round-robin to break ties
            min_load = min(self._replica_dp_load_tracker.values())
            candidates = [
                (replica_id, dp_id) for (replica_id, dp_id), load
                in self._replica_dp_load_tracker.items()
                if load == min_load
            ]

            # Use round-robin among candidates with minimum load
            selected_idx = self._request_counter % len(candidates)
            selected_replica_id, selected_dp_id = candidates[selected_idx]

            # Assign request to selected replica
            request_mapping.append((selected_replica_id, selected_dp_id, request))

            # Update load tracker for the selected replica
            self._replica_dp_load_tracker[(selected_replica_id, selected_dp_id)] += 1

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
        for (replica_id, dp_id) in self._replica_dp_load_tracker.keys():
            scheduler_key = (replica_id, dp_id)
            current_pending = self._dp_replica_schedulers[scheduler_key].num_pending_requests
            self._replica_dp_load_tracker[(replica_id, dp_id)] = current_pending

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
                # Preserve batch integrity by scheduling the entire batch to its original replica/DP assignment
                # The batch should maintain its original replica_id from when it was first scheduled in decode-attn
                if (
                    batch.decode_attn_original_replica_id is None
                    or batch.decode_attn_original_dp_id is None
                ):
                    raise ValueError(
                        f"Batch {batch.id} returning to DECODE_ATTN cluster without original assignment."
                    )

                target_replica_id = batch.decode_attn_original_replica_id
                target_dp_id = batch.decode_attn_original_dp_id

                # Schedule the entire batch to the original replica and target DP
                scheduler_key = (target_replica_id, target_dp_id)

                # Add the complete batch to the replica scheduler's immediate queue
                # This preserves batch integrity and avoids re-batching overhead
                replica_scheduler = self._dp_replica_schedulers[scheduler_key]

                # Add batch directly to replica scheduler for immediate processing
                # This follows the pattern from schedule_ffn_with_m2n_immediate()
                replica_scheduler.add_batch_to_immediate_queue(batch)

                # Track the affected replica for event scheduling
                # Return the scheduler key as a tuple for ReplicaScheduleEvent creation
                request_mapping.append((target_replica_id, target_dp_id, None))  # None indicates batch-level scheduling

        # Schedule newly arrived requests from PREFILL using the dynamic load-aware policy.
        if self._request_queue:
            request_mapping.extend(self._schedule_dynamic())

        return request_mapping

    def _get_least_loaded_replica(self) -> Tuple[int, int]:
        """
        Find the (replica_id, dp_id) combination with the least load.

        Returns:
            Tuple[int, int]: (replica_id, dp_id) with minimum load
        """
        # Initialize load tracker if not exists
        if not hasattr(self, '_replica_dp_load_tracker') or not self._replica_dp_load_tracker:
            self._replica_dp_load_tracker = {}
            replica_ids = list(self._cluster.replicas.keys())
            for replica_id in replica_ids:
                for dp_id in range(self._replica_dp_size):
                    self._replica_dp_load_tracker[(replica_id, dp_id)] = 0

        # Update load tracker with current pending requests from replica schedulers
        self._update_load_tracker()

        # Find the (replica_id, dp_id) combination with minimum load
        min_load = min(self._replica_dp_load_tracker.values())
        candidates = [
            (replica_id, dp_id) for (replica_id, dp_id), load
            in self._replica_dp_load_tracker.items()
            if load == min_load
        ]

        # Use round-robin among candidates with minimum load
        selected_idx = self._request_counter % len(candidates)
        return candidates[selected_idx]

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

        ready_groups = self._m2n_ready_groups
        assert ready_groups is not None, "M2N ready groups not found in decode-ffn cluster scheduler"

        ep_size = getattr(self, '_replica_ep_size', self._config.replica_config.moe_expert_parallel_size)
        # total_experts = self._config.replica_config.total_expert_num
        experts_per_ep = self._config.replica_config.local_expert_num


        routing_details = getattr(self._predictor, "_decode_ffn_routing_details", None)
        if routing_details is None:
            raise ValueError("Missing _decode_ffn_routing_details on predictor for DECODE_FFN")


        if ready_groups and len(ready_groups) > 0:
            group = ready_groups.popleft()  # List[(batch, transfer_info)]
            replica_ids = list(self._cluster.replicas.keys())
            # group elements are (batch, transfer_info)
            layer_global_id = self._get_ffn_layer_id_from_group(group)

            # Extract source batch IDs for diagnostic logging
            source_batch_ids = [batch.id for (batch, _) in group]
            logger.info(f"[FFN-GROUP][DEBUG] replica_ids={replica_ids}")

            target_replica_ids = {
                getattr(transfer_info, "target_ffn_replica_id", None)
                for (_, transfer_info) in group
            }
            if len(target_replica_ids) != 1:
                raise ValueError(
                    "DECODE_FFN ready group must map to exactly one target replica; "
                    f"got {sorted(target_replica_ids)}"
                )
            target_replica_id = next(iter(target_replica_ids))
            if target_replica_id not in replica_ids:
                raise ValueError(
                    "DECODE_FFN ready group target replica is not available: "
                    f"target={target_replica_id}, available={replica_ids}"
                )
            rd_replicas = list(getattr(routing_details, 'keys', lambda: [])())
            logger.info(
                f"[FFN-GROUP] Consuming group size={len(group)} -> target_replica={target_replica_id}, "
                f"routing_detail_replicas={rd_replicas}, layer_global_id={layer_global_id}"
            )

            # Prepare a shared group_global_id so all EP sub-batches share the same global_id
            shared_group_id = self._batch_group_creation_counter
            self._batch_group_creation_counter += 1

            # DIAGNOSTIC: Log the shared_group_id assignment
            logger.info(f"[EP-GLOBAL-ID][ASSIGN] shared_group_id={shared_group_id} assigned to group with source_batch_ids={source_batch_ids}, layer={layer_global_id}, target_replica={target_replica_id}")

            # Level 2: workload -> ep (corresponding replica)
            # 总体目标：将group中的所有batches按照routing信息，分配到target_replica_id的各个EP中
            # 分配的单位是EPBatchGroup，该实体包含关键metadata (it's a logic batch)

            # 为每个EP 构建一个EPBatchGroup
            created_ep_batches = []  # Track created EP batches for diagnostic logging
            ep_batch_groups = []  # (ep_id, EPBatchGroup)
            group_activation_bytes = int(
                sum(
                    getattr(transfer_info, "activation_size_bytes", 0)
                    for (_, transfer_info) in group
                )
            )
            for ep_id in range(ep_size):
                # 计算当前ep_id对应experts的global_id (based on offset)
                expert_global_ids = list(range(ep_id * experts_per_ep, ep_id * experts_per_ep + experts_per_ep))
                ep_batch_group: EPBatchGroup = self._distribute_tokens_within_ep_replica(
                    group, target_replica_id, ep_id, expert_global_ids, layer_global_id, routing_details
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

                self.get_dp_replica_scheduler(target_replica_id, ep_id).add_batch_to_m2n_queue(ep_batch_group)
                affected_ep_pairs.add((target_replica_id, ep_id))

            # DIAGNOSTIC: Verify all EP batches have the same global_id
            global_ids = [gid for (_, _, gid) in created_ep_batches]
            if len(set(global_ids)) != 1:
                logger.error(f"[EP-GLOBAL-ID][ERROR] EP batches have different global_ids! created_batches={created_ep_batches}")
                raise ValueError(f"EP batches from the same group have different global_ids: {created_ep_batches}")
            else:
                logger.info(f"[EP-GLOBAL-ID][VERIFY] All {len(created_ep_batches)} EP batches share global_id={shared_group_id}")

            logger.info(f"[FFN-GROUP] Affected EP lanes: {sorted(list(affected_ep_pairs))}")
            return sorted(list(affected_ep_pairs))

        return []

    @staticmethod
    def _get_ffn_layer_id_from_group(group: List[Tuple["Batch", Any]]) -> int:
        layer_id = getattr(group[0][1], "layer_id", None)
        if layer_id is None:
            raise ValueError("Missing layer_id in M2N transfer_info for DECODE_FFN group")
        for (_, transfer_info) in group[1:]:
            if getattr(transfer_info, "layer_id", None) != layer_id:
                raise ValueError(
                    "M2N transfer_info layer_id mismatch within DECODE_FFN group: "
                    f"expected={layer_id}, got={getattr(transfer_info, 'layer_id', None)}"
                )
        return layer_id

        #     # Dry-run: build EP assignments for each batch (not enqueued yet)
        #     per_batch_ep_assignments = []  # List[Dict[(replica_id, ep_id), ep_batch]]
        #     for (batch, _ti) in group:
        #         ep_assignments = self._distribute_tokens_within_replica(batch, target_replica_id)
        #         per_batch_ep_assignments.append(ep_assignments)

        #     # Group-level aggregation: sum expert allocations per ep_id across the group
        #     ep_to_group_alloc = defaultdict(lambda: defaultdict(int))  # ep_id -> {expert_id: tokens}
        #     ep_to_batches = defaultdict(list)  # ep_id -> List[Batch]

        #     for ep_assignments in per_batch_ep_assignments:
        #         for (replica_id_key, ep_id), ep_batch in ep_assignments.items():
        #             if replica_id_key != target_replica_id:
        #                 raise ValueError("EP assignment replica_id mismatch with target_replica_id")
        #             alloc = getattr(ep_batch, "moe_expert_token_allocation", None)
        #             if not isinstance(alloc, dict):
        #                 raise ValueError("Missing moe_expert_token_allocation on EP batch before grouping")
        #             for gid, tok in alloc.items():
        #                 if tok < 0:
        #                     raise ValueError("Negative tokens in moe_expert_token_allocation")
        #                 ep_to_group_alloc[ep_id][gid] += tok
        #             ep_to_batches[ep_id].append(ep_batch)

        #     # Assign group metadata and enqueue to EP lanes with single representative per ep_id
        #     if not hasattr(self, "_group_id_counter"):
        #         self._group_id_counter = 0
        #     group_id = self._group_id_counter
        #     self._group_id_counter += 1
        #     group_size = len(group)

        #     for ep_id, ep_batches in ep_to_batches.items():
        #         if not ep_batches:
        #             raise ValueError("Empty EP batch list for a grouped ep_id")
        #         group_alloc = dict(ep_to_group_alloc[ep_id])

        #         # Mark exactly one representative per ep_id
        #         for idx, ep_batch in enumerate(ep_batches):
        #             setattr(ep_batch, "group_id", group_id)
        #             setattr(ep_batch, "group_size", group_size)
        #             setattr(ep_batch, "group_ep_agg_expert_allocation", group_alloc)
        #             setattr(ep_batch, "is_group_compute_representative", idx == 0)

        #     # Enqueue all EP batches and collect affected lanes
        #     for ep_assignments in per_batch_ep_assignments:
        #         for (replica_id_key, ep_id), ep_batch in ep_assignments.items():
        #             self.get_dp_replica_scheduler(replica_id_key, ep_id).add_batch_to_m2n_queue(ep_batch)
        #             affected_ep_pairs.add((replica_id_key, ep_id))

        #     logger.info(f"[FFN-GROUP] Affected EP lanes: {sorted(list(affected_ep_pairs))}")
        #     return sorted(list(affected_ep_pairs))

        # # No ready groups: no scheduling action (idempotent)
        # return []

    def _distribute_batches_to_replicas_round_robin(self, m2n_batches) -> dict:
        """
        Level 1: Distribute M2N batches to replica schedulers using round-robin allocation.

        Args:
            m2n_batches: List of (batch, transfer_info) tuples from M2N transfer

        Returns:
            Dict[replica_id, List[Tuple[batch, transfer_info]]]: Batches assigned to each replica
        """
        replica_ids = list(self._cluster.replicas.keys())
        replica_batch_assignments = {replica_id: [] for replica_id in replica_ids}

        # Distribute batches using round-robin based on batch counter
        for i, batch_tuple in enumerate(m2n_batches):
            target_replica_id = replica_ids[i % len(replica_ids)]
            replica_batch_assignments[target_replica_id].append(batch_tuple)

        return replica_batch_assignments

    def _distribute_tokens_within_replica(self, batch, replica_id) -> dict:
        """
        Level 2: Distribute tokens of a batch across EP replicas within a specific replica.

        Uses predictor routing_details to perform realistic token→EP allocation with per-request conservation
        and per-EP per-expert allocation for Grouped GEMM, with strict Fail Fast error handling.

        Args:
            batch: The original batch to be distributed (from decode-attn → decode-ffn)
            replica_id: The target decode-ffn replica ID selected at Level 1

        Returns:
            Dict[(replica_id, ep_id), Batch]: EP batch assignments with correct token allocation
        """
        from collections import defaultdict
        from frontier.entities import Batch
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Preconditions & config fetch
        if self._predictor is None:
            raise ValueError("Execution time predictor is required for token→expert allocation")
        routing_details = getattr(self._predictor, "_decode_ffn_routing_details", None)
        if routing_details is None:
            raise ValueError("Missing _decode_ffn_routing_details on predictor for DECODE_FFN")

        ep_size = getattr(self, '_replica_ep_size', self._config.replica_config.moe_expert_parallel_size)
        total_experts = self._config.replica_config.total_expert_num
        router_topk = self._config.replica_config.router_topk
        if total_experts % ep_size != 0:
            raise ValueError(
                f"total_experts ({total_experts}) must be divisible by ep_size ({ep_size})"
            )
        experts_per_ep = total_experts // ep_size

        # Determine current global layer id for routing
        try:
            layer_id = batch.af_inflight_layer_count
        except Exception as e:
            raise ValueError(f"Cannot determine af_inflight_layer_count for batch {batch.id}: {e}")

        if replica_id not in routing_details:
            raise KeyError(f"Replica {replica_id} not found in routing details for DECODE_FFN")
        if layer_id not in routing_details[replica_id]:
            raise KeyError(
                f"Layer {layer_id} not found in routing details for replica {replica_id}"
            )
        expert_ratios: dict = routing_details[replica_id][layer_id]

        total_batch_tokens = batch.num_routing_tokens # sum(batch.num_tokens)
        assert total_batch_tokens == batch.total_num_tokens, "num_routing_tokens mismatch"
        total_expert_tokens = total_batch_tokens * router_topk

        # Step 1: compute per-EP total tokens (top-k expanded), with rounding correction
        ep_tokens_total = []
        ep_tokens_float = []
        for ep_id in range(ep_size):
            start = ep_id * experts_per_ep
            end = start + experts_per_ep
            ep_share = sum(expert_ratios.get(gid, 0.0) for gid in range(start, end))
            float_tokens = total_expert_tokens * ep_share
            ep_tokens_float.append((float_tokens, ep_id))
            ep_tokens_total.append(int(round(float_tokens)))

        diff = total_expert_tokens - sum(ep_tokens_total)
        if diff != 0:
            # Adjust by descending fractional parts (stable correction)
            fracs = sorted(
                [
                    (abs(f - int(round(f))), ep_id, f)
                    for (f, ep_id) in ep_tokens_float
                ],
                reverse=True,
            )
            idx = 0
            step = 1 if diff > 0 else -1
            for _ in range(abs(diff)):
                ep_id = fracs[idx % len(fracs)][1]
                ep_tokens_total[ep_id] += step
                idx += 1
        if sum(ep_tokens_total) != total_expert_tokens:
            raise ValueError("Failed to correct ep_tokens_total to match total_expert_tokens")
        if any(t < 0 for t in ep_tokens_total):
            raise ValueError("Negative tokens assigned to an EP after correction")

        # Step 2: per-request split across EPs preserving each request's token count
        per_ep_num_tokens = [[0] * len(batch.requests) for _ in range(ep_size)]
        for i, original in enumerate(batch.num_tokens):
            if original < 0:
                raise ValueError("Negative original token count in batch.num_tokens")
            if total_expert_tokens == 0:
                continue
            # shares proportional to ep_tokens_total
            alloc_base = []
            alloc_frac = []
            for ep_id in range(ep_size):
                # Avoid division by zero (guarded by total_expert_tokens check)
                share = ep_tokens_total[ep_id] / total_expert_tokens
                val = original * share
                base = int(val)
                alloc_base.append(base)
                alloc_frac.append((val - base, ep_id))
            remain = original - sum(alloc_base)
            # Distribute remaining tokens by largest fractional parts
            alloc = alloc_base[:]
            if remain > 0:
                alloc_frac.sort(key=lambda x: x[0], reverse=True)
                for k in range(remain):
                    ep_j = alloc_frac[k % ep_size][1]
                    alloc[ep_j] += 1
            if sum(alloc) != original:
                raise ValueError("Per-request allocation mismatch after correction")
            for ep_id in range(ep_size):
                per_ep_num_tokens[ep_id][i] = alloc[ep_id]

        # Step 3: build EP sub-batches with per-expert allocation saved for predictor
        ep_batch_assignments = {}
        for ep_id in range(ep_size):
            scheduler_key = (replica_id, ep_id)
            ep_batch_portion = Batch(
                replica_id=replica_id,
                requests=batch.requests[:],
                num_tokens=per_ep_num_tokens[ep_id],
                is_moe=self._config.replica_config.model_config.is_moe,
            )

            # Preserve original decode-attn assignment metadata
            ep_batch_portion.decode_attn_original_replica_id = batch.decode_attn_original_replica_id
            ep_batch_portion.decode_attn_original_dp_id = batch.decode_attn_original_dp_id

            # EP AllGather synchronization: share global_id with original batch
            ep_batch_portion.set_global_id(batch.id)

            # Metadata for EP identification
            ep_batch_portion._original_batch_id = batch.id
            ep_batch_portion._ep_id = ep_id
            ep_batch_portion._total_ep_size = ep_size
            ep_batch_portion.time = batch.time

            # Compute per-expert allocation within this EP (top-k expanded)
            start = ep_id * experts_per_ep
            end = start + experts_per_ep
            local_experts = list(range(start, end))
            local_sum = sum(expert_ratios.get(gid, 0.0) for gid in local_experts)
            tokens_for_ep = ep_tokens_total[ep_id]
            expert_alloc = {gid: 0 for gid in local_experts}
            if tokens_for_ep > 0 and local_sum > 0.0:
                tmp = []
                for gid in local_experts:
                    share = expert_ratios.get(gid, 0.0) / local_sum
                    val = tokens_for_ep * share
                    base = int(val)
                    tmp.append((gid, val, base))
                base_sum = sum(b for (_g, _v, b) in tmp)
                remain = tokens_for_ep - base_sum
                # Distribute remainder by largest fractional parts
                tmp.sort(key=lambda x: (x[1] - x[2]), reverse=True)
                for idx in range(len(tmp)):
                    gid, _v, base = tmp[idx]
                    expert_alloc[gid] = base + (1 if idx < remain else 0)
            # Fail Fast checks
            if sum(expert_alloc.values()) != tokens_for_ep:
                raise ValueError("Expert allocation within EP does not sum to tokens_for_ep")
            if any(v < 0 for v in expert_alloc.values()):
                raise ValueError("Negative token in expert allocation within EP")

            # Persist per-EP per-expert allocation for predictor
            setattr(ep_batch_portion, "moe_expert_token_allocation", expert_alloc)
            setattr(ep_batch_portion, "experts_local_to_ep", local_experts)

            logger.info(
                f"[TOKEN_DISTRIBUTION] EP batch (replica={replica_id}, ep_id={ep_id}) "
                f"tokens={per_ep_num_tokens[ep_id]} expert_tokens_sum={sum(expert_alloc.values())}"
            )

            ep_batch_assignments[scheduler_key] = ep_batch_portion

        return ep_batch_assignments
