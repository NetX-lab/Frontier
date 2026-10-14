"""KV block accounting, allocation and preemption for the vLLM V1 scheduler.

This is the memory side of admission: how many tokens a request has already had
accounted, how many blocks its next step needs, whether those blocks are
available, and which request to preempt when they are not.
"""

from math import ceil
from typing import List, Optional

from frontier.attention.gdn.guards import validate_gdn_runtime_support
from frontier.entities.batch import Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheAllocationResult
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.vllm_v1_prefix_cache import (
    PrefixCacheAdmission,
)
from frontier.types import ClusterType


_REQUEST_PROGRESS_PRESERVING_PREEMPTION_CLUSTER_TYPES = frozenset(
    {
        ClusterType.DECODE,
        ClusterType.DECODE_ATTN,
    }
)


class KvBlockAllocation:
    """Token accounting, KV block allocation and preemption."""

    def _find_request_by_id(self, request_id: int) -> Optional[Request]:
        request_groups = [
            getattr(self, "_running_requests", []),
            getattr(self, "_request_queue", []),
            getattr(self, "_preempted_requests", []),
            getattr(self, "_waiting_requests", []),
        ]
        for requests in request_groups:
            for request in requests:
                if request.id == request_id:
                    return request
        return None

    def _free_request_resources(self, request: Request) -> None:
        self._get_monolithic_pp_mtp_near_full_prefill_request_ids().discard(
            request.id
        )
        self._get_monolithic_pp_mtp_single_output_wait_request_ids().discard(
            request.id
        )
        self._get_monolithic_pp_mtp_fractional_output_wait_counts().pop(
            request.id,
            None,
        )
        self._get_monolithic_pp_mtp_output_wait_remaining_iters().pop(
            request.id,
            None,
        )
        self._get_monolithic_pp_mtp_output_wait_request_ids().discard(request.id)
        self._get_monolithic_pp_waiting_admission_delay_iters().pop(
            request.id,
            None,
        )
        if self._is_prefix_caching_enabled():
            assert self._kv_cache_manager is not None
            self._kv_cache_manager.free(request)
            self._allocation_map.pop(request.id, None)
            self._sync_prefix_cache_allocation_state()
            gdn_slot_manager = self._gdn_state_slot_manager
            if gdn_slot_manager is not None:
                gdn_slot_manager.release(request.id)
            return
        self.free(request.id)
        gdn_slot_manager = self._gdn_state_slot_manager
        if gdn_slot_manager is not None:
            gdn_slot_manager.release(request.id)

    def _free_request_resources_by_id(self, request_id: int) -> None:
        request = self._find_request_by_id(request_id)
        if request is not None:
            self._free_request_resources(request)
            return
        self.free(request_id)
        # Completion/cancellation callbacks may arrive after the request has
        # left every scheduler queue.  Release an orphaned ownership token as
        # part of the same idempotent cleanup boundary so a slot cannot leak.
        gdn_slot_manager = self._gdn_state_slot_manager
        if gdn_slot_manager is not None:
            gdn_slot_manager.release(request_id)

    def _get_explicit_scheduler_num_computed_tokens(
        self, request: Request
    ) -> Optional[int]:
        scheduled_frontier = getattr(
            self, "_scheduled_num_computed_tokens_by_request", {}
        ).get(request.id)
        if scheduled_frontier is None:
            return None
        return int(scheduled_frontier)

    def _get_scheduler_num_computed_tokens(self, request: Request) -> int:
        """Return the scheduler-visible computed frontier for a request."""
        scheduled_frontier = self._get_explicit_scheduler_num_computed_tokens(request)
        if scheduled_frontier is not None:
            return scheduled_frontier

        processed_tokens = int(request.num_processed_tokens)
        if (
            getattr(self, "_cluster_type", None) == ClusterType.MONOLITHIC
            and request.is_prefill_complete
            and processed_tokens > int(request.num_prefill_tokens)
        ):
            # MONOLITHIC request metrics grant the first decode token at the
            # prefill-complete boundary, but vLLM's scheduler frontier does not
            # advance to that token until the first decode scheduling step.
            return max(int(request.num_prefill_tokens), processed_tokens - 1)
        return processed_tokens

    def _advance_scheduler_num_computed_tokens(
        self, request: Request, num_scheduled_tokens: int
    ) -> None:
        if num_scheduled_tokens < 0:
            raise ValueError(
                f"num_scheduled_tokens must be >= 0, got {num_scheduled_tokens}"
            )
        self._scheduled_num_computed_tokens_by_request[request.id] = (
            self._get_scheduler_num_computed_tokens(request) + int(num_scheduled_tokens)
        )

    def _get_kv_accounted_processed_tokens(self, request: Request) -> int:
        """Return processed tokens used for KV block accounting.

        In MONOLITHIC mode we intentionally count the first generated token at
        prefill boundary for request-level progression parity. However, vLLM's
        KV block growth does not advance at that boundary; it advances when the
        first decode scheduling step is executed. To align block semantics, KV
        accounting excludes that boundary token.
        """
        explicit_scheduler_frontier = self._get_explicit_scheduler_num_computed_tokens(
            request
        )
        if getattr(self, "_cluster_type", None) != ClusterType.MONOLITHIC:
            if explicit_scheduler_frontier is not None:
                return explicit_scheduler_frontier
            return int(request.num_processed_tokens)
        if not getattr(request, "is_prefill_complete", False):
            if explicit_scheduler_frontier is not None:
                return explicit_scheduler_frontier
            return int(request.num_processed_tokens)
        processed_tokens = int(request.num_processed_tokens)
        inflight_verify_tokens = 1
        if (
            getattr(request, "spec_decode_enabled", False)
            and getattr(request, "spec_method_uses_lookahead_slots", False)
        ):
            inflight_verify_tokens = max(
                1, int(getattr(request, "spec_current_verify_tokens", 1))
            )
        decode_boundary_adjusted_tokens = max(
            int(request.num_prefill_tokens), processed_tokens - inflight_verify_tokens
        )
        if explicit_scheduler_frontier is None:
            return decode_boundary_adjusted_tokens
        return max(explicit_scheduler_frontier, decode_boundary_adjusted_tokens)

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        computed_tokens = self._get_scheduler_num_computed_tokens(request)
        cluster_type = getattr(self, "_cluster_type", None)

        if request.is_prefill_complete:
            if getattr(request, "spec_decode_enabled", False):
                if getattr(request, "spec_method_is_target_embedded_mtp", False):
                    planned_drafts = int(
                        getattr(request, "spec_next_planned_draft_tokens", 0)
                    )
                    if (
                        cluster_type == ClusterType.MONOLITHIC
                        and int(getattr(request, "num_processed_decode_tokens", 0))
                        == 1
                        and computed_tokens <= int(request.num_prefill_tokens)
                    ):
                        return max(planned_drafts, 1)
                return 1 + int(getattr(request, "spec_next_planned_draft_tokens", 0))
            if cluster_type == ClusterType.MONOLITHIC:
                # In MONOLITHIC mode, request.num_processed_tokens includes the
                # post-prefill decode bonus. A new decode step is schedulable
                # only when request-side progress has advanced beyond the
                # scheduler-visible frontier, mirroring vLLM's
                # num_tokens_with_spec/num_computed_tokens gating under PP.
                return max(int(request.num_processed_tokens) - computed_tokens, 0)
            return 1

        remaining_prefill_tokens = int(request.num_prefill_tokens) - computed_tokens
        return max(remaining_prefill_tokens, 0)

    def _get_num_tokens_for_kv_reservation(
        self, request: Request, scheduled_tokens: int
    ) -> int:
        reserved_tokens = int(scheduled_tokens)
        if reserved_tokens <= 0:
            raise ValueError(
                f"scheduled_tokens must be > 0, got={scheduled_tokens}"
            )
        if not getattr(request, "is_prefill_complete", False):
            return reserved_tokens
        if not getattr(request, "spec_decode_enabled", False):
            return reserved_tokens
        if getattr(request, "spec_method_uses_lookahead_slots", False):
            return reserved_tokens
        # ngram/medusa path: no lookahead slot reservation in Phase 1.
        # Strategy A keeps allocation simple and lets future decode iterations
        # amortize accepted-token KV growth without immediate draft-slot reserves.
        return 1

    def _get_initial_allocation_num_blocks(
        self, request: Request, reserved_tokens: int
    ) -> int:
        """Return the block count used to check and commit a first allocation."""
        kv_accounted_tokens = self._get_kv_accounted_processed_tokens(request)
        total_tokens = min(
            kv_accounted_tokens + reserved_tokens, self._max_model_len
        )
        return ceil(total_tokens / self._config.block_size)

    def _can_allocate_request(
        self,
        request: Request,
        num_new_tokens: int = 1,
        new_computed_blocks=None,
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> bool:
        """
        Check if memory can be allocated for a request.

        For new requests: check if prefill blocks can be allocated.
        For running requests: check if at least one block is available.

        Args:
            request: The request to check allocation for
            num_new_tokens: Number of new tokens to allocate (used for decode)

        Returns:
            bool: True if allocation is possible
        """
        gdn_slot_manager = self._gdn_state_slot_manager
        if (
            gdn_slot_manager is not None
            and request.id not in self._allocation_map
            and not gdn_slot_manager.has_slot(request.id)
            and not gdn_slot_manager.has_available_slot
        ):
            return False
        if self._is_prefix_caching_enabled():
            assert self._kv_cache_manager is not None
            return self._kv_cache_manager.can_allocate_slots(
                request,
                num_new_tokens,
                new_computed_blocks=new_computed_blocks,
                scheduler_num_computed_tokens=(
                    self._get_scheduler_num_computed_tokens(request)
                    if scheduler_num_computed_tokens is None
                    else scheduler_num_computed_tokens
                ),
            )

        reserved_tokens = self._get_num_tokens_for_kv_reservation(
            request, num_new_tokens
        )
        if request.id not in self._allocation_map:
            # New request - estimate blocks from current token frontier
            # (already processed + newly scheduled in this iteration), then
            # clamp by max_model_len to keep allocation semantics consistent
            # with chunked prefill scheduling.
            num_required_blocks = self._get_initial_allocation_num_blocks(
                request, reserved_tokens
            )
            available_blocks = (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
            )
            return available_blocks >= self._watermark_blocks

        # Running request - check if we need additional blocks for decode
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        kv_accounted_tokens = self._get_kv_accounted_processed_tokens(request)
        num_tokens_required = max(
            0, kv_accounted_tokens + reserved_tokens - num_tokens_reserved
        )

        if num_tokens_required <= 0:
            return True

        # Need additional blocks
        num_additional_blocks = ceil(num_tokens_required / self._config.block_size)
        return self.can_allocate(num_additional_blocks)

    def _allocate_request(
        self,
        request: Request,
        num_new_tokens: int = 1,
        new_computed_blocks=None,
        prefix_cache_admission: Optional[PrefixCacheAdmission] = None,
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> Optional[KVCacheAllocationResult]:
        """
        Allocate memory blocks for a request.

        For new requests: allocate blocks for prefill tokens + first decode token.
        For running requests: allocate additional blocks if needed.

        Args:
            request: The request to allocate for
            num_new_tokens: Number of new tokens being processed
        """
        if type(num_new_tokens) is not int or num_new_tokens <= 0:
            raise ValueError(
                "num_new_tokens must be a positive integer, "
                f"got {num_new_tokens!r}"
            )
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        if self._is_prefix_caching_enabled():
            assert self._kv_cache_manager is not None
            computed_blocks = list(new_computed_blocks or [])
            if prefix_cache_admission is not None:
                expected_blocks = prefix_cache_admission.effective_hit_blocks
                if len(computed_blocks) != len(expected_blocks) or any(
                    actual is not expected
                    for actual, expected in zip(computed_blocks, expected_blocks)
                ):
                    raise ValueError(
                        "Committed Prefix cache admission blocks differ from "
                        "the allocation input."
                    )
                if int(num_new_tokens) != int(prefix_cache_admission.num_new_tokens):
                    raise ValueError(
                        "Committed Prefix cache admission token count differs "
                        "from the allocation input."
                    )
            allocation = self._kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                new_computed_blocks=computed_blocks,
                scheduler_num_computed_tokens=(
                    self._get_scheduler_num_computed_tokens(request)
                    if scheduler_num_computed_tokens is None
                    else scheduler_num_computed_tokens
                ),
            )
            if allocation is None:
                raise ValueError(
                    f"Failed to allocate prefix-cache-managed KV blocks for request {request.id}"
                )
            self._sync_prefix_cache_allocation_state(request)
            self._emit_prefix_cache_identity_events(
                request=request,
                num_new_tokens=num_new_tokens,
                allocation=allocation,
                admission=prefix_cache_admission,
            )
            return allocation

        reserved_tokens = self._get_num_tokens_for_kv_reservation(
            request, num_new_tokens
        )

        if request.id not in self._allocation_map:
            # Commit the exact block count used by the allocation preflight.
            # This includes a transferred token frontier for a first
            # DECODE_ATTN allocation after the PREFILL handoff.
            num_required_blocks = self._get_initial_allocation_num_blocks(
                request, reserved_tokens
            )
            self.allocate(request.id, num_required_blocks)
            gdn_slot_manager = self._gdn_state_slot_manager
            if gdn_slot_manager is not None:
                try:
                    gdn_slot_manager.allocate(request.id)
                except Exception:
                    # KV and state ownership must commit atomically from the
                    # scheduler's perspective.  Roll back the KV allocation
                    # before exposing the slot failure to admission.
                    self.free(request.id)
                    raise
            logger.debug(
                f"[VLLMv1Engine] Allocated {num_required_blocks} blocks for request {request.id} "
                f"(scheduled_tokens={num_new_tokens}, reserved_tokens={reserved_tokens})"
            )
            return None

        # Running request - check if additional blocks needed
        gdn_slot_manager = self._gdn_state_slot_manager
        if gdn_slot_manager is not None:
            if not gdn_slot_manager.has_slot(request.id):
                raise RuntimeError(
                    f"GDN state slot missing for admitted request {request.id}"
                )
            gdn_slot_manager.resume(request.id)
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        kv_accounted_tokens = self._get_kv_accounted_processed_tokens(request)
        num_tokens_required = max(
            0, kv_accounted_tokens + reserved_tokens - num_tokens_reserved
        )

        if num_tokens_required <= 0:
            return None

        # Allocate additional blocks
        num_additional_blocks = ceil(num_tokens_required / self._config.block_size)
        self.allocate(request.id, num_additional_blocks)
        return None

    def _select_preemption_victim(
        self, exclude: Optional[Request] = None
    ) -> Optional[Request]:
        """
        Select a victim request for preemption based on scheduling policy.

        FCFS policy: Preempt the most recently added request (queue tail).
        Priority policy: Preempt the request with lowest priority
                        (highest priority value, then latest arrival).

        Args:
            exclude: Optional request to exclude from victim selection (typically the requesting request)

        Returns:
            Optional[Request]: The victim request, or None if no victims available
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        if not self._running_requests:
            return None

        # Filter out excluded request
        candidates = (
            [r for r in self._running_requests if r != exclude]
            if exclude
            else self._running_requests
        )

        if not candidates:
            return None

        if self._scheduling_policy == "priority":
            # Priority policy: preempt request with highest priority value (lowest priority)
            # Tie-breaker: latest arrival time
            victim = max(candidates, key=lambda r: (r.priority, r.arrived_at))

            # Flow validation: log victim selection
            logger.info(
                f"[VICTIM_SELECTION] policy=PRIORITY, "
                f"victim={victim.id}, "
                f"priority={victim.priority}, "
                f"reason=highest_priority_value"
            )
            return victim
        else:
            # FCFS policy: preempt most recently added (queue tail)
            # If exclude is specified and is the tail, select the second-to-last
            if exclude and candidates and candidates[-1] != self._running_requests[-1]:
                # exclude was the tail, use candidates[-1] which is second-to-last
                victim = candidates[-1]
            else:
                victim = candidates[-1] if candidates else None

            if victim is None:
                return None

            # Flow validation: log victim selection
            logger.info(
                f"[VICTIM_SELECTION] policy=FCFS, "
                f"victim={victim.id}, "
                f"position=tail, "
                f"reason=last_in_running_queue"
            )
            return victim

    def _preempt_request(
        self, victim: Request, preempted_requests: List[Request]
    ) -> None:
        """
        Preempt a request - free its resources and move to waiting queue.

        Args:
            victim: The request to preempt
            preempted_requests: List to track preempted requests for this iteration
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        # GDN state cannot be dropped and restored by the simulator. Reject
        # before touching request counters, allocations, or queue membership.
        validate_gdn_runtime_support(
            self._replica_config.model_config,
            preemption_requires_state_drop=True,
        )

        # Capture state before modification
        num_computed_tokens_before = self._get_scheduler_num_computed_tokens(victim)
        freed_blocks = self._allocation_map.get(victim.id, 0)
        running_count_before = len(self._running_requests)
        queue_position_before = (
            self._running_requests.index(victim)
            if victim in self._running_requests
            else -1
        )

        # Record preemption statistics in the request entity
        # This must be done BEFORE resetting num_processed_tokens
        victim.record_preemption(self._cluster_type, num_computed_tokens_before)
        victim.advance_runtime_epoch()

        # Remove from running requests
        if victim in self._running_requests:
            self._running_requests.remove(victim)

        # Free allocated blocks
        if victim.id in self._allocation_map:
            self._free_request_resources(victim)

        # Mark as preempted and reset the scheduler-visible computed frontier.
        # Disaggregated decode requests arrive after PREFILL has completed and
        # the prompt KV frontier has transferred to the decode-side cluster.
        # Their Request-level token lifecycle must survive memory preemption;
        # only scheduler-local computed state and KV allocation are restarted.
        victim._preempted = True
        if (
            self._cluster_type
            not in _REQUEST_PROGRESS_PRESERVING_PREEMPTION_CLUSTER_TYPES
        ):
            victim._num_processed_tokens = 0  # Reset computed tokens as in vLLM v1
        self._scheduled_num_computed_tokens_by_request.pop(victim.id, None)

        # Record re-entry to waiting queue for waiting time tracking after the
        # lifecycle decision above and before adding the request to the queue.
        victim.on_enter_waiting_queue(self._current_schedule_time, self._cluster_type)

        # Add to front of appropriate waiting queue (prepend)
        # DECODE and DECODE_ATTN clusters use _waiting_requests, others use _request_queue
        if self._cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]:
            self._waiting_requests.insert(0, victim)
        else:
            self._request_queue.insert(0, victim)

        # Track for this iteration
        preempted_requests.append(victim)

        logger.info(
            f"[VLLMv1Engine] Preempted request {victim.id} "
            f"(policy={self._scheduling_policy}), "
            f"running_reqs={len(self._running_requests)}"
        )

        # Flow validation: log preemption event
        logger.info(
            f"[PREEMPTION] req={victim.id} preempted, "
            f"policy={self._scheduling_policy}, "
            f"freed_blocks={freed_blocks}"
        )
        available_blocks_preempt = int(self._config.num_blocks - self._num_allocated_blocks)
        self._emit_schedule_decision_event(
            event="decision",
            decision_result="PREEMPTED",
            request_id=victim.id,
            token_budget=self._current_iteration_token_budget,
            available_blocks=available_blocks_preempt,
            num_tokens=0,
        )

        # Flow validation: log detailed preemption info
        victim_selection_reason = (
            "lowest_priority"
            if self._scheduling_policy == "priority"
            else "tail_of_running_queue"
        )
        logger.info(
            f"[PREEMPTION_DETAIL] req={victim.id}, "
            f"num_computed_tokens_before={num_computed_tokens_before}, "
            f"freed_blocks={freed_blocks}, "
            f"policy={self._scheduling_policy}, "
            f"victim_selection_reason={victim_selection_reason}, "
            f"queue_position_before={queue_position_before}, "
            f"running_count_before={running_count_before}, "
            f"running_count_after={len(self._running_requests)}"
        )

    def _try_allocate_with_preemption(
        self,
        request: Request,
        num_new_tokens: int,
        preempted_requests: List[Request],
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> bool:
        """
        Try to allocate memory for a request, preempting other requests if necessary.

        This implements the core preemption loop from vLLM v1 scheduler.

        Args:
            request: The request to allocate for
            num_new_tokens: Number of new tokens to process
            preempted_requests: List to track preempted requests

        Returns:
            bool: True if allocation succeeded (possibly after preemption)
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        while True:
            if scheduler_num_computed_tokens is None:
                can_allocate = self._can_allocate_request(request, num_new_tokens)
            else:
                can_allocate = self._can_allocate_request(
                    request,
                    num_new_tokens,
                    scheduler_num_computed_tokens=scheduler_num_computed_tokens,
                )
            if can_allocate:
                if scheduler_num_computed_tokens is None:
                    self._allocate_request(request, num_new_tokens)
                else:
                    self._allocate_request(
                        request,
                        num_new_tokens,
                        scheduler_num_computed_tokens=scheduler_num_computed_tokens,
                    )
                return True

            if not self._enable_preemption:
                return False

            # Flow validation: log memory pressure
            available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
            logger.info(
                f"[MEMORY_PRESSURE] trigger=allocation_failed, "
                f"requesting_req={request.id}, "
                f"requested_tokens={num_new_tokens}, "
                f"available_blocks={available_blocks}, "
                f"running_queue_size={len(self._running_requests)}"
            )

            # Select victim for preemption (exclude current request)
            victim = self._select_preemption_victim(exclude=request)

            if victim is None:
                # No victims available (all other requests have higher priority or no other requests)
                # Preempt self and move to waiting queue
                self._preempt_request(request, preempted_requests)
                return False

            # Preempt victim and try again
            self._preempt_request(victim, preempted_requests)

    def _rollback_current_iteration_preempted_requests(
        self,
        *,
        scheduled_requests: List[Request],
        scheduled_num_tokens: List[int],
        newly_preempted_requests: List[Request],
        token_budget: int,
    ) -> int:
        if not newly_preempted_requests:
            return token_budget

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        preempted_request_ids = {
            int(request.id) for request in newly_preempted_requests
        }
        if not preempted_request_ids:
            return token_budget

        kept_requests: List[Request] = []
        kept_num_tokens: List[int] = []
        refunded_tokens = 0

        for scheduled_request, scheduled_tokens in zip(
            scheduled_requests, scheduled_num_tokens
        ):
            if int(scheduled_request.id) in preempted_request_ids:
                refunded_tokens += int(scheduled_tokens)
                logger.info(
                    "[RUNNING-SCHEDULE-ROLLBACK] req=%s removed from current iteration "
                    "after same-iteration preemption, refunded_tokens=%s",
                    scheduled_request.id,
                    scheduled_tokens,
                )
                continue
            kept_requests.append(scheduled_request)
            kept_num_tokens.append(int(scheduled_tokens))

        if refunded_tokens == 0:
            return token_budget

        scheduled_requests[:] = kept_requests
        scheduled_num_tokens[:] = kept_num_tokens
        token_budget += refunded_tokens
        self._current_iteration_token_budget = token_budget
        return token_budget
