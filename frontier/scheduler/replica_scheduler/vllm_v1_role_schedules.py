"""Single-role scheduling entry points for disaggregated clusters.

Co-location schedules prefill and decode together through the two-phase path
that stays on the scheduler itself.  A disaggregated cluster instead drives one
role per replica, and each of those roles has its own entry point here.
"""

from collections import deque
from typing import List, Optional, Tuple

from frontier.config import global_vars
from frontier.entities.batch import Batch, Request
from frontier.logger import get_cluster_logger
from frontier.types import ClusterType


class DisaggregatedRoleScheduling:
    """Prefill-only, decode-only and decode-attention scheduling entry points."""

    def _schedule_prefill_only(self) -> Optional[Batch]:
        """
        Scheduling for PREFILL cluster.

        In PD-disaggregation, the prefill cluster only handles new requests
        that need prefill computation. With chunked prefill enabled, running
        partial-prefill requests are also scheduled in Phase 1.

        Returns:
            Optional[Batch]: The scheduled batch
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        token_budget = self._max_num_scheduled_tokens
        available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
        waiting_count = len(self._request_queue) + len(self._preempted_requests)

        # Flow validation: log iteration start
        logger.info(
            f"[ITERATION_START] token_budget={token_budget}, "
            f"running_count={len(self._running_requests)}, "
            f"waiting_count={waiting_count}, "
            f"available_blocks={available_blocks}, "
            f"max_running_reqs={self._max_num_running_reqs}"
        )
        self._emit_schedule_decision_event(
            event="iteration_start",
            decision_result=None,
            request_id=None,
            token_budget=token_budget,
            available_blocks=available_blocks,
            num_tokens=0,
        )

        # Flow validation: log memory state
        total_blocks = int(self._config.num_blocks)
        allocated_blocks = int(self._num_allocated_blocks)
        usage_ratio = allocated_blocks / total_blocks if total_blocks > 0 else 0.0
        watermark = self._watermark_blocks
        logger.info(
            f"[MEMORY_STATE] total_blocks={total_blocks}, "
            f"allocated_blocks={allocated_blocks}, "
            f"free_blocks={available_blocks}, "
            f"usage_ratio={usage_ratio:.4f}, "
            f"watermark_blocks={watermark}"
        )
        reclaimed_requests = self._reclaim_borrowed_final_running_slots(
            waiting_requests=self._preempted_requests + self._request_queue,
            final_predicate=self._is_final_prefill_fast_lane_request,
            reserved_slots=self._final_prefill_reserved_slots,
            lane_name="prefill",
        )
        if reclaimed_requests:
            waiting_count = len(self._request_queue) + len(self._preempted_requests)

        all_scheduled_requests: List[Request] = []
        all_num_tokens: List[int] = []
        preempted_requests: List[Request] = []
        waiting_scheduled: List[Request] = []
        waiting_tokens: List[int] = []

        # Phase 1: schedule running requests (partial prefill continuation)
        logger.info(
            f"[PHASE1_START] running_count={len(self._running_requests)}, "
            f"token_budget={token_budget}, "
            f"waiting_count={waiting_count}"
        )
        token_budget, running_scheduled, running_tokens = self._schedule_running_requests(
            token_budget, preempted_requests
        )
        all_scheduled_requests.extend(running_scheduled)
        all_num_tokens.extend(running_tokens)

        available_blocks_p1 = int(self._config.num_blocks - self._num_allocated_blocks)
        logger.info(
            f"[PHASE1_END] scheduled_count={len(running_scheduled)}, "
            f"preempted_count={len(preempted_requests)}, "
            f"token_budget_remaining={token_budget}, "
            f"available_blocks={available_blocks_p1}"
        )

        # Phase 2: schedule waiting requests only when Phase 1 has no preemption
        if not preempted_requests:
            logger.info(
                f"[PHASE2_START] waiting_count={waiting_count}, "
                f"token_budget={token_budget}, "
                f"running_count={len(self._running_requests)}"
            )
            token_budget, waiting_scheduled, waiting_tokens = (
                self._schedule_waiting_requests(token_budget)
            )
            all_scheduled_requests.extend(waiting_scheduled)
            all_num_tokens.extend(waiting_tokens)

            available_blocks_p2 = int(
                self._config.num_blocks - self._num_allocated_blocks
            )
            logger.info(
                f"[PHASE2_END] admitted_count={len(waiting_scheduled)}, "
                f"token_budget_remaining={token_budget}, "
                f"available_blocks={available_blocks_p2}, "
                f"running_count={len(self._running_requests)}"
            )

        if not all_scheduled_requests:
            self._emit_schedule_decision_event(
                event="iteration_end",
                decision_result=None,
                request_id=None,
                token_budget=token_budget,
                num_tokens=0,
                available_blocks=int(self._config.num_blocks - self._num_allocated_blocks),
                batch_request_ids=[],
                request_num_tokens=[],
                batch_size=0,
                batch_num_tokens=0,
            )
            return None

        ordered_scheduled_requests = waiting_scheduled + running_scheduled
        ordered_num_tokens = waiting_tokens + running_tokens

        # Flow validation: log batch formation
        total_tokens = sum(all_num_tokens)
        new_admitted = len(waiting_scheduled)
        resumed = len(
            [r for r in all_scheduled_requests if getattr(r, "_preempted", False)]
        )
        running_continued = len(running_scheduled)
        batch_size = len(all_scheduled_requests)

        logger.info(
            f"[BATCH_FORMATION] total_tokens={total_tokens}, "
            f"new_admitted={new_admitted}, "
            f"resumed={resumed}, "
            f"running_continued={running_continued}, "
            f"batch_size={batch_size}"
        )
        self._emit_schedule_decision_event(
            event="iteration_end",
            decision_result=None,
            request_id=None,
            token_budget=token_budget,
            num_tokens=total_tokens,
            available_blocks=int(self._config.num_blocks - self._num_allocated_blocks),
            batch_request_ids=[request.id for request in ordered_scheduled_requests],
            request_num_tokens=ordered_num_tokens,
            batch_size=batch_size,
            batch_num_tokens=total_tokens,
        )

        return self._create_batch(ordered_scheduled_requests, ordered_num_tokens)

    def _schedule_decode_only(self) -> Optional[Batch]:
        """
        Scheduling for DECODE cluster - two-phase scheduling matching vLLM v1.

        Phase 1: Schedule RUNNING requests (ongoing decode iterations)
        Phase 2: Admit WAITING requests (new arrivals from prefill cluster)

        This matches vLLM v1's scheduling algorithm where requests must be
        admitted from waiting queue to running queue before generating tokens.

        Returns:
            Optional[Batch]: The scheduled batch
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        all_scheduled_requests: List[Request] = []
        all_num_tokens: List[int] = []
        preempted_requests: List[Request] = []
        waiting_scheduled: List[Request] = []
        waiting_tokens: List[int] = []
        token_budget = self._max_num_scheduled_tokens
        available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
        waiting_final_decode_count = self._count_final_fast_lane_requests(
            self._waiting_requests,
            final_predicate=self._is_final_decode_fast_lane_request,
        )
        self._decode_iteration_reserved_slots_remaining = (
            self._final_decode_reserved_slots if waiting_final_decode_count > 0 else 0
        )

        # Flow validation: log iteration start
        logger.info(
            f"[ITERATION_START] token_budget={token_budget}, "
            f"running_count={len(self._running_requests)}, "
            f"waiting_count={len(self._waiting_requests)}, "
            f"available_blocks={available_blocks}, "
            f"max_running_reqs={self._max_num_running_reqs}"
        )
        self._emit_schedule_decision_event(
            event="iteration_start",
            decision_result=None,
            request_id=None,
            token_budget=token_budget,
            available_blocks=available_blocks,
            num_tokens=0,
        )

        # Flow validation: log memory state
        total_blocks = int(self._config.num_blocks)
        allocated_blocks = int(self._num_allocated_blocks)
        usage_ratio = allocated_blocks / total_blocks if total_blocks > 0 else 0.0
        watermark = self._watermark_blocks
        logger.info(
            f"[MEMORY_STATE] total_blocks={total_blocks}, "
            f"allocated_blocks={allocated_blocks}, "
            f"free_blocks={available_blocks}, "
            f"usage_ratio={usage_ratio:.4f}, "
            f"watermark_blocks={watermark}"
        )
        self._reclaim_borrowed_final_running_slots(
            waiting_requests=self._waiting_requests,
            final_predicate=self._is_final_decode_fast_lane_request,
            reserved_slots=self._final_decode_reserved_slots,
            lane_name="decode",
        )

        # Flow validation: log Phase 1 start
        logger.info(
            f"[PHASE1_START] running_count={len(self._running_requests)}, "
            f"token_budget={token_budget}"
        )

        # === Phase 1: Schedule RUNNING requests ===
        token_budget, running_scheduled, running_tokens = (
            self._schedule_running_requests(token_budget, preempted_requests)
        )
        all_scheduled_requests.extend(running_scheduled)
        all_num_tokens.extend(running_tokens)

        # Flow validation: log Phase 1 end
        available_blocks_p1 = int(self._config.num_blocks - self._num_allocated_blocks)
        logger.info(
            f"[PHASE1_END] scheduled_count={len(running_scheduled)}, "
            f"preempted_count={len(preempted_requests)}, "
            f"token_budget_remaining={token_budget}, "
            f"available_blocks={available_blocks_p1}"
        )

        # === Phase 2: Admit WAITING requests (only if no preemption) ===
        if not preempted_requests:
            # Flow validation: log Phase 2 start
            logger.info(
                f"[PHASE2_START] waiting_count={len(self._waiting_requests)}, "
                f"token_budget={token_budget}, "
                f"running_count={len(self._running_requests)}"
            )

            token_budget, waiting_scheduled, waiting_tokens = (
                self._schedule_decode_waiting_requests(token_budget)
            )
            all_scheduled_requests.extend(waiting_scheduled)
            all_num_tokens.extend(waiting_tokens)

            # Flow validation: log Phase 2 end
            available_blocks_p2 = int(
                self._config.num_blocks - self._num_allocated_blocks
            )
            logger.info(
                f"[PHASE2_END] admitted_count={len(waiting_scheduled)}, "
                f"token_budget_remaining={token_budget}, "
                f"available_blocks={available_blocks_p2}, "
                f"running_count={len(self._running_requests)}"
            )

        if not all_scheduled_requests:
            self._emit_schedule_decision_event(
                event="iteration_end",
                decision_result=None,
                request_id=None,
                token_budget=token_budget,
                num_tokens=0,
                available_blocks=int(self._config.num_blocks - self._num_allocated_blocks),
                batch_request_ids=[],
                request_num_tokens=[],
                batch_size=0,
                batch_num_tokens=0,
            )
            return None

        # Match vLLM v1 output order: new admissions first, then running.
        ordered_scheduled_requests = waiting_scheduled + running_scheduled
        ordered_num_tokens = waiting_tokens + running_tokens

        # Flow validation: log batch formation
        total_tokens = sum(all_num_tokens)
        new_admitted = len(
            [r for r in all_scheduled_requests if r not in running_scheduled]
        )
        resumed = 0  # DECODE doesn't handle preempted requests (they come from prefill)
        running_continued = len(running_scheduled)
        batch_size = len(all_scheduled_requests)

        logger.info(
            f"[BATCH_FORMATION] total_tokens={total_tokens}, "
            f"new_admitted={new_admitted}, "
            f"resumed={resumed}, "
            f"running_continued={running_continued}, "
            f"batch_size={batch_size}"
        )
        self._emit_schedule_decision_event(
            event="iteration_end",
            decision_result=None,
            request_id=None,
            token_budget=token_budget,
            num_tokens=total_tokens,
            available_blocks=int(self._config.num_blocks - self._num_allocated_blocks),
            batch_request_ids=[request.id for request in ordered_scheduled_requests],
            request_num_tokens=ordered_num_tokens,
            batch_size=batch_size,
            batch_num_tokens=total_tokens,
        )

        return self._create_batch(ordered_scheduled_requests, ordered_num_tokens)

    def _schedule_decode_waiting_requests(
        self, token_budget: int
    ) -> Tuple[int, List[Request], List[int]]:
        """
        Phase 2 for DECODE cluster: Admit requests from waiting queue.

        This method handles requests that have arrived from the prefill cluster
        and are waiting to be admitted to the running queue for decode iterations.
        Matches vLLM v1's Phase 2 scheduling behavior.

        Args:
            token_budget: Remaining token budget for this iteration

        Returns:
            Tuple of (remaining_budget, scheduled_requests, num_tokens_list)
        """
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        scheduled: List[Request] = []
        num_tokens_list: List[int] = []

        fast_lane_decode_enabled = self._cluster_type == ClusterType.DECODE and (
            self._final_decode_reserved_slots > 0
        )
        waiting_queue = self._build_decode_waiting_queue()
        skipped_waiting_requests: deque[Request] = deque()

        self._current_iteration_token_budget = token_budget
        while waiting_queue and token_budget > 0:
            self._current_iteration_token_budget = token_budget
            final_waiting_count = (
                self._count_final_fast_lane_requests(
                    waiting_queue,
                    final_predicate=self._is_final_decode_fast_lane_request,
                )
                if fast_lane_decode_enabled
                else 0
            )
            has_final_waiting = final_waiting_count > 0
            # Check max concurrent requests limit
            if len(self._running_requests) >= self._max_num_running_reqs:
                logger.debug(
                    f"[VLLMv1Engine][DECODE] Phase 2: max running requests "
                    f"reached ({self._max_num_running_reqs}), stopping admission"
                )
                break

            request = waiting_queue[0]
            is_final_decode_request = fast_lane_decode_enabled and (
                self._is_final_decode_fast_lane_request(request)
            )
            is_hidden_decode_request = (
                fast_lane_decode_enabled and not is_final_decode_request
            )

            if (
                is_hidden_decode_request
                and has_final_waiting
                and self._decode_iteration_reserved_slots_remaining > 0
                and len(self._running_requests)
                >= (
                    self._max_num_running_reqs
                    - self._decode_iteration_reserved_slots_remaining
                )
            ):
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue

            num_new_tokens = self._get_request_next_num_tokens(request)

            # Apply max_model_len limit
            scheduler_num_computed_tokens = self._get_scheduler_num_computed_tokens(
                request
            )
            max_allowed = self._max_model_len - scheduler_num_computed_tokens
            num_new_tokens = min(num_new_tokens, max_allowed)

            # Apply token budget limit
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens <= 0:
                # Request has reached max length, remove from queue
                waiting_queue.popleft()
                logger.debug(
                    f"[VLLMv1Engine][DECODE] Phase 2: req={request.id} "
                    f"reached max length, removing from waiting queue"
                )
                continue

            # Try to allocate (no preemption for waiting requests in Phase 2)
            if not self._can_allocate_request(
                request,
                num_new_tokens,
                scheduler_num_computed_tokens=scheduler_num_computed_tokens,
            ):
                # Cannot allocate - stop admitting new requests
                logger.debug(
                    f"[VLLMv1Engine][DECODE] Phase 2: cannot allocate "
                    f"req={request.id}, stopping admission"
                )
                break

            # Check if this request was previously preempted
            was_preempted = getattr(request, "_preempted", False)

            # Remove from waiting queue and allocate
            waiting_queue.popleft()

            # Record leaving waiting queue for waiting time tracking
            request.on_leave_waiting_queue(
                self._current_schedule_time, self._cluster_type
            )

            self._allocate_request(
                request,
                num_new_tokens,
                scheduler_num_computed_tokens=scheduler_num_computed_tokens,
            )
            self._advance_scheduler_num_computed_tokens(request, num_new_tokens)

            # Add to running requests
            self._running_requests.append(request)

            # Clear preempted flag if set
            if was_preempted:
                request._preempted = False

            scheduled.append(request)
            num_tokens_list.append(num_new_tokens)
            token_budget -= num_new_tokens
            self._current_iteration_token_budget = token_budget
            if is_final_decode_request:
                self._decode_iteration_reserved_slots_remaining = max(
                    self._decode_iteration_reserved_slots_remaining - 1,
                    0,
                )

            # Flow validation: log ADMISSION event (matching vLLM v1)
            logger.info(
                f"[ADMISSION] req={request.id} admitted, "
                f"num_tokens={num_new_tokens}, "
                f"running_count={len(self._running_requests)}, "
                f"token_budget_remaining={token_budget}"
            )
            available_blocks_admission = int(
                self._config.num_blocks - self._num_allocated_blocks
            )
            self._emit_schedule_decision_event(
                event="decision",
                decision_result="ADMISSION",
                request_id=request.id,
                token_budget=token_budget,
                available_blocks=available_blocks_admission,
                num_tokens=num_new_tokens,
            )

            # Flow validation: log preemption recovery if applicable
            if was_preempted:
                # For DECODE cluster, preempted requests need full recomputation
                # from their original prefill tokens
                recompute_tokens = request.num_prefill_tokens
                logger.info(
                    f"[PREEMPTION_RECOVERY] req={request.id}, "
                    f"was_preempted=True, "
                    f"recompute_tokens={recompute_tokens}"
                )

        if skipped_waiting_requests:
            waiting_queue.extend(skipped_waiting_requests)
        self._waiting_requests = list(waiting_queue)

        return token_budget, scheduled, num_tokens_list

    def _should_use_dense_decode_attn_metadata_wave(self) -> bool:
        """Return whether dense PP=1 PDAF needs one DES macro-wave batch."""
        if self._cluster_type != ClusterType.DECODE_ATTN:
            return False
        if self._replica_is_moe:
            return False
        return (
            self._num_stages == 1
            and self._af_pipeline_num_micro_batch > 1
        )

    def _schedule_decode_attn_only(
        self, is_micro_batch: bool = True
    ) -> Optional[Batch]:
        """
        Scheduling for DECODE_ATTN cluster in PD-AF disaggregation mode.

        This method is called ONLY for Priority 2 scheduling (new micro-batch formation).
        Priority 1 (AF immediate inflight batches) is handled by on_schedule() directly.

        Two-level scheduling strategy based on decode step:
        - Incomplete decode step (is_mb_last_layer=False): batch-level, via _af_immediate_batch_queue
        - Complete decode step (is_mb_last_layer=True): request-level, via this method

        Phase 1: Schedule running requests (ongoing decode from _running_requests)
            - For each request in _running_requests:
              - Calculate new tokens to process (usually 1 for decode)
              - Allocate memory for new tokens
              - If allocation fails: trigger preemption following vLLM v1 behavior
              - Add to scheduled batch

        Phase 2: Admit new requests from _waiting_requests (if Phase 1 had no preemption)
            - Check memory budget and token budget
            - Form micro-batch with layer-consistent grouping (fix: do we need it? all requests are layer-0)
            - All new requests start at layer 0, so naturally layer-consistent

        Layer-consistent grouping is implicitly guaranteed:
        - Running requests have _completed_layer_count = 0 (reset after decode step completion)
        - New requests also start at layer 0
        - Therefore, all requests in a micro-batch are layer-consistent

        Note on initial state:
        - On first scheduling, _running_requests is empty, so Phase 1 produces no output
        - Phase 2 will admit new requests from _waiting_requests to _running_requests
        - Subsequent decode steps will have Phase 1 populated from previous on_batch_end()

        Args:
            is_micro_batch: Should always be True for DECODE_ATTN

        Returns:
            Optional[Batch]: The scheduled micro-batch, or None if no requests available
        """
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        if is_micro_batch and self._af_pending_micro_batches:
            return self._af_pending_micro_batches.popleft()

        # Enable preemption for DECODE_ATTN to handle memory pressure
        # Preemption logic follows vLLM v1 behavior for running requests
        preemption_enabled = True
        preempted_requests: List[Request] = []

        # Phase 1: Schedule running requests
        scheduled_requests = []
        scheduled_tokens = []

        # Get request IDs to exclude (already scheduled in inflight batches)
        continuation_request_ids = getattr(self, "_continuation_request_ids", set())
        # _running_requests in inclued reqs: inflight(layer!=0) req, completed req (really?) 

        for request in self._running_requests:
            # ISSUE-008 FIX: Check batch size limit at start of Phase 1 loop.
            # This prevents scheduling more requests than _micro_batch_size allows,
            # ensuring proper batch size enforcement in DECODE_ATTN cluster.
            if len(scheduled_requests) >= self._micro_batch_size:
                logger.debug(
                    f"[VLLMv1Engine][DECODE_ATTN] Phase 1: reached micro_batch_size limit "
                    f"({self._micro_batch_size}), stopping"
                )
                break

            if request.completed:
                # why would a running request be completed but still in _running_requests?
                raise ValueError(f"Request {request.id} is already completed")
                continue

            # CRITICAL FIX: Only schedule requests ready for new decode step (layer_count = 0)
            # Requests with layer_count > 0 are still in-flight (mid-layer processing)
            # and should NOT be re-scheduled until their current decode step completes.
            # This ensures layer-consistent grouping in micro-batches.
            if request.completed_layer_count != 0:
                logger.debug(
                    f"[VLLMv1Engine][DECODE_ATTN] Phase 1: skipping in-flight req={request.id} "
                    f"with layer_count={request.completed_layer_count} (not ready for new decode step)"
                )
                continue

            # Requests in active A->F->A roundtrip must not be re-scheduled until
            # F->A transfer end clears the in-flight marker.
            if request.af_roundtrip_inflight:
                logger.debug(
                    f"[VLLMv1Engine][DECODE_ATTN] Phase 1: skipping req={request.id} "
                    f"(AF roundtrip still in-flight)"
                )
                continue

            # CRITICAL FIX: Skip requests already scheduled in continuation batches (Priority 1)
            # This prevents the same request from being scheduled into multiple batches
            if request.id in continuation_request_ids:
                logger.debug(
                    f"[VLLMv1Engine][DECODE_ATTN] Phase 1: skipping req={request.id} "
                    f"(already in continuation batch from Priority 1)"
                )
                continue

            # Calculate tokens for decode: usually 1
            num_new_tokens = 1

            # Try to allocate memory
            if self._can_allocate_request(request, num_new_tokens):
                self._allocate_request(request, num_new_tokens)
                scheduled_requests.append(request)
                scheduled_tokens.append(num_new_tokens)
                logger.debug(
                    f"[VLLMv1Engine][DECODE_ATTN] Phase 1: scheduled running req={request.id}, "
                    f"num_tokens={num_new_tokens}"
                )
            else:
                # Memory pressure - try allocation with preemption
                if not preemption_enabled:
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Phase 1: cannot allocate req={request.id}, "
                        f"preemption disabled, skipping"
                    )
                    continue

                # Try to allocate with preemption (follows vLLM v1 behavior)
                preempted_count_before = len(preempted_requests)
                success = self._try_allocate_with_preemption(
                    request, num_new_tokens, preempted_requests
                )
                self._current_iteration_token_budget = (
                    self._rollback_current_iteration_preempted_requests(
                        scheduled_requests=scheduled_requests,
                        scheduled_num_tokens=scheduled_tokens,
                        newly_preempted_requests=preempted_requests[
                            preempted_count_before:
                        ],
                        token_budget=self._current_iteration_token_budget,
                    )
                )
                if success:
                    scheduled_requests.append(request)
                    scheduled_tokens.append(num_new_tokens)
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Phase 1: scheduled req={request.id} "
                        f"after preemption, num_tokens={num_new_tokens}"
                    )
                else:
                    # Request itself was preempted or no victim available
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Phase 1: req={request.id} "
                        f"preempted or allocation failed"
                    )

        # Check micro-batch size limit
        remaining_slots = self._micro_batch_size - len(scheduled_requests)

        logger.debug(
            f"[VLLMv1Engine][DECODE_ATTN] After Phase 1: scheduled={len(scheduled_requests)}, "
            f"remaining_slots={remaining_slots}, micro_batch_size={self._micro_batch_size}"
        )

        # Phase 2: Admit new requests (only if no preemption occurred)
        if len(preempted_requests) == 0 and remaining_slots > 0:
            for request in list(self._waiting_requests):
                if remaining_slots <= 0:
                    break

                # New requests start at layer 0 - naturally layer-consistent
                assert request.completed_layer_count == 0, (
                    f"New request {request.id} should have completed_layer_count=0, got {request.completed_layer_count}"
                )

                # Allocate decode token
                num_tokens = 1
                if self._can_allocate_request(request, num_tokens):
                    self._waiting_requests.remove(request)
                    request.on_leave_waiting_queue(
                        self._current_schedule_time, self._cluster_type
                    )
                    self._allocate_request(request, num_tokens)
                    self._running_requests.append(request)
                    scheduled_requests.append(request)
                    scheduled_tokens.append(num_tokens)
                    remaining_slots -= 1
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Phase 2: admitted new req={request.id}, "
                        f"num_tokens={num_tokens}, running_count={len(self._running_requests)}"
                    )
                else:
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Phase 2: cannot allocate req={request.id}, "
                        f"stopping admission"
                    )
                    break

        # (scheduled_requests, scheduled_tokens) is the scheduler's output
        # we should use scheduler_output to creat microbatch for pd-af

        # Create batch if we have scheduled requests
        if scheduled_requests:
            logger.info(
                f"[VLLMv1Engine][DECODE_ATTN] Created micro-batch with {len(scheduled_requests)} requests"
            )

            num_reqs = len(scheduled_requests)
            num_stages = self._af_pipeline_num_micro_batch
            if num_stages is None or num_stages <= 0:
                raise ValueError(
                    "af_pipeline_num_micro_batch must be positive for DECODE_ATTN"
                )

            replay_decode_token_index = int(
                scheduled_requests[0].current_decode_token_index
            )
            decode_attn_cohort_id = self._allocate_decode_attn_cohort_id()
            decode_attn_cohort_request_ids = tuple(
                request.id for request in scheduled_requests
            )
            cohort_state = self._get_decode_attn_active_cohort_states().setdefault(
                decode_attn_cohort_id,
                {
                    "all_request_ids": set(),
                    "pending_request_ids": set(),
                    "af_phase": "local_attn",
                    "active_stage_indices": set(),
                    "stage_phases": {},
                    "stage_current_layer_ids": {},
                },
            )
            cohort_state["all_request_ids"].update(
                decode_attn_cohort_request_ids
            )
            cohort_state["pending_request_ids"].update(
                decode_attn_cohort_request_ids
            )
            cohort_state["current_layer_id"] = int(
                scheduled_requests[0].completed_layer_count
            )

            # StepFun-vLLM partitioning: split requests by stage
            if num_reqs >= num_stages:
                num_reqs_per_stage = num_reqs // num_stages
                stage_reqs_start_loc = [
                    num_reqs_per_stage * i for i in range(num_stages + 1)
                ]
                stage_reqs_start_loc[-1] = num_reqs
            else:
                stage_reqs_start_loc = list(range(num_reqs + 1))

            afd_stage_metadata = None
            if self._cluster_type == ClusterType.DECODE_ATTN and num_stages > 0:
                from frontier.config import global_vars
                from frontier.entities.batch import AFDStageMetadata

                use_cuda_graph = global_vars.get_use_cuda_graph()
                cudagraph_capture_sizes = global_vars.get_cudagraph_capture_sizes()
                if use_cuda_graph and cudagraph_capture_sizes is None:
                    max_num_seqs = (
                        self._micro_batch_size
                        if hasattr(self, "_micro_batch_size")
                        else 64
                    )
                    cudagraph_capture_sizes = [1, 2, 4] + [
                        8 * i for i in range(1, max_num_seqs // 8 + 1)
                    ]

                afd_stage_metadata = AFDStageMetadata.from_batch_params(
                    num_reqs=num_reqs,
                    num_tokens_per_req=scheduled_tokens,
                    num_stages=num_stages,
                    dp_stage_max_tokens=None,
                    use_cuda_graph=use_cuda_graph,
                    cudagraph_capture_sizes=cudagraph_capture_sizes,
                    ffn_use_cuda_graph=use_cuda_graph,
                    ffn_cudagraph_capture_sizes=cudagraph_capture_sizes,
                )

            first_micro_batch = None
            if self._should_use_dense_decode_attn_metadata_wave():
                macro_batch = self._create_batch(
                    scheduled_requests,
                    scheduled_tokens,
                )
                macro_batch.afd_stage_idx = 0
                macro_batch.afd_stage_represents_all_stages = True
                macro_batch.replay_decode_token_index = replay_decode_token_index
                macro_batch.decode_attn_cohort_id = decode_attn_cohort_id
                macro_batch.decode_attn_cohort_request_ids = (
                    decode_attn_cohort_request_ids
                )
                if afd_stage_metadata is not None:
                    macro_batch.afd_stage_metadata = afd_stage_metadata
                cohort_state["active_stage_indices"].add(0)
                cohort_state["stage_phases"][0] = "local_attn"
                cohort_state["stage_current_layer_ids"][0] = int(
                    scheduled_requests[0].completed_layer_count
                )
                return macro_batch

            shared_decode_attn_global_id = int(self._batch_creation_counter)
            for stage_idx in range(len(stage_reqs_start_loc) - 1):
                start_idx = stage_reqs_start_loc[stage_idx]
                end_idx = stage_reqs_start_loc[stage_idx + 1]
                stage_requests = scheduled_requests[start_idx:end_idx]
                stage_tokens = scheduled_tokens[start_idx:end_idx]
                micro_batch = self._create_batch(stage_requests, stage_tokens)
                micro_batch.set_global_id(shared_decode_attn_global_id)
                micro_batch.afd_stage_idx = stage_idx
                micro_batch.replay_decode_token_index = int(
                    stage_requests[0].current_decode_token_index
                )
                micro_batch.decode_attn_cohort_id = decode_attn_cohort_id
                micro_batch.decode_attn_cohort_request_ids = (
                    decode_attn_cohort_request_ids
                )
                if afd_stage_metadata is not None:
                    micro_batch.afd_stage_metadata = afd_stage_metadata
                normalized_stage_idx = int(stage_idx)
                cohort_state["active_stage_indices"].add(normalized_stage_idx)
                cohort_state["stage_phases"][normalized_stage_idx] = "local_attn"
                cohort_state["stage_current_layer_ids"][normalized_stage_idx] = int(
                    scheduled_requests[0].completed_layer_count
                )
                if first_micro_batch is None:
                    first_micro_batch = micro_batch
                else:
                    self._af_pending_micro_batches.append(micro_batch)

            return first_micro_batch


        logger.debug("[VLLMv1Engine][DECODE_ATTN] No requests to schedule")
        return None
