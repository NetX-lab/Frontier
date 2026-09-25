"""
vLLM v1 Engine Replica Scheduler

This scheduler simulates the admission control behavior of the vLLM v1 engine,
implementing two-phase scheduling, token budget management, and preemption mechanisms.

Key Features:
- Two-phase scheduling: Phase 1 (RUNNING requests), Phase 2 (WAITING requests)
- Token budget management per scheduling iteration
- FCFS and Priority-based scheduling policies
- Memory-pressure-driven preemption with policy-aware victim selection
- Support for MONOLITHIC, PREFILL, and DECODE cluster types

Reference:
- vLLM v1 scheduler: sota-infer-engine/vllm/vllm/v1/core/sched/scheduler.py
- Admission control guide: tests/debug/flow-level/admission_control_dev_guide_en.md
"""

from collections import deque
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frontier.config import global_vars
from frontier.attention.gdn.guards import model_has_gdn, validate_gdn_runtime_support
from frontier.attention.gdn.state import GatedDeltaNetStateSlotManager
from frontier.entities.batch import Batch, Request
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_decision_log import (
    _log_frontier_vllm_v1_schedule_decision,
)
from frontier.scheduler.replica_scheduler.vllm_v1_decode_attn_cohort import (
    DecodeAttentionCohort,
)
from frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy import (
    IterationSchedulingPolicy,
)
from frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation import KvBlockAllocation
from frontier.scheduler.replica_scheduler.vllm_v1_mtp_wait import (
    TargetEmbeddedMtpWaitPolicy,
)
from frontier.scheduler.replica_scheduler.vllm_v1_prefix_cache import (
    PrefixCacheAdmission,
    PrefixCacheLedger,
)
from frontier.scheduler.replica_scheduler.vllm_v1_role_schedules import (
    DisaggregatedRoleScheduling,
)
from frontier.spec_decode import is_spec_decode_enabled, method_uses_lookahead_slots
from frontier.types import ClusterType


class VLLMv1EngineReplicaScheduler(
    IterationSchedulingPolicy,
    KvBlockAllocation,
    PrefixCacheLedger,
    TargetEmbeddedMtpWaitPolicy,
    DecodeAttentionCohort,
    DisaggregatedRoleScheduling,
    BaseReplicaScheduler,
):
    """
    Replica scheduler that simulates vLLM v1 engine admission control.

    This scheduler implements the core scheduling algorithm from vLLM v1,
    including two-phase scheduling for RUNNING and WAITING requests,
    token budget enforcement, and memory-pressure-driven preemption.

    Attributes:
        _running_requests: List of requests currently being processed (RUNNING state)
        _preempted_requests: List of requests that have been preempted
        _max_num_running_reqs: Maximum number of concurrent requests
        _max_num_scheduled_tokens: Maximum tokens per scheduling iteration
        _scheduling_policy: Scheduling policy ('fcfs' or 'priority')
        _enable_preemption: Whether preemption is enabled
        _watermark_blocks: Number of blocks to keep as watermark
        _max_model_len: Maximum sequence length from model config
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GDN state is request-owned for the lifetime of a monolithic
        # admission.  Keep ownership at the scheduler boundary so waiting
        # continuations retain their slot while unadmitted requests consume
        # no state capacity.
        self._gdn_state_slot_manager = None
        if self._cluster_type == ClusterType.MONOLITHIC and model_has_gdn(
            self._replica_config.model_config
        ):
            validate_gdn_runtime_support(
                self._replica_config.model_config,
                num_pipeline_stages=self._replica_config.num_pipeline_stages,
                moe_expert_parallel_size=self._replica_config.moe_expert_parallel_size,
                attn_dp=self._replica_config.attn_dp,
            )
            self._gdn_state_slot_manager = GatedDeltaNetStateSlotManager(
                self._admitted_request_capacity
            )

        # vLLM v1 specific state - running requests tracking
        self._running_requests: List[Request] = []
        self._preempted_requests: List[Request] = []
        # Waiting queue for DECODE cluster - matches vLLM v1's two-phase scheduling
        # Requests arriving from prefill cluster enter here first before being
        # admitted to _running_requests during Phase 2 scheduling
        self._waiting_requests: List[Request] = []
        # Requests waiting for prefill-side KV transfer completion.
        self._pending_kv_transfer_requests: set[int] = set()
        self._scheduled_num_computed_tokens_by_request: Dict[int, int] = {}
        self._monolithic_pp_pending_terminal_release_iters: Dict[int, int] = {}
        self._monolithic_pp_waiting_sensitive_release_extensions: set[int] = set()
        self._monolithic_pp_terminal_release_followup_poll_pending = False
        self._monolithic_pp_mtp_output_wait_request_ids: set[int] = set()
        self._monolithic_pp_mtp_output_wait_remaining_iters: Dict[int, int] = {}
        self._monolithic_pp_mtp_near_full_prefill_request_ids: set[int] = set()
        self._monolithic_pp_mtp_single_output_wait_request_ids: set[int] = set()
        self._monolithic_pp_mtp_fractional_output_wait_counts: Dict[int, int] = {}
        self._monolithic_pp_mtp_output_wait_followup_poll_pending = False
        self._monolithic_pp_waiting_admission_delay_iters: Dict[int, int] = {}
        self._active_batch_request_counts: Dict[int, int] = {}

        # Configuration mapping from vLLM v1 parameters
        self._max_num_running_reqs = self._config.batch_size_cap
        self._max_num_scheduled_tokens = self._config.max_tokens_in_batch

        # TEMPORARY: Hardcode scheduling policy to 'priority' for Task 3 validation
        # TODO: In future work, expose this as a command-line parameter via config
        # Design note: The policy selection logic below uses a clean interface
        # that will make it easy to add parameter control without major refactoring
        self._scheduling_policy = self._get_scheduling_policy()

        self._enable_preemption = getattr(self._config, "enable_preemption", True)
        self._enable_chunked_prefill = bool(
            getattr(self._config, "enable_chunked_prefill", False)
        )
        self._enable_phase_aware_thinking_profile = bool(
            getattr(self._config, "enable_phase_aware_thinking_profile", False)
        )
        self._enable_final_round_priority_boost = bool(
            getattr(self._config, "enable_final_round_priority_boost", False)
        )
        self._final_round_priority_value = int(
            getattr(self._config, "final_round_priority_value", -1)
        )
        self._final_prefill_reserved_slots = int(
            getattr(self._config, "final_prefill_reserved_slots", 0)
        )
        self._final_prefill_reserved_tokens = int(
            getattr(self._config, "final_prefill_reserved_tokens", 0)
        )
        self._final_decode_reserved_slots = int(
            getattr(self._config, "final_decode_reserved_slots", 0)
        )
        self._enable_final_running_request_reclaim = bool(
            getattr(self._config, "enable_final_running_request_reclaim", False)
        )
        self._active_iteration_round_class: Optional[str] = None
        self._long_prefill_token_threshold = int(
            getattr(self._config, "long_prefill_token_threshold", 0)
        )
        if self._long_prefill_token_threshold < 0:
            raise ValueError(
                "long_prefill_token_threshold must be >= 0, got "
                f"{self._long_prefill_token_threshold}"
            )
        if self._long_prefill_token_threshold > 0 and not self._enable_chunked_prefill:
            raise ValueError(
                "long_prefill_token_threshold > 0 requires enable_chunked_prefill=True"
            )

        # Block management - watermark for memory safety
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

        # Max model length from replica config
        self._max_model_len = getattr(
            self._request_generator_config, "max_tokens", 8192
        )

        # Speculative decoding runtime (Phase 1)
        self._spec_decode_config = getattr(
            self._replica_config, "speculative_decoding_config", None
        )
        self._spec_decode_enabled = is_spec_decode_enabled(self._spec_decode_config)
        self._spec_method_uses_lookahead_slots = False
        if self._spec_decode_enabled:
            self._spec_method_uses_lookahead_slots = method_uses_lookahead_slots(
                self._spec_decode_config.method
            )
            if self._cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
                raise ValueError(
                    "Speculative decoding Phase 1 does not support DECODE_ATTN/DECODE_FFN "
                    f"cluster scheduling, got cluster_type={self._cluster_type}."
                )

        # Initialize micro-batch size for DECODE_ATTN (PD+AF) use case
        #  - prefer cluster-specific decode_attn_micro_batch_size from ClusterConfig
        if self._cluster_type == ClusterType.DECODE_ATTN:
            mbs = None

            if getattr(self, "_cluster_scheduler", None) is not None:
                cfg = getattr(self._cluster_scheduler, "_config", None)
                if cfg is not None:
                    mbs = getattr(cfg, "decode_attn_micro_batch_size", None)

            if mbs is None:
                raise ValueError("Missing decode_attn_micro_batch_size in ClusterConfig")
                # mbs = 1  # Conservative default
            self._micro_batch_size = int(mbs)
            logger = get_cluster_logger(__name__, self._cluster_type.name)
            logger.info(
                f"[VLLMv1Engine][DECODE_ATTN] Initialized micro_batch_size={self._micro_batch_size}"
            )
            self._af_pending_micro_batches = deque()

        # Validate scheduling policy
        if self._scheduling_policy not in ("fcfs", "priority"):
            raise ValueError(
                f"Invalid scheduling policy: {self._scheduling_policy}. "
                "Must be 'fcfs' or 'priority'."
            )

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        logger.info(
            f"[VLLMv1Engine] Initialized scheduler for replica {self._replica_id}: "
            f"max_running_reqs={self._max_num_running_reqs}, "
            f"max_tokens={self._max_num_scheduled_tokens}, "
            f"policy={self._scheduling_policy}, "
            f"preemption={self._enable_preemption}, "
            f"chunked_prefill={self._enable_chunked_prefill}, "
            f"final_prefill_reserved_slots={self._final_prefill_reserved_slots}, "
            f"final_prefill_reserved_tokens={self._final_prefill_reserved_tokens}, "
            f"final_decode_reserved_slots={self._final_decode_reserved_slots}, "
            f"enable_final_running_request_reclaim={self._enable_final_running_request_reclaim}, "
            f"long_prefill_token_threshold={self._long_prefill_token_threshold}, "
            f"watermark_blocks={self._watermark_blocks}, "
            f"spec_decode_enabled={self._spec_decode_enabled}, "
            f"spec_method={getattr(self._spec_decode_config, 'method', None)}, "
            f"spec_num_tokens={getattr(self._spec_decode_config, 'num_speculative_tokens', 0)}, "
            f"spec_lookahead_slots={self._spec_method_uses_lookahead_slots}"
        )

        self._schedule_iteration_id = 0
        self._active_schedule_iteration_id = -1
        self._prefix_cache_identity_event_seq = 0
        self._decode_attn_next_cohort_id = 0
        self._current_iteration_token_budget = 0
        self._prefill_iteration_reserved_slots_remaining = 0
        self._prefill_iteration_reserved_tokens_remaining = 0
        self._decode_iteration_reserved_slots_remaining = 0
        self._kv_cache_manager: Optional[ReplicaKVCacheManager] = None
        if (
            bool(getattr(self._config, "enable_prefix_caching", False))
            and self._cluster_type in (ClusterType.MONOLITHIC, ClusterType.PREFILL)
        ):
            self._kv_cache_manager = ReplicaKVCacheManager(
                block_size=int(self._config.block_size),
                num_gpu_blocks=int(self._config.num_blocks),
                enable_caching=True,
                caching_hash_algo=str(
                    getattr(self._config, "prefix_caching_hash_algo", "builtin")
                ),
                num_preallocate_tokens=int(
                    getattr(self._config, "num_preallocate_tokens", 0)
                ),
            )

    def _create_batch(self, requests: List[Request], num_tokens: List[int]) -> Batch:
        # The scheduler frontier already counts this batch's tokens. A prompt
        # chunk scheduled while the request's previous chunk is in flight
        # attends to tokens the Request counts only when that chunk ends.
        num_context_tokens = [
            request.num_context_tokens
            if request.is_decoding
            else self._get_scheduler_num_computed_tokens(request) - scheduled_tokens
            for request, scheduled_tokens in zip(requests, num_tokens)
        ]
        batch = super()._create_batch(requests, num_tokens, num_context_tokens)
        metadata = self._build_decode_cuda_graph_metadata(batch)
        if metadata is not None:
            batch.decode_cuda_graph_metadata = metadata
        spec_metadata = self._build_spec_decode_batch_metadata(batch)
        if spec_metadata is not None:
            batch.spec_decode_metadata = spec_metadata
        self._record_monolithic_pp_mtp_near_full_prefill_slices(batch)
        self._mark_batch_requests_active(batch)
        return batch

    def _get_active_batch_request_counts(self) -> Dict[int, int]:
        active_counts = getattr(self, "_active_batch_request_counts", None)
        if active_counts is None:
            active_counts = {}
            self._active_batch_request_counts = active_counts
        return active_counts

    def _mark_batch_requests_active(self, batch: Batch) -> None:
        active_counts = self._get_active_batch_request_counts()
        for request in batch.requests:
            active_counts[request.id] = active_counts.get(request.id, 0) + 1

    def _release_batch_requests_active(self, batch: Batch) -> None:
        # A request preempted while this batch was in flight already left it.
        active_counts = self._get_active_batch_request_counts()
        for request in batch.current_execution_requests:
            current_count = active_counts.get(request.id, 0)
            if current_count <= 1:
                active_counts.pop(request.id, None)
            else:
                active_counts[request.id] = current_count - 1

    def _is_request_active_in_batch(self, request: Request) -> bool:
        return self._get_active_batch_request_counts().get(request.id, 0) > 0

    def _roll_back_rejected_drafts(self, batch: Batch) -> None:
        """Take a speculative step's rejected drafts off the scheduler frontier.

        vLLM advances num_computed_tokens by the whole verify width when it
        schedules the step and subtracts the rejected drafts when the step's
        output arrives (scheduler.py update_from_output).
        """
        metadata = batch.spec_decode_metadata
        if metadata is None:
            return
        rejected_by_request_id = {
            request.id: rejected
            for request, rejected in zip(
                batch.requests, metadata.rejected_draft_tokens_per_request
            )
        }
        for request in batch.current_execution_requests:
            rejected = rejected_by_request_id[request.id]
            if rejected:
                self._scheduled_num_computed_tokens_by_request[request.id] -= rejected

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
                    f"source_dp={self._replica_local_id}"
                )

            if request.id in self._allocation_map:
                self._free_request_resources(request)
            self._pending_kv_transfer_requests.discard(request.id)

    def on_batch_end(self, batch: Batch) -> None:
        """
        Handle batch completion - update running requests state.

        For completed requests: free resources and remove from running list.
        For ongoing requests: keep in running list for next iteration.

        Special handling for PREFILL cluster in disaggregated mode:
        - Requests are transferred to DECODE cluster after prefill completion
        - Partially-prefilled requests stay in PREFILL running list for next chunk

        Args:
            batch: The batch that has completed execution
        """
        self._num_running_batches -= 1

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        self._release_batch_requests_active(batch)
        self._roll_back_rejected_drafts(batch)

        for request in batch.requests:
            self._refresh_target_embedded_mtp_prefill_boundary_state(batch, request)
            if self._cluster_type == ClusterType.DECODE_ATTN:
                decode_attn_cohort_id = getattr(
                    batch,
                    "decode_attn_cohort_id",
                    None,
                )
                if decode_attn_cohort_id is None:
                    self._decode_attn_active_request_ids.discard(request.id)
                else:
                    cohort_states = self._get_decode_attn_active_cohort_states()
                    cohort_state = cohort_states.get(int(decode_attn_cohort_id))
                    if cohort_state is None:
                        self._decode_attn_active_request_ids.discard(request.id)
                    else:
                        cohort_state["pending_request_ids"].discard(request.id)
                        if not cohort_state["pending_request_ids"]:
                            for cohort_request_id in cohort_state["all_request_ids"]:
                                self._decode_attn_active_request_ids.discard(
                                    cohort_request_id
                                )
                            cohort_states.pop(int(decode_attn_cohort_id), None)
                            if (
                                getattr(self, "_decode_attn_open_cohort_id", None)
                                == decode_attn_cohort_id
                            ):
                                self._decode_attn_open_cohort_id = None

            if request.completed:
                if any(request in queue for queue in self._waiting_queues()):
                    # It stopped on the sample of the step it was preempted
                    # from, and preemption already freed its KV.
                    self.remove_stopped_waiting_request(
                        request, batch.completed_at
                    )
                    continue
                extra_release_iters = (
                    self._get_monolithic_pp_extra_terminal_release_iters()
                )
                if extra_release_iters > 0:
                    pending_release_iters = (
                        self._get_monolithic_pp_pending_terminal_release_iters()
                    )
                    pending_release_iters[request.id] = max(
                        pending_release_iters.get(request.id, 0),
                        extra_release_iters,
                    )
                    logger.debug(
                        "[VLLMv1Engine] Request %s completed, deferring free for "
                        "%s extra MONOLITHIC+PP terminal iteration(s)",
                        request.id,
                        extra_release_iters,
                    )
                    continue
                # Request finished - free resources and remove from running
                self._free_request_resources(request)
                self._scheduled_num_computed_tokens_by_request.pop(request.id, None)
                if request in self._running_requests:
                    self._running_requests.remove(request)
                logger.debug(
                    f"[VLLMv1Engine] Request {request.id} completed, "
                    f"freed resources, running_reqs={len(self._running_requests)}"
                )
            elif self._cluster_type == ClusterType.PREFILL:
                # PREFILL cluster in disaggregated mode:
                # Requests are transferred to DECODE cluster after prefill completion
                # MODIFIED: Do NOT free KV cache here - it will be freed when transfer completes
                # This matches vLLM v1 behavior (scheduler.py:1480-1501)
                # where blocks are freed on finished_sending event

                if request.is_prefill_complete:
                    # Remove from running list only after prefill is fully complete.
                    self._scheduled_num_computed_tokens_by_request.pop(request.id, None)
                    if request in self._running_requests:
                        self._running_requests.remove(request)

                    # Track that this request's KV cache is pending transfer.
                    self._pending_kv_transfer_requests.add(request.id)

                    logger.info(
                        f"[VLLMv1Engine] Request {request.id} prefill complete, "
                        f"KV cache retained for transfer (blocks={self._allocation_map.get(request.id, 0)}), "
                        f"running_reqs={len(self._running_requests)}"
                    )
                else:
                    # Partial prefill: keep request in running queue for the next chunk.
                    logger.debug(
                        f"[VLLMv1Engine] Request {request.id} partial prefill complete, "
                        f"processed_tokens={request.num_processed_tokens}, "
                        f"running_reqs={len(self._running_requests)}"
                    )
            elif self._cluster_type == ClusterType.DECODE_ATTN:
                # DECODE_ATTN in PD-AF mode:
                # This method is called ONLY by GlobalBatchEndEvent (decode step complete)
                # NOT called for intermediate layers (those go through _af_immediate_batch_queue)

                # Note: _num_running_batches already decremented at method start (line 151)
                # This is correct - decode step completed, release pipeline slot

                # For completed requests: free resources and remove from running list
                # For ongoing requests: keep in _running_requests for next decode step
                #
                # Note: request._completed_layer_count is already reset to 0 by request.on_batch_end()
                # This ensures layer-consistent grouping in next _schedule_decode_attn_only() call

                if request.completed:
                    # Request finished all decode tokens - free resources and remove
                    self._free_request_resources(request)
                    if request in self._running_requests:
                        self._running_requests.remove(request)
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Request {request.id} completed, "
                        f"freed resources, running_reqs={len(self._running_requests)}"
                    )
                else:
                    # Request continues with next decode token - keep in _running_requests
                    # Phase 1 of _schedule_decode_attn_only() will pick this up
                    logger.debug(
                        f"[VLLMv1Engine][DECODE_ATTN] Request {request.id} continues to next decode step, "
                        f"processed_tokens={request.num_processed_tokens}"
                    )
            else:
                # Request continues - keep in running list
                # (will be scheduled again in next iteration)
                if self._should_apply_monolithic_pp_mtp_output_wait(request):
                    output_wait_iters = (
                        self._get_monolithic_pp_mtp_output_wait_iters_for_request(
                            request
                        )
                    )
                    if output_wait_iters > 0:
                        self._add_monolithic_pp_mtp_output_wait(
                            request.id,
                            wait_iters=output_wait_iters,
                        )
                logger.debug(
                    f"[VLLMv1Engine] Request {request.id} continues, "
                    f"processed_tokens={request.num_processed_tokens}"
                )

    def _waiting_queues(self) -> Tuple[List[Request], List[Request], List[Request]]:
        # Preemption inserts a victim into `_request_queue`, or into
        # `_waiting_requests` on DECODE and DECODE_ATTN. A later waiting-queue
        # rebuild may move it into `_preempted_requests`.
        return (self._request_queue, self._preempted_requests, self._waiting_requests)

    def remove_stopped_waiting_request(self, request: Request, time: float) -> None:
        """Remove a preempted request that stopped on its in-flight sample.

        vLLM removes such a request from waiting, so it never resumes.
        """
        for waiting_queue in self._waiting_queues():
            if request in waiting_queue:
                waiting_queue.remove(request)
                break
        request.on_leave_waiting_queue(time, self._cluster_type)

    def _schedule_running_requests(
        self, token_budget: int, preempted_requests: List[Request]
    ) -> Tuple[int, List[Request], List[int]]:
        """
        Phase 1: Schedule requests currently in RUNNING state.

        Iterate through running requests and try to allocate memory for
        their next tokens. May trigger preemption if memory is insufficient.

        Args:
            token_budget: Remaining token budget for this iteration
            preempted_requests: List to track preempted requests

        Returns:
            Tuple of (remaining_budget, scheduled_requests, num_tokens_list)
        """
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        scheduled = []
        num_tokens_list = []
        waiting_final_prefill_count = (
            self._count_final_fast_lane_requests(
                self._preempted_requests + self._request_queue,
                final_predicate=self._is_final_prefill_fast_lane_request,
            )
            if self._cluster_type == ClusterType.PREFILL
            else 0
        )

        self._current_iteration_token_budget = token_budget
        req_index = 0
        while req_index < len(self._running_requests) and token_budget > 0:
            self._current_iteration_token_budget = token_budget
            request = self._running_requests[req_index]
            is_final_prefill_running_request = (
                self._cluster_type == ClusterType.PREFILL
                and self._is_final_prefill_fast_lane_request(request)
            )
            is_hidden_prefill_running_request = (
                self._cluster_type == ClusterType.PREFILL
                and not request.is_prefill_complete
                and not is_final_prefill_running_request
            )

            if (
                request.id
                in self._get_monolithic_pp_pending_terminal_release_iters()
            ):
                req_index += 1
                continue

            continuation_request_ids = getattr(
                self, "_continuation_request_ids", set()
            )
            if request.id in continuation_request_ids:
                logger.debug(
                    "[VLLMv1Engine] Phase 1: skipping req=%s "
                    "(already scheduled in current cycle)",
                    request.id,
                )
                req_index += 1
                continue

            if (
                self._cluster_type == ClusterType.MONOLITHIC
                and self._num_stages > 1
                and request.id
                in self._get_monolithic_pp_mtp_output_wait_request_ids()
            ):
                logger.debug(
                    "[VLLMv1Engine][MONOLITHIC] Phase 1: delaying req=%s "
                    "for one PP output-visible MTP scheduler step",
                    request.id,
                )
                req_index += 1
                continue

            # vLLM skips an in-flight request only once every prompt token is
            # scheduled; the next chunk goes out while the previous one is in
            # flight (scheduler.py, the num_new_tokens == 0 branch).
            active_in_pp_batch = (
                self._cluster_type in {ClusterType.MONOLITHIC, ClusterType.DECODE}
                and self._num_stages > 1
                and self._is_request_active_in_batch(request)
                and (
                    request.is_decoding
                    or self._get_request_next_num_tokens(request) == 0
                )
            )
            if active_in_pp_batch:
                if self._cluster_type == ClusterType.MONOLITHIC:
                    reserved_tokens = (
                        self._get_monolithic_pp_mtp_visible_budget_reservation_tokens(
                            request,
                            token_budget,
                        )
                    )
                    if reserved_tokens > 0:
                        token_budget -= reserved_tokens
                        self._current_iteration_token_budget = token_budget
                        logger.debug(
                            "[VLLMv1Engine][MONOLITHIC] Phase 1: reserving "
                            "%s token(s) for active output-visible MTP req=%s",
                            reserved_tokens,
                            request.id,
                        )
                logger.debug(
                    "[VLLMv1Engine][%s] Phase 1: skipping req=%s "
                    "(already active in a PP batch)",
                    self._cluster_type.name,
                    request.id,
                )
                req_index += 1
                continue

            if (
                self._cluster_type in {ClusterType.MONOLITHIC, ClusterType.DECODE}
                and self._num_stages > 1
                and getattr(request, "completed_layer_count", 0) != 0
            ):
                logger.debug(
                    "[VLLMv1Engine][%s] Phase 1: skipping in-flight req=%s "
                    "with layer_count=%s (PP continuation still active)",
                    self._cluster_type.name,
                    request.id,
                    getattr(request, "completed_layer_count", None),
                )
                req_index += 1
                continue

            # Calculate number of new tokens to process
            num_new_tokens = self._get_request_next_num_tokens(request)

            # Apply max_model_len limit
            scheduler_num_computed_tokens = self._get_scheduler_num_computed_tokens(
                request
            )
            max_allowed = self._max_model_len - scheduler_num_computed_tokens
            num_new_tokens = min(num_new_tokens, max_allowed)
            num_new_tokens = self._apply_long_prefill_token_threshold(
                request, num_new_tokens
            )

            # Apply token budget limit
            effective_token_budget = token_budget
            if (
                is_hidden_prefill_running_request
                and waiting_final_prefill_count > 0
                and self._prefill_iteration_reserved_tokens_remaining > 0
            ):
                effective_token_budget = max(
                    token_budget
                    - min(
                        self._prefill_iteration_reserved_tokens_remaining,
                        token_budget,
                    ),
                    0,
                )
                if effective_token_budget <= 0:
                    req_index += 1
                    continue
            num_new_tokens = min(num_new_tokens, effective_token_budget)

            if num_new_tokens <= 0:
                req_index += 1
                continue

            # Try to allocate with preemption
            preempted_count_before = len(preempted_requests)
            can_schedule = self._try_allocate_with_preemption(
                request,
                num_new_tokens,
                preempted_requests,
                scheduler_num_computed_tokens=scheduler_num_computed_tokens,
            )
            token_budget = self._rollback_current_iteration_preempted_requests(
                scheduled_requests=scheduled,
                scheduled_num_tokens=num_tokens_list,
                newly_preempted_requests=preempted_requests[
                    preempted_count_before:
                ],
                token_budget=token_budget,
            )

            if can_schedule:
                self._advance_scheduler_num_computed_tokens(request, num_new_tokens)
                scheduled.append(request)
                num_tokens_list.append(num_new_tokens)
                token_budget -= num_new_tokens
                self._current_iteration_token_budget = token_budget
                if is_final_prefill_running_request:
                    self._prefill_iteration_reserved_tokens_remaining = max(
                        self._prefill_iteration_reserved_tokens_remaining
                        - num_new_tokens,
                        0,
                    )
                req_index += 1

                # Flow validation: log RUNNING request scheduled
                logger.info(
                    f"[RUNNING_SCHEDULED] req={request.id}, "
                    f"num_new_tokens={num_new_tokens}, "
                    f"blocks_allocated={self._allocation_map.get(request.id, 0)}"
                )
                available_blocks_running = int(
                    self._config.num_blocks - self._num_allocated_blocks
                )
                self._emit_schedule_decision_event(
                    event="decision",
                    decision_result="RUNNING_SCHEDULED",
                    request_id=request.id,
                    token_budget=token_budget,
                    available_blocks=available_blocks_running,
                    num_tokens=num_new_tokens,
                )
            else:
                # Request was preempted, stop processing running requests
                break

        return token_budget, scheduled, num_tokens_list

    def _get_sorted_waiting_queue(self) -> List[Request]:
        """
        Get waiting requests sorted by scheduling policy.

        FCFS: Original queue order (first arrived first).
        Priority: Sorted by (priority, arrival_time) ascending.
        Thinking-round priority: Final-round requests first, then by
        existing policy within each tier.

        Returns:
            List of requests in scheduling order
        """
        # Combine main queue and preempted requests
        # Preempted requests should be prioritized (at front of queue)
        combined = self._preempted_requests + self._request_queue

        if getattr(
            getattr(self, "_config", None), "enable_thinking_round_priority", False
        ):
            # Final-round requests first, then by priority, then FIFO
            return sorted(
                combined,
                key=lambda r: (
                    0 if r.is_final_thinking_round else 1,
                    r.priority,
                    r.arrived_at,
                ),
            )
        elif self._scheduling_policy == "priority":
            # Sort by priority (ascending) then arrival time (ascending)
            return sorted(combined, key=lambda r: (r.priority, r.arrived_at))
        else:
            # FCFS: maintain insertion order (preempted first)
            return combined

    def _set_waiting_queues_from_ordered_requests(
        self, ordered_requests: List[Request]
    ) -> None:
        """Rebuild waiting queues from ordered requests.

        Requests with `_preempted=True` stay in `_preempted_requests` to keep
        preemption recovery semantics and queue priority.
        """
        self._preempted_requests = []
        self._request_queue = []
        for request in ordered_requests:
            if getattr(request, "_preempted", False):
                self._preempted_requests.append(request)
            else:
                self._request_queue.append(request)

    def _schedule_waiting_requests(
        self, token_budget: int
    ) -> Tuple[int, List[Request], List[int]]:
        """
        Phase 2: Schedule requests in WAITING state.

        Only called when no preemption occurred in Phase 1.
        Attempts to admit new requests from the waiting queue.

        Args:
            token_budget: Remaining token budget for this iteration

        Returns:
            Tuple of (remaining_budget, scheduled_requests, num_tokens_list)
        """
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        scheduled = []
        num_tokens_list = []

        fast_lane_prefill_enabled = self._cluster_type == ClusterType.PREFILL and (
            self._final_prefill_reserved_slots > 0
            or self._final_prefill_reserved_tokens > 0
        )

        # Get sorted waiting queue based on policy
        waiting_queue = (
            self._build_prefill_waiting_queue()
            if fast_lane_prefill_enabled
            else deque(self._get_sorted_waiting_queue())
        )
        skipped_waiting_requests: deque[Request] = deque()

        self._current_iteration_token_budget = token_budget
        while waiting_queue and token_budget > 0:
            self._current_iteration_token_budget = token_budget
            final_waiting_count = (
                self._count_final_fast_lane_requests(
                    waiting_queue,
                    final_predicate=self._is_final_prefill_fast_lane_request,
                )
                if fast_lane_prefill_enabled
                else 0
            )
            has_final_waiting = final_waiting_count > 0
            # Check max concurrent requests limit
            if len(self._running_requests) >= self._max_num_running_reqs:
                break

            request = waiting_queue[0]
            if self._should_defer_monolithic_pp_waiting_admission(request):
                logger.debug(
                    "[VLLMv1Engine][MONOLITHIC] Phase 2: delaying req=%s "
                    "until a PP output-visible scheduler boundary",
                    request.id,
                )
                break

            is_final_prefill_request = fast_lane_prefill_enabled and (
                self._is_final_prefill_fast_lane_request(request)
            )
            is_hidden_prefill_request = (
                fast_lane_prefill_enabled
                and not request.is_prefill_complete
                and not is_final_prefill_request
            )
            computed_blocks = None
            prefix_cached_tokens = 0
            prefix_cache_admission: Optional[PrefixCacheAdmission] = None
            scheduler_num_computed_tokens = self._get_scheduler_num_computed_tokens(
                request
            )

            # Calculate number of new tokens to process
            if self._is_prefix_caching_enabled() and not request.is_decoding:
                prefix_cache_admission = self._prepare_prefix_cache_admission(
                    request
                )
                computed_blocks = list(
                    prefix_cache_admission.effective_hit_blocks
                )
                prefix_cached_tokens = int(
                    prefix_cache_admission.effective_cached_tokens
                )
                num_new_tokens = int(prefix_cache_admission.num_new_tokens)
                max_allowed = self._max_model_len - prefix_cached_tokens
            else:
                num_new_tokens = self._get_request_next_num_tokens(request)
                max_allowed = self._max_model_len - scheduler_num_computed_tokens

            # Apply max_model_len limit
            num_new_tokens = min(num_new_tokens, max_allowed)
            num_new_tokens = self._apply_long_prefill_token_threshold(
                request, num_new_tokens
            )

            effective_token_budget = token_budget
            if (
                is_hidden_prefill_request
                and has_final_waiting
                and self._prefill_iteration_reserved_slots_remaining > 0
                and len(self._running_requests)
                >= (
                    self._max_num_running_reqs
                    - self._prefill_iteration_reserved_slots_remaining
                )
            ):
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue
            if (
                is_hidden_prefill_request
                and has_final_waiting
                and self._prefill_iteration_reserved_tokens_remaining > 0
            ):
                effective_token_budget = max(
                    token_budget
                    - min(
                        self._prefill_iteration_reserved_tokens_remaining,
                        token_budget,
                    ),
                    0,
                )
                if effective_token_budget <= 0:
                    waiting_queue.popleft()
                    skipped_waiting_requests.append(request)
                    continue

            # When chunked prefill is disabled, waiting prefills that exceed token
            # budget are skipped for this iteration.
            if (
                not self._enable_chunked_prefill
                and not request.is_decoding
                and num_new_tokens > effective_token_budget
            ):
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue

            # Apply token budget limit after chunked-prefill guard
            num_new_tokens = min(num_new_tokens, effective_token_budget)

            if num_new_tokens <= 0:
                waiting_queue.popleft()
                continue

            # Try to allocate (no preemption for waiting requests in Phase 2)
            if not self._can_allocate_request(
                request,
                num_new_tokens,
                new_computed_blocks=computed_blocks,
                scheduler_num_computed_tokens=scheduler_num_computed_tokens,
            ):
                # Flow validation: log memory pressure for waiting queue admission
                available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
                logger.info(
                    f"[MEMORY_PRESSURE] trigger=waiting_allocation_failed, "
                    f"requesting_req={request.id}, "
                    f"requested_tokens={num_new_tokens}, "
                    f"available_blocks={available_blocks}, "
                    f"running_queue_size={len(self._running_requests)}, "
                    f"waiting_queue_size={len(waiting_queue)}"
                )
                # Cannot allocate - stop scheduling new requests
                break

            self._allocate_request(
                request,
                num_new_tokens,
                new_computed_blocks=computed_blocks,
                prefix_cache_admission=(
                    replace(
                        prefix_cache_admission,
                        num_new_tokens=int(num_new_tokens),
                    )
                    if prefix_cache_admission is not None
                    else None
                ),
                scheduler_num_computed_tokens=scheduler_num_computed_tokens,
            )

            # Commit queue ownership only after KV and state admission succeeds
            waiting_queue.popleft()
            was_preempted = request in self._preempted_requests
            if request in self._preempted_requests:
                self._preempted_requests.remove(request)
            if request in self._request_queue:
                self._request_queue.remove(request)
            self._get_monolithic_pp_waiting_admission_delay_iters().pop(
                request.id, None
            )

            # Record leaving waiting queue for waiting time tracking
            request.on_leave_waiting_queue(
                self._current_schedule_time, self._cluster_type
            )

            if prefix_cached_tokens > 0:
                request.on_cache_hit(prefix_cached_tokens)
            self._advance_scheduler_num_computed_tokens(request, num_new_tokens)

            # Add to running requests
            self._running_requests.append(request)

            # Clear preempted flag if set
            request._preempted = False

            scheduled.append(request)
            num_tokens_list.append(num_new_tokens)
            token_budget -= num_new_tokens
            self._current_iteration_token_budget = token_budget
            if is_final_prefill_request:
                self._prefill_iteration_reserved_slots_remaining = max(
                    self._prefill_iteration_reserved_slots_remaining - 1,
                    0,
                )
                self._prefill_iteration_reserved_tokens_remaining = max(
                    self._prefill_iteration_reserved_tokens_remaining - num_new_tokens,
                    0,
                )

            # Flow validation: log WAITING request admission
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
                # Preempted requests need full recomputation from prefill tokens
                recompute_tokens = request.num_prefill_tokens
                logger.info(
                    f"[PREEMPTION_RECOVERY] req={request.id}, "
                    f"was_preempted=True, "
                    f"recompute_tokens={recompute_tokens}"
                )

        # vLLM parity for skipped waiting requests:
        # prepend skipped queue back to waiting queue.
        if skipped_waiting_requests:
            if self._scheduling_policy == "priority":
                merged_requests = list(waiting_queue) + list(skipped_waiting_requests)
                waiting_queue = deque(
                    sorted(merged_requests, key=lambda r: (r.priority, r.arrived_at))
                )
            else:
                waiting_queue.extend(skipped_waiting_requests)

        self._set_waiting_queues_from_ordered_requests(list(waiting_queue))

        return token_budget, scheduled, num_tokens_list

    def _get_next_batch(self, is_micro_batch: bool = False) -> Optional[Batch]:
        """
        Build the next batch using vLLM v1 two-phase scheduling algorithm.

        Phase 1: Schedule RUNNING requests (decode phase)
        Phase 2: Schedule WAITING requests (prefill phase) - only if no preemption

        This method handles cluster-type-specific behavior:
        - MONOLITHIC: Full two-phase scheduling
        - PREFILL: Two-phase scheduling for running partial-prefill + waiting admission
        - DECODE: Only Phase 1 (scheduling running requests)

        Args:
            is_micro_batch: Whether this is for micro-batch (ignored in vLLM v1)

        Returns:
            Optional[Batch]: The next batch to execute, or None if no work
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        self._active_schedule_iteration_id = self._schedule_iteration_id
        self._schedule_iteration_id += 1
        self._refresh_iteration_scheduler_profile()
        logger.info(
            "[ITERATION_PROFILE] round_class=%s max_tokens=%s batch_size_cap=%s chunked_prefill=%s",
            self._active_iteration_round_class,
            self._max_num_scheduled_tokens,
            self._max_num_running_reqs,
            self._enable_chunked_prefill,
        )

        # Route to cluster-specific scheduling
        if self._cluster_type == ClusterType.PREFILL:
            return self._schedule_prefill_only()
        elif self._cluster_type == ClusterType.DECODE:
            return self._schedule_decode_only()
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            return self._schedule_decode_attn_only(is_micro_batch)
        else:
            # MONOLITHIC or other: full two-phase scheduling
            return self._schedule_two_phase()

    def _schedule_two_phase(self) -> Optional[Batch]:
        """
        Full two-phase scheduling for MONOLITHIC cluster.

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
        released = self._materialize_monolithic_pp_terminal_release_before_iteration_start()
        token_budget = self._max_num_scheduled_tokens
        available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
        waiting_count = len(self._request_queue) + len(self._preempted_requests)
        waiting_final_prefill_count = self._count_final_fast_lane_requests(
            self._preempted_requests + self._request_queue,
            final_predicate=self._is_final_prefill_fast_lane_request,
        )
        self._prefill_iteration_reserved_slots_remaining = (
            self._final_prefill_reserved_slots
            if (
                self._enable_final_running_request_reclaim
                and waiting_final_prefill_count > 0
            )
            else 0
        )
        self._prefill_iteration_reserved_tokens_remaining = (
            self._final_prefill_reserved_tokens
            if (
                self._enable_final_running_request_reclaim
                and waiting_final_prefill_count > 0
            )
            else 0
        )
        waiting_final_prefill_count = self._count_final_fast_lane_requests(
            self._preempted_requests + self._request_queue,
            final_predicate=self._is_final_prefill_fast_lane_request,
        )
        self._prefill_iteration_reserved_slots_remaining = (
            self._final_prefill_reserved_slots if waiting_final_prefill_count > 0 else 0
        )
        self._prefill_iteration_reserved_tokens_remaining = (
            self._final_prefill_reserved_tokens if waiting_final_prefill_count > 0 else 0
        )

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

        if self._monolithic_pp_terminal_release_followup_poll_pending:
            self._emit_schedule_decision_event(
                event="iteration_end",
                decision_result=None,
                request_id=None,
                token_budget=token_budget,
                num_tokens=0,
                available_blocks=int(
                    self._config.num_blocks - self._num_allocated_blocks
                ),
                batch_request_ids=[],
                request_num_tokens=[],
                batch_size=0,
                batch_num_tokens=0,
            )
            return None

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

        # === Phase 2: Schedule WAITING requests (only if no preemption) ===
        if not preempted_requests and not self._has_monolithic_pp_pending_terminal_release():
            # Flow validation: log Phase 2 start
            waiting_count_p2 = len(self._request_queue) + len(self._preempted_requests)
            logger.info(
                f"[PHASE2_START] waiting_count={waiting_count_p2}, "
                f"token_budget={token_budget}, "
                f"running_count={len(self._running_requests)}"
            )

            token_budget, waiting_scheduled, waiting_tokens = (
                self._schedule_waiting_requests(token_budget)
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
        elif not preempted_requests and self._has_monolithic_pp_pending_terminal_release():
            logger.info(
                "[PHASE2_SKIPPED] waiting admission blocked by pending "
                "MONOLITHIC+PP terminal release boundary"
            )

        if not all_scheduled_requests:
            if self._has_monolithic_pp_mtp_output_wait():
                self._clear_monolithic_pp_mtp_output_wait()
                self._monolithic_pp_mtp_output_wait_followup_poll_pending = True
            released += self._advance_monolithic_pp_terminal_release_boundary()
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
            if released > 0:
                # The iteration applied an output and scheduled nothing, and vLLM publishes after such a step.
                self._cluster_scheduler.on_replica_batch_end(
                    self._current_schedule_time,
                    self._replica_id,
                    self._replica_local_id,
                    None,
                )
            return None

        # Match vLLM v1 output order: new/resumed admissions first, then running.
        ordered_scheduled_requests = waiting_scheduled + running_scheduled
        ordered_num_tokens = waiting_tokens + running_tokens

        # Flow validation: log batch formation
        total_tokens = sum(all_num_tokens)
        new_admitted = len(
            [r for r in all_scheduled_requests if r not in running_scheduled]
        )
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
        self._advance_monolithic_pp_terminal_release_boundary()

        return self._create_batch(ordered_scheduled_requests, ordered_num_tokens)

    @property

    def num_pending_requests(self) -> int:
        """
        Return total schedulable pending requests for this cluster.

        DECODE clusters admit requests from _waiting_requests. PREFILL and
        MONOLITHIC clusters admit from _request_queue plus _preempted_requests.
        """
        if self._cluster_type in (ClusterType.DECODE, ClusterType.DECODE_ATTN):
            return len(self._request_queue) + len(self._waiting_requests)
        return len(self._request_queue) + len(self._preempted_requests)

    def peek_waiting_requests(self) -> List[Request]:
        requests: List[Request] = []
        seen_request_ids: set[int] = set()
        for queue in (
            list(self._preempted_requests),
            list(self._request_queue),
            list(self._waiting_requests),
        ):
            for request in queue:
                if request.id in seen_request_ids:
                    continue
                seen_request_ids.add(request.id)
                requests.append(request)
        return requests

    def is_empty(self) -> bool:
        """
        Check if scheduler has no pending work.

        For DECODE cluster, also checks _waiting_requests queue.
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        stages_empty = all(
            stage_scheduler.is_empty()
            for stage_scheduler in self._replica_stage_schedulers.values()
        )
        af_len = (
            len(self._af_immediate_batch_queue)
            if hasattr(self, "_af_immediate_batch_queue")
            else 0
        )
        waiting_len = len(self._waiting_requests)
        running_len = len(self._running_requests)

        logger.info(
            f"[RS-IDLE-CHECK][replica={self._replica_id}][dp={self._replica_local_id}] "
            f"num_pending_requests={self.num_pending_requests}, waiting_requests={waiting_len}, "
            f"running_requests={running_len}, allocated_blocks={len(self._allocation_map)}, "
            f"num_running_batches={self._num_running_batches}, stages_empty={stages_empty}, af_immediate_len={af_len}"
        )
        # If AF immediate queue has pending batches, the replica is not idle
        if af_len > 0:
            return False
        return (
            self.num_pending_requests == 0
            and waiting_len == 0
            and running_len == 0
            and len(self._allocation_map) == 0
            and self._num_running_batches == 0
            and stages_empty
        )

    # ========== Request Addition Override ==========

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        For DECODE cluster: requests coming from prefill enter waiting queue
        first, then get admitted to running queue during Phase 2 scheduling.
        This matches vLLM v1's two-phase scheduling behavior.

        For other clusters: add to waiting queue.

        Args:
            request: The request to add
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        self._initialize_request_spec_decode_state(request)
        self._maybe_promote_final_round_priority(request)

        if self._cluster_type == ClusterType.DECODE:
            # For DECODE cluster, incoming requests enter waiting queue first
            # This matches vLLM v1's behavior where requests are admitted
            # from waiting to running during Phase 2 scheduling
            self._waiting_requests.append(request)

            # Flow validation: log KV transfer completion (request arrived from prefill)
            num_blocks_allocated = self._allocation_map.get(request.id, 0)
            logger.info(
                f"[KV_TRANSFER_STATE] req={request.id}, "
                f"status=TRANSFER_COMPLETE, "
                f"num_blocks_received={num_blocks_allocated}, "
                f"num_computed_tokens={request.num_processed_tokens}"
            )
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            # For DECODE_ATTN, new requests enter _waiting_requests for Phase 2 admission
            # This is consistent with DECODE cluster behavior
            #
            # Note: F→A returning batches do NOT go through this method
            # They are handled by add_batch_to_immediate_queue() -> _af_immediate_batch_queue
            #
            # Request entry points:
            # 1. New requests from prefill: add_request() -> _waiting_requests -> Phase 2
            # 2. F→A continuation: add_batch_to_immediate_queue() -> _af_immediate_batch_queue -> Priority 1

            self._waiting_requests.append(request)
            logger.debug(
                f"[VLLMv1Engine][DECODE_ATTN] Request {request.id} added to _waiting_requests, "
                f"queue_size={len(self._waiting_requests)}"
            )
        else:
            # For PREFILL/MONOLITHIC, add to waiting queue
            self._request_queue.append(request)
            if self._should_delay_monolithic_pp_waiting_admission_on_add(request):
                self._add_monolithic_pp_waiting_admission_delay(request.id)
