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

import math
from collections import deque
from dataclasses import replace
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from frontier.attention.gdn.guards import model_has_gdn, validate_gdn_runtime_support
from frontier.attention.gdn.state import GatedDeltaNetStateSlotManager
from frontier.entities.batch import Batch, DummyForwardBatch, Request
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_decode_attn_cohort import (
    DecodeAttentionCohort,
)
from frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy import (
    IterationSchedulingPolicy,
    priority_policy_key,
)
from frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation import KvBlockAllocation
from frontier.scheduler.replica_scheduler.vllm_v1_mtp_wait import (
    TargetEmbeddedMtpWaitPolicy,
)
from frontier.scheduler.replica_scheduler.vllm_v1_pp_terminal_release import (
    PipelineTerminalRelease,
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

# vLLM's DPEngineCoreProc runs its DP finish-sync all-reduce after every 32nd
# busy-loop iteration of a wave (`_has_global_unfinished_reqs`).
VLLM_DP_SYNC_INTERVAL = 32


class VLLMv1EngineReplicaScheduler(
    IterationSchedulingPolicy,
    KvBlockAllocation,
    PrefixCacheLedger,
    TargetEmbeddedMtpWaitPolicy,
    PipelineTerminalRelease,
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
        _max_model_len: Longest context (prompt plus output) a request may reach
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
        self._preemption_followup_poll_pending = False

        # The EngineCore batch queue: in-flight batch ids in admission order,
        # and the oldest one the engine blocks on once a pass leaves work in
        # flight. The PD-AF roles schedule micro-batches in their own loops.
        self._has_engine_batch_queue = self._cluster_type in (
            ClusterType.MONOLITHIC,
            ClusterType.PREFILL,
            ClusterType.DECODE,
        )
        self._in_flight_batch_ids: Deque[int] = deque()
        self._blocking_batch_id: Optional[int] = None
        # vLLM's DPEngineCoreProc: after a step that executed no batch, an
        # attention-DP engine of a MoE model runs a dummy forward on every
        # pipeline stage while its DP wave runs (`run_busy_loop`), so the other
        # engines' MoE layers always find a partner.
        self._runs_dp_dummy_passes = (
            self._has_engine_batch_queue
            and self._replica_is_moe
            and self._replica_config.attn_dp > 1
        )
        self._dummy_pass_due = False
        self._dummy_forwards_in_flight = 0
        # Such an engine counts the busy-loop iterations of its DP wave, each
        # running one forward, a batch or a dummy forward. After every
        # VLLM_DP_SYNC_INTERVAL-th, its host waits in an all-reduce over the DP
        # group, which ends the wave once no engine has unfinished requests.
        # A batch that ends during the wait keeps its output, and its queue
        # slot, until a later iteration pops it and applies the output.
        self._dp_wave_running = False
        self._dp_wave_steps = 0
        self._iteration_open = False
        self._waits_at_dp_sync = False
        self._unfinished_at_dp_sync = False
        self._dp_sync_released_lane_ids: List[int] = []
        self._unpopped_outputs: Deque[Callable[[float], List]] = deque()
        self._popped_output_events: List = []
        self._iteration_running_limit = math.inf
        # End of the engine loop's latest iteration: a schedule call that formed
        # a batch, a batch output, or a dummy forward. None before the first.
        # Only the batch-queue roles run vLLM's engine loop.
        self._last_iteration_end: Optional[float] = None

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

        # vLLM's max_model_len: set by --max-model-len, otherwise derived
        # from the model config.
        self._max_model_len = (
            self._config.max_model_len
            if self._config.max_model_len is not None
            else self._replica_config.model_config.max_position_embeddings
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

        vLLM advances num_computed_tokens by the scheduled width when it
        schedules the step and, when the step's output arrives, subtracts the
        scheduled tokens that produced no output (scheduler.py
        update_from_output). The scheduled width is one token short of the
        verify width on a MONOLITHIC target-embedded MTP request's first
        decode step, so the rollback is taken against the scheduled width.
        """
        metadata = batch.spec_decode_metadata
        if metadata is None:
            return
        rejected_by_request_id = {
            request.id: scheduled - committed
            for request, scheduled, committed in zip(
                batch.requests,
                batch.num_tokens,
                metadata.committed_tokens_per_request,
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

    def on_schedule(self, time: float = 0.0) -> List[Batch]:
        # vLLM runs iterations until its batch queue is full or an iteration
        # schedules nothing, then blocks on the oldest in-flight batch
        # (`EngineCore.step_with_batch_queue`). A request that arrives during
        # the block waits for that batch's output.
        if not self._has_engine_batch_queue:
            return super().on_schedule(time)
        if (
            self._blocking_batch_id is not None
            or self._dummy_forwards_in_flight
            or self._waits_at_dp_sync
        ):
            return []
        if self._runs_dp_dummy_passes:
            return self._run_dp_iterations(time)
        batches = self._start_iterations(super().on_schedule(time), time)
        self._in_flight_batch_ids.extend(batch.id for batch in batches)
        if self._in_flight_batch_ids:
            self._blocking_batch_id = self._in_flight_batch_ids[0]
            self._clear_followup_polls()
        return batches

    def _run_dp_iterations(self, time: float) -> List[Batch]:
        """Run DPEngineCoreProc's busy-loop iterations from `time` until the host waits.

        Each iteration schedules at most one batch. It returns at once while
        the batch queue has room and its oldest batch is in flight; otherwise
        it pops the oldest batch, waiting for it if it has not ended, and runs
        a dummy forward if it scheduled nothing (`run_busy_loop`). An
        iteration that waited or ran a dummy forward ends when the engine's
        next pass starts.
        """
        if self._dummy_pass_due:
            self._dummy_pass_due = False
            return self._issue_dummy_pass()
        if self._iteration_open:
            self._iteration_open = False
            if self._end_iteration():
                return []
        if not self._dp_wave_running and self.has_unfinished_requests:
            # An engine that receives a request starts a wave on every engine.
            for lane in self._dp_group():
                lane._dp_wave_running = True
        batches: List[Batch] = []
        while True:
            formed = self._schedule_iteration(time)
            batches.extend(formed)
            if self._unpopped_outputs:
                # The oldest batch ended while the host waited at the DP sync,
                # so this iteration pops it at once and applies its output.
                # The iteration ends with its own forward.
                apply_output = self._unpopped_outputs.popleft()
                self._popped_output_events.extend(apply_output(time))
                if not formed:
                    return batches + self._issue_dummy_pass()
            elif not formed or self._num_running_batches == self._num_stages:
                if self._in_flight_batch_ids:
                    self._blocking_batch_id = self._in_flight_batch_ids[0]
                    self._dummy_pass_due = not formed
                    self._iteration_open = True
                    self._clear_followup_polls()
                elif self._dp_wave_running:
                    return self._issue_dummy_pass()
                return batches
            if self._end_iteration():
                self._clear_followup_polls()
                return batches

    def _schedule_iteration(self, time: float) -> List[Batch]:
        """Schedule call of one busy-loop iteration: at most one batch."""
        self._iteration_running_limit = self._num_running_batches + 1
        batches = self._start_iterations(super().on_schedule(time), time)
        self._iteration_running_limit = math.inf
        self._in_flight_batch_ids.extend(batch.id for batch in batches)
        return batches

    def _end_iteration(self) -> bool:
        """Count one ended busy-loop iteration; return whether the host then waits at the DP sync."""
        self._dp_wave_steps += 1
        if self._dp_wave_steps % VLLM_DP_SYNC_INTERVAL:
            return False
        self._waits_at_dp_sync = True
        self._unfinished_at_dp_sync = self.has_unfinished_requests
        group = self._dp_group()
        if all(lane._waits_at_dp_sync for lane in group):
            # The all-reduce returns once the last engine joins it, and it ends
            # the wave when no engine joined with unfinished requests.
            wave_continues = any(lane._unfinished_at_dp_sync for lane in group)
            for lane in group:
                lane._waits_at_dp_sync = False
                if not wave_continues:
                    lane._dp_wave_running = False
                    lane._dp_wave_steps = 0
            self._dp_sync_released_lane_ids = [
                lane._replica_local_id for lane in group if lane is not self
            ]
        return self._waits_at_dp_sync

    def consume_dp_sync_release(self) -> List[int]:
        """Lanes whose host this engine's latest pass released from the DP sync."""
        released, self._dp_sync_released_lane_ids = self._dp_sync_released_lane_ids, []
        return released

    def _dp_group(self) -> List["VLLMv1EngineReplicaScheduler"]:
        return [
            self._cluster_scheduler.get_replica_scheduler(self._replica_id, lane_id)
            for lane_id in range(self._replica_config.attn_dp)
        ]

    def _start_iterations(self, batches: List[Batch], time: float) -> List[Batch]:
        """Give each batch the engine's idle time before the iteration that forms it.

        Each batch is one loop iteration of vLLM's engine, so a second batch of
        the same call follows the first with no idle time. An engine's first
        batch follows no iteration.
        """
        for batch in batches:
            batch.engine_idle_time = (
                math.inf if self._last_iteration_end is None else time - self._last_iteration_end
            )
            self._last_iteration_end = time
        return batches

    def _clear_followup_polls(self) -> None:
        # No iteration runs while the engine waits; the output or the dummy
        # forward that ends the wait starts the next one.
        self._monolithic_pp_terminal_release_followup_poll_pending = False
        self._monolithic_pp_mtp_output_wait_followup_poll_pending = False

    @property
    def has_unfinished_requests(self) -> bool:
        """vLLM's `Scheduler.has_unfinished_requests`: a waiting or running request."""
        return bool(self._running_requests) or any(self._waiting_queues())

    @property
    def engine_loop_idle(self) -> bool:
        """Whether the engine waits for work: nothing in flight and no DP sync to finish."""
        return (
            not self._in_flight_batch_ids
            and not self._dummy_forwards_in_flight
            and not self._waits_at_dp_sync
        )

    def _issue_dummy_pass(self) -> List[Batch]:
        """Start one dummy forward on every pipeline stage at once.

        `execute_dummy_batch` is one RPC to every worker of the engine. Each
        stage runs its part once its earlier work ends, with no PP transfer
        between parts, and the engine waits for every part.
        """
        dummy_forwards = []
        for stage_id in range(self._num_stages):
            dummy_forward = DummyForwardBatch(
                self._replica_id, stage_id, self._batch_creation_counter
            )
            self._assign_lane_identity(dummy_forward, self._batch_creation_counter)
            dummy_forwards.append(dummy_forward)
        self._batch_creation_counter += 1
        self._dummy_forwards_in_flight = self._num_stages
        self._iteration_open = True
        self._clear_followup_polls()
        return dummy_forwards

    def on_dummy_forward_end(self, time: float) -> bool:
        """Record one ended dummy forward; return whether the engine iterates again."""
        self._dummy_forwards_in_flight -= 1
        if self._dummy_forwards_in_flight:
            return False
        self._last_iteration_end = time
        return True

    def hold_batch_output(self, apply_output: Callable[[float], List]) -> bool:
        """Keep the output of a batch that just ended while the host waits at the DP sync.

        vLLM applies a batch's output (`Scheduler.update_from_output`) only
        when an iteration pops it from the batch queue. Its effects on
        requests, KV blocks and metrics wait for that pop, where
        `apply_output(pop_time)` applies them and returns its events. Return
        whether the output is held.
        """
        if not self._waits_at_dp_sync:
            return False
        self._unpopped_outputs.append(apply_output)
        return True

    def consume_popped_output_events(self) -> List:
        """Events of the outputs this engine's latest pass popped."""
        events, self._popped_output_events = self._popped_output_events, []
        return events

    def _pop_batch_output(self, batch: Batch) -> None:
        """Take a batch's output from the engine: its queue slot and its requests are free."""
        self._num_running_batches -= 1
        if self._has_engine_batch_queue:
            self._in_flight_batch_ids.remove(batch.id)
            if batch.id == self._blocking_batch_id:
                self._blocking_batch_id = None
        self._release_batch_requests_active(batch)

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
        self._pop_batch_output(batch)
        if self._has_engine_batch_queue:
            # The iteration that pops a batch's output ends with it.
            self._last_iteration_end = batch.completed_at

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
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

    def _update_preemption_followup_poll(self, preempted_requests: List[Request]) -> None:
        # vLLM runs its next step right after a step that schedules nothing,
        # unless it waits for a batch in flight (`EngineCore.run_busy_loop`,
        # `step_with_batch_queue`). Frontier starts a pass on an arrival or a
        # batch end, so with no batch in flight an empty pass has no successor.
        # A preemption in that pass freed blocks for the requests still
        # running, so their next step follows at once. Each such pass removes
        # a request from running, which bounds the chain.
        self._preemption_followup_poll_pending = bool(
            preempted_requests
            and self._running_requests
            and self._num_running_batches == 0
        )

    def consume_preemption_followup_poll(self) -> bool:
        pending = self._preemption_followup_poll_pending
        self._preemption_followup_poll_pending = False
        return pending

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
        reserved_tokens_remaining = (
            self._final_prefill_reserved_tokens if waiting_final_prefill_count > 0 else 0
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

            # Keep draft tokens within max_model_len: vLLM computes at most
            # max_model_len - 1 tokens so the token a step samples fits too
            # (scheduler.py, running phase). The DECODE frontier is one token
            # ahead of vLLM's num_computed_tokens: after the KV transfer vLLM's
            # decode side leaves the last prompt token to compute
            # (get_num_new_matched_tokens, or _update_waiting_for_remote_kv
            # for async connectors), so there the frontier already leaves that
            # token of room. The request itself fits (checked on arrival).
            scheduler_num_computed_tokens = self._get_scheduler_num_computed_tokens(
                request
            )
            max_allowed = self._max_model_len - scheduler_num_computed_tokens
            if self._cluster_type != ClusterType.DECODE:
                max_allowed -= 1
            num_new_tokens = min(num_new_tokens, max_allowed)
            num_new_tokens = self._apply_long_prefill_token_threshold(
                request, num_new_tokens
            )

            # Apply token budget limit; a hidden-round prefill leaves the
            # reserved tokens to the waiting final-round requests.
            effective_token_budget = token_budget
            if is_hidden_prefill_running_request:
                effective_token_budget = max(token_budget - reserved_tokens_remaining, 0)
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
                    reserved_tokens_remaining = max(
                        reserved_tokens_remaining - num_new_tokens, 0
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
            return sorted(combined, key=priority_policy_key)
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
            # Check max concurrent requests limit
            if len(self._running_requests) >= self._max_num_running_reqs:
                break

            request = waiting_queue[0]
            if request.stops_on_preempted_step:
                # vLLM would admit it again only to free it when the sample of
                # the step it was preempted from arrives.
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue
            if request.id in self._get_monolithic_pp_pending_terminal_release_iters():
                # Its previous thinking round still waits for the terminal
                # release that resets its scheduler frontier. Only SGLang
                # reaches this: the vllm_v1 pass admits nothing while any
                # release is pending.
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue
            if self._should_defer_monolithic_pp_waiting_admission(request):
                logger.debug(
                    "[VLLMv1Engine][MONOLITHIC] Phase 2: delaying req=%s "
                    "until a PP output-visible scheduler boundary",
                    request.id,
                )
                break

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
            else:
                num_new_tokens = self._get_request_next_num_tokens(request)

            num_new_tokens = self._apply_long_prefill_token_threshold(
                request, num_new_tokens
            )

            # When chunked prefill is disabled, waiting prefills that exceed token
            # budget are skipped for this iteration.
            if (
                not self._enable_chunked_prefill
                and not request.is_decoding
                and num_new_tokens > token_budget
            ):
                waiting_queue.popleft()
                skipped_waiting_requests.append(request)
                continue

            # Apply token budget limit after chunked-prefill guard
            num_new_tokens = min(num_new_tokens, token_budget)

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

            # A victim admitted before the step it was preempted from ends
            # keeps that step's sample pending. As in vLLM, this admission is
            # sized without it, and the step's end appends it at the time its
            # output arrives, before this admission's step can end.
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

        # vLLM (ea95f571) prepends the skipped requests to the waiting queue in
        # the order they were skipped (scheduler.py:882-883, request_queue.py:102-105).
        if skipped_waiting_requests:
            if self._scheduling_policy == "priority":
                merged_requests = list(waiting_queue) + list(skipped_waiting_requests)
                waiting_queue = deque(
                    sorted(merged_requests, key=priority_policy_key)
                )
            else:
                waiting_queue.extendleft(reversed(skipped_waiting_requests))

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
        # DPEngineCoreProc schedules at most one batch per busy-loop iteration.
        if self._num_running_batches >= self._iteration_running_limit:
            return None
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
            self._update_preemption_followup_poll(preempted_requests)
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

    def _check_request_fits_max_model_len(self, request: Request) -> None:
        """Reject a request that does not fit max_model_len, as vLLM's OpenAI server does.

        Each thinking round reaches vLLM as a new request, so every round is
        checked when it arrives.
        """
        num_prompt_tokens = request.num_prefill_tokens
        num_output_tokens = request.num_decode_tokens
        if (
            num_prompt_tokens >= self._max_model_len
            or num_prompt_tokens + num_output_tokens > self._max_model_len
        ):
            raise ValueError(
                f"Request {request.id} (round index "
                f"{request.current_thinking_round_index}) has {num_prompt_tokens} "
                f"prompt and {num_output_tokens} output tokens, which do not fit "
                f"max_model_len={self._max_model_len}: vLLM requires "
                "prompt < max_model_len and prompt + output <= max_model_len. "
                "Raise max_model_len (default: the model's max_position_embeddings) "
                "or shorten the request."
            )

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

        self._check_request_fits_max_model_len(request)
        self._check_request_fits_kv_pool(request)
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
