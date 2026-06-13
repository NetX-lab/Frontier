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
import json
import logging
from math import ceil
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frontier.config import global_vars
from frontier.entities.batch import (
    Batch,
    DecodeCudaGraphMetadata,
    Request,
    SpecDecodeBatchMetadata,
)
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.spec_decode import (
    compute_iteration_outcome,
    get_planned_draft_tokens,
    is_spec_decode_enabled,
    method_uses_lookahead_slots,
)
from frontier.types import ClusterType


_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH = os.environ.get(
    "FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH", ""
)
_frontier_vllm_v1_sched_decision_logger: Optional[logging.Logger] = None

if _FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH:
    _frontier_vllm_v1_sched_decision_logger = logging.getLogger(
        "frontier.vllm_v1_sched_decision"
    )
    _frontier_vllm_v1_sched_decision_logger.setLevel(logging.INFO)
    _frontier_vllm_v1_sched_decision_logger.propagate = False

    _decision_log_dir = os.path.dirname(_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH)
    if _decision_log_dir:
        os.makedirs(_decision_log_dir, exist_ok=True)

    _decision_handler = logging.FileHandler(_FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH)
    _decision_handler.setFormatter(logging.Formatter("%(message)s"))
    _frontier_vllm_v1_sched_decision_logger.addHandler(_decision_handler)


def _log_frontier_vllm_v1_schedule_decision(event: Dict[str, Any]) -> None:
    if _frontier_vllm_v1_sched_decision_logger is None:
        return
    _frontier_vllm_v1_sched_decision_logger.info(json.dumps(event))


class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):
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
        batch = super()._create_batch(requests, num_tokens)
        metadata = self._build_decode_cuda_graph_metadata(batch)
        if metadata is not None:
            batch.decode_cuda_graph_metadata = metadata
        spec_metadata = self._build_spec_decode_batch_metadata(batch)
        if spec_metadata is not None:
            batch.spec_decode_metadata = spec_metadata
        self._record_monolithic_pp_mtp_near_full_prefill_slices(batch)
        self._mark_batch_requests_active(batch)
        return batch

    def _refresh_target_embedded_mtp_prefill_boundary_state(
        self, batch: Batch, request: Request
    ) -> None:
        metadata = batch.spec_decode_metadata
        if metadata is not None:
            for _, batch_request in enumerate(batch.requests):
                if batch_request is request:
                    # The metadata row is authoritative for this batch. A
                    # positive verify width was already recorded when metadata
                    # was built; a zero verify width means the request did not
                    # participate in this spec-decode iteration.
                    return
        if self._cluster_type != ClusterType.MONOLITHIC:
            return
        if not getattr(request, "spec_decode_enabled", False):
            return
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return
        if not getattr(request, "is_prefill_complete", False):
            return
        spec_decode_config = getattr(self, "_spec_decode_config", None)
        if spec_decode_config is None:
            raise ValueError("Speculative decoding config is not initialized")
        if int(getattr(request, "spec_total_iterations", 0)) == 0:
            planned_drafts = get_planned_draft_tokens(
                spec_decode_config,
                request.remaining_decode_tokens,
                iteration_index=0,
                request_id=str(request.id),
            )
            outcome = compute_iteration_outcome(
                spec_decode_config,
                request.remaining_decode_tokens,
                planned_draft_tokens=planned_drafts,
                iteration_index=0,
                request_id=str(request.id),
            )
            if planned_drafts == 0 and outcome.committed_tokens == 1:
                # Some vLLM target-embedded MTP requests emit a real prefill
                # commit row: one sampled token, no scheduled drafts. Frontier
                # already advanced that token at the prefill boundary, so only
                # the trace cursor and spec stats need to catch up here.
                request.record_spec_decode_iteration(
                    verify_tokens=outcome.verify_tokens,
                    accepted_drafts=outcome.accepted_draft_tokens,
                    rejected_drafts=outcome.rejected_draft_tokens,
                    committed_tokens=outcome.committed_tokens,
                )
            else:
                request.set_spec_next_planned_draft_tokens(planned_drafts)
                return
        if int(getattr(request, "spec_total_iterations", 0)) != 1:
            return
        request.set_spec_next_planned_draft_tokens(
            get_planned_draft_tokens(
                spec_decode_config,
                request.remaining_decode_tokens,
                iteration_index=request.spec_total_iterations,
                request_id=str(request.id),
            )
        )

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
        active_counts = self._get_active_batch_request_counts()
        for request in batch.requests:
            current_count = active_counts.get(request.id, 0)
            if current_count <= 1:
                active_counts.pop(request.id, None)
            else:
                active_counts[request.id] = current_count - 1

    def _is_request_active_in_batch(self, request: Request) -> bool:
        return self._get_active_batch_request_counts().get(request.id, 0) > 0

    def _get_monolithic_pp_waiting_admission_delay_iters(self) -> Dict[int, int]:
        delay_iters = getattr(
            self,
            "_monolithic_pp_waiting_admission_delay_iters",
            None,
        )
        if delay_iters is None:
            delay_iters = {}
            self._monolithic_pp_waiting_admission_delay_iters = delay_iters
        return delay_iters

    def _is_target_embedded_mtp_request(self, request: Request) -> bool:
        return (
            bool(getattr(request, "spec_decode_enabled", False))
            and bool(getattr(request, "spec_method_is_target_embedded_mtp", False))
        )

    def _has_future_planned_draft_tokens(self, request: Request) -> bool:
        spec_config = getattr(
            getattr(self, "_replica_config", None),
            "speculative_decoding_config",
            None,
        )
        per_request_trace = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        if per_request_trace is None:
            return False
        request_id = str(request.id)
        if request_id not in per_request_trace:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        next_iteration = int(getattr(request, "spec_total_iterations", 0)) + 1
        return any(int(tokens) > 0 for tokens in per_request_trace[request_id][next_iteration:])

    def _get_target_embedded_mtp_terminal_overshoot_rows(
        self,
        request: Request,
        *,
        start_iteration_index: int,
    ) -> List[Tuple[int, int, int, int, int]]:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return []
        if self._num_stages <= 1:
            return []
        if not getattr(request, "spec_decode_enabled", False):
            return []
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return []

        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            raise ValueError("Speculative decoding config is not initialized")
        per_request_planned_trace = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        per_request_committed_trace = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        if per_request_planned_trace is None and per_request_committed_trace is None:
            return []
        if per_request_planned_trace is None or per_request_committed_trace is None:
            raise ValueError(
                "terminal target-embedded MTP overshoot modeling requires both "
                "per-request scheduled draft and committed token traces"
            )

        request_id = str(request.id)
        if request_id not in per_request_planned_trace:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        if request_id not in per_request_committed_trace:
            raise ValueError(
                "per-request acceptance trace missing request_id="
                f"{request_id!r}"
            )

        planned_trace = per_request_planned_trace[request_id]
        committed_trace = per_request_committed_trace[request_id]
        if len(planned_trace) != len(committed_trace):
            raise ValueError(
                "per-request MTP trace length mismatch: "
                f"request_id={request_id!r}, planned_len={len(planned_trace)}, "
                f"committed_len={len(committed_trace)}"
            )

        start_idx = int(start_iteration_index)
        if start_idx < 0:
            raise ValueError(
                f"start_iteration_index must be >= 0, got={start_idx}"
            )
        if start_idx >= len(planned_trace):
            return []

        terminal_rows: List[Tuple[int, int, int, int, int]] = []
        for idx in range(start_idx, len(planned_trace)):
            planned_drafts = int(planned_trace[idx])
            raw_committed = int(committed_trace[idx])
            if planned_drafts < 0:
                raise ValueError(
                    "terminal scheduled draft tokens must be >= 0, "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"got={planned_drafts}"
                )
            if raw_committed < 0:
                raise ValueError(
                    "terminal committed tokens must be >= 0, "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"got={raw_committed}"
                )
            trace_verify_tokens = 1 + planned_drafts
            if raw_committed > trace_verify_tokens:
                raise ValueError(
                    "terminal committed tokens cannot exceed verify window: "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"committed={raw_committed}, "
                    f"verify_tokens={trace_verify_tokens}"
                )
            if planned_drafts == 0 and raw_committed == 0:
                continue

            # Once the logical response is complete, vLLM's online scheduler no
            # longer replays the full forced-acceptance scheduled-draft window
            # for request latency. The diagnostic acceptance trace can still
            # contain scheduled_draft_tokens=32 for the next trace row, while
            # the clean scheduler batch log exposes only a one-token cleanup row
            # for that completed request. Model that terminal cleanup as one
            # target token and keep the raw committed count only for audit
            # provenance. Replaying the full trace window here over-extends
            # short decode-tail request latency and violates the clean online
            # metric scope.
            terminal_cleanup_verify_tokens = 1
            terminal_rows.append(
                (
                    0,
                    terminal_cleanup_verify_tokens,
                    0,
                    0,
                    raw_committed,
                )
            )
        return terminal_rows

    def _should_delay_monolithic_pp_waiting_admission_on_add(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not self._is_target_embedded_mtp_request(request):
            return False
        planned_drafts = int(getattr(request, "spec_next_planned_draft_tokens", 0))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if planned_drafts <= 0:
            # A zero first scheduled-draft step still has PP lookahead admission
            # visibility cost when a long decode will enter later MTP draft steps.
            if num_speculative_tokens > 2:
                return False
            if int(getattr(request, "num_decode_tokens", 0)) < 128:
                return False
            if not self._has_future_planned_draft_tokens(request):
                return False
        else:
            block_size = int(getattr(self._config, "block_size", 16))
            if num_speculative_tokens >= block_size:
                # Full-block-or-wider target-embedded MTP request admission is
                # already protected by output-visible guards after it starts
                # running. Adding a pre-admission PP boundary here makes late
                # online arrivals miss the vLLM-visible scheduler slot and
                # under-batches wide verify traces relative to vLLM.
                return False
            if (
                num_speculative_tokens > 2
                and int(getattr(request, "num_prefill_tokens", 0))
                >= int(self._max_num_scheduled_tokens)
                + max(1, int(self._max_num_scheduled_tokens) // 2)
            ):
                # Long multi-chunk prefills have enough remaining prefill work
                # to expose the next PP boundary through the prefill chunks
                # themselves. Adding a separate half-block MTP admission delay
                # over-queues these arrivals and inflates p90 TTFT; keep the
                # delay for shorter prefills where r106 showed global
                # half-block skipping over-corrects TPOT tails.
                return False
        return (
            self._num_running_batches > 0
            or bool(self._running_requests)
            or bool(self._get_active_batch_request_counts())
        )

    def _add_monolithic_pp_waiting_admission_delay(
        self, request_id: int, *, wait_iters: Optional[int] = None
    ) -> None:
        resolved_wait_iters = int(
            wait_iters if wait_iters is not None else max(1, self._num_stages - 1)
        )
        if resolved_wait_iters <= 0:
            raise ValueError(
                f"wait_iters must be positive, got={resolved_wait_iters}"
            )
        delay_iters = self._get_monolithic_pp_waiting_admission_delay_iters()
        delay_iters[request_id] = max(
            int(delay_iters.get(request_id, 0)),
            resolved_wait_iters,
        )

    def _should_defer_monolithic_pp_waiting_admission(
        self, request: Request
    ) -> bool:
        delay_iters = self._get_monolithic_pp_waiting_admission_delay_iters()
        remaining = int(delay_iters.get(request.id, 0))
        if remaining <= 0:
            delay_iters.pop(request.id, None)
            return False
        if self._cluster_type != ClusterType.MONOLITHIC or self._num_stages <= 1:
            delay_iters.pop(request.id, None)
            return False
        if not self._is_target_embedded_mtp_request(request):
            delay_iters.pop(request.id, None)
            return False
        if (
            not self._running_requests
            and self._num_running_batches <= 0
            and not self._get_active_batch_request_counts()
        ):
            # No active PP work remains to provide a future output-visible
            # scheduler boundary; fail open to avoid deadlocking the queue.
            delay_iters.pop(request.id, None)
            return False
        remaining -= 1
        if remaining > 0:
            delay_iters[request.id] = remaining
        else:
            delay_iters.pop(request.id, None)
        return True

    def _get_monolithic_pp_mtp_near_full_prefill_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_near_full_prefill_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_near_full_prefill_request_ids = request_ids
        return request_ids

    def _get_monolithic_pp_mtp_single_output_wait_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_single_output_wait_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_single_output_wait_request_ids = request_ids
        return request_ids

    def _get_target_embedded_mtp_request_acceptance_ratio(
        self, request: Request
    ) -> Optional[float]:
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        if spec_config is None:
            return None
        committed_trace_map = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        scheduled_trace_map = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        if committed_trace_map is None and scheduled_trace_map is None:
            return None
        if committed_trace_map is None or scheduled_trace_map is None:
            raise ValueError(
                "MTP request acceptance audit requires both per-request "
                "committed and scheduled-draft traces"
            )
        request_id = str(request.id)
        if request_id not in committed_trace_map:
            raise ValueError(
                "per-request acceptance trace missing request_id="
                f"{request_id!r}"
            )
        if request_id not in scheduled_trace_map:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        committed_trace = committed_trace_map[request_id]
        scheduled_trace = scheduled_trace_map[request_id]
        if len(committed_trace) != len(scheduled_trace):
            raise ValueError(
                "MTP request acceptance audit trace length mismatch: "
                f"request_id={request_id!r}, "
                f"committed_len={len(committed_trace)}, "
                f"scheduled_len={len(scheduled_trace)}"
            )
        accepted_drafts = 0
        scheduled_drafts = 0
        for committed_tokens, planned_drafts in zip(
            committed_trace,
            scheduled_trace,
        ):
            planned = int(planned_drafts)
            if planned <= 0:
                continue
            committed = int(committed_tokens)
            accepted_drafts += min(max(committed - 1, 0), planned)
            scheduled_drafts += planned
        if scheduled_drafts <= 0:
            return None
        return accepted_drafts / scheduled_drafts

    def _get_monolithic_pp_mtp_output_wait_prefill_threshold(self) -> int:
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        block_size = int(getattr(self._config, "block_size", 16))
        headroom_tokens = 4 * block_size
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens >= block_size:
            # Wide target-embedded MTP carries a larger PP lookahead payload than
            # the narrow-window cases that established the original four-block
            # threshold. Reserve window-proportional headroom so long chunked
            # prefill slices enter the same output-visible continuation lane
            # instead of being treated as ordinary prefill chunks.
            headroom_tokens = max(headroom_tokens, 8 * num_speculative_tokens)
        return max(1, max_scheduled_tokens - headroom_tokens)

    def _get_monolithic_pp_mtp_output_wait_iters(self) -> int:
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        if spec_config is None:
            return 2

        block_size = int(getattr(self._config, "block_size", 16))
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < block_size:
            return 2

        committed_trace_map = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        scheduled_trace_map = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        accepted_drafts = 0
        scheduled_drafts = 0
        if committed_trace_map is not None or scheduled_trace_map is not None:
            if committed_trace_map is None or scheduled_trace_map is None:
                raise ValueError(
                    "MTP output-wait acceptance audit requires both "
                    "per-request committed and scheduled-draft traces"
                )
            if set(committed_trace_map.keys()) != set(scheduled_trace_map.keys()):
                raise ValueError(
                    "MTP output-wait acceptance audit trace keys mismatch"
                )
            for request_id, committed_trace in committed_trace_map.items():
                scheduled_trace = scheduled_trace_map[request_id]
                if len(committed_trace) != len(scheduled_trace):
                    raise ValueError(
                        "MTP output-wait acceptance audit trace length mismatch: "
                        f"request_id={request_id!r}, "
                        f"committed_len={len(committed_trace)}, "
                        f"scheduled_len={len(scheduled_trace)}"
                    )
                for committed_tokens, planned_drafts in zip(
                    committed_trace,
                    scheduled_trace,
                ):
                    planned = int(planned_drafts)
                    if planned <= 0:
                        continue
                    committed = int(committed_tokens)
                    accepted_drafts += min(max(committed - 1, 0), planned)
                    scheduled_drafts += planned
        else:
            committed_trace = getattr(
                spec_config,
                "_committed_tokens_trace",
                None,
            )
            scheduled_trace = getattr(
                spec_config,
                "_scheduled_draft_tokens_trace",
                None,
            )
            if committed_trace is None or scheduled_trace is None:
                return 2
            if len(committed_trace) != len(scheduled_trace):
                raise ValueError(
                    "MTP output-wait acceptance audit global trace length mismatch: "
                    f"committed_len={len(committed_trace)}, "
                    f"scheduled_len={len(scheduled_trace)}"
                )
            for committed_tokens, planned_drafts in zip(
                committed_trace,
                scheduled_trace,
            ):
                planned = int(planned_drafts)
                if planned <= 0:
                    continue
                committed = int(committed_tokens)
                accepted_drafts += min(max(committed - 1, 0), planned)
                scheduled_drafts += planned

        if scheduled_drafts <= 0:
            return 2
        acceptance_ratio = accepted_drafts / scheduled_drafts
        if acceptance_ratio >= 0.5:
            if self._has_monolithic_pp_visible_waiting_requests():
                # High-acceptance wide MTP should still preserve the extra
                # output-visible boundary while fresh prefill admissions are
                # visible. Otherwise decode continuation can consume the
                # online token budget before late-arriving prefills enter,
                # inflating TTFT tails. Once no waiting prefill/resume request
                # is visible, shorten the wait to protect short decode tails.
                return 2
            # High-acceptance wide target-embedded MTP has fewer decode
            # scheduler turns per request. A single output-visible turn is
            # enough to expose PP continuation without repeatedly holding
            # short tail requests behind terminal trace rows.
            return 1
        return 2

    def _get_monolithic_pp_mtp_output_wait_iters_for_request(
        self, request: Request
    ) -> int:
        wait_iters = self._get_monolithic_pp_mtp_output_wait_iters()
        if request.id in self._get_monolithic_pp_mtp_single_output_wait_request_ids():
            return min(wait_iters, 1)
        if self._should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
            request
        ):
            counts = self._get_monolithic_pp_mtp_fractional_output_wait_counts()
            previous_count = int(counts.get(request.id, 0))
            counts[request.id] = previous_count + 1
            if previous_count == 0:
                return 0
            if previous_count % 2 == 1:
                return wait_iters
            return min(wait_iters, 1)
        if self._should_extend_monolithic_pp_mtp_long_decode_output_wait(request):
            return wait_iters + 1
        return wait_iters

    def _is_monolithic_pp_mtp_half_block_low_acceptance_request(
        self,
        request: Request,
        *,
        acceptance_ratio_limit: float,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < max(1, block_size // 2):
            return False
        if num_speculative_tokens >= block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False
        return acceptance_ratio < acceptance_ratio_limit

    def _is_monolithic_pp_mtp_mid_prefill_request(
        self,
        request: Request,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        num_prefill_tokens = int(getattr(request, "num_prefill_tokens", 0))
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        if num_prefill_tokens < max(1, max_scheduled_tokens - 4 * block_size):
            return False
        if num_prefill_tokens > (
            max_scheduled_tokens + max(1, max_scheduled_tokens // 2)
        ):
            return False
        return True

    def _should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
        self,
        request: Request,
    ) -> bool:
        if not self._is_target_embedded_mtp_request(request):
            return False
        if not self._is_monolithic_pp_mtp_mid_prefill_request(request):
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < max(1, block_size // 2):
            return False
        if num_speculative_tokens >= block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        num_decode_tokens = int(getattr(request, "num_decode_tokens", 0))
        if num_decode_tokens < 16 * block_size:
            return False
        if num_decode_tokens < 24 * block_size:
            if acceptance_ratio >= 0.5:
                return False
        else:
            if acceptance_ratio >= 0.55:
                return False

        # The v8/a0.3 request-level RCA shows that medium-length decode tails
        # and long decode tails need about one and a half PP output-visible
        # wait turns after the first visible decode result. Skip the first
        # wait so TTFT remains a prefill/first-token metric, then alternate
        # the regular low-acceptance wait with a single-turn wait. This keeps
        # the correction as a scheduler visibility family rather than an
        # op-runtime calibration scale.
        return True

    def _should_extend_monolithic_pp_mtp_long_decode_output_wait(
        self, request: Request
    ) -> bool:
        if not self._is_monolithic_pp_mtp_half_block_low_acceptance_request(
            request,
            acceptance_ratio_limit=0.25,
        ):
            return False
        if not self._is_monolithic_pp_mtp_mid_prefill_request(request):
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        num_decode_tokens = int(getattr(request, "num_decode_tokens", 0))
        if num_decode_tokens < 24 * block_size:
            return False

        # Low-acceptance half-block MTP keeps many long decode continuations
        # alive after a near-full prefill admission. vLLM exposes an additional
        # PP-visible output boundary for the very long decode tail; model that
        # boundary as one extra scheduler wait turn instead of
        # hiding the residual in compute calibration scale.
        return True

    def _record_monolithic_pp_mtp_near_full_prefill_slices(self, batch: Batch) -> None:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return
        if self._num_stages <= 1:
            return
        near_full_prefill_threshold = (
            self._get_monolithic_pp_mtp_output_wait_prefill_threshold()
        )
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            if getattr(request, "is_prefill_complete", False):
                continue
            if not getattr(request, "spec_decode_enabled", False):
                continue
            if not getattr(request, "spec_method_is_target_embedded_mtp", False):
                continue
            if int(num_tokens) < near_full_prefill_threshold:
                if not self._should_record_monolithic_pp_mtp_subthreshold_single_wait_prefill(
                    request,
                    num_tokens=int(num_tokens),
                    near_full_prefill_threshold=near_full_prefill_threshold,
                ):
                    if not self._should_record_monolithic_pp_mtp_subthreshold_long_decode_prefill(
                        request,
                        num_tokens=int(num_tokens),
                        near_full_prefill_threshold=near_full_prefill_threshold,
                    ):
                        continue
                else:
                    self._get_monolithic_pp_mtp_single_output_wait_request_ids().add(
                        request.id
                    )
            self._get_monolithic_pp_mtp_near_full_prefill_request_ids().add(
                request.id
            )

    def _should_record_monolithic_pp_mtp_subthreshold_single_wait_prefill(
        self,
        request: Request,
        *,
        num_tokens: int,
        near_full_prefill_threshold: int,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens >= block_size:
            return False

        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        min_subthreshold_tokens = max(
            1,
            max_scheduled_tokens - 8 * block_size,
        )
        if int(num_tokens) < min_subthreshold_tokens:
            return False
        if int(num_tokens) >= int(near_full_prefill_threshold):
            return False
        if int(getattr(request, "num_prefill_tokens", 0)) < (
            max_scheduled_tokens + max(1, max_scheduled_tokens // 2)
        ):
            return False
        if int(getattr(request, "num_decode_tokens", 0)) > 4 * block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False
        if acceptance_ratio >= 0.5:
            return False

        # Low-acceptance narrow MTP preserves more decode turns than the
        # high-acceptance case, so a two-turn output wait over-delays the
        # terminal request tail. A single PP-visible wait models the missing
        # output boundary without absorbing the residual into compute scale.
        return True

    def _should_record_monolithic_pp_mtp_subthreshold_long_decode_prefill(
        self,
        request: Request,
        *,
        num_tokens: int,
        near_full_prefill_threshold: int,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        min_subthreshold_tokens = max(
            1,
            max_scheduled_tokens - 8 * block_size,
        )
        if int(num_tokens) < min_subthreshold_tokens:
            return False
        if int(num_tokens) >= int(near_full_prefill_threshold):
            return False
        if not (
            self._should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
                request
            )
            or self._should_extend_monolithic_pp_mtp_long_decode_output_wait(
                request
            )
        ):
            return False

        # These subthreshold prefill slices are close enough to the online
        # max-token boundary to expose the same PP output-visible behavior as
        # near-full chunks, but only for the low-acceptance half-block
        # medium/long decode slices identified by the request-level TPOT RCA.
        return True

    def _is_prefix_caching_enabled(self) -> bool:
        return getattr(self, "_kv_cache_manager", None) is not None

    def _sync_prefix_cache_allocation_state(
        self, request: Optional[Request] = None
    ) -> None:
        if not self._is_prefix_caching_enabled():
            return
        assert self._kv_cache_manager is not None
        self._num_allocated_blocks = int(self._kv_cache_manager.num_used_blocks)
        if request is not None:
            num_blocks = int(self._kv_cache_manager.get_num_blocks_for_request(request))
            if num_blocks > 0:
                self._allocation_map[request.id] = num_blocks
            else:
                self._allocation_map.pop(request.id, None)

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
            return
        self.free(request.id)

    def _free_request_resources_by_id(self, request_id: int) -> None:
        request = self._find_request_by_id(request_id)
        if request is not None:
            self._free_request_resources(request)
            return
        self.free(request_id)

    def _prepare_prefix_cache_admission(
        self, request: Request
    ) -> Tuple[List[object], int, int]:
        if not self._is_prefix_caching_enabled():
            return [], 0, self._get_request_next_num_tokens(request)
        if request.block_hash_ids is None:
            raise ValueError(
                "block_hash_ids are required when enable_prefix_caching=True"
            )
        assert self._kv_cache_manager is not None
        computed_blocks, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(
            request
        )
        num_new_tokens = int(request.num_prefill_tokens) - int(num_computed_tokens)
        if num_new_tokens == 0 and computed_blocks:
            num_computed_tokens -= int(self._config.block_size)
            num_new_tokens = int(self._config.block_size)
            computed_blocks = list(computed_blocks[:-1])
            self._kv_cache_manager.prefix_cache_stats.hits -= 1
        return computed_blocks, int(num_computed_tokens), int(num_new_tokens)

    def _build_decode_cuda_graph_metadata(
        self, batch: Batch
    ) -> Optional[DecodeCudaGraphMetadata]:
        if self._cluster_type not in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return None
        if (
            getattr(self, "_spec_decode_enabled", False)
            and not global_vars.get_allow_spec_decode_cuda_graph_diagnostic()
        ):
            # Phase 2+ baseline: speculative decoding always runs in eager mode.
            # We intentionally disable decode CUDA graph modeling for all
            # speculative batches to reduce alignment complexity; future work
            # can reintroduce method-specific CUDA graph semantics.
            return None

        config_mode = global_vars.get_decode_cuda_graph_mode()
        if config_mode == "none":
            return None

        capture_hit, capture_size = self._resolve_decode_cuda_graph_capture_size(
            batch.total_num_tokens
        )
        decode_query_lens = [
            int(num_tokens)
            for request, num_tokens in zip(batch.requests, batch.num_tokens)
            if request.is_prefill_complete
        ]
        original_decode_batch_size = len(decode_query_lens)

        # Align with vLLM's uniform_decode_query_len semantics.
        # When speculative decoding is enabled, FULL decode cudagraphs are only
        # valid for uniform batches whose query_len matches
        # 1 + num_speculative_tokens. Non-uniform speculative verify batches
        # must dispatch to mixed/piecewise graphs or fall back to eager.
        uniform_decode_query_len = 1
        if getattr(self, "_spec_decode_enabled", False):
            spec_decode_config = getattr(self, "_spec_decode_config", None)
            if spec_decode_config is None:
                raise ValueError("Speculative decoding config is not initialized")
            uniform_decode_query_len += int(spec_decode_config.num_speculative_tokens)

        is_uniform_decode_batch = (
            bool(decode_query_lens)
            and len(decode_query_lens) == len(batch.requests)
            and all(
                query_len == uniform_decode_query_len
                for query_len in decode_query_lens
            )
        )
        is_mixed_batch = not is_uniform_decode_batch

        runtime_mode = "NONE"
        if config_mode == "full_decode_only":
            if is_uniform_decode_batch and capture_hit:
                runtime_mode = "FULL"
        elif config_mode == "piecewise" and capture_hit:
            runtime_mode = "PIECEWISE"

        if runtime_mode == "NONE":
            capture_hit = False
            capture_size = batch.total_num_tokens

        padded_decode_batch_size = (
            capture_size if capture_hit else original_decode_batch_size
        )
        padded_total_tokens = capture_size if capture_hit else batch.total_num_tokens

        return DecodeCudaGraphMetadata(
            config_mode=config_mode,
            runtime_mode=runtime_mode,
            capture_hit=capture_hit,
            is_mixed_batch=is_mixed_batch,
            original_total_tokens=batch.total_num_tokens,
            padded_total_tokens=padded_total_tokens,
            original_decode_batch_size=original_decode_batch_size,
            padded_decode_batch_size=padded_decode_batch_size,
        )

    def _resolve_decode_cuda_graph_capture_size(self, total_tokens: int) -> Tuple[bool, int]:
        cudagraph_capture_sizes = global_vars.get_cudagraph_capture_sizes()
        if cudagraph_capture_sizes is None:
            max_num_seqs = getattr(
                self,
                "_max_num_running_reqs",
                getattr(self, "_max_batch_size", total_tokens),
            )
            max_num_seqs = max(int(max_num_seqs), total_tokens)
            cudagraph_capture_sizes = [1, 2, 4] + [
                8 * i for i in range(1, max_num_seqs // 8 + 1)
            ]

        for capture_size in sorted(cudagraph_capture_sizes):
            if total_tokens <= capture_size:
                return True, int(capture_size)
        return False, int(total_tokens)

    def _build_spec_decode_batch_metadata(
        self, batch: Batch
    ) -> Optional[SpecDecodeBatchMetadata]:
        if not getattr(self, "_spec_decode_enabled", False):
            return None
        if self._cluster_type not in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return None
        if batch.num_decode_tokens <= 0:
            return None
        spec_decode_config = getattr(self, "_spec_decode_config", None)
        if spec_decode_config is None:
            raise ValueError("Speculative decoding config is not initialized")

        planned_drafts_list: List[int] = []
        verify_tokens_list: List[int] = []
        accepted_drafts_list: List[int] = []
        rejected_drafts_list: List[int] = []
        committed_tokens_list: List[int] = []
        terminal_planned_drafts_list: List[List[int]] = []
        terminal_verify_tokens_list: List[List[int]] = []
        terminal_accepted_drafts_list: List[List[int]] = []
        terminal_rejected_drafts_list: List[List[int]] = []
        terminal_raw_committed_tokens_list: List[List[int]] = []
        per_request_outcomes: Dict[int, Tuple[int, Any, List[Tuple[int, int, int, int, int]]]] = {}

        for request, scheduled_tokens in zip(batch.requests, batch.num_tokens):
            if not getattr(request, "is_prefill_complete", False) or not getattr(
                request, "spec_decode_enabled", False
            ):
                planned_drafts_list.append(0)
                verify_tokens_list.append(0)
                accepted_drafts_list.append(0)
                rejected_drafts_list.append(0)
                committed_tokens_list.append(int(scheduled_tokens))
                terminal_planned_drafts_list.append([])
                terminal_verify_tokens_list.append([])
                terminal_accepted_drafts_list.append([])
                terminal_rejected_drafts_list.append([])
                terminal_raw_committed_tokens_list.append([])
                continue

            request_id = int(request.id)
            scheduled_tokens_int = int(scheduled_tokens)
            if request_id in per_request_outcomes:
                (
                    recorded_scheduled_tokens,
                    recorded_outcome,
                    recorded_terminal_rows,
                ) = per_request_outcomes[request_id]
                if recorded_scheduled_tokens != scheduled_tokens_int:
                    raise ValueError(
                        "Inconsistent scheduled_tokens for duplicated request in the "
                        "same batch: "
                        f"request_id={request_id}, "
                        f"first={recorded_scheduled_tokens}, "
                        f"current={scheduled_tokens_int}"
                    )
                outcome = recorded_outcome
                terminal_rows = recorded_terminal_rows
            else:
                if getattr(request, "spec_method_is_target_embedded_mtp", False):
                    planned_drafts = int(request.spec_next_planned_draft_tokens)
                else:
                    planned_drafts = max(scheduled_tokens_int - 1, 0)
                remaining_decode = request.remaining_decode_tokens
                outcome = compute_iteration_outcome(
                    spec_decode_config,
                    remaining_decode,
                    planned_draft_tokens=planned_drafts,
                    iteration_index=request.spec_total_iterations,
                    request_id=str(request.id),
                )
                request.record_spec_decode_iteration(
                    verify_tokens=outcome.verify_tokens,
                    accepted_drafts=outcome.accepted_draft_tokens,
                    rejected_drafts=outcome.rejected_draft_tokens,
                    committed_tokens=outcome.committed_tokens,
                )

                next_remaining_decode = max(
                    remaining_decode - outcome.committed_tokens,
                    0,
                )
                terminal_rows: List[Tuple[int, int, int, int, int]] = []
                if next_remaining_decode == 0:
                    terminal_rows = (
                        self._get_target_embedded_mtp_terminal_overshoot_rows(
                            request,
                            start_iteration_index=request.spec_total_iterations,
                        )
                    )
                request.set_spec_next_planned_draft_tokens(
                    get_planned_draft_tokens(
                        spec_decode_config,
                        next_remaining_decode,
                        iteration_index=request.spec_total_iterations,
                        request_id=str(request.id),
                    )
                )
                per_request_outcomes[request_id] = (
                    scheduled_tokens_int,
                    outcome,
                    terminal_rows,
                )

            planned_drafts_list.append(outcome.planned_draft_tokens)
            verify_tokens_list.append(outcome.verify_tokens)
            accepted_drafts_list.append(outcome.accepted_draft_tokens)
            rejected_drafts_list.append(outcome.rejected_draft_tokens)
            committed_tokens_list.append(outcome.committed_tokens)
            terminal_planned_drafts_list.append(
                [int(row[0]) for row in terminal_rows]
            )
            terminal_verify_tokens_list.append([int(row[1]) for row in terminal_rows])
            terminal_accepted_drafts_list.append(
                [int(row[2]) for row in terminal_rows]
            )
            terminal_rejected_drafts_list.append(
                [int(row[3]) for row in terminal_rows]
            )
            terminal_raw_committed_tokens_list.append(
                [int(row[4]) for row in terminal_rows]
            )

        metadata = SpecDecodeBatchMetadata(
            method=spec_decode_config.method,
            planned_draft_tokens_per_request=planned_drafts_list,
            verify_tokens_per_request=verify_tokens_list,
            accepted_draft_tokens_per_request=accepted_drafts_list,
            rejected_draft_tokens_per_request=rejected_drafts_list,
            committed_tokens_per_request=committed_tokens_list,
            uses_lookahead_slots=getattr(
                self, "_spec_method_uses_lookahead_slots", False
            ),
            terminal_overshoot_planned_draft_tokens_per_request=(
                terminal_planned_drafts_list
            ),
            terminal_overshoot_verify_tokens_per_request=(
                terminal_verify_tokens_list
            ),
            terminal_overshoot_accepted_draft_tokens_per_request=(
                terminal_accepted_drafts_list
            ),
            terminal_overshoot_rejected_draft_tokens_per_request=(
                terminal_rejected_drafts_list
            ),
            terminal_overshoot_raw_committed_tokens_per_request=(
                terminal_raw_committed_tokens_list
            ),
        )
        metadata.validate(len(batch.requests))
        return metadata

    # ========== Scheduling Policy Selection ==========

    def _get_scheduling_policy(self) -> str:
        """
        Get the scheduling policy to use.

        This method provides a clean interface for policy selection that can be
        easily extended in future work to support command-line parameter control.

        Returns:
            str: The scheduling policy ('fcfs' or 'priority')
        """
        # Use the scheduling policy from configuration
        return self._config.scheduling_policy

    def _get_iteration_phase_aware_waiting_requests(self) -> List[Request]:
        if self._cluster_type not in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
            return []
        return list(self._preempted_requests) + list(self._request_queue)

    def _resolve_iteration_round_class(self) -> Optional[str]:
        if not self._enable_phase_aware_thinking_profile:
            return None

        waiting_requests = self._get_iteration_phase_aware_waiting_requests()
        thinking_requests = [
            request
            for request in waiting_requests
            if getattr(request, "is_thinking_mode_enabled", False)
        ]
        if not thinking_requests:
            return None
        if any(request.is_final_thinking_round for request in thinking_requests):
            return "final"
        return "hidden"

    def _get_iteration_scheduler_profile(self) -> Dict[str, Any]:
        round_class = self._resolve_iteration_round_class()
        profile = {
            "round_class": round_class,
            "max_num_running_reqs": int(self._config.batch_size_cap),
            "max_num_scheduled_tokens": int(self._config.max_tokens_in_batch),
            "enable_chunked_prefill": bool(
                getattr(self._config, "enable_chunked_prefill", False)
            ),
        }
        if round_class is None:
            return profile

        prefix = f"{round_class}_phase_"
        max_tokens_override = getattr(self._config, f"{prefix}max_tokens_in_batch")
        chunked_override = getattr(
            self._config, f"{prefix}enable_chunked_prefill"
        )
        batch_size_override = getattr(self._config, f"{prefix}batch_size_cap")
        if max_tokens_override is not None:
            profile["max_num_scheduled_tokens"] = int(max_tokens_override)
        if chunked_override is not None:
            profile["enable_chunked_prefill"] = bool(chunked_override)
        if batch_size_override is not None:
            profile["max_num_running_reqs"] = int(batch_size_override)
        return profile

    def _refresh_iteration_scheduler_profile(self) -> None:
        profile = self._get_iteration_scheduler_profile()
        self._active_iteration_round_class = profile["round_class"]
        self._max_num_running_reqs = int(profile["max_num_running_reqs"])
        self._max_num_scheduled_tokens = int(profile["max_num_scheduled_tokens"])
        self._enable_chunked_prefill = bool(profile["enable_chunked_prefill"])

    def _maybe_promote_final_round_priority(self, request: Request) -> None:
        if not self._enable_final_round_priority_boost:
            return
        if not getattr(request, "is_thinking_mode_enabled", False):
            return
        if not request.is_final_thinking_round or request.completed_thinking_rounds <= 0:
            return
        request.set_priority(min(request.priority, self._final_round_priority_value))

    def _is_final_prefill_fast_lane_request(self, request: Request) -> bool:
        return bool(
            getattr(request, "is_thinking_mode_enabled", False)
            and request.is_final_thinking_round
            and not request.is_prefill_complete
        )

    def _is_final_decode_fast_lane_request(self, request: Request) -> bool:
        return bool(
            getattr(request, "is_thinking_mode_enabled", False)
            and request.is_final_thinking_round
            and request.is_prefill_complete
            and not request.completed
        )

    def _ordered_requests_with_final_lane(
        self,
        requests: List[Request],
        *,
        final_predicate,
    ) -> List[Request]:
        final_requests: List[Request] = []
        non_final_requests: List[Request] = []
        for request in requests:
            if final_predicate(request):
                final_requests.append(request)
            else:
                non_final_requests.append(request)
        if not final_requests:
            return requests
        return final_requests + non_final_requests

    def _build_prefill_waiting_queue(self) -> deque[Request]:
        ordered_requests = self._get_sorted_waiting_queue()
        if (
            self._cluster_type == ClusterType.PREFILL
            and (
                self._final_prefill_reserved_slots > 0
                or self._final_prefill_reserved_tokens > 0
            )
        ):
            ordered_requests = self._ordered_requests_with_final_lane(
                ordered_requests,
                final_predicate=self._is_final_prefill_fast_lane_request,
            )
        return deque(ordered_requests)

    def _build_decode_waiting_queue(self) -> deque[Request]:
        ordered_requests = list(self._waiting_requests)
        if getattr(
            getattr(self, "_config", None), "enable_thinking_round_priority", False
        ):
            ordered_requests.sort(
                key=lambda r: (
                    0 if r.is_final_thinking_round else 1,
                    r.priority,
                    r.arrived_at,
                )
            )
        elif self._scheduling_policy == "priority":
            ordered_requests.sort(key=lambda r: (r.priority, r.arrived_at))

        if (
            self._cluster_type == ClusterType.DECODE
            and self._final_decode_reserved_slots > 0
        ):
            ordered_requests = self._ordered_requests_with_final_lane(
                ordered_requests,
                final_predicate=self._is_final_decode_fast_lane_request,
            )
        return deque(ordered_requests)

    def _count_final_fast_lane_requests(
        self,
        requests: List[Request] | deque[Request],
        *,
        final_predicate,
    ) -> int:
        return sum(1 for request in requests if final_predicate(request))

    def _select_final_running_reclaim_victim(
        self,
        *,
        final_predicate,
    ) -> Optional[Request]:
        candidates = [
            request
            for request in self._running_requests
            if not final_predicate(request)
        ]
        if not candidates:
            return None
        if self._scheduling_policy == "priority":
            return max(candidates, key=lambda r: (r.priority, r.arrived_at))
        return candidates[-1]

    def _reclaim_borrowed_final_running_slots(
        self,
        *,
        waiting_requests: List[Request] | deque[Request],
        final_predicate,
        reserved_slots: int,
        lane_name: str,
    ) -> List[Request]:
        if not self._enable_final_running_request_reclaim or reserved_slots <= 0:
            return []

        final_waiting_count = self._count_final_fast_lane_requests(
            waiting_requests,
            final_predicate=final_predicate,
        )
        if final_waiting_count <= 0:
            return []

        final_running_count = self._count_final_fast_lane_requests(
            self._running_requests,
            final_predicate=final_predicate,
        )
        remaining_reserved_slots = max(
            reserved_slots - min(final_running_count, reserved_slots),
            0,
        )
        target_new_final_admissions = min(
            final_waiting_count,
            remaining_reserved_slots,
        )
        if target_new_final_admissions <= 0:
            return []

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        reclaimed_requests: List[Request] = []
        while (
            max(self._max_num_running_reqs - len(self._running_requests), 0)
            < target_new_final_admissions
        ):
            victim = self._select_final_running_reclaim_victim(
                final_predicate=final_predicate,
            )
            if victim is None:
                logger.info(
                    "[FINAL-SLICE-RECLAIM] lane=%s stopped_without_victim "
                    "target_new_final_admissions=%s running_count=%s",
                    lane_name,
                    target_new_final_admissions,
                    len(self._running_requests),
                )
                break
            logger.info(
                "[FINAL-SLICE-RECLAIM] lane=%s reclaiming_hidden_req=%s "
                "target_new_final_admissions=%s running_count_before=%s",
                lane_name,
                victim.id,
                target_new_final_admissions,
                len(self._running_requests),
            )
            self._preempt_request(victim, reclaimed_requests)

        if reclaimed_requests:
            logger.info(
                "[FINAL-SLICE-RECLAIM] lane=%s reclaimed_count=%s "
                "running_count_after=%s",
                lane_name,
                len(reclaimed_requests),
                len(self._running_requests),
            )
        return reclaimed_requests

    def _get_num_waiting_reqs_for_decision_log(self) -> int:
        if self._cluster_type in (ClusterType.DECODE, ClusterType.DECODE_ATTN):
            return len(self._waiting_requests)
        return len(self._request_queue) + len(self._preempted_requests)

    def _apply_long_prefill_token_threshold(
        self, request: Request, num_new_tokens: int
    ) -> int:
        """Apply long prefill threshold only for prefill-phase requests."""
        if request.is_prefill_complete or self._long_prefill_token_threshold <= 0:
            return num_new_tokens
        return min(num_new_tokens, self._long_prefill_token_threshold)

    def _emit_schedule_decision_event(
        self,
        *,
        event: str,
        decision_result: Optional[str],
        request_id: Optional[int],
        token_budget: int,
        num_tokens: int,
        available_blocks: Optional[int] = None,
        batch_request_ids: Optional[List[int]] = None,
        request_num_tokens: Optional[List[int]] = None,
        batch_size: int = 0,
        batch_num_tokens: int = 0,
    ) -> None:
        if _frontier_vllm_v1_sched_decision_logger is None:
            return

        if available_blocks is None:
            available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)

        cluster_name = self._cluster_type.name if self._cluster_type else "MONOLITHIC"
        payload: Dict[str, Any] = {
            "event": event,
            "source": "frontier",
            "scheduler": "vllm_v1",
            "cluster_type": cluster_name,
            "iteration_id": int(self._active_schedule_iteration_id),
            "decision_result": decision_result,
            "request_id": None if request_id is None else str(request_id),
            "token_budget": int(token_budget),
            "available_blocks": int(available_blocks),
            "num_tokens": int(num_tokens),
            "num_running_reqs": len(self._running_requests),
            "num_waiting_reqs": self._get_num_waiting_reqs_for_decision_log(),
            "max_num_running_reqs": int(self._max_num_running_reqs),
            "max_num_scheduled_tokens": int(self._max_num_scheduled_tokens),
            "batch_request_ids": [str(req_id) for req_id in (batch_request_ids or [])],
            "request_num_tokens": [int(v) for v in (request_num_tokens or [])],
            "batch_size": int(batch_size),
            "batch_num_tokens": int(batch_num_tokens),
            "timestamp": time.time(),
            "timestamp_semantics": "wall_clock_epoch_seconds",
            "simulation_time": float(self._current_schedule_time),
            "simulation_time_semantics": "frontier_event_time_seconds",
        }
        if self._kv_cache_manager is not None:
            prefix_cache_stats = self._kv_cache_manager.prefix_cache_stats
            payload.update(
                {
                    "prefix_cache_metric_semantics": "block_level",
                    "prefix_cache_unit": "blocks",
                    "prefix_cache_block_size": int(self._config.block_size),
                    "prefix_cache_requests": int(prefix_cache_stats.requests),
                    "prefix_cache_queries": int(prefix_cache_stats.queries),
                    "prefix_cache_hits": int(prefix_cache_stats.hits),
                }
            )
        _log_frontier_vllm_v1_schedule_decision(payload)

    # ========== Batch Completion Handling ==========

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

        for request in batch.requests:
            self._refresh_target_embedded_mtp_prefill_boundary_state(batch, request)
            if request.completed:
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

    def _get_monolithic_pp_pending_terminal_release_iters(self) -> Dict[int, int]:
        pending = getattr(
            self,
            "_monolithic_pp_pending_terminal_release_iters",
            None,
        )
        if pending is None:
            pending = {}
            self._monolithic_pp_pending_terminal_release_iters = pending
        return pending

    def _get_monolithic_pp_extra_terminal_release_iters(self) -> int:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return 0

        pp = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp <= 1:
            return 0

        # Frontier's last-stage batch-end already accounts for one terminal
        # drain iteration. Deeper PP still needs the sampled-token-return
        # boundary to reach the scheduler before blocks can be released.
        return max(pp // 2 - 1, 0)

    def _has_monolithic_pp_pending_terminal_release(self) -> bool:
        return bool(self._get_monolithic_pp_pending_terminal_release_iters())

    def _has_monolithic_pp_visible_waiting_requests(self) -> bool:
        return bool(self._request_queue or self._preempted_requests)

    def _get_monolithic_pp_iteration_start_release_threshold(self) -> int:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return 1

        pp = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp <= 4:
            return 1

        # Once terminal release is materialized at iteration_start, deeper
        # MONOLITHIC+PP pipelines expose the release boundary earlier than the
        # old end-of-iteration bookkeeping. The validated scheduler-visible
        # contracts are pp4->1 and pp8->2, so keep the threshold PP-depth
        # aware instead of assuming a single remaining hop for every PP size.
        return max(pp // 4, 1)

    def _advance_monolithic_pp_terminal_release_boundary(self) -> None:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return

        release_visible_threshold = (
            self._get_monolithic_pp_iteration_start_release_threshold()
        )
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        ready_request_ids: List[int] = []
        for request_id, remaining_iters in list(pending_release_iters.items()):
            if (
                remaining_iters <= release_visible_threshold
                and self._has_monolithic_pp_visible_waiting_requests()
                and request_id
                not in self._monolithic_pp_waiting_sensitive_release_extensions
            ):
                self._monolithic_pp_waiting_sensitive_release_extensions.add(request_id)
                pending_release_iters[request_id] = 1
                logger.debug(
                    "[VLLMv1Engine] Delaying MONOLITHIC+PP terminal release for "
                    "request %s by one extra empty iteration because waiting "
                    "requests are already visible",
                    request_id,
                )
                continue
            if remaining_iters <= 1:
                ready_request_ids.append(request_id)
                pending_release_iters.pop(request_id, None)
            else:
                pending_release_iters[request_id] = remaining_iters - 1

        if not ready_request_ids:
            if pending_release_iters:
                self._monolithic_pp_terminal_release_followup_poll_pending = True
                logger.debug(
                    "[VLLMv1Engine] Keeping MONOLITHIC+PP terminal release self-driven "
                    "with one follow-up schedule poll while pending state remains: %s",
                    dict(pending_release_iters),
                )
            return

        ready_request_id_set = set(ready_request_ids)
        for request_id in ready_request_ids:
            self._free_request_resources_by_id(request_id)
            self._scheduled_num_computed_tokens_by_request.pop(request_id, None)
            self._monolithic_pp_waiting_sensitive_release_extensions.discard(request_id)

        self._running_requests = [
            request
            for request in self._running_requests
            if request.id not in ready_request_id_set
        ]
        self._monolithic_pp_terminal_release_followup_poll_pending = bool(
            pending_release_iters
        ) or self._has_monolithic_pp_visible_waiting_requests()

        logger.debug(
            "[VLLMv1Engine] Released %s MONOLITHIC+PP terminal request(s) "
            "after sampled-token-return-equivalent boundary: %s",
            len(ready_request_ids),
            ready_request_ids,
        )

    def _materialize_monolithic_pp_terminal_release_before_iteration_start(
        self,
    ) -> None:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return
        if self._has_monolithic_pp_visible_waiting_requests():
            return

        release_visible_threshold = (
            self._get_monolithic_pp_iteration_start_release_threshold()
        )
        ready_request_ids = [
            request_id
            for request_id, remaining_iters in list(pending_release_iters.items())
            if remaining_iters <= release_visible_threshold
        ]
        if not ready_request_ids:
            return

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        ready_request_id_set = set(ready_request_ids)
        for request_id in ready_request_ids:
            pending_release_iters.pop(request_id, None)
            self._free_request_resources_by_id(request_id)
            self._scheduled_num_computed_tokens_by_request.pop(request_id, None)
            self._monolithic_pp_waiting_sensitive_release_extensions.discard(
                request_id
            )

        self._running_requests = [
            request
            for request in self._running_requests
            if request.id not in ready_request_id_set
        ]
        logger.debug(
            "[VLLMv1Engine] Materialized %s MONOLITHIC+PP terminal release(s) "
            "before iteration_start because no waiting request is visible: %s",
            len(ready_request_ids),
            ready_request_ids,
        )

    def consume_monolithic_pp_terminal_release_followup_poll(self) -> bool:
        pending = bool(
            getattr(
                self,
                "_monolithic_pp_terminal_release_followup_poll_pending",
                False,
            )
        )
        self._monolithic_pp_terminal_release_followup_poll_pending = False
        return pending

    def _get_monolithic_pp_mtp_output_wait_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_output_wait_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_output_wait_request_ids = request_ids
        return request_ids

    def _get_monolithic_pp_mtp_output_wait_remaining_iters(self) -> Dict[int, int]:
        remaining_iters = getattr(
            self,
            "_monolithic_pp_mtp_output_wait_remaining_iters",
            None,
        )
        if remaining_iters is None:
            remaining_iters = {}
            self._monolithic_pp_mtp_output_wait_remaining_iters = remaining_iters
        return remaining_iters

    def _get_monolithic_pp_mtp_fractional_output_wait_counts(self) -> Dict[int, int]:
        counts = getattr(
            self,
            "_monolithic_pp_mtp_fractional_output_wait_counts",
            None,
        )
        if counts is None:
            counts = {}
            self._monolithic_pp_mtp_fractional_output_wait_counts = counts
        return counts

    def _add_monolithic_pp_mtp_output_wait(
        self, request_id: int, *, wait_iters: int = 2
    ) -> None:
        if wait_iters <= 0:
            raise ValueError(f"wait_iters must be positive, got={wait_iters}")
        self._get_monolithic_pp_mtp_output_wait_request_ids().add(request_id)
        remaining_iters = self._get_monolithic_pp_mtp_output_wait_remaining_iters()
        remaining_iters[request_id] = max(
            int(remaining_iters.get(request_id, 0)),
            int(wait_iters),
        )

    def _should_apply_monolithic_pp_mtp_output_wait(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not getattr(request, "spec_decode_enabled", False):
            return False
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return False
        if not getattr(request, "is_prefill_complete", False):
            return False
        if (
            request.id
            not in self._get_monolithic_pp_mtp_near_full_prefill_request_ids()
        ):
            return False
        processed_decode_tokens = int(
            getattr(request, "num_processed_decode_tokens", 0)
        )
        if processed_decode_tokens <= 0:
            return False
        if self._get_monolithic_pp_mtp_output_wait_iters() == 1:
            block_size = int(getattr(self._config, "block_size", 16))
            remaining_decode_tokens = (
                int(getattr(request, "num_decode_tokens", 0))
                - processed_decode_tokens
            )
            if remaining_decode_tokens <= block_size:
                # High-acceptance wide-MTP short tails have no useful future
                # prefill visibility to protect once the request is within the
                # final block-sized decode window. Skipping this idle turn
                # avoids a terminal PP output-wait residual without changing
                # CUDA op calibration.
                return False
        return True

    def _has_monolithic_pp_mtp_output_wait(self) -> bool:
        return bool(self._get_monolithic_pp_mtp_output_wait_request_ids())

    def _should_reserve_monolithic_pp_mtp_visible_budget(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not self._is_target_embedded_mtp_request(request):
            return False
        if not getattr(request, "is_prefill_complete", False):
            return False
        block_size = int(getattr(self._config, "block_size", 16))
        active_verify_window_tokens = int(
            getattr(request, "spec_current_verify_tokens", 0)
        )
        if active_verify_window_tokens <= 0:
            spec_config = getattr(self, "_spec_decode_config", None)
            if spec_config is None:
                spec_config = getattr(
                    getattr(self, "_replica_config", None),
                    "speculative_decoding_config",
                    None,
                )
            active_verify_window_tokens = int(
                getattr(spec_config, "num_speculative_tokens", 0)
            )
        if active_verify_window_tokens < block_size:
            # Narrow target-embedded MTP verify windows do not consume a full
            # cache block of output-visible scheduler budget. Reserving a
            # synthetic block-sized budget for them under MONOLITHIC+PP
            # under-batches waiting prefill relative to vLLM clean traces.
            return False
        return self._has_monolithic_pp_visible_waiting_requests()

    def _get_monolithic_pp_mtp_visible_budget_reservation_tokens(
        self, request: Request, token_budget: int
    ) -> int:
        if not self._should_reserve_monolithic_pp_mtp_visible_budget(request):
            return 0
        reserved_tokens = max(
            int(getattr(request, "spec_current_verify_tokens", 1)),
            self._get_request_next_num_tokens(request),
        )
        return min(max(reserved_tokens, 0), token_budget)

    def _clear_monolithic_pp_mtp_output_wait(self) -> None:
        request_ids = self._get_monolithic_pp_mtp_output_wait_request_ids()
        remaining_iters = self._get_monolithic_pp_mtp_output_wait_remaining_iters()
        if not remaining_iters:
            request_ids.clear()
            return
        next_waiting_request_ids: set[int] = set()
        for request_id in list(request_ids):
            remaining = int(remaining_iters.get(request_id, 1)) - 1
            if remaining > 0:
                remaining_iters[request_id] = remaining
                next_waiting_request_ids.add(request_id)
            else:
                remaining_iters.pop(request_id, None)
        request_ids.clear()
        request_ids.update(next_waiting_request_ids)

    def consume_monolithic_pp_mtp_output_wait_followup_poll(self) -> bool:
        pending = bool(
            getattr(
                self,
                "_monolithic_pp_mtp_output_wait_followup_poll_pending",
                False,
            )
        )
        self._monolithic_pp_mtp_output_wait_followup_poll_pending = False
        return pending

    # ========== Memory Allocation Helpers ==========

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

    def _can_allocate_request(
        self,
        request: Request,
        num_new_tokens: int = 1,
        new_computed_blocks=None,
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
        if self._is_prefix_caching_enabled():
            assert self._kv_cache_manager is not None
            return self._kv_cache_manager.can_allocate_slots(
                request,
                num_new_tokens,
                new_computed_blocks=new_computed_blocks,
            )

        reserved_tokens = self._get_num_tokens_for_kv_reservation(
            request, num_new_tokens
        )
        if request.id not in self._allocation_map:
            # New request - estimate blocks from current token frontier
            # (already processed + newly scheduled in this iteration), then
            # clamp by max_model_len to keep allocation semantics consistent
            # with chunked prefill scheduling.
            kv_accounted_tokens = self._get_kv_accounted_processed_tokens(request)
            total_tokens = min(
                kv_accounted_tokens + reserved_tokens, self._max_model_len
            )
            num_required_blocks = ceil(total_tokens / self._config.block_size)
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
    ) -> None:
        """
        Allocate memory blocks for a request.

        For new requests: allocate blocks for prefill tokens + first decode token.
        For running requests: allocate additional blocks if needed.

        Args:
            request: The request to allocate for
            num_new_tokens: Number of new tokens being processed
        """
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        if self._is_prefix_caching_enabled():
            assert self._kv_cache_manager is not None
            allocated_blocks = self._kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                new_computed_blocks=new_computed_blocks,
            )
            if allocated_blocks is None:
                raise ValueError(
                    f"Failed to allocate prefix-cache-managed KV blocks for request {request.id}"
                )
            self._sync_prefix_cache_allocation_state(request)
            return

        reserved_tokens = self._get_num_tokens_for_kv_reservation(
            request, num_new_tokens
        )

        if request.id not in self._allocation_map:
            # New request - allocate blocks only for tokens scheduled in this iteration.
            # This aligns with vLLM v1 allocate_slots(request, num_new_tokens + external_tokens).
            num_required_blocks = ceil(reserved_tokens / self._config.block_size)
            self.allocate(request.id, num_required_blocks)
            logger.debug(
                f"[VLLMv1Engine] Allocated {num_required_blocks} blocks for request {request.id} "
                f"(scheduled_tokens={num_new_tokens}, reserved_tokens={reserved_tokens})"
            )
            return

        # Running request - check if additional blocks needed
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        kv_accounted_tokens = self._get_kv_accounted_processed_tokens(request)
        num_tokens_required = max(
            0, kv_accounted_tokens + reserved_tokens - num_tokens_reserved
        )

        if num_tokens_required <= 0:
            return

        # Allocate additional blocks
        num_additional_blocks = ceil(num_tokens_required / self._config.block_size)
        self.allocate(request.id, num_additional_blocks)

    # ========== Preemption Logic ==========

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

        # Mark as preempted and reset computed tokens
        victim._preempted = True
        victim._num_processed_tokens = 0  # Reset computed tokens as in vLLM v1
        self._scheduled_num_computed_tokens_by_request.pop(victim.id, None)

        # Record re-entry to waiting queue for waiting time tracking
        # This must be called AFTER resetting num_processed_tokens but BEFORE
        # adding to the waiting queue
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
        self, request: Request, num_new_tokens: int, preempted_requests: List[Request]
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
            if self._can_allocate_request(request, num_new_tokens):
                self._allocate_request(request, num_new_tokens)
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

    # ========== Phase 1: RUNNING Requests Scheduling ==========

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

            active_in_pp_batch = (
                self._cluster_type in {ClusterType.MONOLITHIC, ClusterType.DECODE}
                and self._num_stages > 1
                and self._is_request_active_in_batch(request)
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
            max_allowed = self._max_model_len - self._get_scheduler_num_computed_tokens(
                request
            )
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
                request, num_new_tokens, preempted_requests
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

    # ========== Phase 2: WAITING Requests Scheduling ==========

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

            # Calculate number of new tokens to process
            if self._is_prefix_caching_enabled() and not request.is_prefill_complete:
                (
                    computed_blocks,
                    prefix_cached_tokens,
                    num_new_tokens,
                ) = self._prepare_prefix_cache_admission(request)
                max_allowed = self._max_model_len - prefix_cached_tokens
            else:
                num_new_tokens = self._get_request_next_num_tokens(request)
                max_allowed = self._max_model_len - self._get_scheduler_num_computed_tokens(
                    request
                )

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
                and not request.is_prefill_complete
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

            # Remove from waiting queues and allocate
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

            self._allocate_request(
                request,
                num_new_tokens,
                new_computed_blocks=computed_blocks,
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

    # ========== Main Scheduling Entry Point ==========

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
        self._materialize_monolithic_pp_terminal_release_before_iteration_start()
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
            self._advance_monolithic_pp_terminal_release_boundary()
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
            max_allowed = self._max_model_len - self._get_scheduler_num_computed_tokens(
                request
            )
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
            if not self._can_allocate_request(request, num_new_tokens):
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

            self._allocate_request(request, num_new_tokens)
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
                success = self._try_allocate_with_preemption(
                    request, num_new_tokens, preempted_requests
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
                    self._allocate_request(request, num_tokens)
                    self._waiting_requests.remove(request)
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
            if self._cluster_type == ClusterType.DECODE_ATTN and num_stages > 1:
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
            for stage_idx in range(len(stage_reqs_start_loc) - 1):
                start_idx = stage_reqs_start_loc[stage_idx]
                end_idx = stage_reqs_start_loc[stage_idx + 1]
                stage_requests = scheduled_requests[start_idx:end_idx]
                stage_tokens = scheduled_tokens[start_idx:end_idx]
                micro_batch = self._create_batch(stage_requests, stage_tokens)
                micro_batch.afd_stage_idx = stage_idx
                if afd_stage_metadata is not None:
                    micro_batch.afd_stage_metadata = afd_stage_metadata
                if first_micro_batch is None:
                    first_micro_batch = micro_batch
                else:
                    self._af_pending_micro_batches.append(micro_batch)

            return first_micro_batch


        logger.debug(f"[VLLMv1Engine][DECODE_ATTN] No requests to schedule")
        return None

    def _attach_afd_metadata_if_needed(self, batch: Batch) -> Batch:
        """Attach AFD stage metadata to batch if num_stages > 1.

        This method generates AFDStageMetadata following StepFun-vLLM's three-layer
        padding strategy. The metadata is used for compute time and communication
        volume prediction.

        Note: DP padding (Layer 2) requires per-stage max tokens across DP ranks.
        In simulator, we can compute this at cluster scheduler level where all DP
        lanes are visible. For now, we skip DP padding here and let cluster scheduler
        handle it when aggregating batches.

        Args:
            batch: The batch to attach metadata to

        Returns:
            The batch with afd_stage_metadata attached (if applicable)
        """
        from frontier.entities.batch import AFDStageMetadata

        # Get num_stages from cluster config (af_pipeline_num_micro_batch)
        num_stages = self._af_pipeline_num_micro_batch
        if num_stages is None or num_stages <= 1:
            return batch

        if self._cluster_type != ClusterType.DECODE_ATTN:
            return batch

        # Get CUDA Graph configuration from global simulation config
        from frontier.config import global_vars

        use_cuda_graph = global_vars.get_use_cuda_graph()
        cudagraph_capture_sizes = global_vars.get_cudagraph_capture_sizes()

        # Generate default capture sizes if not specified
        # Aligned with StepFun-vLLM's default: [1, 2, 4] + [8 * i for i in range(1, max_num_seqs // 8 + 1)]
        if use_cuda_graph and cudagraph_capture_sizes is None:
            max_num_seqs = self._micro_batch_size if hasattr(self, "_micro_batch_size") else 64
            cudagraph_capture_sizes = [1, 2, 4] + [
                8 * i for i in range(1, max_num_seqs // 8 + 1)
            ]

        # Compute AFD metadata
        # Note: dp_stage_max_tokens is None here - cluster scheduler will handle DP padding
        batch.afd_stage_metadata = AFDStageMetadata.from_batch_params(
            num_reqs=len(batch.requests),
            num_tokens_per_req=batch.num_tokens,
            num_stages=num_stages,
            dp_stage_max_tokens=None,  # Cluster scheduler handles DP padding
            use_cuda_graph=use_cuda_graph,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            ffn_use_cuda_graph=use_cuda_graph,
            ffn_cudagraph_capture_sizes=cudagraph_capture_sizes,
        )

        logger = get_cluster_logger(__name__, self._cluster_type.name)
        logger.debug(
            f"[AFD-METADATA] batch={batch.id} "
            f"num_stages={num_stages} "
            f"original_tokens={batch.afd_stage_metadata.original_total_tokens} "
            f"padded_tokens={batch.afd_stage_metadata.padded_total_tokens} "
            f"padding_overhead={batch.afd_stage_metadata.num_pad_tokens}"
        )

        return batch

    # ========== Property Overrides ==========

    @property
    def num_pending_requests(self) -> int:
        """
        Return total pending requests including waiting queue for DECODE cluster.

        For DECODE cluster, this includes requests in _waiting_requests.
        For other clusters, uses the base implementation.
        """
        base_count = len(self._request_queue)
        waiting_count = len(self._waiting_requests)
        return base_count + waiting_count

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
            f"[RS-IDLE-CHECK][replica={self._replica_id}][dp={self._dp_id}] "
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
