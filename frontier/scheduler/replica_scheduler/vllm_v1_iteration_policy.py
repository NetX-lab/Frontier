"""Per-iteration scheduling policy, fast lanes, CUDA graph and spec-decode metadata.

These methods decide the shape of one scheduling iteration: which policy orders
the waiting queue, whether a request takes a final-round fast lane, which CUDA
graph capture size the decode batch maps onto, and what speculative-decoding
metadata the batch carries.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple

from frontier.config import global_vars
from frontier.entities.batch import (
    Batch,
    DecodeCudaGraphMetadata,
    Request,
    SpecDecodeBatchMetadata,
)
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.vllm_v1_decision_log import (
    _log_frontier_vllm_v1_schedule_decision,
    schedule_decision_logging_enabled,
)
from frontier.scheduler.request_load import RequestLoad
from frontier.spec_decode import compute_iteration_outcome, get_planned_draft_tokens
from frontier.types import ClusterType


class IterationSchedulingPolicy:
    """Iteration ordering, fast lanes, CUDA graph sizing and batch metadata."""

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
            if request.is_decoding
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
            if not request.is_decoding or not getattr(
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

    def get_request_load(self) -> RequestLoad:
        """Report this lane's populations from the one waiting definition.

        Running counts every admitted request, including one that is admitted
        but not scheduled in the current iteration. Waiting reuses the
        decision-log accessor above so a load balancer and the decision log can
        never disagree about what is waiting.
        """

        return RequestLoad(
            self._get_num_waiting_reqs_for_decision_log(),
            len(self._running_requests),
        )

    def _apply_long_prefill_token_threshold(
        self, request: Request, num_new_tokens: int
    ) -> int:
        """Apply the long-prefill token cap to prefill and recompute chunks."""
        if request.is_decoding or self._long_prefill_token_threshold <= 0:
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
        if not schedule_decision_logging_enabled():
            return

        if available_blocks is None:
            available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)

        cluster_name = self._cluster_type.name if self._cluster_type else "MONOLITHIC"
        request_load = self.get_request_load()
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
            "num_running_reqs": request_load.running,
            "num_waiting_reqs": request_load.waiting,
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
