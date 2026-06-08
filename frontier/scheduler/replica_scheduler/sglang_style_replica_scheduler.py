from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from frontier.entities.batch import Batch, Request
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
    _log_frontier_vllm_v1_schedule_decision,
)
from frontier.types import ClusterType


class SGLangStyleReplicaScheduler(VLLMv1EngineReplicaScheduler):
    """
    Frontier scheduler with SGLang-style MONOLITHIC batch selection.

    The implementation intentionally reuses the existing Frontier vLLM v1
    allocation and preemption helpers. The only behavioral change is the
    high-level scheduling order:

    1. Prefer any schedulable prefill-stage work first.
    2. Fall back to running decode only when no prefill batch can be formed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._cluster_type not in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
            raise ValueError(
                "SGLangStyleReplicaScheduler only supports MONOLITHIC or PREFILL cluster_type, "
                f"got {self._cluster_type!r}"
            )

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
        if available_blocks is None:
            available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)

        cluster_name = self._cluster_type.name if self._cluster_type else "MONOLITHIC"
        payload: Dict[str, Any] = {
            "event": event,
            "source": "frontier",
            "scheduler": "sglang_style",
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

    def _is_prefill_stage_request(self, request: Request) -> bool:
        return bool(getattr(request, "_preempted", False)) or not request.is_prefill_complete

    def _get_request_next_num_tokens(self, request: Request) -> int:
        if getattr(request, "_preempted", False):
            computed_tokens = self._get_scheduler_num_computed_tokens(request)
            remaining_prefill_tokens = int(request.num_prefill_tokens) - computed_tokens
            return max(remaining_prefill_tokens, 0)
        return super()._get_request_next_num_tokens(request)

    def _get_split_waiting_requests(self) -> Tuple[List[Request], List[Request]]:
        ordered_waiting_requests = self._get_sorted_waiting_queue()
        prefill_waiting_requests = [
            request
            for request in ordered_waiting_requests
            if self._is_prefill_stage_request(request)
        ]
        other_waiting_requests = [
            request
            for request in ordered_waiting_requests
            if not self._is_prefill_stage_request(request)
        ]
        return prefill_waiting_requests, other_waiting_requests

    def _schedule_prefill_stage_first(
        self, token_budget: int
    ) -> Tuple[int, List[Request], List[int], List[Request], List[int]]:
        original_running_requests = list(self._running_requests)
        original_waiting_requests = self._get_sorted_waiting_queue()

        prefill_running_requests = [
            request
            for request in original_running_requests
            if self._is_prefill_stage_request(request)
        ]
        decode_running_requests = [
            request
            for request in original_running_requests
            if not self._is_prefill_stage_request(request)
        ]
        prefill_waiting_requests, other_waiting_requests = self._get_split_waiting_requests()

        if not prefill_running_requests and not prefill_waiting_requests:
            return token_budget, [], [], [], []

        self._running_requests = list(prefill_running_requests)
        self._set_waiting_queues_from_ordered_requests(prefill_waiting_requests)

        preempted_requests: List[Request] = []
        token_budget, running_scheduled, running_tokens = self._schedule_running_requests(
            token_budget,
            preempted_requests,
        )
        waiting_scheduled: List[Request] = []
        waiting_tokens: List[int] = []
        if not preempted_requests:
            token_budget, waiting_scheduled, waiting_tokens = (
                self._schedule_waiting_requests(token_budget)
            )

        scheduled_any = bool(running_scheduled or waiting_scheduled)
        updated_prefill_running_requests = list(self._running_requests)
        updated_prefill_waiting_requests = self._get_sorted_waiting_queue()

        if not scheduled_any:
            self._running_requests = original_running_requests
            self._set_waiting_queues_from_ordered_requests(original_waiting_requests)
            return token_budget, [], [], [], []

        self._running_requests = (
            decode_running_requests + list(updated_prefill_running_requests)
        )
        self._set_waiting_queues_from_ordered_requests(
            list(updated_prefill_waiting_requests) + other_waiting_requests
        )
        return (
            token_budget,
            waiting_scheduled,
            waiting_tokens,
            running_scheduled,
            running_tokens,
        )

    def _schedule_decode_fallback_running_requests(
        self, token_budget: int
    ) -> Tuple[int, List[Request], List[int]]:
        original_running_requests = list(self._running_requests)
        prefill_running_requests = [
            request
            for request in original_running_requests
            if self._is_prefill_stage_request(request)
        ]
        decode_running_requests = [
            request
            for request in original_running_requests
            if not self._is_prefill_stage_request(request)
        ]

        self._running_requests = list(decode_running_requests)
        preempted_requests: List[Request] = []
        token_budget, running_scheduled, running_tokens = self._schedule_running_requests(
            token_budget,
            preempted_requests,
        )
        updated_decode_running_requests = list(self._running_requests)
        self._running_requests = (
            list(prefill_running_requests) + updated_decode_running_requests
        )
        return token_budget, running_scheduled, running_tokens

    def _schedule_two_phase(self) -> Optional[Batch]:
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        self._materialize_monolithic_pp_terminal_release_before_iteration_start()
        token_budget = self._max_num_scheduled_tokens
        available_blocks = int(self._config.num_blocks - self._num_allocated_blocks)
        waiting_count = len(self._request_queue) + len(self._preempted_requests)

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

        total_blocks = int(self._config.num_blocks)
        allocated_blocks = int(self._num_allocated_blocks)
        usage_ratio = allocated_blocks / total_blocks if total_blocks > 0 else 0.0
        logger.info(
            f"[MEMORY_STATE] total_blocks={total_blocks}, "
            f"allocated_blocks={allocated_blocks}, "
            f"free_blocks={available_blocks}, "
            f"usage_ratio={usage_ratio:.4f}, "
            f"watermark_blocks={self._watermark_blocks}"
        )

        if self._monolithic_pp_terminal_release_followup_poll_pending:
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

        logger.info(
            f"[PREFILL_FIRST_START] running_count={len(self._running_requests)}, "
            f"waiting_count={waiting_count}, token_budget={token_budget}"
        )
        (
            token_budget,
            waiting_scheduled,
            waiting_tokens,
            running_prefill_scheduled,
            running_prefill_tokens,
        ) = self._schedule_prefill_stage_first(token_budget)

        if waiting_scheduled or running_prefill_scheduled:
            ordered_scheduled_requests = waiting_scheduled + running_prefill_scheduled
            ordered_num_tokens = waiting_tokens + running_prefill_tokens
            total_tokens = sum(ordered_num_tokens)

            logger.info(
                f"[PREFILL_FIRST_END] admitted_count={len(waiting_scheduled)}, "
                f"running_prefill_count={len(running_prefill_scheduled)}, "
                f"token_budget_remaining={token_budget}"
            )
            logger.info(
                f"[BATCH_FORMATION] total_tokens={total_tokens}, "
                f"new_admitted={len(waiting_scheduled)}, "
                f"resumed=0, "
                f"running_continued={len(running_prefill_scheduled)}, "
                f"batch_size={len(ordered_scheduled_requests)}"
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
                batch_size=len(ordered_scheduled_requests),
                batch_num_tokens=total_tokens,
            )
            self._advance_monolithic_pp_terminal_release_boundary()
            return self._create_batch(ordered_scheduled_requests, ordered_num_tokens)

        logger.info(
            "[PREFILL_FIRST_FALLBACK] no schedulable prefill-stage batch; "
            "falling back to running decode"
        )
        token_budget, running_decode_scheduled, running_decode_tokens = (
            self._schedule_decode_fallback_running_requests(token_budget)
        )

        if not running_decode_scheduled:
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

        total_tokens = sum(running_decode_tokens)
        logger.info(
            f"[DECODE_FALLBACK_END] scheduled_count={len(running_decode_scheduled)}, "
            f"token_budget_remaining={token_budget}"
        )
        logger.info(
            f"[BATCH_FORMATION] total_tokens={total_tokens}, "
            f"new_admitted=0, resumed=0, "
            f"running_continued={len(running_decode_scheduled)}, "
            f"batch_size={len(running_decode_scheduled)}"
        )
        self._emit_schedule_decision_event(
            event="iteration_end",
            decision_result=None,
            request_id=None,
            token_budget=token_budget,
            num_tokens=total_tokens,
            available_blocks=int(self._config.num_blocks - self._num_allocated_blocks),
            batch_request_ids=[request.id for request in running_decode_scheduled],
            request_num_tokens=running_decode_tokens,
            batch_size=len(running_decode_scheduled),
            batch_num_tokens=total_tokens,
        )
        self._advance_monolithic_pp_terminal_release_boundary()
        return self._create_batch(running_decode_scheduled, running_decode_tokens)
