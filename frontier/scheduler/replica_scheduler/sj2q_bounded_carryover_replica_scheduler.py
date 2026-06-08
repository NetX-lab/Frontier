from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from frontier.entities.batch import Batch, Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


QSHORT = "Qshort"
QLONG = "Qlong"


@dataclass
class SJ2QBoundedCarryoverSessionState:
    session_id: int
    queue_level: str = QSHORT
    long_history: bool = False
    served_new_tokens_total: int = 0
    current_round_new_prompt_tokens: int = 0
    current_round_prefill_spilled: bool = False
    qlong_wait_started_at: Optional[float] = None
    last_round_total_tokens: int = 0
    qshort_entry_count: int = 0
    qshort_long_history_entry_count: int = 0
    qlong_entry_count: int = 0
    long_history_to_qshort_reentry_count: int = 0
    first_long_history_round_number: int = 0
    carryover_release_pending: bool = False
    carryover_release_consumed_count: int = 0


class SJ2QBoundedCarryoverReplicaScheduler(VLLMv1EngineReplicaScheduler):
    """Two-queue bounded-carryover scheduler without long-history rescue."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sj2q_bounded_session_states: Dict[int, SJ2QBoundedCarryoverSessionState] = {}
        self._sj2q_bounded_long_round_threshold = int(
            getattr(self._config, "long_round_new_prompt_threshold", 4096)
        )
        self._sj2q_bounded_service_cap_tokens = int(
            getattr(self._config, "service_cap_tokens", 8192)
        )
        self._sj2q_bounded_long_liveness_quota = int(
            getattr(self._config, "long_liveness_quota", 32)
        )
        self._sj2q_bounded_short_streak_counter = 0
        self._sj2q_bounded_forced_qlong_request_id: Optional[int] = None
        self._sj2q_bounded_release_request_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Oracle-prohibition overrides
    # ------------------------------------------------------------------
    def _resolve_iteration_round_class(self) -> Optional[str]:
        return None

    def _get_iteration_scheduler_profile(self) -> Dict[str, object]:
        return {
            "round_class": None,
            "max_num_running_reqs": int(self._config.batch_size_cap),
            "max_num_scheduled_tokens": int(self._config.max_tokens_in_batch),
            "enable_chunked_prefill": bool(
                getattr(self._config, "enable_chunked_prefill", False)
            ),
        }

    def _maybe_promote_final_round_priority(self, request: Request) -> None:
        return None

    def _is_final_prefill_fast_lane_request(self, request: Request) -> bool:
        return False

    def _is_final_decode_fast_lane_request(self, request: Request) -> bool:
        return False

    # ------------------------------------------------------------------
    # Session-state helpers
    # ------------------------------------------------------------------
    def _get_session_id(self, request: Request) -> int:
        session_id = getattr(request, "session_id", None)
        if session_id is None:
            raise ValueError(
                "SJ2QBoundedCarryoverReplicaScheduler requires request.session_id."
            )
        return int(session_id)

    def _get_request_arrival_time(self, request: Request) -> float:
        cluster_arrival_getter = getattr(request, "get_cluster_arrival_time", None)
        if callable(cluster_arrival_getter):
            try:
                return float(cluster_arrival_getter(self._cluster_type))
            except (TypeError, ValueError, IndexError):
                pass
        return float(getattr(request, "arrived_at", 0.0))

    def _get_request_state_snapshot(self, request: Request) -> Optional[dict]:
        snapshot = getattr(request, "_sj2q_bounded_state_snapshot", None)
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            raise TypeError(
                "request._sj2q_bounded_state_snapshot must be a dict when present."
            )
        return snapshot

    def _load_or_create_session_state(self, request: Request) -> SJ2QBoundedCarryoverSessionState:
        session_id = self._get_session_id(request)
        existing_state = self._sj2q_bounded_session_states.get(session_id)
        if existing_state is not None:
            return existing_state

        snapshot = self._get_request_state_snapshot(request)
        if snapshot is not None:
            state = SJ2QBoundedCarryoverSessionState(
                session_id=session_id,
                queue_level=str(snapshot.get("queue_level", QSHORT)),
                long_history=bool(snapshot.get("long_history", False)),
                served_new_tokens_total=int(snapshot.get("served_new_tokens_total", 0)),
                current_round_new_prompt_tokens=int(
                    snapshot.get("current_round_new_prompt_tokens", 0)
                ),
                current_round_prefill_spilled=bool(
                    snapshot.get("current_round_prefill_spilled", False)
                ),
                qlong_wait_started_at=(
                    None
                    if snapshot.get("qlong_wait_started_at") is None
                    else float(snapshot["qlong_wait_started_at"])
                ),
                last_round_total_tokens=int(snapshot.get("last_round_total_tokens", 0)),
                qshort_entry_count=int(snapshot.get("qshort_entry_count", 0)),
                qshort_long_history_entry_count=int(
                    snapshot.get("qshort_long_history_entry_count", 0)
                ),
                qlong_entry_count=int(snapshot.get("qlong_entry_count", 0)),
                long_history_to_qshort_reentry_count=int(
                    snapshot.get("long_history_to_qshort_reentry_count", 0)
                ),
                first_long_history_round_number=int(
                    snapshot.get("first_long_history_round_number", 0)
                ),
                carryover_release_pending=bool(
                    snapshot.get("carryover_release_pending", False)
                ),
                carryover_release_consumed_count=int(
                    snapshot.get("carryover_release_consumed_count", 0)
                ),
            )
        else:
            state = SJ2QBoundedCarryoverSessionState(session_id=session_id)

        self._sj2q_bounded_session_states[session_id] = state
        return state

    def _sync_request_state_snapshot(
        self,
        request: Request,
        state: SJ2QBoundedCarryoverSessionState,
    ) -> None:
        setattr(request, "_sj2q_bounded_state_snapshot", asdict(state))
        setattr(request, "_sj2q_bounded_queue_level", state.queue_level)
        setattr(request, "_sj2q_bounded_qshort_entries_total", state.qshort_entry_count)
        setattr(
            request,
            "_sj2q_bounded_qshort_long_history_entries_total",
            state.qshort_long_history_entry_count,
        )
        setattr(request, "_sj2q_bounded_qlong_entries_total", state.qlong_entry_count)
        setattr(
            request,
            "_sj2q_bounded_long_history_to_qshort_reentry_count",
            state.long_history_to_qshort_reentry_count,
        )
        setattr(
            request,
            "_sj2q_bounded_first_long_history_round_number",
            state.first_long_history_round_number,
        )
        setattr(
            request,
            "_sj2q_bounded_carryover_release_pending",
            state.carryover_release_pending,
        )
        setattr(
            request,
            "_sj2q_bounded_carryover_release_consumed_count",
            state.carryover_release_consumed_count,
        )

    def _compute_current_round_new_prompt_tokens(
        self,
        request: Request,
        state: SJ2QBoundedCarryoverSessionState,
    ) -> int:
        return max(int(request.num_prefill_tokens) - int(state.last_round_total_tokens), 0)

    def _get_current_round_number(self, request: Request) -> int:
        round_number = getattr(request, "current_thinking_round_number", 1)
        try:
            round_number = int(round_number)
        except (TypeError, ValueError):
            round_number = 1
        return max(round_number, 1)

    def _mark_long_history(
        self,
        request: Request,
        state: SJ2QBoundedCarryoverSessionState,
    ) -> None:
        if state.long_history:
            return
        state.long_history = True
        if state.first_long_history_round_number <= 0:
            state.first_long_history_round_number = self._get_current_round_number(
                request
            )

    def _record_queue_assignment(
        self,
        request: Request,
        state: SJ2QBoundedCarryoverSessionState,
        *,
        queue_level: str,
        assigned_at: Optional[float] = None,
    ) -> None:
        if queue_level == QSHORT:
            state.qshort_entry_count += 1
            state.qlong_wait_started_at = None
            if state.long_history:
                state.qshort_long_history_entry_count += 1
                state.long_history_to_qshort_reentry_count += 1
        elif queue_level == QLONG:
            state.qlong_entry_count += 1
            state.qlong_wait_started_at = float(
                self._get_request_arrival_time(request)
                if assigned_at is None
                else assigned_at
            )
        else:
            raise ValueError(f"Unsupported queue_level={queue_level!r}")

        state.queue_level = queue_level
        self._sync_request_state_snapshot(request, state)

    def _classify_new_round(
        self,
        request: Request,
        state: SJ2QBoundedCarryoverSessionState,
    ) -> None:
        round_new_prompt_tokens = self._compute_current_round_new_prompt_tokens(
            request, state
        )
        state.current_round_new_prompt_tokens = int(round_new_prompt_tokens)
        state.current_round_prefill_spilled = False
        assigned_at = self._get_request_arrival_time(request)

        if state.long_history:
            queue_level = QLONG
        elif state.served_new_tokens_total > self._sj2q_bounded_service_cap_tokens:
            self._mark_long_history(request, state)
            queue_level = QLONG
        elif round_new_prompt_tokens > self._sj2q_bounded_long_round_threshold:
            self._mark_long_history(request, state)
            queue_level = QLONG
        else:
            queue_level = QSHORT

        self._record_queue_assignment(
            request,
            state,
            queue_level=queue_level,
            assigned_at=assigned_at,
        )

    def _get_state_for_request(
        self,
        request: Request,
    ) -> Optional[SJ2QBoundedCarryoverSessionState]:
        try:
            session_id = self._get_session_id(request)
        except ValueError:
            return None
        return self._sj2q_bounded_session_states.get(session_id)

    def _is_decode_slice(self, request: Request) -> bool:
        if self._cluster_type in {ClusterType.DECODE, ClusterType.DECODE_ATTN}:
            return True
        return bool(getattr(request, "is_prefill_complete", False))

    def _select_forced_qlong_request_id(self, requests: Sequence[Request]) -> Optional[int]:
        if self._sj2q_bounded_short_streak_counter < self._sj2q_bounded_long_liveness_quota:
            return None
        qlong_requests = [
            request
            for request in requests
            if (state := self._get_state_for_request(request)) is not None
            and state.queue_level == QLONG
        ]
        if not qlong_requests:
            return None
        oldest = min(qlong_requests, key=self._get_request_arrival_time)
        return int(getattr(oldest, "id"))

    def _select_carryover_release_request_id(
        self,
        requests: Sequence[Request],
    ) -> Optional[int]:
        carryover_requests = [
            request
            for request in requests
            if (state := self._get_state_for_request(request)) is not None
            and state.queue_level == QLONG
            and state.carryover_release_pending
            and not self._is_decode_slice(request)
        ]
        if not carryover_requests:
            return None

        qshort_arrivals = [
            self._get_request_arrival_time(request)
            for request in requests
            if (state := self._get_state_for_request(request)) is not None
            and state.queue_level == QSHORT
        ]
        if not qshort_arrivals:
            oldest = min(carryover_requests, key=self._get_request_arrival_time)
            return int(getattr(oldest, "id"))

        qshort_head_arrival = min(qshort_arrivals)
        eligible_requests = [
            request
            for request in carryover_requests
            if self._get_request_arrival_time(request) <= qshort_head_arrival
        ]
        if not eligible_requests:
            return None

        oldest = min(eligible_requests, key=self._get_request_arrival_time)
        return int(getattr(oldest, "id"))

    def _queue_rank(self, request: Request) -> int:
        request_id = int(getattr(request, "id"))
        if self._sj2q_bounded_release_request_id == request_id:
            return -2
        if self._sj2q_bounded_forced_qlong_request_id == request_id:
            return -1
        state = self._get_state_for_request(request)
        if state is None:
            return 0
        return 0 if state.queue_level == QSHORT else 1

    def _request_sort_tuple(self, request: Request) -> Tuple[int, int, int, float]:
        state = self._get_state_for_request(request)
        queue_rank = self._queue_rank(request)
        arrival = float(getattr(request, "arrived_at", 0.0))
        is_decode_slice = self._is_decode_slice(request)
        if state is None:
            return (queue_rank, 0, 0, arrival)
        if state.queue_level == QSHORT and queue_rank >= 0:
            return (
                queue_rank,
                int(state.current_round_new_prompt_tokens),
                0 if not is_decode_slice else 1,
                arrival,
            )
        if queue_rank < 0:
            return (queue_rank, 0, 0 if is_decode_slice else 1, arrival)
        return (queue_rank, 0 if is_decode_slice else 1, 0, arrival)

    def _sort_requests_by_queue_level(
        self,
        requests: Sequence[Request],
    ) -> List[Request]:
        self._sj2q_bounded_release_request_id = self._select_carryover_release_request_id(
            requests
        )
        self._sj2q_bounded_forced_qlong_request_id = self._select_forced_qlong_request_id(
            requests
        )
        if self._scheduling_policy == "priority":
            return sorted(
                requests,
                key=lambda request: (
                    *self._request_sort_tuple(request),
                    int(getattr(request, "priority", 0)),
                ),
            )
        return sorted(requests, key=self._request_sort_tuple)

    def _mark_prefill_spill_if_needed(
        self,
        request: Request,
        *,
        completed_at: float,
    ) -> None:
        if self._cluster_type not in {ClusterType.MONOLITHIC, ClusterType.PREFILL}:
            return
        state = self._get_state_for_request(request)
        if state is None:
            return
        if state.current_round_prefill_spilled:
            return
        if bool(request.is_prefill_complete):
            return
        if int(getattr(request, "num_processed_tokens", 0)) <= 0:
            return
        if int(getattr(request, "num_processed_tokens", 0)) >= int(
            request.num_prefill_tokens
        ):
            return

        state.current_round_prefill_spilled = True
        self._mark_long_history(request, state)
        state.carryover_release_pending = True
        self._record_queue_assignment(
            request,
            state,
            queue_level=QLONG,
            assigned_at=float(completed_at),
        )

    def _account_service_tokens(
        self,
        request: Request,
        *,
        executed_tokens: int,
        completed_at: float,
    ) -> None:
        state = self._get_state_for_request(request)
        if state is None:
            return

        state.served_new_tokens_total += int(executed_tokens)
        if (
            state.served_new_tokens_total > self._sj2q_bounded_service_cap_tokens
            and not state.long_history
        ):
            self._mark_long_history(request, state)
            if not bool(request.completed):
                self._record_queue_assignment(
                    request,
                    state,
                    queue_level=QLONG,
                    assigned_at=float(completed_at),
                )
                return

        if bool(request.completed):
            state.last_round_total_tokens = int(request.total_tokens)

        self._sync_request_state_snapshot(request, state)

    def _consume_carryover_release_credit_if_used(self, request: Request) -> None:
        if self._cluster_type not in {ClusterType.MONOLITHIC, ClusterType.PREFILL}:
            return
        state = self._get_state_for_request(request)
        if state is None or not state.carryover_release_pending:
            return
        if self._is_decode_slice(request):
            return
        state.carryover_release_pending = False
        state.carryover_release_consumed_count += 1
        self._sync_request_state_snapshot(request, state)

    def _apply_long_liveness(self, scheduled_levels: Sequence[str]) -> None:
        if any(level == QLONG for level in scheduled_levels):
            self._sj2q_bounded_short_streak_counter = 0
            self._sj2q_bounded_forced_qlong_request_id = None
            self._sj2q_bounded_release_request_id = None
            return
        self._sj2q_bounded_short_streak_counter += sum(
            1 for level in scheduled_levels if level == QSHORT
        )
        self._sj2q_bounded_forced_qlong_request_id = None
        self._sj2q_bounded_release_request_id = None

    # ------------------------------------------------------------------
    # vLLM-v1 hook overrides
    # ------------------------------------------------------------------
    def add_request(self, request: Request) -> None:
        self._initialize_request_spec_decode_state(request)
        state = self._load_or_create_session_state(request)
        if self._cluster_type in {ClusterType.DECODE, ClusterType.DECODE_ATTN} and bool(
            getattr(request, "is_prefill_complete", False)
        ):
            self._record_queue_assignment(
                request,
                state,
                queue_level=state.queue_level,
                assigned_at=self._get_request_arrival_time(request),
            )
            self._waiting_requests.append(request)
            return

        self._classify_new_round(request, state)
        self._request_queue.append(request)

    def _get_sorted_waiting_queue(self) -> List[Request]:
        combined = list(self._preempted_requests) + list(self._request_queue)
        return self._sort_requests_by_queue_level(combined)

    def _build_decode_waiting_queue(self):
        return deque(self._sort_requests_by_queue_level(list(self._waiting_requests)))

    def _schedule_running_requests(
        self,
        token_budget: int,
        preempted_requests: List[Request],
    ):
        self._running_requests[:] = self._sort_requests_by_queue_level(
            self._running_requests
        )
        return super()._schedule_running_requests(token_budget, preempted_requests)

    def on_batch_end(self, batch: Batch) -> None:
        scheduled_levels: List[str] = []
        pending_release_before_schedule: Dict[int, bool] = {}
        for request in batch.requests:
            state = self._get_state_for_request(request)
            scheduled_levels.append(QSHORT if state is None else state.queue_level)
            pending_release_before_schedule[int(getattr(request, "id"))] = bool(
                state is not None and state.carryover_release_pending
            )

        super().on_batch_end(batch)
        completed_at = float(batch.completed_at)
        if completed_at < 0:
            raise ValueError(
                f"Batch completion time must be non-negative, got {completed_at}"
            )

        for request, executed_tokens in zip(batch.requests, batch.num_tokens):
            self._account_service_tokens(
                request,
                executed_tokens=int(executed_tokens),
                completed_at=completed_at,
            )
            self._mark_prefill_spill_if_needed(
                request,
                completed_at=completed_at,
            )
            if pending_release_before_schedule.get(int(getattr(request, "id")), False):
                self._consume_carryover_release_credit_if_used(request)

        self._apply_long_liveness(scheduled_levels)
