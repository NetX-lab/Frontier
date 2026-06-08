from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from frontier.entities.batch import Batch, Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


QH = "QH"
QL = "QL"


@dataclass
class SJ2QSessionState:
    session_id: int
    queue_level: str = QH
    long_history: bool = False
    served_new_tokens_total: int = 0
    boost_credit_tokens: int = 0
    prefill_release_boost_active: bool = False
    current_round_new_prompt_tokens: int = 0
    current_round_prefill_spilled: bool = False
    ql_wait_started_at: Optional[float] = None
    last_round_total_tokens: int = 0


class SJ2QFastServeLiteReplicaScheduler(VLLMv1EngineReplicaScheduler):
    """Simplified two-queue FastServe-style scheduler without final-round oracle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sj2q_session_states: Dict[int, SJ2QSessionState] = {}
        self._sj2q_long_round_threshold = int(
            getattr(self._config, "long_round_new_prompt_threshold", 2048)
        )
        self._sj2q_short_round_boost_threshold = int(
            getattr(self._config, "short_round_boost_threshold", 512)
        )
        self._sj2q_boost_credit_token_budget = int(
            getattr(self._config, "boost_credit_token_budget", 2048)
        )
        self._sj2q_enable_aging = bool(getattr(self._config, "enable_aging", False))
        self._sj2q_aging_wait_threshold_ms = float(
            getattr(self._config, "aging_wait_threshold_ms", 7.5)
        )
        self._sj2q_aging_boost_token_budget = int(
            getattr(self._config, "aging_boost_token_budget", 512)
        )

    # ---------------------------------------------------------------------
    # Oracle-prohibition overrides
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Session-state helpers
    # ---------------------------------------------------------------------
    def _get_session_id(self, request: Request) -> int:
        session_id = getattr(request, "session_id", None)
        if session_id is None:
            raise ValueError(
                "SJ2QFastServeLiteReplicaScheduler requires request.session_id."
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
        snapshot = getattr(request, "_sj2q_state_snapshot", None)
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            raise TypeError(
                "request._sj2q_state_snapshot must be a dict when present."
            )
        return snapshot

    def _load_or_create_session_state(self, request: Request) -> SJ2QSessionState:
        session_id = self._get_session_id(request)
        existing_state = self._sj2q_session_states.get(session_id)
        if existing_state is not None:
            return existing_state

        snapshot = self._get_request_state_snapshot(request)
        if snapshot is not None:
            state = SJ2QSessionState(
                session_id=session_id,
                queue_level=str(snapshot.get("queue_level", QH)),
                long_history=bool(snapshot.get("long_history", False)),
                served_new_tokens_total=int(
                    snapshot.get("served_new_tokens_total", 0)
                ),
                boost_credit_tokens=int(snapshot.get("boost_credit_tokens", 0)),
                prefill_release_boost_active=bool(
                    snapshot.get("prefill_release_boost_active", False)
                ),
                current_round_new_prompt_tokens=int(
                    snapshot.get("current_round_new_prompt_tokens", 0)
                ),
                current_round_prefill_spilled=bool(
                    snapshot.get("current_round_prefill_spilled", False)
                ),
                ql_wait_started_at=(
                    None
                    if snapshot.get("ql_wait_started_at") is None
                    else float(snapshot["ql_wait_started_at"])
                ),
                last_round_total_tokens=int(snapshot.get("last_round_total_tokens", 0)),
            )
        else:
            state = SJ2QSessionState(session_id=session_id)

        self._sj2q_session_states[session_id] = state
        return state

    def _sync_request_state_snapshot(
        self,
        request: Request,
        state: SJ2QSessionState,
    ) -> None:
        setattr(request, "_sj2q_state_snapshot", asdict(state))
        setattr(request, "_sj2q_queue_level", state.queue_level)

    def _compute_current_round_new_prompt_tokens(
        self,
        request: Request,
        state: SJ2QSessionState,
    ) -> int:
        return max(
            int(request.num_prefill_tokens) - int(state.last_round_total_tokens),
            0,
        )

    def _classify_new_round(
        self,
        request: Request,
        state: SJ2QSessionState,
    ) -> None:
        round_new_prompt_tokens = self._compute_current_round_new_prompt_tokens(
            request, state
        )
        state.current_round_new_prompt_tokens = int(round_new_prompt_tokens)
        state.current_round_prefill_spilled = False
        state.prefill_release_boost_active = False

        queue_level = QH
        boost_credit_tokens = 0
        ql_wait_started_at = None

        if round_new_prompt_tokens > self._sj2q_long_round_threshold:
            queue_level = QL
            state.long_history = True
            ql_wait_started_at = self._get_request_arrival_time(request)
        elif not state.long_history:
            queue_level = QH
        elif round_new_prompt_tokens <= self._sj2q_short_round_boost_threshold:
            queue_level = QH
            state.prefill_release_boost_active = True
        else:
            queue_level = QL
            ql_wait_started_at = self._get_request_arrival_time(request)

        state.queue_level = queue_level
        state.boost_credit_tokens = int(boost_credit_tokens)
        state.ql_wait_started_at = ql_wait_started_at
        self._sync_request_state_snapshot(request, state)

    def _get_state_for_request(
        self,
        request: Request,
    ) -> Optional[SJ2QSessionState]:
        try:
            session_id = self._get_session_id(request)
        except ValueError:
            return None
        return self._sj2q_session_states.get(session_id)

    def _queue_rank(self, request: Request) -> int:
        state = self._get_state_for_request(request)
        if state is None:
            return 0
        return 0 if state.queue_level == QH else 1

    def _qh_subrank(self, request: Request) -> int:
        state = self._get_state_for_request(request)
        if state is None or state.queue_level != QH:
            return 0

        if self._cluster_type in {ClusterType.DECODE, ClusterType.DECODE_ATTN}:
            return 1

        is_decode_slice = bool(getattr(request, "is_prefill_complete", False))
        if self._cluster_type == ClusterType.MONOLITHIC and is_decode_slice:
            return 1

        if (
            state.current_round_new_prompt_tokens
            <= self._sj2q_short_round_boost_threshold
        ):
            return 0
        return 2

    def _sort_requests_by_queue_level(
        self,
        requests: Sequence[Request],
    ) -> List[Request]:
        if self._scheduling_policy == "priority":
            return sorted(
                requests,
                key=lambda request: (
                    self._queue_rank(request),
                    self._qh_subrank(request),
                    int(getattr(request, "priority", 0)),
                    float(getattr(request, "arrived_at", 0.0)),
                ),
            )
        return sorted(
            requests,
            key=lambda request: (
                self._queue_rank(request),
                self._qh_subrank(request),
                float(getattr(request, "arrived_at", 0.0)),
            ),
        )

    def _promote_request_via_aging(
        self,
        request: Request,
        *,
        current_time: float,
    ) -> None:
        state = self._get_state_for_request(request)
        if state is None:
            return
        if state.queue_level != QL or not self._sj2q_enable_aging:
            return
        if state.ql_wait_started_at is None:
            return
        waited_ms = (float(current_time) - float(state.ql_wait_started_at)) * 1000.0
        if waited_ms < self._sj2q_aging_wait_threshold_ms:
            return
        state.queue_level = QH
        state.boost_credit_tokens = self._sj2q_aging_boost_token_budget
        state.ql_wait_started_at = None
        self._sync_request_state_snapshot(request, state)

    def _apply_aging_promotions(self, requests: Iterable[Request]) -> None:
        current_time = float(getattr(self, "_current_schedule_time", 0.0))
        for request in requests:
            self._promote_request_via_aging(request, current_time=current_time)

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
        state.long_history = True
        state.queue_level = QL
        state.prefill_release_boost_active = False
        state.boost_credit_tokens = 0
        state.ql_wait_started_at = float(completed_at)
        self._sync_request_state_snapshot(request, state)

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
        if state.prefill_release_boost_active and bool(request.is_prefill_complete):
            state.prefill_release_boost_active = False
            state.boost_credit_tokens = 0
            if state.long_history and not bool(request.completed):
                state.queue_level = QL
                state.ql_wait_started_at = float(completed_at)
            else:
                state.queue_level = QH
                state.ql_wait_started_at = None

        if bool(request.completed):
            state.last_round_total_tokens = int(request.total_tokens)

        self._sync_request_state_snapshot(request, state)

    # ---------------------------------------------------------------------
    # vLLM-v1 hook overrides
    # ---------------------------------------------------------------------
    def add_request(self, request: Request) -> None:
        self._initialize_request_spec_decode_state(request)
        state = self._load_or_create_session_state(request)
        if self._cluster_type in {ClusterType.DECODE, ClusterType.DECODE_ATTN} and bool(
            getattr(request, "is_prefill_complete", False)
        ):
            self._sync_request_state_snapshot(request, state)
        else:
            self._classify_new_round(request, state)

        if self._cluster_type in {ClusterType.DECODE, ClusterType.DECODE_ATTN}:
            self._waiting_requests.append(request)
            return
        self._request_queue.append(request)

    def _get_sorted_waiting_queue(self) -> List[Request]:
        combined = list(self._preempted_requests) + list(self._request_queue)
        self._apply_aging_promotions(combined)
        return self._sort_requests_by_queue_level(combined)

    def _build_decode_waiting_queue(self):
        ordered_requests = list(self._waiting_requests)
        self._apply_aging_promotions(ordered_requests)
        return deque(self._sort_requests_by_queue_level(ordered_requests))

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
