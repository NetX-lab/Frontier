from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Sequence
from collections import defaultdict

from frontier.entities.base_entity import BaseEntity
from frontier.logger import init_logger
from frontier.types import ClusterType

logger = init_logger(__name__)


@dataclass(frozen=True)
class RequestRoundPlan:
    num_prefill_tokens: int
    num_decode_tokens: int


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
        priority: int = 0,
        block_hash_ids: Optional[List[int]] = None,
        session_id: Optional[int] = None,
        cohort: Optional[str] = None,
        thinking_depth: int = 1,
        tool_call_latency: float = 0.001,
        thinking_round_plans: Optional[Sequence[RequestRoundPlan]] = None,
    ):
        resolved_round_plans = self._resolve_thinking_round_plans(
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            thinking_depth=thinking_depth,
            thinking_round_plans=thinking_round_plans,
        )
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._thinking_depth = thinking_depth
        self._tool_call_latency = tool_call_latency
        self._thinking_round_plans = tuple(resolved_round_plans)
        self._current_thinking_round_index = 0
        self._user_facing_num_prefill_tokens = resolved_round_plans[-1].num_prefill_tokens
        self._user_facing_num_decode_tokens = resolved_round_plans[-1].num_decode_tokens
        self._num_prefill_tokens = resolved_round_plans[0].num_prefill_tokens
        self._num_decode_tokens = resolved_round_plans[0].num_decode_tokens
        self._num_processed_tokens = num_processed_tokens
        self._priority = priority  # Lower value = higher priority (matches vLLM v1)
        self._block_hash_ids = (
            list(block_hash_ids) if block_hash_ids is not None else None
        )
        self._session_id = session_id
        self._cohort = cohort
        self._num_prefill_tokens_cached = 0

        # Multi-round scheduling support for pingpong-pipeline mode
        # Each cluster can have multiple scheduling rounds (for different layers)
        self._cluster_arrival_times: Dict[ClusterType, List[float]] = defaultdict(list)
        self._scheduled_at: Dict[ClusterType, List[float]] = defaultdict(list)
        self._execution_time: Dict[ClusterType, List[float]] = defaultdict(list)
        self._model_execution_time: Dict[ClusterType, List[float]] = defaultdict(list)
        self._scheduling_delay: Dict[ClusterType, List[float]] = defaultdict(list)
        self._preempted_time: Dict[ClusterType, List[float]] = defaultdict(list)

        # Preemption tracking attributes
        # Track how many times this request has been preempted in each cluster type
        self._preemption_count: Dict[ClusterType, int] = defaultdict(int)

        # Track the number of tokens completed (num_processed_tokens) at each preemption event
        # Only tracked for DECODE clusters (DECODE, DECODE_ATTN)
        # Each list contains the token count at each preemption event in chronological order
        self._tokens_at_preemption: Dict[ClusterType, List[int]] = defaultdict(list)

        # Per-cluster waiting time tracking for preemption analysis
        # Tracks cumulative time spent waiting in each cluster's queue
        self._cluster_waiting_time: Dict[ClusterType, float] = defaultdict(float)
        # Records the time when request entered the waiting queue (for current waiting period)
        self._queue_entry_time: Dict[ClusterType, float] = defaultdict(float)
        # Tracks whether request is currently waiting in each cluster's queue
        self._is_waiting: Dict[ClusterType, bool] = defaultdict(lambda: False)
        # Minimal round-class ledger for use case3 observability
        self._queue_entry_round_class: Dict[ClusterType, str] = {}
        self._queue_entry_round_number: Dict[ClusterType, int] = {}
        self._round_class_waiting_time: Dict[str, float] = defaultdict(float)
        self._round_class_service_time: Dict[str, float] = defaultdict(float)
        self._round_class_cluster_waiting_time: Dict[Tuple[str, ClusterType], float] = (
            defaultdict(float)
        )
        self._round_class_cluster_service_time: Dict[Tuple[str, ClusterType], float] = (
            defaultdict(float)
        )
        self._round_number_cluster_waiting_time: Dict[
            Tuple[int, ClusterType], float
        ] = defaultdict(float)
        self._round_class_preemption_count: Dict[str, int] = defaultdict(int)
        self._round_class_tokens_at_preemption: Dict[str, List[int]] = defaultdict(
            list
        )

        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0
        self._latest_iteration_round_class: Optional[str] = None
        self._latest_iteration_round_number: Optional[int] = None

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0
        self._runtime_epoch = 0
        # Monotonic execution epoch for stale-event detection. This advances
        # whenever the request re-enters any waiting queue, which covers initial
        # arrival, cross-cluster handoff, preemption recovery, and thinking-mode
        # requeue within the same round.
        self._execution_epoch = 0

        # Decode-attn cluster ping-pong pipeline state tracking
        self._current_decode_token_index = 1  # Start from 1 (token 0 generated by prefill)
        self._completed_layer_count = 0  # Number of layers completed for current decode token
        self._af_roundtrip_inflight = False  # True from A->F transfer start until F->A transfer end

        # Transfer time tracking for disaggregated architectures
        self._kv_cache_transfer_time: float = 0.0  # Total KV cache transfer time
        self._kv_cache_transfer_start_time: Optional[float] = None
        self._kv_cache_transfer_end_time: Optional[float] = None
        self._m2n_transfer_time_attn_to_ffn: float = 0.0  # Total A→F transfer time
        self._m2n_transfer_time_ffn_to_attn: float = 0.0  # Total F→A transfer time

        # First decode token tracking for TTFT metric
        self._first_decode_token_completed_at: float = 0.0
        # First token generated by decode process (second token in full forward).
        self._decode_first_token_completed_at: float = 0.0

        # Request-level DECODE_FFN residence timing (A→F arrival to F→A departure)
        self._decode_ffn_enter_time: Optional[float] = None
        self._decode_ffn_residence_time: float = 0.0

        # Metrics recording flags to prevent duplicate recording in disaggregated mode
        self._metrics_recorded: bool = False  # For arrival metrics (histogram)
        self._end_metrics_recorded: bool = False  # For completion metrics (time distributions)

        # Speculative decoding runtime state (Phase 1).
        self._spec_decode_enabled: bool = False
        self._spec_method: Optional[str] = None
        self._spec_method_uses_lookahead_slots: bool = False
        self._spec_num_speculative_tokens: int = 0
        self._spec_next_planned_draft_tokens: int = 0
        self._spec_current_verify_tokens: int = 1
        self._spec_last_committed_tokens: int = 1
        self._spec_total_iterations: int = 0
        self._spec_total_accepted_drafts: int = 0
        self._spec_total_rejected_drafts: int = 0
        self._spec_total_committed_tokens: int = 0
        self._spec_post_first_service_delay: float = 0.0

        self._thinking_home_cluster_type: Optional[ClusterType] = None
        self._thinking_home_replica_id: Optional[int] = None
        self._thinking_home_dp_id: Optional[int] = None
        self._pending_thinking_requeue: bool = False
        self._thinking_tool_wait_started_at: Optional[float] = None
        self._thinking_time_total: float = 0.0
        self._tool_call_time_total: float = 0.0
        self._completed_thinking_rounds: int = 0

    @staticmethod
    def _resolve_thinking_round_plans(
        *,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        thinking_depth: int,
        thinking_round_plans: Optional[Sequence[RequestRoundPlan]],
    ) -> List[RequestRoundPlan]:
        if thinking_depth < 1:
            raise ValueError(f"thinking_depth must be >= 1, got {thinking_depth}")
        if thinking_round_plans is None:
            if thinking_depth != 1:
                raise ValueError(
                    "thinking_round_plans must be provided when thinking_depth > 1."
                )
            return [RequestRoundPlan(num_prefill_tokens, num_decode_tokens)]

        resolved_round_plans = list(thinking_round_plans)
        if len(resolved_round_plans) != thinking_depth:
            raise ValueError(
                "thinking_round_plans length must match thinking_depth, "
                f"got len={len(resolved_round_plans)} depth={thinking_depth}"
            )
        final_round_plan = resolved_round_plans[-1]
        if (
            final_round_plan.num_prefill_tokens != num_prefill_tokens
            or final_round_plan.num_decode_tokens != num_decode_tokens
        ):
            raise ValueError(
                "Final thinking round plan must match the user-facing request lengths."
            )
        return resolved_round_plans

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        # Return the time it was first scheduled in any cluster
        if not self._scheduled_at:
            return 0
        # Get the first scheduling time from all clusters
        first_times = []
        for cluster_times in self._scheduled_at.values():
            if cluster_times:
                first_times.append(cluster_times[0])
        return min(first_times) if first_times else 0

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        # Return the scheduling delay of the first time it was scheduled
        if not self._scheduling_delay:
            return 0
        # Get the first scheduling delay from all clusters
        for cluster_delays in self._scheduling_delay.values():
            if cluster_delays:
                return cluster_delays[0]
        return 0

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        # Deprecated: preemption delay is now modeled as waiting-queue time.
        # Keep this property for backward compatibility with legacy metrics fields.
        return 0.0

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        # Sum all execution times across all clusters and rounds
        total_execution = 0
        for cluster_times in self._execution_time.values():
            total_execution += sum(cluster_times)
        return total_execution

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self.execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        # Sum all model execution times across all clusters and rounds
        total_model_execution = 0
        for cluster_times in self._model_execution_time.values():
            total_model_execution += sum(cluster_times)
        return total_model_execution

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self.model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def thinking_depth(self) -> int:
        return self._thinking_depth

    @property
    def tool_call_latency(self) -> float:
        return self._tool_call_latency

    @property
    def current_thinking_round_index(self) -> int:
        return self._current_thinking_round_index

    @property
    def current_thinking_round_number(self) -> int:
        return self._current_thinking_round_index + 1

    @property
    def thinking_round_plans(self) -> Tuple[RequestRoundPlan, ...]:
        return self._thinking_round_plans

    @property
    def is_thinking_mode_enabled(self) -> bool:
        return self._thinking_depth > 1

    @property
    def is_final_thinking_round(self) -> bool:
        return self._current_thinking_round_index == self._thinking_depth - 1

    @property
    def current_round_class(self) -> str:
        return "final" if self.is_final_thinking_round else "hidden"

    @property
    def user_facing_num_prefill_tokens(self) -> int:
        return self._user_facing_num_prefill_tokens

    @property
    def user_facing_num_decode_tokens(self) -> int:
        return self._user_facing_num_decode_tokens

    @property
    def user_facing_total_tokens(self) -> int:
        return self._user_facing_num_prefill_tokens + self._user_facing_num_decode_tokens

    @property
    def user_facing_pd_ratio(self) -> float:
        return self._user_facing_num_prefill_tokens / self._user_facing_num_decode_tokens

    @property
    def thinking_home_cluster_type(self) -> Optional[ClusterType]:
        return self._thinking_home_cluster_type

    @property
    def thinking_home_replica_id(self) -> Optional[int]:
        return self._thinking_home_replica_id

    @property
    def thinking_home_dp_id(self) -> Optional[int]:
        return self._thinking_home_dp_id

    @property
    def pending_thinking_requeue(self) -> bool:
        return self._pending_thinking_requeue

    @property
    def thinking_time_total(self) -> float:
        return self._thinking_time_total

    @property
    def tool_call_time_total(self) -> float:
        return self._tool_call_time_total

    @property
    def completed_thinking_rounds(self) -> int:
        return self._completed_thinking_rounds

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def runtime_epoch(self) -> int:
        return self._runtime_epoch

    def advance_runtime_epoch(self) -> int:
        self._runtime_epoch += 1
        return self._runtime_epoch

    @property
    def execution_epoch(self) -> int:
        return self._execution_epoch

    @property
    def priority(self) -> int:
        """
        Request priority value.
        Lower value = higher priority (processed first).
        Default is 0 (highest priority).
        Matches vLLM v1 semantics.
        """
        return self._priority

    def set_priority(self, priority: int) -> None:
        self._priority = priority

    @property
    def block_hash_ids(self) -> Optional[List[int]]:
        return self._block_hash_ids

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    @property
    def cohort(self) -> Optional[str]:
        return self._cohort

    @property
    def num_prefill_tokens_cached(self) -> int:
        return self._num_prefill_tokens_cached

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    def on_cache_hit(self, num_tokens_cached: int) -> None:
        if self._scheduled:
            raise ValueError(f"Request {self._id} already scheduled.")
        if self._num_processed_tokens != 0:
            raise ValueError(
                f"Request {self._id} already has processed tokens: {self._num_processed_tokens}"
            )
        if num_tokens_cached < 0 or num_tokens_cached > self._num_prefill_tokens:
            raise ValueError(
                f"Invalid cached token count for request {self._id}: {num_tokens_cached}"
            )
        self._num_processed_tokens = int(num_tokens_cached)
        self._num_prefill_tokens_cached = int(num_tokens_cached)

    # Preemption tracking accessor methods
    def get_preemption_count(self, cluster_type: ClusterType) -> int:
        """
        Get the number of times this request has been preempted in a specific cluster.

        Args:
            cluster_type: The cluster type to query

        Returns:
            Number of preemption events in the specified cluster
        """
        return self._preemption_count.get(cluster_type, 0)

    def get_total_preemption_count(self) -> int:
        """
        Get the total number of times this request has been preempted across all clusters.

        Returns:
            Total number of preemption events across all clusters
        """
        return sum(self._preemption_count.values())

    def get_round_class_preemption_count(self, round_class: str) -> int:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return int(self._round_class_preemption_count.get(round_class, 0))

    def get_tokens_at_preemption(self, cluster_type: ClusterType) -> List[int]:
        """
        Get the list of token counts at each preemption event for a specific cluster.

        This is only tracked for DECODE clusters (DECODE, DECODE_ATTN).
        Each element in the list represents the number of tokens that had been
        completed (num_processed_tokens) when a preemption event occurred.

        Args:
            cluster_type: The cluster type to query

        Returns:
            List of token counts at each preemption event (chronological order)
        """
        return self._tokens_at_preemption.get(cluster_type, [])

    def get_round_class_tokens_at_preemption(self, round_class: str) -> List[int]:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return list(self._round_class_tokens_at_preemption.get(round_class, []))

    # Waiting time tracking accessor methods
    def get_cluster_waiting_time(self, cluster_type: ClusterType) -> float:
        """
        Get cumulative waiting time for a specific cluster.

        This returns the total time this request has spent waiting in the
        specified cluster's queue, including time after preemption events.

        Args:
            cluster_type: The cluster type to query

        Returns:
            Cumulative waiting time in the specified cluster (in simulation time units)
        """
        return self._cluster_waiting_time.get(cluster_type, 0.0)

    def get_total_waiting_time(self) -> float:
        """
        Get total waiting time across all clusters.

        This returns the sum of waiting times across all cluster types,
        providing a complete picture of queue waiting time for this request.

        Returns:
            Sum of waiting times across all clusters (in simulation time units)
        """
        return float(sum(self._cluster_waiting_time.values()))

    def get_round_class_waiting_time(self, round_class: str) -> float:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return float(self._round_class_waiting_time.get(round_class, 0.0))

    def get_round_class_service_time(self, round_class: str) -> float:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return float(self._round_class_service_time.get(round_class, 0.0))

    def get_round_class_cluster_waiting_time(
        self, round_class: str, cluster_type: ClusterType
    ) -> float:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return float(
            self._round_class_cluster_waiting_time.get((round_class, cluster_type), 0.0)
        )

    def get_round_class_cluster_service_time(
        self, round_class: str, cluster_type: ClusterType
    ) -> float:
        if round_class not in {"hidden", "final"}:
            raise ValueError(f"Unsupported round_class={round_class!r}")
        return float(
            self._round_class_cluster_service_time.get((round_class, cluster_type), 0.0)
        )

    def get_round_number_cluster_waiting_time(
        self, round_number: int, cluster_type: ClusterType
    ) -> float:
        if round_number < 1:
            raise ValueError(f"round_number must be >= 1, got={round_number}")
        return float(
            self._round_number_cluster_waiting_time.get((round_number, cluster_type), 0.0)
        )

    def get_round_numbers_cluster_waiting_time(
        self, round_numbers: Sequence[int], cluster_type: ClusterType
    ) -> float:
        return float(
            sum(
                self.get_round_number_cluster_waiting_time(
                    round_number=round_number,
                    cluster_type=cluster_type,
                )
                for round_number in round_numbers
            )
        )

    # not include the second decode token (if we call in pd+af m2n arrive scenario)
    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    # A request is in ongoing decoding if prefill is complete.
    # This is used by batch.all_requests_ongoing_decoding to determine
    # whether to use decode-phase token counting (1 token per request)
    # or prefill-phase token counting (num_prefill_tokens).
    #
    # In PD+AF disaggregation mode:
    # - After PREFILL completes: num_processed_tokens = num_prefill_tokens
    # - The first decode token is granted by DECODE_ATTN via GlobalBatchEndEvent
    # - Therefore, ongoing_decoding should be True when is_prefill_complete is True
    @property
    def ongoing_decoding(self) -> bool:
        return self._is_prefill_complete

    @property
    def spec_decode_enabled(self) -> bool:
        return self._spec_decode_enabled

    @property
    def spec_method(self) -> Optional[str]:
        return self._spec_method

    @property
    def spec_method_is_target_embedded_mtp(self) -> bool:
        from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_method

        return bool(self._spec_method is not None and is_target_embedded_mtp_method(self._spec_method))

    @property
    def spec_method_uses_lookahead_slots(self) -> bool:
        return self._spec_method_uses_lookahead_slots

    @property
    def spec_num_speculative_tokens(self) -> int:
        return self._spec_num_speculative_tokens

    @property
    def spec_next_planned_draft_tokens(self) -> int:
        return self._spec_next_planned_draft_tokens

    @property
    def spec_current_verify_tokens(self) -> int:
        return self._spec_current_verify_tokens

    @property
    def spec_total_iterations(self) -> int:
        return self._spec_total_iterations

    @property
    def spec_total_accepted_drafts(self) -> int:
        return self._spec_total_accepted_drafts

    @property
    def spec_total_rejected_drafts(self) -> int:
        return self._spec_total_rejected_drafts

    @property
    def spec_total_committed_tokens(self) -> int:
        return self._spec_total_committed_tokens

    @property
    def remaining_decode_tokens(self) -> int:
        return max(self._num_decode_tokens - self.num_processed_decode_tokens, 0)

    def initialize_spec_decode_state(
        self,
        *,
        enabled: bool,
        method: Optional[str] = None,
        num_speculative_tokens: int = 0,
        method_uses_lookahead_slots: bool = False,
    ) -> None:
        self._spec_decode_enabled = bool(enabled)
        self._spec_method = method if enabled else None
        self._spec_method_uses_lookahead_slots = (
            bool(method_uses_lookahead_slots) if enabled else False
        )
        self._spec_num_speculative_tokens = (
            int(num_speculative_tokens) if enabled else 0
        )
        self._spec_next_planned_draft_tokens = 0
        self._spec_current_verify_tokens = 1
        self._spec_last_committed_tokens = 1
        self._spec_total_iterations = 0
        self._spec_total_accepted_drafts = 0
        self._spec_total_rejected_drafts = 0
        self._spec_total_committed_tokens = 0

    def set_spec_next_planned_draft_tokens(self, planned_draft_tokens: int) -> None:
        planned = int(planned_draft_tokens)
        if planned < 0:
            raise ValueError(
                f"planned_draft_tokens must be >= 0, got={planned_draft_tokens}"
            )
        self._spec_next_planned_draft_tokens = planned

    def record_spec_decode_iteration(
        self,
        *,
        verify_tokens: int,
        accepted_drafts: int,
        rejected_drafts: int,
        committed_tokens: int,
    ) -> None:
        if verify_tokens < 0:
            raise ValueError(f"verify_tokens must be >= 0, got={verify_tokens}")
        if accepted_drafts < 0:
            raise ValueError(
                f"accepted_drafts must be >= 0, got={accepted_drafts}"
            )
        if rejected_drafts < 0:
            raise ValueError(
                f"rejected_drafts must be >= 0, got={rejected_drafts}"
            )
        if committed_tokens < 0:
            raise ValueError(
                f"committed_tokens must be >= 0, got={committed_tokens}"
            )
        self._spec_current_verify_tokens = int(verify_tokens)
        self._spec_last_committed_tokens = int(committed_tokens)
        self._spec_total_iterations += 1
        self._spec_total_accepted_drafts += int(accepted_drafts)
        self._spec_total_rejected_drafts += int(rejected_drafts)
        self._spec_total_committed_tokens += int(committed_tokens)

    def reset_spec_verify_state_after_batch_end(self) -> None:
        self._spec_current_verify_tokens = 1

    def set_spec_current_verify_tokens_for_reservation(self, verify_tokens: int) -> None:
        value = int(verify_tokens)
        if value <= 0:
            raise ValueError(f"verify_tokens must be > 0, got={verify_tokens}")
        self._spec_current_verify_tokens = value

    @property
    def early_decoding_on_first_layer(self) -> bool:
        # Early decoding on first layer: first decode token, layer 0
        # In PD+AF mode: num_processed_tokens = num_prefill_tokens when first decode starts
        # After first decode token completes: num_processed_tokens = num_prefill_tokens + 1
        return self._is_prefill_complete and self._current_decode_token_index <= 1 and self._completed_layer_count == 0

    @property
    def current_decode_token_index(self) -> int:
        """Current decode token index being processed (1-based, token 0 generated by prefill)"""
        return self._current_decode_token_index

    @property
    def completed_layer_count(self) -> int:
        """Number of layers completed for current decode token"""
        return self._completed_layer_count

    @property
    def af_roundtrip_inflight(self) -> bool:
        """Whether the request is in-flight across DECODE_ATTN <-> DECODE_FFN."""
        return self._af_roundtrip_inflight

    def mb_on_step_layer_count_increment(self, num_layers_completed: int = 1) -> None:
        self._completed_layer_count += num_layers_completed

    @property
    def kv_cache_transfer_time(self) -> float:
        """Total KV cache transfer time for this request."""
        return self._kv_cache_transfer_time

    @property
    def kv_cache_transfer_start_time(self) -> Optional[float]:
        """Request-level KV transfer contract start timestamp in seconds."""
        return self._kv_cache_transfer_start_time

    @property
    def kv_cache_transfer_end_time(self) -> Optional[float]:
        """Request-level KV transfer contract end timestamp in seconds."""
        return self._kv_cache_transfer_end_time

    @property
    def total_m2n_transfer_time(self) -> float:
        """Total M2N transfer time (A→F + F→A) for this request."""
        return self._m2n_transfer_time_attn_to_ffn + self._m2n_transfer_time_ffn_to_attn

    @property
    def m2n_transfer_time_attn_to_ffn(self) -> float:
        """Total A→F transfer time for this request."""
        return self._m2n_transfer_time_attn_to_ffn

    @property
    def m2n_transfer_time_ffn_to_attn(self) -> float:
        """Total F→A transfer time for this request."""
        return self._m2n_transfer_time_ffn_to_attn

    @property
    def first_decode_token_completed_at(self) -> float:
        """Time when first decode token was completed."""
        return self._first_decode_token_completed_at

    @property
    def spec_post_first_service_delay(self) -> float:
        """Request-local post-first-token speculative service delay."""
        return self._spec_post_first_service_delay

    def add_spec_post_first_service_delay(self, delay: float) -> None:
        """Accumulate speculative work that should affect TPOT/E2E, not TTFT."""
        delay_value = float(delay)
        if delay_value < 0.0:
            raise ValueError(
                f"spec post-first service delay must be >= 0, got={delay_value}"
            )
        if delay_value == 0.0:
            return
        self._spec_post_first_service_delay += delay_value

    @property
    def decode_ffn_residence_time(self) -> float:
        """
        Total request-level residence time in DECODE_FFN cluster.

        This accumulates time windows defined by:
        - Entry: A→F transfer end (request arrives at DECODE_FFN)
        - Exit:  F→A transfer start (request leaves DECODE_FFN)
        """
        return self._decode_ffn_residence_time

    @property
    def decode_first_token_completed_at(self) -> float:
        """Time when decode process generated its first token."""
        return self._decode_first_token_completed_at

    @property
    def ttft(self) -> float:
        """
        Time To First Token (TTFT).
        Measured from request arrival to prefill completion.
        Returns 0 if prefill hasn't completed yet.
        """
        if self._prefill_completed_at == 0:
            return 0
        return self._prefill_completed_at - self._arrived_at

    @property
    def tpot(self) -> float:
        """
        Time Per Output Token (TPOT).
        Average time per output token excluding the first token.
        Returns 0 if there's only one or no decode tokens.
        """
        if self._num_decode_tokens <= 1 or self._first_decode_token_completed_at == 0:
            return 0
        total_decode_time = self._completed_at - self._first_decode_token_completed_at
        return total_decode_time / (self._num_decode_tokens - 1)

    def on_kv_cache_transfer_start(self, transfer_start_time: float) -> None:
        """Record the earliest request-level KV transfer start timestamp."""
        start_time = float(transfer_start_time)
        if self._kv_cache_transfer_start_time is None:
            self._kv_cache_transfer_start_time = start_time
            return
        self._kv_cache_transfer_start_time = min(
            self._kv_cache_transfer_start_time,
            start_time,
        )

    def on_kv_cache_transfer_complete(
        self,
        transfer_end_time: float,
        transfer_time: float,
    ) -> None:
        """
        Record KV cache transfer time for this request.

        Args:
            transfer_end_time: End timestamp of the request-level transfer window in
                seconds.
            transfer_time: Duration of the KV cache transfer in seconds
        """
        end_time = float(transfer_end_time)
        if self._kv_cache_transfer_end_time is None:
            self._kv_cache_transfer_end_time = end_time
        else:
            self._kv_cache_transfer_end_time = max(
                self._kv_cache_transfer_end_time,
                end_time,
            )
        self._kv_cache_transfer_time += transfer_time

    def on_m2n_transfer_complete(self, transfer_time: float, is_attn_to_ffn: bool) -> None:
        """
        Record M2N transfer time for this request.

        Args:
            transfer_time: Duration of the M2N transfer in seconds
            is_attn_to_ffn: True if this is an A→F transfer, False if F→A
        """
        if is_attn_to_ffn:
            self._m2n_transfer_time_attn_to_ffn += transfer_time
        else:
            self._m2n_transfer_time_ffn_to_attn += transfer_time

    def mark_first_decode_token_complete(self, time: float) -> None:
        """
        Mark the completion of the first decode token.
        This is used for TTFT calculation.

        Args:
            time: Time when the first decode token was completed
        """
        if self._first_decode_token_completed_at != 0:
            return
        if self._decode_first_token_completed_at > 0:
            # Keep timestamp ordering invariant:
            # first_decode_token_completed_at <= decode_first_token_completed_at.
            self._first_decode_token_completed_at = min(
                time, self._decode_first_token_completed_at
            )
            return
        self._first_decode_token_completed_at = time

    def record_preemption(self, cluster_type: ClusterType, num_tokens_completed: int) -> None:
        """
        Record a preemption event for this request.

        This method should be called by the scheduler when preempting a request.
        It increments the preemption count for the specified cluster and records
        the number of tokens that had been completed at the time of preemption.

        Args:
            cluster_type: The cluster type where preemption occurred
            num_tokens_completed: Number of tokens completed (num_processed_tokens) at preemption time
        """
        # Increment preemption count for this cluster
        self._preemption_count[cluster_type] += 1
        round_class = self.current_round_class
        self._round_class_preemption_count[round_class] += 1

        # Record tokens at preemption for DECODE clusters only
        # This is meaningful for DECODE (PD mode) and DECODE_ATTN (PD+AF mode)
        if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]:
            self._tokens_at_preemption[cluster_type].append(num_tokens_completed)
            self._round_class_tokens_at_preemption[round_class].append(
                num_tokens_completed
            )

    def is_finished_for_cluster(self, cluster_type: ClusterType) -> bool:
        """
        Checks if the request has completed its lifecycle for a specific cluster.
        """
        if cluster_type == ClusterType.PREFILL:
            return self.is_prefill_complete

        # For decode clusters or monolithic mode, completion is global.
        return self.completed

    def on_arrival(self, time: float, cluster_type: ClusterType) -> None:
        assert time >= 0, f"Invalid arrival time: {time}"
        # Delegate to on_enter_waiting_queue for consistent waiting time tracking
        # This ensures initial arrival also triggers waiting time tracking
        self.on_enter_waiting_queue(time, cluster_type)

    def on_enter_waiting_queue(self, time: float, cluster_type: ClusterType) -> None:
        """
        Record when request enters waiting queue (initial arrival or after preemption).

        This method should be called:
        1. When a request first arrives at a cluster (via on_arrival integration)
        2. When a request is preempted and re-enters the waiting queue

        Args:
            time: Current simulation time
            cluster_type: The cluster type where the request is entering the queue
        """
        assert time >= 0, f"Invalid queue entry time: {time}"
        self._execution_epoch += 1
        self._queue_entry_time[cluster_type] = time
        self._is_waiting[cluster_type] = True
        self._queue_entry_round_class[cluster_type] = self.current_round_class
        self._queue_entry_round_number[cluster_type] = self.current_thinking_round_number
        # Append to arrival times list (not replace!) for multi-round tracking
        # This ensures scheduling_delay calculation uses the most recent arrival time
        self._cluster_arrival_times[cluster_type].append(time)

    def on_leave_waiting_queue(self, time: float, cluster_type: ClusterType) -> None:
        """
        Record when request leaves waiting queue (is scheduled).
        Updates cumulative waiting time for the cluster.

        Args:
            time: Current simulation time
            cluster_type: The cluster type where the request is being scheduled
        """
        assert time >= 0, f"Invalid queue leave time: {time}"
        if self._is_waiting.get(cluster_type, False):
            entry_time = self._queue_entry_time.get(cluster_type, time)
            waiting_duration = time - entry_time
            self._cluster_waiting_time[cluster_type] += waiting_duration
            round_class = self._queue_entry_round_class.get(
                cluster_type, self.current_round_class
            )
            round_number = self._queue_entry_round_number.get(
                cluster_type, self.current_thinking_round_number
            )
            self._round_class_waiting_time[round_class] += max(0.0, waiting_duration)
            self._round_class_cluster_waiting_time[(round_class, cluster_type)] += max(
                0.0, waiting_duration
            )
            self._round_number_cluster_waiting_time[(round_number, cluster_type)] += max(
                0.0, waiting_duration
            )
            self._is_waiting[cluster_type] = False
        # If not waiting, this is a no-op (idempotent behavior)

    def set_arrived_at(self, time: float) -> None:
        assert time >= 0, f"Invalid arrival time: {time}"
        self._arrived_at = time

    # Triggered only when the request is scheduled.
    def on_batch_schedule(
        self,
        time: float,
        cluster_type: ClusterType,
    ) -> None:
        assert time >= 0, f"Invalid scheduling time: {time}"
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_round_class = self.current_round_class
        self._latest_iteration_round_number = self.current_thinking_round_number
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        # Record this scheduling event for multi-round support
        self._scheduled_at[cluster_type].append(time)

        # Calculate scheduling delay based on the most recent arrival time for this cluster
        cluster_arrival_times = self._cluster_arrival_times.get(cluster_type, [])
        if cluster_arrival_times:
            # Use the most recent arrival time for this cluster
            cluster_arrival_time = cluster_arrival_times[-1]
            scheduling_delay = time - cluster_arrival_time
            self._scheduling_delay[cluster_type].append(scheduling_delay)
        else:
            # No arrival time recorded for this cluster (e.g., offline mode)
            # Use global arrival time as fallback
            scheduling_delay = time - self._arrived_at
            self._scheduling_delay[cluster_type].append(scheduling_delay)

        # Mark as scheduled overall if it's the first time ever
        # TODO: should we replace _scheduled as _scheduled[cluster_type]? why?
        if not self._scheduled:
            self._scheduled = True

    # decode-attn: state update for token completion is triggered only at GlobalBatchEndEvent
    # decode-ffn: middle stage, no token-level state mutation here
    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
        cluster_type: ClusterType,
    ) -> None:
        # NOTE:
        # - For PREFILL/non-decode paths, num_tokens_processed means "scheduled tokens
        #   executed in this callback".
        # - For DECODE/MONOLITHIC decode rollout in speculative mode, callers pass
        #   committed tokens (may be > 1) via batch.spec_decode_metadata.
        # ISSUE-005 FIX: Skip if request is already completed to prevent token count overflow.
        # This can happen when a batch contains multiple requests and some complete before others.
        # The batch continues A↔F ping-pong for remaining requests, but GlobalBatchEndEvent
        # still calls on_batch_end() for all requests in the batch.
        if self._completed:
            logger.debug(
                f"[TOKEN-ROLLOUT][SKIP] req={self._id} already completed, "
                f"skipping on_batch_end (cluster={cluster_type.name})"
            )
            return

        if num_tokens_processed < 0:
            raise ValueError(
                f"num_tokens_processed must be >= 0, got={num_tokens_processed}"
            )
        if num_tokens_processed > self.total_tokens:
            raise ValueError(
                f"Invalid number of tokens processed: {num_tokens_processed}, "
                f"total_tokens={self.total_tokens}"
            )
        decode_tokens_before = self.num_processed_decode_tokens

        # Track whether prefill completion happened in this callback.
        prefill_completed_this_call = False

        # For non-decode clusters (e.g., PREFILL), advance by processed tokens.
        # Exclude DECODE (PD mode), DECODE_ATTN (PD+AF mode), DECODE_FFN (PD+AF mode),
        # and MONOLITHIC because those paths interpret num_tokens_processed as committed
        # decode tokens rather than raw scheduled prefill width.
        if cluster_type not in [ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, ClusterType.MONOLITHIC]:
            self._num_processed_tokens += num_tokens_processed

        # MONOLITHIC cluster: prefill callbacks still pass prefill width, while decode
        # callbacks pass committed decode tokens for the current rollout.
        if cluster_type == ClusterType.MONOLITHIC:
            if not self._is_prefill_complete:
                # Prefill phase: add prefill tokens
                self._num_processed_tokens += num_tokens_processed
            # Decode phase token counting is handled in the MONOLITHIC block below

        # Record iteration completion time
        self._latest_iteration_completed_at = time
        if self._scheduled or self._latest_iteration_scheduled_at > 0:
            service_duration = time - self._latest_iteration_scheduled_at
            round_class = (
                self._latest_iteration_round_class
                if self._latest_iteration_round_class is not None
                else self.current_round_class
            )
            self._round_class_service_time[round_class] += max(
                0.0, service_duration
            )
            self._round_class_cluster_service_time[(round_class, cluster_type)] += max(
                0.0, service_duration
            )

        assert self._num_processed_tokens <= self.total_tokens

        # PREFILL cluster: when prefill completes, mark completion
        # Note: In disaggregated mode (PD or PD+AF), the first decode token is NOT
        # granted here. It will be granted by DECODE/DECODE_ATTN cluster via
        # GlobalBatchEndEvent. In monolithic mode, this block is not reached since
        # cluster_type would be MONOLITHIC.
        if self._num_processed_tokens == self._num_prefill_tokens and not self._is_prefill_complete:
            self._is_prefill_complete = True
            prefill_completed_this_call = True

            # Record prefill completion time only once
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # DECODE cluster (PD mode) or DECODE_ATTN cluster (PD+AF mode):
        # GlobalBatchEndEvent semantics guarantee this is called when a decode step
        # completes. In speculative mode this step may commit >1 token.
        if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]:
            committed_tokens = int(num_tokens_processed)
            if committed_tokens < 0:
                raise ValueError(
                    f"Decode rollout requires committed_tokens >= 0, got={committed_tokens}"
                )
            if committed_tokens == 0:
                if not self._spec_decode_enabled:
                    raise ValueError(
                        "Decode rollout requires committed_tokens > 0 when "
                        f"speculative decoding is disabled, got={committed_tokens}"
                    )
                self._spec_last_committed_tokens = 0
                logger.info(
                    f"[TOKEN-ROLLOUT][ZERO-COMMIT] req={self._id} "
                    f"cluster={cluster_type.name} decode_progress="
                    f"{self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                    f"total_progress={self._num_processed_tokens}/{self.total_tokens} "
                    "accepted_drafts=0"
                )
            else:
                decode_remaining_before = self.remaining_decode_tokens
                if committed_tokens > decode_remaining_before:
                    raise ValueError(
                        "Committed decode tokens exceed remaining decode tokens: "
                        f"committed={committed_tokens}, remaining={decode_remaining_before}, "
                        f"request_id={self._id}"
                    )
                self._num_processed_tokens += committed_tokens
                # Reset layer count for next token and advance token index
                self._completed_layer_count = 0
                self._current_decode_token_index += committed_tokens
                self._spec_last_committed_tokens = committed_tokens
                logger.info(
                    f"[TOKEN-ROLLOUT] req={self._id} cluster={cluster_type.name} "
                    f"decode_progress={self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                    f"total_progress={self._num_processed_tokens}/{self.total_tokens} "
                    f"committed_tokens={committed_tokens} "
                    f"num_prefill_tokens={self._num_prefill_tokens} "
                    f"num_decode_tokens={self._num_decode_tokens} "
                )

        # MONOLITHIC cluster: rollout decode token whenever prefill is already
        # complete (including the callback that completes prefill). This matches
        # vLLM V1 behavior where decode progression starts at the final prefill
        # completion boundary.
        if cluster_type == ClusterType.MONOLITHIC and self._is_prefill_complete:
            if self._num_processed_tokens < self.total_tokens:
                # Keep MONOLITHIC prefill boundary rollout parity: the callback that
                # just finished prefill grants one decode-progress token. Subsequent
                # decode callbacks consume committed_tokens from scheduler metadata.
                committed_tokens = (
                    1 if prefill_completed_this_call else int(num_tokens_processed)
                )
                if committed_tokens < 0:
                    raise ValueError(
                        "Monolithic decode rollout requires committed_tokens >= 0, "
                        f"got={committed_tokens}"
                    )
                if committed_tokens == 0:
                    if not self._spec_decode_enabled or prefill_completed_this_call:
                        raise ValueError(
                            "Monolithic decode rollout requires committed_tokens > 0 "
                            "unless speculative decoding is already active after the "
                            f"prefill boundary, got={committed_tokens}"
                        )
                    self._spec_last_committed_tokens = 0
                    logger.info(
                        f"[TOKEN-ROLLOUT][ZERO-COMMIT] req={self._id} "
                        f"cluster={cluster_type.name} decode_progress="
                        f"{self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                        f"total_progress={self._num_processed_tokens}/{self.total_tokens}"
                    )
                else:
                    decode_remaining_before = self.total_tokens - self._num_processed_tokens
                    if committed_tokens > decode_remaining_before:
                        raise ValueError(
                            "Monolithic committed tokens exceed remaining tokens: "
                            f"committed={committed_tokens}, remaining={decode_remaining_before}, "
                            f"request_id={self._id}"
                        )
                    self._num_processed_tokens += committed_tokens
                    # Reset layer count for next token and advance token index.
                    self._completed_layer_count = 0
                    self._current_decode_token_index += committed_tokens
                    self._spec_last_committed_tokens = committed_tokens
                    if prefill_completed_this_call:
                        logger.info(
                            f"[TOKEN-ROLLOUT] req={self._id} cluster={cluster_type.name} "
                            f"decode_progress={self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                            f"total_progress={self._num_processed_tokens}/{self.total_tokens} "
                            f"committed_tokens={committed_tokens} "
                            f"num_prefill_tokens={self._num_prefill_tokens} "
                            f"num_decode_tokens={self._num_decode_tokens} "
                            "(first decode token after prefill)"
                        )
                    else:
                        logger.info(
                            f"[TOKEN-ROLLOUT] req={self._id} cluster={cluster_type.name} "
                            f"decode_progress={self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                            f"total_progress={self._num_processed_tokens}/{self.total_tokens} "
                            f"committed_tokens={committed_tokens} "
                            f"num_prefill_tokens={self._num_prefill_tokens} "
                            f"num_decode_tokens={self._num_decode_tokens} "
                        )

        # Check if request is completed
        decode_tokens_after = self.num_processed_decode_tokens
        if self._first_decode_token_completed_at == 0:
            first_decode_threshold = None
            if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]:
                first_decode_threshold = 1
            elif cluster_type == ClusterType.MONOLITHIC:
                # MONOLITHIC advances decode progress at the prefill-complete boundary.
                # For online vLLM parity, that first rollout corresponds to the first
                # observable output token and must seed TTFT / TPOT bookkeeping.
                first_decode_threshold = 1

            if (
                first_decode_threshold is not None
                and decode_tokens_before < first_decode_threshold <= decode_tokens_after
            ):
                self.mark_first_decode_token_complete(time)

        if (
            self._decode_first_token_completed_at == 0
            and decode_tokens_before < 2 <= decode_tokens_after
        ):
            self._decode_first_token_completed_at = time
            if (
                self._first_decode_token_completed_at == 0
                or self._first_decode_token_completed_at
                > self._decode_first_token_completed_at
            ):
                self._first_decode_token_completed_at = (
                    self._decode_first_token_completed_at
                )

        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.info(
                f"[TOKEN-ROLLOUT-COMPLETE] req={self._id} completed_at={self._completed_at}, total_tokens={self.total_tokens}"
            )

        # One decode iteration has finished; reset verify in-flight hint.
        self.reset_spec_verify_state_after_batch_end()

    def on_batch_stage_schedule(
        self,
        time: float,
        cluster_type: ClusterType,
    ) -> None:
        self._latest_stage_scheduled_at = time
        # Preemption duration is intentionally not tracked here.
        # Queue delay after preemption is accounted by waiting-time metrics:
        # on_enter_waiting_queue() -> on_leave_waiting_queue().
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
        cluster_type: ClusterType,
    ) -> None:
        # Record execution times for this stage completion
        self._execution_time[cluster_type].append(execution_time)
        self._model_execution_time[cluster_type].append(model_execution_time)
        self._latest_stage_completed_at = time
        # NOTE: Do NOT set self._preempted = True here!
        # The _preempted flag should only be set in _preempt_request() method
        # when a request is actually preempted. Setting it here causes all
        # requests to be incorrectly marked as preempted after batch stage completion.

    # Inter-cluster transfer timeline hooks
    def on_inter_cluster_transfer_start(
        self,
        time: float,
        source_cluster: ClusterType,
        target_cluster: ClusterType,
        activation_size_bytes: int,
    ) -> None:
        """Mark the beginning of an inter-cluster transfer for this request.

        This is a lightweight hook to improve timeline completeness and decode-ffn
        residence tracking.
        """
        # Mark request as in-flight for the whole A->F->A roundtrip.
        if (
            source_cluster == ClusterType.DECODE_ATTN
            and target_cluster == ClusterType.DECODE_FFN
        ):
            if self._af_roundtrip_inflight:
                raise ValueError(
                    f"A->F transfer start while roundtrip already in-flight for request {self._id}"
                )
            self._af_roundtrip_inflight = True

        # DECODE_FFN request-level residence window closes at F→A transfer start.
        if (
            source_cluster == ClusterType.DECODE_FFN
            and target_cluster == ClusterType.DECODE_ATTN
        ):
            if not self._af_roundtrip_inflight:
                raise ValueError(
                    f"F->A transfer start without active roundtrip for request {self._id}"
                )
            if self._decode_ffn_enter_time is None:
                raise ValueError(
                    f"DECODE_FFN exit without entry for request {self._id} at {time}"
                )
            residence = time - self._decode_ffn_enter_time
            if residence < 0:
                raise ValueError(
                    f"Negative DECODE_FFN residence for request {self._id}: "
                    f"start={self._decode_ffn_enter_time}, end={time}"
                )
            self._decode_ffn_residence_time += residence
            self._decode_ffn_enter_time = None

        # Record as preemption start boundary for the source cluster
        self._latest_stage_completed_at = max(self._latest_stage_completed_at, time)
        # Optional: we could append a 0-length preemption bucket here; keep noop for stability
        return

    def on_inter_cluster_transfer_end(
        self,
        time: float,
        source_cluster: ClusterType,
        target_cluster: ClusterType,
        activation_size_bytes: int,
    ) -> None:
        """Mark the end of an inter-cluster transfer for this request."""
        # DECODE_FFN request-level residence window opens at A→F transfer end.
        if (
            source_cluster == ClusterType.DECODE_ATTN
            and target_cluster == ClusterType.DECODE_FFN
        ):
            if not self._af_roundtrip_inflight:
                raise ValueError(
                    f"A->F transfer end without active roundtrip for request {self._id}"
                )
            if self._decode_ffn_enter_time is not None:
                raise ValueError(
                    f"DECODE_FFN entry already open for request {self._id}: "
                    f"existing_start={self._decode_ffn_enter_time}, new_start={time}"
                )
            self._decode_ffn_enter_time = time

        if (
            source_cluster == ClusterType.DECODE_FFN
            and target_cluster == ClusterType.DECODE_ATTN
        ):
            if not self._af_roundtrip_inflight:
                raise ValueError(
                    f"F->A transfer end without active roundtrip for request {self._id}"
                )
            self._af_roundtrip_inflight = False

        # Arrival is recorded elsewhere via on_arrival; keep this as a marker.
        return

    def get_cluster_arrival_time(self, cluster_type: ClusterType, round_index: int = -1) -> float:
        """
        Get the arrival time for a specific cluster and round.

        Args:
            cluster_type: The cluster type to get arrival time for
            round_index: The round index (default: -1 for most recent, 0 for first)

        Returns:
            Arrival time for the cluster and round, or original arrival time if not found
        """
        cluster_times = self._cluster_arrival_times.get(cluster_type, [])
        if cluster_times:
            try:
                return cluster_times[round_index]
            except IndexError:
                # If round_index is out of bounds, return the most recent time
                return cluster_times[-1] if cluster_times else self._arrived_at
        return self._arrived_at

    def get_cluster_scheduled_at(self, cluster_type: ClusterType, round_index: int = -1) -> float:
        """
        Get the scheduled time for a specific cluster and round.

        Args:
            cluster_type: The cluster type to get scheduled time for
            round_index: The round index (default: -1 for most recent, 0 for first)

        Returns:
            Scheduled time for the cluster and round, or 0 if not scheduled in that cluster
        """
        cluster_times = self._scheduled_at.get(cluster_type, [])
        if cluster_times:
            try:
                return cluster_times[round_index]
            except IndexError:
                # If round_index is out of bounds, return the most recent time
                return cluster_times[-1] if cluster_times else 0
        return 0

    def get_cluster_scheduling_delay(self, cluster_type: ClusterType, round_index: int = -1) -> float:
        """
        Get the scheduling delay for a specific cluster and round.

        Args:
            cluster_type: The cluster type to get scheduling delay for
            round_index: The round index (default: -1 for most recent, 0 for first)

        Returns:
            Scheduling delay for the cluster and round, or 0 if not scheduled in that cluster
        """
        cluster_delays = self._scheduling_delay.get(cluster_type, [])
        if cluster_delays:
            try:
                return cluster_delays[round_index]
            except IndexError:
                # If round_index is out of bounds, return the most recent delay
                return cluster_delays[-1] if cluster_delays else 0
        return 0

    def has_been_scheduled_in_cluster(self, cluster_type: ClusterType) -> bool:
        """
        Check if the request has been scheduled in a specific cluster.

        Args:
            cluster_type: The cluster type to check

        Returns:
            True if the request has been scheduled in the cluster, False otherwise
        """
        cluster_times = self._scheduled_at.get(cluster_type, [])
        return len(cluster_times) > 0

    def get_cluster_scheduling_round_count(self, cluster_type: ClusterType) -> int:
        """
        Get the number of scheduling rounds for a specific cluster.

        Args:
            cluster_type: The cluster type to check

        Returns:
            Number of times the request has been scheduled in the cluster
        """
        cluster_times = self._scheduled_at.get(cluster_type, [])
        return len(cluster_times)

    def get_total_cluster_execution_time(self, cluster_type: ClusterType) -> float:
        """
        Get the total execution time for a specific cluster across all rounds.

        Args:
            cluster_type: The cluster type to get execution time for

        Returns:
            Total execution time for the cluster across all rounds
        """
        cluster_times = self._execution_time.get(cluster_type, [])
        return sum(cluster_times)

    def get_total_cluster_model_execution_time(self, cluster_type: ClusterType) -> float:
        """
        Get the total model execution time for a specific cluster across all rounds.

        Args:
            cluster_type: The cluster type to get model execution time for

        Returns:
            Total model execution time for the cluster across all rounds
        """
        cluster_times = self._model_execution_time.get(cluster_type, [])
        return sum(cluster_times)

    def get_cluster_execution_time_by_round(self, cluster_type: ClusterType, round_index: int = -1) -> float:
        """
        Get the execution time for a specific cluster and round.

        Args:
            cluster_type: The cluster type to get execution time for
            round_index: The round index (default: -1 for most recent, 0 for first)

        Returns:
            Execution time for the cluster and round, or 0 if not found
        """
        cluster_times = self._execution_time.get(cluster_type, [])
        if cluster_times:
            try:
                return cluster_times[round_index]
            except IndexError:
                return 0
        return 0

    def complete_decode_attn_processing(self) -> None:
        """
        Mark that this request has completed attention processing in decode-attn cluster.
        This method is called when a request finishes processing in the decode-attn cluster
        and is ready to be transferred to decode-ffn cluster.
        """
        # For now, we don't need to update any internal state since the request
        # will be handled by the cluster transfer logic in BatchEndEvent
        pass

    # TODO: legacy, to be removed
    def advance_decode_layer(self, total_layers: int, time: float, num_layers_completed: int = 1, num_token_completed: int = 1) -> bool:
        """
        Advance the decode processing by completing additional layers.

        Note: This method is kept for compatibility but may not be used in the
        current decode-attn cluster implementation where each cluster processes
        only specific layers.

        Args:
            num_layers_completed: Number of layers completed in this processing step
        """
        # Enhanced debugging: Log layer advancement (before)
        logger.info(
            f"[ADVANCE-LAYER][BEFORE] req={self._id} token_idx={self._current_decode_token_index} "
            f"completed_layer_count={self._completed_layer_count}/{total_layers} num_processed_tokens={self._num_processed_tokens}/{self.total_tokens}"
        )
        if self._id == 0:
            logger.info(
                f"[TRACE-R0][ADVANCE-LAYER-BEFORE] token_idx={self._current_decode_token_index} "
                f"layer={self._completed_layer_count}/{total_layers} decode_progress={self.num_processed_decode_tokens}/{self._num_decode_tokens}"
            )

        assert self._completed_layer_count <= total_layers, f"Request {self._id} completed_layer_count ({self._completed_layer_count}) " \
            f"does not match total_layers ({total_layers}) in complete_decode_token"

        assert self._num_processed_tokens <= self.total_tokens, f"Request {self._id} already completed, " \
            f"num_processed_tokens={self._num_processed_tokens}, total_tokens={self.total_tokens}"

        self._completed_layer_count += num_layers_completed

        is_final_layer = self._completed_layer_count == total_layers
        is_final_token = False
        if is_final_layer:
            before_tokens = self._num_processed_tokens
            # rollout
            self._num_processed_tokens += num_token_completed
            # Reset layer count for next token
            self._completed_layer_count = 0
            # Advance to next decode token index
            self._current_decode_token_index += 1

            is_final_token = self._num_processed_tokens == self.total_tokens

            # self._completed will be updated in requests.on_batch_end()
            if is_final_token:
                self._completed = True
                self._completed_at = time

            # Log token increment
            logger.info(
                f"[ADVANCE-LAYER][AFTER] req={self._id} FINAL_LAYER=True token_inc {before_tokens}->{self._num_processed_tokens} "
                f"new_token_idx={self._current_decode_token_index} is_final_token={is_final_token}"
            )
            if self._id == 0:
                logger.info(
                    f"[TRACE-R0][ADVANCE-LAYER-AFTER] decode_progress={self.num_processed_decode_tokens}/{self._num_decode_tokens} "
                    f"is_final_token={is_final_token}"
                )
        else:
            logger.info(
                f"[ADVANCE-LAYER][AFTER] req={self._id} FINAL_LAYER=False layer_progress={self._completed_layer_count}/{total_layers}"
            )
            if self._id == 0:
                logger.info(
                    f"[TRACE-R0][ADVANCE-LAYER-AFTER] layer_progress={self._completed_layer_count}/{total_layers}"
                )

        # return status: whether the is the final layer and whether request finished rollout process.
        return is_final_layer, is_final_token



    # def complete_decode_token(self, total_layers: int) -> bool:
    #     """
    #     Complete the current decode token and advance to the next one.

    #     Note: This method is kept for compatibility but may not be used in the
    #     current disaggregated implementation where token completion is handled
    #     differently across clusters.

    #     Args:
    #         total_layers: Total number of layers in the model

    #     Returns:
    #         True if a new decode token was generated, False if request is completed
    #     """
    #     # Enhanced debugging: Log detailed token completion state
    #     from frontier.logger import init_logger
    #     logger = init_logger(__name__)

    #     logger.info(f"🔍 [TOKEN_DEBUG] Request {self._id} complete_decode_token called: "
    #                f"completed_layer_count={self._completed_layer_count}, "
    #                f"total_layers={total_layers}, "
    #                f"current_decode_token_index={self._current_decode_token_index}, "
    #                f"num_processed_tokens={self._num_processed_tokens}, "
    #                f"total_tokens={self.total_tokens}, "
    #                f"num_decode_tokens={self._num_decode_tokens}")


    #     # CRITICAL FIX: Check if request is already completed before incrementing tokens
    #     # This prevents _num_processed_tokens from exceeding total_tokens
    #     if self._num_processed_tokens == self.total_tokens:
    #         self._completed = True
    #         logger.info(f"✅ [TOKEN_DEBUG] Request {self._id} ALREADY COMPLETED! "
    #                     f"All {self._num_decode_tokens} decode tokens generated, "
    #                     f"processed_tokens={self._num_processed_tokens}, total_tokens={self.total_tokens}")
    #         return False

    #     # Reset layer count for next token
    #     self._completed_layer_count = 0

    #     # Advance to next decode token index
    #     self._current_decode_token_index += 1

    #     logger.info(f"🎯 [TOKEN_DEBUG] Request {self._id} TOKEN GENERATED! "
    #                 f"New token index: {self._current_decode_token_index}, "
    #                 f"Total processed tokens: {self._num_processed_tokens}, "
    #                 f"Decode tokens progress: {self.num_processed_decode_tokens}/{self._num_decode_tokens}")

    #     # Check if request is fully completed after token generation
    #     if self._num_processed_tokens >= self.total_tokens:
    #         self._completed = True
    #         logger.info(f"✅ [TOKEN_DEBUG] Request {self._id} COMPLETED! "
    #                     f"Generated all {self._num_decode_tokens} decode tokens")
    #         return False

    #     logger.info(f"🔄 [TOKEN_DEBUG] Request {self._id} CONTINUING: "
    #                 f"Need {self._num_decode_tokens - self.num_processed_decode_tokens} more decode tokens")
    #     return True




    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "priority": self._priority,
            "execution_time": self.execution_time,
            "model_execution_time": self.model_execution_time,
            "scheduled_at": self.scheduled_at,
            "scheduling_delay": self.scheduling_delay,
            "preempted_time": self.preempted_time,
            "completed_at": self._completed_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "thinking_depth": self._thinking_depth,
            "current_thinking_round_index": self._current_thinking_round_index,
            "thinking_round_plans": [
                {
                    "num_prefill_tokens": round_plan.num_prefill_tokens,
                    "num_decode_tokens": round_plan.num_decode_tokens,
                }
                for round_plan in self._thinking_round_plans
            ],
            "tool_call_latency": self._tool_call_latency,
            "user_facing_num_prefill_tokens": self._user_facing_num_prefill_tokens,
            "user_facing_num_decode_tokens": self._user_facing_num_decode_tokens,
            "thinking_home_cluster_type": (
                self._thinking_home_cluster_type.name
                if self._thinking_home_cluster_type is not None
                else None
            ),
            "thinking_home_replica_id": self._thinking_home_replica_id,
            "thinking_home_dp_id": self._thinking_home_dp_id,
            "pending_thinking_requeue": self._pending_thinking_requeue,
            "thinking_time_total": self._thinking_time_total,
            "tool_call_time_total": self._tool_call_time_total,
            "completed_thinking_rounds": self._completed_thinking_rounds,
            "num_processed_tokens": self._num_processed_tokens,
            "num_prefill_tokens_cached": self._num_prefill_tokens_cached,
            "block_hash_ids": self._block_hash_ids,
            "session_id": self._session_id,
            "cohort": self._cohort,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
            # Add cluster-specific information for debugging (multi-round support)
            "cluster_arrival_times": {ct.name: times for ct, times in self._cluster_arrival_times.items()},
            "cluster_scheduled_at": {ct.name: times for ct, times in self._scheduled_at.items()},
            "cluster_scheduling_delays": {ct.name: delays for ct, delays in self._scheduling_delay.items()},
            "cluster_execution_times": {ct.name: times for ct, times in self._execution_time.items()},
            "cluster_model_execution_times": {ct.name: times for ct, times in self._model_execution_time.items()},
            "cluster_preempted_times": {ct.name: times for ct, times in self._preempted_time.items()},
            "decode_ffn_enter_time": self._decode_ffn_enter_time,
            "decode_ffn_residence_time": self._decode_ffn_residence_time,
            "af_roundtrip_inflight": self._af_roundtrip_inflight,
            "first_decode_token_completed_at": self._first_decode_token_completed_at,
            "decode_first_token_completed_at": self._decode_first_token_completed_at,
            "kv_cache_transfer_start_time": self._kv_cache_transfer_start_time,
            "kv_cache_transfer_end_time": self._kv_cache_transfer_end_time,
            # Decode-attn cluster ping-pong pipeline state
            "current_decode_token_index": self._current_decode_token_index,
            "completed_layer_count": self._completed_layer_count,
            # Speculative decoding runtime state
            "spec_decode_enabled": self._spec_decode_enabled,
            "spec_method": self._spec_method,
            "spec_method_uses_lookahead_slots": self._spec_method_uses_lookahead_slots,
            "spec_num_speculative_tokens": self._spec_num_speculative_tokens,
            "spec_next_planned_draft_tokens": self._spec_next_planned_draft_tokens,
            "spec_current_verify_tokens": self._spec_current_verify_tokens,
            "spec_last_committed_tokens": self._spec_last_committed_tokens,
            "spec_total_iterations": self._spec_total_iterations,
            "spec_total_accepted_drafts": self._spec_total_accepted_drafts,
            "spec_total_rejected_drafts": self._spec_total_rejected_drafts,
            "spec_total_committed_tokens": self._spec_total_committed_tokens,
        }

    def restart(self):
        logger.debug(f"Restarting request {self._id}")
        assert not self._pending_thinking_requeue, (
            "Cannot restart a request with pending thinking requeue state."
        )
        self.advance_runtime_epoch()

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        # Thinking Mode intentionally preserves the current round index here.
        # A restart represents KV-loss recovery within the active round rather
        # than a rollback to round 0.
        # _thinking_round_plans stay as the initial per-round configuration.
        # After restart, the active round's runtime token split follows the
        # vLLM replay rule below instead of the original round plan values.
        preserve_round_history = (
            self.is_thinking_mode_enabled and self._current_thinking_round_index > 0
        )
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        # Thinking Mode must preserve prior-round history when replaying a later
        # round after preemption. Round-0 restarts keep the legacy behavior.
        if not preserve_round_history:
            self._scheduled_at = defaultdict(list)
            self._execution_time = defaultdict(list)
            self._model_execution_time = defaultdict(list)
            self._scheduling_delay = defaultdict(list)
            self._preempted_time = defaultdict(list)
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0
        self._decode_ffn_enter_time = None
        self._decode_ffn_residence_time = 0.0
        self._af_roundtrip_inflight = False
        self._first_decode_token_completed_at = 0.0
        self._decode_first_token_completed_at = 0.0
        self._kv_cache_transfer_start_time = None
        self._kv_cache_transfer_end_time = None
        self._num_prefill_tokens_cached = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        # Reset decode-attn cluster ping-pong pipeline state
        self._current_decode_token_index = 1
        self._completed_layer_count = 0

        # Reset speculative iteration state, keep static config.
        self._spec_next_planned_draft_tokens = 0
        self._spec_current_verify_tokens = 1
        self._spec_last_committed_tokens = 1
        self._spec_total_iterations = 0
        self._spec_total_accepted_drafts = 0
        self._spec_total_rejected_drafts = 0
        self._spec_total_committed_tokens = 0
        self._spec_post_first_service_delay = 0.0

        self._num_restarts += 1

    def bind_thinking_home_queue(
        self,
        cluster_type: ClusterType,
        replica_id: int,
        dp_id: int,
    ) -> None:
        if not self.is_thinking_mode_enabled:
            return
        if cluster_type not in [ClusterType.MONOLITHIC, ClusterType.PREFILL]:
            raise ValueError(
                "Thinking Mode v1 home queue must be MONOLITHIC or PREFILL, "
                f"got={cluster_type.name}"
            )
        if self._thinking_home_cluster_type is None:
            self._thinking_home_cluster_type = cluster_type
            self._thinking_home_replica_id = int(replica_id)
            self._thinking_home_dp_id = int(dp_id)
            return
        if (
            self._thinking_home_cluster_type != cluster_type
            or self._thinking_home_replica_id != int(replica_id)
            or self._thinking_home_dp_id != int(dp_id)
        ):
            raise ValueError(
                "Thinking Mode v1 home queue affinity changed unexpectedly for "
                f"request {self._id}: "
                f"existing=({self._thinking_home_cluster_type.name}, "
                f"{self._thinking_home_replica_id}, {self._thinking_home_dp_id}), "
                f"new=({cluster_type.name}, {replica_id}, {dp_id})"
            )

    def begin_thinking_tool_wait(
        self,
        time: float,
        round_started_at: float | None = None,
    ) -> None:
        if not self.is_thinking_mode_enabled:
            raise ValueError("begin_thinking_tool_wait requires Thinking Mode.")
        if self.is_final_thinking_round:
            raise ValueError("Final thinking round cannot enter tool wait.")
        if self._thinking_home_cluster_type is None:
            raise ValueError("Thinking Mode home queue affinity has not been bound.")
        if self._pending_thinking_requeue:
            raise ValueError("Thinking Mode request is already waiting for requeue.")

        effective_round_started_at = round_started_at
        if effective_round_started_at is None:
            effective_round_started_at = self.get_cluster_arrival_time(
                self._thinking_home_cluster_type
            )
        round_duration = time - effective_round_started_at
        if round_duration < -1e-9:
            raise ValueError(
                f"Negative thinking round duration for request {self._id}: "
                f"start={effective_round_started_at}, end={time}"
            )

        self._thinking_time_total += max(0.0, round_duration)
        self._completed_thinking_rounds += 1
        self._thinking_tool_wait_started_at = time
        self._pending_thinking_requeue = True
        self._completed = False
        self._scheduled = False
        self._preempted = False

    def _reset_runtime_state_for_next_thinking_round(self) -> None:
        self._num_processed_tokens = 0
        self._num_prefill_tokens_cached = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False
        self._current_decode_token_index = 1
        self._completed_layer_count = 0
        self._af_roundtrip_inflight = False
        self._kv_cache_transfer_time = 0.0
        self._kv_cache_transfer_start_time = None
        self._kv_cache_transfer_end_time = None
        self._m2n_transfer_time_attn_to_ffn = 0.0
        self._m2n_transfer_time_ffn_to_attn = 0.0
        self._first_decode_token_completed_at = 0.0
        self._decode_first_token_completed_at = 0.0
        self._decode_ffn_enter_time = None
        self._decode_ffn_residence_time = 0.0
        self._spec_next_planned_draft_tokens = 0
        self._spec_current_verify_tokens = 1
        self._spec_last_committed_tokens = 1
        self._spec_total_iterations = 0
        self._spec_total_accepted_drafts = 0
        self._spec_total_rejected_drafts = 0
        self._spec_total_committed_tokens = 0
        self._spec_post_first_service_delay = 0.0

    def finish_thinking_tool_wait_and_requeue(self, time: float) -> None:
        if not self._pending_thinking_requeue:
            raise ValueError("Thinking Mode request is not pending requeue.")
        if self._thinking_tool_wait_started_at is None:
            raise ValueError("Thinking Mode tool wait start time is missing.")
        if self._thinking_home_cluster_type is None:
            raise ValueError("Thinking Mode home queue affinity has not been bound.")

        tool_wait = time - self._thinking_tool_wait_started_at
        if tool_wait < -1e-9:
            raise ValueError(
                f"Negative tool wait duration for request {self._id}: "
                f"start={self._thinking_tool_wait_started_at}, end={time}"
            )
        self._tool_call_time_total += max(0.0, tool_wait)
        self._thinking_tool_wait_started_at = None
        self._pending_thinking_requeue = False
        self._reset_runtime_state_for_next_thinking_round()
        self.advance_thinking_round()
        self.on_enter_waiting_queue(time, self._thinking_home_cluster_type)

    def advance_thinking_round(self) -> None:
        # NOTE: This method only advances the round plan cursor.
        # Runtime-state cleanup for an actual requeue belongs in
        # finish_thinking_tool_wait_and_requeue().
        if self.is_final_thinking_round:
            raise ValueError("Request is already on the final thinking round.")

        self._current_thinking_round_index += 1
        next_round_plan = self._thinking_round_plans[self._current_thinking_round_index]
        self._num_prefill_tokens = next_round_plan.num_prefill_tokens
        self._num_decode_tokens = next_round_plan.num_decode_tokens
        self._num_processed_tokens = 0
        self._num_prefill_tokens_cached = 0
        self._is_prefill_complete = False
        self._prefill_completed_at = 0
