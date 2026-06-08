from typing import List

from frontier.events.base_event import BaseEvent
from frontier.types import EventType, ClusterType
from frontier.scheduler import BaseGlobalScheduler, BaseClusterScheduler, BaseReplicaScheduler
from frontier.metrics import MetricsStore
from frontier.entities import Batch
from frontier.logger import get_cluster_logger


def _should_mark_first_decode_token(request, cluster_type: ClusterType) -> bool:
    """Return whether this callback should stamp first_decode_token_completed_at.

    In MONOLITHIC chunked-prefill mode, Frontier rolls request token state once at
    prefill-complete boundary to keep scheduling parity. For online vLLM alignment,
    that first decode-progress rollout corresponds to the first observable output
    token and should seed TTFT / TPOT bookkeeping.
    """
    if request.first_decode_token_completed_at != 0:
        return False

    if cluster_type == ClusterType.MONOLITHIC:
        num_decode_tokens = getattr(request, "num_decode_tokens", None)
        if num_decode_tokens is None:
            # Keep legacy test doubles and lightweight request mocks compatible.
            return request.num_processed_decode_tokens >= 1
        if int(num_decode_tokens) <= 0:
            return False
        return request.num_processed_decode_tokens >= 1

    return request.num_processed_decode_tokens >= 1


class GlobalBatchEndEvent(BaseEvent):
    """
    True global batch completion for decode pipeline.

    Created by the monolithic cluster when both conditions hold:
    - final layer for the current token, and
    - final token for the batch

    Responsibilities:
    - Call batch.on_batch_end(time, cluster=MONOLITHIC)
    - Call replica_scheduler.on_batch_end(batch) to free resources and decrement running batches
    - Record metrics
    - Trigger a ReplicaScheduleEvent for continued scheduling
    """

    def __init__(
        self,
        time: float,
        replica_id: int,
        dp_id: int,
        batch: Batch,
        cluster_type: ClusterType = None,
        batch_schedule_epoch: int | None = None,
        request_execution_signatures: list[tuple[int, int, int]] | None = None,
        request_mutation_signatures: list[tuple[int, int, int, int]] | None = None,
        thinking_round_start_times: list[float | None] | None = None,
    ):
        super().__init__(time, EventType.GLOBAL_BATCH_END)
        self._replica_id = replica_id
        self._dp_id = dp_id
        self._batch = batch
        self._cluster_type = (
            cluster_type if cluster_type is not None else ClusterType.MONOLITHIC
        )
        self._batch_schedule_epoch = (
            batch.schedule_epoch
            if batch_schedule_epoch is None
            else int(batch_schedule_epoch)
        )
        self._request_execution_signatures = (
            batch.request_execution_signatures
            if request_execution_signatures is None
            else list(request_execution_signatures)
        )
        self._request_mutation_signatures = (
            batch.request_mutation_signatures
            if request_mutation_signatures is None
            else list(request_mutation_signatures)
        )
        self._thinking_round_start_times = (
            batch.thinking_round_start_times
            if thinking_round_start_times is None
            else list(thinking_round_start_times)
        )

    def handle_event(self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent
        from frontier.events.thinking_round_requeue_event import (
            ThinkingRoundRequeueEvent,
        )

        logger = get_cluster_logger(__name__, self._cluster_type.name)
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        replica_scheduler: BaseReplicaScheduler = cluster_scheduler.get_dp_replica_scheduler(self._replica_id, self._dp_id)

        if self._batch.schedule_epoch != self._batch_schedule_epoch:
            logger.warning(
                "[STALE-GLOBAL-BATCH-END] Skipping batch %s: expected_schedule_epoch=%s "
                "current_schedule_epoch=%s",
                getattr(self._batch, "id", "?"),
                self._batch_schedule_epoch,
                self._batch.schedule_epoch,
            )
            return []

        # Debug: entering GlobalBatchEndEvent with batch/request states
        try:
            req_states = [
                f"id={r.id}|tok={getattr(r,'current_decode_token_index',None)}|layer={getattr(r,'completed_layer_count',None)}"
                for r in self._batch.requests
            ]
            logger.info(
                f"[GLOBAL-END][ENTER] batch={getattr(self._batch,'id','?')} replica={self._replica_id} dp={self._dp_id} reqs={req_states}"
            )
        except Exception as e:
            logger.debug(f"[GLOBAL-END][ENTER] logging failed: {e}")

        def _current_request_entries() -> list[tuple[int, object]]:
            current_entries: list[tuple[int, object]] = []
            seen_request_ids: set[int] = set()
            for index, request in enumerate(self._batch.requests):
                if request.id in seen_request_ids:
                    continue
                seen_request_ids.add(request.id)
                current_execution_signature = Batch._get_request_execution_signature(
                    request
                )
                current_mutation_signature = Batch._get_request_mutation_signature(
                    request
                )
                if (
                    current_execution_signature
                    == self._request_execution_signatures[index]
                    and current_mutation_signature
                    == self._request_mutation_signatures[index]
                ):
                    thinking_round_start_time = self._thinking_round_start_times[index]
                    if Batch._thinking_round_start_is_in_future(
                        thinking_round_start_time,
                        self.time,
                    ):
                        logger.warning(
                            "[STALE-GLOBAL-BATCH-END-FUTURE-ROUND-START] Skipping "
                            "request %s in batch %s: expected_round_start=%s "
                            "event_time=%s",
                            request.id,
                            getattr(self._batch, "id", "?"),
                            thinking_round_start_time,
                            self.time,
                        )
                        continue
                    current_entries.append((index, request))
            return current_entries

        pre_batch_request_entries = _current_request_entries()

        # PD-disaggregation DECODE keeps the legacy marker timing: stamp TTFT when
        # finishing token index=1 before request token rollout mutates the index.
        if self._cluster_type == ClusterType.DECODE:
            for _, request in pre_batch_request_entries:
                if (
                    request.first_decode_token_completed_at == 0
                    and getattr(request, "current_decode_token_index", 0) == 1
                ):
                    request.mark_first_decode_token_complete(self.time)

        # Finalize at decode-attn
        self._batch.on_batch_end(
            self.time,
            self._cluster_type,
            request_execution_signatures=self._request_execution_signatures,
            request_mutation_signatures=self._request_mutation_signatures,
            thinking_round_start_times=self._thinking_round_start_times,
        )
        replica_scheduler.on_batch_end(self._batch)  # decrement running batches

        thinking_requeue_events: List[BaseEvent] = []
        for index, request in pre_batch_request_entries:
            if (
                request.completed
                and request.is_thinking_mode_enabled
                and not request.is_final_thinking_round
            ):
                request.begin_thinking_tool_wait(
                    self.time,
                    round_started_at=self._thinking_round_start_times[index],
                )
                thinking_requeue_events.append(
                    ThinkingRoundRequeueEvent(
                        self.time + request.tool_call_latency,
                        request,
                    )
                )

        # Mark first decode token completion for TTFT tracking.
        # The marking must happen after request state transitions in on_batch_end().
        # Otherwise, partial prefill iterations (decode index starts at 1) can be
        # incorrectly counted as first-token completion in chunked prefill mode.
        for _, request in pre_batch_request_entries:
            if getattr(request, "pending_thinking_requeue", False):
                continue
            if self._cluster_type != ClusterType.DECODE:
                if _should_mark_first_decode_token(request, self._cluster_type):
                    request.mark_first_decode_token_complete(self.time)
            decode_first_completed_at = getattr(
                request, "decode_first_token_completed_at", 0
            )
            if (
                request.first_decode_token_completed_at == 0
                and decode_first_completed_at > 0
            ):
                request.mark_first_decode_token_complete(
                    decode_first_completed_at
                )

        memory_usage_percent = replica_scheduler.memory_usage_percent
        # Some returned A\u2192F\u2192A batches may be newly constructed during FFN aggregation
        # and lack a valid scheduled_at on decode-attn. Guard metric emission accordingly.
        try:
            logger.info(
                f"[GLOBAL-END][EMIT] batch={getattr(self._batch,'id','?')} -> ReplicaScheduleEvent(replica={self._replica_id}, dp={self._dp_id})"
            )
        except Exception:
            pass

        if getattr(self._batch, "scheduled", False):
            metrics_store.on_batch_end(
                self.time,
                self._batch,
                self._replica_id,
                memory_usage_percent,
                self._cluster_type,
                self._dp_id,
            )
        else:
            logger.error("[GLOBAL-END] Skipping metrics_store.on_batch_end: batch was not scheduled on decode-attn")

        # Ensure completion counters update even if metrics writing is disabled.
        # Request metric validation errors must propagate; otherwise malformed KPI
        # state can be silently reported as a successful simulation.
        for _, request in pre_batch_request_entries:
            if not getattr(request, "completed", False):
                continue
            metrics_store._on_request_end(self.time, request)

        # Schedule next batch for this DP replica
        # Note: Idle batch mechanism now handles MoE synchronization when num_requests < dp_size
        next_events = [ReplicaScheduleEvent(self.time, self._replica_id, self._cluster_type, self._dp_id)]

        return next_events + thinking_requeue_events

    def get_target_cluster(self) -> ClusterType:
        # Processed by DECODE_ATTN cluster
        return self._cluster_type
