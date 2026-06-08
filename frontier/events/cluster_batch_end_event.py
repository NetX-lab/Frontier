from typing import List

from frontier.events.base_event import BaseEvent
from frontier.types import EventType, ClusterType
from frontier.scheduler import BaseGlobalScheduler
from frontier.metrics import MetricsStore
from frontier.entities import Batch
from frontier.logger import get_cluster_logger
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR


class ClusterBatchEndEvent(BaseEvent):
    """
    Cluster-internal batch stage completion event.

    This release supports only the MONOLITHIC co-location path. Disaggregated
    cluster types fail fast before any cluster-local completion logic runs.
    """

    def __init__(
        self,
        time: float,
        replica_id: int,
        batch: Batch,
        cluster_type: ClusterType,
        dp_id: int,
        batch_schedule_epoch: int | None = None,
        request_execution_signatures: list[tuple[int, int, int]] | None = None,
        request_mutation_signatures: list[tuple[int, int, int, int]] | None = None,
        thinking_round_start_times: list[float | None] | None = None,
    ):
        super().__init__(time, EventType.CLUSTER_BATCH_END)
        self._replica_id = replica_id
        self._batch = batch
        self._cluster_type = cluster_type
        self._dp_id = dp_id
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

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

        cluster_scheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(
            self._replica_id, self._dp_id
        )

        logger = get_cluster_logger(__name__, self._cluster_type.name)
        next_events: List[BaseEvent] = []

        if self._batch.schedule_epoch != self._batch_schedule_epoch:
            logger.warning(
                "[STALE-CLUSTER-BATCH-END] Skipping batch %s: expected_schedule_epoch=%s "
                "current_schedule_epoch=%s",
                self._batch.id,
                self._batch_schedule_epoch,
                self._batch.schedule_epoch,
            )
            return []

        # Always record cluster-internal stage completion hooks
        try:
            # Entities-level hook (lightweight; can be a no-op)
            if hasattr(self._batch, "on_cluster_stage_end"):
                self._batch.on_cluster_stage_end(self.time, self._cluster_type)
            # Replica-scheduler-level hook (lightweight; can be a no-op)
            if hasattr(replica_scheduler, "on_cluster_stage_end"):
                replica_scheduler.on_cluster_stage_end(self._batch)
        except Exception as e:
            logger.info(f"[CLUSTER-END][WARN] on_cluster_stage_end hooks error: {e}")

        # MONOLITHIC cluster: Complete batch processing
        # In co-location mode, MONOLITHIC processes everything: prefill + all decode tokens
        # IMPORTANT: In MONOLITHIC mode, ReplicaStageScheduleEvent uses the generic path
        # which processes ALL layers in one shot (not layer-by-layer like disaggregated mode).
        # Therefore, when ClusterBatchEndEvent is triggered, all layers have already been
        # processed, and we should directly emit GlobalBatchEndEvent.
        if self._cluster_type == ClusterType.MONOLITHIC:
            # IMPORTANT: Handle idle batches specially
            if self._batch.is_idle:
                logger.info(
                    f"[MONOLITHIC-END][IDLE] batch_id={self._batch.id} is idle batch, skipping normal end logic"
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._dp_id
                    )
                )
                return next_events

            # Check if this is a dense model (non-MoE) for logging purposes
            replica = cluster_scheduler._cluster.replicas[self._replica_id]
            is_moe = replica.is_moe

            # For both dense and MoE models in MONOLITHIC mode:
            # All layers are processed in one shot by ReplicaStageScheduleEvent (generic path)
            # So we should directly emit GlobalBatchEndEvent
            logger.info(
                f"[MONOLITHIC-END] batch_id={self._batch.id} is_moe={is_moe}, "
                f"emitting GlobalBatchEndEvent (all layers processed in one shot)"
            )
            from frontier.events.global_batch_end_event import GlobalBatchEndEvent

            next_events.append(
                GlobalBatchEndEvent(
                    self.time,
                    self._replica_id,
                    self._dp_id,
                    self._batch,
                    self._cluster_type,
                    batch_schedule_epoch=self._batch_schedule_epoch,
                    request_execution_signatures=self._request_execution_signatures,
                    request_mutation_signatures=self._request_mutation_signatures,
                    thinking_round_start_times=self._thinking_round_start_times,
                )
            )
            return next_events

        # Fallback - should never reach here
        logger.warning(
            f"[CLUSTER-END] Unhandled cluster type: {self._cluster_type}; no-op"
        )
        return []

    def _get_current_layer_id_from_batch(self, batch: "Batch") -> int:
        if not batch.requests:
            raise ValueError(
                "_get_current_layer_id_from_batch: batch.requests is empty"
            )
        # ISSUE-006 FIX: Use layer count from first non-completed request to avoid
        # using an overflowed layer_id from a completed request.
        for request in batch.requests:
            if not request.completed:
                return request.completed_layer_count
        # All requests completed - return the first request's layer count
        # (this case should be handled by the caller before reaching here)
        return batch.requests[0].completed_layer_count

    def get_target_cluster(self) -> ClusterType:
        # Cluster-internal event, processed by current cluster
        return self._cluster_type
