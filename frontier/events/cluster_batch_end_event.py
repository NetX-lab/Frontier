from typing import List

from frontier.events.base_event import BaseEvent
from frontier.types import EventType, ClusterType
from frontier.scheduler import BaseGlobalScheduler
from frontier.metrics import MetricsStore
from frontier.entities import Batch
from frontier.logger import get_cluster_logger


class ClusterBatchEndEvent(BaseEvent):
    """
    Cluster-internal batch stage completion event.

    PREFILL completes local batch work and emits KV cache transfers to the decode
    cluster. MONOLITHIC keeps the existing co-location completion path.
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
        from frontier.events.kv_cache_transfer_start_event import (
            KVCacheTransferStartEvent,
        )
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

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

        # Always record cluster-internal stage completion hooks.
        if hasattr(self._batch, "on_cluster_stage_end"):
            self._batch.on_cluster_stage_end(self.time, self._cluster_type)
        if hasattr(replica_scheduler, "on_cluster_stage_end"):
            replica_scheduler.on_cluster_stage_end(self._batch)

        if self._cluster_type == ClusterType.PREFILL:
            self._batch.on_batch_end(
                self.time,
                self._cluster_type,
            )
            replica_scheduler.on_batch_end(self._batch)

            memory_usage_percent = replica_scheduler.memory_usage_percent
            metrics_store.on_batch_end(
                self.time,
                self._batch,
                self._replica_id,
                memory_usage_percent,
                self._cluster_type,
                self._dp_id,
            )

            kv_pred = cluster_scheduler._kv_cache_transfer_predictor
            if kv_pred is None:
                raise ValueError(
                    "KV cache transfer predictor not found in ClusterScheduler"
                )

            replica_config = cluster_scheduler._config.replica_config
            target_cluster = cluster_scheduler._get_decode_target_cluster()

            for request in self._batch.requests:
                if request.is_prefill_complete and request.num_decode_tokens > 0:
                    kv_cache_size_bytes, transfer_time_ms = (
                        kv_pred.get_transfer_info_for_request(
                            source_cluster_type=self._cluster_type,
                            target_cluster_type=target_cluster,
                            request=request,
                            replica_config=replica_config,
                        )
                    )

                    from frontier.entities.batch import Batch as SingleBatch

                    single_request_batch = SingleBatch(
                        replica_id=self._replica_id,
                        requests=[request],
                        num_tokens=[request.num_prefill_tokens],
                        is_moe=replica_config.model_config.is_moe,
                    )
                    next_events.append(
                        KVCacheTransferStartEvent(
                            self.time,
                            source_replica_id=self._replica_id,
                            source_dp_id=self._dp_id,
                            target_cluster_type=target_cluster,
                            batch=single_request_batch,
                            kv_cache_size_bytes=kv_cache_size_bytes,
                            transfer_time_ms=transfer_time_ms,
                            source_cluster_type=self._cluster_type,
                        )
                    )

            next_events.append(
                ReplicaScheduleEvent(
                    self.time, self._replica_id, self._cluster_type, self._dp_id
                )
            )
            return next_events

        if self._cluster_type == ClusterType.DECODE:
            if self._batch.is_idle:
                logger.info(
                    f"[DECODE-END][IDLE] batch_id={self._batch.id} is idle batch, skipping normal end logic"
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._dp_id
                    )
                )
                return next_events

            replica = cluster_scheduler._cluster.replicas[self._replica_id]
            is_moe = replica.is_moe

            if not is_moe:
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

            model_config = cluster_scheduler._config.replica_config.model_config
            total_layers = model_config.num_layers
            current_layer_id = self._get_current_layer_id_from_batch(self._batch)
            is_final_layer = current_layer_id >= total_layers - 1

            if is_final_layer:
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
            else:
                memory_usage_percent = replica_scheduler.memory_usage_percent
                metrics_store.on_batch_end(
                    self.time,
                    self._batch,
                    self._replica_id,
                    memory_usage_percent,
                    self._cluster_type,
                    self._dp_id,
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._dp_id
                    )
                )

            return next_events

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
