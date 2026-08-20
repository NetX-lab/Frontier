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
        replica_local_id: int | None,
        batch_schedule_epoch: int | None = None,
        request_execution_signatures: list[tuple[int, int, int]] | None = None,
        request_mutation_signatures: list[tuple[int, int, int, int]] | None = None,
        thinking_round_start_times: list[float | None] | None = None,
    ):
        super().__init__(time, EventType.CLUSTER_BATCH_END)
        self._replica_id = replica_id
        self._batch = batch
        self._cluster_type = cluster_type
        self._replica_local_id = replica_local_id
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
        replica_scheduler = cluster_scheduler.get_replica_scheduler(
            self._replica_id, self._replica_local_id
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

        if self._cluster_type == ClusterType.DECODE_FFN:
            from frontier.entities.batch import DenseFFNBatchGroup, EPBatchGroup
            from frontier.events.m2n_transfer_start_event import (
                M2NTransferStartEvent,
            )

            m2n_pred = getattr(
                cluster_scheduler, "_m2n_transfer_predictor", None
            )
            if m2n_pred is None:
                raise ValueError(
                    "M2N transfer predictor not found in decode-ffn cluster "
                    "scheduler"
                )
            replica_config = cluster_scheduler._config.replica_config

            if not isinstance(self._batch, (EPBatchGroup, DenseFFNBatchGroup)):
                raise ValueError(
                    "DECODE_FFN F→A return path received "
                    "non-EPBatchGroup/non-DenseFFNBatchGroup unsupported batch "
                    f"(type={type(self._batch).__name__}, "
                    f"id={self._batch.id}). Expected EPBatchGroup for MoE or "
                    "DenseFFNBatchGroup for dense Llama."
                )

            source_batch_ids = list(
                getattr(self._batch, "source_batch_ids", [])
            )
            if not source_batch_ids:
                raise ValueError(
                    f"{type(self._batch).__name__} {self._batch.id} has empty "
                    "source_batch_ids in DECODE_FFN return path"
                )
            if len(set(source_batch_ids)) != len(source_batch_ids):
                raise ValueError(
                    f"{type(self._batch).__name__} {self._batch.id} has "
                    f"duplicate source_batch_ids: {source_batch_ids}"
                )
            raw_batch_waiting = getattr(
                cluster_scheduler, "_raw_batch_waiting_for_m2n_back", None
            )
            if raw_batch_waiting is None:
                raise ValueError(
                    "_raw_batch_waiting_for_m2n_back is missing in "
                    "decode-ffn cluster scheduler"
                )

            batches_for_transfer = []
            for source_batch_id in source_batch_ids:
                original_batch = raw_batch_waiting.get(source_batch_id)
                if original_batch is None:
                    raise ValueError(
                        "Missing original batch for "
                        f"source_batch_id={source_batch_id} from "
                        f"{type(self._batch).__name__} {self._batch.id}"
                    )
                batches_for_transfer.append(original_batch)

            ffn_execution_time = getattr(self._batch, "execution_time", 0.0)
            if ffn_execution_time <= 0:
                raise ValueError(
                    "Invalid DECODE_FFN execution_time on "
                    f"{type(self._batch).__name__} {self._batch.id}: "
                    f"{ffn_execution_time}. ReplicaStageScheduleEvent must "
                    "store stage execution time."
                )

            activation_bytes = getattr(self._batch, "activation_bytes", 0)
            activation_bytes = (
                int(activation_bytes) if activation_bytes else 0
            )

            prepared_m2n_events = []
            for batch_for_transfer in batches_for_transfer:
                current_layer_id = self._get_current_layer_id_from_batch(
                    batch_for_transfer
                )
                if (
                    batch_for_transfer.decode_attn_original_replica_id is None
                ):
                    raise ValueError(
                        f"Batch {batch_for_transfer.id} missing "
                        "decode_attn_original_replica_id metadata"
                    )

                activation_size, transfer_time = m2n_pred.get_transfer_info(
                    source_cluster_type=ClusterType.DECODE_FFN,
                    target_cluster_type=ClusterType.DECODE_ATTN,
                    batch=batch_for_transfer,
                    replica_config=replica_config,
                )
                try:
                    req_ids = [r.id for r in batch_for_transfer.requests]
                    logger.info(
                        f"[M2N][F2A][CREATE] batch_id={batch_for_transfer.id} "
                        f"reqs={req_ids} "
                        f"batch_global_id={getattr(batch_for_transfer, 'global_id', '?')} "
                        "decode_attn_orig="
                        f"(replica={getattr(batch_for_transfer, 'decode_attn_original_replica_id', '?')},"
                        "replica_local_id="
                        f"{getattr(batch_for_transfer, 'decode_attn_original_replica_local_id', '?')}) "
                        f"target={ClusterType.DECODE_ATTN.name} "
                        f"size={activation_size}B t_ms={transfer_time:.3f}"
                    )
                except Exception:
                    logger.info(
                        f"[M2N][F2A][CREATE] batch_id={batch_for_transfer.id} "
                        "(details unavailable)"
                    )

                prepared_m2n_events.append(
                    M2NTransferStartEvent(
                        time=self.time,
                        source_replica_id=(
                            batch_for_transfer.decode_attn_original_replica_id
                        ),
                        source_replica_local_id=(
                            batch_for_transfer.decode_attn_original_replica_local_id
                        ),
                        source_cluster_type=ClusterType.DECODE_FFN,
                        target_cluster_type=ClusterType.DECODE_ATTN,
                        batch=batch_for_transfer,
                        activation_size_bytes=activation_size,
                        transfer_time_ms=transfer_time,
                        layer_id=current_layer_id,
                        afd_stage_idx=batch_for_transfer.afd_stage_idx,
                        source_execution_replica_id=self._replica_id,
                        source_execution_replica_local_id=self._replica_local_id,
                        target_execution_replica_id=(
                            batch_for_transfer.decode_attn_original_replica_id
                        ),
                        target_execution_replica_local_id=(
                            batch_for_transfer.decode_attn_original_replica_local_id
                        ),
                    )
                )

            prepared_schedule_event = ReplicaScheduleEvent(
                self.time,
                self._replica_id,
                self._cluster_type,
                self._replica_local_id,
            )

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
                self._replica_local_id,
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
                            source_replica_local_id=self._replica_local_id,
                            target_cluster_type=target_cluster,
                            batch=single_request_batch,
                            kv_cache_size_bytes=kv_cache_size_bytes,
                            transfer_time_ms=transfer_time_ms,
                            source_cluster_type=self._cluster_type,
                        )
                    )

            next_events.append(
                ReplicaScheduleEvent(
                    self.time, self._replica_id, self._cluster_type, self._replica_local_id
                )
            )
            return next_events

        # DECODE_ATTN: after attention, either recover an already-complete /
        # overflowed state or emit A→F M2N transfer for the matching FFN.
        if self._cluster_type == ClusterType.DECODE_ATTN:
            active_requests = [r for r in self._batch.requests if not r.completed]

            if not active_requests:
                from frontier.events.global_batch_end_event import GlobalBatchEndEvent

                global_end_time = (
                    cluster_scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
                        self.time,
                        self._batch,
                    )
                )
                logger.info(
                    f"[DECODE_ATTN-END] batch_id={self._batch.id} all requests completed, "
                    f"emitting GlobalBatchEndEvent at {global_end_time:.6f}s "
                    f"(skipping A→F transfer)"
                )
                next_events.append(
                    GlobalBatchEndEvent(
                        global_end_time,
                        self._replica_id,
                        self._replica_local_id,
                        self._batch,
                        self._cluster_type,
                    )
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._replica_local_id
                    )
                )
                return next_events

            model_config = cluster_scheduler._config.replica_config.model_config
            total_layers = model_config.num_layers

            current_layer_id = self._get_current_layer_id_from_batch(self._batch)
            is_final_layer = current_layer_id >= total_layers

            logger.info(
                f"[DECODE_ATTN-END] batch_id={self._batch.id} layer={current_layer_id}/{total_layers - 1} "
                f"is_final_layer={is_final_layer} active_requests={len(active_requests)}"
            )

            if is_final_layer:
                from frontier.events.global_batch_end_event import GlobalBatchEndEvent

                global_end_time = (
                    cluster_scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
                        self.time,
                        self._batch,
                    )
                )
                logger.info(
                    f"[DECODE_ATTN-END] Final layer completed, emitting GlobalBatchEndEvent "
                    f"at {global_end_time:.6f}s"
                )
                next_events.append(
                    GlobalBatchEndEvent(
                        global_end_time,
                        self._replica_id,
                        self._replica_local_id,
                        self._batch,
                        self._cluster_type,
                    )
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._replica_local_id
                    )
                )
                return next_events
            else:
                logger.info(
                    f"[ISSUE-007][A2F][READY] batch_id={self._batch.id}, "
                    f"decode_attn_original_replica_id={getattr(self._batch, 'decode_attn_original_replica_id', 'MISSING')}, "
                    f"decode_attn_original_replica_local_id={getattr(self._batch, 'decode_attn_original_replica_local_id', 'MISSING')}, "
                    f"layer_id={current_layer_id}"
                )
                next_events.extend(
                    cluster_scheduler.on_decode_attn_a2f_ready(
                        self.time,
                        self._batch,
                        replica_id=self._replica_id,
                        replica_local_id=self._replica_local_id,
                        layer_id=current_layer_id,
                        logger=logger,
                    )
                )
                return next_events

        # DECODE_FFN: emit F→A M2N transfer; do not decrement here
        if self._cluster_type == ClusterType.DECODE_FFN:
            for source_batch_id in source_batch_ids:
                raw_batch_waiting.pop(source_batch_id)

            logger.info(
                f"[ISSUE-007][F2A][RESOLVE] "
                f"{type(self._batch).__name__} {self._batch.id} "
                f"expanded to source batches {source_batch_ids}"
            )

            for original_batch in batches_for_transfer:
                for request in original_batch.requests:
                    request.on_batch_stage_end(
                        self.time,
                        ffn_execution_time,
                        ffn_execution_time,
                        ClusterType.DECODE_FFN,
                    )

            memory_usage_percent = replica_scheduler.memory_usage_percent
            for original_batch in batches_for_transfer:
                metrics_store.on_batch_end(
                    self.time,
                    original_batch,
                    self._replica_id,
                    memory_usage_percent,
                    ClusterType.DECODE_FFN,
                    self._replica_local_id,
                )

            replica_scheduler.decrement_num_running_batches()
            if activation_bytes:
                replica_scheduler.release_activation_memory_bytes(
                    activation_bytes
                )
                metrics_store.on_replica_schedule(
                    self.time,
                    self._replica_id,
                    replica_scheduler.memory_usage_percent,
                    ClusterType.DECODE_FFN,
                    replica_local_id=self._replica_local_id,
                )

            next_events.extend(prepared_m2n_events)
            next_events.append(prepared_schedule_event)
            return next_events

        if self._cluster_type == ClusterType.DECODE:
            if self._batch.is_idle:
                logger.info(
                    f"[DECODE-END][IDLE] batch_id={self._batch.id} is idle batch, skipping normal end logic"
                )
                next_events.append(
                    ReplicaScheduleEvent(
                        self.time, self._replica_id, self._cluster_type, self._replica_local_id
                    )
                )
                return next_events

            replica = cluster_scheduler._cluster.replicas[self._replica_id]
            is_moe = replica.is_moe
            # MoE layer stepping is required even for EP=1; the complete
            # one-lane EP_WAVE remains the canonical protocol.
            moe_sync_required = bool(is_moe)

            if not is_moe or not moe_sync_required:
                from frontier.events.global_batch_end_event import GlobalBatchEndEvent

                next_events.append(
                    GlobalBatchEndEvent(
                        self.time,
                        self._replica_id,
                        self._replica_local_id,
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
            active_requests = []
            active_request_ids = set()
            for request in self._batch.requests:
                if request.completed or request.id in active_request_ids:
                    continue
                active_request_ids.add(request.id)
                active_requests.append(request)

            if not active_requests:
                raise ValueError(
                    "Distributed MoE DECODE terminal batch has no active request: "
                    f"batch_id={self._batch.id}, total_layers={total_layers}"
                )

            layer_counts = {
                request.id: request.completed_layer_count
                for request in active_requests
            }
            unique_layer_counts = set(layer_counts.values())
            if len(unique_layer_counts) != 1:
                raise ValueError(
                    "Distributed MoE DECODE terminal batch has inconsistent "
                    f"active request layer counts: batch_id={self._batch.id}, "
                    f"total_layers={total_layers}, layer_counts={layer_counts}"
                )

            completed_layer_count = next(iter(unique_layer_counts))
            if completed_layer_count < total_layers:
                raise ValueError(
                    "Distributed MoE DECODE terminal layer undercount: "
                    f"batch_id={self._batch.id}, total_layers={total_layers}, "
                    f"layer_counts={layer_counts}"
                )
            if completed_layer_count > total_layers:
                raise ValueError(
                    "Distributed MoE DECODE terminal layer overflow: "
                    f"batch_id={self._batch.id}, total_layers={total_layers}, "
                    f"layer_counts={layer_counts}"
                )

            from frontier.events.global_batch_end_event import GlobalBatchEndEvent

            next_events.append(
                GlobalBatchEndEvent(
                    self.time,
                    self._replica_id,
                    self._replica_local_id,
                    self._batch,
                    self._cluster_type,
                    batch_schedule_epoch=self._batch_schedule_epoch,
                    request_execution_signatures=self._request_execution_signatures,
                    request_mutation_signatures=self._request_mutation_signatures,
                    thinking_round_start_times=self._thinking_round_start_times,
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
                        self.time, self._replica_id, self._cluster_type, self._replica_local_id
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
                    self._replica_local_id,
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
