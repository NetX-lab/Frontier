from typing import Optional, Tuple
import heapq
import logging

from frontier.entities import Batch, BatchStage, ExecutionTime, EPBatchGroup
from frontier.entities.batch import DenseFFNBatchGroup
from frontier.execution_time_predictor import BaseExecutionTimePredictor
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageAdmissionTicket,
    StageExecutionContext,
)
from frontier.types import ClusterType


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        is_moe: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        cluster_type: ClusterType,
        replica_local_id: int | None,
        stage_execution_context: StageExecutionContext,
    ) -> None:
        if not isinstance(stage_execution_context, StageExecutionContext):
            raise TypeError(
                "stage_execution_context must be a StageExecutionContext"
            )
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._is_moe = is_moe
        self._execution_time_predictor = execution_time_predictor
        self._cluster_type = cluster_type
        self._replica_local_id = replica_local_id
        self._stage_execution_context = stage_execution_context

        # Priority queue implementation to prevent EP synchronization deadlock
        # Batches are ordered by (global_id, insertion_order) to ensure:
        # 1. Batches with smaller global_id are always processed first
        # 2. Batches with the same global_id maintain FIFO order (deterministic)
        # This prevents circular dependencies where batch A waits for batch B,
        # but batch B is blocked in the queue by batch C that waits for batch A.
        self._batch_queue = []  # Priority queue: list of (global_id, insertion_counter, schedule_epoch, batch)
        self._insertion_counter = 0  # Monotonically increasing counter for FIFO tie-breaking
        self._is_busy = False
        self._last_stale_drop_count = 0

    # gurantee only one batch is in current stage at a time;
    # other batches are in the self._batch_queue
    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def get_debug_state(self) -> dict:
        """Return scheduler state for fail-fast sequential-end diagnostics."""
        queued_batches = self.get_queue_batches()
        return {
            "replica_id": self._replica_id,
            "replica_local_id": self._replica_local_id,
            "stage_id": self._stage_id,
            "is_busy": bool(self._is_busy),
            "is_empty": self.is_empty(),
            "batch_queue": {
                "count": len(queued_batches),
                "batch_ids": [batch.id for batch in queued_batches],
                "batch_global_ids": [
                    getattr(batch, "global_id", None) for batch in queued_batches
                ],
                "request_ids": [
                    list(getattr(batch, "request_ids", []))
                    for batch in queued_batches
                ],
            },
        }

    def _copy_source_batch_metadata_to_stage(
        self, batch: Batch, batch_stage: BatchStage
    ) -> None:
        if not isinstance(batch, (EPBatchGroup, DenseFFNBatchGroup)):
            return

        source_batches = getattr(batch, "source_batches", None)
        if not isinstance(source_batches, (list, tuple)) or not source_batches:
            raise ValueError(
                "DECODE_FFN synthetic batch requires non-empty source_batches"
            )
        batch_stage.source_batch_ids = [int(batch_id) for batch_id in batch.source_batch_ids]
        batch_stage.source_request_ids = [
            str(request_id)
            for source_batch in source_batches
            for request_id in source_batch.request_ids
        ]
        batch_stage.source_request_runtime_epochs = [
            int(runtime_epoch)
            for source_batch in source_batches
            for runtime_epoch in source_batch.request_runtime_epochs
        ]
        batch_stage.source_request_num_tokens = [
            int(token_count)
            for source_batch in source_batches
            for token_count in source_batch.num_tokens
        ]
        source_batch_arrival_times = []
        for source_batch in source_batches:
            if not hasattr(source_batch, "decode_ffn_m2n_arrival_time"):
                raise ValueError(
                    "DECODE_FFN source batch is missing "
                    "decode_ffn_m2n_arrival_time"
                )
            source_batch_arrival_times.append(
                float(source_batch.decode_ffn_m2n_arrival_time)
            )
        batch_stage.source_batch_arrival_times = source_batch_arrival_times
        batch_stage.source_group_ready_ts = max(source_batch_arrival_times)

        if isinstance(batch, EPBatchGroup):
            batch_stage.ep_id = int(batch.ep_id)
            batch_stage.attach_lane_workload(batch.lane_workload)

    def add_batch(self, batch: Batch) -> None:
        """
        Add a batch to the priority queue.

        Batches are ordered by (global_id, insertion_counter) to ensure:
        - Batches with smaller global_id are processed first (prevents deadlock)
        - Batches with same global_id maintain FIFO order (deterministic)

        microbatch is organized similar to batch and all microbatches used for 
        pd-af will be put into the queue in a while loop 
        in base_replica_scheduler.py 's scheudle method

        Args:
            batch: The batch to add to the queue
        """
        admission_ticket = getattr(batch, "_stage_admission_ticket", None)
        if admission_ticket is None:
            if self._cluster_type == ClusterType.DECODE_FFN and isinstance(
                batch, EPBatchGroup
            ):
                raise ValueError(
                    "DECODE_FFN EPBatchGroup must carry a complete EP_WAVE "
                    "admission ticket before queue insertion"
                )
            operation_id = (
                "stage_batch",
                int(batch.id),
                int(batch.schedule_epoch),
            )
            admission_ticket = self._stage_execution_context.enqueue_full_stage(
                operation_id=operation_id,
            )
            batch._stage_admission_ticket = admission_ticket

        # Use heapq to maintain priority queue invariant
        # Tuple comparison: (global_id, insertion_counter) ensures correct ordering
        queue_item = None
        try:
            queue_item = (
                batch.global_id,
                self._insertion_counter,
                batch.schedule_epoch,
                batch,
            )
            heapq.heappush(self._batch_queue, queue_item)
        except Exception:
            if queue_item is not None:
                self._batch_queue[:] = [
                    queued_item
                    for queued_item in self._batch_queue
                    if queued_item is not queue_item
                ]
                heapq.heapify(self._batch_queue)
            if self._stage_execution_context.is_queued(admission_ticket):
                self._stage_execution_context.cancel(admission_ticket)
            batch.__dict__.pop("_stage_admission_ticket", None)
            raise
        self._insertion_counter += 1

    def on_stage_end(self) -> None:
        self._is_busy = False

    def consume_last_stale_drop_count(self) -> int:
        count = self._last_stale_drop_count
        self._last_stale_drop_count = 0
        return count

    def _materialize_runtime_live_batch(self, batch: Batch) -> Optional[Batch]:
        live_indices = [
            index
            for index in range(len(batch.requests))
            if batch._request_execution_matches_snapshot(index)
        ]
        if not live_indices:
            return None
        if len(live_indices) == len(batch.requests):
            return batch

        live_requests = [batch.requests[index] for index in live_indices]
        live_num_tokens = [batch.num_tokens[index] for index in live_indices]
        live_batch = Batch(
            replica_id=batch.replica_id,
            requests=live_requests,
            num_tokens=live_num_tokens,
            is_idle=batch.is_idle,
            is_moe=batch.is_moe,
        )
        live_batch._id = batch.id
        live_batch.set_global_id(batch.global_id)
        live_batch.decode_attn_original_replica_id = (
            batch.decode_attn_original_replica_id
        )
        live_batch.decode_attn_original_replica_local_id = batch.decode_attn_original_replica_local_id
        live_batch.decode_cuda_graph_metadata = batch.decode_cuda_graph_metadata
        live_batch.afd_stage_idx = batch.afd_stage_idx
        live_batch.afd_stage_metadata = batch.afd_stage_metadata
        live_batch.spec_decode_metadata = batch.spec_decode_metadata
        live_batch.time = batch.time
        live_batch._scheduled = batch.scheduled
        live_batch._scheduled_at = batch._scheduled_at
        live_batch._schedule_epoch = batch.schedule_epoch
        live_batch._request_execution_signatures = [
            batch.request_execution_signatures[index] for index in live_indices
        ]
        live_batch._thinking_round_start_times = [
            batch.thinking_round_start_times[index] for index in live_indices
        ]
        if hasattr(batch, "_stage_admission_ticket"):
            live_batch._stage_admission_ticket = batch._stage_admission_ticket
        return live_batch

    def _drop_queued_lanes_for_ticket(
        self, admission_ticket: StageAdmissionTicket
    ) -> int:
        """Drop every queued sibling that belongs to one invalid EP wave."""

        retained = []
        dropped = 0
        for queue_item in self._batch_queue:
            queued_batch = queue_item[3]
            if getattr(queued_batch, "_stage_admission_ticket", None) == admission_ticket:
                dropped += 1
                queued_batch.__dict__.pop("_stage_admission_ticket", None)
            else:
                retained.append(queue_item)
        if dropped:
            self._batch_queue = retained
            heapq.heapify(self._batch_queue)
        return dropped

    def _discard_stale_ticket(self, admission_ticket: StageAdmissionTicket) -> None:
        """Cancel a queued stale ticket without touching another active wave."""

        context = self._stage_execution_context
        if context.is_queued(admission_ticket):
            context.cancel(admission_ticket)
        elif context.is_active(admission_ticket):
            # A sibling lane may already own the wave. Its active ticket is
            # released only at the true completion boundary.
            return
        elif not context.is_cancelled(admission_ticket):
            raise ValueError(
                "stale batch carries an unknown stage admission ticket: "
                f"{admission_ticket.operation_id!r}"
            )

    def pop_batch_if_not_busy(self) -> Batch:
        """
        Pop the batch with smallest (global_id, insertion_counter) from the queue.

        Returns None if:
        - The stage is busy processing another batch
        - The queue is empty

        Returns:
            The batch with smallest global_id, or None if cannot pop
        """
        self._last_stale_drop_count = 0
        if self._is_busy or not self._batch_queue:
            return None
        while self._batch_queue:
            # Inspect and admit the same queue head. The old implementation
            # admitted the first ticket before a stale-drop loop could pop a
            # different batch, allowing that batch to bypass the parent owner.
            _, _, expected_schedule_epoch, batch = self._batch_queue[0]
            admission_ticket = getattr(batch, "_stage_admission_ticket", None)
            if not isinstance(admission_ticket, StageAdmissionTicket):
                raise ValueError(
                    "Queued batch must carry a StageAdmissionTicket"
                )
            if self._stage_execution_context.is_cancelled(admission_ticket):
                heapq.heappop(self._batch_queue)
                batch.__dict__.pop("_stage_admission_ticket", None)
                self._last_stale_drop_count += 1
                self._last_stale_drop_count += (
                    self._drop_queued_lanes_for_ticket(admission_ticket)
                )
                continue
            if batch.schedule_epoch != expected_schedule_epoch:
                heapq.heappop(self._batch_queue)
                self._discard_stale_ticket(admission_ticket)
                batch.__dict__.pop("_stage_admission_ticket", None)
                self._last_stale_drop_count += (
                    self._drop_queued_lanes_for_ticket(admission_ticket)
                )
                self._last_stale_drop_count += 1
                continue
            parent_acquired = False
            if not self._stage_execution_context.owns(admission_ticket):
                if not self._stage_execution_context.try_acquire(admission_ticket):
                    return None
                parent_acquired = True
            # Remove the same candidate whose ticket was just acquired.
            heapq.heappop(self._batch_queue)
            live_batch = self._materialize_runtime_live_batch(batch)
            if live_batch is None:
                context = self._stage_execution_context
                if parent_acquired and context.is_active(admission_ticket):
                    context.release(admission_ticket)
                elif context.is_queued(admission_ticket):
                    context.cancel(admission_ticket)
                batch.__dict__.pop("_stage_admission_ticket", None)
                self._last_stale_drop_count += (
                    self._drop_queued_lanes_for_ticket(admission_ticket)
                )
                self._last_stale_drop_count += 1
                continue
            self._is_busy = True
            return live_batch
        return None

    def get_queue_batches(self):
        """
        Get list of batches currently in the queue (for debugging/logging).

        Returns batches in priority order (smallest global_id first).

        Returns:
            List of Batch objects in priority order
        """
        # Return batches sorted by priority (global_id, insertion_counter)
        # This is used by logging code that accesses _batch_queue directly
        return [batch for _, _, _, batch in sorted(self._batch_queue)]

    def predict_and_create_stage(
        self, batch: Batch, skip_get_execution_time: bool = False
    ) -> Tuple[BatchStage, ExecutionTime]:
        from frontier.logger import get_cluster_logger
        debug_logger = get_cluster_logger(__name__, self._cluster_type.name)
        info_logging_enabled = debug_logger.isEnabledFor(logging.INFO)
        if info_logging_enabled:
            debug_logger.info(
                "[PREDICT_STAGE_ENTER] batch_id=%s, cluster=%s, "
                "skip_get_execution_time=%s",
                batch.id,
                self._cluster_type.name,
                skip_get_execution_time,
            )

        # Phase 2: Unified API for both MoE and dense models
        # Get num_layers from the execution time predictor (calculated from model config)
        num_layers = self._execution_time_predictor._num_layers_per_pipeline_stage
        layer_ids = None

        # In PD+AF disaggregation mode, DECODE_ATTN and DECODE_FFN process one
        # layer per A↔F ping-pong iteration. The loop in ClusterBatchEndEvent
        # already iterates num_layers times, so each call here must predict
        # single-layer time. Using the full num_layers would cause num_layers^2
        # overestimation (e.g., 61^2 = 3721x for a 61-layer model).
        if self._cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            num_layers = 1
        layer_id = 0
        if self._cluster_type == ClusterType.DECODE_FFN:
            layer_id = getattr(batch, "decode_ffn_layer_id", None)
            if layer_id is None:
                raise ValueError(
                    "DECODE_FFN batch is missing decode_ffn_layer_id"
                )
        elif self._cluster_type not in (
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            # A regular stage owns a global, contiguous layer interval.  Keep
            # the existing scalar ``layer_id`` as the first-layer compatibility
            # field while carrying the complete identity for typed aggregates.
            from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
                BaseClusterScheduler,
            )

            first_layer_id, layer_end = (
                BaseClusterScheduler.get_pipeline_stage_layer_bounds(
                    self._stage_id,
                    num_layers,
                )
            )
            layer_id = first_layer_id
            layer_ids = tuple(range(first_layer_id, layer_end))
        effective_tokens_compute = batch.get_effective_total_tokens_for_compute(
            self._cluster_type
        )
        effective_tokens_transfer = batch.get_effective_total_tokens_for_transfer(
            self._cluster_type
        )
        effective_tokens_rounded = batch.get_effective_total_tokens_rounded(
            self._cluster_type
        )
        tokens_are_post_routing = isinstance(
            batch, (EPBatchGroup, DenseFFNBatchGroup)
        )

        if not skip_get_execution_time:
            if info_logging_enabled:
                debug_logger.info(
                    "[PREDICT_STAGE_CALLING] Calling predict_stage_execution_time "
                    "for batch %s, num_layers=%s",
                    batch.id,
                    num_layers,
                )
            prediction_kwargs = {
                "cluster_type": self._cluster_type,
                "num_layers": num_layers,
                "layer_id": layer_id,
            }
            if layer_ids is not None:
                prediction_kwargs["layer_ids"] = layer_ids
            execution_time = self._execution_time_predictor.predict_stage_execution_time(
                batch,
                self._stage_id,
                **prediction_kwargs,
            )
            if info_logging_enabled:
                debug_logger.info(
                    "[PREDICT_STAGE_RETURNED] batch_id=%s, total_time=%s",
                    batch.id,
                    execution_time.total_time if execution_time else "None",
                )
        else:
            batch_stage = BatchStage(
                batch.id,
                self._replica_id,
                self._stage_id,
                0,
                0,
                batch.requests,
                batch.num_tokens,
                self._cluster_type,
                effective_total_tokens_compute=effective_tokens_compute,
                effective_total_tokens_transfer=effective_tokens_transfer,
                effective_total_tokens_rounded=effective_tokens_rounded,
                tokens_are_post_routing=tokens_are_post_routing,
            )
            self._copy_source_batch_metadata_to_stage(batch, batch_stage)
            batch_stage.attach_runtime_identity(batch)
            return batch_stage, None

        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
            self._cluster_type,
            effective_total_tokens_compute=effective_tokens_compute,
            effective_total_tokens_transfer=effective_tokens_transfer,
            effective_total_tokens_rounded=effective_tokens_rounded,
            tokens_are_post_routing=tokens_are_post_routing,
        )
        self._copy_source_batch_metadata_to_stage(batch, batch_stage)
        batch_stage.attach_runtime_identity(batch)

        return batch_stage, execution_time

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        batch = self.pop_batch_if_not_busy()
        if not batch:
            return None, None, None

        batch_stage, execution_time = self.predict_and_create_stage(batch)

        return batch, batch_stage, execution_time
