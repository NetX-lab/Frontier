from typing import Optional, Tuple
import heapq
import logging

from frontier.entities import Batch, BatchStage, ExecutionTime, EPBatchGroup
from frontier.execution_time_predictor import BaseExecutionTimePredictor
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
        dp_id: int,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._is_moe = is_moe
        self._execution_time_predictor = execution_time_predictor
        self._cluster_type = cluster_type
        self._dp_id = dp_id

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
            "dp_id": self._dp_id,
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

    def _copy_ep_batch_metadata_to_stage(
        self, batch: Batch, batch_stage: BatchStage
    ) -> None:
        if not isinstance(batch, EPBatchGroup):
            return
        batch_stage.ep_id = int(batch.ep_id)
        batch_stage.source_batch_ids = [int(batch_id) for batch_id in batch.source_batch_ids]
        batch_stage.source_request_ids = [
            str(request_id)
            for source_batch in getattr(batch, "source_batches", [])
            for request_id in source_batch.request_ids
        ]
        batch_stage.source_request_num_tokens = [
            int(token_count)
            for source_batch in getattr(batch, "source_batches", [])
            for token_count in source_batch.num_tokens
        ]
        source_batch_arrival_times = []
        for source_batch in getattr(batch, "source_batches", []):
            if not hasattr(source_batch, "decode_ffn_m2n_arrival_time"):
                raise ValueError(
                    "DECODE_FFN EP source batch is missing "
                    "decode_ffn_m2n_arrival_time"
                )
            source_batch_arrival_times.append(
                float(source_batch.decode_ffn_m2n_arrival_time)
            )
        batch_stage.source_batch_arrival_times = source_batch_arrival_times
        if source_batch_arrival_times:
            batch_stage.source_group_ready_ts = max(source_batch_arrival_times)
        batch_stage.per_expert_tokens = {
            int(expert_id): int(token_count)
            for expert_id, token_count in batch.per_expert_tokens.items()
        }

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
        # Use heapq to maintain priority queue invariant
        # Tuple comparison: (global_id, insertion_counter) ensures correct ordering
        heapq.heappush(
            self._batch_queue,
            (batch.global_id, self._insertion_counter, batch.schedule_epoch, batch),
        )
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
        live_batch.decode_attn_original_dp_id = batch.decode_attn_original_dp_id
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
        return live_batch

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
            # heappop returns the smallest element:
            # (global_id, insertion_counter, schedule_epoch, batch)
            _, _, expected_schedule_epoch, batch = heapq.heappop(self._batch_queue)
            if batch.schedule_epoch != expected_schedule_epoch:
                self._last_stale_drop_count += 1
                continue
            live_batch = self._materialize_runtime_live_batch(batch)
            if live_batch is None:
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

        # In PD+AF disaggregation mode, DECODE_ATTN and DECODE_FFN process one
        # layer per A↔F ping-pong iteration. The loop in ClusterBatchEndEvent
        # already iterates num_layers times, so each call here must predict
        # single-layer time. Using the full num_layers would cause num_layers^2
        # overestimation (e.g., 61^2 = 3721x for a 61-layer model).
        if self._cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
            num_layers = 1
        effective_tokens_compute = batch.get_effective_total_tokens_for_compute(
            self._cluster_type
        )
        effective_tokens_transfer = batch.get_effective_total_tokens_for_transfer(
            self._cluster_type
        )
        effective_tokens_rounded = batch.get_effective_total_tokens_rounded(
            self._cluster_type
        )
        tokens_are_post_routing = isinstance(batch, EPBatchGroup)

        if not skip_get_execution_time:
            if info_logging_enabled:
                debug_logger.info(
                    "[PREDICT_STAGE_CALLING] Calling predict_stage_execution_time "
                    "for batch %s, num_layers=%s",
                    batch.id,
                    num_layers,
                )
            execution_time = self._execution_time_predictor.predict_stage_execution_time(
                batch,
                self._stage_id,
                cluster_type=self._cluster_type,
                num_layers=num_layers,
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
            self._copy_ep_batch_metadata_to_stage(batch, batch_stage)
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
        self._copy_ep_batch_metadata_to_stage(batch, batch_stage)

        return batch_stage, execution_time

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        batch = self.pop_batch_if_not_busy()
        if not batch:
            return None, None, None

        batch_stage, execution_time = self.predict_and_create_stage(batch)

        return batch, batch_stage, execution_time
