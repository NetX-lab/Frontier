import logging
from typing import List, TYPE_CHECKING

from frontier.events import BaseEvent
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.entities.batch import DenseFFNBatchGroup, EPBatchGroup
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseClusterScheduler
from frontier.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from frontier.types import EventType, ClusterType

if TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler

logger = init_logger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int,
    ):
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._cluster_type = cluster_type
        self._dp_id = dp_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    def handle_event(
        self, scheduler: "BaseGlobalScheduler", metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        """
        Schedule the next batch for a replica stage and emit synchronization events.

        Communication skip rules:
        - DP sync for MoE is needed only when a replica has more than one DP lane.
        - EP synchronization is needed only when moe_expert_parallel_size > 1.
        - Local MoE (dp_size == 1 and moe_ep_size <= 1) uses direct stage
          execution; TP communication remains an analytical predictor term.
        """
        from frontier.logger import get_cluster_logger

        debug_logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(
            self._cluster_type
        )
        stage_scheduler: ReplicaStageScheduler = (
            cluster_scheduler.get_dp_replica_stage_scheduler(
                self._replica_id, self._dp_id, self._stage_id
            )
        )

        # Debug: Check stage scheduler state before popping batch
        if debug_logger.isEnabledFor(logging.INFO):
            debug_logger.info(
                f"[STAGE] ReplicaStageScheduleEvent at {self.time:.3f}s: "
                f"replica={self._replica_id}, dp_id={self._dp_id}, stage={self._stage_id}"
            )
            # Use get_queue_batches() to get batches in priority order
            queue_batches = stage_scheduler.get_queue_batches()
            debug_logger.info(
                f"[STAGE] Stage scheduler state: is_busy={stage_scheduler.is_busy}, "
                f"queue_size={len(queue_batches)}, "
                f"queue_batches={[b.id for b in queue_batches]}"
            )

        batch = stage_scheduler.pop_batch_if_not_busy()
        stale_drop_consumer = getattr(
            stage_scheduler,
            "consume_last_stale_drop_count",
            None,
        )
        stale_drop_count = (
            stale_drop_consumer() if callable(stale_drop_consumer) else 0
        )
        replica_scheduler = None
        if stale_drop_count > 0:
            replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(
                self._replica_id,
                self._dp_id,
            )
            if replica_scheduler.num_running_batches < stale_drop_count:
                raise ValueError(
                    "Fully stale stage-drop would make num_running_batches negative: "
                    f"replica={self._replica_id}, dp_id={self._dp_id}, "
                    f"stage={self._stage_id}, stale_drop_count={stale_drop_count}, "
                    f"num_running_batches={replica_scheduler.num_running_batches}"
                )
            for _ in range(stale_drop_count):
                replica_scheduler.decrement_num_running_batches()
            debug_logger.info(
                "[STAGE][STALE-DROP-ACCOUNTING] replica=%s dp_id=%s stage=%s "
                "dropped_batches=%s num_running_batches=%s",
                self._replica_id,
                self._dp_id,
                self._stage_id,
                stale_drop_count,
                replica_scheduler.num_running_batches,
            )
        if not batch:
            if (
                stale_drop_count > 0
                and replica_scheduler is not None
                and not replica_scheduler.is_empty()
            ):
                from frontier.events.replica_schedule_event import ReplicaScheduleEvent

                debug_logger.info(
                    "[STAGE][STALE-DROP-RESCHEDULE] replica=%s dp_id=%s stage=%s",
                    self._replica_id,
                    self._dp_id,
                    self._stage_id,
                )
                return [
                    ReplicaScheduleEvent(
                        self.time,
                        self._replica_id,
                        self._cluster_type,
                        self._dp_id,
                    )
                ]
            debug_logger.info(
                f"[STAGE] No batch to schedule: is_busy={stage_scheduler.is_busy}, "
                f"queue_empty={stage_scheduler.is_empty()}"
            )
            debug_logger.info(
                f"No batch to schedule for replica {self._replica_id}, "
                f"stage {self._stage_id}, dp_id {self._dp_id}"
            )
            return []

        debug_logger.info(
            f"[STAGE] Popped batch {batch.id} for processing, "
            f"requests={[r.id for r in batch.requests]}, global_id={batch.global_id}"
        )

        self._batch = batch
        # replica = scheduler.get_replica(self._replica_id)

        # pd-af-disagg and pd-disagg
        from frontier.config.global_vars import is_disaggregated_mode

        replica = cluster_scheduler.get_replica(self._replica_id)
        is_moe = replica.is_moe
        is_monolithic_prefill_moe = (
            self._cluster_type == ClusterType.MONOLITHIC
            and is_moe
            and batch.num_prefill_tokens > 0
        )
        is_monolithic_decode_moe = (
            self._cluster_type == ClusterType.MONOLITHIC
            and is_moe
            and batch.num_prefill_tokens <= 0
            and batch.num_decode_tokens > 0
        )
        # Every MoE layer uses the canonical per-layer protocol, including
        # EP=1.  A one-lane wave still establishes the same dispatch/compute/
        # combine ordering; dense models never enter this branch.
        moe_sync_required = bool(is_moe)
        uses_prefill_sync_path = (
            (self._cluster_type == ClusterType.PREFILL and is_moe)
            or is_monolithic_prefill_moe
        ) and moe_sync_required
        uses_decode_sync_path = (
            self._cluster_type == ClusterType.DECODE_FFN and is_moe
        ) or (
            (
                (self._cluster_type == ClusterType.DECODE and is_moe)
                or is_monolithic_decode_moe
            )
            and moe_sync_required
        )

        if is_disaggregated_mode() or uses_prefill_sync_path or uses_decode_sync_path:
            # In disaggregated mode MoE PREFILL/DECODE clusters use sync events.
            # For monolithic MoE prefill/decode, reuse the same per-layer sync paths.
            if uses_prefill_sync_path or uses_decode_sync_path:
                # use sync event for MoE models only

                # Implement prefill cluster staged processing
                if uses_prefill_sync_path:
                    # Prefill cluster with MoE: implement the staged processing

                    # Use layer-by-layer DP sync path for MoE processing.
                    from frontier.events.prefill_sync_event import PrefillSyncEvent

                    # Initialize batch metadata for layer-by-layer processing
                    batch._prefill_stage_start_time = self.time

                    # Start with first layer (layer_id = 0)
                    first_layer_id = 0

                    # Predict first-layer timing directly (avoid using aggregated stage prediction).
                    execution_time = stage_scheduler._execution_time_predictor.predict_stage_execution_time(
                        batch,
                        self._stage_id,
                        self._cluster_type,
                        num_layers=1,
                        layer_id=first_layer_id,
                    )
                    # Predictor single-layer components are in milliseconds.
                    # Event queue timestamps are in seconds.
                    attention_time_ms = execution_time.get_single_layer_attention_time()

                    # Diagnostic logging for execution time
                    import math

                    if (
                        math.isnan(attention_time_ms)
                        or math.isinf(attention_time_ms)
                        or attention_time_ms < 0
                    ):
                        debug_logger.error(
                            f"[EXEC_TIME_ERROR] Invalid attention_time detected in {self._cluster_type.name}!"
                        )
                        debug_logger.error(f"  Batch ID: {batch.id}")
                        debug_logger.error(f"  Attention time (ms): {attention_time_ms}")
                        debug_logger.error(
                            f"  Total time: {execution_time.total_time if execution_time else 'None'}"
                        )
                        debug_logger.error(
                            f"  Model time: {execution_time.model_time if execution_time else 'None'}"
                        )
                        raise ValueError(f"Invalid attention_time_ms: {attention_time_ms}")

                    batch._prefill_model_execution_components_ms_by_stage = {
                        self._stage_id: [attention_time_ms]
                    }
                    attention_time = attention_time_ms * 1e-3

                    debug_logger.info(
                        f"[EXEC_TIME_OK_{self._cluster_type.name}] batch_id={batch.id}, attention_time_ms={attention_time_ms:.6f}, "
                        f"attention_time_s={attention_time:.6f}"
                    )

                    # Schedule first sync point (pre_moe) after first layer's attention computation
                    return [
                        PrefillSyncEvent(
                            self.time + attention_time,
                            self._replica_id,
                            self._stage_id,
                            batch,
                            self._dp_id,
                            "pre_moe",
                            first_layer_id,
                            attention_time,
                            cluster_type=self._cluster_type,
                        )
                    ]

                elif self._cluster_type == ClusterType.DECODE_FFN:
                    # Decode FFN cluster with MoE: implement EP dispatch + expert compute + EP combine sync
                    if isinstance(batch, DenseFFNBatchGroup):
                        batch_stage, execution_time = (
                            stage_scheduler.predict_and_create_stage(batch)
                        )
                        self._batch_stage = batch_stage
                        self._is_last_stage = stage_scheduler.is_last_stage

                        batch.execution_time = self._batch_stage.execution_time
                        self._batch_stage.on_schedule(self.time)
                        metrics_store.on_replica_stage_schedule(
                            self.time,
                            self._replica_id,
                            self._stage_id,
                            self._batch_stage,
                            execution_time,
                            self._cluster_type,
                            self._dp_id,
                        )

                        return [
                            BatchStageEndEvent(
                                self.time + self._batch_stage.execution_time,
                                self._replica_id,
                                self._stage_id,
                                self._is_last_stage,
                                self._batch,
                                self._batch_stage,
                                self._cluster_type,
                                self._dp_id,
                            )
                        ]
                    if not isinstance(batch, EPBatchGroup):
                        raise ValueError(
                            "MoE DECODE_FFN scheduling requires EPBatchGroup or "
                            f"DenseFFNBatchGroup, got {type(batch).__name__}"
                        )

                    batch_stage, execution_time = (
                        stage_scheduler.predict_and_create_stage(batch)
                    )
                    self._batch_stage = batch_stage
                    self._is_last_stage = stage_scheduler.is_last_stage

                    moe_ep_size = replica.num_moe_expert_parallel_size
                    if type(moe_ep_size) is not int or moe_ep_size <= 0:
                        raise ValueError(
                            "MoE DECODE_FFN requires a positive integer "
                            f"moe_expert_parallel_size, got {moe_ep_size!r}"
                        )

                    # Use EP dispatch -> expert compute -> combine path for
                    # every EP size.  EP=1 is a one-participant EP_WAVE, not a
                    # dense shortcut.
                    # FFN-EP runtime follows the explicit operation order:
                    # share-expert + gating + shuffling -> dispatch -> grouped_gemm -> combine.
                    pre_dispatch_compute_time_ms = (
                        execution_time.get_single_layer_moe_pre_dispatch_time()
                    )
                    expert_comp_time_ms = (
                        execution_time.get_single_layer_moe_post_dispatch_compute_time()
                    )
                    pre_dispatch_compute_time = pre_dispatch_compute_time_ms * 1e-3
                    expert_comp_time = expert_comp_time_ms * 1e-3

                    import math

                    if (
                        math.isnan(expert_comp_time)
                        or math.isinf(expert_comp_time)
                        or expert_comp_time < 0
                    ):
                        raise ValueError(
                            f"Invalid expert_comp_time: {expert_comp_time}"
                        )

                    debug_logger.info(
                        f"[EXEC_TIME_OK_FFN] batch_id={batch.id}, "
                        f"pre_dispatch_ms={pre_dispatch_compute_time_ms:.6f}, "
                        f"expert_comp_time_ms={expert_comp_time_ms:.6f}"
                    )

                    # Emit op-level traces for EP>1 before synchronization
                    self._batch_stage.on_schedule(self.time)
                    batch.record_decode_ffn_stage_execution_time_once(
                        self._stage_id,
                        self._batch_stage.execution_time,
                    )
                    metrics_store.on_replica_stage_schedule(
                        self.time,
                        self._replica_id,
                        self._stage_id,
                        self._batch_stage,
                        execution_time,
                        self._cluster_type,
                        self._dp_id,
                    )

                    # Store expert compute time for use after dispatch collective completes
                    batch.expert_compute_time = expert_comp_time

                    if debug_logger.isEnabledFor(logging.INFO):
                        debug_logger.info(
                            f"[EXEC_TIME_STAGE] batch_id={batch.id}, "
                            f"stage_execution_time={self._batch_stage.execution_time:.6f}s, "
                            f"pre_dispatch_compute_time={pre_dispatch_compute_time:.6f}s, "
                            f"expert_comp_time={expert_comp_time:.6f}s"
                        )

                    # Pre-dispatch compute must complete before EP dispatch collective starts
                    batch.time = self.time + pre_dispatch_compute_time
                    from frontier.events.ep_alltoall_dispatch_ready_event import (
                        EPAllToAllDispatchReadyEvent,
                    )

                    debug_logger.info(
                        f"[EP>1] Creating EPAllToAllDispatchReadyEvent for batch {batch.id} "
                        f"(moe_ep_size={moe_ep_size})"
                    )
                    return [
                        EPAllToAllDispatchReadyEvent(
                            batch.time,
                            self._replica_id,
                            self._stage_id,
                            batch,
                            self._dp_id,
                        )
                    ]

                elif self._cluster_type == ClusterType.DECODE or is_monolithic_decode_moe:
                    # DECODE cluster (PD-disaggregation) and MONOLITHIC pure-decode MoE
                    # reuse the same layer-by-layer decode sync processing path.

                    # Get num_layers from the predictor
                    num_layers = stage_scheduler._execution_time_predictor._num_layers_per_pipeline_stage

                    # Get execution time components for all layers in this pipeline stage
                    execution_time = stage_scheduler._execution_time_predictor.predict_stage_execution_time(
                        batch, self._stage_id, self._cluster_type, num_layers=num_layers
                    )

                    # Use layer-by-layer DP sync path for MoE processing.
                    from frontier.events.decode_sync_event import DecodeSyncEvent

                    # Initialize batch metadata for layer-by-layer processing
                    batch._decode_stage_start_time = self.time

                    # Start with first layer (layer_id = 0)
                    first_layer_id = 0
                    # Predictor single-layer attention component is in milliseconds;
                    # event queue timestamps are in seconds.
                    attention_time_ms = execution_time.get_single_layer_attention_time()
                    attention_time = attention_time_ms * 1e-3

                    # Diagnostic logging for execution time
                    import math

                    if (
                        math.isnan(attention_time_ms)
                        or math.isinf(attention_time_ms)
                        or attention_time_ms < 0
                    ):
                        debug_logger.error(
                            f"[EXEC_TIME_ERROR] Invalid attention_time detected in DECODE!"
                        )
                        debug_logger.error(f"  Batch ID: {batch.id}")
                        debug_logger.error(f"  Attention time (ms): {attention_time_ms}")
                        debug_logger.error(
                            f"  Total time: {execution_time.total_time if execution_time else 'None'}"
                        )
                        debug_logger.error(
                            f"  Model time: {execution_time.model_time if execution_time else 'None'}"
                        )
                        raise ValueError(f"Invalid attention_time_ms: {attention_time_ms}")

                    decode_cluster_name = self._cluster_type.name
                    debug_logger.info(
                        f"[EXEC_TIME_OK_{decode_cluster_name}] batch_id={batch.id}, "
                        f"attention_time_ms={attention_time_ms:.6f}, "
                        f"attention_time_s={attention_time:.6f}"
                    )

                    # Schedule first sync point (pre_moe) after first layer's attention computation
                    return [
                        DecodeSyncEvent(
                            self.time + attention_time,
                            self._replica_id,
                            self._stage_id,
                            batch,
                            self._dp_id,
                            "pre_moe",
                            first_layer_id,
                            attention_time,
                            cluster_type=self._cluster_type,
                        )
                    ]


            elif self._cluster_type == ClusterType.DECODE_ATTN:
                # decode attn cluster, without moe structure (attn only) - use direct execution
                # no sync for attn part
                batch_stage, execution_time = stage_scheduler.predict_and_create_stage(
                    batch
                )
                self._batch_stage = batch_stage
                self._is_last_stage = stage_scheduler.is_last_stage

                # Diagnostic logging for execution time
                import math
                # todo: check the component of execution time
                exec_time = self._batch_stage.execution_time
                if math.isnan(exec_time) or math.isinf(exec_time) or exec_time < 0:
                    debug_logger.error(
                        f"[EXEC_TIME_ERROR] Invalid execution time detected!"
                    )
                    debug_logger.error(f"  Batch ID: {batch.id}")
                    debug_logger.error(f"  Execution time: {exec_time}")
                    debug_logger.error(
                        f"  Total time: {execution_time.total_time if execution_time else 'None'}"
                    )
                    debug_logger.error(
                        f"  Model time: {execution_time.model_time if execution_time else 'None'}"
                    )
                    raise ValueError(f"Invalid execution time: {exec_time}")

                debug_logger.info(
                    f"[EXEC_TIME_OK] batch_id={batch.id}, exec_time={exec_time:.6f}s, "
                    f"event_time={self.time + exec_time:.6f}s"
                )

                self._batch_stage.on_schedule(self.time)
                metrics_store.on_replica_stage_schedule(
                    self.time,
                    self._replica_id,
                    self._stage_id,
                    self._batch_stage,
                    execution_time,
                    self._cluster_type,
                    self._dp_id,
                )

                return [
                    BatchStageEndEvent(
                        self.time + exec_time,
                        self._replica_id,
                        self._stage_id,
                        self._is_last_stage,
                        self._batch,
                        self._batch_stage,
                        self._cluster_type,
                        self._dp_id,
                    ),
                ]

            elif self._cluster_type == ClusterType.DECODE_FFN and not is_moe:
                # Dense DECODE_FFN: single-layer direct execution, no EP dispatch/combine
                batch_stage, execution_time = stage_scheduler.predict_and_create_stage(
                    batch
                )
                self._batch_stage = batch_stage
                self._is_last_stage = stage_scheduler.is_last_stage

                batch.execution_time = self._batch_stage.execution_time
                self._batch_stage.on_schedule(self.time)
                metrics_store.on_replica_stage_schedule(
                    self.time,
                    self._replica_id,
                    self._stage_id,
                    self._batch_stage,
                    execution_time,
                    self._cluster_type,
                    self._dp_id,
                )

                return [
                    BatchStageEndEvent(
                        self.time + self._batch_stage.execution_time,
                        self._replica_id,
                        self._stage_id,
                        self._is_last_stage,
                        self._batch,
                        self._batch_stage,
                        self._cluster_type,
                        self._dp_id,
                    ),
                ]

            elif (
                self._cluster_type in [ClusterType.PREFILL, ClusterType.DECODE]
                and not is_moe
            ):
                # Dense model path: simplified processing without sync events
                # This matches Vidur's approach for co-location mode
                # All layers in the pipeline stage are processed in one shot

                debug_logger.info(
                    f"[DENSE_MODEL] Processing dense model in {self._cluster_type.name} cluster, "
                    f"batch_id={batch.id}, replica={self._replica_id}, stage={self._stage_id}"
                )

                # Get num_layers_per_pipeline_stage from predictor
                num_layers = stage_scheduler._execution_time_predictor._num_layers_per_pipeline_stage

                debug_logger.info(
                    f"[DENSE_MODEL] Processing {num_layers} layers in one shot"
                )

                # Predict execution time for all layers in this pipeline stage
                execution_time = stage_scheduler._execution_time_predictor.predict_stage_execution_time(
                    batch,
                    self._stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=num_layers,  # All layers in one shot
                )

                # Create batch stage
                from frontier.entities import BatchStage

                total_execution_time = execution_time.total_time
                model_execution_time = execution_time.model_time
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
                batch_stage = BatchStage(
                    batch.id,
                    self._replica_id,
                    self._stage_id,
                    total_execution_time,
                    model_execution_time,
                    batch.requests,
                    batch.num_tokens,
                    cluster_type=self._cluster_type,
                    effective_total_tokens_compute=effective_tokens_compute,
                    effective_total_tokens_transfer=effective_tokens_transfer,
                    effective_total_tokens_rounded=effective_tokens_rounded,
                    tokens_are_post_routing=tokens_are_post_routing,
                )

                # Mark stage as busy
                stage_scheduler._is_busy = True
                self._batch_stage = batch_stage
                self._is_last_stage = stage_scheduler.is_last_stage

                # Diagnostic logging for execution time
                import math

                exec_time = self._batch_stage.execution_time
                if math.isnan(exec_time) or math.isinf(exec_time) or exec_time < 0:
                    debug_logger.error(
                        f"[EXEC_TIME_ERROR] Invalid execution time detected in dense model path!"
                    )
                    debug_logger.error(f"  Batch ID: {batch.id}")
                    debug_logger.error(f"  Execution time: {exec_time}")
                    debug_logger.error(f"  num_layers: {num_layers}")
                    debug_logger.error(
                        f"  Total time: {execution_time.total_time if execution_time else 'None'}"
                    )
                    debug_logger.error(
                        f"  Model time: {execution_time.model_time if execution_time else 'None'}"
                    )
                    raise ValueError(f"Invalid execution time: {exec_time}")

                debug_logger.info(
                    f"[DENSE_MODEL] Execution time: {exec_time:.6f}s for {num_layers} layers, "
                    f"event_time={self.time + exec_time:.6f}s"
                )

                # Record metrics
                self._batch_stage.on_schedule(self.time)
                metrics_store.on_replica_stage_schedule(
                    self.time,
                    self._replica_id,
                    self._stage_id,
                    self._batch_stage,
                    execution_time,
                    self._cluster_type,
                    self._dp_id,
                )

                # Schedule batch stage end event directly (no sync events)
                return [
                    BatchStageEndEvent(
                        self.time + self._batch_stage.execution_time,
                        self._replica_id,
                        self._stage_id,
                        self._is_last_stage,
                        self._batch,
                        self._batch_stage,
                        self._cluster_type,
                        self._dp_id,
                    )
                ]

        # for local MoE or dense model, we donot need to use moe_sync_event, because
        # batches are same across tp ranks (tp_size == ep_size or ep_size == 1)
        try:
            replica = cluster_scheduler.get_replica(self._replica_id)
        except KeyError as e:
            from frontier.logger import get_cluster_logger

            cluster_logger = get_cluster_logger(__name__, self._cluster_type.name)
            cluster_logger.error(
                f"Failed to get replica {self._replica_id} from cluster {self._cluster_type.name}"
            )
            cluster_logger.error(
                f"Available replica IDs: {list(cluster_scheduler._cluster.replicas.keys())}"
            )
            raise e
        
        batch_stage, execution_time = stage_scheduler.predict_and_create_stage(
            batch
        )
        self._batch_stage = batch_stage
        self._is_last_stage = stage_scheduler.is_last_stage

        # Diagnostic logging for execution time
        import math

        exec_time = self._batch_stage.execution_time
        if math.isnan(exec_time) or math.isinf(exec_time) or exec_time < 0:
            debug_logger.error(
                f"[EXEC_TIME_ERROR] Invalid execution time detected (generic path)!"
            )
            debug_logger.error(f"  Batch ID: {batch.id}")
            debug_logger.error(f"  Execution time: {exec_time}")
            debug_logger.error(
                f"  Total time: {execution_time.total_time if execution_time else 'None'}"
            )
            debug_logger.error(
                f"  Model time: {execution_time.model_time if execution_time else 'None'}"
            )
            raise ValueError(f"Invalid execution time: {exec_time}")

        debug_logger.info(
            f"[EXEC_TIME_OK_GENERIC] batch_id={batch.id}, exec_time={exec_time:.6f}s"
        )

        self._batch_stage.on_schedule(self.time)
        metrics_store.on_replica_stage_schedule(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
            self._cluster_type,
            self._dp_id,
        )

        return [
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
                self._cluster_type,
                self._dp_id,
            ),
        ]



    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "cluster_type": self._cluster_type.name,
            "dp_id": self._dp_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
