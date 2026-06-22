import json
import os
from functools import reduce
from typing import Dict, List, TYPE_CHECKING, Any, Optional

import pandas as pd
import plotly.express as px

from frontier.config import SimulationConfig, ClusterConfig, get_quantization_manager
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.entities import Batch, BatchStage, ExecutionTime, Request
from frontier.logger import get_cluster_logger

if TYPE_CHECKING:
    from frontier.metrics.trace_store import TraceStore
from frontier.metrics.cdf_sketch import CDFSketch
from frontier.metrics.trace_store import TraceStore
from frontier.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CpuOperationMetrics,
    OperationMetrics,
    RequestCompletionMetricsTimeSeries,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenCompletionMetricsTimeSeries,
    TokenMetricsTimeDistribution,
)
from frontier.metrics.data_series import DataSeries
from frontier.metrics.series_average_meter import SeriesAverageMeter
from frontier.metrics.trace_store import TraceStore
from frontier.metrics.op_trace_utils import (
    OpTraceContext,
    build_kv_cache_transfer_meta,
    build_parallel_context,
    compute_op_trace_meta,
)
from frontier.metrics.wandb_utils import get_wandb, require_wandb
from frontier.utils.mfu_calculator import MFUCalculator
from frontier.types import ClusterType

wandb = get_wandb()


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"


def _round_ledger_ms(value: float) -> float:
    return round(float(value), 9)


class MetricsStore:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        cluster_configs: Dict[ClusterType, ClusterConfig],
        trace_store: Optional["TraceStore"] = None,
    ) -> None:
        self._simulation_config = simulation_config
        self._config = self._simulation_config.metrics_config
        self._trace_store = trace_store
        self._last_request_arrived_at = None

        # Completion tracking for termination logic
        self._total_requests = 0
        self._completed_requests = 0
        self._completed_request_ids = set()

        self._cluster_configs = cluster_configs

        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise batch metrics per cluster
        self._batch_metrics_count_distribution: Dict[
            ClusterType, Dict[BatchMetricsCountDistribution, CDFSketch]
        ] = {}
        self._batch_metrics_count_distribution_per_batch: Dict[
            ClusterType, Dict[BatchMetricsCountDistribution, DataSeries]
        ] = {}
        self._batch_metrics_time_distribution: Dict[
            ClusterType, Dict[BatchMetricsTimeDistribution, CDFSketch]
        ] = {}
        self._batch_metrics_time_distribution_per_batch: Dict[
            ClusterType, Dict[BatchMetricsTimeDistribution, DataSeries]
        ] = {}

        # Initialise operation metrics per cluster
        self._operation_metrics: Dict[
            ClusterType, Dict[OperationMetrics, CDFSketch]
        ] = {}
        self._operation_metrics_per_batch: Dict[
            ClusterType, Dict[OperationMetrics, DataSeries]
        ] = {}
        self._cpu_operation_metrics: Dict[
            ClusterType, Dict[CpuOperationMetrics, CDFSketch]
        ] = {}
        self._cpu_operation_metrics_per_batch: Dict[
            ClusterType, Dict[CpuOperationMetrics, DataSeries]
        ] = {}

        # per replica metrics per cluster
        self._replica_memory_usage: Dict[
            ClusterType, List[List[SeriesAverageMeter]]
        ] = {}
        # per replica stage metrics per cluster
        self._replica_busy_time: Dict[
            ClusterType, List[List[List[SeriesAverageMeter]]]
        ] = {}
        self._replica_mfu: Dict[ClusterType, List[List[List[SeriesAverageMeter]]]] = {}
        self._mfu_calculator: Dict[ClusterType, MFUCalculator] = {}
        self._pending_frontier_stage_batch_ledger_rows: Dict[int, dict[str, Any]] = {}
        self._pending_frontier_stage_batch_ledger_row_keys: Dict[
            tuple[str, int, int, int, int], int
        ] = {}
        self._pending_frontier_stage_batch_ledger_rows_by_key: Dict[
            tuple[str, int, int, int, int], dict[str, Any]
        ] = {}
        self._frontier_stage_batch_ledger_rows: list[dict[str, Any]] = []
        self._frontier_stage_batch_ledger_summary = (
            self._new_frontier_stage_batch_ledger_summary()
        )
        self._frontier_stage_batch_ledger_summary_groups: dict[
            tuple[int, int, float, tuple[int, ...]], dict[str, Any]
        ] = {}

        for cluster_type, cluster_config in self._cluster_configs.items():
            self._init_per_cluster_metrics(cluster_type, cluster_config)

        # Initialise completion metrics (these are global)
        self._request_completion_metrics_time_series: Dict[
            RequestCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestCompletionMetricsTimeSeries:
            self._request_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._token_completion_metrics_time_series: Dict[
            TokenCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in TokenCompletionMetricsTimeSeries:
            self._token_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialize KV cache transfer metrics
        self._kv_cache_transfer_metrics = {
            "transfer_count": 0,
            "total_transfer_time": 0.0,
            "total_data_transferred": 0,
            "transfer_times": DataSeries(
                "Transfer ID",
                "Transfer Time (ms)",
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            ),
            "transfer_sizes": DataSeries(
                "Transfer ID",
                "Transfer Size (bytes)",
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            ),
        }

        # Initialize preemption statistics collector for system-level metrics
        # NOTE: Separation of Concerns - Request-Level vs. System-Level Metrics
        # - Request-level metrics (request_metrics.csv): Per-request preemption details
        #   (e.g., how many times each request was preempted, tokens-at-preemption statistics)
        # - System-level metrics (system_metrics.json): Cluster-wide aggregate statistics
        #   (e.g., total preemption events, preemption rate, tokens-at-preemption distribution)
        # This stores per-cluster aggregate statistics computed from all requests
        self._preemption_statistics = {
            "total_preemption_events": 0,  # Total preemption events across all clusters
            "preempted_request_ids": set(),  # Set of unique request IDs that were preempted
            "by_cluster_type": {
                ClusterType.MONOLITHIC: {
                    "preemption_events": 0,
                    "preempted_request_ids": set(),
                    "tokens_at_preemption": [],  # Not applicable for MONOLITHIC
                },
                ClusterType.PREFILL: {
                    "preemption_events": 0,
                    "preempted_request_ids": set(),
                    "tokens_at_preemption": [],  # Not applicable for PREFILL
                },
                ClusterType.DECODE: {
                    "preemption_events": 0,
                    "preempted_request_ids": set(),
                    "tokens_at_preemption": [],  # List of all tokens-at-preemption values
                },
                ClusterType.DECODE_ATTN: {
                    "preemption_events": 0,
                    "preempted_request_ids": set(),
                    "tokens_at_preemption": [],  # List of all tokens-at-preemption values
                },
                ClusterType.DECODE_FFN: {
                    "preemption_events": 0,
                    "preempted_request_ids": set(),
                    "tokens_at_preemption": [],  # Not applicable for DECODE_FFN
                },
            },
        }

        # Per-layer expansion tracking (cluster-aware): request IDs traced per cluster.
        # Limited by num_requests_to_trace_per_layer config for each cluster.
        self._per_layer_traced_requests_by_cluster = {
            cluster_type: set() for cluster_type in self._cluster_configs
        }
        self._metrics_ground_truth_trace_path = os.path.join(
            self._config.output_dir,
            self._config.metrics_ground_truth_trace_file,
        )
        if self._config.enable_metrics_ground_truth_trace:
            os.makedirs(self._config.output_dir, exist_ok=True)
            with open(self._metrics_ground_truth_trace_path, "w", encoding="utf-8"):
                pass

        self._init_wandb()

    @property
    def trace_store(self) -> Optional["TraceStore"]:
        """Get the trace store for op-level tracing."""
        return self._trace_store

    def _write_metrics_ground_truth_record(self, record: dict[str, Any]) -> None:
        if not self._config.enable_metrics_ground_truth_trace:
            return
        with open(self._metrics_ground_truth_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # =========================================================================
    # Op-Level Tracing: Emit Traces for Stage Execution
    # =========================================================================

    def _build_op_trace_context(
        self, batch_stage: BatchStage, cluster_type: ClusterType
    ) -> OpTraceContext:
        cluster_config = self._cluster_configs.get(cluster_type)
        if cluster_config is None:
            raise ValueError(f"Cluster config not found for {cluster_type}")
        total_tokens = sum(batch_stage.num_tokens)
        return OpTraceContext(
            cluster_type=cluster_type,
            model_config=cluster_config.replica_config.model_config,
            replica_config=cluster_config.replica_config,
            total_tokens=total_tokens,
            effective_tokens_compute=batch_stage.effective_total_tokens_compute,
            effective_tokens_transfer=batch_stage.effective_total_tokens_transfer,
            effective_tokens_rounded=batch_stage.effective_total_tokens_rounded,
            tokens_are_post_routing=batch_stage.tokens_are_post_routing,
        )

    def _emit_op_level_traces(
        self,
        time: float,
        batch_stage: BatchStage,
        replica_id: int,
        execution_time: ExecutionTime,
        cluster_type: ClusterType,
        request_ids: List[str] = None,
    ) -> None:
        """
        Emit op-level trace events for a batch stage execution.

        Unrolls the ExecutionTime object into a sequence of trace events,
        placing them sequentially on the timeline (cursor-based timing).

        Semantic Model: This represents PREDICTED execution, not actual
        hardware execution. Operations are serialized; overlap is not modeled.

        Per-layer expansion: When enable_per_layer_expansion is True and we
        haven't traced num_requests_to_trace_per_layer requests yet, emit
        individual layer traces instead of aggregated spans.

        Args:
            time: Simulation time when stage execution begins (seconds)
            batch_stage: BatchStage with request and token context
            replica_id: Replica ID
            execution_time: ExecutionTime object with all component times
            cluster_type: Type of cluster (PREFILL, DECODE_ATTN, DECODE_FFN, etc.)
            request_ids: List of request IDs in the batch (for traceability)
        """
        if not self._trace_store or not self._config.enable_op_level_tracing:
            return

        from frontier.metrics.trace_store import TraceEvent

        # Determine if we should emit per-layer traces for this batch
        should_expand_layers = self._should_expand_layers(cluster_type, request_ids)

        trace_context = self._build_op_trace_context(batch_stage, cluster_type)
        parallel_context = build_parallel_context(trace_context)
        cluster_config = self._cluster_configs.get(cluster_type)
        model_name = (
            cluster_config.replica_config.model_name if cluster_config else "unknown"
        )
        model_config = cluster_config.replica_config.model_config if cluster_config else None
        skip_ffn_attn_norm_residual = (
            cluster_type == ClusterType.DECODE_FFN
            and model_config is not None
            and model_config.is_step3_text()
        )
        skip_add_attn_residual = (
            cluster_type == ClusterType.DECODE_ATTN
            and model_config is not None
            and model_config.is_step3_text()
        )
        use_step3_ep_alltoall = (
            model_config is not None
            and model_config.model_type == "step3_text"
            and cluster_type
            in (
                ClusterType.PREFILL,
                ClusterType.DECODE,
                ClusterType.DECODE_FFN,
                ClusterType.MONOLITHIC,
            )
        )
        use_ep_alltoall_dispatch_combine = self._use_ep_alltoall_dispatch_combine(
            cluster_type=cluster_type,
            batch_stage=batch_stage,
        )

        # Convert seconds to milliseconds for cursor tracking
        cursor_ms = time * 1000.0
        cluster_name = cluster_type.name
        num_layers = execution_time.num_layers

        # Common metadata for all events in this stage
        base_meta = {
            "num_tokens": batch_stage.num_tokens,
            "total_tokens": trace_context.total_tokens,
            "effective_total_tokens_compute": trace_context.effective_tokens_compute,
            "effective_total_tokens_transfer": trace_context.effective_tokens_transfer,
            "effective_total_tokens_rounded": trace_context.effective_tokens_rounded,
            "tokens_are_post_routing": batch_stage.tokens_are_post_routing,
            "num_layers": num_layers,
            "request_ids": request_ids or [],
            "model_name": model_name,
            "parallel_context": parallel_context,
        }

        def emit(
            op_type: str,
            op_name: str,
            duration_ms: float,
            layer_id: int = -1,
            extra_meta: dict = None,
        ):
            """Helper to emit a single trace event and advance cursor."""
            nonlocal cursor_ms
            if duration_ms <= 0:
                return  # Skip zero-duration ops

            if extra_meta is None and op_type in ("COMPUTE", "COMM"):
                extra_meta = compute_op_trace_meta(op_name, op_type, trace_context)

            if op_type in ("COMPUTE", "COMM") and not extra_meta:
                raise ValueError(
                    f"Missing op trace metadata for op={op_name} type={op_type}"
                )

            meta = base_meta.copy()
            if extra_meta:
                meta.update(extra_meta)

            event = TraceEvent(
                type=op_type,
                name=op_name,
                ts_start=cursor_ms / 1000.0,  # Convert back to seconds for ts_start
                duration_ms=duration_ms,
                cluster=cluster_name,
                replica_id=replica_id,
                batch_id=batch_stage._batch_id,
                layer_id=layer_id,
                meta=meta,
            )
            self._trace_store.log_event(event)
            cursor_ms += duration_ms

        def emit_related_collective_wait_traces() -> None:
            wait_specs = getattr(execution_time, "_trace_related_collective_waits", None)
            if not isinstance(wait_specs, list) or not wait_specs:
                return

            for spec in wait_specs:
                if not isinstance(spec, dict):
                    continue
                base_op_name = str(spec.get("op_name") or "").strip()
                if not base_op_name:
                    continue
                related_wait_ms = float(spec.get("related_wait_ms", 0.0) or 0.0)
                per_layer_related_wait_ms = float(
                    spec.get("per_layer_related_wait_ms", 0.0) or 0.0
                )
                collective_domain = str(spec.get("collective_domain") or "").strip()
                scope_alignment_mode = str(
                    spec.get("scope_alignment_mode") or "wait_inclusive"
                ).strip()
                reason = str(spec.get("reason") or "").strip()

                wait_meta = compute_op_trace_meta(base_op_name, "COMM", trace_context)
                wait_meta.update(
                    {
                        "collective_scope_component": "related_wait",
                        "collective_base_op_name": base_op_name,
                        "collective_domain": collective_domain,
                        "collective_scope_alignment_mode": scope_alignment_mode,
                        "collective_wait_reason": reason,
                    }
                )
                wait_event_name = f"{base_op_name}_wait"

                if should_expand_layers and num_layers > 1:
                    if per_layer_related_wait_ms <= 0.0:
                        continue
                    for layer_idx in range(num_layers):
                        emit(
                            "COMM",
                            wait_event_name,
                            per_layer_related_wait_ms,
                            layer_idx,
                            {"layer_idx": layer_idx, **wait_meta},
                        )
                    continue

                if related_wait_ms <= 0.0:
                    continue
                emit(
                    "COMM",
                    wait_event_name,
                    related_wait_ms,
                    -1,
                    wait_meta,
                )

        # =====================================================================
        # Emit Overhead (CPU) Operations - Before Model Execution
        # These are NOT per-layer, always emit as aggregated
        # =====================================================================
        emit("OVERHEAD", "schedule", execution_time.schedule_time)
        emit("OVERHEAD", "prepare_inputs_e2e", execution_time.prepare_inputs_e2e_time)
        emit(
            "OVERHEAD",
            "pp_receiver_head_runtime",
            execution_time.pp_receiver_head_runtime_time,
        )
        emit(
            "OVERHEAD",
            "pp_prefill_consumer_active_runtime",
            execution_time.pp_prefill_consumer_active_runtime_time,
        )
        emit(
            "COMPUTE",
            "decode_draft_proposer",
            execution_time.decode_draft_proposer_time,
            extra_meta={
                "residual_family": "mtp_draft_proposer",
                "spec_decode_component": "draft_proposer",
            },
        )
        emit(
            "COMPUTE",
            "mtp_terminal_overshoot",
            execution_time.mtp_terminal_overshoot_time,
            extra_meta={
                "residual_family": "mtp_terminal_overshoot_compute",
                "spec_decode_component": "terminal_overshoot",
            },
        )

        # =====================================================================
        # Emit Model Execution Operations (Attention + MLP/MoE)
        # =====================================================================
        moe_tp_enabled = False
        ep_enabled = False
        if execution_time._is_moe:
            cluster_config = self._cluster_configs.get(cluster_type)
            if cluster_config is None:
                raise ValueError(f"Cluster config not found for {cluster_type}")
            moe_tp_enabled = cluster_config.replica_config.moe_tensor_parallel_size > 1
            # COMM_SKIP: EP communication only needed when ep_size > 1 (experts distributed across devices)
            ep_enabled = cluster_config.replica_config.moe_expert_parallel_size > 1

        if should_expand_layers and num_layers > 1:
            # Per-layer expansion mode: emit individual layer traces
            self._emit_per_layer_traces(
                emit,
                execution_time,
                num_layers,
                base_meta,
                moe_tp_enabled,
                ep_enabled,
                cluster_type,
                skip_ffn_attn_norm_residual=skip_ffn_attn_norm_residual,
                skip_add_attn_residual=skip_add_attn_residual,
                use_step3_ep_alltoall=use_step3_ep_alltoall,
                use_ep_alltoall_dispatch_combine=use_ep_alltoall_dispatch_combine,
            )
        else:
            # Aggregated mode (default): emit single spans for all layers
            self._emit_aggregated_traces(
                emit,
                execution_time,
                moe_tp_enabled,
                ep_enabled,
                cluster_type,
                skip_ffn_attn_norm_residual=skip_ffn_attn_norm_residual,
                skip_add_attn_residual=skip_add_attn_residual,
                use_step3_ep_alltoall=use_step3_ep_alltoall,
                use_ep_alltoall_dispatch_combine=use_ep_alltoall_dispatch_combine,
            )

        emit_related_collective_wait_traces()

        # =====================================================================
        # Emit Pipeline Parallel Communication (if applicable)
        # This is NOT per-layer, always emit as aggregated
        # =====================================================================
        emit(
            "COMM",
            "pipeline_parallel_send_recv",
            execution_time.pipeline_parallel_communication_time,
        )
        emit(
            "OVERHEAD",
            "pp_stage_boundary_handoff",
            execution_time.pp_stage_boundary_handoff_time,
        )

        # =====================================================================
        # Emit Overhead (CPU) Operations - After Model Execution
        # These are NOT per-layer, always emit as aggregated
        # =====================================================================
        emit("OVERHEAD", "sampler_e2e", execution_time.sampler_e2e_time)
        emit(
            "OVERHEAD",
            "process_model_outputs",
            execution_time.process_model_outputs_time,
        )
        emit("OVERHEAD", "ray_comm_time", execution_time.ray_comm_time)

    def _should_expand_layers(
        self, cluster_type: ClusterType, request_ids: List[str] = None
    ) -> bool:
        """
        Determine if per-layer expansion should be used for this batch.

        Returns True if:
        1. enable_per_layer_expansion is True
        2. We haven't traced num_requests_to_trace_per_layer requests yet
        3. At least one request in the batch hasn't been traced yet

        Note: The per-layer trace quota is tracked per cluster to avoid
        prefill traces consuming decode-attn/ffn expansion slots.
        """
        if not self._config.enable_per_layer_expansion:
            return False

        if not request_ids:
            return False

        if cluster_type not in self._per_layer_traced_requests_by_cluster:
            self._per_layer_traced_requests_by_cluster[cluster_type] = set()

        traced_requests = self._per_layer_traced_requests_by_cluster[cluster_type]
        max_requests = self._config.num_requests_to_trace_per_layer
        if len(traced_requests) >= max_requests:
            return False

        # Check if any request in this batch is new (not yet traced)
        for req_id in request_ids:
            if req_id not in traced_requests:
                # Mark all requests in this batch as traced
                for rid in request_ids:
                    if len(traced_requests) < max_requests:
                        traced_requests.add(rid)
                return True

        return False

    def _use_ep_alltoall_dispatch_combine(
        self,
        *,
        cluster_type: ClusterType,
        batch_stage: BatchStage,
    ) -> bool:
        if cluster_type == ClusterType.DECODE_FFN:
            return True
        if batch_stage.tokens_are_post_routing:
            return True
        cluster_config = self._cluster_configs.get(cluster_type)
        if cluster_config is None:
            return False
        return int(
            getattr(cluster_config.replica_config, "attn_data_parallel_size", 1)
        ) > 1

    def _emit_aggregated_traces(
        self,
        emit,
        execution_time: ExecutionTime,
        moe_tp_enabled: bool,
        ep_enabled: bool,
        cluster_type: ClusterType,
        skip_ffn_attn_norm_residual: bool = False,
        skip_add_attn_residual: bool = False,
        use_step3_ep_alltoall: bool = False,
        use_ep_alltoall_dispatch_combine: bool = True,
    ) -> None:
        """
        Emit aggregated traces (one span per op type across all layers).

        Communication skip rules:
        - EP communication ops (expert_parallel_alltoall_*) are emitted only when ep_enabled.
        - MoE TP all-reduce is emitted only when moe_tp_enabled.
        """
        is_moe = execution_time._is_moe

        # Attention Block
        emit("COMPUTE", "input_layernorm", execution_time.attn_norm_time)
        emit("COMPUTE", "attn_pre_proj", execution_time.attention_pre_proj_time)
        emit("COMPUTE", "attn_rope", execution_time.attention_rope_execution_time)

        if execution_time.attention_prefill_execution_time > 0:
            emit(
                "COMPUTE",
                "attn_prefill",
                execution_time.attention_prefill_execution_time,
            )
        if execution_time.attention_decode_execution_time > 0:
            emit(
                "COMPUTE", "attn_decode", execution_time.attention_decode_execution_time
            )

        emit(
            "COMPUTE",
            "attn_kv_cache_save",
            execution_time.attention_kv_cache_save_execution_time,
        )
        emit("COMPUTE", "attn_post_proj", execution_time.attention_post_proj_time)
        emit(
            "COMM",
            "attn_tensor_parallel_allreduce",
            execution_time.attention_all_reduce_time,
        )

        # MLP or MoE Block
        if not skip_ffn_attn_norm_residual:
            emit("COMPUTE", "post_attention_layernorm", execution_time.mlp_norm_time)
            if not skip_add_attn_residual:
                emit(
                    "COMPUTE",
                    "add_attn_residual",
                    execution_time.add_attn_residual_time,
                )

        if cluster_type == ClusterType.DECODE_ATTN:
            return

        if is_moe:
            emit(
                "COMM",
                "moe_tensor_parallel_allgather",
                execution_time.moe_tensor_parallel_allgather_time,
            )
            if execution_time.share_expert_time > 0:
                emit(
                    "COMPUTE",
                    "share_expert_up_proj",
                    execution_time.share_expert_up_proj_time,
                )
                emit("COMPUTE", "share_expert_act", execution_time.share_expert_act_time)
                emit(
                    "COMPUTE",
                    "share_expert_down_proj",
                    execution_time.share_expert_down_proj_time,
                )
                emit(
                    "COMM",
                    "share_expert_tensor_parallel_allreduce",
                    execution_time.share_expert_tensor_parallel_allreduce_time,
                )
            emit("COMPUTE", "moe_gating_linear", execution_time.moe_gating_linear_time)
            emit(
                "COMPUTE",
                "moe_gating_routing_topk",
                execution_time.moe_gating_routing_topk_time,
            )
            emit("COMPUTE", "moe_shuffling", execution_time.moe_shuffling_time)
            # COMM_SKIP: EP communication only needed when ep_size > 1 (experts distributed)
            if ep_enabled:
                if use_step3_ep_alltoall:
                    emit(
                        "COMM",
                        "expert_parallel_alltoall",
                        execution_time.expert_parallel_communication_time / 2,
                    )
                elif use_ep_alltoall_dispatch_combine:
                    emit(
                        "COMM",
                        "expert_parallel_alltoall_dispatch",
                        execution_time.expert_parallel_communication_time / 2,
                    )
            emit("COMPUTE", "moe_grouped_gemm", execution_time.moe_grouped_gemm_time)
            if ep_enabled:
                if use_step3_ep_alltoall or use_ep_alltoall_dispatch_combine:
                    emit(
                        "COMM",
                        "expert_parallel_alltoall_combine",
                        execution_time.expert_parallel_communication_time / 2,
                    )
                else:
                    emit(
                        "COMM",
                        "expert_parallel_allreduce",
                        execution_time.expert_parallel_communication_time,
                    )
            if moe_tp_enabled:
                emit(
                    "COMM",
                    "moe_tensor_parallel_allreduce",
                    execution_time.mlp_all_reduce_time,
                )

            dense_trace_up = float(
                getattr(
                    execution_time,
                    "_trace_dense_mlp_layer_up_proj_execution_time",
                    0.0,
                )
            )
            dense_trace_act = float(
                getattr(
                    execution_time,
                    "_trace_dense_mlp_layer_act_execution_time",
                    0.0,
                )
            )
            dense_trace_down = float(
                getattr(
                    execution_time,
                    "_trace_dense_mlp_layer_down_proj_execution_time",
                    0.0,
                )
            )
            if dense_trace_up > 0.0 or dense_trace_act > 0.0 or dense_trace_down > 0.0:
                emit("COMPUTE", "mlp_up_proj", dense_trace_up)
                emit("COMPUTE", "mlp_act", dense_trace_act)
                emit("COMPUTE", "mlp_down_proj", dense_trace_down)
        else:
            emit(
                "COMPUTE",
                "mlp_up_proj",
                execution_time.mlp_layer_up_proj_execution_time,
            )
            emit("COMPUTE", "mlp_act", execution_time.mlp_layer_act_execution_time)
            emit(
                "COMPUTE",
                "mlp_down_proj",
                execution_time.mlp_layer_down_proj_execution_time,
            )
            emit(
                "COMM",
                "mlp_tensor_parallel_allreduce",
                execution_time.mlp_all_reduce_time,
            )

        emit("COMPUTE", "add_ffn_residual", execution_time.add_ffn_residual_time)

    def _emit_per_layer_traces(
        self,
        emit,
        execution_time: ExecutionTime,
        num_layers: int,
        base_meta: dict,
        moe_tp_enabled: bool,
        ep_enabled: bool,
        cluster_type: ClusterType,
        skip_ffn_attn_norm_residual: bool = False,
        skip_add_attn_residual: bool = False,
        use_step3_ep_alltoall: bool = False,
        use_ep_alltoall_dispatch_combine: bool = True,
    ) -> None:
        """
        Emit per-layer traces (individual spans for each layer).

        Divides aggregated times by num_layers to get per-layer times.

        Communication skip rules:
        - EP communication ops (expert_parallel_alltoall_*) are emitted only when ep_enabled.
        - MoE TP all-reduce is emitted only when moe_tp_enabled.
        """
        is_moe = execution_time._is_moe

        # Get per-layer times by dividing aggregated times
        per_layer_attn_norm = execution_time.attn_norm_time / num_layers
        per_layer_attn_pre_proj = execution_time.attention_pre_proj_time / num_layers
        per_layer_attn_rope = execution_time.attention_rope_execution_time / num_layers
        per_layer_attn_prefill = (
            execution_time.attention_prefill_execution_time / num_layers
        )
        per_layer_attn_decode = (
            execution_time.attention_decode_execution_time / num_layers
        )
        per_layer_attn_kv_save = (
            execution_time.attention_kv_cache_save_execution_time / num_layers
        )
        per_layer_attn_post_proj = execution_time.attention_post_proj_time / num_layers
        per_layer_attn_allreduce = execution_time.attention_all_reduce_time / num_layers
        per_layer_mlp_norm = execution_time.mlp_norm_time / num_layers
        per_layer_add_attn_residual = execution_time.add_attn_residual_time / num_layers
        per_layer_add_ffn_residual = execution_time.add_ffn_residual_time / num_layers
        per_layer_share_expert_up = execution_time.share_expert_up_proj_time / num_layers
        per_layer_share_expert_act = execution_time.share_expert_act_time / num_layers
        per_layer_share_expert_down = execution_time.share_expert_down_proj_time / num_layers
        per_layer_moe_tp_allgather = (
            execution_time.moe_tensor_parallel_allgather_time / num_layers
        )
        per_layer_share_expert_allreduce = (
            execution_time.share_expert_tensor_parallel_allreduce_time / num_layers
        )

        if is_moe:
            per_layer_moe_gating_linear = (
                execution_time.moe_gating_linear_time / num_layers
            )
            per_layer_moe_gating_routing_topk = (
                execution_time.moe_gating_routing_topk_time / num_layers
            )
            per_layer_moe_shuffling = execution_time.moe_shuffling_time / num_layers
            per_layer_ep_comm = (
                execution_time.expert_parallel_communication_time / num_layers
            )
            per_layer_moe_gemm = execution_time.moe_grouped_gemm_time / num_layers
            per_layer_moe_tp_allreduce = execution_time.mlp_all_reduce_time / num_layers
        else:
            per_layer_mlp_up = (
                execution_time.mlp_layer_up_proj_execution_time / num_layers
            )
            per_layer_mlp_act = execution_time.mlp_layer_act_execution_time / num_layers
            per_layer_mlp_down = (
                execution_time.mlp_layer_down_proj_execution_time / num_layers
            )
            per_layer_mlp_allreduce = execution_time.mlp_all_reduce_time / num_layers

        # Emit traces for each layer
        for layer_idx in range(num_layers):
            layer_meta = {"layer_idx": layer_idx}

            # Attention Block
            emit(
                "COMPUTE", "input_layernorm", per_layer_attn_norm, layer_idx, layer_meta
            )
            emit(
                "COMPUTE",
                "attn_pre_proj",
                per_layer_attn_pre_proj,
                layer_idx,
                layer_meta,
            )
            emit("COMPUTE", "attn_rope", per_layer_attn_rope, layer_idx, layer_meta)

            if per_layer_attn_prefill > 0:
                emit(
                    "COMPUTE",
                    "attn_prefill",
                    per_layer_attn_prefill,
                    layer_idx,
                    layer_meta,
                )
            if per_layer_attn_decode > 0:
                emit(
                    "COMPUTE",
                    "attn_decode",
                    per_layer_attn_decode,
                    layer_idx,
                    layer_meta,
                )

            emit(
                "COMPUTE",
                "attn_kv_cache_save",
                per_layer_attn_kv_save,
                layer_idx,
                layer_meta,
            )
            emit(
                "COMPUTE",
                "attn_post_proj",
                per_layer_attn_post_proj,
                layer_idx,
                layer_meta,
            )
            emit(
                "COMM",
                "attn_tensor_parallel_allreduce",
                per_layer_attn_allreduce,
                layer_idx,
                layer_meta,
            )

            # MLP or MoE Block
            if not skip_ffn_attn_norm_residual:
                emit(
                    "COMPUTE",
                    "post_attention_layernorm",
                    per_layer_mlp_norm,
                    layer_idx,
                    layer_meta,
                )
                if not skip_add_attn_residual:
                    emit(
                        "COMPUTE",
                        "add_attn_residual",
                        per_layer_add_attn_residual,
                        layer_idx,
                        layer_meta,
                    )

            if cluster_type == ClusterType.DECODE_ATTN:
                continue

            if is_moe:
                emit(
                    "COMM",
                    "moe_tensor_parallel_allgather",
                    per_layer_moe_tp_allgather,
                    layer_idx,
                    layer_meta,
                )
                if execution_time.share_expert_time > 0:
                    emit(
                        "COMPUTE",
                        "share_expert_up_proj",
                        per_layer_share_expert_up,
                        layer_idx,
                        layer_meta,
                    )
                    emit(
                        "COMPUTE",
                        "share_expert_act",
                        per_layer_share_expert_act,
                        layer_idx,
                        layer_meta,
                    )
                    emit(
                        "COMPUTE",
                        "share_expert_down_proj",
                        per_layer_share_expert_down,
                        layer_idx,
                        layer_meta,
                    )
                    emit(
                        "COMM",
                        "share_expert_tensor_parallel_allreduce",
                        per_layer_share_expert_allreduce,
                        layer_idx,
                        layer_meta,
                    )
                emit(
                    "COMPUTE",
                    "moe_gating_linear",
                    per_layer_moe_gating_linear,
                    layer_idx,
                    layer_meta,
                )
                emit(
                    "COMPUTE",
                    "moe_gating_routing_topk",
                    per_layer_moe_gating_routing_topk,
                    layer_idx,
                    layer_meta,
                )
                emit(
                    "COMPUTE",
                    "moe_shuffling",
                    per_layer_moe_shuffling,
                    layer_idx,
                    layer_meta,
                )
                # COMM_SKIP: EP communication only needed when ep_size > 1 (experts distributed)
                if ep_enabled:
                    if use_step3_ep_alltoall:
                        emit(
                            "COMM",
                            "expert_parallel_alltoall",
                            per_layer_ep_comm / 2,
                            layer_idx,
                            layer_meta,
                        )
                    elif use_ep_alltoall_dispatch_combine:
                        emit(
                            "COMM",
                            "expert_parallel_alltoall_dispatch",
                            per_layer_ep_comm / 2,
                            layer_idx,
                            layer_meta,
                        )
                emit(
                    "COMPUTE",
                    "moe_grouped_gemm",
                    per_layer_moe_gemm,
                    layer_idx,
                    layer_meta,
                )
                if ep_enabled:
                    if use_step3_ep_alltoall or use_ep_alltoall_dispatch_combine:
                        emit(
                            "COMM",
                            "expert_parallel_alltoall_combine",
                            per_layer_ep_comm / 2,
                            layer_idx,
                            layer_meta,
                        )
                    else:
                        emit(
                            "COMM",
                            "expert_parallel_allreduce",
                            per_layer_ep_comm,
                            layer_idx,
                            layer_meta,
                        )
                if moe_tp_enabled:
                    emit(
                        "COMM",
                        "moe_tensor_parallel_allreduce",
                        per_layer_moe_tp_allreduce,
                        layer_idx,
                        layer_meta,
                    )

                dense_layer_id = getattr(execution_time, "_trace_dense_layer_id", None)
                dense_trace_up = float(
                    getattr(
                        execution_time,
                        "_trace_dense_mlp_layer_up_proj_execution_time",
                        0.0,
                    )
                )
                dense_trace_act = float(
                    getattr(
                        execution_time,
                        "_trace_dense_mlp_layer_act_execution_time",
                        0.0,
                    )
                )
                dense_trace_down = float(
                    getattr(
                        execution_time,
                        "_trace_dense_mlp_layer_down_proj_execution_time",
                        0.0,
                    )
                )
                if dense_layer_id == layer_idx and (
                    dense_trace_up > 0.0
                    or dense_trace_act > 0.0
                    or dense_trace_down > 0.0
                ):
                    emit("COMPUTE", "mlp_up_proj", dense_trace_up, layer_idx, layer_meta)
                    emit("COMPUTE", "mlp_act", dense_trace_act, layer_idx, layer_meta)
                    emit(
                        "COMPUTE",
                        "mlp_down_proj",
                        dense_trace_down,
                        layer_idx,
                        layer_meta,
                    )
            else:
                emit("COMPUTE", "mlp_up_proj", per_layer_mlp_up, layer_idx, layer_meta)
                emit("COMPUTE", "mlp_act", per_layer_mlp_act, layer_idx, layer_meta)
                emit(
                    "COMPUTE",
                    "mlp_down_proj",
                    per_layer_mlp_down,
                    layer_idx,
                    layer_meta,
                )
                emit(
                    "COMM",
                    "mlp_tensor_parallel_allreduce",
                    per_layer_mlp_allreduce,
                    layer_idx,
                    layer_meta,
                )

            emit(
                "COMPUTE",
                "add_ffn_residual",
                per_layer_add_ffn_residual,
                layer_idx,
                layer_meta,
            )

    def _should_store_memory_time_series(self) -> bool:
        if not self._config.enable_memory_time_series:
            return False
        if str(self._simulation_config.log_level).lower() != "debug":
            raise ValueError(
                "metrics_config.enable_memory_time_series requires log_level='debug'"
            )
        return True

    @staticmethod
    def _get_parallel_lane_count(cluster_type: ClusterType, replica_config) -> int:
        """Return the lane count used for utilization metrics indexing.

        Frontier models DECODE_FFN with EP lanes (ep_id reused as dp_id in schedulers),
        so utilization tensors must be sized by moe_expert_parallel_size there.
        Other clusters keep data_parallel_size semantics.
        """
        if cluster_type == ClusterType.DECODE_FFN:
            lane_count = int(replica_config.moe_expert_parallel_size)
        else:
            lane_count = int(replica_config.data_parallel_size)

        if lane_count <= 0:
            raise ValueError(
                f"Invalid lane count for {cluster_type.name}: {lane_count}"
            )

        return lane_count

    def _init_per_cluster_metrics(self, cluster_type: ClusterType, cluster_config):
        # Batch metrics
        self._batch_metrics_count_distribution[cluster_type] = {
            metric_name: CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in BatchMetricsCountDistribution
        }
        self._batch_metrics_count_distribution_per_batch[cluster_type] = {
            metric_name: DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in BatchMetricsCountDistribution
        }
        self._batch_metrics_time_distribution[cluster_type] = {
            metric_name: CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in BatchMetricsTimeDistribution
        }
        self._batch_metrics_time_distribution_per_batch[cluster_type] = {
            metric_name: DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in BatchMetricsTimeDistribution
        }

        # Operation metrics
        self._operation_metrics[cluster_type] = {
            metric_name: CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in OperationMetrics
        }
        self._operation_metrics_per_batch[cluster_type] = {
            metric_name: DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in OperationMetrics
        }
        self._cpu_operation_metrics[cluster_type] = {
            metric_name: CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in CpuOperationMetrics
        }
        self._cpu_operation_metrics_per_batch[cluster_type] = {
            metric_name: DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            for metric_name in CpuOperationMetrics
        }

        # Utilization metrics - now support dp_id
        num_replicas = cluster_config.num_replicas
        num_pipeline_stages = cluster_config.replica_config.num_pipeline_stages
        dp_size = self._get_parallel_lane_count(cluster_type, cluster_config.replica_config)

        self._replica_memory_usage[cluster_type] = []
        self._replica_busy_time[cluster_type] = []
        self._replica_mfu[cluster_type] = []
        self._mfu_calculator[cluster_type] = MFUCalculator(
            cluster_config.replica_config, cluster_type
        )

        for replica_idx in range(num_replicas):
            # Create dp_size slots for each replica
            self._replica_memory_usage[cluster_type].append([])
            self._replica_busy_time[cluster_type].append([])
            self._replica_mfu[cluster_type].append([])

            for dp_idx in range(dp_size):
                self._replica_memory_usage[cluster_type][replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        MEMORY_USAGE_STR,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                        store_data_series=self._should_store_memory_time_series(),
                    )
                )
                self._replica_memory_usage[cluster_type][replica_idx][dp_idx].put(0, 0)

                self._replica_busy_time[cluster_type][replica_idx].append([])
                self._replica_mfu[cluster_type][replica_idx].append([])

                for stage_idx in range(num_pipeline_stages):
                    self._replica_busy_time[cluster_type][replica_idx][dp_idx].append(
                        SeriesAverageMeter(
                            TIME_STR,
                            BUSY_TIME_PERCENT,
                            save_table_to_wandb=self._config.save_table_to_wandb,
                        )
                    )
                    self._replica_busy_time[cluster_type][replica_idx][dp_idx][
                        stage_idx
                    ].put(0, 0)

                    self._replica_mfu[cluster_type][replica_idx][dp_idx].append(
                        SeriesAverageMeter(
                            TIME_STR,
                            UTILIZATION_STR,
                            save_table_to_wandb=self._config.save_table_to_wandb,
                        )
                    )
                    self._replica_mfu[cluster_type][replica_idx][dp_idx][stage_idx].put(
                        0, 0
                    )

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._config.wandb_project
            or not self._config.wandb_group
        ):
            return

        global wandb
        wandb = require_wandb()
        wandb.init(
            project=self._config.wandb_project,
            group=self._config.wandb_group,
            name=self._config.wandb_run_name,
            config=self._simulation_config.to_dict(),
        )

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
        collapse_duplicates: bool = False,
    ):
        from frontier.logger import init_logger

        logger = init_logger(__name__)

        logger.info(f"Saving {file_name}.csv: {len(dataseries_list)} dataseries")
        os.makedirs(base_path, exist_ok=True)

        # Convert dataseries to DataFrames (detailed logging at DEBUG level)
        dfs = []
        empty_count = 0
        for i, dataseries in enumerate(dataseries_list):
            logger.debug(
                f"Converting dataseries {i + 1}/{len(dataseries_list)}: {dataseries._metric_name}"
            )
            df = dataseries._to_df()
            logger.debug(f"  DataFrame shape: {df.shape}")
            if collapse_duplicates and not df.empty and key_to_join in df.columns:
                value_cols = [col for col in df.columns if col != key_to_join]
                if value_cols:
                    df = (
                        df.groupby(key_to_join, as_index=False)[value_cols]
                        .sum()
                    )
            # Skip empty DataFrames to avoid merge issues
            if not df.empty:
                dfs.append(df)
            else:
                empty_count += 1
                logger.debug(
                    f"  Skipping empty DataFrame for {dataseries._metric_name}"
                )

        if empty_count > 0:
            logger.info(f"Skipped {empty_count} empty DataFrames")

        # Merge DataFrames
        if not dfs:
            logger.warning(
                "No non-empty DataFrames to merge! Creating empty DataFrame."
            )
            merged_df = pd.DataFrame()
        elif len(dfs) == 1:
            logger.info("Only one DataFrame, no merge needed")
            merged_df = dfs[0]
        else:
            logger.info(f"Starting merge of {len(dfs)} DataFrames...")
            # Perform merge iteratively with progress logging
            merged_df = dfs[0]
            logger.info(
                f"Initial DataFrame: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns"
            )

            for i, df in enumerate(dfs[1:], start=2):
                logger.info(f"Merging DataFrame {i}/{len(dfs)} (shape: {df.shape})...")
                try:
                    before_rows = merged_df.shape[0]
                    merged_df = pd.merge(merged_df, df, on=[key_to_join], how="outer")
                    after_rows = merged_df.shape[0]
                    logger.info(
                        f"  Merge {i}/{len(dfs)} complete: {before_rows} → {after_rows} rows, {merged_df.shape[1]} columns"
                    )
                except Exception as e:
                    logger.error(
                        f"Error merging DataFrame {i}/{len(dfs)}: {e}", exc_info=True
                    )
                    raise

            logger.info(f"All merges complete!")

        # Log summary statistics
        if not merged_df.empty:
            num_rows = merged_df.shape[0]
            num_cols = merged_df.shape[1]
            logger.info(f"Computing unique {key_to_join} count...")
            unique_keys = (
                merged_df[key_to_join].nunique()
                if key_to_join in merged_df.columns
                else 0
            )
            logger.info(
                f"Final merged DataFrame: {num_rows} rows × {num_cols} columns (unique {key_to_join}: {unique_keys})"
            )
        else:
            logger.info("Final merged DataFrame: empty")

        # Normalize exported request-level time metrics to milliseconds.
        # Internal simulation and scheduler timelines remain in seconds.
        if file_name == "request_metrics" and not merged_df.empty:
            for metric_name in RequestMetricsTimeDistributions:
                column_name = metric_name.value
                if column_name not in merged_df.columns:
                    continue
                merged_df[column_name] = (
                    pd.to_numeric(merged_df[column_name], errors="coerce") * 1000.0
                )

        # Write to CSV
        csv_path = f"{base_path}/{file_name}.csv"
        merged_df.to_csv(csv_path, index=False)
        logger.info(f"Saved to {csv_path}")

        if wandb is not None and wandb.run and self._config.save_table_to_wandb:
            logger.info("Logging to wandb...")
            wand_table = wandb.Table(dataframe=merged_df)
            wandb.log({f"{file_name}_table": wand_table}, step=0)
            logger.info("Wandb logging complete")

    def _store_bar_plot(
        self,
        base_path: str,
        plot_name: str,
        x_label: str,
        y_label: str,
        data: Dict[str, float],
    ):
        if not data:
            return
        if wandb is not None and wandb.run:
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )
        if self._config.store_plots:
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                labels={"x": x_label, "y": y_label},
            )
            try:
                fig.write_image(f"{base_path}/{plot_name}.png")
            except Exception as exc:
                logger.warning(
                    "Skipping bar plot image export because image rendering is unavailable. "
                    "plot_name=%s base_path=%s error=%s",
                    plot_name,
                    base_path,
                    exc,
                )

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:
            return

        for cluster_type in self._cluster_configs.keys():
            cluster_plot_path = f"{base_plot_path}/{cluster_type.name.lower()}"
            os.makedirs(cluster_plot_path, exist_ok=True)

            # Plot operation metrics (only if store_plots=True)
            if self._config.store_plots:
                total_operation_runtimes: Dict[str, float] = {}
                total_operation_runtimes["model_execution_e2e"] = 0

                for dataseries in self._operation_metrics[cluster_type].values():
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_execution_time"
                    dataseries.plot_cdf(
                        cluster_plot_path, plot_name_with_cluster, TIME_STR_MS
                    )
                    total_operation_runtimes[dataseries._metric_name] = dataseries.sum
                    total_operation_runtimes["model_execution_e2e"] += dataseries.sum

                for dataseries in self._cpu_operation_metrics[cluster_type].values():
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_execution_time"
                    dataseries.plot_cdf(
                        cluster_plot_path, plot_name_with_cluster, TIME_STR_MS
                    )
                    total_operation_runtimes[dataseries._metric_name] = dataseries.sum

                self._store_bar_plot(
                    cluster_plot_path,
                    f"{cluster_type.name.lower()}_total_operation_runtimes",
                    OPERATION_STR,
                    TIME_STR_MS,
                    total_operation_runtimes,
                )

            if not self._config.keep_individual_batch_metrics:
                continue

            # Plot per-batch operation metrics (only if store_plots=True)
            if self._config.store_plots:
                for dataseries in self._operation_metrics_per_batch[
                    cluster_type
                ].values():
                    dataseries.consolidate()
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_per_batch"
                    dataseries.plot_step(
                        cluster_plot_path,
                        plot_name_with_cluster,
                        y_axis_label=TIME_STR_MS,
                        y_cumsum=False,
                    )

                for dataseries in self._cpu_operation_metrics_per_batch[
                    cluster_type
                ].values():
                    dataseries.consolidate()
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_per_batch"
                    dataseries.plot_step(
                        cluster_plot_path,
                        plot_name_with_cluster,
                        y_axis_label=TIME_STR_MS,
                        y_cumsum=False,
                    )

            # Save to CSV (always execute if keep_individual_batch_metrics=True)
            operations_dataseries_list = list(
                self._operation_metrics_per_batch[cluster_type].values()
            )
            self._save_as_csv(
                dataseries_list=operations_dataseries_list,
                key_to_join=BATCH_ID_STR,
                base_path=self._config.output_dir,
                file_name=f"{cluster_type.name.lower()}_operation_metrics",
                collapse_duplicates=True,
            )

            cpu_operations_dataseries_list = list(
                self._cpu_operation_metrics_per_batch[cluster_type].values()
            )
            self._save_as_csv(
                dataseries_list=cpu_operations_dataseries_list,
                key_to_join=BATCH_ID_STR,
                base_path=self._config.output_dir,
                file_name=f"{cluster_type.name.lower()}_cpu_operation_metrics",
                collapse_duplicates=True,
            )

    def _store_request_metrics(self, base_plot_path: str):
        from frontier.logger import init_logger

        logger = init_logger(__name__)

        if not self._config.store_request_metrics:
            logger.debug("store_request_metrics=False, skipping")
            return

        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())

        if not all_request_metrics:
            logger.warning("No request metrics to save!")
            return

        # Save to CSV (detailed logging in _save_as_csv)
        # CSV generation is always executed when store_request_metrics=True
        self._save_as_csv(
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        # Plot histograms and CDFs (only if store_plots=True)
        if self._config.store_plots:
            # Plot histograms (detailed logging at DEBUG level)
            num_histograms = len(self._request_metrics_histogram)
            logger.info(f"Plotting {num_histograms} histograms...")
            for i, dataseries in enumerate(self._request_metrics_histogram.values()):
                logger.debug(
                    f"Plotting histogram {i + 1}/{num_histograms}: {dataseries._y_name}"
                )
                dataseries.plot_histogram(base_plot_path, dataseries._y_name)

            # Plot CDFs (detailed logging at DEBUG level)
            num_cdfs = len(self._request_metrics_time_distributions)
            logger.info(f"Plotting {num_cdfs} CDFs...")
            for i, dataseries in enumerate(
                self._request_metrics_time_distributions.values()
            ):
                logger.debug(f"Plotting CDF {i + 1}/{num_cdfs}: {dataseries._y_name}")
                dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)
            logger.info("Plotting completed")
        else:
            logger.info("Skipping plots (store_plots=False)")

        logger.info("Request metrics storage completed")

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        for cluster_type in self._cluster_configs.keys():
            cluster_plot_path = f"{base_plot_path}/{cluster_type.name.lower()}"
            os.makedirs(cluster_plot_path, exist_ok=True)

            # Plot CDFs (only if store_plots=True)
            if self._config.store_plots:
                for dataseries in self._batch_metrics_time_distribution[
                    cluster_type
                ].values():
                    y_axis_label = (
                        TIME_STR_MS
                        if "model_execution" in dataseries._metric_name
                        else TIME_STR
                    )
                    plot_name_with_cluster = (
                        f"{cluster_type.name.lower()}_{dataseries._metric_name}"
                    )
                    dataseries.plot_cdf(
                        cluster_plot_path, plot_name_with_cluster, y_axis_label
                    )

                for dataseries in self._batch_metrics_count_distribution[
                    cluster_type
                ].values():
                    plot_name_with_cluster = (
                        f"{cluster_type.name.lower()}_{dataseries._metric_name}"
                    )
                    dataseries.plot_cdf(
                        cluster_plot_path, plot_name_with_cluster, COUNT_STR
                    )

            # Save individual batch metrics (if enabled)
            if not self._config.keep_individual_batch_metrics:
                continue

            # Plot per-batch metrics (only if store_plots=True)
            if self._config.store_plots:
                for dataseries in self._batch_metrics_time_distribution_per_batch[
                    cluster_type
                ].values():
                    y_axis_label = (
                        TIME_STR_MS
                        if "model_execution" in dataseries._metric_name
                        else TIME_STR
                    )
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_per_batch"
                    dataseries.plot_step(
                        cluster_plot_path,
                        plot_name_with_cluster,
                        y_axis_label=y_axis_label,
                        y_cumsum=False,
                    )

                for dataseries in self._batch_metrics_count_distribution_per_batch[
                    cluster_type
                ].values():
                    plot_name_with_cluster = f"{cluster_type.name.lower()}_{dataseries._metric_name}_per_batch"
                    dataseries.plot_step(
                        cluster_plot_path,
                        plot_name_with_cluster,
                        y_axis_label=COUNT_STR,
                        y_cumsum=False,
                    )

            # Save to CSV (always execute if keep_individual_batch_metrics=True)
            all_batch_metrics = list(
                self._batch_metrics_count_distribution_per_batch[cluster_type].values()
            ) + list(
                self._batch_metrics_time_distribution_per_batch[cluster_type].values()
            )
            self._save_as_csv(
                dataseries_list=all_batch_metrics,
                key_to_join=BATCH_ID_STR,
                base_path=self._config.output_dir,
                file_name=f"{cluster_type.name.lower()}_batch_metrics",
                collapse_duplicates=True,
            )

    def _store_completion_metrics(self, base_plot_path: str):
        # Plot completion metrics (only if store_plots=True)
        if self._config.store_plots:
            if self._config.store_request_metrics:
                for dataseries in self._request_completion_metrics_time_series.values():
                    dataseries.plot_step(
                        base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                    )

            if self._config.store_token_completion_metrics:
                for dataseries in self._token_metrics_time_distribution.values():
                    dataseries.plot_cdf(
                        base_plot_path, dataseries._metric_name, TIME_STR
                    )

                for dataseries in self._token_completion_metrics_time_series.values():
                    dataseries.plot_step(
                        base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                    )

    def _store_utilization_metrics(self, base_plot_path: str):
        if not self._config.store_utilization_metrics:
            return

        # Plot utilization metrics (only if store_plots=True)
        if self._config.store_plots:
            for cluster_type, cluster_config in self._cluster_configs.items():
                cluster_plot_path = f"{base_plot_path}/{cluster_type.name.lower()}"
                os.makedirs(cluster_plot_path, exist_ok=True)
                num_replicas = cluster_config.num_replicas
                num_pipeline_stages = cluster_config.replica_config.num_pipeline_stages
                dp_size = self._get_parallel_lane_count(cluster_type, cluster_config.replica_config)

                for replica_idx in range(num_replicas):
                    for dp_idx in range(dp_size):
                        self._replica_memory_usage[cluster_type][replica_idx][
                            dp_idx
                        ].print_stats(
                            f"replica_{replica_idx + 1}_dp_{dp_idx}_memory_usage",
                            cluster_plot_path,
                        )
                        for stage_idx in range(num_pipeline_stages):
                            self._replica_busy_time[cluster_type][replica_idx][dp_idx][
                                stage_idx
                            ].print_stats(
                                f"replica_{replica_idx + 1}_dp_{dp_idx}_stage_{stage_idx + 1}_busy_time_percent",
                                cluster_plot_path,
                            )
                            self._replica_mfu[cluster_type][replica_idx][dp_idx][
                                stage_idx
                            ].print_stats(
                                f"replica_{replica_idx + 1}_dp_{dp_idx}_stage_{stage_idx + 1}_mfu",
                                cluster_plot_path,
                            )

    # ----- Completion tracking API -----
    def register_total_requests(self, n: int) -> None:
        """Register total number of requests generated for this simulation."""
        self._total_requests = int(n)

    def all_requests_completed(self) -> bool:
        """Return True if all requests have completed (and at least one was registered)."""
        return (self._total_requests > 0) and (
            self._completed_requests >= self._total_requests
        )

    def get_total_requests(self) -> int:
        return getattr(self, "_total_requests", 0)

    def get_completed_requests(self) -> int:
        return getattr(self, "_completed_requests", 0)

    def _write_op_precision_metadata(self) -> None:
        from frontier.logger import init_logger

        logger = init_logger(__name__)
        quant_manager = get_quantization_manager()
        metadata = quant_manager.get_operation_precision_metadata()
        if not metadata:
            logger.info("No op precision metadata to write")
            return
        output_path = os.path.join(self._config.output_dir, "op_precision_metadata.csv")
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        logger.info("Wrote op precision metadata to %s", output_path)

    @if_write_metrics
    def plot(self) -> None:
        from frontier.logger import init_logger

        logger = init_logger(__name__)

        logger.info(f"Writing metrics to {self._config.output_dir}")
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)
        self._store_completion_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics(dir_plot_path)
        self._write_system_metrics()
        self._write_op_precision_metadata()
        self._write_frontier_stage_batch_ledger()

        # Flush and close TraceStore if active
        if self.trace_store:
            self.trace_store.close()

        logger.info("Metrics output completed")

    def _write_system_metrics(self) -> None:
        """
        Write system-level aggregate metrics to system_metrics.json.

        This file contains aggregate statistics computed from per-request metrics,
        providing a high-level summary of simulation performance.
        """
        from frontier.logger import init_logger
        import numpy as np

        logger = init_logger(__name__)

        logger.info("Computing system-level aggregate metrics...")

        system_metrics: Dict[str, Any] = {}

        # 1. Simulation metadata
        system_metrics["simulation_metadata"] = {
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "system_architecture": self._simulation_config.sys_arch,
        }
        server_count_metadata = (
            self._simulation_config.cluster_config.get_server_count_metadata(
                self._simulation_config.sys_arch
            )
        )
        system_metrics["simulation_metadata"].update(server_count_metadata)
        quant_manager = get_quantization_manager()
        quant_config_path = quant_manager.get_config_path() or "none"
        system_metrics["quantization_config"] = {
            "config_path": quant_config_path,
            "default_precision": quant_manager.get_default_precision().name,
            "custom_operation_count": quant_manager.get_custom_operation_count(),
            "config_hash": quant_manager.get_config_hash(),
        }

        # 2. TTFT statistics (Time To First Token)
        ttft_data = self._request_metrics_time_distributions.get(
            RequestMetricsTimeDistributions.TTFT
        )
        if ttft_data and len(ttft_data) > 0:
            ttft_values = [y * 1000.0 for _, y in ttft_data._data_series]
            system_metrics["ttft_statistics"] = self._compute_percentile_stats(
                ttft_values, "ms"
            )
        else:
            system_metrics["ttft_statistics"] = {"error": "No TTFT data available"}

        # 3. TPOT statistics (Time Per Output Token)
        tpot_data = self._request_metrics_time_distributions.get(
            RequestMetricsTimeDistributions.TPOT
        )
        if tpot_data and len(tpot_data) > 0:
            tpot_values = [y * 1000.0 for _, y in tpot_data._data_series]
            system_metrics["tpot_statistics"] = self._compute_percentile_stats(
                tpot_values, "ms"
            )
            # Add note about requests with single decode token
            system_metrics["tpot_statistics"]["note"] = (
                f"TPOT is only computed for requests with num_decode_tokens > 1. "
                f"Computed from {len(tpot_values)} out of {self._completed_requests} requests."
            )
        else:
            system_metrics["tpot_statistics"] = {
                "error": "No TPOT data available (all requests may have num_decode_tokens=1)"
            }

        # 4. Request E2E time statistics
        e2e_data = self._request_metrics_time_distributions.get(
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        )
        if e2e_data and len(e2e_data) > 0:
            e2e_values = [y * 1000.0 for _, y in e2e_data._data_series]
            system_metrics["request_e2e_time_statistics"] = (
                self._compute_percentile_stats(e2e_values, "ms")
            )
        else:
            system_metrics["request_e2e_time_statistics"] = {
                "error": "No E2E time data available"
            }

        # 5. Throughput metrics
        if self._completed_requests > 0:
            # Get first request arrival and last request completion times
            arrival_data = self._request_completion_metrics_time_series.get(
                RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
            )
            completion_data = self._request_completion_metrics_time_series.get(
                RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
            )

            if (
                arrival_data
                and completion_data
                and len(arrival_data) > 0
                and len(completion_data) > 0
            ):
                first_arrival = min(x for x, _ in arrival_data._data_series)
                last_completion = max(x for x, _ in completion_data._data_series)
                # NOTE: Simulator time is in SECONDS, convert to milliseconds for reporting
                total_duration_s = last_completion - first_arrival
                total_duration_ms = total_duration_s * 1000.0  # Convert to milliseconds

                if total_duration_ms > 0:
                    # Calculate total tokens processed
                    total_tokens = 0
                    total_decode_tokens = 0
                    decode_tokens_data = self._request_metrics_histogram.get(
                        RequestMetricsHistogram.REQUEST_DECODE_TOKENS
                    )
                    num_tokens_data = self._request_metrics_histogram.get(
                        RequestMetricsHistogram.REQUEST_NUM_TOKENS
                    )

                    if decode_tokens_data and len(decode_tokens_data) > 0:
                        total_decode_tokens = sum(
                            y for _, y in decode_tokens_data._data_series
                        )
                    if num_tokens_data and len(num_tokens_data) > 0:
                        total_tokens = sum(y for _, y in num_tokens_data._data_series)

                    system_metrics["throughput_metrics"] = {
                        "total_duration_ms": total_duration_ms,
                        "total_duration_seconds": total_duration_s,
                        "requests_per_second": self._completed_requests
                        / total_duration_s,
                        "total_tokens_processed": int(total_tokens),
                        "total_decode_tokens_generated": int(total_decode_tokens),
                        "tokens_per_second": total_tokens / total_duration_s
                        if total_tokens > 0
                        else 0,
                        "decode_tokens_per_second": total_decode_tokens
                        / total_duration_s
                        if total_decode_tokens > 0
                        else 0,
                    }
                else:
                    system_metrics["throughput_metrics"] = {
                        "error": "Total duration is zero"
                    }
            else:
                system_metrics["throughput_metrics"] = {
                    "error": "Missing arrival or completion data"
                }
        else:
            system_metrics["throughput_metrics"] = {"error": "No completed requests"}

        # 6. Speculative decoding aggregate statistics (Phase 1)
        spec_iter_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_SPEC_TOTAL_ITERATIONS
        )
        spec_accepted_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_SPEC_ACCEPTED_DRAFTS
        )
        spec_rejected_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_SPEC_REJECTED_DRAFTS
        )
        spec_committed_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_SPEC_COMMITTED_TOKENS
        )
        if (
            spec_iter_data
            and spec_accepted_data
            and spec_rejected_data
            and spec_committed_data
            and len(spec_iter_data) > 0
        ):
            total_iterations = float(sum(y for _, y in spec_iter_data._data_series))
            total_accepted = float(sum(y for _, y in spec_accepted_data._data_series))
            total_rejected = float(sum(y for _, y in spec_rejected_data._data_series))
            total_committed = float(sum(y for _, y in spec_committed_data._data_series))
            total_drafts = total_accepted + total_rejected
            acceptance_ratio = total_accepted / total_drafts if total_drafts > 0 else 0.0
            committed_per_iteration = (
                total_committed / total_iterations if total_iterations > 0 else 0.0
            )
            system_metrics["spec_decode_statistics"] = {
                "total_iterations": int(total_iterations),
                "total_accepted_drafts": int(total_accepted),
                "total_rejected_drafts": int(total_rejected),
                "total_committed_tokens": int(total_committed),
                "acceptance_ratio": float(acceptance_ratio),
                "avg_committed_tokens_per_iteration": float(committed_per_iteration),
            }
        else:
            system_metrics["spec_decode_statistics"] = {
                "total_iterations": 0,
                "total_accepted_drafts": 0,
                "total_rejected_drafts": 0,
                "total_committed_tokens": 0,
                "acceptance_ratio": 0.0,
                "avg_committed_tokens_per_iteration": 0.0,
            }

        prefix_cache_statistics = self._compute_prefix_cache_statistics()
        if prefix_cache_statistics is not None:
            system_metrics["prefix_cache_statistics"] = prefix_cache_statistics

        # 7. KV Cache Transfer statistics (for disaggregated mode)
        if self._kv_cache_transfer_metrics["transfer_count"] > 0:
            transfer_times = self._kv_cache_transfer_metrics["transfer_times"]
            if len(transfer_times) > 0:
                transfer_time_values = [y for _, y in transfer_times._data_series]
                system_metrics["kv_cache_transfer_statistics"] = {
                    "total_transfers": self._kv_cache_transfer_metrics[
                        "transfer_count"
                    ],
                    "total_data_transferred_bytes": self._kv_cache_transfer_metrics[
                        "total_data_transferred"
                    ],
                    "total_transfer_time_ms": self._kv_cache_transfer_metrics[
                        "total_transfer_time"
                    ],
                    "transfer_time_stats": self._compute_percentile_stats(
                        transfer_time_values, "ms"
                    ),
                }
                system_metrics["kv_cache_transfer_total_bytes"] = (
                    self._kv_cache_transfer_metrics["total_data_transferred"]
                )

        # 8. M2N Transfer statistics (for PD+AF disaggregated mode)
        if hasattr(self, "_m2n_transfer_metrics") and self._m2n_transfer_metrics[
            "transfer_count"
        ] > 0:
            transfer_times = self._m2n_transfer_metrics["transfer_times"]
            if len(transfer_times) > 0:
                transfer_time_values = [y for _, y in transfer_times._data_series]
                system_metrics["m2n_transfer_statistics"] = {
                    "total_transfers": self._m2n_transfer_metrics["transfer_count"],
                    "total_data_transferred_bytes": self._m2n_transfer_metrics[
                        "total_data_transferred"
                    ],
                    "total_transfer_time_ms": self._m2n_transfer_metrics[
                        "total_transfer_time"
                    ],
                    "attn_to_ffn_transfers": self._m2n_transfer_metrics[
                        "attn_to_ffn_transfers"
                    ],
                    "ffn_to_attn_transfers": self._m2n_transfer_metrics[
                        "ffn_to_attn_transfers"
                    ],
                    "transfer_time_stats": self._compute_percentile_stats(
                        transfer_time_values, "ms"
                    ),
                }
                system_metrics["m2n_transfer_total_bytes"] = (
                    self._m2n_transfer_metrics["total_data_transferred"]
                )

        # 9. Per-cluster utilization summary
        cluster_utilization = {}
        for cluster_type in self._cluster_configs.keys():
            if cluster_type in self._replica_memory_usage:
                memory_usage_data = self._replica_memory_usage[cluster_type]
                if memory_usage_data:
                    # Compute average memory usage across all replicas and DPs
                    all_memory_values = []
                    for replica_data in memory_usage_data:
                        for dp_data in replica_data:
                            if (
                                hasattr(dp_data, "_data_series")
                                and dp_data._data_series is not None
                                and len(dp_data._data_series) > 0
                            ):
                                all_memory_values.extend(
                                    [y for _, y in dp_data._data_series]
                                )

                    if all_memory_values:
                        cluster_utilization[cluster_type.name] = {
                            "avg_memory_usage_percent": float(
                                np.mean(all_memory_values)
                            ),
                            "max_memory_usage_percent": float(
                                np.max(all_memory_values)
                            ),
                        }

        if cluster_utilization:
            system_metrics["cluster_utilization"] = cluster_utilization

        # 10. Preemption statistics (cluster-level aggregate metrics)
        if self._preemption_statistics["total_preemption_events"] > 0:
            preemption_stats = {
                "total_preemption_events": self._preemption_statistics[
                    "total_preemption_events"
                ],
                "total_preempted_requests": len(
                    self._preemption_statistics["preempted_request_ids"]
                ),
                "preemption_rate": len(
                    self._preemption_statistics["preempted_request_ids"]
                )
                / self._total_requests
                if self._total_requests > 0
                else 0.0,
                "by_cluster_type": {},
            }

            # Compute per-cluster statistics
            for cluster_type, cluster_data in self._preemption_statistics[
                "by_cluster_type"
            ].items():
                if cluster_data["preemption_events"] > 0:
                    cluster_stats = {
                        "total_preemption_events": cluster_data["preemption_events"],
                        "preempted_request_count": len(
                            cluster_data["preempted_request_ids"]
                        ),
                        "preemption_rate": len(cluster_data["preempted_request_ids"])
                        / self._total_requests
                        if self._total_requests > 0
                        else 0.0,
                    }

                    # Add tokens-at-preemption statistics for DECODE clusters
                    if (
                        cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]
                        and cluster_data["tokens_at_preemption"]
                    ):
                        tokens_list = cluster_data["tokens_at_preemption"]
                        cluster_stats["tokens_at_preemption"] = {
                            "mean": float(np.mean(tokens_list)),
                            "max": int(np.max(tokens_list)),
                            "min": int(np.min(tokens_list)),
                        }

                    preemption_stats["by_cluster_type"][cluster_type.name] = (
                        cluster_stats
                    )

            system_metrics["preemption_statistics"] = preemption_stats
        else:
            # No preemptions occurred
            system_metrics["preemption_statistics"] = {
                "total_preemption_events": 0,
                "total_preempted_requests": 0,
                "preemption_rate": 0.0,
                "by_cluster_type": {},
            }

        # 11. System architecture-specific configuration and model weight memory information
        system_architecture_info = {
            "system_architecture": self._simulation_config.sys_arch,
            "cluster_configurations": {},
        }

        model_weight_memory = {}
        for cluster_type, cluster_config in self._cluster_configs.items():
            try:
                # Import ParamCounter for model weight calculations
                from frontier.utils.param_counter import ParamCounter

                # Create ParamCounter for this cluster configuration
                param_counter = ParamCounter(
                    replica_config=cluster_config.replica_config,
                    cluster_type=cluster_type,
                )

                # Calculate model parameters and memory
                num_params_per_device = param_counter.get_num_parameters_per_device()
                num_attention_params_per_layer = (
                    param_counter.get_num_attention_params_per_layer()
                )
                num_mlp_params_per_layer = param_counter.get_num_mlp_params_per_layer()
                num_mlp_params_per_device = (
                    param_counter.get_num_mlp_parameters_per_device()
                )

                # Calculate memory in bytes (FP16: 2 bytes per parameter)
                model_weight_memory_bytes = 2 * num_params_per_device
                model_weight_memory_gb = model_weight_memory_bytes / (1024**3)

                # Calculate memory breakdown
                attention_memory_bytes = (
                    2
                    * num_attention_params_per_layer
                    * (
                        cluster_config.replica_config.model_config.num_layers
                        // cluster_config.replica_config.num_pipeline_stages
                    )
                )
                mlp_memory_bytes = 2 * num_mlp_params_per_device

                # Calculate memory utilization percentage
                total_device_memory_gb = (
                    cluster_config.replica_config.device_config.total_memory_gb
                )
                memory_utilization_percent = (
                    (model_weight_memory_gb / total_device_memory_gb) * 100
                    if total_device_memory_gb > 0
                    else 0
                )

                # Extract parallel parameters based on cluster type
                replica_config = cluster_config.replica_config
                parallel_params = {
                    "PP": replica_config.num_pipeline_stages,
                }

                # Add cluster-specific parallel parameters
                if cluster_type == ClusterType.DECODE_FFN:
                    # DECODE_FFN cluster only has MoE parameters
                    parallel_params.update(
                        {
                            "MOE_TP": replica_config.moe_tensor_parallel_size,
                            "MOE_EP": replica_config.moe_expert_parallel_size,
                        }
                    )
                elif cluster_type == ClusterType.DECODE_ATTN:
                    # DECODE_ATTN cluster only has Attention parameters
                    parallel_params.update(
                        {
                            "ATTN_TP": replica_config.attn_tensor_parallel_size,
                        }
                    )
                else:
                    # MONOLITHIC, PREFILL, DECODE clusters have both Attention and MoE parameters
                    parallel_params.update(
                        {
                            "ATTN_TP": replica_config.attn_tensor_parallel_size,
                            "MOE_TP": replica_config.moe_tensor_parallel_size,
                            "MOE_EP": replica_config.moe_expert_parallel_size,
                        }
                    )

                cluster_memory_info = {
                    "total_parameters": int(num_params_per_device),
                    "total_memory_bytes": int(model_weight_memory_bytes),
                    "total_memory_gb": float(model_weight_memory_gb),
                    "memory_utilization_percent": float(memory_utilization_percent),
                    "breakdown": {
                        "attention_parameters": int(
                            num_attention_params_per_layer
                            * (
                                cluster_config.replica_config.model_config.num_layers
                                // cluster_config.replica_config.num_pipeline_stages
                            )
                        ),
                        "attention_memory_bytes": int(attention_memory_bytes),
                        "attention_memory_gb": float(
                            attention_memory_bytes / (1024**3)
                        ),
                        "ffn_parameters": int(num_mlp_params_per_device),
                        "ffn_memory_bytes": int(mlp_memory_bytes),
                        "ffn_memory_gb": float(mlp_memory_bytes / (1024**3)),
                    },
                    "device_info": {
                        "device_type": cluster_config.replica_config.device_config.get_type().name,
                        "total_device_memory_gb": float(total_device_memory_gb),
                        "num_replicas": cluster_config.num_replicas,
                        "expert_parallel_size": getattr(
                            cluster_config.replica_config, "moe_expert_parallel_size", 1
                        ),
                        "moe_tensor_parallel_size": getattr(
                            cluster_config.replica_config, "moe_tensor_parallel_size", 1
                        ),
                    },
                }

                model_weight_memory[cluster_type.name] = cluster_memory_info

                # Add cluster configuration to system architecture info
                system_architecture_info["cluster_configurations"][
                    cluster_type.name
                ] = {
                    "num_replicas": cluster_config.num_replicas,
                    "parallel_parameters": parallel_params,
                    "device_type": cluster_config.replica_config.device_config.get_type().name,
                    "model_name": cluster_config.replica_config.model_name,
                }

            except Exception as e:
                logger.warning(
                    f"Failed to calculate model weight memory for cluster {cluster_type.name}: {e}"
                )
                model_weight_memory[cluster_type.name] = {
                    "error": f"Failed to calculate: {str(e)}"
                }
                # Still add basic cluster configuration info even if memory calculation fails
                system_architecture_info["cluster_configurations"][
                    cluster_type.name
                ] = {
                    "num_replicas": cluster_config.num_replicas,
                    "error": f"Failed to extract configuration: {str(e)}",
                }

        if model_weight_memory:
            system_metrics["model_weight_memory"] = model_weight_memory
            system_metrics["memory_utilization_percent"] = {
                cluster_name: float(cluster_info.get("memory_utilization_percent", 0.0))
                for cluster_name, cluster_info in model_weight_memory.items()
            }

        if system_architecture_info["cluster_configurations"]:
            system_metrics["system_architecture_info"] = system_architecture_info

        # Write to JSON file
        output_path = f"{self._config.output_dir}/system_metrics.json"
        with open(output_path, "w") as f:
            json.dump(system_metrics, f, indent=2)

        logger.info(f"System metrics written to {output_path}")

    def _compute_percentile_stats(
        self, values: List[float], unit: str = "ms"
    ) -> Dict[str, Any]:
        """Compute percentile statistics for a list of values."""
        import numpy as np

        if not values:
            return {"error": "No data available"}

        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p50": float(np.percentile(values_array, 50)),
            "p90": float(np.percentile(values_array, 90)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
            "unit": unit,
        }

    @if_write_metrics
    def on_request_arrival(
        self, time: float, request: Request, cluster_type: ClusterType
    ) -> None:
        if not self._config.store_request_metrics:
            return

        # The following metrics are global and should only be recorded once per request
        # In disaggregated mode, requests arrive at multiple clusters (e.g., PREFILL → DECODE)
        # Use a flag to prevent duplicate recording
        if hasattr(request, "_metrics_recorded") and request._metrics_recorded:
            return

        # Mark as recorded to prevent duplicate recording in subsequent cluster arrivals
        request._metrics_recorded = True

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put(time, 1)
        self._write_metrics_ground_truth_record(
            {
                "event_type": "request_arrival",
                "time_s": float(time),
                "time_ms": float(time) * 1000.0,
                "request_id": int(request.id),
                "cluster_type": cluster_type.name,
                "arrived_at": float(request.arrived_at),
                "arrived_at_ms": float(request.arrived_at) * 1000.0,
                "num_prefill_tokens": int(request.num_prefill_tokens),
                "num_decode_tokens": int(request.num_decode_tokens),
            }
        )

        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id,
            request.user_facing_total_tokens,
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(
            request.id,
            request.user_facing_num_prefill_tokens,
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(
            request.id,
            request.user_facing_num_decode_tokens,
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PD_RATIO
        ].put(
            request.id,
            request.user_facing_pd_ratio,
        )
        if self._last_request_arrived_at is not None:
            self._request_metrics_histogram[
                RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(request.id, request.arrived_at - self._last_request_arrived_at)
        self._last_request_arrived_at = request.arrived_at

    def _get_ttft_export_value(self, request: Request) -> float:
        return request.ttft

    def _get_sj2q_pen_export_value(
        self,
        request: Request,
        penalty_attr_name: str,
        bounded_attr_name: str,
    ) -> int:
        if hasattr(request, penalty_attr_name):
            return int(getattr(request, penalty_attr_name, 0))
        return int(getattr(request, bounded_attr_name, 0))

    def _get_prefix_cache_block_size(self) -> Optional[int]:
        cluster_configs = getattr(self, "_cluster_configs", {})
        for cluster_type, cluster_config in cluster_configs.items():
            if cluster_type not in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
                continue
            replica_scheduler_config = getattr(
                cluster_config,
                "replica_scheduler_config",
                None,
            )
            if replica_scheduler_config is None:
                continue
            if not bool(
                getattr(replica_scheduler_config, "enable_prefix_caching", False)
            ):
                continue
            block_size = int(getattr(replica_scheduler_config, "block_size", 0))
            if block_size > 0:
                return block_size
        return None

    def _record_prefix_cache_request_metrics(self, request: Request) -> None:
        block_size = self._get_prefix_cache_block_size()
        if block_size is None or block_size <= 0:
            return
        query_blocks = len(request.block_hash_ids or [])
        hit_blocks = int(request.num_prefill_tokens_cached // block_size)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_CACHED_PREFILL_TOKENS
        ].put(request.id, int(request.num_prefill_tokens_cached))
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFIX_CACHE_QUERY_BLOCKS
        ].put(request.id, int(query_blocks))
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFIX_CACHE_HIT_BLOCKS
        ].put(request.id, int(hit_blocks))

    def _compute_prefix_cache_statistics(self) -> Optional[Dict[str, Any]]:
        cached_tokens_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_CACHED_PREFILL_TOKENS
        )
        query_blocks_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_PREFIX_CACHE_QUERY_BLOCKS
        )
        hit_blocks_data = self._request_metrics_histogram.get(
            RequestMetricsHistogram.REQUEST_PREFIX_CACHE_HIT_BLOCKS
        )
        if cached_tokens_data is None or len(cached_tokens_data) == 0:
            return None

        cached_values = [int(value) for _, value in cached_tokens_data._data_series]
        query_values = (
            [int(value) for _, value in query_blocks_data._data_series]
            if query_blocks_data is not None
            else []
        )
        hit_values = (
            [int(value) for _, value in hit_blocks_data._data_series]
            if hit_blocks_data is not None
            else []
        )
        total_query_blocks = int(sum(query_values))
        total_hit_blocks = int(sum(hit_values))
        requests = len(cached_values)
        requests_with_hits = sum(1 for value in cached_values if value > 0)

        return {
            "block_size_tokens": self._get_prefix_cache_block_size(),
            "requests": int(requests),
            "requests_with_hits": int(requests_with_hits),
            "total_cached_prefill_tokens": int(sum(cached_values)),
            "mean_cached_prefill_tokens": (
                float(sum(cached_values) / requests) if requests > 0 else 0.0
            ),
            "total_query_blocks": total_query_blocks,
            "total_hit_blocks": total_hit_blocks,
            "hit_ratio": (
                float(total_hit_blocks / total_query_blocks)
                if total_query_blocks > 0
                else 0.0
            ),
        }

    def _on_request_end(self, time: float, request: Request) -> None:
        # Update completion counters regardless of plotting preferences. Counter
        # update failures must propagate so corrupted completion accounting is
        # not reported as a successful simulation.
        if (
            getattr(self, "_completed_request_ids", None) is not None
            and request.id not in self._completed_request_ids
        ):
            self._completed_request_ids.add(request.id)
            self._completed_requests += 1

        if not self._config.store_request_metrics:
            return

        # Prevent duplicate recording of request end metrics in disaggregated mode
        # In disaggregated mode, on_batch_end() may be called multiple times for the same request
        # (e.g., once in PREFILL cluster, once in DECODE cluster)
        # Use a flag to ensure metrics are only recorded once when the request truly completes
        if hasattr(request, "_end_metrics_recorded") and request._end_metrics_recorded:
            return

        # Mark as recorded to prevent duplicate recording
        request._end_metrics_recorded = True

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)
        request_waiting_time_total = request.get_total_waiting_time()
        self._write_metrics_ground_truth_record(
            {
                "event_type": "request_completion",
                "time_s": float(time),
                "time_ms": float(time) * 1000.0,
                "request_id": int(request.id),
                "arrived_at": float(request.arrived_at),
                "arrived_at_ms": float(request.arrived_at) * 1000.0,
                "prefill_completed_at": float(request.prefill_completed_at),
                "prefill_completed_at_ms": float(request.prefill_completed_at)
                * 1000.0,
                "first_decode_token_completed_at": float(
                    request.first_decode_token_completed_at
                ),
                "first_decode_token_completed_at_ms": float(
                    request.first_decode_token_completed_at
                )
                * 1000.0,
                "decode_first_token_completed_at": float(
                    request.decode_first_token_completed_at
                ),
                "decode_first_token_completed_at_ms": float(
                    request.decode_first_token_completed_at
                )
                * 1000.0,
                "completed_at": float(request.completed_at),
                "completed_at_ms": float(request.completed_at) * 1000.0,
                "num_prefill_tokens": int(request.num_prefill_tokens),
                "num_decode_tokens": int(request.num_decode_tokens),
                "request_e2e_time_s": float(request.e2e_time),
                "request_e2e_time_ms": float(request.e2e_time) * 1000.0,
                "request_waiting_time_total_s": float(request_waiting_time_total),
                "request_waiting_time_total_ms": float(request_waiting_time_total)
                * 1000.0,
                "ttft_s": float(request.ttft),
                "ttft_ms": float(request.ttft) * 1000.0,
                "tpot_s": float(request.tpot),
                "tpot_ms": float(request.tpot) * 1000.0,
            }
        )

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, 0.0)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_FIRST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)

        # Per-cluster waiting time metrics (includes time after preemption)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_TOTAL
        ].put(request.id, request.get_total_waiting_time())
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_PREFILL
        ].put(request.id, request.get_cluster_waiting_time(ClusterType.PREFILL))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_DECODE
        ].put(request.id, request.get_cluster_waiting_time(ClusterType.DECODE))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_DECODE_ATTN
        ].put(request.id, request.get_cluster_waiting_time(ClusterType.DECODE_ATTN))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_DECODE_FFN
        ].put(request.id, request.get_cluster_waiting_time(ClusterType.DECODE_FFN))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_HIDDEN_WAITING_TIME
        ].put(request.id, request.get_round_class_waiting_time("hidden"))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_FINAL_WAITING_TIME
        ].put(request.id, request.get_round_class_waiting_time("final"))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_HIDDEN_SERVICE_TIME
        ].put(request.id, request.get_round_class_service_time("hidden"))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_FINAL_SERVICE_TIME
        ].put(request.id, request.get_round_class_service_time("final"))
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.FINAL_ROUND_PREFILL_WAIT_MS
        ].put(
            request.id,
            request.get_round_class_cluster_waiting_time(
                "final",
                ClusterType.PREFILL,
            ),
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.FINAL_ROUND_DECODE_WAIT_MS
        ].put(
            request.id,
            request.get_round_class_cluster_waiting_time(
                "final",
                ClusterType.DECODE,
            ),
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.FINAL_ROUND_PREFILL_SERVICE_MS
        ].put(
            request.id,
            request.get_round_class_cluster_service_time(
                "final",
                ClusterType.PREFILL,
            ),
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.LATE_HIDDEN_PREFILL_WAIT_MS
        ].put(
            request.id,
            request.get_round_numbers_cluster_waiting_time(
                round_numbers=(3, 4),
                cluster_type=ClusterType.PREFILL,
            ),
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_THINKING_TIME
        ].put(request.id, request.thinking_time_total)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_TOOL_CALL_TIME
        ].put(request.id, request.tool_call_time_total)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            request.execution_time / request.num_decode_tokens,
        )

        if request.is_prefill_complete:
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.PREFILL_E2E_TIME
            ].put(request.id, request.prefill_completed_at - request.arrived_at)
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.PREFILL_EXECUTION_PLUS_PREEMPTION
            ].put(request.id, request.prefill_completed_at - request.scheduled_at)
            # Guard against division by zero (defensive programming)
            # Normal requests should always have num_prefill_tokens >= 1, but
            # this protects against edge cases or synthetic test requests
            if request.num_prefill_tokens > 0:
                self._request_metrics_time_distributions[
                    RequestMetricsTimeDistributions.PREFILL_EXECUTION_PLUS_PREEMPTION_PER_TOKEN
                ].put(
                    request.id,
                    (request.prefill_completed_at - request.scheduled_at)
                    / request.num_prefill_tokens,
                )
            #
            # Guard against division by zero for decode tokens
            if request.num_decode_tokens > 0:
                self._request_metrics_time_distributions[
                    RequestMetricsTimeDistributions.DECODE_E2E_TIME_PER_TOKEN
                ].put(
                    request.id,
                    (request.completed_at - request.prefill_completed_at)
                    / request.num_decode_tokens,
                )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_THINKING_ROUND_COUNT
        ].put(request.id, request.completed_thinking_rounds)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SESSION_ID
        ].put(request.id, request.session_id)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_COHORT
        ].put(request.id, request.cohort)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SJ2Q_PEN_QSHORT_ENTRIES_TOTAL
        ].put(
            request.id,
            self._get_sj2q_pen_export_value(
                request,
                "_sj2q_penalty_qshort_entries_total",
                "_sj2q_bounded_qshort_entries_total",
            ),
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SJ2Q_PEN_QSHORT_LONG_HISTORY_ENTRIES_TOTAL
        ].put(
            request.id,
            self._get_sj2q_pen_export_value(
                request,
                "_sj2q_penalty_qshort_long_history_entries_total",
                "_sj2q_bounded_qshort_long_history_entries_total",
            ),
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SJ2Q_PEN_QLONG_ENTRIES_TOTAL
        ].put(
            request.id,
            self._get_sj2q_pen_export_value(
                request,
                "_sj2q_penalty_qlong_entries_total",
                "_sj2q_bounded_qlong_entries_total",
            ),
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SJ2Q_PEN_LONG_HISTORY_TO_QSHORT_REENTRY_COUNT
        ].put(
            request.id,
            self._get_sj2q_pen_export_value(
                request,
                "_sj2q_penalty_long_history_to_qshort_reentry_count",
                "_sj2q_bounded_long_history_to_qshort_reentry_count",
            ),
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SJ2Q_PEN_FIRST_LONG_HISTORY_ROUND_NUMBER
        ].put(
            request.id,
            self._get_sj2q_pen_export_value(
                request,
                "_sj2q_penalty_first_long_history_round_number",
                "_sj2q_bounded_first_long_history_round_number",
            ),
        )
        self._record_prefix_cache_request_metrics(request)

        spec_total_iterations = int(getattr(request, "spec_total_iterations", 0))
        spec_accepted_drafts = int(getattr(request, "spec_total_accepted_drafts", 0))
        spec_rejected_drafts = int(getattr(request, "spec_total_rejected_drafts", 0))
        spec_committed_tokens = int(getattr(request, "spec_total_committed_tokens", 0))
        spec_acceptance_ratio = 0.0
        spec_committed_per_iter = 0.0
        total_drafts = spec_accepted_drafts + spec_rejected_drafts
        if total_drafts > 0:
            spec_acceptance_ratio = spec_accepted_drafts / total_drafts
        if spec_total_iterations > 0:
            spec_committed_per_iter = spec_committed_tokens / spec_total_iterations

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_TOTAL_ITERATIONS
        ].put(request.id, spec_total_iterations)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_ACCEPTED_DRAFTS
        ].put(request.id, spec_accepted_drafts)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_REJECTED_DRAFTS
        ].put(request.id, spec_rejected_drafts)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_COMMITTED_TOKENS
        ].put(request.id, spec_committed_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_ACCEPTANCE_RATIO
        ].put(request.id, spec_acceptance_ratio)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_SPEC_COMMITTED_PER_ITER
        ].put(request.id, spec_committed_per_iter)

        # Record preemption tracking metrics (request-level)
        # NOTE: These metrics describe what happened to each individual request
        # For cluster-wide aggregate statistics, see self._preemption_statistics
        # which is exported to system_metrics.json
        # ClusterType is already imported at module level (line 51)

        # Total preemption count across all clusters
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_TOTAL_PREEMPTION_COUNT
        ].put(request.id, request.get_total_preemption_count())

        # Per-cluster preemption counts (request-level)
        # These provide detailed per-cluster breakdown for each request
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_PREEMPTION_COUNT
        ].put(request.id, request.get_preemption_count(ClusterType.PREFILL))

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_PREEMPTION_COUNT
        ].put(request.id, request.get_preemption_count(ClusterType.DECODE))

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_ATTN_PREEMPTION_COUNT
        ].put(request.id, request.get_preemption_count(ClusterType.DECODE_ATTN))
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_HIDDEN_PREEMPTION_COUNT
        ].put(request.id, request.get_round_class_preemption_count("hidden"))
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_FINAL_PREEMPTION_COUNT
        ].put(request.id, request.get_round_class_preemption_count("final"))

        # Complete tokens-at-preemption lists (request-level, raw data)
        # Format: comma-separated integers (e.g., "257,15" for 2 preemptions)
        # Empty string "" if never preempted in that cluster
        tokens_at_preemption_decode = request.get_tokens_at_preemption(
            ClusterType.DECODE
        )
        tokens_at_preemption_decode_str = (
            ",".join(map(str, tokens_at_preemption_decode))
            if tokens_at_preemption_decode
            else ""
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS_AT_PREEMPTION_ALL
        ].put(request.id, tokens_at_preemption_decode_str)

        tokens_at_preemption_decode_attn = request.get_tokens_at_preemption(
            ClusterType.DECODE_ATTN
        )
        tokens_at_preemption_decode_attn_str = (
            ",".join(map(str, tokens_at_preemption_decode_attn))
            if tokens_at_preemption_decode_attn
            else ""
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_ATTN_TOKENS_AT_PREEMPTION_ALL
        ].put(request.id, tokens_at_preemption_decode_attn_str)

        # Decode tokens-at-preemption statistics (aggregated summary)
        # Aggregate tokens-at-preemption data from both DECODE and DECODE_ATTN clusters
        # This provides a unified view of decode preemption behavior regardless of architecture mode
        all_decode_tokens_at_preemption = []

        # Collect from DECODE cluster (PD-disaggregation mode)
        if tokens_at_preemption_decode:
            all_decode_tokens_at_preemption.extend(tokens_at_preemption_decode)

        # Collect from DECODE_ATTN cluster (PD+AF-disaggregation mode)
        if tokens_at_preemption_decode_attn:
            all_decode_tokens_at_preemption.extend(tokens_at_preemption_decode_attn)

        # Calculate statistics from aggregated data
        if all_decode_tokens_at_preemption:
            mean_tokens = sum(all_decode_tokens_at_preemption) / len(
                all_decode_tokens_at_preemption
            )
            max_tokens = max(all_decode_tokens_at_preemption)
            min_tokens = min(all_decode_tokens_at_preemption)
        else:
            # No decode preemptions occurred for this request
            mean_tokens = 0
            max_tokens = 0
            min_tokens = 0

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS_AT_PREEMPTION_MEAN
        ].put(request.id, mean_tokens)

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS_AT_PREEMPTION_MAX
        ].put(request.id, max_tokens)

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS_AT_PREEMPTION_MIN
        ].put(request.id, min_tokens)

        # Collect system-level preemption statistics for aggregate metrics
        # This data will be used to generate cluster-level statistics in system_metrics.json
        total_preemptions = request.get_total_preemption_count()
        if total_preemptions > 0:
            # Track this request as preempted
            self._preemption_statistics["preempted_request_ids"].add(request.id)
            self._preemption_statistics["total_preemption_events"] += total_preemptions

            # Collect per-cluster statistics
            for cluster_type in [
                ClusterType.MONOLITHIC,
                ClusterType.PREFILL,
                ClusterType.DECODE,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE_FFN,
            ]:
                cluster_preemption_count = request.get_preemption_count(cluster_type)
                if cluster_preemption_count > 0:
                    cluster_stats = self._preemption_statistics["by_cluster_type"][
                        cluster_type
                    ]
                    cluster_stats["preemption_events"] += cluster_preemption_count
                    cluster_stats["preempted_request_ids"].add(request.id)

                    # Collect tokens-at-preemption for DECODE clusters
                    if cluster_type in [ClusterType.DECODE, ClusterType.DECODE_ATTN]:
                        tokens_list = request.get_tokens_at_preemption(cluster_type)
                        cluster_stats["tokens_at_preemption"].extend(tokens_list)

        # TTFT (Time To First Token) metrics
        if request.is_prefill_complete:
            ttft_value = self._get_ttft_export_value(request)
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TTFT
            ].put(request.id, ttft_value)

            # TTFT breakdown components
            if request.is_prefill_complete:
                # Prefill execution time
                prefill_exec_time = request.prefill_completed_at - request.scheduled_at
                self._request_metrics_time_distributions[
                    RequestMetricsTimeDistributions.TTFT_PREFILL_ONLY
                ].put(request.id, prefill_exec_time)

                # KV cache transfer time
                self._request_metrics_time_distributions[
                    RequestMetricsTimeDistributions.TTFT_KV_TRANSFER
                ].put(request.id, request.kv_cache_transfer_time)

                # Legacy TTFT residual kept for backward-compatible reporting.
                # This is no longer the canonical first pure decode latency.
                # Use decode_first_token_latency for the actual first pure decode
                # token duration after TTFT re-anchoring.
                ttft_decode_first = (
                    ttft_value - prefill_exec_time - request.kv_cache_transfer_time
                )
                self._request_metrics_time_distributions[
                    RequestMetricsTimeDistributions.TTFT_DECODE_FIRST_TOKEN
                ].put(request.id, max(0, ttft_decode_first))  # Ensure non-negative

        if request.num_decode_tokens > 1:
            if request.first_decode_token_completed_at <= 0:
                raise ValueError(
                    f"Missing first token completion timestamp for request_id={request.id}"
                )
            if request.decode_first_token_completed_at <= 0:
                raise ValueError(
                    f"Missing decode-first-token completion timestamp for request_id={request.id}"
                )
            decode_first_token_latency = (
                request.decode_first_token_completed_at - request.first_decode_token_completed_at
            )
            if decode_first_token_latency < -1e-9:
                raise ValueError(
                    "Negative decode-first-token latency for "
                    f"request_id={request.id}: {decode_first_token_latency} "
                    f"(first_decode_token_completed_at={request.first_decode_token_completed_at}, "
                    f"decode_first_token_completed_at={request.decode_first_token_completed_at})"
                )
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.DECODE_FIRST_TOKEN_LATENCY
            ].put(request.id, max(0, decode_first_token_latency))

        # TPOT (Time Per Output Token) metrics
        if (
            request.num_decode_tokens > 1
            and request.first_decode_token_completed_at > 0
        ):
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TPOT
            ].put(request.id, request.tpot)

            # TPOT breakdown: computation vs transfer
            # Total decode time after first token
            total_decode_time_after_first = (
                request.completed_at - request.first_decode_token_completed_at
            )
            # M2N transfer time per token (excluding first token)
            tpot_transfer = request.total_m2n_transfer_time / (
                request.num_decode_tokens - 1
            )
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TPOT_TRANSFER
            ].put(request.id, tpot_transfer)

            # Computation time per token = TPOT - transfer_per_token
            tpot_computation = request.tpot - tpot_transfer
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TPOT_COMPUTATION
            ].put(request.id, max(0, tpot_computation))  # Ensure non-negative

        # Transfer time metrics (disaggregated modes)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.TRANSFER_KV_CACHE
        ].put(request.id, request.kv_cache_transfer_time)
        if request.kv_cache_transfer_start_time is not None:
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TRANSFER_KV_CACHE_REQUEST_START_TS
            ].put(request.id, request.kv_cache_transfer_start_time)
        if request.kv_cache_transfer_end_time is not None:
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TRANSFER_KV_CACHE_REQUEST_END_TS
            ].put(request.id, request.kv_cache_transfer_end_time)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.TRANSFER_M2N_TOTAL
        ].put(request.id, request.total_m2n_transfer_time)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.TRANSFER_M2N_ATTN_TO_FFN
        ].put(request.id, request.m2n_transfer_time_attn_to_ffn)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.TRANSFER_M2N_FFN_TO_ATTN
        ].put(request.id, request.m2n_transfer_time_ffn_to_attn)

        # Cluster computation time metrics
        # Get per-cluster execution times from request's cluster-specific tracking
        # ClusterType is already imported at module level (line 51)

        prefill_computation = sum(request._execution_time.get(ClusterType.PREFILL, []))
        decode_attn_computation = sum(
            request._execution_time.get(ClusterType.DECODE_ATTN, [])
        )
        decode_ffn_computation = sum(
            request._execution_time.get(ClusterType.DECODE_FFN, [])
        )
        decode_computation = sum(
            request._execution_time.get(ClusterType.DECODE, [])
        )  # PD mode

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_PREFILL_COMPUTATION
        ].put(request.id, prefill_computation)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_DECODE_ATTN_COMPUTATION
        ].put(request.id, decode_attn_computation)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_DECODE_FFN_COMPUTATION
        ].put(request.id, decode_ffn_computation)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_DECODE_COMPUTATION
        ].put(request.id, decode_computation)

        # Decode phase E2E time metrics
        # decode_e2e_time: Total decode phase time (from prefill completion to request completion)
        decode_e2e_time = request.completed_at - request.prefill_completed_at
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_E2E_TIME
        ].put(request.id, decode_e2e_time)

        # cluster_decode_attn_e2e_time: DECODE_ATTN cluster computation time only (PD+AF mode)
        # This is the same as cluster_decode_attn_computation (pure computation, no transfer)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_DECODE_ATTN_E2E_TIME
        ].put(request.id, decode_attn_computation)

        # cluster_decode_ffn_e2e_time: request-level DECODE_FFN residence time (PD+AF mode)
        # AUDIT decision: request-level DECODE_FFN timing should track residence time
        # defined by (A→F transfer end) -> (F→A transfer start).
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLUSTER_DECODE_FFN_E2E_TIME
        ].put(request.id, request.decode_ffn_residence_time)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        if not self._config.store_token_completion_metrics:
            return

        # if prefill has just finished in this iteration, update the prefill completion time series
        prefill_completed_at = request._prefill_completed_at
        if prefill_completed_at > 0 and time == prefill_completed_at:
            self._token_completion_metrics_time_series[
                TokenCompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,
            )

        # determine if this was prefill or decode token
        if not request.has_started_decode:
            return

        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(
            time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
        )

        self._token_completion_metrics_time_series[
            TokenCompletionMetricsTimeSeries.DECODE_COMPLETIONS
        ].put(time, 1)

    def _push_metric(
        self,
        metric_name: OperationMetrics,
        batch_id: int,
        value: float,
        cluster_type: ClusterType,
    ) -> None:
        if metric_name in OperationMetrics:
            self._operation_metrics[cluster_type][metric_name].put(value)
            self._operation_metrics_per_batch[cluster_type][metric_name].put(
                batch_id, value
            )
        elif metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[cluster_type][metric_name].put(value)
            self._cpu_operation_metrics_per_batch[cluster_type][metric_name].put(
                batch_id, value
            )
        elif metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[cluster_type][metric_name].put(value)
            self._batch_metrics_time_distribution_per_batch[cluster_type][
                metric_name
            ].put(batch_id, value)
        elif metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[cluster_type][metric_name].put(value)
            self._batch_metrics_count_distribution_per_batch[cluster_type][
                metric_name
            ].put(batch_id, value)
        else:
            raise ValueError(f"Invalid metric name {metric_name}")

    @if_write_metrics
    def on_batch_end(
        self,
        time: float,
        batch: Batch,
        replica_id: int,
        memory_usage_percent: int,
        cluster_type: ClusterType,
        dp_id: int = 0,
    ) -> None:
        if (
            self._config.min_batch_index and batch.id < self._config.min_batch_index
        ) or (self._config.max_batch_index and batch.id > self._config.max_batch_index):
            return

        for request in batch.completed_requests:
            self._on_request_end(time, request)

        if self._config.store_utilization_metrics:
            replica_index = self._get_cluster_replica_index(cluster_type, replica_id)
            self._replica_memory_usage[cluster_type][replica_index][dp_id].put(
                time, memory_usage_percent
            )

        for request in batch.requests:
            self._update_per_token_execution_times(time, request, batch)

        if not self._config.store_batch_metrics:
            return

        self._push_metric(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME,
            batch.id,
            time - batch.scheduled_at,
            cluster_type,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS,
            batch.id,
            batch.total_num_tokens,
            cluster_type,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS,
            batch.id,
            batch.num_prefill_tokens,
            cluster_type,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS,
            batch.id,
            batch.num_decode_tokens,
            cluster_type,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_SIZE, batch.id, batch.size, cluster_type
        )

    @if_write_metrics
    def on_replica_schedule(
        self,
        time: float,
        replica_id: int,
        memory_usage_percent: int,
        cluster_type: ClusterType,
        dp_id: int = 0,
    ) -> None:
        if not self._config.store_utilization_metrics:
            return

        # Convert global replica_id to cluster-relative index
        replica_index = self._get_cluster_replica_index(cluster_type, replica_id)
        self._replica_memory_usage[cluster_type][replica_index][dp_id].put(
            time, memory_usage_percent
        )

    def _get_cluster_replica_index(
        self, cluster_type: ClusterType, replica_id: int
    ) -> int:
        """
        Convert global replica_id to cluster-relative index.

        In disaggregated systems, replica_ids are globally unique across clusters,
        but metrics arrays are indexed by cluster-relative positions (0, 1, 2, ...).

        Args:
            cluster_type: The cluster type
            replica_id: Global replica ID

        Returns:
            Cluster-relative replica index (0-based)
        """
        # Get all replica IDs for this cluster type
        cluster_config = self._cluster_configs[cluster_type]

        # For now, use a simple mapping: assume replicas are assigned sequentially
        # within each cluster starting from 0
        # This works for the current disaggregated setup where each cluster has 1 replica
        num_replicas = cluster_config.num_replicas

        # Simple approach: use modulo to map global replica_id to cluster-relative index
        # This assumes replica_ids are assigned in a predictable pattern
        replica_index = 0  # For single-replica clusters, always use index 0

        # Validate the index is within bounds
        if replica_index >= num_replicas:
            raise ValueError(
                f"Replica index {replica_index} out of bounds for cluster {cluster_type.name} "
                f"with {num_replicas} replicas. Global replica_id: {replica_id}"
            )

        return replica_index

    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
        cluster_type: ClusterType,
        dp_id: int = 0,
    ) -> None:
        # Emit op-level traces if enabled (independent of write_metrics flag)
        # This must be called FIRST because it uses the raw simulation time,
        # and trace events should capture the start time of the stage execution.
        # Note: Op-level tracing is controlled by enable_op_level_tracing, NOT write_metrics
        if execution_time is not None:
            request_ids = (
                [str(rid) for rid in batch_stage.request_ids] if batch_stage else []
            )
            trace_execution_time = getattr(
                execution_time,
                "_trace_execution_time_override",
                execution_time,
            )
            self._emit_op_level_traces(
                time=time,
                batch_stage=batch_stage,
                replica_id=replica_id,
                execution_time=trace_execution_time,
                cluster_type=cluster_type,
                request_ids=request_ids,
            )

        if (
            self._config.write_metrics
            and self._should_capture_frontier_stage_batch_ledger()
            and execution_time is not None
            and batch_stage is not None
        ):
            ledger_row_id = id(batch_stage)
            ledger_row = self._build_frontier_stage_batch_ledger_capture_row(
                batch_stage=batch_stage,
                execution_time=execution_time,
                replica_id=replica_id,
                stage_id=stage_id,
                cluster_type=cluster_type,
                dp_id=dp_id,
                stage_end_time=time + batch_stage.execution_time,
            )
            ledger_key = self._frontier_stage_batch_ledger_key(
                batch_id=batch_stage._batch_id,
                replica_id=replica_id,
                stage_id=stage_id,
                cluster_type=cluster_type,
                dp_id=dp_id,
            )
            self._pending_frontier_stage_batch_ledger_rows[ledger_row_id] = ledger_row
            self._pending_frontier_stage_batch_ledger_row_keys[ledger_key] = ledger_row_id
            self._pending_frontier_stage_batch_ledger_rows_by_key[ledger_key] = ledger_row

        if not self._config.write_metrics:
            return

        if not self._config.store_utilization_metrics:
            return

        replica_index = self._get_cluster_replica_index(cluster_type, replica_id)
        dp_stage_busy = self._replica_busy_time[cluster_type][replica_index]
        if dp_id < 0 or dp_id >= len(dp_stage_busy):
            raise ValueError(
                f"Invalid lane id for utilization metrics: cluster={cluster_type.name} "
                f"replica={replica_id} lane={dp_id} available_lanes={len(dp_stage_busy)}"
            )
        if stage_id < 0 or stage_id >= len(dp_stage_busy[dp_id]):
            raise ValueError(
                f"Invalid stage id for utilization metrics: cluster={cluster_type.name} "
                f"replica={replica_id} lane={dp_id} stage={stage_id} "
                f"available_stages={len(dp_stage_busy[dp_id])}"
            )

        self._replica_busy_time[cluster_type][replica_index][dp_id][stage_id].put(
            time, 100
        )
        mfu = self._mfu_calculator[cluster_type].get_mfu(batch_stage)
        self._replica_mfu[cluster_type][replica_index][dp_id][stage_id].put(
            time, mfu
        )

        if not self._config.store_operation_metrics:
            return

        cluster_config = self._cluster_configs.get(cluster_type)
        moe_tp_enabled = False
        ep_enabled = False
        use_step3_ep_alltoall = False
        use_ep_alltoall_dispatch_combine = False
        if execution_time._is_moe:
            if cluster_config is None:
                raise ValueError(f"Cluster config not found for {cluster_type}")
            moe_tp_enabled = cluster_config.replica_config.moe_tensor_parallel_size > 1
            ep_enabled = cluster_config.replica_config.moe_expert_parallel_size > 1
            model_config = cluster_config.replica_config.model_config
            use_step3_ep_alltoall = (
                model_config is not None
                and model_config.model_type == "step3_text"
                and cluster_type
                in (
                    ClusterType.PREFILL,
                    ClusterType.DECODE,
                    ClusterType.DECODE_FFN,
                    ClusterType.MONOLITHIC,
                )
            )
            use_ep_alltoall_dispatch_combine = (
                self._use_ep_alltoall_dispatch_combine(
                    cluster_type=cluster_type,
                    batch_stage=batch_stage,
                )
            )

        batch_id = batch_stage._batch_id
        for _ in range(execution_time.num_layers):
            self._push_metric(
                OperationMetrics.MLP_UP_PROJ,
                batch_id,
                execution_time.mlp_layer_up_proj_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MLP_ACTIVATION,
                batch_id,
                execution_time.mlp_layer_act_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ,
                batch_id,
                execution_time.mlp_layer_down_proj_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.mlp_all_reduce_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_PRE_PROJ,
                batch_id,
                execution_time.attention_pre_proj_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ,
                batch_id,
                execution_time.attention_post_proj_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.attention_all_reduce_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_TENSOR_PARALLEL_ALLREDUCE,
                batch_id,
                execution_time.attention_all_reduce_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MOE_TENSOR_PARALLEL_ALLGATHER,
                batch_id,
                execution_time.moe_tensor_parallel_allgather_time,
                cluster_type,
            )
            if moe_tp_enabled:
                self._push_metric(
                    OperationMetrics.MOE_TENSOR_PARALLEL_ALLREDUCE,
                    batch_id,
                    execution_time.mlp_all_reduce_time,
                    cluster_type,
                )
            self._push_metric(
                OperationMetrics.SHARE_EXPERT_TENSOR_PARALLEL_ALLREDUCE,
                batch_id,
                execution_time.share_expert_tensor_parallel_allreduce_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.INPUT_LAYERNORM,
                batch_id,
                execution_time.attn_norm_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.POST_ATTENTION_LAYERNORM,
                batch_id,
                execution_time.mlp_norm_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ADD,
                batch_id,
                execution_time.add_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ADD_ATTN_RESIDUAL,
                batch_id,
                execution_time.add_attn_residual_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ADD_FFN_RESIDUAL,
                batch_id,
                execution_time.add_ffn_residual_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_PREFILL,
                batch_id,
                execution_time.attention_prefill_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_DECODE,
                batch_id,
                execution_time.attention_decode_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_KV_CACHE_SAVE,
                batch_id,
                execution_time.attention_kv_cache_save_execution_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.ATTN_ROPE,
                batch_id,
                execution_time.attention_rope_execution_time,
                cluster_type,
            )
            if ep_enabled and (
                use_step3_ep_alltoall or use_ep_alltoall_dispatch_combine
            ):
                self._push_metric(
                    OperationMetrics.EXPERT_PARALLEL_ALLTOALL_DISPATCH,
                    batch_id,
                    execution_time.expert_parallel_communication_time / 2,
                    cluster_type,
                )
            self._push_metric(
                OperationMetrics.MOE_GATING_LINEAR,
                batch_id,
                execution_time.moe_gating_linear_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MOE_GATING_ROUTING_TOPK,
                batch_id,
                execution_time.moe_gating_routing_topk_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MOE_SHUFFLING,
                batch_id,
                execution_time.moe_shuffling_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.MOE_GROUPED_GEMM,
                batch_id,
                execution_time.moe_grouped_gemm_time,
                cluster_type,
            )
            if ep_enabled and (
                use_step3_ep_alltoall or use_ep_alltoall_dispatch_combine
            ):
                self._push_metric(
                    OperationMetrics.EXPERT_PARALLEL_ALLTOALL_COMBINE,
                    batch_id,
                    execution_time.expert_parallel_communication_time / 2,
                    cluster_type,
                )
            self._push_metric(
                OperationMetrics.SHARE_EXPERT_UP_PROJ,
                batch_id,
                execution_time.share_expert_up_proj_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.SHARE_EXPERT_ACT,
                batch_id,
                execution_time.share_expert_act_time,
                cluster_type,
            )
            self._push_metric(
                OperationMetrics.SHARE_EXPERT_DOWN_PROJ,
                batch_id,
                execution_time.share_expert_down_proj_time,
                cluster_type,
            )

    @if_write_metrics
    def on_batch_stage_end(
        self,
        batch_stage: BatchStage,
        time: float,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int = 0,
    ) -> None:
        if not self._config.store_utilization_metrics:
            return
        replica_index = self._get_cluster_replica_index(cluster_type, replica_id)
        dp_stage_busy = self._replica_busy_time[cluster_type][replica_index]
        if dp_id < 0 or dp_id >= len(dp_stage_busy):
            raise ValueError(
                f"Invalid lane id for utilization metrics end hook: cluster={cluster_type.name} "
                f"replica={replica_id} lane={dp_id} available_lanes={len(dp_stage_busy)}"
            )
        if stage_id < 0 or stage_id >= len(dp_stage_busy[dp_id]):
            raise ValueError(
                f"Invalid stage id for utilization metrics end hook: cluster={cluster_type.name} "
                f"replica={replica_id} lane={dp_id} stage={stage_id} "
                f"available_stages={len(dp_stage_busy[dp_id])}"
            )

        self._replica_busy_time[cluster_type][replica_index][dp_id][stage_id].put(
            time, 0
        )
        self._replica_mfu[cluster_type][replica_index][dp_id][stage_id].put(time, 0)

        ledger_row = self._pending_frontier_stage_batch_ledger_rows.pop(id(batch_stage), None)
        if ledger_row is not None:
            self._pending_frontier_stage_batch_ledger_row_keys.pop(
                ledger_key := self._frontier_stage_batch_ledger_key(
                    batch_id=ledger_row["batch_id"],
                    replica_id=ledger_row["replica_id"],
                    stage_id=ledger_row["stage_id"],
                    cluster_type=ledger_row["cluster_type"],
                    dp_id=ledger_row["dp_id"],
                ),
                None,
            )
            self._pending_frontier_stage_batch_ledger_rows_by_key.pop(ledger_key, None)
            ledger_row["stage_end_ts"] = time
            self._complete_frontier_stage_batch_ledger_row(ledger_row)

    def _build_frontier_stage_batch_component_ledger(
        self,
        execution_time: ExecutionTime,
    ) -> dict[str, float]:
        component_ledger = {
            "attention_prefill_execution_time": _round_ledger_ms(
                execution_time.attention_prefill_execution_time
            ),
            "attention_decode_execution_time": _round_ledger_ms(
                execution_time.attention_decode_execution_time
            ),
            "attention_pre_proj_time": _round_ledger_ms(execution_time.attention_pre_proj_time),
            "attention_post_proj_time": _round_ledger_ms(
                execution_time.attention_post_proj_time
            ),
            "attention_kv_cache_save_execution_time": _round_ledger_ms(
                execution_time.attention_kv_cache_save_execution_time
            ),
            "attention_rope_execution_time": _round_ledger_ms(
                execution_time.attention_rope_execution_time
            ),
            "attn_norm_time": _round_ledger_ms(execution_time.attn_norm_time),
            "mlp_layer_up_proj_execution_time": _round_ledger_ms(
                execution_time.mlp_layer_up_proj_execution_time
            ),
            "mlp_layer_act_execution_time": _round_ledger_ms(
                execution_time.mlp_layer_act_execution_time
            ),
            "mlp_layer_down_proj_execution_time": _round_ledger_ms(
                execution_time.mlp_layer_down_proj_execution_time
            ),
            "mlp_norm_time": _round_ledger_ms(execution_time.mlp_norm_time),
            "attention_all_reduce_time": _round_ledger_ms(execution_time.attention_all_reduce_time),
            "mlp_all_reduce_time": _round_ledger_ms(execution_time.mlp_all_reduce_time),
            "moe_tensor_parallel_allgather_time": _round_ledger_ms(
                execution_time.moe_tensor_parallel_allgather_time
            ),
            "share_expert_tensor_parallel_allreduce_time": _round_ledger_ms(
                execution_time.share_expert_tensor_parallel_allreduce_time
            ),
            "dp_input_allreduce_time": _round_ledger_ms(execution_time.dp_input_allreduce_time),
            "dp_output_allreduce_time": _round_ledger_ms(execution_time.dp_output_allreduce_time),
            "pipeline_parallel_communication_time": _round_ledger_ms(
                execution_time.pipeline_parallel_communication_time
            ),
            "expert_parallel_communication_time": _round_ledger_ms(
                execution_time.expert_parallel_communication_time
            ),
            "moe_gating_linear_time": _round_ledger_ms(execution_time.moe_gating_linear_time),
            "moe_gating_routing_topk_time": _round_ledger_ms(
                execution_time.moe_gating_routing_topk_time
            ),
            "moe_shuffling_time": _round_ledger_ms(execution_time.moe_shuffling_time),
            "moe_grouped_gemm_time": _round_ledger_ms(execution_time.moe_grouped_gemm_time),
            "share_expert_up_proj_time": _round_ledger_ms(
                execution_time.share_expert_up_proj_time
            ),
            "share_expert_act_time": _round_ledger_ms(execution_time.share_expert_act_time),
            "share_expert_down_proj_time": _round_ledger_ms(
                execution_time.share_expert_down_proj_time
            ),
            "add_attn_residual_time": _round_ledger_ms(
                execution_time.add_attn_residual_time
            ),
            "add_ffn_residual_time": _round_ledger_ms(execution_time.add_ffn_residual_time),
            "schedule_time": _round_ledger_ms(execution_time.schedule_time),
            "sampler_e2e_time": _round_ledger_ms(execution_time.sampler_e2e_time),
            "prepare_inputs_e2e_time": _round_ledger_ms(
                execution_time.prepare_inputs_e2e_time
            ),
            "pp_producer_send_path_runtime_time": _round_ledger_ms(
                execution_time.pp_producer_send_path_runtime_time
            ),
            "pp_receiver_head_runtime_time": _round_ledger_ms(
                execution_time.pp_receiver_head_runtime_time
            ),
            "pp_prefill_consumer_active_runtime_time": _round_ledger_ms(
                execution_time.pp_prefill_consumer_active_runtime_time
            ),
            "pp_stage_boundary_residual_runtime_time": _round_ledger_ms(
                execution_time.pp_stage_boundary_residual_runtime_time
            ),
            "process_model_outputs_time": _round_ledger_ms(
                execution_time.process_model_outputs_time
            ),
            "ray_comm_time": _round_ledger_ms(execution_time.ray_comm_time),
            "decode_draft_proposer_time": _round_ledger_ms(
                execution_time.decode_draft_proposer_time
            ),
            "mtp_terminal_overshoot_time": _round_ledger_ms(
                execution_time.mtp_terminal_overshoot_time
            ),
        }
        return component_ledger

    def _build_frontier_stage_batch_diagnostic_component_ledger(
        self,
        execution_time: ExecutionTime,
    ) -> dict[str, float]:
        return {
            "pp_stage_boundary_handoff_time": _round_ledger_ms(
                execution_time.pp_stage_boundary_handoff_time
            ),
        }

    def _frontier_stage_batch_ledger_key(
        self,
        *,
        batch_id: int,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType | str,
        dp_id: int,
    ) -> tuple[str, int, int, int, int]:
        cluster_type_name = (
            cluster_type.name if isinstance(cluster_type, ClusterType) else str(cluster_type)
        )
        return (
            cluster_type_name,
            int(replica_id),
            int(dp_id),
            int(stage_id),
            int(batch_id),
        )

    def flush_frontier_stage_batch_ledger_row(
        self,
        *,
        time: float,
        batch_id: int,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int,
        completion_source: str = "manual_flush",
    ) -> None:
        if (
            not self._config.write_metrics
            or not self._should_capture_frontier_stage_batch_ledger()
        ):
            return

        ledger_key = self._frontier_stage_batch_ledger_key(
            batch_id=batch_id,
            replica_id=replica_id,
            stage_id=stage_id,
            cluster_type=cluster_type,
            dp_id=dp_id,
        )
        ledger_row = self._pending_frontier_stage_batch_ledger_rows_by_key.pop(
            ledger_key, None
        )
        if ledger_row is None:
            raise ValueError(
                "Missing pending Frontier stage-batch ledger row for manual flush: "
                f"cluster={cluster_type.name}, replica_id={replica_id}, dp_id={dp_id}, "
                f"stage_id={stage_id}, batch_id={batch_id}"
            )
        ledger_row_id = self._pending_frontier_stage_batch_ledger_row_keys.pop(
            ledger_key, None
        )
        if (
            ledger_row_id is not None
            and self._pending_frontier_stage_batch_ledger_rows.get(ledger_row_id)
            is ledger_row
        ):
            self._pending_frontier_stage_batch_ledger_rows.pop(ledger_row_id, None)

        ledger_row["stage_completion_observed_ts"] = float(time)
        ledger_row["stage_completion_observed_source"] = completion_source
        ledger_row["observed_stage_duration_ms"] = _round_ledger_ms(
            (float(time) - float(ledger_row["stage_start_ts"])) * 1e3
        )
        self._complete_frontier_stage_batch_ledger_row(ledger_row)

    def _build_frontier_stage_batch_ledger_row(
        self,
        *,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int,
        stage_end_time: float,
    ) -> dict[str, Any]:
        component_ledger_ms = self._build_frontier_stage_batch_component_ledger(execution_time)
        diagnostic_component_ledger_ms = (
            self._build_frontier_stage_batch_diagnostic_component_ledger(execution_time)
        )
        total_time_ms = _round_ledger_ms(execution_time.total_time * 1e3)
        component_sum_ms = _round_ledger_ms(sum(component_ledger_ms.values()))
        if abs(component_sum_ms - total_time_ms) > 1e-6:
            raise ValueError(
                "Frontier stage-batch component ledger must sum to total_time_ms: "
                f"component_sum_ms={component_sum_ms}, total_time_ms={total_time_ms}, "
                f"batch_id={batch_stage._batch_id}, stage_id={stage_id}"
            )
        diagnostic_total_time_ms = _round_ledger_ms(execution_time.diagnostic_total_time_ms)
        diagnostic_component_sum_ms = _round_ledger_ms(
            total_time_ms + sum(diagnostic_component_ledger_ms.values())
        )
        if abs(diagnostic_component_sum_ms - diagnostic_total_time_ms) > 1e-6:
            raise ValueError(
                "Frontier stage-batch diagnostic ledger must sum to diagnostic_total_time_ms: "
                f"diagnostic_component_sum_ms={diagnostic_component_sum_ms}, "
                f"diagnostic_total_time_ms={diagnostic_total_time_ms}, "
                f"batch_id={batch_stage._batch_id}, stage_id={stage_id}"
            )

        row = {
            "batch_id": int(batch_stage._batch_id),
            "stage_id": int(stage_id),
            "cluster_type": cluster_type.name,
            "replica_id": int(replica_id),
            "dp_id": int(dp_id),
            "request_ids": [str(request_id) for request_id in batch_stage.request_ids],
            "request_num_tokens": [int(token_count) for token_count in batch_stage.num_tokens],
            "stage_start_ts": float(batch_stage.scheduled_at),
            "stage_end_ts": float(stage_end_time),
            "execution_time": {
                "total_time_ms": total_time_ms,
                "diagnostic_total_time_ms": diagnostic_total_time_ms,
                "model_time_ms": _round_ledger_ms(execution_time.model_time_ms),
                "component_ledger_ms": component_ledger_ms,
                "diagnostic_component_ledger_ms": diagnostic_component_ledger_ms,
            },
        }
        if hasattr(batch_stage, "source_batch_ids"):
            row["source_batch_ids"] = [
                int(batch_id) for batch_id in batch_stage.source_batch_ids
            ]
        if hasattr(batch_stage, "source_request_ids"):
            row["source_request_ids"] = [
                str(request_id) for request_id in batch_stage.source_request_ids
            ]
        if hasattr(batch_stage, "source_request_num_tokens"):
            row["source_request_num_tokens"] = [
                int(token_count)
                for token_count in batch_stage.source_request_num_tokens
            ]
        if hasattr(batch_stage, "source_batch_arrival_times"):
            row["source_batch_arrival_times"] = [
                float(arrival_time)
                for arrival_time in batch_stage.source_batch_arrival_times
            ]
        if hasattr(batch_stage, "source_group_ready_ts"):
            row["source_group_ready_ts"] = float(batch_stage.source_group_ready_ts)
        if hasattr(batch_stage, "ep_id"):
            row["ep_id"] = int(batch_stage.ep_id)
        if hasattr(batch_stage, "per_expert_tokens"):
            row["per_expert_tokens"] = {
                str(expert_id): int(token_count)
                for expert_id, token_count in batch_stage.per_expert_tokens.items()
            }
        return row

    def _build_frontier_stage_batch_ledger_capture_row(
        self,
        *,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int,
        stage_end_time: float,
    ) -> dict[str, Any]:
        if getattr(self._config, "store_frontier_stage_batch_ledger", True):
            return self._build_frontier_stage_batch_ledger_row(
                batch_stage=batch_stage,
                execution_time=execution_time,
                replica_id=replica_id,
                stage_id=stage_id,
                cluster_type=cluster_type,
                dp_id=dp_id,
                stage_end_time=stage_end_time,
            )
        return self._build_frontier_stage_batch_ledger_summary_row(
            batch_stage=batch_stage,
            execution_time=execution_time,
            replica_id=replica_id,
            stage_id=stage_id,
            cluster_type=cluster_type,
            dp_id=dp_id,
            stage_end_time=stage_end_time,
        )

    def _build_frontier_stage_batch_ledger_summary_row(
        self,
        *,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
        replica_id: int,
        stage_id: int,
        cluster_type: ClusterType,
        dp_id: int,
        stage_end_time: float,
    ) -> dict[str, Any]:
        row = {
            "batch_id": int(batch_stage._batch_id),
            "stage_id": int(stage_id),
            "cluster_type": cluster_type.name,
            "replica_id": int(replica_id),
            "dp_id": int(dp_id),
            "stage_start_ts": float(batch_stage.scheduled_at),
            "stage_end_ts": float(stage_end_time),
            "execution_time": {
                "total_time_ms": _round_ledger_ms(execution_time.total_time * 1e3),
            },
        }
        if hasattr(batch_stage, "source_batch_ids"):
            row["source_batch_ids"] = [
                int(batch_id) for batch_id in batch_stage.source_batch_ids
            ]
        if hasattr(batch_stage, "source_request_ids"):
            row["source_request_ids"] = [
                str(request_id) for request_id in batch_stage.source_request_ids
            ]
        if hasattr(batch_stage, "source_batch_arrival_times"):
            row["source_batch_arrival_times"] = [
                float(arrival_time)
                for arrival_time in batch_stage.source_batch_arrival_times
            ]
        if hasattr(batch_stage, "source_group_ready_ts"):
            row["source_group_ready_ts"] = float(batch_stage.source_group_ready_ts)
        return row

    def _should_capture_frontier_stage_batch_ledger(self) -> bool:
        return bool(
            getattr(self._config, "store_frontier_stage_batch_ledger", True)
            or getattr(self._config, "store_frontier_stage_batch_ledger_summary", False)
        )

    def _complete_frontier_stage_batch_ledger_row(self, row: dict[str, Any]) -> None:
        if getattr(self._config, "store_frontier_stage_batch_ledger_summary", False):
            self._update_frontier_stage_batch_ledger_summary(row)
        if getattr(self._config, "store_frontier_stage_batch_ledger", True):
            self._frontier_stage_batch_ledger_rows.append(row)

    def _new_frontier_stage_batch_ledger_summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "total_rows": 0,
            "cluster_counts": {},
            "decode_ffn": {
                "rows": 0,
                "completion_sources": {},
                "missing_source_rows": 0,
                "rows_missing_required_metadata": 0,
                "negative_queue_groups": 0,
            },
        }

    def _increment_frontier_stage_batch_summary_counter(
        self,
        container: dict[str, int],
        key: str,
    ) -> None:
        container[key] = int(container.get(key, 0)) + 1

    def _update_frontier_stage_batch_ledger_summary(
        self, row: dict[str, Any]
    ) -> None:
        summary = self._frontier_stage_batch_ledger_summary
        summary["total_rows"] += 1
        cluster_type = str(row["cluster_type"])
        self._increment_frontier_stage_batch_summary_counter(
            summary["cluster_counts"], cluster_type
        )

        if cluster_type != ClusterType.DECODE_FFN.name:
            return

        decode_ffn = summary["decode_ffn"]
        decode_ffn["rows"] += 1
        completion_source = row.get("stage_completion_observed_source")
        if completion_source:
            self._increment_frontier_stage_batch_summary_counter(
                decode_ffn["completion_sources"], str(completion_source)
            )
        else:
            decode_ffn["missing_source_rows"] += 1

        required_metadata_present = self._decode_ffn_summary_required_metadata_present(
            row
        )
        if not required_metadata_present:
            decode_ffn["rows_missing_required_metadata"] += 1
            return

        source_batch_ids = tuple(int(batch_id) for batch_id in row["source_batch_ids"])
        group_key = (
            int(row["replica_id"]),
            int(row["stage_id"]),
            float(row["source_group_ready_ts"]),
            source_batch_ids,
        )
        group = self._frontier_stage_batch_ledger_summary_groups.setdefault(
            group_key,
            {
                "rows": 0,
                "max_source_arrival_ts": float("-inf"),
                "source_group_ready_ts": float(row["source_group_ready_ts"]),
                "stage_start_ts": float(row["stage_start_ts"]),
                "observed_stage_duration_ms": 0.0,
                "model_stage_time_ms": 0.0,
                "source_request_ids": set(),
            },
        )
        group["rows"] += 1
        for arrival_ts in row["source_batch_arrival_times"]:
            group["max_source_arrival_ts"] = max(
                group["max_source_arrival_ts"], float(arrival_ts)
            )
        group["observed_stage_duration_ms"] = max(
            group["observed_stage_duration_ms"],
            float(row.get("observed_stage_duration_ms", 0.0)),
        )
        group["model_stage_time_ms"] = max(
            group["model_stage_time_ms"],
            float(row["execution_time"]["total_time_ms"]),
        )
        group["source_request_ids"].update(
            str(request_id) for request_id in row.get("source_request_ids", [])
        )

    def _decode_ffn_summary_required_metadata_present(
        self, row: dict[str, Any]
    ) -> bool:
        return (
            bool(row.get("source_batch_ids"))
            and bool(row.get("source_batch_arrival_times"))
            and row.get("source_group_ready_ts") is not None
            and row.get("stage_start_ts") is not None
            and row.get("observed_stage_duration_ms") is not None
            and row.get("execution_time", {}).get("total_time_ms") is not None
        )

    def _summarize_frontier_stage_batch_values(
        self, values: list[float]
    ) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "p50": None, "p95": None, "max": None}
        sorted_values = sorted(values)
        return {
            "count": len(sorted_values),
            "p50": self._percentile_from_sorted_values(sorted_values, 50.0),
            "p95": self._percentile_from_sorted_values(sorted_values, 95.0),
            "max": sorted_values[-1],
        }

    def _percentile_from_sorted_values(
        self, sorted_values: list[float], percentile: float
    ) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * percentile / 100.0
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        if lower_index == upper_index:
            return sorted_values[lower_index]
        fraction = position - lower_index
        return (
            sorted_values[lower_index] * (1.0 - fraction)
            + sorted_values[upper_index] * fraction
        )

    def _build_frontier_stage_batch_ledger_summary_output(self) -> dict[str, Any]:
        summary = json.loads(json.dumps(self._frontier_stage_batch_ledger_summary))
        row_counts_per_group: dict[str, int] = {}
        source_request_counts_per_group: dict[str, int] = {}
        grouping_barrier_wait_ms: list[float] = []
        ffn_replica_queue_residence_ms: list[float] = []
        observed_decode_ffn_stage_duration_ms: list[float] = []
        model_decode_ffn_stage_time_ms: list[float] = []
        event_level_residual_ms: list[float] = []
        negative_queue_groups = 0

        for group in self._frontier_stage_batch_ledger_summary_groups.values():
            row_count_key = str(int(group["rows"]))
            row_counts_per_group[row_count_key] = (
                row_counts_per_group.get(row_count_key, 0) + 1
            )
            source_request_count_key = str(len(group["source_request_ids"]))
            source_request_counts_per_group[source_request_count_key] = (
                source_request_counts_per_group.get(source_request_count_key, 0) + 1
            )
            grouping_barrier_wait_ms.append(
                (group["source_group_ready_ts"] - group["max_source_arrival_ts"])
                * 1e3
            )
            queue_residence_ms = (
                group["stage_start_ts"] - group["source_group_ready_ts"]
            ) * 1e3
            ffn_replica_queue_residence_ms.append(queue_residence_ms)
            if queue_residence_ms < -1e-9:
                negative_queue_groups += 1
            observed_decode_ffn_stage_duration_ms.append(
                group["observed_stage_duration_ms"]
            )
            model_decode_ffn_stage_time_ms.append(group["model_stage_time_ms"])
            event_level_residual_ms.append(
                group["observed_stage_duration_ms"] - group["model_stage_time_ms"]
            )

        decode_ffn = summary["decode_ffn"]
        decode_ffn["groups"] = len(self._frontier_stage_batch_ledger_summary_groups)
        decode_ffn["rows_per_group_histogram"] = row_counts_per_group
        decode_ffn["rows_per_group_unique"] = sorted(
            int(row_count) for row_count in row_counts_per_group
        )
        decode_ffn["source_request_count_per_group_histogram"] = (
            source_request_counts_per_group
        )
        decode_ffn["source_request_count_per_group_unique"] = sorted(
            int(request_count) for request_count in source_request_counts_per_group
        )
        decode_ffn["negative_queue_groups"] = negative_queue_groups
        decode_ffn["group_metrics"] = {
            "grouping_barrier_wait_ms": self._summarize_frontier_stage_batch_values(
                grouping_barrier_wait_ms
            ),
            "ffn_replica_queue_residence_ms": self._summarize_frontier_stage_batch_values(
                ffn_replica_queue_residence_ms
            ),
            "observed_decode_ffn_stage_duration_ms": self._summarize_frontier_stage_batch_values(
                observed_decode_ffn_stage_duration_ms
            ),
            "model_decode_ffn_stage_time_ms": self._summarize_frontier_stage_batch_values(
                model_decode_ffn_stage_time_ms
            ),
            "event_level_residual_ms": self._summarize_frontier_stage_batch_values(
                event_level_residual_ms
            ),
        }
        return summary

    def _write_frontier_stage_batch_ledger(self) -> None:
        if (
            not self._frontier_stage_batch_ledger_rows
            and self._frontier_stage_batch_ledger_summary["total_rows"] == 0
        ):
            return
        os.makedirs(self._config.output_dir, exist_ok=True)
        if getattr(self._config, "store_frontier_stage_batch_ledger", True):
            output_path = os.path.join(
                self._config.output_dir, "frontier_stage_batch_ledger.jsonl"
            )
            with open(output_path, "w", encoding="utf-8") as handle:
                for row in self._frontier_stage_batch_ledger_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        if getattr(self._config, "store_frontier_stage_batch_ledger_summary", False):
            summary_output_path = os.path.join(
                self._config.output_dir, "frontier_stage_batch_ledger_summary.json"
            )
            with open(summary_output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    self._build_frontier_stage_batch_ledger_summary_output(),
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")

    def on_kv_cache_transfer_start(
        self,
        time: float,
        source_replica_id: int,
        source_dp_id: int,
        target_cluster_type: ClusterType,
        kv_cache_size_bytes: int,
        transfer_info: Any,
    ) -> None:
        if self._trace_store and self._config.enable_op_level_tracing:
            from frontier.metrics.trace_store import TraceEvent

            batch_id = transfer_info.batch.id if transfer_info.batch else None
            request_ids = (
                [str(req.id) for req in transfer_info.batch.requests]
                if transfer_info.batch and transfer_info.batch.requests
                else []
            )

            cluster_config = self._cluster_configs.get(transfer_info.source_cluster_type)
            if cluster_config is None:
                raise ValueError(
                    f"Cluster config not found for {transfer_info.source_cluster_type}"
                )
            transfer_meta = build_kv_cache_transfer_meta(
                transfer_info.batch,
                cluster_config.replica_config,
                transfer_info.source_cluster_type,
                kv_cache_size_bytes,
            )
            total_tokens = transfer_meta["total_tokens"]
            trace_context = OpTraceContext(
                cluster_type=transfer_info.source_cluster_type,
                model_config=cluster_config.replica_config.model_config,
                replica_config=cluster_config.replica_config,
                total_tokens=total_tokens,
                effective_tokens_compute=total_tokens,
                effective_tokens_transfer=total_tokens,
                effective_tokens_rounded=(total_tokens + 7) // 8 * 8,
                tokens_are_post_routing=False,
            )
            transfer_meta["parallel_context"] = build_parallel_context(trace_context)
            transfer_meta["model_name"] = cluster_config.replica_config.model_name
            transfer_meta["request_ids"] = request_ids
            transfer_meta["source_dp_id"] = source_dp_id

            event = TraceEvent(
                type="TRANSFER",
                name="kv_cache_transfer",
                ts_start=time,
                duration_ms=transfer_info.transfer_time_ms,
                cluster=transfer_info.source_cluster_type.name,
                replica_id=source_replica_id,
                batch_id=batch_id,
                target_cluster=target_cluster_type.name,
                meta=transfer_meta,
            )
            self._trace_store.log_event(event)

        if not self._config.write_metrics:
            return

        self._kv_cache_transfer_metrics["transfer_count"] += 1

    def on_kv_cache_transfer_end(
        self,
        time: float,
        duration: float,
        size_bytes: int,
        target_cluster_type: ClusterType,
        transfer_info: Any,
    ) -> None:
        if not self._config.write_metrics:
            return

        self._kv_cache_transfer_metrics["total_transfer_time"] += duration
        self._kv_cache_transfer_metrics["total_data_transferred"] += size_bytes

        request_info = ""
        if transfer_info.batch and transfer_info.batch.requests:
            request_ids = [str(req.id) for req in transfer_info.batch.requests]
            request_info = f"_req_{'_'.join(request_ids)}"

        transfer_id = (
            f"transfer_{self._kv_cache_transfer_metrics['transfer_count']}"
            f"{request_info}"
        )
        self._kv_cache_transfer_metrics["transfer_times"].put(transfer_id, duration)
        self._kv_cache_transfer_metrics["transfer_sizes"].put(transfer_id, size_bytes)

    def on_m2n_transfer_start(
        self,
        time: float,
        source_replica_id: int,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        activation_size_bytes: int,
        transfer_info: Any,
    ) -> None:
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def on_m2n_transfer_end(
        self,
        time: float,
        duration: float,
        size_bytes: int,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        transfer_info: Any,
    ) -> None:
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
