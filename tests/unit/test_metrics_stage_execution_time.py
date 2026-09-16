"""Regression coverage for StageExecutionTime-aware operation metrics."""

from __future__ import annotations

from types import SimpleNamespace

from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.metrics.op_trace_utils import OpTraceContext
from frontier.types import ClusterType


class _Series:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def put(self, *args) -> None:
        self.calls.append(args)


def _layer(
    layer_id: int,
    family_id: str,
    op_times: dict[str, float],
    pipeline_parallel_communication_time: float = 0.0,
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=pipeline_parallel_communication_time,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        global_layer_id=layer_id,
        attention_family_id=family_id,
        attention_variant_id="standard" if family_id == "dense_attention" else "qwen3_5",
        op_times=op_times,
    )


def _store() -> tuple[MetricsStore, dict[OperationMetrics, _Series]]:
    store = MetricsStore.__new__(MetricsStore)
    store._config = SimpleNamespace(
        write_metrics=True,
        store_utilization_metrics=True,
        store_operation_metrics=True,
        enable_op_level_tracing=False,
        enable_per_layer_expansion=False,
        num_requests_to_trace_per_layer=0,
        enable_metrics_ground_truth_trace=False,
    )
    store._trace_store = None
    store._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(num_replicas=1)}
    store._replica_busy_time = {ClusterType.MONOLITHIC: [[[_Series()]]]}
    store._replica_mfu = {ClusterType.MONOLITHIC: [[[_Series()]]]}
    store._replica_full_stage_busy_time = {ClusterType.MONOLITHIC: [[_Series()]]}
    store._replica_full_stage_mfu = {ClusterType.MONOLITHIC: [[_Series()]]}
    store._mfu_calculator = {
        ClusterType.MONOLITHIC: SimpleNamespace(get_mfu=lambda _stage: 0.0)
    }
    operation = {metric: _Series() for metric in OperationMetrics}
    per_batch = {metric: _Series() for metric in OperationMetrics}
    store._operation_metrics = {ClusterType.MONOLITHIC: operation}
    store._operation_metrics_per_batch = {ClusterType.MONOLITHIC: per_batch}
    store._pending_frontier_stage_batch_ledger_rows = {}
    store._pending_frontier_stage_batch_ledger_row_keys = {}
    store._pending_frontier_stage_batch_ledger_rows_by_key = {}
    store._should_capture_frontier_stage_batch_ledger = lambda: False
    store._emit_op_level_traces = lambda **_kwargs: None
    return store, per_batch


def test_stage_operation_metrics_emit_each_layer_once_with_its_attention_family() -> None:
    dense = _layer(
        0,
        "dense_attention",
        {"attn_kv_cache_save": 2.0, "attn_prefill": 3.0, "attn_decode": 5.0},
    )
    gdn = _layer(
        1,
        "gated_delta_net",
        {
            "gdn_input_projections": 11.0,
            "gdn_core_prefill": 13.0,
            "gdn_core_decode": 17.0,
            "gdn_output_projection": 19.0,
        },
    )
    stage = StageExecutionTime((dense, gdn), stage_execution_time=dense)
    store, per_batch = _store()

    store.on_replica_stage_schedule(
        time=0.0,
        replica_id=0,
        stage_id=0,
        batch_stage=SimpleNamespace(
            _batch_id=7,
            request_ids=[],
            num_tokens=[1],
            execution_time=stage.total_time,
        ),
        execution_time=stage,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert [call[1] for call in per_batch[OperationMetrics.ATTN_PREFILL].calls] == [3.0]
    assert [call[1] for call in per_batch[OperationMetrics.ATTN_KV_CACHE_SAVE].calls] == [2.0]
    assert [call[1] for call in per_batch[OperationMetrics.GDN_INPUT_PROJECTIONS].calls] == [11.0]
    assert [call[1] for call in per_batch[OperationMetrics.GDN_CORE_PREFILL].calls] == [13.0]
    assert [call[1] for call in per_batch[OperationMetrics.GDN_CORE_DECODE].calls] == [17.0]
    assert [call[1] for call in per_batch[OperationMetrics.GDN_OUTPUT_PROJECTION].calls] == [19.0]
    assert not per_batch[OperationMetrics.ATTN_PREFILL].calls[0][1] == 16.0


def test_stage_trace_helper_keeps_dense_and_gdn_names_on_their_layers() -> None:
    dense = _layer(
        0,
        "dense_attention",
        {"attn_kv_cache_save": 2.0, "attn_prefill": 3.0, "attn_decode": 5.0},
    )
    gdn = _layer(
        1,
        "gated_delta_net",
        {
            "gdn_input_projections": 11.0,
            "gdn_core_prefill": 13.0,
            "gdn_core_decode": 17.0,
            "gdn_output_projection": 19.0,
        },
    )
    stage = StageExecutionTime((dense, gdn), stage_execution_time=dense)
    events: list[tuple[str, str, float, int, dict[str, object]]] = []

    MetricsStore.__new__(MetricsStore)._emit_stage_layer_traces(
        lambda kind, name, duration, layer_idx, metadata: events.append(
            (kind, name, duration, layer_idx, metadata)
        ),
        stage,
        ClusterType.MONOLITHIC,
        moe_tp_enabled=False,
        ep_enabled=False,
    )

    assert {
        name
        for _kind, name, _duration, layer_idx, _metadata in events
        if layer_idx == 0
    } >= {"attn_kv_cache_save", "attn_prefill", "attn_decode"}
    assert {
        name
        for _kind, name, _duration, layer_idx, _metadata in events
        if layer_idx == 1
    } >= {
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_core_decode",
        "gdn_output_projection",
    }
    assert not {
        name
        for _kind, name, _duration, layer_idx, _metadata in events
        if layer_idx == 1
    }.intersection(
        {"attn_kv_cache_save", "attn_prefill", "attn_decode", "attn_pre_proj", "attn_rope", "attn_post_proj"}
    )


class _TraceStore:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event) -> None:
        self.events.append(event)


def test_stage_op_level_traces_include_gdn_metadata_without_duplication() -> None:
    dense = _layer(
        0,
        "dense_attention",
        {"attn_kv_cache_save": 2.0, "attn_prefill": 3.0, "attn_decode": 5.0},
    )
    gdn = _layer(
        1,
        "gated_delta_net",
        {
            "gdn_input_projections": 11.0,
            "gdn_core_prefill": 13.0,
            "gdn_core_decode": 17.0,
            "gdn_output_projection": 19.0,
        },
    )
    stage = StageExecutionTime((dense, gdn), stage_execution_time=dense)
    store, _per_batch = _store()
    del store._emit_op_level_traces
    store._config.enable_op_level_tracing = True
    store._trace_store = _TraceStore()
    store._per_layer_traced_requests_by_cluster = {}
    store._cluster_configs = {
        ClusterType.MONOLITHIC: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=None,
                attn_tensor_parallel_size=2,
                attn_dp=1,
                moe_tensor_parallel_size=1,
                moe_expert_parallel_size=1,
                num_pipeline_stages=1,
                model_name="synthetic",
            )
        )
    }
    model_config = SimpleNamespace(
        embedding_dim=16,
        num_q_heads=4,
        num_kv_heads=2,
        mlp_hidden_dim=32,
        num_experts=0,
        num_experts_per_tok=0,
        is_moe=False,
        model_type="synthetic",
        get_head_dim=lambda: 4,
        uses_mla=lambda: False,
    )
    replica_config = store._cluster_configs[ClusterType.MONOLITHIC].replica_config
    store._build_op_trace_context = lambda _batch, _cluster: OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=1,
        effective_tokens_compute=1,
        effective_tokens_transfer=1,
        effective_tokens_rounded=1,
        tokens_are_post_routing=False,
    )

    store._emit_op_level_traces(
        time=0.0,
        batch_stage=SimpleNamespace(
            _batch_id=7,
            request_ids=[],
            num_tokens=[1],
            effective_total_tokens_compute=1,
            effective_total_tokens_transfer=1,
            effective_total_tokens_rounded=1,
            tokens_are_post_routing=False,
        ),
        replica_id=0,
        execution_time=stage,
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=[],
    )

    events = store._trace_store.events
    gdn_events = [event for event in events if event.name.startswith("gdn_")]
    assert [event.name for event in gdn_events] == [
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_core_decode",
        "gdn_output_projection",
    ]
    assert [event.layer_id for event in gdn_events] == [-1, -1, -1, -1]
    assert [event.meta["global_layer_ids"] for event in gdn_events] == [[1]] * 4
    assert [event.duration_ms for event in gdn_events] == [11.0, 13.0, 17.0, 19.0]
    assert {event.meta["attention_family_id"] for event in gdn_events} == {
        "gated_delta_net"
    }


def test_stage_component_ledger_preserves_family_names_and_owner_once() -> None:
    dense = _layer(
        0,
        "dense_attention",
        {
            "attn_kv_cache_save": 2.0,
            "attn_prefill": 3.0,
            "attn_decode": 5.0,
        },
    )
    gdn = _layer(
        1,
        "gated_delta_net",
        {
            "gdn_input_projections": 11.0,
            "gdn_core_prefill": 13.0,
            "gdn_core_decode": 17.0,
            "gdn_output_projection": 19.0,
        },
    )
    owner = _layer(
        0,
        "dense_attention",
        {
            "attn_kv_cache_save": 2.0,
            "attn_prefill": 3.0,
            "attn_decode": 5.0,
        },
        pipeline_parallel_communication_time=23.0,
    )
    stage = StageExecutionTime((owner, gdn), stage_execution_time=owner)

    ledger = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(
        stage
    )

    assert ledger["gdn_core_prefill"] == 13.0
    assert ledger["gdn_core_decode"] == 17.0
    assert ledger["attention_prefill_execution_time"] == 3.0
    assert ledger["pipeline_parallel_communication_time"] == 23.0
    assert sum(ledger.values()) == stage.total_time * 1e3
