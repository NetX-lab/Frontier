from __future__ import annotations

import pytest

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import (
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
)
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import AttentionOperatorTimes
from frontier.metrics.constants import OperationMetrics as MetricsOperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.profiling.common.constants import (
    OperationMetrics as ProfilingOperationMetrics,
)
from frontier.types import ClusterType


MLA_OPS = (
    "attn_mla_kv_cache_save",
    "attn_mla_prefill_kv_up_proj",
    "attn_mla_prefill",
    "attn_mla_decode_q_latent_proj",
    "attn_mla_decode",
    "attn_mla_v_up_proj",
)


def _fake_dense_attention_operator(
    name: str,
    role: AttentionOperatorRole,
) -> AttentionOperatorSpec:
    execution_time_attr_by_role = {
        AttentionOperatorRole.CACHE_WRITE: "attention_kv_cache_save_execution_time",
        AttentionOperatorRole.PREFILL_KERNEL: "attention_prefill_execution_time",
        AttentionOperatorRole.DECODE_KERNEL: "attention_decode_execution_time",
    }
    return AttentionOperatorSpec(
        name=name,
        role=role,
        phases=(
            AttentionPhase.PREFILL,
            AttentionPhase.DECODE,
            AttentionPhase.MIXED,
        ),
        execution_time_attr=execution_time_attr_by_role[role],
    )


class _DummySeries:
    def __init__(self) -> None:
        self.calls = []

    def put(self, *args) -> None:
        self.calls.append(args)


class _DummyMFU:
    def get_mfu(self, _batch_stage) -> float:
        return 0.0


class _DummyConfig:
    write_metrics = True
    store_utilization_metrics = True
    store_operation_metrics = True
    enable_op_level_tracing = False
    enable_per_layer_expansion = False
    num_requests_to_trace_per_layer = 0
    store_frontier_stage_batch_ledger = True
    store_frontier_stage_batch_ledger_summary = False


class _DummyReplicaConfig:
    moe_tensor_parallel_size = 1
    moe_expert_parallel_size = 1
    attn_tensor_parallel_size = 2
    attn_data_parallel_size = 1
    num_pipeline_stages = 1
    router_topk = 1
    model_config = None
    model_name = "dummy-model"


class _DummyClusterConfig:
    num_replicas = 1
    replica_config = _DummyReplicaConfig()


class _DummyBatchStage:
    def __init__(self) -> None:
        self._batch_id = 17
        self.request_ids = [101]
        self.num_tokens = [1]
        self.scheduled_at = 0.0
        self.execution_time = 0.0
        self.tokens_are_post_routing = False
        self.effective_total_tokens_compute = 8
        self.effective_total_tokens_transfer = 8
        self.effective_total_tokens_rounded = 8


def _build_metrics_store() -> MetricsStore:
    metrics_store = MetricsStore.__new__(MetricsStore)
    metrics_store._config = _DummyConfig()
    metrics_store._trace_store = None
    metrics_store._cluster_configs = {ClusterType.MONOLITHIC: _DummyClusterConfig()}
    metrics_store._mfu_calculator = {ClusterType.MONOLITHIC: _DummyMFU()}
    metrics_store._operation_metrics = {
        ClusterType.MONOLITHIC: {
            metric: _DummySeries() for metric in MetricsOperationMetrics
        }
    }
    metrics_store._operation_metrics_per_batch = {
        ClusterType.MONOLITHIC: {
            metric: _DummySeries() for metric in MetricsOperationMetrics
        }
    }
    metrics_store._pending_frontier_stage_batch_ledger_rows = {}
    metrics_store._pending_frontier_stage_batch_ledger_row_keys = {}
    metrics_store._pending_frontier_stage_batch_ledger_rows_by_key = {}
    metrics_store._frontier_stage_batch_ledger_rows = []

    dummy_series = _DummySeries()
    metrics_store._replica_busy_time = {ClusterType.MONOLITHIC: [[[dummy_series]]]}
    metrics_store._replica_mfu = {ClusterType.MONOLITHIC: [[[dummy_series]]]}
    return metrics_store


def _build_execution_time(
    num_layers: int = 2,
    dense_kernel_times: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=0.5,
        attention_kv_cache_save_execution_time=dense_kernel_times[0],
        attention_decode_execution_time=dense_kernel_times[2],
        attention_prefill_execution_time=dense_kernel_times[1],
        attention_layer_pre_proj_execution_time=0.7,
        attention_layer_post_proj_execution_time=0.8,
        attn_norm_time=0.4,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        attn_mla_kv_cache_save_time=0.11,
        attn_mla_prefill_kv_up_proj_time=0.12,
        attn_mla_prefill_time=0.13,
        attn_mla_decode_q_latent_proj_time=0.14,
        attn_mla_decode_time=0.15,
        attn_mla_v_up_proj_time=0.16,
    )


def _build_structured_mla_execution_time(num_layers: int = 2) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
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
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        attention_operator_times=AttentionOperatorTimes(
            {
                "attn_mla_kv_cache_save": 0.01,
                "attn_mla_prefill_kv_up_proj": 0.02,
                "attn_mla_prefill": 0.03,
                "attn_mla_decode_q_latent_proj": 0.04,
                "attn_mla_decode": 0.05,
                "attn_mla_v_up_proj": 0.06,
            }
        ),
    )


class _DummyTraceStore:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event) -> None:
        self.events.append(event)


class _DummyMlaModelConfig:
    embedding_dim = 16
    num_q_heads = 4
    num_kv_heads = 1
    mlp_hidden_dim = 32
    num_experts = 0
    num_experts_per_tok = 0
    is_moe = False
    model_type = "generic"
    use_mla = True
    kv_lora_rank = 6
    qk_nope_head_dim = 3
    qk_rope_head_dim = 2
    qk_head_dim = 5
    v_head_dim = 4

    def is_step3_text(self) -> bool:
        return False

    def get_head_dim(self) -> int:
        return 4

    def uses_mla(self) -> bool:
        return True

    def get_runtime_num_kv_heads(self) -> int:
        return 1

    def get_runtime_head_size(self) -> int:
        return 8

    def get_qk_head_dim(self) -> int:
        return 5


class _DummyMlaReplicaConfig(_DummyReplicaConfig):
    model_config = _DummyMlaModelConfig()
    model_name = "dummy-mla-model"


class _DummyMlaClusterConfig:
    num_replicas = 1
    replica_config = _DummyMlaReplicaConfig()


def _collect_emitted_ops():
    emitted: list[tuple[str, float, int | None]] = []

    def _emit(
        _op_type: str,
        op_name: str,
        duration_ms: float,
        layer_idx: int | None = None,
        *_args,
        **_kwargs,
    ) -> None:
        if duration_ms > 0.0:
            emitted.append((op_name, duration_ms, layer_idx))

    return emitted, _emit


def test_mla_operator_names_are_registered_for_metrics_and_profiling() -> None:
    assert {metric.value for metric in MetricsOperationMetrics}.issuperset(MLA_OPS)
    assert {metric.value for metric in ProfilingOperationMetrics}.issuperset(MLA_OPS)


def test_execution_time_carries_and_aggregates_mla_attention_ops() -> None:
    execution_time = _build_execution_time(num_layers=2)

    assert execution_time.attn_mla_kv_cache_save_time == pytest.approx(0.22)
    assert execution_time.attn_mla_prefill_kv_up_proj_time == pytest.approx(0.24)
    assert execution_time.attn_mla_prefill_time == pytest.approx(0.26)
    assert execution_time.attn_mla_decode_q_latent_proj_time == pytest.approx(0.28)
    assert execution_time.attn_mla_decode_time == pytest.approx(0.30)
    assert execution_time.attn_mla_v_up_proj_time == pytest.approx(0.32)

    single_layer_attention = (
        0.5
        + 0.7
        + 0.8
        + 0.4
        + 0.11
        + 0.12
        + 0.13
        + 0.14
        + 0.15
        + 0.16
    )
    assert execution_time.get_single_layer_attention_time() == pytest.approx(
        single_layer_attention
    )
    assert execution_time.attention_time == pytest.approx(single_layer_attention * 2)


def test_metrics_store_emits_aggregated_mla_attention_ops() -> None:
    execution_time = _build_execution_time(num_layers=2)
    emitted, emit = _collect_emitted_ops()

    MetricsStore.__new__(MetricsStore)._emit_aggregated_traces(
        emit,
        execution_time,
        moe_tp_enabled=False,
        ep_enabled=False,
        cluster_type=ClusterType.MONOLITHIC,
    )

    durations_by_name = {name: duration for name, duration, _ in emitted}

    assert [name for name in durations_by_name if name in MLA_OPS] == list(MLA_OPS)
    assert durations_by_name["attn_mla_kv_cache_save"] == pytest.approx(0.22)
    assert durations_by_name["attn_mla_prefill_kv_up_proj"] == pytest.approx(0.24)
    assert durations_by_name["attn_mla_prefill"] == pytest.approx(0.26)
    assert durations_by_name["attn_mla_decode_q_latent_proj"] == pytest.approx(0.28)
    assert durations_by_name["attn_mla_decode"] == pytest.approx(0.30)
    assert durations_by_name["attn_mla_v_up_proj"] == pytest.approx(0.32)


def test_metrics_store_emits_per_layer_mla_attention_ops() -> None:
    execution_time = _build_execution_time(num_layers=2)
    emitted, emit = _collect_emitted_ops()

    MetricsStore.__new__(MetricsStore)._emit_per_layer_traces(
        emit,
        execution_time,
        num_layers=2,
        base_meta={},
        moe_tp_enabled=False,
        ep_enabled=False,
        cluster_type=ClusterType.MONOLITHIC,
    )

    mla_rows = [row for row in emitted if row[0] in MLA_OPS]

    assert [row[0] for row in mla_rows[:6]] == list(MLA_OPS)
    assert [row[0] for row in mla_rows[6:]] == list(MLA_OPS)
    assert [row[2] for row in mla_rows[:6]] == [0] * 6
    assert [row[2] for row in mla_rows[6:]] == [1] * 6
    assert [row[1] for row in mla_rows[:6]] == pytest.approx(
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    )


def test_operation_metrics_record_mla_attention_ops() -> None:
    metrics_store = _build_metrics_store()
    execution_time = _build_execution_time(num_layers=1)

    metrics_store.on_replica_stage_schedule(
        time=0.0,
        replica_id=0,
        stage_id=0,
        batch_stage=_DummyBatchStage(),
        execution_time=execution_time,
        cluster_type=ClusterType.MONOLITHIC,
        dp_id=0,
    )

    per_batch = metrics_store._operation_metrics_per_batch[ClusterType.MONOLITHIC]

    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_KV_CACHE_SAVE
    ].calls[0][1] == pytest.approx(0.11)
    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_PREFILL_KV_UP_PROJ
    ].calls[0][1] == pytest.approx(0.12)
    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_PREFILL
    ].calls[0][1] == pytest.approx(0.13)
    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_DECODE_Q_LATENT_PROJ
    ].calls[0][1] == pytest.approx(0.14)
    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_DECODE
    ].calls[0][1] == pytest.approx(0.15)
    assert per_batch[
        MetricsOperationMetrics.ATTN_MLA_V_UP_PROJ
    ].calls[0][1] == pytest.approx(0.16)


def test_operation_metrics_dense_attention_uses_shared_mapper(monkeypatch) -> None:
    metrics_store = _build_metrics_store()
    metrics_store._config.store_frontier_stage_batch_ledger = False
    metrics_store._config.store_frontier_stage_batch_ledger_summary = False
    execution_time = _build_execution_time(
        num_layers=1,
        dense_kernel_times=(0.2, 0.4, 0.3),
    )
    mapper_calls: list[tuple[str, int | None, bool]] = []

    def _fake_get_attention_trace_op_times(
        mapped_execution_time,
        family,
        *,
        per_layer_count=None,
        skip_zero=True,
    ):
        assert mapped_execution_time is execution_time
        if family is not DENSE_ATTENTION_FAMILY:
            return ()
        mapper_calls.append((family.family_id, per_layer_count, skip_zero))
        return (
            (
                _fake_dense_attention_operator(
                    "role_cache",
                    AttentionOperatorRole.CACHE_WRITE,
                ),
                0.35,
            ),
            (
                _fake_dense_attention_operator(
                    "role_prefill",
                    AttentionOperatorRole.PREFILL_KERNEL,
                ),
                0.25,
            ),
            (
                _fake_dense_attention_operator(
                    "role_decode",
                    AttentionOperatorRole.DECODE_KERNEL,
                ),
                0.30,
            ),
        )

    monkeypatch.setattr(
        "frontier.metrics.metrics_store.get_attention_trace_op_times",
        _fake_get_attention_trace_op_times,
    )

    metrics_store.on_replica_stage_schedule(
        time=0.0,
        replica_id=0,
        stage_id=0,
        batch_stage=_DummyBatchStage(),
        execution_time=execution_time,
        cluster_type=ClusterType.MONOLITHIC,
        dp_id=0,
    )

    per_batch = metrics_store._operation_metrics_per_batch[ClusterType.MONOLITHIC]

    assert mapper_calls == [("dense_attention", None, False)]
    assert per_batch[MetricsOperationMetrics.ATTN_PREFILL].calls[0][1] == pytest.approx(
        0.25
    )
    assert per_batch[MetricsOperationMetrics.ATTN_DECODE].calls[0][1] == pytest.approx(
        0.30
    )
    assert per_batch[
        MetricsOperationMetrics.ATTN_KV_CACHE_SAVE
    ].calls[0][1] == pytest.approx(0.35)


def test_frontier_stage_batch_component_ledger_records_mla_attention_ops() -> None:
    execution_time = _build_execution_time(num_layers=1)

    ledger = MetricsStore.__new__(
        MetricsStore
    )._build_frontier_stage_batch_component_ledger(execution_time)

    assert ledger["attn_mla_kv_cache_save_time"] == pytest.approx(0.11)
    assert ledger["attn_mla_prefill_kv_up_proj_time"] == pytest.approx(0.12)
    assert ledger["attn_mla_prefill_time"] == pytest.approx(0.13)
    assert ledger["attn_mla_decode_q_latent_proj_time"] == pytest.approx(0.14)
    assert ledger["attn_mla_decode_time"] == pytest.approx(0.15)
    assert ledger["attn_mla_v_up_proj_time"] == pytest.approx(0.16)


def test_frontier_stage_batch_ledger_uses_structured_mla_operator_times() -> None:
    execution_time = _build_structured_mla_execution_time(num_layers=2)
    batch_stage = _DummyBatchStage()

    row = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_ledger_row(
        batch_stage=batch_stage,
        execution_time=execution_time,
        replica_id=0,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        dp_id=0,
        stage_end_time=1.0,
    )

    component_ledger = row["execution_time"]["component_ledger_ms"]
    assert component_ledger["attn_mla_kv_cache_save_time"] == pytest.approx(0.02)
    assert component_ledger["attn_mla_prefill_kv_up_proj_time"] == pytest.approx(0.04)
    assert component_ledger["attn_mla_prefill_time"] == pytest.approx(0.06)
    assert component_ledger["attn_mla_decode_q_latent_proj_time"] == pytest.approx(0.08)
    assert component_ledger["attn_mla_decode_time"] == pytest.approx(0.10)
    assert component_ledger["attn_mla_v_up_proj_time"] == pytest.approx(0.12)
    assert sum(component_ledger.values()) == pytest.approx(0.42)
    assert row["execution_time"]["total_time_ms"] == pytest.approx(0.42)


def test_op_level_tracing_generates_metadata_for_structured_mla_ops() -> None:
    metrics_store = _build_metrics_store()
    metrics_store._config.enable_op_level_tracing = True
    metrics_store._trace_store = _DummyTraceStore()
    metrics_store._cluster_configs = {ClusterType.MONOLITHIC: _DummyMlaClusterConfig()}
    metrics_store._per_layer_traced_requests_by_cluster = {}

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_DummyBatchStage(),
        replica_id=0,
        execution_time=_build_structured_mla_execution_time(num_layers=1),
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=["101"],
    )

    mla_events = [
        event
        for event in metrics_store._trace_store.events
        if event.name in MLA_OPS
    ]

    assert [event.name for event in mla_events] == list(MLA_OPS)
    for event in mla_events:
        assert event.type == "COMPUTE"
        assert event.meta["precision_op"] == event.name
        assert event.meta["dtype"] == "FP16"
        assert event.meta["dtype_bytes"] == 2
        assert event.meta["tensor_shape"]
        assert event.meta["tensor_size_bytes"]


def test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper(
    monkeypatch,
) -> None:
    execution_time = _build_execution_time(
        num_layers=1,
        dense_kernel_times=(0.2, 0.4, 0.3),
    )
    mapper_calls: list[tuple[str, int | None, bool]] = []

    def _fake_get_attention_trace_op_times(
        mapped_execution_time,
        family,
        *,
        per_layer_count=None,
        skip_zero=True,
    ):
        assert mapped_execution_time is execution_time
        if family is not DENSE_ATTENTION_FAMILY:
            mapper_calls.append((family.family_id, per_layer_count, skip_zero))
            return tuple((operator, 0.0) for operator in family.e2e_trace_ops())
        mapper_calls.append((family.family_id, per_layer_count, skip_zero))
        return (
            (
                _fake_dense_attention_operator(
                    "role_cache",
                    AttentionOperatorRole.CACHE_WRITE,
                ),
                9.2,
            ),
            (
                _fake_dense_attention_operator(
                    "role_prefill",
                    AttentionOperatorRole.PREFILL_KERNEL,
                ),
                9.4,
            ),
            (
                _fake_dense_attention_operator(
                    "role_decode",
                    AttentionOperatorRole.DECODE_KERNEL,
                ),
                9.3,
            ),
        )

    monkeypatch.setattr(
        "frontier.metrics.metrics_store.get_attention_trace_op_times",
        _fake_get_attention_trace_op_times,
    )

    ledger = MetricsStore.__new__(
        MetricsStore
    )._build_frontier_stage_batch_component_ledger(execution_time)

    assert mapper_calls == [
        ("dense_attention", None, False),
        ("latent_mla_attention", None, False),
    ]
    assert ledger["attention_prefill_execution_time"] == pytest.approx(9.4)
    assert ledger["attention_decode_execution_time"] == pytest.approx(9.3)
    assert ledger["attention_kv_cache_save_execution_time"] == pytest.approx(9.2)


def test_dense_attention_trace_does_not_emit_mla_ops_without_mla_timings() -> None:
    execution_time = ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.5,
        attention_kv_cache_save_execution_time=0.2,
        attention_decode_execution_time=0.3,
        attention_prefill_execution_time=0.4,
        attention_layer_pre_proj_execution_time=0.7,
        attention_layer_post_proj_execution_time=0.8,
        attn_norm_time=0.4,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
    )
    emitted, emit = _collect_emitted_ops()

    MetricsStore.__new__(MetricsStore)._emit_aggregated_traces(
        emit,
        execution_time,
        moe_tp_enabled=False,
        ep_enabled=False,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    emitted_names = [name for name, _, _ in emitted]

    assert not any(name in MLA_OPS for name in emitted_names)
    assert {"attn_kv_cache_save", "attn_prefill", "attn_decode"}.issubset(
        emitted_names
    )


def test_metrics_store_dense_attention_trace_uses_shared_mapper(monkeypatch) -> None:
    execution_time = _build_execution_time(
        num_layers=2,
        dense_kernel_times=(0.2, 0.4, 0.3),
    )
    emitted, emit = _collect_emitted_ops()
    mapper_calls: list[tuple[str, int | None]] = []

    def _fake_get_attention_trace_op_times(
        mapped_execution_time,
        family,
        *,
        per_layer_count=None,
        skip_zero=True,
    ):
        assert mapped_execution_time is execution_time
        if family is not DENSE_ATTENTION_FAMILY:
            return ()
        assert skip_zero is True
        mapper_calls.append((family.family_id, per_layer_count))
        return (
            (
                _fake_dense_attention_operator(
                    "role_cache",
                    AttentionOperatorRole.CACHE_WRITE,
                ),
                9.2,
            ),
            (
                _fake_dense_attention_operator(
                    "role_prefill",
                    AttentionOperatorRole.PREFILL_KERNEL,
                ),
                9.4,
            ),
            (
                _fake_dense_attention_operator(
                    "role_decode",
                    AttentionOperatorRole.DECODE_KERNEL,
                ),
                9.3,
            ),
        )

    monkeypatch.setattr(
        "frontier.metrics.metrics_store.get_attention_trace_op_times",
        _fake_get_attention_trace_op_times,
    )

    MetricsStore.__new__(MetricsStore)._emit_aggregated_traces(
        emit,
        execution_time,
        moe_tp_enabled=False,
        ep_enabled=False,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    dense_rows = [
        (name, duration_ms)
        for name, duration_ms, _ in emitted
        if name in {"attn_prefill", "attn_decode", "attn_kv_cache_save"}
    ]

    assert mapper_calls == [("dense_attention", None)]
    assert dense_rows == [
        ("attn_prefill", pytest.approx(9.4)),
        ("attn_decode", pytest.approx(9.3)),
        ("attn_kv_cache_save", pytest.approx(9.2)),
    ]


def test_metrics_store_per_layer_dense_attention_trace_uses_shared_mapper(
    monkeypatch,
) -> None:
    execution_time = _build_execution_time(
        num_layers=2,
        dense_kernel_times=(0.2, 0.4, 0.3),
    )
    emitted, emit = _collect_emitted_ops()
    mapper_calls: list[tuple[str, int | None]] = []

    def _fake_get_attention_trace_op_times(
        mapped_execution_time,
        family,
        *,
        per_layer_count=None,
        skip_zero=True,
    ):
        assert mapped_execution_time is execution_time
        if family is not DENSE_ATTENTION_FAMILY:
            return ()
        assert skip_zero is True
        mapper_calls.append((family.family_id, per_layer_count))
        return (
            (
                _fake_dense_attention_operator(
                    "role_cache",
                    AttentionOperatorRole.CACHE_WRITE,
                ),
                9.2,
            ),
            (
                _fake_dense_attention_operator(
                    "role_prefill",
                    AttentionOperatorRole.PREFILL_KERNEL,
                ),
                9.4,
            ),
            (
                _fake_dense_attention_operator(
                    "role_decode",
                    AttentionOperatorRole.DECODE_KERNEL,
                ),
                9.3,
            ),
        )

    monkeypatch.setattr(
        "frontier.metrics.metrics_store.get_attention_trace_op_times",
        _fake_get_attention_trace_op_times,
    )

    MetricsStore.__new__(MetricsStore)._emit_per_layer_traces(
        emit,
        execution_time,
        num_layers=2,
        base_meta={},
        moe_tp_enabled=False,
        ep_enabled=False,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    dense_rows = [
        (name, duration_ms, layer_id)
        for name, duration_ms, layer_id in emitted
        if name in {"attn_prefill", "attn_decode", "attn_kv_cache_save"}
    ]

    assert mapper_calls == [("dense_attention", 2)]
    assert dense_rows == [
        ("attn_prefill", pytest.approx(9.4), 0),
        ("attn_decode", pytest.approx(9.3), 0),
        ("attn_kv_cache_save", pytest.approx(9.2), 0),
        ("attn_prefill", pytest.approx(9.4), 1),
        ("attn_decode", pytest.approx(9.3), 1),
        ("attn_kv_cache_save", pytest.approx(9.2), 1),
    ]
