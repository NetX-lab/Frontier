"""Regression tests for profile-owned mixed-layer MetricsStore traces."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import CommunicationOperatorTimes
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType


class _TraceStore:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event) -> None:
        self.events.append(event)


class _MetricsConfig:
    enable_op_level_tracing = True
    enable_per_layer_expansion = True
    num_requests_to_trace_per_layer = 1


def _build_step3_execution_time(
    *,
    num_layers: int = 61,
    layer_ids: list[int] | tuple[int, ...] | None = None,
    attention_time: float = 0.0,
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=attention_time,
        attention_kv_cache_save_execution_time=attention_time,
        attention_decode_execution_time=attention_time,
        attention_prefill_execution_time=attention_time,
        attention_layer_pre_proj_execution_time=attention_time,
        attention_layer_post_proj_execution_time=attention_time,
        attn_norm_time=attention_time,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=2.0,
        moe_gating_time=2.0,
        moe_shuffling_time=1.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=True,
        moe_grouped_gemm_time=3.0,
        moe_gating_linear_time=1.0,
        moe_gating_routing_topk_time=1.0,
        share_expert_up_proj_time=1.0,
        share_expert_down_proj_time=1.0,
        share_expert_act_time=1.0,
        tensor_parallel_allgather_time=0.0,
        share_expert_tensor_parallel_allreduce_time=1.0,
        attn_tensor_parallel_allreduce_time=0.0,
        moe_tensor_parallel_allreduce_time=1.0,
        communication_operator_times=CommunicationOperatorTimes(
            {
                "expert_parallel_alltoall_dispatch": 1.0,
                "expert_parallel_alltoall_combine": 1.0,
            }
        ),
        layer_ids=layer_ids,
    )


def _build_dense_execution_time(
    num_layers: int = 32,
    *,
    layer_ids: list[int] | tuple[int, ...] | None = None,
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=0.1,
        attention_kv_cache_save_execution_time=0.1,
        attention_decode_execution_time=0.1,
        attention_prefill_execution_time=0.1,
        attention_layer_pre_proj_execution_time=0.1,
        attention_layer_post_proj_execution_time=0.1,
        attn_norm_time=0.1,
        mlp_norm_time=0.1,
        add_time=0.1,
        tensor_parallel_communication_time=0.1,
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
        mlp_layer_up_proj_execution_time=1.0,
        mlp_layer_down_proj_execution_time=1.0,
        mlp_layer_act_execution_time=1.0,
        attn_tensor_parallel_allreduce_time=0.1,
        moe_tensor_parallel_allreduce_time=0.0,
        layer_ids=layer_ids,
    )


def _build_pure_moe_execution_time(num_layers: int = 32) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=0.1,
        attention_kv_cache_save_execution_time=0.1,
        attention_decode_execution_time=0.1,
        attention_prefill_execution_time=0.1,
        attention_layer_pre_proj_execution_time=0.1,
        attention_layer_post_proj_execution_time=0.1,
        attn_norm_time=0.1,
        mlp_norm_time=0.1,
        add_time=0.1,
        tensor_parallel_communication_time=0.1,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.1,
        moe_gating_time=0.1,
        moe_shuffling_time=0.1,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=True,
        moe_grouped_gemm_time=1.0,
        moe_gating_linear_time=0.1,
        moe_gating_routing_topk_time=0.1,
        tensor_parallel_allgather_time=0.1,
        moe_tensor_parallel_allreduce_time=0.1,
        communication_operator_times=CommunicationOperatorTimes(
            {
                "expert_parallel_alltoall_dispatch": 0.1,
                "expert_parallel_alltoall_combine": 0.1,
            }
        ),
        layer_ids=None,
    )


def _build_metrics_store(
    model_config: BaseModelConfig,
    *,
    expand_layers: bool = True,
) -> tuple[MetricsStore, _TraceStore]:
    replica_config = SimpleNamespace(
        model_config=model_config,
        model_name="step3-moe-noquant",
        attn_tensor_parallel_size=8,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
        router_topk=3,
    )
    cluster_config = SimpleNamespace(replica_config=replica_config)
    trace_store = _TraceStore()
    metrics_store = MetricsStore.__new__(MetricsStore)
    metrics_store._config = type(
        "_TestMetricsConfig",
        (_MetricsConfig,),
        {"enable_per_layer_expansion": expand_layers},
    )()
    metrics_store._trace_store = trace_store
    metrics_store._cluster_configs = {
        ClusterType.MONOLITHIC: cluster_config,
        ClusterType.DECODE_ATTN: cluster_config,
    }
    metrics_store._per_layer_traced_requests_by_cluster = {
        ClusterType.MONOLITHIC: set()
    }
    return metrics_store, trace_store


def _build_batch_stage() -> SimpleNamespace:
    return SimpleNamespace(
        _batch_id=17,
        num_tokens=[8],
        effective_total_tokens_compute=8,
        effective_total_tokens_transfer=8,
        effective_total_tokens_rounded=8,
        tokens_are_post_routing=False,
    )


def test_execution_time_layer_ids_are_normalized_and_exposed() -> None:
    execution_time = _build_dense_execution_time(
        num_layers=2,
        layer_ids=[4, 5],
    )
    assert execution_time.layer_ids == (4, 5)
    assert execution_time.layer_id is None


@pytest.mark.parametrize(
    "layer_ids, expected_error",
    [
        ([1], "length"),
        ([1, 1], "duplicate"),
        ([-1, 2], "non-negative"),
        ([1, "2"], "exact integers"),
    ],
)
def test_execution_time_layer_ids_reject_invalid_shape(
    layer_ids: list[object], expected_error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_error):
        _build_dense_execution_time(num_layers=2, layer_ids=layer_ids)


def test_mixed_step3_per_layer_trace_uses_layer_contract() -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    execution_time = _build_step3_execution_time()
    execution_time._trace_dense_layer_id = 0
    execution_time._trace_dense_mlp_layer_up_proj_execution_time = 4.0
    execution_time._trace_dense_mlp_layer_act_execution_time = 4.0
    execution_time._trace_dense_mlp_layer_down_proj_execution_time = 4.0
    metrics_store, trace_store = _build_metrics_store(model_config)

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_build_batch_stage(),
        replica_id=0,
        execution_time=execution_time,
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=["request-1"],
    )

    events_by_layer = {}
    for event in trace_store.events:
        if event.layer_id is not None and event.layer_id >= 0:
            events_by_layer.setdefault(event.layer_id, set()).add(event.name)

    dense_names = events_by_layer[0]
    routed_names = events_by_layer[4]
    assert "mlp_up_proj" in dense_names
    assert "moe_grouped_gemm" not in dense_names
    assert "moe_gating_linear" not in dense_names
    assert "share_expert_up_proj" not in dense_names
    assert "expert_parallel_alltoall" not in dense_names
    assert "expert_parallel_alltoall_dispatch" not in dense_names
    assert "moe_grouped_gemm" in routed_names
    assert "moe_gating_linear" in routed_names
    assert "share_expert_up_proj" in routed_names
    # Step3's profile-owned collective contract uses the generic alltoall
    # trace name; dispatch/combine remain the split-phase contract for roles
    # that do not select the profile-owned representation.
    assert "expert_parallel_alltoall" in routed_names
    assert "expert_parallel_alltoall_dispatch" not in routed_names
    assert "expert_parallel_alltoall_combine" in routed_names

    dense_mlp_event = next(
        event
        for event in trace_store.events
        if event.layer_id == 0 and event.name == "mlp_up_proj"
    )
    assert dense_mlp_event.meta["precision_op"] == "mlp_up_proj"
    assert dense_mlp_event.meta["tensor_shape"]
    assert dense_mlp_event.meta["tensor_size_bytes"]


@pytest.mark.parametrize("layer_id", [0, 1, 2, 3, 60])
def test_step3_dense_layer_map_is_profile_owned(layer_id: int) -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    profile = model_config.get_model_architecture_profile()
    resolved = profile.resolve_layer_contract(
        model_config,
        layer_id=layer_id,
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    assert resolved.layer_kind.value == "dense"


def test_mixed_multi_layer_aggregate_requires_layer_identity() -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    metrics_store, _trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    with pytest.raises(ValueError, match="layer identity"):
        metrics_store._emit_op_level_traces(
            time=0.0,
            batch_stage=_build_batch_stage(),
            replica_id=0,
            execution_time=_build_step3_execution_time(),
            cluster_type=ClusterType.MONOLITHIC,
            request_ids=[],
        )


def test_mixed_multi_layer_aggregate_accepts_complete_layer_identity() -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    execution_time = _build_step3_execution_time(
        layer_ids=list(range(model_config.num_layers))
    )
    metrics_store, trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_build_batch_stage(),
        replica_id=0,
        execution_time=execution_time,
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=[],
    )

    assert trace_store.events
    assert all(
        event.meta.get("layer_ids") == tuple(range(model_config.num_layers))
        for event in trace_store.events
        if event.type in {"COMPUTE", "COMM"}
    )


@pytest.mark.parametrize(
    "layer_ids, expected_error",
    [
        ([0, 0, 1], "duplicate"),
        (list(range(60)) + [61], "out of range"),
    ],
)
def test_mixed_multi_layer_aggregate_rejects_invalid_layer_identity(
    layer_ids: list[int],
    expected_error: str,
) -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    if expected_error == "duplicate":
        with pytest.raises(ValueError, match="duplicate"):
            _build_step3_execution_time(
                num_layers=len(layer_ids),
                layer_ids=layer_ids,
            )
        return
    execution_time = _build_step3_execution_time(
        num_layers=len(layer_ids),
        layer_ids=layer_ids,
    )
    metrics_store, _trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    with pytest.raises(ValueError, match=expected_error):
        metrics_store._emit_op_level_traces(
            time=0.0,
            batch_stage=_build_batch_stage(),
            replica_id=0,
            execution_time=execution_time,
            cluster_type=ClusterType.MONOLITHIC,
            request_ids=[],
        )


def test_pure_dense_multi_layer_aggregate_keeps_existing_behavior() -> None:
    model_config = BaseModelConfig.create_from_name("llama3.1-8b")
    execution_time = _build_dense_execution_time(num_layers=model_config.num_layers)
    metrics_store, trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_build_batch_stage(),
        replica_id=0,
        execution_time=execution_time,
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=[],
    )

    assert any(event.name == "mlp_up_proj" for event in trace_store.events)


def test_pure_moe_multi_layer_aggregate_keeps_existing_behavior() -> None:
    model_config = BaseModelConfig.create_from_name("mixtral_8x7b_moe")
    metrics_store, trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_build_batch_stage(),
        replica_id=0,
            execution_time=_build_pure_moe_execution_time(
                num_layers=model_config.num_layers
            ),
        cluster_type=ClusterType.MONOLITHIC,
        request_ids=[],
    )

    assert any(event.name == "moe_grouped_gemm" for event in trace_store.events)


def test_attention_only_aggregate_keeps_existing_behavior() -> None:
    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    metrics_store, trace_store = _build_metrics_store(
        model_config,
        expand_layers=False,
    )

    metrics_store._emit_op_level_traces(
        time=0.0,
        batch_stage=_build_batch_stage(),
        replica_id=0,
        execution_time=_build_step3_execution_time(attention_time=0.1),
        cluster_type=ClusterType.DECODE_ATTN,
        request_ids=[],
    )

    assert trace_store.events
    assert not any(event.name == "moe_grouped_gemm" for event in trace_store.events)
