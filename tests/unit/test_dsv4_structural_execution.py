"""Canonical timing conservation for explicitly synthetic V4 execution."""

import pytest

from frontier.entities import ExecutionTime
from frontier.types import ClusterType
from frontier.scheduler.utils.expert_parallel import get_ep_phase_times_ms
from frontier.moe_ep_workload import (
    build_contiguous_expert_ownership, materialize_layer_ep_workload,
)
from test_dsv4_source_execution import predictor, source_batch


def test_actual_dummy_constructor_preserves_integer_only_precision(tmp_path):
    from frontier.config import (
        MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
        VllmV1SchedulerConfig, get_quantization_manager,
    )
    from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
        RandomForrestExecutionTimePredictor,
    )

    replica = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=4,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=32,
        total_expert_num=256, router_topk=6,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
        indexer_kv_cache_dtype="fp8",
    )
    result = RandomForrestExecutionTimePredictor(
        RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True), replica,
        VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "metrics")),
    )
    assert result._enable_dummy_mode
    with pytest.raises(ValueError, match="integer-only source tensor"):
        get_quantization_manager().get_precision("dsv4_index_short_context_fill")


def destination(layer=0):
    workload = materialize_layer_ep_workload(
        routing_ratios={i: 1 / 256 for i in range(256)},
        target_replica_id=0, global_layer_id=layer, routing_token_count=40,
        router_topk=6, total_expert_num=256, moe_expert_parallel_size=64,
        expert_to_ep=build_contiguous_expert_ownership(256, 64),
    )
    return workload.lane(0)


def execution_time(op_times, layer=0):
    return ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0, mlp_norm_time=0.0, add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0, moe_shuffling_time=0.0,
        schedule_time=0.0, sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0, process_model_outputs_time=0.0,
        ray_comm_time=0.0, is_moe=True, op_times=op_times,
        global_layer_id=layer, attention_family_id="deepseek_v4_attention",
        attention_variant_id="uncompressed" if layer < 2 else "csa",
    )


def test_physical_scopes_conserve_attention_and_five_ep_phases():
    result = execution_time({
        "dsv4_embedding": 1.0, "dsv4_mhc_first_pre": 2.0,
        "dsv4_sp_attention_allgather": 3.0,
        "dsv4_sparse_attention": 4.0,
        "dsv4_sp_attention_reduce_scatter": 5.0,
        "dsv4_mhc_attn_post_ffn_pre": 6.0,
        "share_expert_up_proj": 7.0, "moe_gating_linear": 8.0,
        "moe_gating_routing_topk": 9.0, "moe_grouped_gemm": 10.0,
    })
    assert result.get_single_layer_attention_scope_time() == 15.0
    assert result.get_single_layer_moe_pre_dispatch_time() == 30.0
    assert result.get_single_layer_moe_post_dispatch_compute_time() == 10.0
    assert result.get_single_layer_block_time() == 55.0
    assert result.attn_norm_time == result.mlp_norm_time == 0.0
    assert result.add_ffn_residual_time == result.add_attn_residual_time == 0.0


def test_source_and_destination_timing_use_distinct_physical_rows(predictor):
    predictor._replica_config.router_topk = 6
    predictor._replica_config.total_expert_num = 256
    result = predictor._predict_dsv4_dummy_structural_execution(
        source_batch(), 0, 0, lane_workload=destination(),
        execution_mode="eager", has_attention_metadata=True,
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=False, input_padding_mask=None, selected_logits_rows=None,
    )
    assert result.get_single_layer_attention_scope_time() == 30.0
    assert get_ep_phase_times_ms(
        result, cluster_type=ClusterType.MONOLITHIC, batch_id=0, layer_id=0, ep_id=0,
    ) == (12.0, 9.0, 2.0, 5.0, 0.0)
    assert result.get_single_layer_block_time() == 58.0
    ep_calls = [value for _, value in predictor._cc_backend.calls
                if value["comm_domain"] == "MOE_EP"]
    assert [value["data_size_bytes"] for value in ep_calls] == [24576, 72, 72, 327680]


def test_full_43_layer_stage_preserves_every_canonical_physical_scope(predictor):
    from frontier.entities import StageExecutionTime
    from frontier.execution_time_predictor.deepseek_v4 import component_ledger

    predictor._replica_config.router_topk = 6
    predictor._replica_config.total_expert_num = 256
    layers = [predictor._predict_dsv4_dummy_structural_execution(
        source_batch(), layer, 0, lane_workload=destination(layer),
        include_stage_owned=layer == 0,
        execution_mode="eager", has_attention_metadata=True,
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=False, input_padding_mask=None,
        selected_logits_rows=1 if layer == 42 else None,
    ) for layer in range(43)]
    stage = StageExecutionTime(layers)
    assert stage.op_times["dsv4_embedding"] == 2.0
    assert stage.op_times["dsv4_mhc_first_pre"] == 2.0
    assert stage.op_times["dsv4_mhc_ffn_post_attn_pre"] == 84.0
    assert stage.op_times["dsv4_mhc_attn_post_ffn_pre"] == 86.0
    assert stage.op_times["dsv4_sparse_attention"] == 86.0
    assert stage.op_times["dsv4_logits_gather"] == 2.0
    assert stage.global_layer_ids == tuple(range(43))
    assert sum(component_ledger(stage).values()) == pytest.approx(stage.total_time * 1000)


def test_registered_legacy_scope_cannot_silently_enter_v4_phase_projection():
    result = execution_time({"dsv4_sparse_attention": 1.0, "attn_prefill": 2.0})
    with pytest.raises(ValueError, match="no phase owner"):
        result.get_single_layer_block_time()


@pytest.mark.parametrize("layer", [0, 2, 3, 42])
def test_explicit_idle_forward_has_its_own_no_metadata_operator_scope(predictor, layer):
    from frontier.attention.deepseek_v4 import build_deepseek_v4_dummy_attention_workload
    from frontier.entities import DummyForwardBatch

    batch = DummyForwardBatch(0, 0, 0)
    workload = build_deepseek_v4_dummy_attention_workload(
        predictor._model_config, predictor._replica_config, batch, layer,
        cluster_type=ClusterType.MONOLITHIC, execution_mode="eager",
        has_attention_metadata=False, prefix_caching=False,
        decode_context_parallel_size=1, prefill_context_parallel_size=1,
    )
    names = set(workload["operators"])
    assert {"dsv4_query_padding", "dsv4_attention_output_zero"} <= names
    assert not names.intersection({
        "dsv4_fused_q_kv_prepare", "dsv4_sparse_attention", "dsv4_index_scores",
        "dsv4_index_topk", "dsv4_index_short_context_fill",
        "dsv4_compressor_save_partial_states", "dsv4_compressor_fused_compress_store",
        "dsv4_index_compressor_save_partial_states",
        "dsv4_index_compressor_fused_compress_store",
    })
    assert workload["operators"]["dsv4_fused_qkv_projection"][0]["input_shape"] == (1, 4096)
    assert ("dsv4_index_query_projection" in names) == (layer >= 2 and layer % 2 == 0)


def test_idle_branch_never_substitutes_for_real_batch_metadata(predictor):
    from frontier.attention.deepseek_v4 import (
        build_deepseek_v4_attention_workload,
        build_deepseek_v4_dummy_attention_workload,
    )
    from frontier.entities import DummyForwardBatch

    context = dict(cluster_type=ClusterType.MONOLITHIC, execution_mode="eager",
                   has_attention_metadata=False, prefix_caching=False,
                   decode_context_parallel_size=1, prefill_context_parallel_size=1)
    with pytest.raises(NotImplementedError, match="real metadata"):
        build_deepseek_v4_attention_workload(
            predictor._model_config, predictor._replica_config, source_batch(), 0, **context)
    with pytest.raises(ValueError, match="explicit DummyForwardBatch"):
        build_deepseek_v4_dummy_attention_workload(
            predictor._model_config, predictor._replica_config, source_batch(), 0, **context)
    context["has_attention_metadata"] = True
    with pytest.raises(ValueError, match="absent attention metadata"):
        build_deepseek_v4_dummy_attention_workload(
            predictor._model_config, predictor._replica_config, DummyForwardBatch(0, 0, 0),
            0, **context)


def test_idle_geometry_has_no_request_rows_or_lm_head(predictor):
    from frontier.entities import DummyForwardBatch
    from frontier.execution_time_predictor.deepseek_v4 import source_physical_rows

    predictor._replica_config.router_topk = 6
    predictor._replica_config.total_expert_num = 256
    dummy = DummyForwardBatch(0, 0, 0)
    result = predictor._predict_dsv4_dummy_structural_execution(
        dummy, 42, 0, lane_workload=destination(42),
        execution_mode="eager", has_attention_metadata=False,
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=False, input_padding_mask=None, selected_logits_rows=0,
    )
    assert result._dsv4_source_rows["source_rows"] == 1
    assert result._dsv4_source_rows["physical_local_rows"] == 1
    assert result._dsv4_source_rows["valid_local_rows"] == 1
    assert result._dsv4_source_rows["request_valid_local_rows"] == 0
    assert result._dsv4_source_rows["source_forward_kind"] == "request-free dummy"
    assert "dsv4_lm_head" not in result.op_times
    assert "dsv4_logits_gather" not in result.op_times
    assert "dsv4_sparse_attention" not in result.op_times
    predictor._replica_config.attn_dp = 2
    assert source_physical_rows(predictor, {0: source_batch(17), 1: dummy}) == 32


@pytest.mark.parametrize("expand", [False, True])
def test_real_stage_trace_preserves_physical_scope_and_source_metadata(predictor, tmp_path, expand):
    from types import SimpleNamespace
    from frontier.config import (
        ClusterConfig, MetricsConfig, ReplicaConfig, SimulationConfig, global_vars,
        get_quantization_manager,
    )
    from frontier.entities import BatchStage, StageExecutionTime
    from frontier.execution_time_predictor.deepseek_v4 import serving_context
    from frontier.metrics.metrics_store import MetricsStore

    global_vars.reset_global_vars()

    get_quantization_manager().configure_from_model_config(predictor._model_config)
    replica = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=4, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=32, total_expert_num=256, router_topk=6,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8", indexer_kv_cache_dtype="fp8")
    cluster = ClusterConfig(replica_config=replica)
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(
        output_dir=str(tmp_path), write_metrics=False, store_plots=False,
        enable_op_level_tracing=True, enable_per_layer_expansion=expand,
        num_requests_to_trace_per_layer=1 if expand else 0))
    sink = SimpleNamespace(events=[])
    sink.log_event = sink.events.append
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, sink)
    batch = source_batch()
    layers = [predictor._predict_dsv4_dummy_structural_execution(
        batch, layer, 0, include_ffn=False, include_stage_owned=layer == 0,
        **serving_context(predictor, batch, layer),
    ) for layer in range(43)]
    stage = StageExecutionTime(layers)
    batch_stage = BatchStage(batch.id, 0, 0, stage.total_time, stage.model_time,
                             batch.requests, batch.num_tokens, batch.request_is_decoding,
                             ClusterType.MONOLITHIC)
    batch_stage.on_schedule(1.0)
    store._emit_op_level_traces(time=1.0, replica_id=0,
                               batch_stage=batch_stage, execution_time=stage,
                               cluster_type=ClusterType.MONOLITHIC)
    physical = [event for event in sink.events if event.type in {"COMPUTE", "COMM"}]
    assert sum(event.duration_ms for event in sink.events) == pytest.approx(stage.total_time * 1000)
    assert sum(event.duration_ms for event in physical) == pytest.approx(sum(stage.op_times.values()))
    names = {event.name for event in physical}
    assert {"dsv4_embedding", "dsv4_embedding_allreduce", "dsv4_mhc_first_pre"} <= names
    assert "attn_prefill" not in names
    sparse = [event for event in physical if event.name == "dsv4_sparse_attention"]
    covered = {layer for event in sparse for layer in event.meta.get("global_layer_ids", [event.layer_id])}
    assert covered == set(range(43))
    assert all(event.meta["source_rows"] == 17 for event in sparse)
    assert all(event.meta["native_routed_assignments"] == "UNKNOWN" for event in sparse)
    assert all(event.meta["source_tensor_calls"] for event in sparse)
    global_vars.reset_global_vars()


def test_ep_expansion_quota_uses_real_source_ids_before_virtual_destinations():
    from types import MethodType
    from frontier.metrics.ep_wave_metrics import record_ep_wave
    from frontier.metrics.metrics_store import MetricsStore
    from test_ep_wave_trace_context import _reporting_case

    store, plan = _reporting_case(operations=False, ledger=False)
    store._config.enable_per_layer_expansion = True
    store._config.num_requests_to_trace_per_layer = 1
    store._per_layer_traced_requests_by_cluster = {}
    store._expanded_trace_batches = set()
    store._should_expand_layers = MethodType(MetricsStore._should_expand_layers, store)
    record = plan.phase_times.lane_records[0]
    virtual_ids = [str(request.id) for request in record.batch.requests]
    layer = record.execution_time.layer_execution_times[0]
    layer._dsv4_source_rows = {"source_request_ids": [], "physical_local_rows": 1}
    record_ep_wave(store, plan, time=1.0, replica_id=0, stage_id=2,
                   cluster_type=ClusterType.MONOLITHIC)
    assert not store._per_layer_traced_requests_by_cluster
    assert not store._expanded_trace_batches
    idle_events = [call.args[0] for call in store.trace_store.log_event.call_args_list]
    assert idle_events
    assert all(event.layer_id == -1 for event in idle_events)
    assert all(event.meta["request_ids"] == [] for event in idle_events)
    assert all(event.meta["synthetic_destination_request_ids"] == virtual_ids for event in idle_events)

    store.trace_store.log_event.reset_mock()
    layer._dsv4_source_rows = {"source_request_ids": ["actual-source-request"],
                              "physical_local_rows": 1}
    record_ep_wave(store, plan, time=1.0, replica_id=0, stage_id=2,
                   cluster_type=ClusterType.MONOLITHIC)
    assert store._per_layer_traced_requests_by_cluster == {
        ClusterType.MONOLITHIC: {"actual-source-request"}}
    real_events = [call.args[0] for call in store.trace_store.log_event.call_args_list]
    assert real_events
    assert all(event.layer_id == 7 for event in real_events)
    assert all(event.meta["request_ids"] == ["actual-source-request"] for event in real_events)
    assert all(event.meta["synthetic_destination_request_ids"] == virtual_ids for event in real_events)
