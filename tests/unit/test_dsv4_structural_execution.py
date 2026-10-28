"""Canonical timing conservation for explicitly synthetic V4 execution."""

import pytest

from frontier.entities import ExecutionTime
from frontier.types import ClusterType
from frontier.scheduler.utils.expert_parallel import get_ep_phase_times_ms
from frontier.moe_ep_workload import (
    build_contiguous_expert_ownership, materialize_layer_ep_workload,
)
from test_dsv4_source_execution import predictor, source_batch


def test_structural_driver_accepts_boundary_workload_inputs(tmp_path):
    import runpy
    from pathlib import Path

    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "integration/run_dsv4_structural.py"))
    argv = driver["command"](
        "online", tmp_path, 2, prefill_tokens=2051, chunk_size=2050,
        decode_tokens=3, num_blocks=1024,
    )
    assert argv[argv.index("--vllm_v1_scheduler_config_max_model_len") + 1] == "2054"
    assert argv[argv.index("--vllm_v1_scheduler_config_max_tokens_in_batch") + 1] == "2050"
    assert argv[argv.index("--vllm_v1_scheduler_config_long_prefill_token_threshold") + 1] == "2050"
    assert argv[argv.index("--trace_request_generator_config_max_tokens") + 1] == "2054"
    assert argv[argv.index("--vllm_v1_scheduler_config_num_blocks") + 1] == "1024"


def test_structural_driver_preserves_default_workload_and_capacity(tmp_path):
    import runpy
    from pathlib import Path

    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "integration/run_dsv4_structural.py"))
    argv = driver["command"]("offline", tmp_path, 4)
    for option, value in (
        ("--vllm_v1_scheduler_config_max_model_len", "256"),
        ("--vllm_v1_scheduler_config_num_blocks", "256"),
        ("--vllm_v1_scheduler_config_max_tokens_in_batch", "16"),
        ("--vllm_v1_scheduler_config_long_prefill_token_threshold", "16"),
        ("--trace_request_generator_config_max_tokens", "256"),
    ):
        assert argv[argv.index(option) + 1] == value


def test_structural_driver_admits_only_evidenced_dp8_scope(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path

    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "integration/run_dsv4_structural.py"))
    captured = []

    def capture(case, output_root, dp, **workload):
        argv = driver["command"](case, output_root, dp, **workload)
        captured.append((case, dp, argv))
        return {"case": case, "logical_gpus": dp * 8}

    driver["main"].__globals__["run_case"] = capture
    output = tmp_path / "dp8"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "run_dsv4_structural.py", "--output-root", str(output),
        "--dp", "8", "--case", "offline",
    ])

    driver["main"]()

    case, dp, argv = captured.pop()
    assert (case, dp) == ("offline", 8)
    assert argv[argv.index("--replica_config_attn_dp") + 1] == "8"
    assert argv[argv.index("--replica_config_moe_expert_parallel_size") + 1] == "64"
    assert argv[argv.index("--metrics_config_num_requests_to_trace_per_layer") + 1] == "8"
    assert json.loads((output / "report.json").read_text()) == [
        {"case": "offline", "logical_gpus": 64},
    ]

    monkeypatch.setattr(sys, "argv", [
        "run_dsv4_structural.py", "--output-root", str(output), "--dp", "16",
    ])
    with pytest.raises(SystemExit):
        driver["main"]()


@pytest.mark.parametrize("field,value", [
    ("prefill_tokens", 0), ("chunk_size", -1), ("decode_tokens", True),
    ("num_blocks", 0), ("num_blocks", -1), ("num_blocks", True),
])
def test_structural_driver_rejects_invalid_workload_inputs(tmp_path, field, value):
    import runpy
    from pathlib import Path

    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "integration/run_dsv4_structural.py"))
    with pytest.raises(ValueError, match="positive integers"):
        driver["command"]("offline", tmp_path, 2, **{field: value})


def test_structural_boundary_validator_rejects_missing_store_and_wrong_write_rows():
    import copy
    import runpy
    from pathlib import Path
    from types import SimpleNamespace
    from frontier.attention.deepseek_v4 import build_deepseek_v4_attention_workload
    from frontier.config.model_config import BaseModelConfig
    from frontier.entities import Batch, Request

    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "integration/run_dsv4_structural.py"))
    validate = driver["validate_attention_boundary"]
    model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    replica = SimpleNamespace(
        num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=2,
        attention_backend="FLASHMLA_SPARSE_DSV4",
        speculative_decoding_config=SimpleNamespace(enabled=False),
    )
    batch = Batch(0, [Request(0.0, 128, 3)], [1], num_context_tokens=[127],
                  request_is_decoding=[False], is_moe=True)
    calls = build_deepseek_v4_attention_workload(
        model, replica, batch, 3, cluster_type=ClusterType.MONOLITHIC,
        execution_mode="eager",
        prefix_caching=False,
        decode_context_parallel_size=1, prefill_context_parallel_size=1,
    )["operators"]
    validate(calls, query=1, context=127, layer=3)
    missing = dict(calls)
    del missing["dsv4_compressor_fused_compress_store"]
    with pytest.raises(ValueError, match="compressor_fused_compress_store"):
        validate(missing, query=1, context=127, layer=3)
    incorrect = copy.deepcopy(calls)
    incorrect["dsv4_compressor_fused_compress_store"][0]["new_compressed_rows"] = (0,)
    with pytest.raises(ValueError, match="new_compressed_rows"):
        validate(incorrect, query=1, context=127, layer=3)
    aggregated = {name: ({**entries[0], "global_layer_id": 3},
                         {**entries[0], "global_layer_id": 5, "input_shape": (-1,)})
                  for name, entries in calls.items()}
    validate(aggregated, query=1, context=127, layer=3)
    duplicate = dict(aggregated)
    duplicate["dsv4_fused_qkv_projection"] = (aggregated["dsv4_fused_qkv_projection"][0],) * 2
    with pytest.raises(ValueError, match="duplicate boundary call"):
        validate(duplicate, query=1, context=127, layer=3)


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
    from frontier.execution_time_predictor.deepseek_v4 import timing_operator_times
    from frontier.execution_time_predictor.dsv4_composition import serial_attention_phase

    timing = ExecutionTime(
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
    timing._dsv4_attention_schedule = serial_attention_phase(
        (name, value) for _, name, value in timing_operator_times(timing, "attention")
    )
    return timing


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
    result = predictor._predict_dsv4_structural_execution(
        source_batch(), 0, 0, lane_workload=destination(),
        execution_mode="eager",
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
    layers = [predictor._predict_dsv4_structural_execution(
        source_batch(), layer, 0, lane_workload=destination(layer),
        include_stage_owned=layer == 0,
        execution_mode="eager",
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
    with pytest.raises(ValueError, match="no phase owner"):
        execution_time({"dsv4_sparse_attention": 1.0, "attn_prefill": 2.0})


@pytest.mark.parametrize("layer,base_count", [(0, 8), (2, 19), (3, 11), (42, 19)])
@pytest.mark.parametrize("mode,rows", [("NONE", 1), ("FULL", 4), ("PIECEWISE", 2), ("PIECEWISE", 4)])
def test_dummy_attention_uses_v2_descriptor_geometry(predictor, layer, base_count, mode, rows):
    from frontier.attention.deepseek_v4 import build_deepseek_v4_attention_workload
    from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch

    batch = DummyForwardBatch(0, 0, 0)
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, 1, rows, 1, rows,
        )
    context = dict(cluster_type=ClusterType.MONOLITHIC,
                   execution_mode="eager" if mode == "NONE" else "cuda_graph",
                   prefix_caching=False, decode_context_parallel_size=1,
                   prefill_context_parallel_size=1)
    result = build_deepseek_v4_attention_workload(
        predictor._model_config, predictor._replica_config, batch, layer, **context)
    operators = result["operators"]
    csa = result["variant"] == "csa"
    expected = 16 if csa and mode == "NONE" else 18 if csa and mode == "PIECEWISE" and rows == 2 else base_count
    assert len(operators) == expected
    assert not {"dsv4_query_padding", "dsv4_attention_output_zero"} & set(operators)
    assert result["indexer_short_context"] == (csa and mode == "NONE")
    assert result["num_tokens"] == rows and result["actual_num_tokens"] == 0
    query_rows = (1,) * rows if mode == "FULL" else (rows,)
    assert result["request_intervals"] == tuple((0, q) for q in query_rows)
    ratio = predictor._model_config.compress_ratios[layer]
    assert result["new_compressed_rows"] == tuple(q // ratio if ratio > 1 else 0 for q in query_rows)
    sparse = operators["dsv4_sparse_attention"][0]
    assert sparse["swa_logical_window_span_rows"] == query_rows
    assert sparse["compressed_history_rows"] == result["new_compressed_rows"]
    assert sparse["query_intervals"] == result["request_intervals"]
    assert sparse["actual_query_rows"] == 0 and sparse["invalid_padding_rows"] == rows
    prepare = operators["dsv4_fused_q_kv_prepare"][0]
    assert prepare["kv_cache_write_rows"] == 0 and prepare["invalid_padding_rows"] == rows
    for field in ("semantic_is_decoding", "source_request_order", "decode_request_indices", "prefill_request_indices"):
        assert result[field] == ()
    for field in ("decode_request_indices", "prefill_request_indices"):
        assert sparse[field] == ()
    for field in ("num_decode_query_tokens", "num_prefill_query_tokens"):
        assert result[field] == 0
    timing = predictor._predict_attention_layer_time(batch, layer, ClusterType.MONOLITHIC)
    assert timing.operator_times.op_times == {scope: predictor._dummy_execution_time * len(calls)
                                             for scope, calls in operators.items()}


def test_idle_geometry_has_no_request_rows_or_lm_head(predictor):
    from frontier.entities import DummyForwardBatch
    from frontier.execution_time_predictor.deepseek_v4 import source_physical_rows

    predictor._replica_config.router_topk = 6
    predictor._replica_config.total_expert_num = 256
    dummy = DummyForwardBatch(0, 0, 0)
    result = predictor._predict_dsv4_structural_execution(
        dummy, 42, 0, lane_workload=destination(42),
        execution_mode="eager",
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
    assert "dsv4_sparse_attention" in result.op_times
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
    layers = [predictor._predict_dsv4_structural_execution(
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
                               cluster_type=ClusterType.MONOLITHIC,
                               request_ids=[str(request.id) for request in batch.requests])
    assert {event.layer_id >= 0 for event in sink.events if event.type == "COMPUTE"} == {expand}
    physical = [event for event in sink.events if event.type in {"COMPUTE", "COMM"}]
    assert sum(event.duration_ms for event in sink.events) == pytest.approx(stage.total_time * 1000)
    # The serial attention schedule keeps every event back to back on the timeline.
    ends = [event.ts_start * 1000 + event.duration_ms for event in sink.events]
    assert [event.ts_start * 1000 for event in sink.events[1:]] == pytest.approx(ends[:-1])
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
