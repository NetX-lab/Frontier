"""Source-defined Step scopes priced with controlled durations through real owners.

These CPU checks establish a limited structural path, not native execution,
CUDA graph replay, hardware capacity, or timing accuracy.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from predictor_cache_fixtures import predictor_fixture_config
from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.config import (
    BaseModelConfig, ClusterConfig, MetricsConfig,
    RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
    SimulationConfig, VllmV1SchedulerConfig, global_vars,
)
from frontier.config.quantization_manager import QuantizationManager
from frontier.entities import Batch, BatchStage, Request
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.entities.time_components import MLPTime, MoETime
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType


REPO = Path(__file__).resolve().parents[2]
DUMMY_MS = .25
ROUTED_LAYERS = tuple(range(3, 91))
CPU_FIELDS = (
    "schedule_time", "prepare_inputs_e2e_time", "sampler_e2e_time",
    "process_model_outputs_time", "ray_comm_time",
)


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    for name, value in (
        ("SYS_ARCH", "co-location"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("KV_CONNECTOR", "none"), ("QUANTIZATION_MANAGER", None),
        ("CUDAGRAPH_CAPTURE_SIZES", None),
        ("ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC", False),
        ("KV_SEND_SLOWDOWN_PER_LAYER_MS", 0.0),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
    ):
        monkeypatch.setattr(global_vars, name, value)
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)
    replica = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=16, attn_dp=1, moe_tensor_parallel_size=16,
        moe_expert_parallel_size=1, attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    replica.cluster_num_replicas = 1
    result = predictor_fixture_config(
        model_config=replica.model_config, total_experts=352, router_topk=8,
    )
    result.update(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(
            block_size=64, max_model_len=8192, batch_size_cap=1,
            max_tokens_in_batch=512, long_prefill_token_threshold=512,
            enable_chunked_prefill=True,
        ),
        predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True, dummy_execution_time_ms=DUMMY_MS,
        ),
        metrics_config=MetricsConfig(output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache")),
    )
    return result


def batch(phase="prefill", context=0, query=512, graph="NONE", rows=None):
    request = Request(
        arrived_at=0.0, num_prefill_tokens=4105, num_decode_tokens=9,
        num_processed_tokens=context,
    )
    result = Batch(
        replica_id=0, requests=[request], num_tokens=[query], is_moe=True,
        request_is_decoding=[phase == "decode"], num_context_tokens=[context],
    )
    if graph != "NONE":
        result.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            config_mode="full_decode_only" if graph == "FULL" else "piecewise",
            runtime_mode=graph, capture_hit=True, is_mixed_batch=False,
            original_total_tokens=query, padded_total_tokens=rows or query,
            original_decode_batch_size=1, padded_decode_batch_size=rows or 1,
        )
    return result


def predict(predictor, source_batch, *, first=0, count=92, include_attention=True):
    return predictor.predict_stage_execution_time(
        source_batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        layer_id=first, num_layers=count, include_attention=include_attention,
    )


@pytest.mark.parametrize(("phase", "context", "query", "graph", "rows", "sparse_scopes", "total_ms"), [
    ("prefill", 0, 512, "NONE", None, 6, 389.25),
    ("prefill", 512, 512, "NONE", None, 6, 389.25),
    ("prefill", 4096, 9, "NONE", None, 6, 389.25),
    ("decode", 4105, 1, "NONE", None, 5, 383.5),
    ("decode", 4105, 1, "FULL", 8, 5, 383.5),
    ("decode", 4105, 1, "PIECEWISE", 8, 6, 389.25),
    ("decode", 4105, 1, "FULL", 129, 6, 389.25),
])
def test_full_backbone_scopes_layers_and_cpu_are_owned_once(
    inputs, phase, context, query, graph, rows, sparse_scopes, total_ms,
):
    predictor = RandomForrestExecutionTimePredictor(**inputs)
    source_batch = batch(phase, context, query, graph, rows)
    stage = predict(predictor, source_batch)
    assert stage.global_layer_ids == tuple(range(92))
    assert stage.attention_family_ids == ("step5_attention",) * 92
    assert Counter(stage.attention_variant_ids) == {"swa": 69, "sparse_gqa": 23}
    assert [layer.global_layer_id for layer in stage.layer_execution_times
            if isinstance(layer.moe_or_mlp_time_component, MLPTime)] == [0, 1, 2, 91]
    assert [layer.global_layer_id for layer in stage.layer_execution_times
            if isinstance(layer.moe_or_mlp_time_component, MoETime)] == list(ROUTED_LAYERS)
    assert stage.step5_ffn_merge_time == len(ROUTED_LAYERS) * DUMMY_MS
    assert stage.attention_operator_times.total_time() == (69 * 6 + 23 * sparse_scopes) * DUMMY_MS
    assert stage.total_time * 1000 == total_ms
    assert stage.model_time_ms == total_ms - len(CPU_FIELDS) * DUMMY_MS
    assert stage.pipeline_parallel_communication_time == 0
    for field in ("embedding_time", "embedding_allreduce_time", "input_cast_time", "final_residual_time"):
        assert getattr(stage, field) == DUMMY_MS
        assert all(getattr(layer, field) == 0 for layer in stage.layer_execution_times)
    assert stage.communication_time_component.total_time() == 185 * DUMMY_MS
    assert stage.op_times["emb"] == stage.op_times["input_cast"] == DUMMY_MS
    assert stage.op_times["final_residual"] == DUMMY_MS
    assert stage.op_times["embedding_tensor_parallel_allreduce"] == DUMMY_MS
    assert stage.attention_all_reduce_time == stage.mlp_all_reduce_time == 92 * DUMMY_MS
    assert stage.expert_parallel_communication_time == 0
    for field in CPU_FIELDS:
        assert getattr(stage, field) == DUMMY_MS
        assert [getattr(layer, field) for layer in stage.layer_execution_times] == [
            DUMMY_MS, *([0.0] * 91),
        ]
    for layer in stage.layer_execution_times:
        attention = layer.attention_time_component
        assert attention.attention_layer_pre_proj_execution_time == 0
        assert attention.attention_layer_post_proj_execution_time == 0
        assert attention.attention_rope_execution_time == 0
        assert attention.attention_kv_cache_save_execution_time == 0
        assert attention.attention_prefill_execution_time == 0
        assert attention.attention_decode_execution_time == 0
        assert layer.add_attn_residual_time == layer.add_ffn_residual_time == 0
    sparse = stage.layer_execution_times[3]
    expected = ["step5_qkv_projection"]
    if sparse_scopes == 5:
        expected += ["step5_indexer_weights", "step5_attention_prepare"]
    else:
        expected += ["step5_attention_prepare", "step5_indexer_weights", "step5_indexer_prepare"]
    expected += ["step5_sparse_attention", "step5_output_projection"]
    for payload in (sparse, sparse.finalized_copy(), sparse.as_single_layer(
        global_layer_id=3, attention_family_id="step5_attention", attention_variant_id="sparse_gqa",
    )):
        assert [op.name for op, _ in get_attention_trace_op_times(
            payload, STEP5_ATTENTION_FAMILY, allow_structural=True,
        )] == expected
    assert source_batch.num_context_tokens == [context]
    assert source_batch.num_tokens == [query]
    assert source_batch.requests[0].num_processed_tokens == context



@pytest.mark.parametrize("phase,context,query", [("prefill", 0, 512), ("decode", 4105, 1)])
@pytest.mark.parametrize("first,count", [(0, 1), (1, 1), (91, 1), (0, 92)])
def test_attention_only_serving_prediction_preserves_stage_boundaries(
    inputs, phase, context, query, first, count,
):
    predictor = RandomForrestExecutionTimePredictor(**inputs)
    stage = predictor.predict_stage_execution_time(
        batch(phase, context, query), stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        layer_id=first, num_layers=count, include_ffn=False,
    )
    prefix = DUMMY_MS if first == 0 else 0.0
    assert stage.embedding_time == stage.embedding_allreduce_time == stage.input_cast_time == prefix
    assert stage.final_residual_time == (DUMMY_MS if first + count == 92 else 0.0)
    assert all(layer.embedding_time == layer.embedding_allreduce_time == layer.input_cast_time
               == layer.final_residual_time == 0 for layer in stage.layer_execution_times)



@pytest.mark.parametrize("phase,context,query", [("prefill", 0, 512), ("decode", 4105, 1)])
def test_first_attention_event_and_final_timing_charge_model_boundaries_once(inputs, phase, context, query):
    from frontier.scheduler.utils.collective_timing import (
        attention_delay_seconds, prepare_decode_final_timing, prepare_prefill_final_timing,
    )
    predictor = RandomForrestExecutionTimePredictor(**inputs)
    source_batch = batch(phase, context, query)
    first = predictor.predict_stage_execution_time(
        source_batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        layer_id=0, num_layers=1, include_ffn=False,
    )
    assert first.layer_execution_times[0].get_single_layer_attention_scope_time() == 2.0
    assert first.get_single_layer_attention_scope_time() == 2.75
    assert attention_delay_seconds(first) == .00275
    last = predictor.predict_stage_execution_time(
        source_batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        layer_id=91, num_layers=1, include_ffn=False,
    )
    assert last.get_single_layer_attention_scope_time() == (1.75 if phase == "decode" else 2.0)
    prefill = prepare_prefill_final_timing(last, [100.0], .1, 0.0, 0.0)
    assert prefill.explicit_model_time == pytest.approx(.10025)
    assert prefill.total_time == pytest.approx(.001)
    assert prefill.completion_time == pytest.approx(.101)
    decode = prepare_decode_final_timing(last, .1)
    assert decode.total_time == pytest.approx(.001)


def test_merge_then_fp32_tp_reduction_has_one_phase_and_one_ledger_owner(inputs):
    stage = predict(RandomForrestExecutionTimePredictor(**inputs), batch("decode", 4105, 1))
    layer = stage.layer_execution_times[3]
    routed = layer.moe_phase_operator_times("routed_compute")
    post = layer.moe_phase_operator_times("post_combine")
    assert routed == (("COMPUTE", "moe_grouped_gemm", DUMMY_MS),)
    assert [name for _, name, value in post if value > 0] == [
        "step5_ffn_merge", "moe_tensor_parallel_allreduce",
    ]
    names = Counter(name for phase in ("pre_dispatch", "routed_compute", "post_combine")
                    for _, name, _ in layer.moe_phase_operator_times(phase))
    assert names["step5_ffn_merge"] == names["moe_tensor_parallel_allreduce"] == 1
    cluster = ClusterConfig(replica_config=inputs["replica_config"])
    store = MetricsStore(SimulationConfig(
        cluster_config=cluster, metrics_config=replace(inputs["metrics_config"], write_metrics=False),
    ), {ClusterType.MONOLITHIC: cluster})
    ledger = store._build_frontier_stage_batch_component_ledger(stage)
    assert ledger["step5_ffn_merge"] == 22.0
    assert sum(ledger.values()) == pytest.approx(stage.total_time * 1000)


class TraceSink:
    def __init__(self):
        self.events = []

    def log_event(self, event):
        self.events.append(event)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("graph", ["FULL", "PIECEWISE"])
def test_actual_metrics_preserve_layer_precision_fusion_and_trace_totals(inputs, expand, graph):
    source_batch = batch("decode", 4105, 1, graph, 8)
    stage = predict(RandomForrestExecutionTimePredictor(**inputs), source_batch)
    cluster = ClusterConfig(replica_config=inputs["replica_config"])
    config = SimulationConfig(
        cluster_config=cluster, metrics_config=replace(
            inputs["metrics_config"], write_metrics=False, store_plots=False,
            enable_op_level_tracing=True, enable_per_layer_expansion=expand,
            num_requests_to_trace_per_layer=1,
        ),
    )
    sink = TraceSink()
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, sink)
    stage_batch = BatchStage(
        batch_id=73, replica_id=0, pipeline_stage=0,
        execution_time=stage.total_time, model_execution_time=stage.model_time,
        requests=source_batch.requests, num_tokens=[1], request_is_decoding=[True],
        cluster_type=ClusterType.MONOLITHIC,
        effective_total_tokens_compute=source_batch.get_effective_total_tokens_for_compute(
            ClusterType.MONOLITHIC,
        ),
    )
    stage_batch.decode_cuda_graph_metadata = source_batch.decode_cuda_graph_metadata
    stage_batch.on_schedule(1.0)
    store.on_replica_stage_schedule(
        time=1.0, replica_id=0, stage_id=0, batch_stage=stage_batch,
        execution_time=stage, cluster_type=ClusterType.MONOLITHIC,
    )
    assert sum(event.duration_ms for event in sink.events) == stage.total_time * 1000
    by_layer = {}
    for event in sink.events:
        if event.type == "OVERHEAD":
            assert event.layer_id == -1
            if event.name == "sampler_e2e":
                assert event.meta["time_ownership"] == "inclusive_sampler_e2e"
                assert event.meta["logits_rows"] == 1
                assert event.meta["tensor_shape"]["selected_hidden"] == [1, 4096]
                assert event.meta["covered_scopes"] == [
                    "select_hidden_states", "logits_input_cast", "final_norm",
                    "lm_head_linear", "logits_gather", "sampling",
                ]
            continue
        metadata = event.meta
        if metadata.get("scope") == "stage":
            assert event.layer_id == -1
            assert event.name in ("emb", "embedding_tensor_parallel_allreduce", "input_cast", "final_residual")
            assert metadata["effective_total_tokens_compute"] == 8
            assert metadata["total_tokens"] == 1
            if event.name == "embedding_tensor_parallel_allreduce":
                assert metadata["data_size_bytes"] == 8 * 4096 * 2
            elif event.name == "input_cast":
                assert metadata["tensor_dtype"] == {"input": "BF16", "output": "FP32"}
            elif event.name == "final_residual":
                assert metadata["tensor_dtype"] == {"input": "BF16", "residual": "FP32", "output": "FP32"}
            continue
        assert metadata["evidence_kind"] == "source_structural_metadata"
        assert metadata["effective_total_tokens_compute"] == 8
        assert metadata["total_tokens"] == 1
        layer = metadata["global_layer_id"]
        by_layer.setdefault(layer, []).append(event)
        assert metadata["source_geometry"]["global_layer_id"] == layer
    assert set(by_layer) == set(range(92))
    stage_names = [event.name for event in sink.events if event.meta.get("scope") == "stage"
                   and event.type != "OVERHEAD"]
    assert stage_names == ["emb", "embedding_tensor_parallel_allreduce", "input_cast", "final_residual"]
    assert sum(event.name == "step5_ffn_merge" for event in sink.events) == 88
    sparse = {event.name: event for event in by_layer[3]}
    prepare = sparse["step5_attention_prepare"].meta
    assert ("index_q" in prepare["tensor_shape"]) is (graph == "FULL")
    assert ("step5_indexer_prepare" in sparse) is (graph == "PIECEWISE")
    assert sparse["step5_ffn_merge"].meta["tensor_dtype"] == {
        "routed": "BF16", "shared": "BF16", "output": "FP32",
    }
    reduction = sparse["moe_tensor_parallel_allreduce"].meta
    assert reduction["dtype"] == "FP32"
    assert reduction["data_size_bytes"] == 8 * 4096 * 4
    dense_reduction = next(event for event in by_layer[0]
                           if event.name == "mlp_tensor_parallel_allreduce")
    assert dense_reduction.meta["dtype"] == "BF16"
    for layer_id, events in by_layer.items():
        names = [event.name for event in events]
        assert names.count("step5_qkv_projection") == names.count("step5_output_projection") == 1
        if layer_id in ROUTED_LAYERS:
            assert names.index("step5_ffn_merge") < names.index("moe_tensor_parallel_allreduce")


def test_metrics_catalog_uses_existing_registry_and_does_not_label_legacy_as_step(inputs, monkeypatch):
    for name, expected_step in (("stepfun-ai/Step-5-Preview", True), ("meta-llama/Llama-2-7b-hf", False)):
        monkeypatch.setattr(global_vars, "IS_MOE", None)
        monkeypatch.setattr(global_vars, "_IS_MOE_INITIALIZED", False)
        replica = inputs["replica_config"] if expected_step else ReplicaConfig(model_name=name)
        cluster = ClusterConfig(replica_config=replica)
        store = MetricsStore(SimulationConfig(
            cluster_config=cluster, metrics_config=replace(inputs["metrics_config"], write_metrics=False),
        ), {ClusterType.MONOLITHIC: cluster})
        names = {metric.value for metric in store._operation_metrics[ClusterType.MONOLITHIC]}
        assert (OperationMetrics.STEP5_FFN_MERGE.value in names) is expected_step
        assert (OperationMetrics.STEP5_SPARSE_ATTENTION.value in names) is expected_step


@pytest.mark.parametrize(("field", "value"), [
    ("num_pipeline_stages", 2), ("attn_dp", 2), ("moe_expert_parallel_size", 16),
    ("moe_tensor_parallel_size", 1),
])
def test_predictor_rejects_unadmitted_parallel_layout_before_pricing(inputs, field, value):
    setattr(inputs["replica_config"], field, value)
    with pytest.raises(NotImplementedError, match="MONOLITHIC, PREFILL or DECODE"):
        RandomForrestExecutionTimePredictor(**inputs)


@pytest.mark.parametrize(
    "role", [None, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, ClusterType.TRANS],
)
def test_predictor_requires_an_admitted_role(inputs, role):
    inputs["cluster_type"] = role
    with pytest.raises(NotImplementedError, match="MONOLITHIC, PREFILL or DECODE"):
        RandomForrestExecutionTimePredictor(**inputs)


def test_full_expert_parallel_layout_is_admitted_only_on_pd_roles(inputs):
    replica = inputs["replica_config"]
    replica.attn_tensor_parallel_size = 8
    replica.attn_dp = 4
    replica.moe_tensor_parallel_size = 1
    replica.moe_expert_parallel_size = 32
    with pytest.raises(NotImplementedError, match="MONOLITHIC, PREFILL or DECODE"):
        RandomForrestExecutionTimePredictor(**inputs)
    for role in (ClusterType.PREFILL, ClusterType.DECODE):
        RandomForrestExecutionTimePredictor(**{**inputs, "cluster_type": role})


def test_real_predictor_requires_measured_profiles(inputs, tmp_path):
    missing = str(tmp_path / "missing_linear_op.csv")
    inputs["predictor_config"] = replace(
        inputs["predictor_config"], enable_dummy_mode=False,
        linear_op_input_file=missing, linear_op_kernel_only_input_file=missing,
    )
    with pytest.raises(FileNotFoundError) as error:
        RandomForrestExecutionTimePredictor(**inputs)
    assert error.value.filename == missing


def test_missing_explicit_layer_is_not_silently_aggregated(inputs):
    predictor = RandomForrestExecutionTimePredictor(**inputs)
    with pytest.raises(ValueError):
        predict(predictor, batch(), first=91, count=2)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_first_serving_event_includes_input_scope_and_preserves_launch_start(inputs, phase):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
    predictor = RandomForrestExecutionTimePredictor(**inputs)
    source_batch = batch(phase, 4105 if phase == "decode" else 0, 1 if phase == "decode" else 512)
    if phase == "decode":
        source_batch.requests[0]._is_prefill_complete = True
    stage_scheduler = SimpleNamespace(
        is_busy=False, get_queue_batches=Mock(return_value=[source_batch]),
        pop_batch_if_not_busy=Mock(return_value=source_batch),
        consume_last_stale_drops=Mock(return_value=[]), _execution_time_predictor=predictor,
    )
    cluster = SimpleNamespace(
        get_replica_stage_scheduler=Mock(return_value=stage_scheduler),
        get_replica=Mock(return_value=SimpleNamespace(is_moe=True)),
    )
    events = ReplicaStageScheduleEvent(
        time=2.0, replica_id=0, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        replica_local_id=0,
    ).handle_event(SimpleNamespace(get_cluster_scheduler=lambda _: cluster), Mock())
    assert len(events) == 1
    assert source_batch._forward_launch_start_time == pytest.approx(2.0005)
    assert events[0].time == pytest.approx(2.00325)
    if phase == "prefill":
        assert source_batch._prefill_model_execution_components_ms_by_stage == {0: [2.75]}


def test_stage_and_wave_share_one_cached_layer_metadata_owner(inputs, monkeypatch):
    from unittest.mock import Mock
    from frontier.attention import step5
    from frontier.metrics import ep_wave_metrics
    from tests.unit.test_ep_wave_trace_context import _reporting_case
    from frontier.entities import StageExecutionTime
    from frontier.scheduler.utils.expert_parallel import EPWaveLaneTiming

    step_layer = predict(RandomForrestExecutionTimePredictor(**inputs), batch("decode", 4105, 1)).layer_execution_times[7]
    cluster = ClusterConfig(replica_config=inputs["replica_config"])
    store = MetricsStore(SimulationConfig(
        cluster_config=cluster, metrics_config=replace(
            inputs["metrics_config"], write_metrics=False, enable_op_level_tracing=True,
        ),
    ), {ClusterType.MONOLITHIC: cluster}, TraceSink())
    counter = Mock(wraps=step5.describe_step5_layer)
    monkeypatch.setattr(step5, "describe_step5_layer", counter)
    metadata = store._get_step5_layer_metadata(inputs["replica_config"], 7)
    assert store._get_step5_layer_metadata(inputs["replica_config"], 7) is metadata
    _, plan = _reporting_case(tracing=True, operations=False, ledger=False)
    wave_batch = plan.phase_times.lane_records[0].batch
    plan.phase_times = plan.phase_times._replace(lane_records=(
        EPWaveLaneTiming(0, wave_batch, StageExecutionTime((step_layer,))),
    ))
    context_spy = Mock(wraps=ep_wave_metrics.OpTraceContext)
    monkeypatch.setattr(ep_wave_metrics, "OpTraceContext", context_spy)
    store._should_expand_layers = Mock(return_value=True)
    for _ in range(2):
        ep_wave_metrics.record_ep_wave(
            store, plan, time=1.0, replica_id=0, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        )
    assert counter.call_count == 1
    assert context_spy.call_count == 6
    assert all(call.kwargs["layer_metadata"] is metadata for call in context_spy.call_args_list)
    other = replace(inputs["replica_config"], attn_tensor_parallel_size=8, moe_tensor_parallel_size=8)
    assert store._get_step5_layer_metadata(other, 7) is not metadata
    assert counter.call_count == 2


def test_routed_trace_keeps_assignment_rows_inside_the_expert_owner(inputs):
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta

    replica = inputs["replica_config"]
    context = OpTraceContext(
        ClusterType.MONOLITHIC, replica.model_config, replica,
        1, 64, 64, 64, True, global_layer_id=3, pre_routing_tokens=8,
    )
    meta = compute_op_trace_meta("moe_grouped_gemm", "COMPUTE", context)
    assert meta["tensor_shape"] == {"input": [8, 4096], "output": [8, 4096]}
    assert meta["internal_tensor_shape"] == {
        "gate_up": [8, 8, 192], "activation": [64, 96], "down": [8, 8, 4096],
    }
    assert meta["covered_scopes"] == ["gate_up_gemm", "swiglu", "down_gemm", "topk_reduce"]
    assert meta["tensor_size_bytes"] == {"input": 65536, "output": 65536}


def test_routed_step_trace_requires_explicit_pre_routing_rows(inputs):
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta

    replica = inputs["replica_config"]
    context = OpTraceContext(
        ClusterType.MONOLITHIC, replica.model_config, replica,
        1, 64, 64, 64, True, global_layer_id=3,
    )
    with pytest.raises(ValueError, match="pre.routing"):
        compute_op_trace_meta("moe_grouped_gemm", "COMPUTE", context)


def test_assignment_alignment_has_no_materialized_hidden_output(inputs):
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta

    replica = inputs["replica_config"]
    context = OpTraceContext(
        ClusterType.MONOLITHIC, replica.model_config, replica,
        1, 8, 8, 8, False, global_layer_id=3,
    )
    meta = compute_op_trace_meta("moe_shuffling", "COMPUTE", context)
    assert meta["tensor_shape"] == {"expert_ids": [8, 8]}
    assert meta["tensor_dtype"] == {"expert_ids": "INT32"}
    assert meta["tensor_size_bytes"] == {"expert_ids": 256}
    assert meta["alignment_outputs"] == ["sorted_token_ids", "expert_ids", "num_tokens_post_padded"]
    assert meta["alignment_output_shapes"] == "unresolved native dispatch and alignment padding"
    assert meta["sorted_token_ids_nullable"] is True
