"""Public predictor acceptance using labelled, unadmitted CPU measurement fixtures."""

import math
from types import SimpleNamespace

import pandas as pd

from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.config.quantization_manager import QuantizationManager
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType

import test_step5_measured_ffn as ffn_fixture
from test_step5_measured_training import _row, _write


def test_normal_public_constructor_and_dense_ffn_stage(tmp_path, monkeypatch):
    """Train all categories normally, construct the public MoE predictor, replay FFN."""
    for name, value in (
        ("SYS_ARCH", "co-location"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "full"),
        ("KV_CONNECTOR", "none"), ("QUANTIZATION_MANAGER", None),
        ("CUDAGRAPH_CAPTURE_SIZES", None), ("IS_MOE", None),
        ("_IS_MOE_INITIALIZED", False),
    ):
        monkeypatch.setattr(global_vars, name, value)
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)

    replica = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=1,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=8,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    replica.cluster_num_replicas = 1
    scheduler = VllmV1SchedulerConfig(
        block_size=16, max_model_len=8192, batch_size_cap=1,
        max_tokens_in_batch=32, long_prefill_token_threshold=32,
        enable_chunked_prefill=True,
    )
    attention_dir, compute_dir = tmp_path / "attention", tmp_path / "compute"
    attention_dir.mkdir()
    compute_dir.mkdir()
    attention = _write(attention_dir, [
        _row(replica.model_config, layer, context=8176) for layer in range(92)
    ])
    metadata = ffn_fixture._metadata
    with monkeypatch.context() as context:
        # Generate event-labelled CPU fixtures; never relabel existing measurements.
        context.setattr(ffn_fixture, "_metadata", lambda model: {
            **metadata(model), "measurement_type": MeasurementType.CUDA_EVENT.value,
        })
        linear, moe = ffn_fixture._tables(
            compute_dir, replica.model_config, token_counts=(1, 8, 16, 32),
            include_stage_boundaries=True,
        )
    gap = tmp_path / "kernel_gap.csv"
    pd.DataFrame({
        "execution_mode": ["eager", "cuda_graph"],
        "num_tensor_parallel_workers": [8, 8], "kernel_gap_us": [1., 2.],
    }).to_csv(gap, index=False)
    config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        linear_op_input_file=str(linear), moe_input_file=str(moe),
        atten_input_file=str(tmp_path / "unadmitted_event_attention.csv"),
        atten_kernel_only_input_file=str(attention), kernel_gap_input_file=str(gap),
        prediction_max_tokens_per_request=32, prediction_max_batch_size=1,
        prediction_max_prefill_chunk_size=32,
        num_estimators=[2], max_depth=[3], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )
    metrics = MetricsConfig(
        output_dir=str(tmp_path / "metrics"), cache_dir=str(tmp_path / "cache"),
        write_metrics=False, enable_chrome_trace=False,
    )
    cluster = ClusterConfig(
        replica_config=replica, replica_scheduler_config=scheduler,
        execution_time_predictor_config=config,
        cc_backend_config=CollectiveSimCCBackendConfig(), cluster_type=ClusterType.PREFILL,
    )
    manager = ExecutionTimePredictionModelManager({ClusterType.PREFILL: cluster}, metrics)
    assert set(manager._step5_model_bundles[ClusterType.PREFILL]) == {"eager", "kernel_only"}

    predictor = RandomForrestExecutionTimePredictor(
        predictor_config=config, replica_config=replica,
        replica_scheduler_config=scheduler, metrics_config=metrics,
        model_manager=manager, cluster_type=ClusterType.PREFILL, actual_replica_ids=[0],
        # Communication is explicitly isolated; this is not a communication profile.
        cc_backend=SimpleNamespace(predict_allreduce=lambda **kwargs: 0.),
    )
    assert isinstance(predictor, SklearnMoEExecutionTimePredictor)
    assert not predictor._enable_dummy_mode
    request = Request(arrived_at=0., num_prefill_tokens=8192, num_decode_tokens=1,
                      num_processed_tokens=8176)
    batch = Batch(replica_id=0, requests=[request], num_tokens=[16], is_moe=True,
                  request_is_decoding=[False], num_context_tokens=[8176])
    stage = predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=ClusterType.PREFILL,
        layer_id=1, num_layers=1, include_attention=False, include_moe=False,
    )
    assert stage.global_layer_ids == (1,)
    layer = stage.layer_execution_times[0]
    assert math.isfinite(layer.get_single_layer_post_attention_time())
    assert layer.get_single_layer_post_attention_time() > 0
    assert layer._step5_ffn_operator_costs["mlp_up_proj"]["precision"] == "BF16"
    assert layer._step5_ffn_operator_costs["mlp_up_proj"]["evidence_kind"] == "cpu_fixture"
    first = predictor.predict_stage_execution_time(
        batch, 0, ClusterType.PREFILL, layer_id=0, include_moe=False,
    )
    assert first.input_cast_time == first._step5_stage_operator_costs["input_cast"]["duration_ms"]
    assert first.input_cast_time > 0
    last = predictor.predict_stage_execution_time(
        batch, 0, ClusterType.PREFILL, layer_id=91, include_ffn=False,
    )
    assert last.final_residual_time == last._step5_stage_operator_costs["final_residual"]["duration_ms"]
    assert last.final_residual_time > 0
