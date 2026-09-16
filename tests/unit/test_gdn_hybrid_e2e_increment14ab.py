"""CPU checkpoint for the supported hybrid GDN layer dispatch path.

The timing values in this test are synthetic CPU values.  The test exercises
the real GDN trainer and artifact loader, then uses deterministic hooks for
the unrelated FFN and communication operators so layer identity and stage
ordering can be checked without claiming GPU or benchmark parity.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention import bind_layer_attention
from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.config import (
    BaseModelConfig,
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.entities.time_components import (
    AttentionOperatorTimes,
    AttentionTime,
    CommunicationOperatorTimes,
    MLPOperatorTimes,
    MoEOperatorTimes,
)
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.execution_time_predictor import ExecutionTimePredictorRegistry
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.simulator import Simulator
from frontier.training.gdn_trainer import GDNTrainer
from frontier.types import ActivationType, ClusterType, MeasurementType, NormType


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "pr31_hybrid"
GDN_FIXTURE = FIXTURE_ROOT / "gdn.csv"


class _HybridCheckpointPredictor(SklearnMoEExecutionTimePredictor):
    """Abstract-method shim; all timing hooks are installed by the test."""

    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _qwen35_fixture_config() -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=8,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=256,
        mlp_hidden_dim=64,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        model_type="qwen3_5_moe_text",
        model_architecture_profile="qwen3_5_moe",
        architectures=("Qwen3_5MoeForCausalLM",),
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=4,
        dense_mlp_hidden_dim=64,
        routed_mlp_hidden_dim=64,
        torch_dtype="bfloat16",
        use_qk_norm=True,
    )


def _train_gdn(
    tmp_path: Path,
    *,
    device: str = "cpu",
    measurement_type: str = "DEVICE_EVENT",
) -> Path:
    dataset_path = tmp_path / "gdn.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        GDN_FIXTURE.read_text(encoding="utf-8")
        .replace("DEVICE_EVENT,qwen3_5_moe,none,cpu,", f"{measurement_type},qwen3_5_moe,none,{device},"),
        encoding="utf-8",
    )
    output = tmp_path / "gdn_models"
    GDNTrainer(
        str(dataset_path),
        str(output),
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
        device=device,
        tensor_parallel_size=1,
        measurement_type=measurement_type,
        num_estimators=[4],
        max_depth=[4],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
    ).train()
    return output


def _write_production_profile_fixture(
    root: Path,
    *,
    device: str = "a100",
    measurement_type: str = "CUDA_EVENT",
) -> tuple[Path, Path, Path]:
    """Write a minimal standard profile set for the real manager constructor."""

    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "profiling_precision": "BF16",
        "model_arch": "generic",
        "model_architecture_profile": "qwen3_5_moe",
        "quant_signature": "none",
        "measurement_type": measurement_type,
    }
    linear_rows = []
    for num_tokens in (1, 4, 16, 32):
        row = {
            **metadata,
            "num_tensor_parallel_workers": 1,
            "num_tokens": num_tokens,
            "n_embd": 256,
            "n_head": 4,
            "n_kv_head": 2,
            "n_expanded_embd": 64,
            "use_gated_mlp": True,
            "use_qk_norm": True,
            "vocab_size": 1024,
        }
        for op_name, value in {
            "input_layernorm": 0.01,
            "post_attention_layernorm": 0.02,
            "attn_pre_proj": 0.03,
            "attn_post_proj": 0.04,
            "attn_rope": 0.005,
            "mlp_up_proj": 0.06,
            "mlp_down_proj": 0.07,
            "mlp_act": 0.01,
        }.items():
            row[f"time_stats.{op_name}.median"] = value
        linear_rows.append(row)
    linear_path = root / "linear_op.csv"
    linear_fields = sorted({key for row in linear_rows for key in row})
    with linear_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=linear_fields)
        writer.writeheader()
        writer.writerows(linear_rows)

    attention_rows = []
    for prefill_chunk_size, is_prefill, kv_cache_size in (
        (16, True, 0),
        (32, True, 0),
        (0, False, 16),
        (0, False, 32),
    ):
        attention_rows.append(
            {
                **metadata,
                "n_embd": 256,
                "n_q_head": 4,
                "n_kv_head": 2,
                "block_size": 16,
                "num_tensor_parallel_workers": 1,
                "batch_size": 1,
                "prefill_chunk_size": prefill_chunk_size,
                "kv_cache_size": kv_cache_size,
                "total_tokens": max(prefill_chunk_size, kv_cache_size, 1),
                "is_prefill": is_prefill,
                "is_mixed_batch": False,
                "is_true_mixed_batch": False,
                "attention_backend": "VLLM_ROCM",
                "time_stats.attn_prefill.median": 0.08 if is_prefill else 0.0,
                "time_stats.attn_decode.median": 0.05 if not is_prefill else 0.0,
                "time_stats.attn_kv_cache_save.median": 0.01,
            }
        )
    attention_path = root / "attention.csv"
    attention_fields = sorted({key for row in attention_rows for key in row})
    with attention_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=attention_fields)
        writer.writeheader()
        writer.writerows(attention_rows)

    moe_rows = []
    for num_tokens in (1, 4, 16, 32):
        moe_rows.append(
            {
                **metadata,
                "num_experts": 8,
                "router_topk": 2,
                "hidden_dim": 256,
                "expert_hidden_dim": 64,
                "num_tensor_parallel_workers": 1,
                "expert_parallel_size": 1,
                "routing_runtime_path": "standard_fused_topk",
                "gating_runtime_context": "standalone_legacy",
                "num_tokens": num_tokens,
                "total_routed_tokens": num_tokens * 2,
                "num_experts_per_device": 8,
                "model_expansion_ratio": 0.25,
                "tokens_per_expert_avg": num_tokens / 4,
                "tokens_to_experts_ratio": num_tokens / 4,
                "expert_utilization": 1.0,
                "min_load_ratio": 1.0,
                "load_imbalance_cv": 0.0,
                "max_load_ratio": 1.0,
                "load_entropy": 1.0,
                "load_gini_coefficient": 0.0,
                "time_stats.moe_gating_linear.median": 0.03,
                "time_stats.moe_gating_routing_topk.median": 0.04,
                "time_stats.moe_shuffling.median": 0.05,
                "time_stats.moe_grouped_gemm.median": 0.12,
            }
        )
    moe_path = root / "moe.csv"
    moe_fields = sorted({key for row in moe_rows for key in row})
    with moe_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=moe_fields)
        writer.writeheader()
        writer.writerows(moe_rows)

    return linear_path, attention_path, moe_path


def _batch(*, phase: str) -> SimpleNamespace:
    if phase == "prefill":
        return SimpleNamespace(
            id=11,
            size=1,
            num_tokens=[16],
            total_num_tokens=16,
            num_prefill_tokens=16,
            num_decode_tokens=0,
            is_idle=False,
            requests=[
                SimpleNamespace(
                    num_processed_tokens=0,
                    num_prefill_tokens=16,
                    is_prefill_complete=False,
                )
            ],
        )
    return SimpleNamespace(
        id=12,
        size=1,
        num_tokens=[1],
        total_num_tokens=1,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        is_idle=False,
        requests=[
            SimpleNamespace(
                num_processed_tokens=128,
                num_prefill_tokens=0,
                is_prefill_complete=True,
            )
        ],
    )


def _build_predictor(config: BaseModelConfig, gdn: GDNPredictor):
    predictor = object.__new__(_HybridCheckpointPredictor)
    predictor._model_config = config
    predictor._gdn_predictor = gdn
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
    )
    predictor._num_layers_per_pipeline_stage = config.num_layers
    predictor._moe_routing_distribution_type = "synthetic"
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    predictor._log_architecture_attention_shape = lambda _batch: None
    predictor._supports_operation = lambda _operation: True
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    predictor._select_measurement_type_for_batch = lambda _batch: MeasurementType.DEVICE_EVENT
    predictor._get_moe_tokens_input = lambda _batch, layer_id: layer_id + 1
    predictor._admit_routed_ep_aggregate = lambda *_args, **kwargs: kwargs.get(
        "lane_workload"
    )
    for method_name in (
        "_get_attn_norm_layer_act_execution_time",
        "_get_attention_prefill_execution_time",
        "_get_attention_decode_execution_time",
        "_get_attention_kv_cache_save_execution_time",
        "_get_attention_layer_pre_proj_execution_time",
        "_get_attention_layer_post_proj_execution_time",
        "_get_attention_rope_execution_time",
    ):
        setattr(predictor, method_name, lambda _batch: 0.1)
    return predictor


def _execution_from_attention(
    predictor, batch, *, layer_id: int, include_moe: bool
) -> ExecutionTime:
    attention = predictor.predict_attention_layer_time(
        batch=batch,
        layer_id=layer_id,
        cluster_type=ClusterType.MONOLITHIC,
    )
    spec = bind_layer_attention(predictor._model_config, layer_id)
    common = dict(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=attention.attention_rope_execution_time,
        attention_kv_cache_save_execution_time=attention.attention_kv_cache_save_execution_time,
        attention_decode_execution_time=attention.attention_decode_execution_time,
        attention_prefill_execution_time=attention.attention_prefill_execution_time,
        attention_layer_pre_proj_execution_time=attention.attention_layer_pre_proj_execution_time,
        attention_layer_post_proj_execution_time=attention.attention_layer_post_proj_execution_time,
        attn_norm_time=attention.attn_norm_time,
        mlp_norm_time=0.2,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.3 if include_moe else 0.0,
        moe_gating_time=0.2 if include_moe else 0.0,
        moe_shuffling_time=0.1 if include_moe else 0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=include_moe,
        mlp_layer_up_proj_execution_time=0.0 if include_moe else 0.4,
        mlp_layer_down_proj_execution_time=0.0 if include_moe else 0.5,
        mlp_layer_act_execution_time=0.0 if include_moe else 0.2,
        attention_operator_times=attention.operator_times,
        global_layer_id=spec.global_layer_id,
        attention_family_id=spec.family_id,
        attention_variant_id=spec.variant_id,
    )
    if include_moe:
        common["moe_grouped_gemm_time"] = 0.6
        common["moe_operator_times"] = MoEOperatorTimes(
            op_times={
                "post_attention_layernorm": 0.2,
                "moe_gating_linear": 0.1,
                "moe_gating_routing_topk": 0.1,
                "moe_shuffling": 0.1,
                "moe_grouped_gemm": 0.6,
            }
        )
    else:
        common["mlp_operator_times"] = MLPOperatorTimes(
            op_times={
                "post_attention_layernorm": 0.2,
                "mlp_up_proj": 0.4,
                "mlp_act": 0.2,
                "mlp_down_proj": 0.5,
            }
        )
    return ExecutionTime(**common)


def test_gdn_training_and_phase_dispatch_are_cpu_safe(tmp_path: Path) -> None:
    output = _train_gdn(tmp_path)
    predictor = GDNPredictor.from_directory(
        output,
        device="cpu",
        tensor_parallel_size=1,
        measurement_type="DEVICE_EVENT",
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
    )
    config = _qwen35_fixture_config()
    runtime = _build_predictor(config, predictor)

    prefill = runtime.predict_attention_layer_time(
        _batch(phase="prefill"), 0, ClusterType.MONOLITHIC
    )
    decode = runtime.predict_attention_layer_time(
        _batch(phase="decode"), 0, ClusterType.MONOLITHIC
    )
    continuation = GDNBatchFeatures.from_batch(
        SimpleNamespace(
            num_tokens=[1],
            total_num_tokens=1,
            num_prefill_tokens=1,
            num_decode_tokens=0,
            requests=[
                SimpleNamespace(
                    num_processed_tokens=128,
                    num_prefill_tokens=1,
                    is_prefill_complete=False,
                )
            ],
        )
    )

    assert prefill.operator_times is not None
    assert prefill.operator_times.op_times["gdn_core_prefill"] == pytest.approx(0.22)
    assert prefill.operator_times.op_times["gdn_core_decode"] == 0.0
    assert decode.operator_times is not None
    assert decode.operator_times.op_times["gdn_core_decode"] == pytest.approx(0.05)
    assert decode.operator_times.op_times["gdn_core_prefill"] == 0.0
    assert continuation.phase == "prefill"
    assert "gdn_layer_e2e" not in prefill.operator_times.op_times


def test_fresh_model_manager_loads_hybrid_gdn_artifact(tmp_path: Path) -> None:
    output = _train_gdn(tmp_path)
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cache_dir = str(output)
    manager._gdn_predictors = {}
    replica_config = SimpleNamespace(
        model_config=_qwen35_fixture_config(),
        device="cpu",
        attn_tensor_parallel_size=1,
    )
    predictor_config = SimpleNamespace(gdn_input_file=str(GDN_FIXTURE))

    manager._load_gdn_predictor_for_cluster(
        ClusterType.MONOLITHIC,
        replica_config,
        predictor_config,
        MeasurementType.DEVICE_EVENT,
    )

    loaded = manager.get_gdn_predictor(ClusterType.MONOLITHIC)
    assert loaded is not None
    assert loaded.identity["measurement_type"] == "DEVICE_EVENT"
    assert loaded.identity["runtime_stack_signature"] == "synthetic_cpu_v1"


def test_hybrid_stage_keeps_layer_order_and_identity(tmp_path: Path) -> None:
    gdn = GDNPredictor.from_directory(_train_gdn(tmp_path))
    config = _qwen35_fixture_config()
    predictor = _build_predictor(config, gdn)

    def fake_internal(*args, batch=None, layer_id, include_moe, **_kwargs):
        if batch is None:
            batch = args[0]
        return _execution_from_attention(
            predictor,
            batch,
            layer_id=layer_id,
            include_moe=bool(include_moe),
        )

    predictor._get_execution_time_internal = fake_internal
    result = predictor.predict_stage_execution_time(
        batch=_batch(phase="prefill"),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=8,
        layer_id=0,
    )

    assert isinstance(result, StageExecutionTime)
    assert result.global_layer_ids == tuple(range(8))
    assert result.attention_family_ids == (
        "gated_delta_net",
        "gated_delta_net",
        "gated_delta_net",
        "dense_attention",
        "gated_delta_net",
        "gated_delta_net",
        "gated_delta_net",
        "dense_attention",
    )
    # Qwen3.5 keeps the routed/shared-expert FFN on both attention families;
    # the G/G/G/A schedule classifies the attention side only.
    assert all(layer._is_moe for layer in result.layer_execution_times)
    assert result.model_time_ms == pytest.approx(
        sum(layer.get_single_layer_block_time() for layer in result.layer_execution_times)
    )


def test_hybrid_gdn_real_simulator_cpu_e2e(tmp_path: Path, monkeypatch) -> None:
    """Run one request through the real CPU Simulator and metrics pipeline.

    GDN training, artifact loading, Replica construction, MemoryPlanner setup,
    scheduler/event processing, and metrics writing are production paths.  The
    unrelated dense/MoE operator timings use deterministic test values because
    this synthetic fixture has no production attention/MoE profiling CSVs.
    """

    model = _qwen35_fixture_config()
    model._model_name = "pr31_hybrid_fixture"
    gdn_output = _train_gdn(tmp_path / "gdn")
    gdn = GDNPredictor.from_directory(
        gdn_output,
        device="cpu",
        tensor_parallel_size=1,
        measurement_type="DEVICE_EVENT",
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
    )

    original_create_from_name = BaseModelConfig.create_from_name

    def create_fixture_model(cls, name):
        if name == model._model_name:
            return model
        return original_create_from_name(name)

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(create_fixture_model),
    )

    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        total_expert_num=8,
        router_topk=2,
        memory_margin_fraction=0.1,
    )
    scheduler = VllmV1SchedulerConfig(
        num_blocks=100,
        block_size=16,
        batch_size_cap=4,
        max_tokens_in_batch=64,
    )
    predictor_config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False,
        gdn_input_file=str(GDN_FIXTURE),
        linear_op_input_file="unused",
        atten_input_file="unused",
        moe_input_file="unused",
    )
    cluster = ClusterConfig(
        cluster_type=ClusterType.MONOLITHIC,
        num_replicas=1,
        replica_config=replica,
        replica_scheduler_config=scheduler,
        execution_time_predictor_config=predictor_config,
    )
    request_generator = SyntheticRequestGeneratorConfig(
        num_requests=1,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=16,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    output_root = tmp_path / "sim_metrics"
    metrics = MetricsConfig(
        output_dir=str(output_root),
        run_id="hybrid_sim",
        write_metrics=True,
        store_request_metrics=True,
        store_batch_metrics=True,
        store_operation_metrics=True,
        keep_individual_batch_metrics=True,
        store_utilization_metrics=True,
        store_frontier_stage_batch_ledger=True,
        store_frontier_stage_batch_ledger_summary=True,
        store_plots=False,
        enable_chrome_trace=False,
        write_json_trace=False,
        enable_op_level_tracing=True,
        trace_output_file="op_traces.jsonl",
        enable_per_layer_expansion=True,
        num_requests_to_trace_per_layer=1,
    )
    config = SimulationConfig(
        simulation_mode="offline",
        sys_arch="co-location",
        cluster_config=cluster,
        request_generator_config=request_generator,
        metrics_config=metrics,
    )

    predictor = _build_predictor(model, gdn)
    predictor._replica_config = replica
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = model.num_layers
    predictor._monolithic_routing_details = {
        request_id: {
            layer_id: {expert_id: 1.0 / 8.0 for expert_id in range(8)}
            for layer_id in range(model.num_layers)
        }
        for request_id in range(32)
    }

    def fake_internal(
        *args,
        batch=None,
        include_moe=True,
        include_ffn=True,
        include_attention=True,
        layer_id=0,
        **_kwargs,
    ):
        if batch is None:
            batch = args[0]
        spec = bind_layer_attention(model, layer_id)
        if include_attention:
            attention = predictor.predict_attention_layer_time(
                batch=batch,
                layer_id=layer_id,
                cluster_type=ClusterType.MONOLITHIC,
            )
        elif spec.family_id == "gated_delta_net":
            attention = AttentionTime(
                operator_times=AttentionOperatorTimes(
                    {
                        "gdn_input_projections": 0.0,
                        "gdn_core_prefill": 0.0,
                        "gdn_core_decode": 0.0,
                        "gdn_output_projection": 0.0,
                    }
                )
            )
        else:
            attention = AttentionTime()

        is_moe = bool(include_moe and include_ffn)
        return ExecutionTime(
            num_layers_per_pipeline_stage=1,
            attention_rope_execution_time=attention.attention_rope_execution_time,
            attention_kv_cache_save_execution_time=attention.attention_kv_cache_save_execution_time,
            attention_decode_execution_time=attention.attention_decode_execution_time,
            attention_prefill_execution_time=attention.attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time=attention.attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=attention.attention_layer_post_proj_execution_time,
            attn_norm_time=attention.attn_norm_time,
            mlp_norm_time=0.2 if include_ffn else 0.0,
            add_time=0.05 if include_ffn else 0.0,
            tensor_parallel_communication_time=0.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            moe_gating_time=0.3 if is_moe else 0.0,
            moe_shuffling_time=0.1 if is_moe else 0.0,
            schedule_time=0.0,
            sampler_e2e_time=0.0,
            prepare_inputs_e2e_time=0.0,
            process_model_outputs_time=0.0,
            ray_comm_time=0.0,
            is_moe=is_moe,
            mlp_layer_up_proj_execution_time=0.0 if is_moe else 0.4,
            mlp_layer_down_proj_execution_time=0.0 if is_moe else 0.5,
            mlp_layer_act_execution_time=0.0 if is_moe else 0.2,
            moe_grouped_gemm_time=0.6 if is_moe else 0.0,
            moe_gating_linear_time=0.15 if is_moe else 0.0,
            moe_gating_routing_topk_time=0.15 if is_moe else 0.0,
            communication_operator_times=CommunicationOperatorTimes(
                {
                    "expert_parallel_alltoall_dispatch": 0.0,
                    "expert_parallel_alltoall_combine": 0.0,
                }
            ),
            attention_operator_times=attention.operator_times,
            global_layer_id=spec.global_layer_id,
            attention_family_id=spec.family_id,
            attention_variant_id=spec.variant_id,
            mlp_operator_times=(
                MLPOperatorTimes(
                    op_times={
                        "post_attention_layernorm": 0.2,
                        "mlp_up_proj": 0.4,
                        "mlp_act": 0.2,
                        "mlp_down_proj": 0.5,
                    }
                )
                if not is_moe
                else None
            ),
            moe_operator_times=(
                MoEOperatorTimes(
                    op_times={
                        "post_attention_layernorm": 0.2,
                        "moe_gating_linear": 0.15,
                        "moe_gating_routing_topk": 0.15,
                        "moe_shuffling": 0.1,
                        "moe_grouped_gemm": 0.6,
                    }
                )
                if is_moe
                else None
            ),
        )

    predictor._get_execution_time_internal = fake_internal
    monkeypatch.setattr(
        ExecutionTimePredictorRegistry,
        "get",
        classmethod(lambda _cls, *_args, **_kwargs: predictor),
    )
    prepared_manager = SimpleNamespace(
        get_training_file_paths=lambda _cluster_type: {},
    )
    monkeypatch.setattr(
        "frontier.simulator.ExecutionTimePredictionModelManager",
        lambda *_args, **_kwargs: prepared_manager,
    )

    Simulator(config).run()

    final_dir = output_root / model._model_name / "offline_batch" / "hybrid_sim"
    request_metrics = final_dir / "request_metrics.csv"
    system_metrics = final_dir / "system_metrics.json"
    stage_ledger = final_dir / "frontier_stage_batch_ledger.jsonl"
    op_trace = final_dir / "op_traces.jsonl"
    for artifact in (request_metrics, system_metrics, stage_ledger, op_trace):
        assert artifact.is_file(), artifact

    import csv
    import json

    with request_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["request_num_tokens"]) == 18
    assert float(rows[0]["request_e2e_time"]) > 0.0
    summary = json.loads(system_metrics.read_text(encoding="utf-8"))
    assert summary["simulation_metadata"]["total_requests"] == 1
    assert summary["simulation_metadata"]["completed_requests"] == 1

    ledger_rows = [
        json.loads(line)
        for line in stage_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 2
    assert {row["operation_kind"] for row in ledger_rows} == {"ep_ffn"}
    assert all(row["stage_end_ts"] >= row["stage_start_ts"] for row in ledger_rows)
    assert [row["request_num_prefill_tokens"] for row in ledger_rows] == [[16], [0]]
    assert ledger_rows[1]["stage_start_ts"] == pytest.approx(
        ledger_rows[0]["stage_end_ts"]
    )


@pytest.mark.parametrize("num_requests", [1, 3])
def test_hybrid_gdn_production_constructor_cpu_e2e(
    tmp_path: Path, monkeypatch, num_requests: int
) -> None:
    """Exercise the normal co-location manager/predictor construction path."""

    model = _qwen35_fixture_config()
    model._model_name = "pr31_hybrid_production_fixture"
    gdn_root = tmp_path / "gdn"
    gdn_output = _train_gdn(
        gdn_root,
        device="a100",
        measurement_type="CUDA_EVENT",
    )
    linear_path, attention_path, moe_path = _write_production_profile_fixture(
        tmp_path / "profiles"
    )

    original_create_from_name = BaseModelConfig.create_from_name

    def create_fixture_model(cls, name):
        if name == model._model_name:
            return model
        return original_create_from_name(name)

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(create_fixture_model),
    )

    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        total_expert_num=8,
        router_topk=2,
        memory_margin_fraction=0.1,
    )
    scheduler = VllmV1SchedulerConfig(
        num_blocks=0,
        block_size=16,
        batch_size_cap=1,
        max_tokens_in_batch=4,
        enable_chunked_prefill=True,
    )
    predictor_config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False,
        gdn_input_file=str(gdn_root / "gdn.csv"),
        linear_op_input_file=str(linear_path),
        atten_input_file=str(attention_path),
        moe_input_file=str(moe_path),
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
        prediction_max_tokens_per_request=64,
        prediction_max_prefill_chunk_size=32,
        prediction_max_batch_size=4,
        kv_cache_prediction_granularity=16,
        skip_cpu_overhead_modeling=True,
    )
    cluster = ClusterConfig(
        cluster_type=ClusterType.MONOLITHIC,
        num_replicas=1,
        replica_config=replica,
        replica_scheduler_config=scheduler,
        execution_time_predictor_config=predictor_config,
    )
    request_generator = SyntheticRequestGeneratorConfig(
        num_requests=num_requests,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=16,
            decode_tokens=2,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1_000_000.0),
    )
    output_root = tmp_path / "sim_metrics"
    metrics = MetricsConfig(
        output_dir=str(output_root),
        cache_dir=str(gdn_output),
        run_id="hybrid_production_sim",
        write_metrics=True,
        store_request_metrics=True,
        store_batch_metrics=True,
        store_operation_metrics=True,
        keep_individual_batch_metrics=True,
        store_utilization_metrics=True,
        store_frontier_stage_batch_ledger=True,
        store_frontier_stage_batch_ledger_summary=True,
        store_plots=False,
        enable_chrome_trace=False,
        write_json_trace=False,
        enable_op_level_tracing=True,
        trace_output_file="op_traces.jsonl",
        enable_per_layer_expansion=True,
        num_requests_to_trace_per_layer=1,
    )
    config = SimulationConfig(
        simulation_mode="online" if num_requests > 1 else "offline",
        sys_arch="co-location",
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        request_generator_config=request_generator,
        metrics_config=metrics,
    )

    simulator = Simulator(config)
    manager = simulator._execution_time_prediction_model_manager
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    assert manager is not None
    assert predictor._model_manager is manager
    assert predictor._gdn_predictor is not None
    assert predictor._gdn_predictor.identity["measurement_type"] == "CUDA_EVENT"
    assert predictor._models_eager
    assert set(predictor._monolithic_routing_details) == set(
        simulator._clusters[ClusterType.MONOLITHIC].replicas
    )
    cluster_scheduler = simulator._global_scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
    replica_id = next(iter(simulator._clusters[ClusterType.MONOLITHIC].replicas))
    replica_scheduler = cluster_scheduler.get_replica_scheduler(replica_id, 0)
    slots = replica_scheduler._gdn_state_slot_manager
    assert slots.capacity == 1
    assert replica_scheduler._config.num_blocks > 0
    admissions = []
    continuations = []
    finishes = []
    exhausted_waiters = []
    original_allocate = replica_scheduler._allocate_request
    original_batch_end = replica_scheduler.on_batch_end

    def observe_allocate(request, *args, **kwargs):
        previous = slots.retain(request.id) if slots.has_slot(request.id) else None
        result = original_allocate(request, *args, **kwargs)
        current = slots.retain(request.id)
        if previous is None:
            admissions.append((request.id, current.slot_id))
        else:
            assert current == previous
            continuations.append(request.id)
        return result

    def observe_batch_end(batch):
        prior_owners = {request_id: slots.retain(request_id) for request_id in slots.active_request_ids}
        original_batch_end(batch)
        for request in batch.requests:
            if request.completed:
                assert not slots.has_slot(request.id)
                assert request.id not in replica_scheduler._allocation_map
                finishes.append(request.id)
            else:
                assert slots.retain(request.id) == prior_owners[request.id]
        waiting = replica_scheduler.peek_waiting_requests()
        if waiting and len(slots.active_request_ids) == slots.capacity:
            exhausted_waiters.extend(request.id for request in waiting)
            assert all(not slots.has_slot(request.id) for request in waiting)
            assert all(request.id not in replica_scheduler._allocation_map for request in waiting)

    monkeypatch.setattr(replica_scheduler, "_allocate_request", observe_allocate)
    monkeypatch.setattr(replica_scheduler, "on_batch_end", observe_batch_end)
    simulator.run()
    assert len(admissions) == num_requests
    assert [slot_id for _, slot_id in admissions] == [0] * num_requests
    assert finishes == [request_id for request_id, _ in admissions]
    assert set(continuations) == set(finishes)
    assert slots.active_request_ids == ()
    assert replica_scheduler._allocation_map == {}
    assert replica_scheduler.num_allocated_blocks == 0
    assert replica_scheduler.peek_waiting_requests() == []
    assert replica_scheduler._running_requests == []
    assert replica_scheduler._num_running_batches == 0
    if num_requests > 1:
        assert set(exhausted_waiters) == set(finishes[1:])

    mode_dir = "online_serving" if num_requests > 1 else "offline_batch"
    final_dir = output_root / model._model_name / mode_dir / "hybrid_production_sim"
    request_metrics = final_dir / "request_metrics.csv"
    system_metrics = final_dir / "system_metrics.json"
    op_trace = final_dir / "op_traces.jsonl"
    assert request_metrics.is_file()
    assert system_metrics.is_file()
    assert op_trace.is_file()

    import json

    with request_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == num_requests
    assert all(int(row["request_num_tokens"]) == 18 for row in rows)
    assert all(float(row["request_e2e_time"]) > 0 for row in rows)
    summary = json.loads(system_metrics.read_text(encoding="utf-8"))
    assert summary["simulation_metadata"]["total_requests"] == num_requests
    assert summary["simulation_metadata"]["completed_requests"] == num_requests

    trace_events = [
        json.loads(line)
        for line in op_trace.read_text(encoding="utf-8").splitlines()
        if line.strip() and '"name"' in line
    ]
    gdn_events = [event for event in trace_events if event["name"].startswith("gdn_")]
    dense_events = [
        event
        for event in trace_events
        if event["name"] in {"attn_pre_proj", "attn_decode"}
        and event.get("layer_id", -1) >= 0
    ]
    assert gdn_events
    assert dense_events
    assert {event["meta"]["attention_family_id"] for event in gdn_events} == {
        "gated_delta_net"
    }
    assert {event["meta"].get("attention_family_id") for event in dense_events} == {
        "dense_attention"
    }
    assert {event["layer_id"] for event in gdn_events} == {0, 1, 2, 4, 5, 6}
    assert {event["layer_id"] for event in dense_events} == {3, 7}
