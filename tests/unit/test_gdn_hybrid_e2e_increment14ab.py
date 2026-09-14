"""CPU checkpoint for the supported hybrid GDN layer dispatch path.

The timing values in this test are synthetic CPU values.  The test exercises
the real GDN trainer and artifact loader, then uses deterministic hooks for
the unrelated FFN and communication operators so layer identity and stage
ordering can be checked without claiming GPU or benchmark parity.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention import bind_layer_attention
from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.config.model_config import BaseModelConfig
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.entities.time_components import MLPOperatorTimes, MoEOperatorTimes
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
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
    )


def _train_gdn(tmp_path: Path) -> Path:
    output = tmp_path / "gdn_models"
    GDNTrainer(
        str(GDN_FIXTURE),
        str(output),
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
        device="cpu",
        tensor_parallel_size=1,
        measurement_type="DEVICE_EVENT",
        num_estimators=[4],
        max_depth=[4],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
    ).train()
    return output


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
    assert decode.operator_times is not None
    assert decode.operator_times.op_times["gdn_core_decode"] == pytest.approx(0.05)
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
