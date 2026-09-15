"""Runtime cache/layout family selection for hybrid attention models."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.model_binding import bind_attention_family
from frontier.config.model_config import BaseModelConfig
from frontier.entities.replica import Replica
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ActivationType, ClusterType, NormType


def _base_hybrid_config() -> BaseModelConfig:
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


def test_hybrid_runtime_family_uses_full_attention_for_cache_metadata() -> None:
    config = _base_hybrid_config()

    assert config.get_attention_family().family_id == "dense_attention"
    assert config.get_runtime_num_kv_heads() == 2
    assert config.get_runtime_head_size() == 64
    assert config.get_qk_head_dim() == 64

    try:
        bind_attention_family(config)
    except ValueError as exc:
        assert "explicit global layer id" in str(exc)
    else:  # pragma: no cover - this is a guard for the homogeneous-only API.
        raise AssertionError("hybrid whole-model binding must remain rejected")


def test_malformed_qwen35_schedule_error_propagates_from_whole_model_binder() -> None:
    config = _base_hybrid_config()
    config.full_attention_interval = 0

    with pytest.raises(ValueError, match="full_attention_interval must be positive"):
        bind_attention_family(config)


def test_real_sklearn_predictor_constructor_registers_hybrid_attention_metadata(
    monkeypatch, tmp_path
) -> None:
    from frontier.config import (
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        VllmV1SchedulerConfig,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    class ModelManager:
        def get_models_for_cluster(self, _cluster_type):
            return {"eager": {}, "kernel_only": {}}

        def get_gdn_predictor(self, _cluster_type):
            return None

    # Keep the constructor and metadata-family selection real while avoiding
    # unrelated profiling-file and PP lookup I/O in this CPU test.
    monkeypatch.setattr(
        Predictor,
        "_register_profiling_metadata_from_file",
        lambda self, _path, operation_names: setattr(
            self, "_registered_hybrid_names", tuple(operation_names)
        ),
    )
    for method_name in (
        "_initialize_pp_stage_boundary_lookup",
        "_initialize_pp_receiver_head_lookup",
        "_initialize_pp_producer_send_path_lookup",
        "_initialize_pp_prefill_consumer_active_lookup",
    ):
        monkeypatch.setattr(Predictor, method_name, lambda self: None)
    monkeypatch.setattr(
        Predictor,
        "_predict_from_models_for_family",
        lambda self, _measurement_type, _models: {},
    )

    replica_config = ReplicaConfig(
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    predictor = Predictor(
        predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=False
        ),
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(),
        metrics_config=MetricsConfig(output_dir=str(tmp_path / "metrics")),
        model_manager=ModelManager(),
        cluster_type=ClusterType.MONOLITHIC,
        training_file_paths=None,
    )

    assert "attn_prefill" in predictor._registered_hybrid_names
    assert predictor._get_attention_family().family_id == "dense_attention"


def test_hybrid_replica_head_properties_are_constructible() -> None:
    model_config = _base_hybrid_config()
    replica_config = SimpleNamespace(
        model_config=model_config,
        device_config=SimpleNamespace(total_memory_gb=80, fp16_tflops=1.0),
        num_pipeline_stages=1,
        attn_tensor_parallel_size=2,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        memory_margin_fraction=0.1,
    )
    generator_config = SimpleNamespace(max_tokens=128)

    replica = Replica(replica_config, generator_config, ClusterType.MONOLITHIC)

    assert replica.attention_head_dim == 64
    assert replica.q_heads_per_tensor_parallel_worker == 2
    assert replica.kv_heads_per_tensor_parallel_worker == 1


def test_hybrid_memory_planner_uses_full_attention_kv_layout() -> None:
    from frontier.config import ReplicaConfig
    from frontier.scheduler.utils.memory_planner import MemoryPlanner

    replica_config = ReplicaConfig(
        model_name="Qwen3.8-2.4T-A95B-Quark-MXFP4",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    replica = Replica(
        replica_config,
        SimpleNamespace(max_tokens=128),
        ClusterType.MONOLITHIC,
    )
    planner = MemoryPlanner(replica_config, replica, ClusterType.MONOLITHIC)

    # Qwen3.8 has 92 layers with one full-attention layer every four layers.
    # The full-attention layer keeps two KV heads per TP worker and a 256-wide
    # head; the layout includes both key and value tensors.
    assert planner._get_kv_cache_elements_per_token_per_worker() == 1024
    assert planner._get_num_full_attention_layers_per_device() == 23
    assert planner._get_num_gdn_layers_per_device() == 69


def test_profiling_model_config_uses_the_same_runtime_policy() -> None:
    config = ModelConfig(
        name="qwen3_5_hybrid_unit",
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

    assert config.get_attention_family().family_id == "dense_attention"
    assert config.get_runtime_num_kv_heads() == 2
    assert config.get_runtime_head_size() == 64
