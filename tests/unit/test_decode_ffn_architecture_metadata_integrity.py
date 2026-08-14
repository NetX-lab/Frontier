from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.model_binding import bind_attention_family
from frontier.config.model_config import BaseModelConfig
from frontier.entities.replica import Replica
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ActivationType, ClusterType, NormType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("Estimator construction is outside this unit-test seam")


def _model_config() -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=2,
        num_q_heads=16,
        num_kv_heads=4,
        embedding_dim=2048,
        mlp_hidden_dim=4096,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=True,
        num_experts=16,
        num_experts_per_tok=2,
        model_type="unit_moe",
        use_mla=False,
    )


def _replica_config(model_config: BaseModelConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config,
        device_config=SimpleNamespace(total_memory_gb=80, fp16_tflops=1.0),
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        total_expert_num=16,
        local_expert_num=16,
        router_topk=2,
        router_load_balancing_type="random",
        memory_margin_fraction=0.1,
    )


def _replica(cluster_type: ClusterType) -> tuple[Replica, SimpleNamespace]:
    replica_config = _replica_config(_model_config())
    generator_config = SimpleNamespace(max_tokens=4096)
    return Replica(replica_config, generator_config, cluster_type), replica_config


def _metadata_predictor(
    cluster_type: ClusterType | None,
) -> SklearnExecutionTimePredictor:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._cluster_type = cluster_type
    predictor._compute_input_file = "compute.csv"
    predictor._attention_input_file = "attention.csv"
    predictor._get_compute_model_names = lambda: ["compute_op"]
    predictor._get_attention_model_names = lambda: ["attention_op"]
    return predictor


def test_decode_ffn_replica_preserves_architecture_heads() -> None:
    replica, replica_config = _replica(ClusterType.DECODE_FFN)

    assert replica_config.model_config.num_q_heads == 16
    assert replica_config.model_config.num_kv_heads == 4
    assert replica.to_dict()["num_q_heads"] == 0
    assert replica.to_dict()["num_kv_heads"] == 0


def test_decode_ffn_replica_reports_attention_as_not_applicable() -> None:
    replica, _ = _replica(ClusterType.DECODE_FFN)

    assert replica.num_q_heads == 0
    assert replica.num_kv_heads == 0
    assert replica.attention_head_dim == 0
    assert replica.q_heads_per_tensor_parallel_worker == 0
    assert replica.kv_heads_per_tensor_parallel_worker == 0


@pytest.mark.parametrize(
    "cluster_type",
    [
        ClusterType.PREFILL,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE,
        ClusterType.MONOLITHIC,
    ],
)
def test_attention_replica_preserves_and_reports_architecture_heads(
    cluster_type: ClusterType,
) -> None:
    replica, replica_config = _replica(cluster_type)

    assert replica_config.model_config.num_q_heads == 16
    assert replica_config.model_config.num_kv_heads == 4
    assert replica.num_q_heads == 16
    assert replica.num_kv_heads == 4
    assert replica.attention_head_dim == 128
    assert replica.q_heads_per_tensor_parallel_worker == 16
    assert replica.kv_heads_per_tensor_parallel_worker == 4


def test_decode_ffn_metadata_registration_skips_attention_and_keeps_ffn_metadata() -> None:
    replica, replica_config = _replica(ClusterType.DECODE_FFN)
    predictor = _metadata_predictor(ClusterType.DECODE_FFN)
    calls: list[tuple[str, str, tuple[str, ...]] | tuple[str]] = []

    def register(file_path: str, operation_names: list[str]) -> None:
        calls.append(("register", file_path, tuple(operation_names)))

    def bind_attention_names() -> list[str]:
        binding = bind_attention_family(replica_config.model_config)
        return list(binding.operation_names)

    predictor._register_profiling_metadata_from_file = register
    predictor._get_attention_model_names = bind_attention_names
    predictor._register_additional_profiling_metadata_from_files = lambda: calls.append(
        ("additional",)
    )

    predictor._register_profiling_metadata_from_files()

    assert replica.num_q_heads == 0
    assert calls == [
        ("register", "compute.csv", ("compute_op",)),
        ("additional",),
    ]


@pytest.mark.parametrize(
    "cluster_type",
    [
        ClusterType.PREFILL,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE,
        ClusterType.MONOLITHIC,
        None,
    ],
)
def test_attention_metadata_registration_remains_enabled(
    cluster_type: ClusterType | None,
) -> None:
    predictor = _metadata_predictor(cluster_type)
    calls: list[tuple[str, str, tuple[str, ...]] | tuple[str]] = []

    predictor._register_profiling_metadata_from_file = (
        lambda file_path, operation_names: calls.append(
            ("register", file_path, tuple(operation_names))
        )
    )
    predictor._register_additional_profiling_metadata_from_files = lambda: calls.append(
        ("additional",)
    )

    predictor._register_profiling_metadata_from_files()

    assert calls == [
        ("register", "compute.csv", ("compute_op",)),
        ("register", "attention.csv", ("attention_op",)),
        ("additional",),
    ]


def test_attention_family_binding_keeps_zero_head_fail_fast_contract() -> None:
    model_config = _model_config()
    model_config.num_q_heads = 0
    model_config.num_kv_heads = 0

    with pytest.raises(ValueError, match="Attention head counts must be positive"):
        bind_attention_family(model_config)
