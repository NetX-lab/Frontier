from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ActivationType, NormType


class _ConcreteSklearnMoEExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def _step3_like_moe_config() -> BaseModelConfig:
    """Build a mixed-layer config whose routed width differs from legacy width."""

    return BaseModelConfig(
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=1,
        embedding_dim=1024,
        mlp_hidden_dim=9999,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum="1,2",
        share_expert_dim=5120,
        share_q_dim=1024,
        head_dim=128,
        use_mfa=True,
        model_type="step3_text",
        model_architecture_profile="step3_text",
    )


def _typed_dense_config() -> BaseModelConfig:
    """Build a profile-backed dense config for selector admission checks."""

    return BaseModelConfig(
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=1,
        embedding_dim=1024,
        mlp_hidden_dim=2048,
        dense_mlp_hidden_dim=2048,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=False,
        model_type="llama",
        model_architecture_profile="generic",
    )


def test_typed_dense_selector_rejects_explicit_moe_override() -> None:
    """A typed dense layer cannot be forced onto the routed MoE path."""

    config = _typed_dense_config()

    with pytest.raises(ValueError, match="include_moe does not match"):
        SklearnMoEExecutionTimePredictor._resolve_moe_layer_classification(
            config,
            layer_id=0,
            num_layers=1,
            include_moe=True,
            include_ffn=True,
        )


def test_moe_dataset_contract_uses_profile_owned_routed_width() -> None:
    """Routed MoE validation must select routed width, not model-wide width."""

    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._model_config = _step3_like_moe_config()
    predictor._cluster_type = None
    predictor._moe_routing_distribution_type = "balanced"

    dataframe = pd.DataFrame(
        {
            "num_experts": [8, 8],
            "router_topk": [2, 2],
            "hidden_dim": [1024, 1024],
            "expert_hidden_dim": [5120, 18432],
            "num_tensor_parallel_workers": [1, 1],
            "expert_parallel_size": [1, 1],
            "time_stats.moe_grouped_gemm.median": [1.0, 2.0],
            "time_stats.moe_gating_linear.median": [1.0, 2.0],
            "time_stats.moe_gating_routing_topk.median": [1.0, 2.0],
            "time_stats.moe_shuffling.median": [1.0, 2.0],
        }
    )

    filtered = predictor._validate_moe_dataset_contract(
        dataframe,
        "step3-like-moe.csv",
        ["moe_grouped_gemm"],
        moe_tp_size=1,
        moe_ep_size=1,
    )

    assert filtered["expert_hidden_dim"].tolist() == [5120]


def test_runtime_load_features_use_profile_owned_routed_width() -> None:
    """Runtime MoE features must use the same routed width as training data."""

    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._model_config = _step3_like_moe_config()
    predictor._cluster_type = None
    predictor._replica_config = SimpleNamespace(
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    predictor._router_topk = 2

    lane_workload = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=1,
        total_expert_num=8,
        owned_expert_ids=tuple(range(8)),
        local_token_counts=(1, 1, 1, 1, 0, 0, 0, 0),
        routed_token_count=4,
        router_topk=2,
    )
    batch = SimpleNamespace(
        get_effective_total_tokens_rounded=lambda _cluster_type: 2,
    )

    features = predictor._build_moe_load_imbalance_features(
        lane_workload,
        batch=batch,
    )

    assert features["expert_hidden_dim"] == 5120
    assert features["model_expansion_ratio"] == 5.0
