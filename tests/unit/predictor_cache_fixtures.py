"""Normally constructed predictors for numerical and routing cache tests."""

from dataclasses import replace

from frontier.config import (
    BaseModelConfig,
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.types import ActivationType, ClusterType, NormType


def cache_model(*, hybrid: bool = False) -> BaseModelConfig:
    hybrid_fields = (
        dict(
            model_type="qwen3_5_moe_text",
            model_architecture_profile="qwen3_5_moe",
            architectures=("Qwen3_5MoeForCausalLM",),
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            full_attention_interval=4,
        )
        if hybrid else {}
    )
    return BaseModelConfig(
        num_layers=8,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        **hybrid_fields,
    )


def predictor_fixture_config(*, hybrid: bool = False, replica_id: int = 0,
                             model_config: BaseModelConfig | None = None,
                             total_experts: int = 8, ep_size: int = 1,
                             router_topk: int = 2) -> dict:
    """Build typed inputs shared by the production predictor constructors."""
    replica = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf", device="a100",
        network_device="a100_pairwise_nvlink", attn_tensor_parallel_size=ep_size,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=ep_size,
        total_expert_num=total_experts, router_topk=router_topk,
    )
    replica.model_config = model_config or replace(
        cache_model(hybrid=hybrid), num_experts=total_experts,
        num_experts_per_tok=router_topk,
    )
    replica.cluster_num_replicas = 1
    return dict(
        predictor_config=RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True),
        replica_config=replica, replica_scheduler_config=VllmV1SchedulerConfig(),
        metrics_config=MetricsConfig(), cluster_type=ClusterType.MONOLITHIC,
        actual_replica_ids=[replica_id],
    )


class CacheFixturePredictor(SklearnMoEExecutionTimePredictor):
    """Use the full production constructor without loading external profiles."""

    def __init__(self, **kwargs):
        super().__init__(**predictor_fixture_config(**kwargs))

    def _get_estimator(self):
        raise AssertionError("Dummy constructor must not train a model")

    def _get_grid_search_params(self):
        return {}


class CacheFixtureDisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    """Initialize the full disaggregation predictor with explicit role configs."""

    def __init__(self, *, cluster_type=ClusterType.PREFILL, **kwargs):
        inputs = predictor_fixture_config(**kwargs)
        replica = inputs["replica_config"]
        cluster = ClusterConfig(replica_config=replica)
        for role in ("prefill", "decode", "decode_attn", "decode_ffn"):
            setattr(cluster, f"{role}_replica_config", replica)
            setattr(cluster, f"{role}_cluster_num_replicas", 1)
        inputs.update(cluster_type=cluster_type, cluster_config=cluster)
        super().__init__(**inputs)

    def _get_estimator(self):
        raise AssertionError("Dummy constructor must not train a model")

    def _get_grid_search_params(self):
        return {}
