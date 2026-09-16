"""TDD coverage for explicit unsupported GDN runtime boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.attention.gdn.guards import validate_gdn_runtime_support
from frontier.config.model_config import BaseModelConfig
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.types import ActivationType, NormType


def _gdn_model():
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefix_cache_enabled": True}, "prefix caching"),
        ({"speculative_enabled": True}, "speculative decoding"),
        ({"num_pipeline_stages": 2}, "pipeline parallelism"),
        ({"moe_expert_parallel_size": 2}, "expert parallelism"),
        ({"attn_dp": 2}, "attention data parallelism"),
        ({"cross_node": True}, "cross-node"),
    ],
)
def test_gdn_runtime_guards_fail_before_unsupported_feature(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_gdn_runtime_support(_gdn_model(), **kwargs)


@pytest.mark.parametrize(
    "schedule",
    [
        {"layer_types": ("linear_attention",) * 8, "full_attention_interval": None},
        {"layer_types": None, "full_attention_interval": 9},
    ],
)
def test_qwen35_hybrid_constructor_rejects_zero_kv_layer_layout(schedule):
    with pytest.raises(ValueError, match="both GDN and full-attention layers"):
        replace(_gdn_model(), **schedule)


def test_waiting_is_not_preemption_but_state_drop_is_rejected_before_mutation():
    validate_gdn_runtime_support(_gdn_model(), waiting=True)
    with pytest.raises(ValueError, match="state recovery"):
        validate_gdn_runtime_support(_gdn_model(), preemption_requires_state_drop=True)


def test_gdn_pd_transfer_fails_before_transfer_size_calculation():
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig())
    replica_config = SimpleNamespace(model_config=_gdn_model())
    request = SimpleNamespace(num_prefill_tokens=8)
    with pytest.raises(ValueError, match="P.*D.*GDN|GDN.*P.*D"):
        predictor.get_kv_cache_size_for_request(request, replica_config)


def test_non_gdn_transfer_remains_available():
    model = BaseModelConfig.create_from_name("llama2_7b_dense_example")
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig())
    replica_config = SimpleNamespace(model_config=model)
    request = SimpleNamespace(num_prefill_tokens=8)
    assert predictor.get_kv_cache_size_for_request(request, replica_config) > 0
