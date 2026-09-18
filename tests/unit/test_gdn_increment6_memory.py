"""TDD coverage for Increment 6 GDN memory and state contracts."""

from types import SimpleNamespace

import pytest

from frontier.attention.gdn.config import GatedDeltaNetConfig
from frontier.config.model_config import BaseModelConfig
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ActivationType, ClusterType, NormType
from frontier.utils.param_counter import ParamCounter


def _qwen35_runtime_config(num_layers: int = 8, **overrides) -> BaseModelConfig:
    values = dict(
        num_layers=num_layers,
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
    values.update(overrides)
    return BaseModelConfig(**values)


def _replica_config(*, tp_size: int = 1, pp_size: int = 1, **overrides):
    model_config = overrides.pop("model_config", _qwen35_runtime_config())
    values = dict(
        model_config=model_config,
        model_name="synthetic-qwen35",
        attn_tensor_parallel_size=tp_size,
        moe_tensor_parallel_size=tp_size,
        moe_expert_parallel_size=1,
        attn_dp=1,
        num_pipeline_stages=pp_size,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _replica(*, num_layers: int = 8, pp_size: int = 1):
    return SimpleNamespace(
        num_layers=num_layers,
        num_layers_per_pipeline_stage=num_layers // pp_size,
        num_pipeline_stages=pp_size,
        max_request_tokens=128,
        kv_heads_per_tensor_parallel_worker=1,
        attention_head_dim=32,
        total_memory_gb=100,
        memory_margin_fraction=0.0,
    )


def test_state_layout_exposes_fixed_components_and_rejects_speculative_sizing():
    config = GatedDeltaNetConfig(
        conv_kernel_size=4,
        key_head_dim=32,
        value_head_dim=32,
        num_key_heads=2,
        num_value_heads=4,
    )
    layout = config.get_state_layout(tensor_parallel_size=2)

    assert layout.conv_state_bytes == 3 * 128 * 2
    assert layout.recurrent_state_bytes == 2 * 32 * 32 * 4
    assert layout.total_bytes == layout.conv_state_bytes + layout.recurrent_state_bytes
    with pytest.raises(TypeError):
        config.get_state_layout(tensor_parallel_size=2, num_speculative_tokens=1)


def test_param_counter_uses_gdn_sharding_and_fp32_a_log_bytes():
    counter = ParamCounter(_replica_config(tp_size=2), ClusterType.MONOLITHIC)
    gdn_config = counter._model_config.get_gdn_config()
    local_value_heads = gdn_config.num_value_heads // 2
    expected = (
        (gdn_config.conv_dim // 2) * gdn_config.conv_kernel_size
        + 256 * (2 * (gdn_config.key_dim // 2) + 2 * (gdn_config.value_dim // 2))
        + 256 * (2 * local_value_heads)
        + 256 * (gdn_config.value_dim // 2)
        + gdn_config.value_head_dim
        + 2 * local_value_heads
    )
    assert counter.get_num_gdn_params_per_layer() == expected
    assert counter.get_gdn_parameter_memory_bytes_per_layer() == 2 * (expected - local_value_heads) + 4 * local_value_heads


def test_param_counter_uses_actual_layer_schedule_per_pipeline_stage():
    model_config = _qwen35_runtime_config(num_layers=8)
    counter = ParamCounter(
        _replica_config(model_config=model_config, pp_size=2),
        ClusterType.MONOLITHIC,
    )
    gdn_params = counter.get_num_gdn_params_per_layer()
    full_params = counter.get_num_attention_params_per_layer()
    assert counter.get_attention_stage_layer_counts() == ((3, 1), (3, 1))
    assert counter.get_num_attention_parameters_per_device() == 3 * gdn_params + full_params


def test_memory_planner_reserves_state_and_counts_only_full_attention_pages():
    model_config = _qwen35_runtime_config(num_layers=8)
    planner = MemoryPlanner(
        replica_config=_replica_config(model_config=model_config, tp_size=2, pp_size=2),
        replica=_replica(num_layers=8, pp_size=2),
        cluster_type=ClusterType.MONOLITHIC,
        max_num_seqs=4,
    )
    planner._get_parameter_memory_per_device = lambda: 0
    state_per_request = planner.get_gdn_state_memory_per_device_per_request_bytes()
    page_size = planner._get_kv_cache_memory_per_layer_per_block(16)
    expected = (100 * 1024**3 - 4 * state_per_request) // page_size // 1
    assert planner._get_num_full_attention_layers_per_device() == 1
    assert planner.get_num_blocks(block_size=16, gpu_memory_utilization=1.0) == expected


def test_non_gdn_memory_planner_keeps_existing_layer_denominator():
    model_config = BaseModelConfig.create_from_name("llama2_7b_dense_example")
    replica_config = SimpleNamespace(
        model_config=model_config,
        model_name="llama2_7b_dense_example",
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
        num_pipeline_stages=1,
    )
    planner = MemoryPlanner(
        replica_config=replica_config,
        replica=_replica(num_layers=model_config.num_layers),
        cluster_type=ClusterType.MONOLITHIC,
    )
    assert planner._get_num_full_attention_layers_per_device() == model_config.num_layers
    assert planner.get_parameter_memory_per_device_bytes() > 0
