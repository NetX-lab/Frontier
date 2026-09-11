from types import SimpleNamespace
from typing import cast

from frontier.config import ReplicaConfig
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


MODEL_NAME = "Qwen3.8-2.4T-A95B-Quark-MXFP4"


def _replica_config(*, tp_size: int = 1, pp_size: int = 1):
    return cast(
        ReplicaConfig,
        SimpleNamespace(
            model_config=ModelConfig.from_model_name(MODEL_NAME),
            model_name=MODEL_NAME,
            attn_tensor_parallel_size=tp_size,
            moe_tensor_parallel_size=tp_size,
            moe_expert_parallel_size=1,
            num_pipeline_stages=pp_size,
        ),
    )


def test_qwen_gdn_parameter_formula_matches_live_vllm_module() -> None:
    counter = ParamCounter(_replica_config(), ClusterType.MONOLITHIC)

    assert counter.get_num_gdn_params_per_layer() == 438_387_072
    assert counter.get_gdn_parameter_memory_bytes_per_layer() == 876_774_400
    assert counter.get_num_attention_params_per_layer() == 285_212_672
    assert counter.get_num_attention_parameters_per_device() == 36_808_599_424
    assert (
        counter.get_attention_parameter_memory_per_device_bytes()
        == 73_617_216_512
    )


def test_qwen_hybrid_parameter_memory_uses_largest_exact_pp_stage() -> None:
    counter = ParamCounter(
        _replica_config(tp_size=1, pp_size=4),
        ClusterType.MONOLITHIC,
    )

    # Stage 0 owns 18 GDN + 5 full-attention layers; the other stages own
    # 17 GDN + 6 full-attention layers, so stage 0 is the largest weight shard.
    assert counter.get_num_attention_parameters_per_device() == 9_317_030_656
    expected_bytes = 18 * 876_774_400 + 5 * (2 * 285_212_672)
    assert counter.get_attention_parameter_memory_per_device_bytes() == expected_bytes


def test_memory_planner_reserves_fixed_gdn_state_before_dense_kv_blocks() -> None:
    replica_config = _replica_config(tp_size=4, pp_size=4)
    replica = SimpleNamespace(
        num_layers=92,
        num_layers_per_pipeline_stage=23,
        num_pipeline_stages=4,
        max_request_tokens=4096,
        kv_heads_per_tensor_parallel_worker=1,
        attention_head_dim=256,
        total_memory_gb=100,
        memory_margin_fraction=0.0,
    )
    planner = MemoryPlanner(
        replica_config=replica_config,
        replica=replica,
        cluster_type=ClusterType.MONOLITHIC,
        max_num_seqs=128,
    )

    state_per_request = 18 * 2_127_872
    assert planner._get_num_gdn_layers_per_device() == 18
    assert planner._get_num_full_attention_layers_per_device() == 6
    assert (
        planner.get_gdn_state_memory_per_device_per_request_bytes()
        == state_per_request
    )

    # Isolate the cache calculation from the model's very large MoE weights.
    planner._get_parameter_memory_per_device = lambda: 0
    requested_bytes = 100 * 1024**3
    page_size = 2 * 16 * (2 * 1 * 256)
    expected_blocks = (
        requested_bytes - 128 * state_per_request
    ) // page_size // 6
    assert planner.get_num_blocks(block_size=16, gpu_memory_utilization=1.0) == (
        expected_blocks
    )


def test_hybrid_kv_transfer_includes_dense_tokens_and_fixed_gdn_state() -> None:
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_cache_dtype_size_bytes=2)
    )
    replica_config = _replica_config()
    request = SimpleNamespace(num_prefill_tokens=16)

    size_bytes = predictor.get_kv_cache_size_for_request(request, replica_config)

    dense_kv_bytes = 16 * 23 * 4 * 256 * 2 * 2
    gdn_state_bytes = 69 * 8_511_488
    assert size_bytes == dense_kv_bytes + gdn_state_bytes


def test_hybrid_kv_transfer_charges_gdn_state_once_per_request() -> None:
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_cache_dtype_size_bytes=2)
    )
    replica_config = _replica_config()
    batch = SimpleNamespace(
        requests=[
            SimpleNamespace(num_prefill_tokens=16),
            SimpleNamespace(num_prefill_tokens=32),
        ]
    )

    size_bytes = predictor.get_kv_cache_size(batch, replica_config)

    dense_kv_bytes = 48 * 23 * 4 * 256 * 2 * 2
    gdn_state_bytes = 2 * 69 * 8_511_488
    assert size_bytes == dense_kv_bytes + gdn_state_bytes
