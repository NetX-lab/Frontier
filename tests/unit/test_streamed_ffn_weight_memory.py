from types import SimpleNamespace

import pytest

from frontier.config import ReplicaConfig
from frontier.errors import FrontierMemoryOOMError
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


def ffn_replica(streamed_layers, **overrides):
    config = dict(
        model_name="Qwen3-4B",
        cluster_prefix="decode_ffn",
        device="hpp2",
        network_device="hpp2_16rank",
        moe_tensor_parallel_size=16,
        streamed_ffn_layers=streamed_layers,
    )
    return ReplicaConfig(**(config | overrides))


@pytest.mark.parametrize("streamed_layers", [0, 1, 2, 3, 36])
def test_dense_ffn_streaming_preserves_logical_weights_and_budgets_one_buffer(streamed_layers):
    config = ffn_replica(streamed_layers)
    counter = ParamCounter(config, ClusterType.DECODE_FFN)
    planner = MemoryPlanner(config, replica=None, cluster_type=ClusterType.DECODE_FFN)
    # Three FP8 weight matrices with TP16: gate, up, down.
    layer_bytes = 3 * 2560 * 9728 // 16
    resident = (36 - streamed_layers) * layer_bytes
    buffer_bytes = layer_bytes if streamed_layers else 0
    assert counter.get_num_mlp_parameters_per_device() == 36 * layer_bytes
    assert counter.get_streamed_ffn_weight_bytes_per_device() == streamed_layers * layer_bytes
    assert counter.get_parameter_memory_per_device_bytes() == resident
    assert counter.get_ffn_streaming_buffer_bytes_per_device() == buffer_bytes
    assert planner.get_parameter_memory_per_device_bytes() == resident + buffer_bytes


@pytest.mark.parametrize("n", [-1, 37, 1.5])
def test_invalid_layer_count(n):
    with pytest.raises(ValueError, match="streamed_ffn_layers"):
        ffn_replica(n)


@pytest.mark.parametrize("overrides", [
    {"cluster_prefix": "decode_attn"},
    {"model_name": "step-moe-noquant"},
    {"num_pipeline_stages": 2},
])
def test_streamed_layers_require_supported_role_and_uniform_dense_weights(overrides):
    with pytest.raises(ValueError, match="streamed_ffn_layers"):
        ffn_replica(1, **overrides)


@pytest.mark.parametrize("tp,streamed_layers,activation_bytes,should_fit", [
    (8, 0, None, False),
    (8, 3, None, False),
    (16, 0, None, True),
    (16, 3, 0, False),
    (16, 3, 1, True),
])
def test_decode_ffn_requires_free_activation_capacity(tp, streamed_layers, activation_bytes, should_fit):
    config = ffn_replica(streamed_layers, moe_tensor_parallel_size=tp)
    rank = SimpleNamespace(total_memory_gb=0.1875, memory_margin_fraction=0)
    planner = MemoryPlanner(config, rank, ClusterType.DECODE_FFN)
    overhead = 0
    if activation_bytes is not None:
        overhead = 192 * 1024**2 - planner.get_parameter_memory_per_device_bytes() - activation_bytes
    if should_fit:
        assert planner.get_max_batch_size(non_kv_cache_overhead_bytes=overhead) >= 1
    else:
        with pytest.raises(FrontierMemoryOOMError) as error:
            planner.get_max_batch_size(non_kv_cache_overhead_bytes=overhead)
        assert error.value.reason == "insufficient_activation_budget"
