"""
Unit tests for op trace shape/precision helpers.

Tests cover:
- Compute op shape/size metadata
- Communication op size metadata
- Transfer shape/size metadata
"""

from unittest.mock import MagicMock

import pytest


def _reset_quantization() -> None:
    from frontier.config import get_quantization_manager

    get_quantization_manager().load_config(None)


def _build_context(is_moe: bool = False, tokens_are_post_routing: bool = False):
    from frontier.metrics.op_trace_utils import OpTraceContext
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 8
    model_config.num_q_heads = 4
    model_config.num_kv_heads = 2
    model_config.mlp_hidden_dim = 16
    model_config.num_experts = 4 if is_moe else 0
    model_config.num_experts_per_tok = 2 if is_moe else 0
    model_config.is_moe = is_moe
    # Mock get_head_dim() to return computed value (embedding_dim // num_q_heads = 8 // 4 = 2)
    model_config.get_head_dim = MagicMock(return_value=2)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 2
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 2
    replica_config.moe_expert_parallel_size = 2
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 2 if is_moe else 1

    return OpTraceContext(
        cluster_type=ClusterType.PREFILL,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=tokens_are_post_routing,
    )


def test_compute_attn_pre_proj_shapes():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context()
    meta = compute_op_trace_meta("attn_pre_proj", "COMPUTE", ctx)

    assert meta["tensor_shape"]["input"] == [8, 8]
    assert meta["tensor_shape"]["output"] == [8, 8]
    assert meta["tensor_size_bytes"]["input"] == 128
    assert meta["tensor_size_bytes"]["output"] == 128
    assert meta["dtype"] == "FP16"


def test_attention_kv_replication_shapes_for_tp8():
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 4096
    model_config.num_q_heads = 32
    model_config.num_kv_heads = 4
    model_config.mlp_hidden_dim = 11008
    model_config.num_experts = 0
    model_config.num_experts_per_tok = 0
    model_config.is_moe = False
    model_config.get_head_dim = MagicMock(return_value=128)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 8
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 1
    replica_config.moe_expert_parallel_size = 1
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 1

    ctx = OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=16,
        effective_tokens_compute=16,
        effective_tokens_transfer=16,
        effective_tokens_rounded=16,
        tokens_are_post_routing=False,
    )

    pre_proj_meta = compute_op_trace_meta("attn_pre_proj", "COMPUTE", ctx)
    assert pre_proj_meta["tensor_shape"]["output"] == [16, 768]

    kv_save_meta = compute_op_trace_meta("attn_kv_cache_save", "COMPUTE", ctx)
    assert kv_save_meta["tensor_shape"]["k"] == [16, 1, 128]
    assert kv_save_meta["tensor_shape"]["v"] == [16, 1, 128]


def test_attention_kv_replication_requires_divisible_tp_ratio():
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 4096
    model_config.num_q_heads = 32
    model_config.num_kv_heads = 3
    model_config.mlp_hidden_dim = 11008
    model_config.num_experts = 0
    model_config.num_experts_per_tok = 0
    model_config.is_moe = False
    model_config.get_head_dim = MagicMock(return_value=128)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 8
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 1
    replica_config.moe_expert_parallel_size = 1
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 1

    ctx = OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=16,
        effective_tokens_compute=16,
        effective_tokens_transfer=16,
        effective_tokens_rounded=16,
        tokens_are_post_routing=False,
    )

    with pytest.raises(ValueError, match="replication requires attn_tp"):
        compute_op_trace_meta("attn_prefill", "COMPUTE", ctx)


def test_comm_allreduce_sizes():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context()
    meta = compute_op_trace_meta("attn_tensor_parallel_allreduce", "COMM", ctx)

    assert meta["tensor_shape"]["data"] == [8, 8]
    assert meta["element_count"] == 64
    assert meta["base_size_bytes"] == 128
    assert meta["data_size_bytes"] == 128
    assert meta["dtype"] == "FP16"


def test_moe_grouped_gemm_shapes():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context(is_moe=True)
    meta = compute_op_trace_meta("moe_grouped_gemm", "COMPUTE", ctx)

    assert meta["tensor_shape"]["input"] == [16, 8]
    assert meta["tensor_shape"]["output"] == [16, 8]
    assert meta["tensor_size_bytes"]["input"] == 256
    assert meta["tensor_size_bytes"]["output"] == 256


def test_moe_grouped_gemm_post_routing_shapes():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context(is_moe=True, tokens_are_post_routing=True)
    meta = compute_op_trace_meta("moe_grouped_gemm", "COMPUTE", ctx)

    assert meta["tensor_shape"]["input"] == [8, 8]
    assert meta["tensor_shape"]["output"] == [8, 8]
    assert meta["tensor_size_bytes"]["input"] == 128
    assert meta["tensor_size_bytes"]["output"] == 128


def test_moe_ep_comm_post_routing_shape():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context(is_moe=True, tokens_are_post_routing=True)
    meta = compute_op_trace_meta("expert_parallel_alltoall_dispatch", "COMM", ctx)

    assert meta["tensor_shape"]["data"] == [4, 2, 8]
    assert meta["element_count"] == 64
    assert meta["base_size_bytes"] == 128


def test_moe_ep_alltoall_post_routing_shape():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context(is_moe=True, tokens_are_post_routing=True)
    meta = compute_op_trace_meta("expert_parallel_alltoall", "COMM", ctx)

    assert meta["tensor_shape"]["data"] == [4, 2, 8]
    assert meta["element_count"] == 64
    assert meta["base_size_bytes"] == 128


def test_moe_shuffling_non_divisible_tokens():
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 8
    model_config.num_q_heads = 4
    model_config.num_kv_heads = 2
    model_config.mlp_hidden_dim = 16
    model_config.num_experts = 4
    model_config.num_experts_per_tok = 3
    model_config.is_moe = True
    model_config.get_head_dim = MagicMock(return_value=2)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 1
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 1
    replica_config.moe_expert_parallel_size = 2
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 3

    ctx = OpTraceContext(
        cluster_type=ClusterType.DECODE_FFN,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=10,
        effective_tokens_compute=10,
        effective_tokens_transfer=10,
        effective_tokens_rounded=10,
        tokens_are_post_routing=True,
    )

    # In Frontier tracing, moe_shuffling is modeled as a compute op (not a comm op).
    meta = compute_op_trace_meta("moe_shuffling", "COMPUTE", ctx)

    assert meta["tensor_shape"]["input"] == [4, 3, 8]
    assert meta["tensor_shape"]["output"] == [4, 3, 8]
    assert meta["tensor_size_bytes"]["input"] == 192
    assert meta["tensor_size_bytes"]["output"] == 192


def test_share_expert_trace_shapes():
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 16
    model_config.num_q_heads = 4
    model_config.num_kv_heads = 2
    model_config.mlp_hidden_dim = 32
    model_config.share_expert_dim = 12
    model_config.num_experts = 8
    model_config.num_experts_per_tok = 2
    model_config.is_moe = True
    model_config.get_head_dim = MagicMock(return_value=4)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 1
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 2
    replica_config.moe_expert_parallel_size = 2
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 2

    ctx = OpTraceContext(
        cluster_type=ClusterType.DECODE_FFN,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=False,
    )

    meta_up = compute_op_trace_meta("share_expert_up_proj", "COMPUTE", ctx)
    assert meta_up["tensor_shape"]["output"] == [8, 6]

    meta_act = compute_op_trace_meta("share_expert_act", "COMPUTE", ctx)
    assert meta_act["tensor_shape"]["input"] == [8, 6]

    meta_down = compute_op_trace_meta("share_expert_down_proj", "COMPUTE", ctx)
    assert meta_down["tensor_shape"]["input"] == [8, 6]
    assert meta_down["tensor_shape"]["output"] == [8, 16]


def test_share_expert_requires_dim():
    from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
    from frontier.types import ClusterType

    _reset_quantization()

    model_config = MagicMock()
    model_config.embedding_dim = 16
    model_config.num_q_heads = 4
    model_config.num_kv_heads = 2
    model_config.mlp_hidden_dim = 32
    model_config.share_expert_dim = None
    model_config.num_experts = 8
    model_config.num_experts_per_tok = 2
    model_config.is_moe = True
    model_config.get_head_dim = MagicMock(return_value=4)

    replica_config = MagicMock()
    replica_config.attn_tensor_parallel_size = 1
    replica_config.attn_data_parallel_size = 1
    replica_config.moe_tensor_parallel_size = 2
    replica_config.moe_expert_parallel_size = 2
    replica_config.num_pipeline_stages = 1
    replica_config.router_topk = 2

    ctx = OpTraceContext(
        cluster_type=ClusterType.DECODE_FFN,
        model_config=model_config,
        replica_config=replica_config,
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=False,
    )

    with pytest.raises(ValueError, match="share_expert_dim must be set"):
        compute_op_trace_meta("share_expert_up_proj", "COMPUTE", ctx)


def test_kv_cache_transfer_meta():
    from frontier.metrics.op_trace_utils import build_kv_cache_transfer_meta
    from frontier.types import ClusterType

    _reset_quantization()
    req_a = MagicMock()
    req_a.num_prefill_tokens = 2
    req_b = MagicMock()
    req_b.num_prefill_tokens = 2

    batch = MagicMock()
    batch.requests = [req_a, req_b]

    replica_config = MagicMock()
    model_config = MagicMock()
    model_config.num_layers = 2
    model_config.num_q_heads = 4
    model_config.num_kv_heads = 2
    model_config.embedding_dim = 8
    model_config.is_moe = False
    # Mock get_head_dim() to return computed value (embedding_dim // num_q_heads = 8 // 4 = 2)
    model_config.get_head_dim = MagicMock(return_value=2)
    replica_config.model_config = model_config

    meta = build_kv_cache_transfer_meta(
        batch, replica_config, ClusterType.PREFILL, transfer_size_bytes=512
    )

    assert meta["total_tokens"] == 4
    assert meta["tensor_shape"]["kv"] == [4, 2, 2, 2, 2]
    assert meta["tensor_size_bytes"]["kv"] == 128
    assert meta["num_heads"] == 2
    assert meta["num_q_heads"] == 4
    assert meta["num_kv_heads"] == 2
    assert meta["dtype"] == "FP16"
    assert meta["transfer_size_bytes"] == 512


def test_m2n_transfer_meta():
    from frontier.metrics.op_trace_utils import build_m2n_transfer_meta
    from frontier.types import ClusterType

    _reset_quantization()
    batch = MagicMock()
    batch.get_effective_total_tokens_for_transfer = MagicMock(return_value=4)

    replica_config = MagicMock()
    model_config = MagicMock()
    model_config.embedding_dim = 8
    model_config.is_moe = False
    replica_config.model_config = model_config

    meta = build_m2n_transfer_meta(
        batch, replica_config, ClusterType.DECODE_ATTN, activation_size_bytes=128
    )

    assert meta["total_tokens"] == 4
    assert meta["tensor_shape"]["activation"] == [4, 8]
    assert meta["dtype"] == "FP16"
    assert meta["activation_size_bytes"] == 128



def test_moe_ep_allreduce_shape():
    from frontier.metrics.op_trace_utils import compute_op_trace_meta

    _reset_quantization()
    ctx = _build_context(is_moe=True, tokens_are_post_routing=False)
    meta = compute_op_trace_meta("expert_parallel_allreduce", "COMM", ctx)

    assert meta["tensor_shape"]["data"] == [8, 8]
    assert meta["element_count"] == 64
    assert meta["base_size_bytes"] == 128
