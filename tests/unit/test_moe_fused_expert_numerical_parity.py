"""Verify the profiled expert path against vLLM with identical routed inputs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from frontier.profiling.moe import moe_vllm_kernel as kernel
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


@pytest.mark.parametrize("num_tokens", [4096, 4097])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_gated_expert_path_matches_vllm(num_tokens, ep_rank):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the vLLM expert kernels")
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.moe.moe_impl import uniform_topk

    model = ModelConfig.from_model_name("qwen3-a3b-30b-moe")
    hidden, width = model.embedding_dim, model.routed_mlp_hidden_dim
    top_k, experts, ep_size = model.num_experts_per_tok, model.num_experts, 8
    local_experts = experts // ep_size
    torch.manual_seed(13)
    a = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(local_experts, 2 * width, hidden, device="cuda", dtype=a.dtype) * 0.1
    w2 = torch.randn(local_experts, hidden, width, device="cuda", dtype=a.dtype) * 0.1
    logits = torch.empty(num_tokens, experts, device="cuda", dtype=a.dtype)
    weights, ids, _ = uniform_topk(a, logits, top_k)
    expert_map = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
    start = ep_rank * local_experts
    expert_map[start:start + local_experts] = torch.arange(local_experts, device="cuda")
    config = kernel.try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, kernel.get_config_dtype_str(a.dtype), num_tokens
    )
    sorted_ids, block_experts, padded = kernel.moe_align_block_size(
        ids, config["BLOCK_SIZE_M"], experts, expert_map=expert_map
    )
    workspace = torch.empty(num_tokens * top_k * max(2 * width, hidden), device="cuda", dtype=a.dtype)
    cache1 = workspace[:num_tokens * top_k * 2 * width].view(num_tokens, top_k, 2 * width)
    cache2 = workspace[:num_tokens * top_k * hidden].view(num_tokens, top_k, hidden)
    activated = torch.empty(num_tokens * top_k, width, device="cuda", dtype=a.dtype)
    actual = torch.empty_like(a)
    kernel._run_fused_moe_iteration(
        a, w1, w2, cache1, cache2, activated, actual, weights,
        sorted_ids, block_experts, padded, top_k, config, None,
    )
    expected = fused_experts(
        a, w1, w2, weights, ids, inplace=False,
        global_num_experts=experts, expert_map=expert_map,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
