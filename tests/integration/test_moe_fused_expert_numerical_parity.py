"""Native parity for the legacy fused-MoE expert path against vLLM's own.

Both sides run on the GPU with the same activations, weights, routing weights,
routing ids, expert map, kernel config and dtype. The comparison is the final
output tensor, including the local top-k reduction, so it covers exactly what
the repaired `_run_fused_moe_iteration` changed.

This runs only where the repaired path is selected: CUDA plus a vLLM build that
exposes the low-level fused-MoE API (`vllm>=0.10,<0.11`). Any other environment
skips, because a newer vLLM routes profiling through `fused_experts` directly
and there is nothing here to compare.

Run on an already authorized worker. This module never provisions workers and
never substitutes a simulated kernel for a native one.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch", reason="the vLLM expert kernels require PyTorch")
pytest.importorskip("vllm", reason="the reference implementation lives in vLLM")

from frontier.profiling.common.vllm_compat import vllm_config_context
from frontier.profiling.moe import moe_vllm_kernel as kernel


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the vLLM expert kernels require a CUDA device",
    ),
    pytest.mark.skipif(
        kernel.VLLM_API_VERSION != "0.10.x",
        reason=(
            "the repaired path is selected only by the low-level vLLM API; this "
            f"build reports VLLM_API_VERSION={kernel.VLLM_API_VERSION!r}"
        ),
    ),
]

# The production case is taken from the checked-in model config rather than
# retyped here, so the shapes stay tied to the model Frontier actually profiles.
PRODUCTION_MODEL = "qwen3-a3b-30b-moe"


def _production_shape():
    config_path = os.path.join(
        "data", "config", "models", f"{PRODUCTION_MODEL}.json"
    )
    if not os.path.exists(config_path):
        pytest.skip(f"run from the repository root; {config_path} not found")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "hidden": config["hidden_size"],
        "width": config["moe_intermediate_size"],
        "top_k": config["num_experts_per_tok"],
        "num_experts": config["num_experts"],
    }


def _uniform_routing(num_tokens, num_experts, top_k, dtype):
    """The routing Frontier's `random` distribution resolves to."""

    from frontier.profiling.moe.moe_impl import uniform_topk

    hidden_proxy = torch.empty(num_tokens, 1, device="cuda", dtype=dtype)
    logits = torch.empty(num_tokens, num_experts, device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = uniform_topk(hidden_proxy, logits, top_k)
    return topk_weights, topk_ids


def _skewed_routing(num_tokens, num_experts, top_k, dtype):
    """Popularity-weighted routing that leaves two experts with no tokens.

    An even split hides a shard that mishandles an expert with an empty block
    range, or one whose block count differs from its neighbours. The last two
    global experts are excluded by construction so the assertion that they stay
    empty does not depend on sampling luck.
    """

    generator = torch.Generator().manual_seed(29)
    allowed = torch.arange(num_experts - 2)
    popularity = 1.0 / (torch.arange(allowed.numel(), dtype=torch.float64) + 1.0)
    probabilities = (popularity / popularity.sum()).repeat(num_tokens, 1)
    picks = torch.multinomial(
        probabilities, top_k, replacement=False, generator=generator
    )
    topk_ids = allowed[picks].to(torch.int32)
    weights = torch.rand(num_tokens, top_k, generator=generator)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (
        weights.to(device="cuda", dtype=torch.float32),
        topk_ids.to("cuda"),
    )


def _build_case(
    *,
    num_tokens,
    hidden,
    width,
    top_k,
    num_experts,
    expert_parallel_size,
    ep_rank,
    seed,
    routing=_uniform_routing,
    dtype=torch.bfloat16,
):
    """Materialize one expert-parallel shard and its routing, on the device."""

    if num_experts % expert_parallel_size:
        raise ValueError("num_experts must divide evenly across the EP ranks")
    local_experts = num_experts // expert_parallel_size

    torch.manual_seed(seed)
    activations = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) * 0.1
    # `w1` carries the gate and up projections stacked, which is the only layout
    # `profile_fused_moe_kernel` produces.
    w1 = torch.randn(
        local_experts, 2 * width, hidden, device="cuda", dtype=dtype
    ) * 0.1
    w2 = torch.randn(local_experts, hidden, width, device="cuda", dtype=dtype) * 0.1

    topk_weights, topk_ids = routing(num_tokens, num_experts, top_k, dtype)

    # Only this rank's experts are local; every other global id maps to -1,
    # exactly as an EP deployment presents its shard to the kernel.
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    start = ep_rank * local_experts
    expert_map[start : start + local_experts] = torch.arange(
        local_experts, device="cuda", dtype=torch.int32
    )
    return activations, w1, w2, topk_weights, topk_ids, expert_map


def _run_repaired_path(activations, w1, w2, topk_weights, topk_ids, expert_map):
    """Drive `_run_fused_moe_iteration` exactly as the profiler drives it.

    The buffer shapes and the config/alignment calls are copied from
    `profile_fused_moe_kernel`, so a change that breaks the profiler's own setup
    breaks this test too.
    """

    num_tokens, hidden = activations.shape
    top_k = topk_ids.shape[1]
    width = w1.shape[1] // 2
    global_num_experts = expert_map.numel()

    config = kernel.try_get_optimal_moe_config(
        w1_shape=w1.shape,
        w2_shape=w2.shape,
        top_k=top_k,
        dtype=kernel.get_config_dtype_str(activations.dtype),
        M=num_tokens,
        block_shape=None,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = kernel.moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map=expert_map,
    )

    intermediate_cache1 = torch.empty(
        num_tokens, top_k, w1.shape[1], device="cuda", dtype=activations.dtype
    )
    intermediate_cache2 = torch.empty(
        num_tokens * top_k, width, device="cuda", dtype=activations.dtype
    )
    intermediate_cache3 = torch.empty(
        num_tokens, top_k, hidden, device="cuda", dtype=activations.dtype
    )
    out_hidden_states = torch.empty(
        num_tokens, hidden, device="cuda", dtype=activations.dtype
    )

    kernel._run_fused_moe_iteration(
        A=activations,
        w1=w1,
        w2=w2,
        intermediate_cache1=intermediate_cache1,
        intermediate_cache2=intermediate_cache2,
        intermediate_cache3=intermediate_cache3,
        out_hidden_states=out_hidden_states,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=top_k,
        config=config,
        block_dims=None,
    )
    return out_hidden_states


def _run_vllm_reference(activations, w1, w2, topk_weights, topk_ids, expert_map):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    return fused_experts(
        activations,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        global_num_experts=expert_map.numel(),
        expert_map=expert_map,
    )


def _assert_matches_reference(case):
    with vllm_config_context():
        actual = _run_repaired_path(*case)
        expected = _run_vllm_reference(*case)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all(), "the repaired path produced a non-finite value"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return actual


@pytest.mark.parametrize("num_tokens", [4096, 4097])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_production_shaped_expert_output_matches_vllm(num_tokens, ep_rank):
    """Production shapes at a prefill-sized token count, on two local shards.

    4097 is not a multiple of any kernel block size, so it exercises the padded
    tail. Two ranks cover the global-to-local expert id remapping. This is the
    token range where the previously missing kernels cost the most, so it is the
    case that most needs to be right.
    """

    case = _build_case(
        num_tokens=num_tokens,
        expert_parallel_size=8,
        ep_rank=ep_rank,
        seed=13,
        **_production_shape(),
    )
    _assert_matches_reference(case)


@pytest.mark.parametrize("top_k", [2, 4])
def test_uneven_expert_occupancy_matches_vllm(top_k):
    """A smaller shape, a different top-k, and experts that receive no tokens."""

    num_experts = 16
    case = _build_case(
        num_tokens=257,
        hidden=512,
        width=256,
        top_k=top_k,
        num_experts=num_experts,
        expert_parallel_size=2,
        ep_rank=1,
        seed=5,
        routing=_skewed_routing,
    )
    topk_ids, expert_map = case[4], case[5]
    local_ids = torch.nonzero(expert_map >= 0).flatten()
    routed_here = torch.isin(topk_ids.to(torch.int64), local_ids)
    assert routed_here.any(), "the case must route some tokens to this shard"
    assert not routed_here.all(), "the case must also route tokens away from it"
    for empty_expert in (num_experts - 2, num_experts - 1):
        assert empty_expert in local_ids, "the empty experts must belong to this shard"
        assert not (topk_ids == empty_expert).any(), "this expert must stay empty"

    _assert_matches_reference(case)


def test_repeated_invocations_do_not_reuse_a_stale_result():
    """Two different inputs through the same code path give two right answers.

    The profiler calls the iteration many times over reused buffers. A result
    that survived into the next call would still look plausible on its own.
    """

    shape = _production_shape()
    first = _build_case(
        num_tokens=512, expert_parallel_size=8, ep_rank=0, seed=101, **shape
    )
    second = _build_case(
        num_tokens=512, expert_parallel_size=8, ep_rank=0, seed=202, **shape
    )

    first_actual = _assert_matches_reference(first)
    second_actual = _assert_matches_reference(second)

    assert not torch.equal(first_actual, second_actual)


def test_fp8_path_runs_on_the_gated_activation():
    """The FP8 path quantizes the gated activation and still produces output.

    This is a structural check on native kernels, not FP8 parity. Frontier
    quantizes weights and activations with its own helpers, so a bit-exact
    comparison against `fused_experts` would first require matching those
    schemes. What it does settle is that the quantizer receives the gated
    activation buffer rather than a raw slice of the first projection, and that
    the real kernels accept that operand and return finite values.
    """

    if not kernel.check_fp8_available():
        pytest.skip("FP8 quantization utilities are unavailable in this build")

    num_tokens, hidden, width, top_k, num_experts = 256, 512, 256, 2, 16
    activations, w1, w2, topk_weights, topk_ids, expert_map = _build_case(
        num_tokens=num_tokens,
        hidden=hidden,
        width=width,
        top_k=top_k,
        num_experts=num_experts,
        expert_parallel_size=2,
        ep_rank=0,
        seed=17,
    )

    block_shape = [128, 128]
    quantized_w1, w1_scale = kernel.quantize_weights_to_fp8(w1, block_shape=block_shape)
    quantized_w2, w2_scale = kernel.quantize_weights_to_fp8(w2, block_shape=block_shape)
    quantized_a, a_scale = kernel.quantize_activations_to_fp8(
        activations, group_size=block_shape[1]
    )

    with vllm_config_context():
        config = kernel.try_get_optimal_moe_config(
            w1_shape=quantized_w1.shape,
            w2_shape=quantized_w2.shape,
            top_k=top_k,
            dtype=kernel.get_config_dtype_str(activations.dtype),
            M=num_tokens,
            block_shape=block_shape,
        )
        sorted_token_ids, expert_ids, padded = kernel.moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, expert_map=expert_map
        )

        observed = {}
        original_quantize = kernel.quantize_activations_to_fp8

        def observing_quantize(tensor, *, group_size):
            observed["shape"] = tuple(tensor.shape)
            return original_quantize(tensor, group_size=group_size)

        kernel.quantize_activations_to_fp8 = observing_quantize
        try:
            out_hidden_states = torch.empty(
                num_tokens, hidden, device="cuda", dtype=activations.dtype
            )
            kernel._run_fused_moe_iteration(
                A=quantized_a,
                w1=quantized_w1,
                w2=quantized_w2,
                intermediate_cache1=torch.empty(
                    num_tokens, top_k, 2 * width, device="cuda", dtype=activations.dtype
                ),
                intermediate_cache2=torch.empty(
                    num_tokens * top_k, width, device="cuda", dtype=activations.dtype
                ),
                intermediate_cache3=torch.empty(
                    num_tokens, top_k, hidden, device="cuda", dtype=activations.dtype
                ),
                out_hidden_states=out_hidden_states,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=padded,
                top_k=top_k,
                config=config,
                block_dims=(block_shape[0], block_shape[1]),
                A_scale=a_scale,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                use_fp8=True,
                # The production block-quantized path hands the kernel its
                # block shape; without it the kernel reads the scales as
                # per-tensor and this check would exercise a different path.
                block_shape=block_shape,
            )
        finally:
            kernel.quantize_activations_to_fp8 = original_quantize
    torch.cuda.synchronize()

    # The gated activation is `(M * top_k, width)`; the raw first projection
    # would be twice as wide.
    assert observed["shape"] == (num_tokens * top_k, width)
    assert out_hidden_states.shape == (num_tokens, hidden)
    assert torch.isfinite(out_hidden_states).all()
