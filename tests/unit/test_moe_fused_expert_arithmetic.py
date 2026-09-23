"""CPU boundary tests for the legacy fused-MoE expert computation.

No native kernel runs here. `_invoke_kernel`, the gated activation and the local
top-k reduction are replaced by plain-Torch references, so what these tests
check is the *composition*: which operand reaches the second expert GEMM, where
the routing weights are applied, and whether the top-k expert outputs are
reduced. That is exactly what the repaired path changes.

This is boundary validation, not native numerical parity. Parity against vLLM's
own kernels needs a GPU and is recorded separately.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the fused-MoE profiler is built on PyTorch")

from frontier.profiling.moe import moe_vllm_kernel as kernel  # noqa: E402


NUM_TOKENS = 5
TOP_K = 2
HIDDEN_DIM = 6
EXPERT_HIDDEN_DIM = 4
NUM_EXPERTS = 3


@pytest.fixture
def expert_problem():
    """One small gated MoE problem with fixed routing, entirely on CPU."""

    generator = torch.Generator().manual_seed(7)
    activations = torch.randn(NUM_TOKENS, HIDDEN_DIM, generator=generator)
    w1 = torch.randn(
        NUM_EXPERTS, 2 * EXPERT_HIDDEN_DIM, HIDDEN_DIM, generator=generator
    )
    w2 = torch.randn(NUM_EXPERTS, HIDDEN_DIM, EXPERT_HIDDEN_DIM, generator=generator)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [0, 2], [1, 1]], dtype=torch.int32
    )
    topk_weights = torch.rand(NUM_TOKENS, TOP_K, generator=generator)
    return SimpleNamespace(
        A=activations, w1=w1, w2=w2, topk_ids=topk_ids, topk_weights=topk_weights
    )


def _expected_output(problem) -> torch.Tensor:
    """The gated expert computation, written out directly."""

    out = torch.zeros(NUM_TOKENS, HIDDEN_DIM)
    for token in range(NUM_TOKENS):
        for slot in range(TOP_K):
            expert = int(problem.topk_ids[token, slot])
            projected = problem.A[token] @ problem.w1[expert].T
            gate, up = projected[:EXPERT_HIDDEN_DIM], projected[EXPERT_HIDDEN_DIM:]
            activated = torch.nn.functional.silu(gate) * up
            expert_out = activated @ problem.w2[expert].T
            out[token] += problem.topk_weights[token, slot] * expert_out
    return out


def _sliced_output(problem) -> torch.Tensor:
    """What the path produced before the repair: the gate half, unactivated."""

    out = torch.zeros(NUM_TOKENS, HIDDEN_DIM)
    for token in range(NUM_TOKENS):
        for slot in range(TOP_K):
            expert = int(problem.topk_ids[token, slot])
            projected = problem.A[token] @ problem.w1[expert].T
            expert_out = projected[:EXPERT_HIDDEN_DIM] @ problem.w2[expert].T
            out[token] += problem.topk_weights[token, slot] * expert_out
    return out


@pytest.fixture
def native_stubs(monkeypatch, expert_problem):
    """Replace the three native calls with plain-Torch references.

    Each stub also records its call, so a test can assert the call order and the
    operands without reaching into the profiling return contract.
    """

    calls = []
    topk_ids = expert_problem.topk_ids

    def fake_invoke_kernel(
        *, A, B, C, topk_weights, mul_routed_weight, top_k, block_shape=None, **_
    ):
        calls.append(
            SimpleNamespace(
                name="invoke_kernel",
                A=A,
                mul_routed_weight=mul_routed_weight,
                top_k=top_k,
                block_shape=block_shape,
            )
        )
        rows = C.view(NUM_TOKENS, TOP_K, C.shape[-1])
        for token in range(NUM_TOKENS):
            for slot in range(TOP_K):
                expert = int(topk_ids[token, slot])
                if top_k == 1:
                    source = A.view(NUM_TOKENS, TOP_K, A.shape[-1])[token, slot]
                else:
                    source = A[token]
                value = source @ B[expert].T
                if mul_routed_weight:
                    value = value * topk_weights[token, slot]
                rows[token, slot] = value

    def fake_silu_and_mul(out, inp):
        calls.append(SimpleNamespace(name="silu_and_mul", out=out, inp=inp))
        half = inp.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(inp[:, :half]) * inp[:, half:])

    def fake_moe_sum(inp, out):
        calls.append(SimpleNamespace(name="moe_sum", inp=inp, out=out))
        out.copy_(inp.sum(dim=1))

    monkeypatch.setattr(kernel, "_invoke_kernel", fake_invoke_kernel)
    monkeypatch.setattr(
        kernel,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(_C=SimpleNamespace(silu_and_mul=fake_silu_and_mul))),
    )
    monkeypatch.setattr(
        kernel, "_vllm_custom_ops", SimpleNamespace(moe_sum=fake_moe_sum)
    )
    return calls


def _run(problem, **overrides):
    cache1 = torch.empty(NUM_TOKENS, TOP_K, 2 * EXPERT_HIDDEN_DIM)
    cache2 = torch.empty(NUM_TOKENS * TOP_K, EXPERT_HIDDEN_DIM)
    cache3 = torch.empty(NUM_TOKENS, TOP_K, HIDDEN_DIM)
    out = torch.empty(NUM_TOKENS, HIDDEN_DIM)
    arguments = dict(
        A=problem.A,
        w1=problem.w1,
        w2=problem.w2,
        intermediate_cache1=cache1,
        intermediate_cache2=cache2,
        intermediate_cache3=cache3,
        out_hidden_states=out,
        topk_weights=problem.topk_weights,
        sorted_token_ids=torch.zeros(1, dtype=torch.int32),
        expert_ids=torch.zeros(1, dtype=torch.int32),
        num_tokens_post_padded=torch.zeros(1, dtype=torch.int32),
        top_k=TOP_K,
        config={"BLOCK_SIZE_M": 16},
    )
    arguments.update(overrides)
    kernel._run_fused_moe_iteration(**arguments)
    return SimpleNamespace(cache1=cache1, cache2=cache2, cache3=cache3, out=out)


def test_the_iteration_computes_the_gated_expert_output(expert_problem, native_stubs):
    """The full local computation matches a directly written reference."""

    result = _run(expert_problem)

    torch.testing.assert_close(result.out, _expected_output(expert_problem))


def test_a_gated_activation_is_not_the_first_half_of_the_projection(
    expert_problem, native_stubs
):
    """The repair is observable: slicing the gate half gives a different answer.

    Without this the test above would pass against the unrepaired path whenever
    the reference happened to agree, which for a gated layout it never does.
    """

    result = _run(expert_problem)
    sliced = _sliced_output(expert_problem)

    assert not torch.allclose(result.out, sliced, atol=1e-4)
    assert (result.out - sliced).abs().max() > 1e-3


def test_the_second_gemm_consumes_the_activation_buffer(expert_problem, native_stubs):
    """Operand provenance, which is what the defect got wrong."""

    result = _run(expert_problem)

    names = [call.name for call in native_stubs]
    assert names == ["invoke_kernel", "silu_and_mul", "invoke_kernel", "moe_sum"]

    first_gemm, activation, second_gemm, reduction = native_stubs
    assert first_gemm.A is expert_problem.A
    assert first_gemm.mul_routed_weight is False
    assert first_gemm.top_k == TOP_K

    # The activation reads the whole `gate | up` projection and writes the
    # buffer the second GEMM then reads.
    assert activation.inp.shape == (NUM_TOKENS * TOP_K, 2 * EXPERT_HIDDEN_DIM)
    assert activation.out is result.cache2
    assert second_gemm.A is result.cache2

    # Routing weights are applied once, on the second GEMM, over one expert.
    assert second_gemm.mul_routed_weight is True
    assert second_gemm.top_k == 1

    assert reduction.inp is result.cache3
    assert reduction.out is result.out


def test_both_gemm_inputs_are_quantized_inside_the_step(
    monkeypatch, expert_problem, native_stubs
):
    """Under FP8 the quantizer sees the hidden state, then the gated activation.

    vLLM quantizes both GEMM inputs inside `fused_experts_impl`, so both belong
    to the profiled step. The second input must be the gated activation buffer,
    not a raw slice of the first projection.
    """

    seen = []

    def fake_quantize(tensor, *, group_size):
        seen.append((tensor, group_size))
        return tensor, torch.ones(1)

    monkeypatch.setattr(kernel, "quantize_activations_to_fp8", fake_quantize)

    result = _run(expert_problem, use_fp8=True, block_shape=[128, 64])

    assert len(seen) == 2
    assert seen[0][0] is expert_problem.A
    assert seen[1][0] is result.cache2
    assert [group_size for _, group_size in seen] == [64, 64]


def test_the_block_shape_reaches_both_expert_gemms(
    monkeypatch, expert_problem, native_stubs
):
    """Under block-quantized FP8 both GEMMs must see the same block shape.

    The kernel reads its scales per block only when told the block shape; a
    call that drops it silently runs the per-tensor path instead. The native
    FP8 check depends on this boundary, so it is pinned here on CPU.
    """

    monkeypatch.setattr(
        kernel,
        "quantize_activations_to_fp8",
        lambda tensor, *, group_size: (tensor, torch.ones(1)),
    )

    _run(expert_problem, use_fp8=True, block_shape=[128, 64])

    gemms = [call for call in native_stubs if call.name == "invoke_kernel"]
    assert len(gemms) == 2
    assert [call.block_shape for call in gemms] == [[128, 64], [128, 64]]


def test_a_missing_block_shape_reaches_the_gemms_as_none(expert_problem, native_stubs):
    """The default is observable, so the FP8 test cannot omit it unnoticed."""

    _run(expert_problem)

    gemms = [call for call in native_stubs if call.name == "invoke_kernel"]
    assert [call.block_shape for call in gemms] == [None, None]


def test_repeated_iterations_do_not_leak_a_previous_result(
    expert_problem, native_stubs
):
    """Workspace reuse across profiling steps must not carry stale outputs."""

    cache1 = torch.empty(NUM_TOKENS, TOP_K, 2 * EXPERT_HIDDEN_DIM)
    cache2 = torch.empty(NUM_TOKENS * TOP_K, EXPERT_HIDDEN_DIM)
    cache3 = torch.empty(NUM_TOKENS, TOP_K, HIDDEN_DIM)
    out = torch.empty(NUM_TOKENS, HIDDEN_DIM)

    def run_with(activations):
        kernel._run_fused_moe_iteration(
            A=activations,
            w1=expert_problem.w1,
            w2=expert_problem.w2,
            intermediate_cache1=cache1,
            intermediate_cache2=cache2,
            intermediate_cache3=cache3,
            out_hidden_states=out,
            topk_weights=expert_problem.topk_weights,
            sorted_token_ids=torch.zeros(1, dtype=torch.int32),
            expert_ids=torch.zeros(1, dtype=torch.int32),
            num_tokens_post_padded=torch.zeros(1, dtype=torch.int32),
            top_k=TOP_K,
            config={"BLOCK_SIZE_M": 16},
        )
        return out.clone()

    first = run_with(expert_problem.A)
    second = run_with(expert_problem.A * 2.0)

    assert not torch.allclose(first, second)
    expert_problem.A = expert_problem.A * 2.0
    torch.testing.assert_close(second, _expected_output(expert_problem))
    assert torch.isfinite(second).all()


def _profile_on_cpu(monkeypatch, **arguments):
    """Run `profile_fused_moe_kernel` on CPU with every native call replaced.

    Returns the keyword arguments of each expert iteration and the dtype flags
    of each kernel-config lookup.
    """

    from contextlib import nullcontext
    from unittest.mock import Mock

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    proxy = SimpleNamespace(
        version=SimpleNamespace(hip=None, cuda="12.8"),
        cuda=SimpleNamespace(Event=Event, synchronize=Mock()),
        randn=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"]),
        empty=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"]),
        bfloat16=torch.bfloat16,
        float16=torch.float16,
        float32=torch.float32,
        int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", proxy)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "0.10.x")
    config_lookups = []
    monkeypatch.setattr(
        kernel,
        "get_config_dtype_str",
        lambda dtype, **flags: config_lookups.append((dtype, flags)) or "float16",
        raising=False,
    )
    monkeypatch.setattr(
        kernel,
        "try_get_optimal_moe_config",
        lambda **kwargs: {"BLOCK_SIZE_M": 16},
        raising=False,
    )
    monkeypatch.setattr(
        kernel,
        "moe_align_block_size",
        lambda *args, **kwargs: (None, None, None),
        raising=False,
    )
    monkeypatch.setattr(
        "frontier.profiling.common.vllm_compat.vllm_config_context",
        lambda **kwargs: nullcontext(),
    )

    iterations = []
    monkeypatch.setattr(
        kernel, "_run_fused_moe_iteration", lambda **kwargs: iterations.append(kwargs)
    )

    routing = Mock()
    routing.to.return_value = routing
    routing.contiguous.return_value = routing
    kernel.profile_fused_moe_kernel(
        topk_weights=routing,
        topk_ids=routing,
        warmup_steps=1,
        active_steps=2,
        profile_method="cuda_event",
        **arguments,
    )
    return iterations, config_lookups


def test_the_profiler_allocates_the_four_buffers_the_computation_needs(monkeypatch):
    """Allocation-site dimensions, including the two buffers the repair adds.

    `intermediate_cache2` holds one activated row per routed token, and
    `out_hidden_states` holds one reduced row per token. Getting either shape
    wrong would silently change what the second GEMM and the reduction see.
    """

    num_tokens, top_k = 3, 2
    hidden_dim, expert_hidden_dim, tensor_parallel_size = 8, 16, 2
    expert_hidden_per_partition = expert_hidden_dim // tensor_parallel_size

    seen, _ = _profile_on_cpu(
        monkeypatch,
        num_tokens=num_tokens,
        num_experts=2,
        hidden_dim=hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        top_k=top_k,
        tensor_parallel_size=tensor_parallel_size,
    )

    assert seen, "the legacy path did not run"
    call = seen[0]
    assert call["intermediate_cache1"].shape == (
        num_tokens,
        top_k,
        2 * expert_hidden_per_partition,
    )
    assert call["intermediate_cache2"].shape == (
        num_tokens * top_k,
        expert_hidden_per_partition,
    )
    assert call["intermediate_cache3"].shape == (num_tokens, top_k, hidden_dim)
    assert call["out_hidden_states"].shape == (num_tokens, hidden_dim)

    # Every profiled step reuses one workspace, so the measured cost is the
    # computation and not repeated allocation.
    assert all(
        later["intermediate_cache2"] is call["intermediate_cache2"] for later in seen
    )
    assert all(later["out_hidden_states"] is call["out_hidden_states"] for later in seen)


@pytest.mark.parametrize("use_fp8", [False, True])
def test_the_kernel_config_is_looked_up_for_the_profiled_quantization(
    monkeypatch, use_fp8
):
    """vLLM's `fused_experts_impl` selects the FP8 tile config under W8A8.

    Looking it up by the unquantized dtype alone returns the BF16 config, whose
    tile sizes differ from the ones vLLM runs for FP8.
    """

    if use_fp8:
        monkeypatch.setattr(
            kernel,
            "quantize_weights_to_fp8",
            lambda weights, **_: (weights, torch.ones(1)),
        )
        monkeypatch.setattr(
            kernel,
            "quantize_activations_to_fp8",
            lambda tensor, *, group_size: (tensor, torch.ones(1)),
        )

    _, lookups = _profile_on_cpu(
        monkeypatch,
        num_tokens=4,
        num_experts=2,
        hidden_dim=8,
        expert_hidden_dim=8,
        top_k=2,
        use_fp8=use_fp8,
        block_shape=[128, 128] if use_fp8 else None,
    )

    assert lookups == [(torch.bfloat16 if use_fp8 else torch.float16, {"use_fp8_w8a8": use_fp8})]


@pytest.mark.parametrize(
    "operand_dtype, output_dtype, use_fp8, expected",
    [
        (torch.float16, torch.float16, False, "fp16"),
        (torch.bfloat16, torch.bfloat16, False, "bf16"),
        (torch.float8_e4m3fn, torch.bfloat16, True, "bf16"),
    ],
)
def test_the_compute_type_follows_the_unquantized_dtype(
    monkeypatch, operand_dtype, output_dtype, use_fp8, expected
):
    """Under FP8 the kernel computes in the hidden-state dtype, as vLLM does."""

    seen = {}
    monkeypatch.setattr(
        kernel,
        "invoke_fused_moe_kernel",
        lambda **kwargs: seen.update(kwargs),
        raising=False,
    )

    kernel._invoke_kernel(
        A=torch.zeros(2, 4).to(operand_dtype),
        B=torch.zeros(1, 4, 4).to(operand_dtype),
        C=torch.zeros(2, 1, 4, dtype=output_dtype),
        topk_weights=torch.zeros(2, 1),
        sorted_token_ids=torch.zeros(1, dtype=torch.int32),
        expert_ids=torch.zeros(1, dtype=torch.int32),
        num_tokens_post_padded=torch.zeros(1, dtype=torch.int32),
        mul_routed_weight=False,
        top_k=1,
        config={"BLOCK_SIZE_M": 16},
        use_fp8=use_fp8,
    )

    assert seen["compute_type"] == {"fp16": kernel.tl.float16, "bf16": kernel.tl.bfloat16}[expected]
    assert seen["use_fp8_w8a8"] is use_fp8
