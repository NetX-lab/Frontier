"""CPU contracts for native Step linear activation, normalization and FFN merge."""

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F

from frontier.profiling.common.layers import activation, layernorm
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_utils import tensor_parallel_layers
from frontier.profiling.linear_op import linear_op_impl as linear
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan


STEP5 = "stepfun-ai/Step-5-Preview"


@pytest.fixture(autouse=True)
def cpu_timing(monkeypatch):
    for module in (layernorm, linear, tensor_parallel_layers):
        monkeypatch.setattr(module, "CudaTimer", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(linear, "raise_if_fp8_requested", lambda *args: None)
    linear._load_step5_add_fp32.cache_clear()
    yield
    linear._load_step5_add_fp32.cache_clear()


@pytest.fixture
def config():
    return ModelConfig.from_model_name(STEP5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1728, 192])
@pytest.mark.parametrize("limit", [7.0, 3.5])
def test_step_activation_clamps_after_silu_in_fp32(dtype, width, limit):
    gate = torch.tensor([-20.0, -8.0, -1.0, 0.0, 7.03125, 8.0, 20.0, 1.0])
    up = torch.tensor([20.0, -20.0, -8.0, 0.0, 8.0, 2.0, 1.015625, 0.5])
    x = torch.cat([gate.repeat(width // 8), up.repeat(width // 8)]).to(dtype)[None, :]
    gate, up = x.float().chunk(2, dim=-1)
    expected = (torch.minimum(F.silu(gate), torch.tensor(limit)) * up.clamp(-limit, limit)).to(dtype)
    result = activation.get_act_fn("swiglustep", limit=limit)(x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == dtype
    assert result.shape == (1, width)
    wrong_order = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(dtype)
    assert not torch.equal(result, wrong_order)


def test_activation_registry_contains_factories_and_preserves_legacy_silu(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(activation, "SiluAndMul", lambda: sentinel)
    monkeypatch.setitem(activation._ACTIVATION_REGISTRY, "silu", activation.SiluAndMul)
    assert activation.get_act_fn("SILU") is sentinel
    assert activation.get_act_fn("gelu") is not activation.get_act_fn("gelu")
    x = torch.tensor([-1.0, 0.0, 1.0])
    for name in ("gelu_fast", "gelu_new", "gelu_pytorch_tanh"):
        torch.testing.assert_close(activation.get_act_fn(name)(x), F.gelu(x, approximate="tanh"))
    with pytest.raises(ValueError, match="requires limit"):
        activation.get_act_fn("swiglustep")
    with pytest.raises(ValueError, match="not supported"):
        activation.get_act_fn("unknown")


def test_step_activation_cuda_dispatch_is_lazy_and_uses_native_signature(monkeypatch):
    module = ModuleType("vllm.model_executor.layers.activation")
    calls = []

    def kernel(output, input, limit):
        calls.append((output, input, limit))
        output.fill_(2)

    module.swiglustep_and_mul_triton = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    act = activation.get_act_fn("swiglustep", limit=7.0)
    assert act._op is None
    x = SimpleNamespace(is_cuda=True, shape=(2, 384), dtype=torch.bfloat16, device="cpu")
    output = act(x)
    assert calls == [(output, x, 7.0)]
    assert output.shape == (2, 192)
    assert output.dtype == torch.bfloat16
    assert act.native_backend.endswith("swiglustep_and_mul_triton")


def test_legacy_silu_keeps_custom_op_dispatch(monkeypatch):
    calls = []

    def kernel(output, input):
        calls.append((output, input))
        output.fill_(4)

    monkeypatch.setattr(torch.ops._C, "silu_and_mul", kernel, raising=False)
    x = torch.zeros(2, 384, dtype=torch.bfloat16)
    output = activation.get_act_fn("silu")(x)
    assert calls == [(output, x)]
    assert output.dtype == x.dtype
    assert output.shape == (2, 192)


def test_step_activation_missing_native_cuda_kernel_fails_without_torch_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.activation", None)
    x = SimpleNamespace(is_cuda=True)
    with pytest.raises(ImportError, match="native.*swiglustep_and_mul_triton"):
        activation.get_act_fn("swiglustep", limit=7.0)(x)


@pytest.mark.parametrize("shared,width", [(False, 13824), (True, 1536)])
def test_mlp_uses_profile_activation_and_native_tp8_widths(config, shared, width):
    mlp = linear.ShareExpertMLP(config, 8) if shared else linear.MLP(config, 8, mlp_hidden_dim=width)
    assert (mlp.activation_name, mlp.activation_limit) == ("swiglustep", 7.0)
    assert isinstance(mlp.act, activation.SwigluStepAndMul)
    assert mlp.up_proj.weight.shape == (2 * width // 8, 4096)
    assert mlp.down_proj.weight.shape == (4096, width // 8)


def test_generic_profile_keeps_legacy_activation(config, monkeypatch):
    from frontier.model_architectures import ModelArchitectureProfile

    monkeypatch.setattr(config, "get_model_architecture_profile", ModelArchitectureProfile.generic)
    sentinel = torch.nn.Identity()
    monkeypatch.setitem(activation._ACTIVATION_REGISTRY, "silu", lambda: sentinel)
    assert linear.MLP(config, 8).act is sentinel
    assert linear.ShareExpertMLP(config, 8).act is sentinel


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_step_norm_reference_uses_zero_centered_fp32_weights_and_bf16_output(config, with_residual, input_dtype):
    norm = linear._build_untimed_norm(config, 8)
    assert isinstance(norm, layernorm.Step5RMSNorm)
    weight = torch.linspace(-0.2, 0.3, 8)
    norm.weight.data.copy_(weight)
    norm.to(dtype=torch.bfloat16)
    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
    x = torch.linspace(-2, 3, 16).reshape(2, 8).to(input_dtype)
    residual = torch.linspace(0, 1, 16).reshape(2, 8) if with_residual else None
    summed = x.float() + residual if with_residual else x.float()
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                * (weight + 1)).to(torch.bfloat16)
    result = norm(x, residual)
    if with_residual:
        result, returned_residual = result
        assert returned_residual.dtype == torch.float32
        torch.testing.assert_close(returned_residual, summed, rtol=0, atol=0)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_step_norm_factory_is_lazy_and_passes_native_dtypes(config, monkeypatch):
    module = ModuleType("vllm.models.step5.nvidia.layernorm")
    calls = []
    forwards = []

    class NativeNorm(torch.nn.Module):
        def forward(self, x, residual, *, out_dtype):
            forwards.append((x, residual, out_dtype, self.weight.dtype))
            return "native norm output"

    def factory(hidden_size, eps=1e-6, zero_centered=False, dtype=None):
        calls.append((hidden_size, eps, zero_centered, dtype))
        return NativeNorm()

    module.make_step5_rms_norm = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    norm = linear._build_untimed_norm(config, 8)
    norm.to(dtype=torch.bfloat16)
    assert calls == []
    x = SimpleNamespace(is_cuda=True)
    residual = torch.zeros(2, 8, dtype=torch.float32)
    assert norm(x, residual) == "native norm output"
    assert calls == [(8, config.rms_norm_eps, True, torch.float32)]
    assert forwards == [(x, residual, torch.bfloat16, torch.float32)]
    assert norm._impl.weight is norm.weight
    norm.to(dtype=torch.bfloat16)
    assert norm._impl.weight.dtype == torch.float32


def test_step_norm_missing_native_cuda_kernel_fails_without_reference_fallback(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.models.step5.nvidia.layernorm", None)
    with pytest.raises(ImportError, match="requires make_step5_rms_norm"):
        linear._build_untimed_norm(config, 8)(SimpleNamespace(is_cuda=True))


def test_step_norm_rejects_non_zero_centered_weights(config):
    config.zero_centered = False
    with pytest.raises(NotImplementedError, match="zero-centered"):
        linear._build_untimed_norm(config, 8)


def test_step_ffn_merge_reference_preserves_fp32_addition():
    routed = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    shared = torch.tensor([[2**-9, 2**-8]], dtype=torch.bfloat16)
    output = linear.step5_ffn_merge(routed, shared)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, routed.float() + shared.float(), rtol=0, atol=0)
    assert not torch.equal(output, (routed + shared).float())


def test_step_ffn_merge_cuda_dispatch_uses_one_native_call(monkeypatch):
    module = ModuleType("vllm.models.step5.nvidia.ops.add")
    calls = []
    expected = torch.ones(2, 8, dtype=torch.float32)

    def add_fp32(x, y):
        calls.append((x, y))
        return expected

    module.add_fp32 = add_fp32
    monkeypatch.setitem(sys.modules, module.__name__, module)
    routed = SimpleNamespace(is_cuda=True)
    shared = object()
    assert linear.step5_ffn_merge(routed, shared) is expected
    assert calls == [(routed, shared)]


def test_step_ffn_merge_missing_native_cuda_kernel_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.models.step5.nvidia.ops.add", None)
    with pytest.raises(ImportError, match="requires native add_fp32"):
        linear.step5_ffn_merge(SimpleNamespace(is_cuda=True), object())


class _Branch(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, x):
        self.calls.append(x)
        return torch.full_like(x, self.value)


@pytest.mark.parametrize("is_moe", [False, True])
def test_step_block_merges_bf16_routed_surrogate_once_without_timing_dense_output(config, monkeypatch, is_moe):
    plan = build_profiling_plan(config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=is_moe)
    dense, shared = _Branch(10), _Branch(3)
    monkeypatch.setattr(linear, "MLP", lambda *args, **kwargs: dense)
    monkeypatch.setattr(linear, "ShareExpertMLP", lambda *args, **kwargs: shared)
    block = linear.GPTBlock(config, 8, profiling_plan=plan)
    assert isinstance(block.attn, linear.DummyAttention)
    assert "add" not in plan["enabled_ops"]
    merges = []
    events = []

    class MergeTimer:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *args):
            events.append("exit")

    def merge(routed, shared_output):
        assert events == ["enter"]
        merges.append((routed, shared_output))
        return routed.float() + shared_output.float()

    block.ffn_merge_timer = MergeTimer()
    monkeypatch.setattr(linear, "step5_ffn_merge", merge)
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    output, residual = block(torch.arange(2), x, torch.zeros_like(x, dtype=torch.float32))
    assert len(merges) == 1
    assert events == ["enter", "exit"]
    assert len(dense.calls) == (0 if is_moe else 1)
    assert len(shared.calls) == 1
    assert merges[0][0].dtype == merges[0][1].dtype == torch.bfloat16
    assert output.dtype == residual.dtype == torch.float32
    assert not torch.equal(merges[0][0], torch.full_like(x, 10))


def test_block_without_merge_keeps_ordinary_branch_addition(config, monkeypatch):
    plan = build_profiling_plan(config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1])
    plan["enabled_ops"].remove("step5_ffn_merge")
    monkeypatch.setattr(linear, "MLP", lambda *args, **kwargs: _Branch(10))
    monkeypatch.setattr(linear, "ShareExpertMLP", lambda *args, **kwargs: _Branch(3))
    block = linear.GPTBlock(config, 8, profiling_plan=plan)
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    output = block._forward_ffn(x)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.full_like(x, 13), rtol=0, atol=0)


def test_attention_builder_ignores_generic_implementation_when_profile_owns_no_projections(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Generic attention must not be constructed for empty sharded_ops")

    monkeypatch.setattr(linear, "CausalSelfAttention", forbidden)
    assert isinstance(linear.build_linear_op_attention_module(config, 8, set(), True), linear.DummyAttention)


@pytest.mark.parametrize("profile_method", ["cuda_event", "record_function"])
def test_model_profiles_native_step_norm_dtypes_for_both_timing_modes(config, monkeypatch, tmp_path, profile_method):
    from frontier.config.quantization_manager import QuantizationManager
    from frontier.profiling.linear_op.linear_op_wrapper import ACTIVE_STEPS, LinearOpWrapper
    from frontier.profiling.utils.singleton import Singleton

    # Keep the real model, block, embedding and normalization adapters. Only
    # device allocation, event timing and the native CUDA factory stand in on
    # CPU; this is not a CUDA timing or first-layer/unfused-norm qualification.
    monkeypatch.setattr(Singleton, "_instances", {})
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    active_scopes = []

    class Scope:
        def __init__(self, name=None, **kwargs):
            self.name = name

        def __enter__(self):
            if self.name is not None:
                active_scopes.append(self.name)

        def __exit__(self, *args):
            if self.name is not None:
                assert active_scopes.pop() == self.name

    for module in (layernorm, linear, tensor_parallel_layers):
        monkeypatch.setattr(module, "CudaTimer", Scope)

    factory_calls, native_calls, embeddings = [], [], []
    native_module = ModuleType("vllm.models.step5.nvidia.layernorm")

    class NativeNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps, zero_centered, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
            self.eps = eps
            self.zero_centered = zero_centered

        def forward(self, x, residual, *, out_dtype):
            native_calls.append((active_scopes[-1], x.dtype, residual.dtype, self.weight.dtype, out_dtype))
            residual.add_(x)
            value = residual.float()
            normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
            return (normalized * (self.weight.float() + float(self.zero_centered))).to(out_dtype), residual

    def factory(hidden_size, eps=1e-6, zero_centered=False, dtype=None):
        factory_calls.append((hidden_size, eps, zero_centered, dtype))
        return NativeNorm(hidden_size, eps, zero_centered, dtype)

    native_module.make_step5_rms_norm = factory
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    config.vocab_size = 16
    plan = build_profiling_plan(config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=True)
    plan["enabled_ops"] = ["emb", "input_layernorm", "post_attention_layernorm"]
    wrapper = LinearOpWrapper(config, 8, profile_method, 0, str(tmp_path), profiling_plan=plan)
    model = wrapper.model
    assert isinstance(model, linear.GPTModel)
    assert isinstance(model.block, linear.GPTBlock)
    expected_repeats = ACTIVE_STEPS if profile_method == "record_function" else 1
    assert model.num_repeat_steps == expected_repeats

    inputs, outputs, weights = {}, {}, {}

    def embedding_hook(module, args, output):
        assert active_scopes == []
        embeddings.append(output.dtype)

    def input_hook(name):
        def hook(module, args):
            assert active_scopes == []
            hidden, residual = args
            assert hidden.data_ptr() != residual.data_ptr()
            inputs[name].append((hidden.dtype, residual.dtype, module.weight.dtype))
        return hook

    def output_hook(name):
        def hook(module, args, output):
            hidden, residual = output
            outputs[name].append((hidden.dtype, residual.dtype))
        return hook

    model.embed_tokens.register_forward_hook(embedding_hook)
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(model.block, name)
        assert isinstance(norm, layernorm.Step5RMSNorm)
        inputs[name], outputs[name] = [], []
        norm.weight.data.copy_(torch.linspace(-0.2, 0.3, config.embedding_dim))
        weights[name] = norm.weight.detach().clone()
        norm.register_forward_pre_hook(input_hook(name))
        norm.register_forward_hook(output_hook(name))

    for _ in range(2):
        # Exercise the actual adapter before and after lazy native creation;
        # converting the enclosing model must not round its FP32 norm weights.
        model.to(dtype=torch.bfloat16)
        for name, expected in weights.items():
            norm = getattr(model.block, name)
            assert norm.weight.dtype == torch.float32
            torch.testing.assert_close(norm.weight, expected, rtol=0, atol=0)
        with torch.inference_mode():
            model(torch.zeros(2, dtype=torch.long), torch.arange(2))

    count = 2 * expected_repeats
    assert embeddings == [torch.bfloat16] * (2 * (expected_repeats + 1))
    assert inputs["input_layernorm"] == [(torch.float32, torch.float32, torch.float32)] * count
    assert inputs["post_attention_layernorm"] == [(torch.bfloat16, torch.float32, torch.float32)] * count
    assert outputs == {
        name: [(torch.bfloat16, torch.float32)] * count
        for name in ("input_layernorm", "post_attention_layernorm")
    }
    assert factory_calls == [(4096, config.rms_norm_eps, True, torch.float32)] * 2
    assert native_calls == [
        (name, input_dtype, torch.float32, torch.float32, torch.bfloat16)
        for _ in range(count)
        for name, input_dtype in (("input_layernorm", torch.float32), ("post_attention_layernorm", torch.bfloat16))
    ]
    for name in weights:
        norm = getattr(model.block, name)
        assert norm._impl.weight is norm.weight
        assert norm._impl.weight.dtype == torch.float32
