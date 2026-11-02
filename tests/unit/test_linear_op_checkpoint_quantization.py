"""Checkpoint quantization overrides, without a native vLLM installation."""

from __future__ import annotations

from types import SimpleNamespace, ModuleType
import sys

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.fp8_quantization import (
    checkpoint_hf_config, resolve_checkpoint_quantization,
)


class Fp8StandIn:
    def __init__(self, checkpoint):
        self.fields = {
            "is_checkpoint_fp8_serialized": "fp8" in checkpoint["quant_method"],
            "activation_scheme": checkpoint["activation_scheme"],
            "weight_block_size": checkpoint["weight_block_size"],
            "ignored_layers": checkpoint.get("ignored_layers") or [],
        }

    @classmethod
    def override_quantization_method(cls, checkpoint, user_quant, hf_config=None):
        assert user_quant is None
        return None

    @classmethod
    def from_config(cls, checkpoint):
        return cls(checkpoint)

    @classmethod
    def get_name(cls):
        return "fp8"


class V4StandIn(Fp8StandIn):
    seen = None

    @classmethod
    def override_quantization_method(cls, checkpoint, user_quant, hf_config=None):
        cls.seen = hf_config
        if hf_config.model_type == "deepseek_v4":
            return "deepseek_v4_fp8"
        return None

    @classmethod
    def get_name(cls):
        return "deepseek_v4_fp8"


def resolve(monkeypatch, hf):
    """Run the resolver against stand-ins for vLLM's detection rule and registry."""

    registry = {"fp8": Fp8StandIn, "deepseek_v4_fp8": V4StandIn}

    class VllmModelConfig:
        @staticmethod
        def _verify_quantization(state):
            assert state.quantization is None and state.allow_deprecated_quantization is False
            checkpoint = state.model_arch_config.quantization_config
            for config in registry.values():
                override = config.override_quantization_method(checkpoint, None, hf_config=state.hf_config)
                if override is not None:
                    state.quantization = override
                    return
            state.quantization = checkpoint["quant_method"]

    model_module = ModuleType("vllm.config.model")
    model_module.ModelConfig = VllmModelConfig
    quantization = ModuleType("vllm.model_executor.layers.quantization")
    quantization.get_quantization_config = registry.__getitem__
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, quantization.__name__, quantization)
    return resolve_checkpoint_quantization(hf)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_v4_uses_checkpoint_expert_dtype_and_native_override(monkeypatch, expert_dtype):
    model = ModelConfig.from_model_name("deepseek-ai/DeepSeek-V4-Flash")
    model.expert_dtype = expert_dtype
    context = checkpoint_hf_config(model)
    method, config = resolve(monkeypatch, context.hf_config)
    assert method == "deepseek_v4_fp8"
    assert type(config) is V4StandIn
    assert V4StandIn.seen.expert_dtype == expert_dtype
    assert V4StandIn.seen.quantization_config["scale_fmt"] == model.quantization_config.scale_fmt
    assert context.dtype is model.dtype
    assert context.hf_config is context.hf_text_config


@pytest.mark.parametrize("model_name", ["GLM-5.1", "GLM-5.3"])
def test_glm_quantization_keeps_head_kernel_fields_and_checkpoint_ignored_layers(monkeypatch, model_name):
    model = ModelConfig.from_model_name("zai-org/GLM-5.3")
    model.name = "zai-org/" + model_name
    method, config = resolve(monkeypatch, checkpoint_hf_config(model).hf_config)
    assert method == "fp8"
    assert type(config) is Fp8StandIn
    assert config.fields == {"is_checkpoint_fp8_serialized": True,
                             "activation_scheme": "dynamic", "weight_block_size": [128, 128],
                             "ignored_layers": model.quantization_config.ignored_layers}
    assert V4StandIn.seen.model_type == "glm_moe_dsa"
    # Native from_config retains the checkpoint list; only get_quant_method reads it
    # (pinned fp8.py:175-205), which the producer does not call.
    assert len(config.fields["ignored_layers"]) == 541


def test_fp8_kernel_passes_config_instance_and_reports_scale_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    from frontier.profiling.common.parallel_utils.fp8_block_linear import Fp8LinearMethodKernel

    config = V4StandIn({"quant_method": "fp8", "activation_scheme": "dynamic",
                       "weight_block_size": [128, 128]})
    seen = {}
    class Method:
        def __init__(self, quant_config):
            seen["config"] = quant_config
            self.fp8_linear = SimpleNamespace()
        def create_weights(self, layer, **kwargs):
            seen["create"] = kwargs
            layer.weight = torch.nn.Parameter(torch.empty(4, 8), requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(torch.empty(1, 1), requires_grad=False)
        def process_weights_after_loading(self, layer):
            seen["unit_scales"] = layer.weight_scale_inv.detach().tolist()
        def apply(self, layer, x, bias):
            return torch.nn.functional.linear(x, layer.weight, bias)
    fp8 = ModuleType("vllm.model_executor.layers.quantization.fp8")
    fp8.Fp8LinearMethod = Method
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(fp8_dtype=lambda: torch.float32)
    monkeypatch.setitem(sys.modules, fp8.__name__, fp8)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    kernel = Fp8LinearMethodKernel(weight=torch.randn(4, 8), quant_config=config,
                                   logical_widths=[2, 2], input_size=8, output_size=4,
                                   tensor_parallel_size=1)
    assert seen["config"] is config
    assert seen["unit_scales"] == [[1.0]]
    assert seen["create"]["output_partition_sizes"] == [2, 2]
    assert kernel.weight_scale_dtype == "torch.float32"
    assert kernel.quantization_method == "deepseek_v4_fp8"
    assert kernel.apply(torch.randn(3, 8), None).shape == (3, 4)
