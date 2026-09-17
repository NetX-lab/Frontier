"""Exact native profiling admission remains independent of resolved profiles."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.profiling.linear_op import linear_op_impl
from frontier.profiling.moe import moe_vllm_kernel as kernel
from frontier import model_architectures
from frontier.model_architectures import (
    MODEL_ARCHITECTURE_REGISTRY, ModelArchitectureProfile, ModelArchitectureRegistry,
)


TYPE_CASES = (
    (None, False, False), ("", False, False), ("other", False, False),
    ("qwen3_next", True, False), ("qwen3_5_moe_text", True, True),
    ("QWEN3_NEXT", False, False), ("QWEN3_5_MOE_TEXT", False, False),
    (" qwen3_next ", False, False), (" qwen3_5_moe_text ", False, False),
    ("Qwen3_5MoeForCausalLM", False, False),
)


@pytest.mark.parametrize("model_type,gemma,mxfp4", TYPE_CASES)
def test_native_type_policy_ignores_profile_alias_and_normalization(model_type, gemma, mxfp4):
    assert MODEL_ARCHITECTURE_REGISTRY.uses_gemma_rms_norm(model_type) is gemma
    assert MODEL_ARCHITECTURE_REGISTRY.supports_mxfp4_moe(model_type) is mxfp4
    for profile, architectures, norm in product(
        (None, "generic", "qwen3_5_moe"),
        ((), ("Qwen3_5MoeForCausalLM",)),
        (None, "rms_norm", "layer_norm", "RMS_NORM"),
    ):
        config = SimpleNamespace(
            model_type=model_type, norm=norm,
            model_architecture_profile=profile, architectures=architectures,
        )
        assert linear_op_impl._uses_gemma_rms_norm(config) is (gemma and norm == "rms_norm")


@pytest.mark.parametrize("model_type,gemma,mxfp4", TYPE_CASES)
@pytest.mark.parametrize("platform", ("rocm", "cuda", "cpu"))
@pytest.mark.parametrize("api", (None, "0.10.x", "functional_fused_experts"))
def test_mxfp4_preserves_type_platform_api_guard_order(
    monkeypatch, model_type, gemma, mxfp4, platform, api,
):
    detector = Mock(return_value=platform)
    monkeypatch.setattr(kernel, "accelerator_platform", detector)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", api)
    if not mxfp4:
        error_type, message = ValueError, "MXFP4 profiling requires model_type='qwen3_5_moe_text'"
    elif platform != "rocm":
        error_type, message = ValueError, "MXFP4/AITER profiling requires ROCm"
    elif api != "functional_fused_experts":
        error_type = NotImplementedError
        message = "MXFP4 profiling requires current vLLM's modular fused MoE API."
    else:
        kernel.validate_mxfp4_runtime(model_type=model_type)
        detector.assert_called_once_with(kernel.torch)
        return
    with pytest.raises(error_type) as error:
        kernel.validate_mxfp4_runtime(model_type=model_type)
    assert str(error.value) == message
    assert detector.call_count == int(mxfp4)


def test_native_policy_extension_uses_one_existing_profile_declaration(monkeypatch):
    registry = ModelArchitectureRegistry()
    registry.register(replace(
        ModelArchitectureProfile.generic(profile_id="unit_native"),
        gemma_rms_norm_model_types=("unit_model",),
        mxfp4_moe_model_types=("unit_model",),
    ))
    monkeypatch.setattr(linear_op_impl, "MODEL_ARCHITECTURE_REGISTRY", registry)
    monkeypatch.setattr(model_architectures, "MODEL_ARCHITECTURE_REGISTRY", registry)
    monkeypatch.setattr(kernel, "accelerator_platform", lambda _: "rocm")
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    monkeypatch.setattr(registry, "resolve", Mock(side_effect=AssertionError("Unexpected resolution")))

    assert linear_op_impl._uses_gemma_rms_norm(SimpleNamespace(norm="rms_norm", model_type="unit_model"))
    kernel.validate_mxfp4_runtime(model_type="unit_model")
    assert not registry.uses_gemma_rms_norm("UNIT_MODEL")
    assert not registry.supports_mxfp4_moe(" unit_model ")
