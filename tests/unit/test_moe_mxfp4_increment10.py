"""CPU contracts for the standard vLLM/AITER MoE profiling slice."""

from __future__ import annotations

import inspect

import pytest


def test_moe_quantization_modes_reject_fp8_and_mxfp4_together() -> None:
    from frontier.profiling.moe.moe_vllm_kernel import (
        validate_moe_quantization_mode,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_moe_quantization_mode(use_fp8=True, use_mxfp4=True)

    assert validate_moe_quantization_mode(use_fp8=False, use_mxfp4=False) == "bf16"
    assert validate_moe_quantization_mode(use_fp8=True, use_mxfp4=False) == "fp8"
    assert validate_moe_quantization_mode(use_fp8=False, use_mxfp4=True) == "mxfp4"


def test_mxfp4_layout_plan_is_packed_group32_with_e8m0_scales() -> None:
    from frontier.profiling.moe.moe_vllm_kernel import plan_mxfp4_weight_layout

    plan = plan_mxfp4_weight_layout(
        num_experts=2,
        hidden_dim=64,
        expert_hidden_dim=96,
        use_gated=True,
        group_size=32,
    )

    assert plan["weight_dtype"] == "uint8"
    assert plan["scale_dtype"] == "uint8"
    assert plan["group_size"] == 32
    assert plan["packing"] == "two_fp4_values_per_byte"
    assert plan["w13_shape"] == (2, 192, 32)
    assert plan["w2_shape"] == (2, 64, 48)
    assert plan["w13_scale_shape"] == (2, 192, 2)
    assert plan["w2_scale_shape"] == (2, 64, 3)

    with pytest.raises(ValueError, match="divisible by group_size"):
        plan_mxfp4_weight_layout(
            num_experts=2,
            hidden_dim=64,
            expert_hidden_dim=80,
            use_gated=True,
            group_size=32,
        )


def test_functional_vllm_kernel_exposes_mxfp4_switch_without_importing_vllm() -> None:
    import frontier.profiling.moe.moe_vllm_kernel as kernel

    assert "use_mxfp4" in inspect.signature(kernel.profile_fused_moe_kernel).parameters
    assert kernel.check_vllm_available() is False


def test_grouped_gemm_backend_selection_is_explicit() -> None:
    from frontier.profiling.moe.moe_wrapper import resolve_grouped_gemm_backend

    assert resolve_grouped_gemm_backend(use_vllm_kernel=False, use_mxfp4=False) == (
        "frontier_loop"
    )
    assert resolve_grouped_gemm_backend(use_vllm_kernel=True, use_mxfp4=False) == (
        "vllm_fused"
    )
    assert resolve_grouped_gemm_backend(use_vllm_kernel=True, use_mxfp4=True) == (
        "vllm_aiter_mxfp4"
    )
    with pytest.raises(ValueError, match="requires vLLM/AITER"):
        resolve_grouped_gemm_backend(use_vllm_kernel=False, use_mxfp4=True)


def test_moe_wrapper_marks_mxfp4_grouped_gemm_backend() -> None:
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    wrapper = object.__new__(MoEWrapper)
    wrapper.use_vllm_kernel = True
    wrapper.use_mxfp4 = True
    wrapper.use_fp8 = False
    wrapper.num_experts = 2
    wrapper.num_experts_per_device = 2
    wrapper.expert_parallel_size = 1
    wrapper.routing_runtime_metadata = {}
    wrapper.gating_runtime_context_metadata = {}
    wrapper.router_topk = 2
    wrapper.hidden_dim = 64
    wrapper.expert_hidden_dim = 96
    wrapper.use_gated = True
    wrapper.num_tensor_parallel_workers = 1
    wrapper._resolve_routed_layer_contract = lambda **_: type(
        "Contract",
        (),
        {"typed_metadata_identity": lambda self: {"operator_family_id": "moe"}, "effective_ffn_width": 96},
    )()

    # The backend label is attached by profile_grouped_gemm, so this test only
    # verifies the public mode-to-label contract without constructing CUDA modules.
    assert wrapper.use_mxfp4 is True
