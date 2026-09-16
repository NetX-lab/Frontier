"""CPU simulation of native MoE preflight and runtime provenance propagation."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from frontier.profiling.moe import moe_vllm_kernel as kernel
from frontier.profiling.moe.moe_wrapper import MoEWrapper


@pytest.fixture
def native_runtime(monkeypatch):
    runtime = SimpleNamespace(
        version=SimpleNamespace(hip="6.4", cuda=None),
        cuda=SimpleNamespace(synchronize=Mock()),
        randn=Mock(side_effect=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"])),
        bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32,
        int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", runtime)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    monkeypatch.setattr("frontier.profiling.common.vllm_compat.vllm_config_context", lambda: nullcontext())
    monkeypatch.setattr(kernel, "_collect_cuda_event_stats", lambda **kwargs: {"mean": 2.5})
    routing = Mock()
    routing.to.return_value = routing
    routing.contiguous.return_value = routing
    args = dict(num_tokens=2, num_experts=2, hidden_dim=64, expert_hidden_dim=256,
                top_k=1, topk_weights=routing, topk_ids=routing, warmup_steps=0,
                active_steps=2, profile_method="device_event", use_mxfp4=True,
                model_type="qwen3_5_moe_text")
    state = dict(backend="Mxfp4MoeBackend.AITER_MXFP4_MXFP4",
                 layer=SimpleNamespace(w13_weight=object(), w2_weight=object(), activation="silu"),
                 quant_method=SimpleNamespace(moe_kernel=SimpleNamespace(apply=Mock())))
    build_state = Mock(return_value=state)
    monkeypatch.setattr(kernel, "_get_functional_mxfp4_state", build_state)
    return runtime, args, build_state


@pytest.mark.parametrize("updates", [
    {"model_type": None}, {"model_type": "llama"}, {"hidden_dim": 65},
    {"expert_hidden_dim": 80}, {"tensor_parallel_size": 0},
    {"tensor_parallel_size": True},
])
def test_invalid_mxfp4_contract_rejects_before_native_allocation(native_runtime, updates):
    runtime, args, build_state = native_runtime
    args.update(updates)
    with pytest.raises(ValueError):
        kernel.profile_fused_moe_kernel(**args)
    runtime.randn.assert_not_called()
    build_state.assert_not_called()


def test_cuda_cannot_be_labeled_as_aiter_mxfp4(native_runtime):
    runtime, args, build_state = native_runtime
    runtime.version = SimpleNamespace(hip=None, cuda="12.8")
    args["profile_method"] = "cuda_event"
    with pytest.raises(ValueError, match="requires ROCm"):
        kernel.profile_fused_moe_kernel(**args)
    runtime.randn.assert_not_called()
    build_state.assert_not_called()


def test_unimplemented_functional_fp8_rejects_before_allocation(native_runtime):
    runtime, args, build_state = native_runtime
    args.update(use_mxfp4=False, use_fp8=True)
    with pytest.raises(NotImplementedError, match="FP8 profiling"):
        kernel.profile_fused_moe_kernel(**args)
    runtime.randn.assert_not_called()
    build_state.assert_not_called()


def test_mxfp4_wrapper_exports_actual_backend_outside_numeric_stats(native_runtime):
    _, args, build_state = native_runtime
    wrapper = object.__new__(MoEWrapper)
    for name, value in dict(
        model_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        use_vllm_kernel=True, use_mxfp4=True, use_fp8=False,
        num_experts=2, num_experts_per_device=2, hidden_dim=64,
        expert_hidden_dim=256, router_topk=1, num_tensor_parallel_workers=1,
        _dtype=torch.bfloat16, per_channel_quant=False, block_shape=None,
        profile_method="device_event", output_dir=None, quantization_mode="mxfp4",
    ).items():
        setattr(wrapper, name, value)
    row = wrapper.profile_grouped_gemm(2, routing_inputs=dict(
        topk_weights=args["topk_weights"], topk_ids=args["topk_ids"],
        expert_token_counts=[1, 1], global_num_experts=2, expert_map=None,
    ))
    assert row["moe_native_backend"] == "Mxfp4MoeBackend.AITER_MXFP4_MXFP4"
    assert row["moe_grouped_gemm_backend"] == "vllm_aiter_mxfp4"
    assert row["time_stats"] == {"moe_grouped_gemm": {"mean": 2.5}}
    assert build_state.call_args.kwargs["model_type"] == "qwen3_5_moe_text"


@pytest.fixture
def packed_layout():
    layout = kernel.plan_mxfp4_weight_layout(num_experts=2, hidden_dim=64,
                                            expert_hidden_dim=256, use_gated=True)
    layer = SimpleNamespace(**{
        name: torch.empty(layout[key], dtype=torch.uint8)
        for name, key in (("w13_weight", "w13_shape"), ("w2_weight", "w2_shape"),
                          ("w13_weight_scale", "w13_scale_shape"),
                          ("w2_weight_scale", "w2_scale_shape"))
    })
    return layer, layout


def test_materialized_packed_layout_allows_native_reshape(packed_layout):
    layer, layout = packed_layout
    layer.w13_weight_scale = layer.w13_weight_scale.reshape(-1)
    kernel.validate_mxfp4_materialized_layout(layer, layout)


@pytest.mark.parametrize("mutation", ["missing", "unpacked", "truncated"])
def test_materialized_packed_layout_rejects_wrong_storage(packed_layout, mutation):
    layer, layout = packed_layout
    if mutation == "missing":
        del layer.w2_weight_scale
    elif mutation == "unpacked":
        layer.w2_weight_scale = layer.w2_weight_scale.to(torch.bfloat16)
    else:
        layer.w2_weight_scale = layer.w2_weight_scale.flatten()[:-1]
    with pytest.raises(ValueError, match="materialized w2_weight_scale"):
        kernel.validate_mxfp4_materialized_layout(layer, layout)
