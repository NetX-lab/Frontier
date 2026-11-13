"""Shared fixtures for the unit tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def cpu_vllm_platform(monkeypatch):
    """Resolve vLLM's platform to CPU so vLLM configs build on a host without GPUs.

    A CUDA build of vLLM on such a host detects no platform, and constructing
    ``VllmConfig`` then fails. Tests that build vLLM MoE kernels on CPU request
    this fixture; they choose the backend themselves, so no GPU code runs.
    """

    pytest.importorskip("torch")
    platforms = pytest.importorskip("vllm.platforms")
    from vllm.platforms.cpu import CpuPlatform

    platform = CpuPlatform()
    monkeypatch.setattr(platforms, "_current_platform", platform)
    return platform


@pytest.fixture
def cuda_runtime(monkeypatch):
    """Run ``profile_fused_moe_kernel`` on CPU with the native experts replaced.

    The CUDA calls of the kernel module become CPU stand-ins, the vLLM expert
    builder and the expert-parallel layout become mocks, and every timed step
    reports a 2.5 ms mean. ``args`` holds a GLM-5.3 block-FP8 EP32 call; tests
    update it for other layer configurations.
    """

    torch = pytest.importorskip("torch")
    from frontier.profiling.moe import moe_vllm_kernel as kernel

    runtime = SimpleNamespace(
        version=SimpleNamespace(hip=None, cuda="12.8"),
        cuda=SimpleNamespace(synchronize=Mock()),
        randn=Mock(side_effect=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"])),
        empty_like=torch.empty_like,
        bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32,
        int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", runtime)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    monkeypatch.setattr(
        "frontier.profiling.common.vllm_compat.vllm_config_context", lambda: nullcontext()
    )
    monkeypatch.setattr(
        kernel,
        "_collect_cuda_event_stats",
        lambda step_fn, active_steps: (step_fn(), {"mean": 2.5})[1],
    )
    routed_experts = SimpleNamespace(
        backend="FLASHINFER_CUTLASS",
        experts_class="FlashInferExperts",
        triton_tuning_config=None,
        apply=Mock(),
    )
    build = Mock(return_value=routed_experts)
    monkeypatch.setattr(kernel, "get_native_routed_experts", build)
    layout = Mock(return_value=(32, 0))
    monkeypatch.setattr(kernel, "expert_parallel_layout", layout)
    routing = Mock()
    routing.to.return_value = routing
    routing.contiguous.return_value = routing
    expert_map = Mock()
    expert_map.numel.return_value = 256
    expert_map.to.return_value = expert_map
    args = dict(
        num_tokens=4, num_experts=8, hidden_dim=6144, expert_hidden_dim=2048, top_k=8,
        topk_weights=routing, topk_ids=routing, warmup_steps=0, active_steps=2,
        profile_method="cuda_event", use_fp8=True, block_shape=[128, 128],
        global_num_experts=256, expert_map=expert_map, model_type="glm_moe_dsa",
    )
    return SimpleNamespace(
        kernel=kernel, runtime=runtime, args=args, build=build, layout=layout,
        routed_experts=routed_experts, routing=routing, expert_map=expert_map,
        torch=torch,
    )
