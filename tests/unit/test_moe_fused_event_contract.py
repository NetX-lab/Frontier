"""CPU boundary tests for fused MoE event timing; no native kernels execute."""

from contextlib import nullcontext
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from frontier.profiling.moe import moe_vllm_kernel as kernel
from frontier.profiling.utils import profile_method_to_measurement_type
from frontier.types import MeasurementType


@pytest.fixture
def event_runtime(monkeypatch):
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.5

    proxy = SimpleNamespace(
        version=SimpleNamespace(hip="6.4", cuda=None),
        cuda=SimpleNamespace(Event=Event, synchronize=Mock()),
        randn=Mock(side_effect=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"])),
        tensor=torch.tensor,
        bfloat16=torch.bfloat16, float16=torch.float16,
        float32=torch.float32, int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", proxy)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    functional = Mock()
    monkeypatch.setattr(kernel, "_functional_fused_experts", functional)
    monkeypatch.setattr(
        "frontier.profiling.common.vllm_compat.vllm_config_context",
        lambda **kwargs: nullcontext(),
    )
    routing = Mock()
    routing.to.return_value = routing
    routing.contiguous.return_value = routing
    arguments = dict(
        num_tokens=2, num_experts=2, hidden_dim=32, expert_hidden_dim=32,
        top_k=1, topk_weights=routing, topk_ids=routing,
        warmup_steps=1, active_steps=2,
    )
    return proxy, functional, arguments


@pytest.mark.parametrize("platform,method,measurement", [
    ("rocm", "device_event", MeasurementType.DEVICE_EVENT),
    ("cuda", "cuda_event", MeasurementType.CUDA_EVENT),
    ("cuda", "cuda", MeasurementType.CUDA_EVENT),
])
def test_event_method_reaches_functional_kernel_without_relabeling(event_runtime, platform, method, measurement):
    runtime, functional, arguments = event_runtime
    runtime.version = SimpleNamespace(
        hip="6.4" if platform == "rocm" else None,
        cuda="12.8" if platform == "cuda" else None,
    )

    result = kernel.profile_fused_moe_kernel(**arguments, profile_method=method)

    assert functional.call_count == 3
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(0.0)
    assert profile_method_to_measurement_type(method) == measurement


@pytest.mark.parametrize("platform,method", [
    ("rocm", "cuda_event"), ("cuda", "device_event"),
])
def test_wrong_platform_event_is_rejected_before_allocation(event_runtime, platform, method):
    runtime, _, arguments = event_runtime
    runtime.version = SimpleNamespace(
        hip="6.4" if platform == "rocm" else None,
        cuda="12.8" if platform == "cuda" else None,
    )

    with pytest.raises(ValueError, match="profiling requires"):
        kernel.profile_fused_moe_kernel(**arguments, profile_method=method)

    runtime.randn.assert_not_called()


def test_single_event_sample_has_finite_zero_standard_deviation(event_runtime):
    result = kernel._collect_cuda_event_stats(Mock(), active_steps=1)

    assert all(math.isfinite(value) for value in result.values())
    assert result["std"] == pytest.approx(0.0)
    assert result["mean"] == pytest.approx(2.5)


def test_event_statistics_match_shared_population_contract(event_runtime, monkeypatch):
    runtime, _, _ = event_runtime
    samples = iter([1.0, 3.0])
    monkeypatch.setattr(runtime.cuda.Event, "elapsed_time", lambda self, other: next(samples))

    result = kernel._collect_cuda_event_stats(Mock(), active_steps=2)

    assert result == pytest.approx({
        "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0, "std": 1.0,
    })


def test_device_events_reach_legacy_low_level_kernel(event_runtime, monkeypatch):
    runtime, _, arguments = event_runtime
    runtime.empty = lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"])
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "0.10.x")
    monkeypatch.setattr(kernel, "get_config_dtype_str", lambda dtype, **flags: "float16", raising=False)
    monkeypatch.setattr(kernel, "try_get_optimal_moe_config", lambda **kwargs: {"BLOCK_SIZE_M": 16}, raising=False)
    monkeypatch.setattr(kernel, "moe_align_block_size", lambda *args, **kwargs: (None, None, None), raising=False)
    iteration = Mock()
    monkeypatch.setattr(kernel, "_run_fused_moe_iteration", iteration)

    result = kernel.profile_fused_moe_kernel(**arguments, profile_method="device_event")

    assert iteration.call_count == 3
    assert result["mean"] == pytest.approx(2.5)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_active_steps_fail_before_allocation(event_runtime, steps):
    runtime, _, arguments = event_runtime
    arguments["active_steps"] = steps

    with pytest.raises(ValueError, match="active_steps"):
        kernel.profile_fused_moe_kernel(**arguments, profile_method="device_event")

    runtime.randn.assert_not_called()


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_invalid_warmup_steps_fail_before_allocation(event_runtime, steps):
    runtime, _, arguments = event_runtime
    arguments["warmup_steps"] = steps

    with pytest.raises(ValueError, match="warmup_steps"):
        kernel.profile_fused_moe_kernel(**arguments, profile_method="device_event")

    runtime.randn.assert_not_called()


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), -0.5])
def test_invalid_event_sample_is_not_exported(event_runtime, monkeypatch, sample):
    runtime, _, _ = event_runtime
    monkeypatch.setattr(runtime.cuda.Event, "elapsed_time", lambda self, other: sample)

    with pytest.raises(ValueError, match="timing sample"):
        kernel._collect_cuda_event_stats(Mock(), active_steps=1)
