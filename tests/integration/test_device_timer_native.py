"""Native CUDA/ROCm timer acceptance; independent of GDN hardware support."""

from contextlib import nullcontext

import pytest

from frontier.profiling.common.device_timer import DeviceTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils.singleton import Singleton


@pytest.fixture
def native_torch():
    torch = pytest.importorskip("torch", reason="Native device timer requires PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("Native device timer requires an available CUDA or ROCm GPU")
    return torch


@pytest.mark.parametrize("method", ["cuda_event", "device_event", "perf_counter", "record_function", "kineto"])
def test_native_timer_records_real_device_work(native_torch, method):
    torch = native_torch
    if method == "cuda_event" and torch.version.hip is not None:
        pytest.skip("CUDA_EVENT belongs to the CUDA measurement family; use DEVICE_EVENT on ROCm")
    previous = Singleton._instances.pop(TimerStatsStore, None)
    try:
        store = TimerStatsStore(profile_method=method)
        operand = torch.randn((256, 256), device="cuda")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) if method == "record_function" else nullcontext() as profiler:
            with DeviceTimer("native_matmul"):
                output = operand @ operand
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        if method == "record_function":
            assert any(event.name == "vidur_native_matmul" for event in profiler.events())
        else:
            samples = store.get_times()["native_matmul"]
            assert len(samples) == 1
            assert samples[0] >= 0
    finally:
        Singleton._instances.pop(TimerStatsStore, None)
        if previous is not None:
            Singleton._instances[TimerStatsStore] = previous
