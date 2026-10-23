"""vLLM-aligned memory accounting primitives for runtime non-KV profiling."""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass, field
from typing import Generator

import torch

from frontier.profiling.non_kv_cache_overhead.types import NonKVMemoryBreakdown


def _ensure_cuda_runtime_available() -> None:
    if not hasattr(torch, "cuda"):
        raise RuntimeError("torch.cuda is unavailable; CUDA runtime is required")

    required_apis = (
        "memory_stats",
        "mem_get_info",
        "memory_reserved",
        "empty_cache",
        "reset_peak_memory_stats",
    )
    for api_name in required_apis:
        if not hasattr(torch.cuda, api_name):
            raise RuntimeError(f"torch.cuda.{api_name} is required for memory profiling")

    if hasattr(torch.cuda, "is_available") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available for memory profiling")


@dataclass
class MemorySnapshot:
    """Memory snapshot in bytes."""

    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self) -> None:
        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        """Capture current CUDA memory statistics."""
        _ensure_cuda_runtime_available()

        stats = torch.cuda.memory_stats()
        self.torch_peak = int(stats.get("allocated_bytes.all.peak", 0))

        free_memory, total_memory = torch.cuda.mem_get_info()
        self.free_memory = int(free_memory)
        self.total_memory = int(total_memory)

        if self.free_memory > self.total_memory:
            raise RuntimeError(
                "Invalid CUDA memory snapshot: free_memory exceeds total_memory, "
                f"free={self.free_memory}, total={self.total_memory}"
            )

        self.cuda_memory = self.total_memory - self.free_memory
        self.torch_memory = int(torch.cuda.memory_reserved())
        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        if not isinstance(other, MemorySnapshot):
            raise TypeError(
                f"MemorySnapshot subtraction requires MemorySnapshot, got {type(other)}"
            )

        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(
        default_factory=lambda: MemorySnapshot(auto_measure=False)
    )
    before_profile: MemorySnapshot = field(
        default_factory=lambda: MemorySnapshot(auto_measure=False)
    )
    after_profile: MemorySnapshot = field(
        default_factory=lambda: MemorySnapshot(auto_measure=False)
    )
    profile_time: float = 0.0

    def to_breakdown(self) -> NonKVMemoryBreakdown:
        return NonKVMemoryBreakdown(
            weights_memory_bytes=int(self.weights_memory),
            torch_peak_increase_bytes=int(self.torch_peak_increase),
            non_torch_increase_bytes=int(self.non_torch_increase),
            non_kv_cache_memory_bytes=int(self.non_kv_cache_memory),
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    *,
    weights_memory: int,
) -> Generator[MemoryProfilingResult, None, None]:
    """vLLM-aligned memory profiling context manager.

    Formula:
        non_kv_cache_memory = weights_memory + torch_peak_increase + non_torch_increase
    """
    if not isinstance(baseline_snapshot, MemorySnapshot):
        raise TypeError(
            "baseline_snapshot must be MemorySnapshot, "
            f"got={type(baseline_snapshot)}"
        )

    if weights_memory < 0:
        raise ValueError(f"weights_memory must be >= 0, got={weights_memory!r}")

    _ensure_cuda_runtime_available()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()
    result.before_create = baseline_snapshot
    result.weights_memory = int(weights_memory)
    result.before_profile.measure()

    try:
        yield result
    finally:
        gc.collect()
        torch.cuda.empty_cache()

        result.after_profile.measure()

        diff_profile = result.after_profile - result.before_profile
        diff_from_create = result.after_profile - result.before_create

        result.torch_peak_increase = int(diff_profile.torch_peak)
        result.non_torch_increase = int(diff_from_create.non_torch_memory)
        result.profile_time = float(diff_profile.timestamp)
        result.non_kv_cache_memory = int(
            result.non_torch_increase
            + result.torch_peak_increase
            + result.weights_memory
        )

        if result.non_kv_cache_memory < 0:
            raise RuntimeError(
                "non_kv_cache_memory is negative after profiling, "
                f"result={result}"
            )
