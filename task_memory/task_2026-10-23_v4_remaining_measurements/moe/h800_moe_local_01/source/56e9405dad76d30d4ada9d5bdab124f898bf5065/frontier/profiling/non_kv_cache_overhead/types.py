"""Types for runtime non-KV cache overhead profiling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NonKVMemoryBreakdown:
    """Structured non-KV memory components in bytes."""

    weights_memory_bytes: int
    torch_peak_increase_bytes: int
    non_torch_increase_bytes: int
    non_kv_cache_memory_bytes: int

    def __post_init__(self) -> None:
        fields = {
            "weights_memory_bytes": self.weights_memory_bytes,
            "torch_peak_increase_bytes": self.torch_peak_increase_bytes,
            "non_torch_increase_bytes": self.non_torch_increase_bytes,
            "non_kv_cache_memory_bytes": self.non_kv_cache_memory_bytes,
        }
        for name, value in fields.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got={value!r}")

        expected_total = (
            self.weights_memory_bytes
            + self.torch_peak_increase_bytes
            + self.non_torch_increase_bytes
        )
        if self.non_kv_cache_memory_bytes != expected_total:
            raise ValueError(
                "non_kv_cache_memory_bytes must equal "
                "weights_memory_bytes + torch_peak_increase_bytes + non_torch_increase_bytes, "
                f"got={self.non_kv_cache_memory_bytes}, expected={expected_total}"
            )
