"""Factory for CPU overhead profiling backends."""

from __future__ import annotations

from frontier.profiling.cpu_overhead.backends.base_backend import (
    BaseCpuOverheadProfilerBackend,
)
from frontier.profiling.cpu_overhead.backends.sarathi_backend import (
    SarathiCpuOverheadProfilerBackend,
)
from frontier.profiling.cpu_overhead.backends.vllm_backend import (
    VllmCpuOverheadProfilerBackend,
)

_BACKEND_BUILDERS = {
    "sarathi": SarathiCpuOverheadProfilerBackend,
    "vllm": VllmCpuOverheadProfilerBackend,
}


def get_available_cpu_overhead_backends() -> tuple[str, ...]:
    """Return supported backend names."""
    return tuple(_BACKEND_BUILDERS.keys())


def create_cpu_overhead_backend(
    name: str, **backend_kwargs
) -> BaseCpuOverheadProfilerBackend:
    """Build backend instance by name."""
    normalized = (name or "").strip().lower()
    if normalized not in _BACKEND_BUILDERS:
        raise ValueError(
            f"Unsupported CPU overhead backend: '{name}'. "
            f"Supported backends: {list(_BACKEND_BUILDERS.keys())}"
        )
    if normalized == "vllm":
        return _BACKEND_BUILDERS[normalized](
            vllm_cpu_overhead_input_file=backend_kwargs.get(
                "vllm_cpu_overhead_input_file"
            )
        )
    return _BACKEND_BUILDERS[normalized]()

