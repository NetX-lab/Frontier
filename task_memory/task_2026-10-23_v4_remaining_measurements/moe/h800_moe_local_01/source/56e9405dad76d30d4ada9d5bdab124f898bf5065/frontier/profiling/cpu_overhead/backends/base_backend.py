"""Backend interface for CPU overhead profiling runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCpuOverheadProfilerBackend(ABC):
    """Abstract backend for creating CPU overhead profiling runners."""

    def start(self) -> None:
        """Initialize backend runtime resources."""

    def stop(self) -> None:
        """Release backend runtime resources."""

    def get_local_gpu_capacity(self) -> int | None:
        """Return measurable TP capacity on one node, if runtime can detect it."""
        return None

    def run_runner(self, runner: Any) -> dict:
        """Execute a backend-specific runner and return one profiling row."""
        raise NotImplementedError(
            f"Backend '{self.name}' must implement run_runner(runner)."
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name used by CLI."""

    @abstractmethod
    def create_runner(
        self,
        model_name: str,
        batch_size: int,
        tensor_parallel_degree: int,
        output_dir: str,
        precision: str,
    ) -> Any:
        """Create a backend-specific runner actor/object."""

