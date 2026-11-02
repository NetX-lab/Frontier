"""Sarathi backend for CPU overhead profiling."""

from __future__ import annotations

import binascii
from typing import Any

from frontier.profiling.cpu_overhead.backends.base_backend import (
    BaseCpuOverheadProfilerBackend,
)


def _hex_to_binary(hex_identifier: str) -> bytes:
    return binascii.unhexlify(hex_identifier)


class SarathiCpuOverheadProfilerBackend(BaseCpuOverheadProfilerBackend):
    """Create Ray remote runners backed by Sarathi runtime."""

    def __init__(self) -> None:
        self._local_gpu_capacity: int | None = None

    @property
    def name(self) -> str:
        return "sarathi"

    def start(self) -> None:
        try:
            import ray  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU overhead profiling backend 'sarathi' requires 'ray'. "
                "Install ray before running this profiler."
            ) from exc

        # Force local runtime for single-node profiling semantics.
        ray.init(address="local", ignore_reinit_error=True, log_to_driver=False)
        self._local_gpu_capacity = int(ray.available_resources().get("GPU", 0))
        if self._local_gpu_capacity <= 0:
            raise RuntimeError(
                "CPU overhead profiling backend 'sarathi' did not detect any local GPU "
                "resource. Single-node profiling requires at least one visible GPU."
            )

    def stop(self) -> None:
        try:
            import ray  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError:
            return
        ray.shutdown()
        self._local_gpu_capacity = None

    def get_local_gpu_capacity(self) -> int | None:
        return self._local_gpu_capacity

    def run_runner(self, runner: Any) -> dict:
        try:
            import ray  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU overhead profiling backend 'sarathi' requires 'ray'. "
                "Install ray before running this profiler."
            ) from exc
        return ray.get(runner.run.remote())

    def create_runner(
        self,
        model_name: str,
        batch_size: int,
        tensor_parallel_degree: int,
        output_dir: str,
        precision: str,
    ) -> Any:
        try:
            import ray  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU overhead profiling backend 'sarathi' requires 'ray'. "
                "Install ray before running this profiler."
            ) from exc

        try:
            from frontier.profiling.cpu_overhead.benchmark_runner import (
                BenchmarkRunner,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU overhead profiling backend 'sarathi' requires sarathi runtime "
                "dependencies. Install sarathi before profiling."
            ) from exc

        placement_group_ids = list(ray.util.placement_group_table().keys())
        for placement_group_id in placement_group_ids:
            ray._private.worker.global_worker.core_worker.remove_placement_group(
                ray.PlacementGroupID(_hex_to_binary(placement_group_id))
            )

        runner_class = (
            ray.remote(num_gpus=0)(BenchmarkRunner)
            .options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
            .remote
        )

        return runner_class(
            model_name,
            batch_size,
            tensor_parallel_degree,
            output_dir,
            precision,
        )

