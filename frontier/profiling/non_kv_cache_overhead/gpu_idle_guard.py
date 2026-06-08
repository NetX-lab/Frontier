"""GPU idle guard utilities for deterministic profiling runs."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class GPUComputeProcess:
    """One compute process reported by nvidia-smi."""

    gpu_id: int
    pid: int
    process_name: str
    used_memory_mib: int


def _run_nvidia_smi_query(gpu_id: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi query failed for GPU "
            f"{gpu_id}: returncode={result.returncode}, stderr={result.stderr.strip()}"
        )

    return result.stdout


def get_gpu_compute_processes(gpu_id: int) -> List[GPUComputeProcess]:
    """Return active compute processes on the given GPU."""
    if gpu_id < 0:
        raise ValueError(f"gpu_id must be >= 0, got={gpu_id!r}")

    raw_output = _run_nvidia_smi_query(gpu_id)
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    processes: List[GPUComputeProcess] = []

    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise RuntimeError(
                "Unexpected nvidia-smi output format for compute apps, "
                f"line={line!r}"
            )

        pid_text, process_name, used_memory_text = parts

        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to parse pid from nvidia-smi output: {pid_text!r}"
            ) from exc

        try:
            used_memory_mib = int(used_memory_text)
        except ValueError as exc:
            raise RuntimeError(
                "Failed to parse used_memory from nvidia-smi output: "
                f"{used_memory_text!r}"
            ) from exc

        processes.append(
            GPUComputeProcess(
                gpu_id=gpu_id,
                pid=pid,
                process_name=process_name,
                used_memory_mib=used_memory_mib,
            )
        )

    return processes


def assert_gpu_idle(gpu_id: int) -> None:
    """Raise RuntimeError if the given GPU has active compute processes."""
    processes = get_gpu_compute_processes(gpu_id)
    if processes:
        details = "; ".join(
            f"pid={proc.pid}, name={proc.process_name}, used_memory_mib={proc.used_memory_mib}"
            for proc in processes
        )
        raise RuntimeError(
            f"GPU {gpu_id} is not idle. Active compute processes: {details}"
        )


def assert_gpus_idle(gpu_ids: Iterable[int]) -> None:
    """Raise RuntimeError if any GPU in the iterable is not idle."""
    checked = list(gpu_ids)
    if not checked:
        raise ValueError("gpu_ids must not be empty")

    for gpu_id in checked:
        assert_gpu_idle(int(gpu_id))


def wait_for_gpus_idle(
    gpu_ids: Iterable[int],
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Wait until all GPUs are idle, or raise TimeoutError."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got={timeout_s!r}")
    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got={poll_interval_s!r}")

    checked = [int(gpu_id) for gpu_id in gpu_ids]
    if not checked:
        raise ValueError("gpu_ids must not be empty")

    deadline = time.monotonic() + float(timeout_s)

    while True:
        busy = []
        for gpu_id in checked:
            processes = get_gpu_compute_processes(gpu_id)
            if processes:
                busy.append((gpu_id, processes))

        if not busy:
            return

        if time.monotonic() >= deadline:
            details = []
            for gpu_id, processes in busy:
                proc_details = ", ".join(
                    f"pid={proc.pid}:{proc.process_name}:{proc.used_memory_mib}MiB"
                    for proc in processes
                )
                details.append(f"gpu={gpu_id}[{proc_details}]")
            raise TimeoutError(
                "Timed out waiting for GPUs to become idle: " + " | ".join(details)
            )

        time.sleep(float(poll_interval_s))
