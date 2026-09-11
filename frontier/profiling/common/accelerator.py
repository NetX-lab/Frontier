"""Platform-neutral accelerator discovery helpers for profiling entrypoints."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, List, Optional


VISIBLE_DEVICE_ENV_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def accelerator_platform(torch_module: Any) -> str:
    """Return the accelerator platform exposed by a torch module."""
    version = getattr(torch_module, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    if getattr(version, "cuda", None):
        return "cuda"
    return "cpu"


def _active_visibility(
    environ: Mapping[str, str],
) -> tuple[Optional[str], Optional[str]]:
    for name in VISIBLE_DEVICE_ENV_VARS:
        value = environ.get(name, "").strip()
        if value:
            return name, value
    return None, None


def _parse_visible_device_ids(name: str, value: str) -> List[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        return []

    try:
        device_ids = [int(token) for token in tokens]
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must contain comma-separated integer GPU IDs for Frontier "
            f"profiling, got {value!r}."
        ) from exc

    if any(device_id < 0 for device_id in device_ids):
        raise RuntimeError(
            f"{name} contains a negative GPU ID: {value!r}."
        )
    if len(set(device_ids)) != len(device_ids):
        raise RuntimeError(f"{name} contains duplicate GPU IDs: {value!r}.")
    return device_ids


def _run_discovery_command(
    command: list[str],
    command_runner: CommandRunner,
) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return command_runner(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return None


def _discover_with_amd_smi(command_runner: CommandRunner) -> List[int]:
    result = _run_discovery_command(
        ["amd-smi", "list", "--json"], command_runner
    )
    if result is not None and result.returncode == 0:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            device_ids = [
                int(entry["gpu"])
                for entry in payload
                if isinstance(entry, dict) and "gpu" in entry
            ]
            if device_ids:
                return device_ids

    result = _run_discovery_command(["amd-smi", "list"], command_runner)
    if result is None or result.returncode != 0:
        return []
    return [
        int(match.group(1))
        for line in result.stdout.splitlines()
        if (match := re.match(r"^\s*GPU:\s*(\d+)\s*$", line))
    ]


def _discover_with_nvidia_smi(command_runner: CommandRunner) -> List[int]:
    result = _run_discovery_command(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        command_runner,
    )
    if result is None or result.returncode != 0:
        return []
    try:
        return [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except ValueError:
        return []


def get_available_gpu_ids(
    num_gpus: int,
    *,
    environ: Optional[Mapping[str, str]] = None,
    command_runner: CommandRunner = subprocess.run,
) -> List[int]:
    """Discover physical GPU IDs without initializing the torch runtime.

    ROCm visibility variables are checked before the CUDA-compatible variable.
    With no explicit visibility constraint, ``amd-smi`` and then ``nvidia-smi``
    provide external inventory evidence so multiprocessing remains safe.
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus!r}")

    effective_environ = os.environ if environ is None else environ
    visibility_name, visibility_value = _active_visibility(effective_environ)
    if visibility_name is not None and visibility_value is not None:
        available = _parse_visible_device_ids(visibility_name, visibility_value)
        source = f"{visibility_name}={visibility_value!r}"
    else:
        available = _discover_with_amd_smi(command_runner)
        source = "amd-smi"
        if not available:
            available = _discover_with_nvidia_smi(command_runner)
            source = "nvidia-smi"

    if not available:
        visibility_hint = ", ".join(VISIBLE_DEVICE_ENV_VARS)
        raise RuntimeError(
            "Unable to discover any GPUs with accelerator visibility variables, "
            f"amd-smi, or nvidia-smi. Set one of {visibility_hint} explicitly."
        )
    if len(available) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only found {len(available)} from "
            f"{source}: {available}."
        )
    return available[:num_gpus]


def set_process_visible_device(
    device_id: int,
    *,
    torch_module: Any,
    environ: Optional[MutableMapping[str, str]] = None,
) -> str:
    """Bind a not-yet-initialized process to one CUDA or ROCm device.

    PyTorch retains the ``torch.cuda`` API on ROCm, but current vLLM warns
    when ROCm jobs use ``CUDA_VISIBLE_DEVICES``. Set the native visibility
    variables before the first runtime call and remove variables for the
    other platform.
    """
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id!r}")

    effective_environ = os.environ if environ is None else environ
    value = str(device_id)
    platform = accelerator_platform(torch_module)
    if platform == "rocm":
        effective_environ["ROCR_VISIBLE_DEVICES"] = value
        # Do not set both ROCm variables: HIP_VISIBLE_DEVICES is interpreted
        # after ROCR_VISIBLE_DEVICES by some runtime versions, which can hide
        # nonzero devices through a second round of index remapping.
        effective_environ.pop("HIP_VISIBLE_DEVICES", None)
        effective_environ.pop("CUDA_VISIBLE_DEVICES", None)
        return f"ROCR_VISIBLE_DEVICES={value}"
    if platform == "cuda":
        effective_environ["CUDA_VISIBLE_DEVICES"] = value
        effective_environ.pop("ROCR_VISIBLE_DEVICES", None)
        effective_environ.pop("HIP_VISIBLE_DEVICES", None)
        return f"CUDA_VISIBLE_DEVICES={value}"
    raise RuntimeError(
        "Cannot bind a GPU visibility variable because torch exposes no CUDA "
        "or ROCm runtime."
    )


__all__ = [
    "VISIBLE_DEVICE_ENV_VARS",
    "accelerator_platform",
    "get_available_gpu_ids",
    "set_process_visible_device",
]
