"""Canonical profiling input-path resolution for predictor families."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from frontier.types import MeasurementType


@dataclass(frozen=True)
class MeasurementInputPaths:
    """Resolved paths shared by model-manager and predictor entry points."""

    compute: str
    attention: str
    all_reduce: str
    send_recv: str
    cpu_overhead: str
    moe: str


def _derive_device_event_path(config: Any, field_name: str, fallback: str) -> str:
    configured = getattr(config, f"{field_name}_device_event_input_file", None)
    if configured:
        return str(configured)
    if not fallback:
        return ""
    root, extension = os.path.splitext(str(fallback))
    return f"{root}_device_event{extension}"


def _template_substitute(path: str, *, device: str, model: str, network_device: str) -> str:
    if not path:
        return ""
    return (
        str(path)
        .replace("{DEVICE}", str(device))
        .replace("{MODEL}", str(model))
        .replace("{NETWORK_DEVICE}", str(network_device))
    )


def resolve_measurement_input_paths(
    config: Any,
    measurement_type: MeasurementType,
    *,
    device: str,
    model: str,
    network_device: str,
) -> MeasurementInputPaths:
    """Resolve one canonical set of profiling paths.

    Empty configured/fallback paths remain empty so required-input validation
    can report the missing artifact.  DEVICE_EVENT paths derive from the
    corresponding CUDA-family path only when that path is present.
    """

    linear_file = getattr(config, "linear_op_input_file", "") or getattr(
        config, "mlp_input_file", ""
    )
    attention_file = getattr(config, "atten_input_file", "")
    moe_file = getattr(config, "moe_input_file", "")
    cpu_overhead_file = getattr(config, "cpu_overhead_input_file", "")

    if measurement_type == MeasurementType.CUDA_EVENT:
        compute = linear_file
        attention = attention_file
        moe = moe_file
    elif measurement_type == MeasurementType.DEVICE_EVENT:
        compute = _derive_device_event_path(config, "linear_op", linear_file)
        attention = _derive_device_event_path(config, "atten", attention_file)
        moe = _derive_device_event_path(config, "moe", moe_file)
    elif measurement_type == MeasurementType.KERNEL_ONLY:
        compute = getattr(config, "linear_op_kernel_only_input_file", "")
        attention = getattr(config, "atten_kernel_only_input_file", "")
        moe = getattr(config, "moe_kernel_only_input_file", "")
        cpu_overhead_file = getattr(
            config,
            "cpu_overhead_kernel_only_input_file",
            "",
        ) or cpu_overhead_file
    else:
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    return MeasurementInputPaths(
        compute=_template_substitute(
            compute, device=device, model=model, network_device=network_device
        ),
        attention=_template_substitute(
            attention, device=device, model=model, network_device=network_device
        ),
        all_reduce=_template_substitute(
            getattr(config, "all_reduce_input_file", ""),
            device=device,
            model=model,
            network_device=network_device,
        ),
        send_recv=_template_substitute(
            getattr(config, "send_recv_input_file", ""),
            device=device,
            model=model,
            network_device=network_device,
        ),
        cpu_overhead=_template_substitute(
            cpu_overhead_file,
            device=device,
            model=model,
            network_device=network_device,
        ),
        moe=_template_substitute(
            moe, device=device, model=model, network_device=network_device
        ),
    )

