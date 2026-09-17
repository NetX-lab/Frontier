"""Canonical profiling input-path resolution for predictor families."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

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
    if configured is not None:
        return str(configured)
    if not fallback:
        return ""
    root, extension = os.path.splitext(str(fallback))
    return f"{root}_device_event{extension}"


def substitute_input_path(
    path: str, *, device: str, model: str, network_device: str | None = None,
) -> str:
    """Bind compute placeholders and, when supplied, the network placeholder."""
    if not path:
        return ""
    resolved = str(path).replace("{DEVICE}", str(device)).replace("{MODEL}", str(model))
    if network_device is not None:
        resolved = resolved.replace("{NETWORK_DEVICE}", str(network_device))
    return resolved


def resolve_measurement_input_paths(
    config: Any,
    measurement_type: MeasurementType,
    *,
    device: str,
    model: str,
    network_device: str,
    overrides: Mapping[str, str] | None = None,
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
        kernel_cpu = getattr(config, "cpu_overhead_kernel_only_input_file", None)
        if kernel_cpu is not None:
            cpu_overhead_file = kernel_cpu
    else:
        raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

    suffix = {
        MeasurementType.CUDA_EVENT: "",
        MeasurementType.DEVICE_EVENT: "_device_event",
        MeasurementType.KERNEL_ONLY: "_kernel_only",
    }[measurement_type]
    explicit = overrides if overrides is not None else {}

    def selected(name: str, configured: str) -> str:
        key = f"{name}{suffix}_input_file"
        alias = f"{name}_input_file{suffix}"
        if key in explicit:
            configured = explicit[key]
        elif alias in explicit:
            configured = explicit[alias]
        return substitute_input_path(
            configured, device=device, model=model, network_device=network_device,
        )

    def common(name: str) -> str:
        key = f"{name}_input_file"
        return substitute_input_path(
            explicit.get(key, getattr(config, key, "")),
            device=device, model=model, network_device=network_device,
        )

    return MeasurementInputPaths(
        compute=selected("compute", compute),
        attention=selected("attention", attention),
        moe=selected("moe", moe),
        all_reduce=common("all_reduce"),
        send_recv=common("send_recv"),
        cpu_overhead=selected("cpu_overhead", cpu_overhead_file),
    )


def resolve_training_file_paths(
    config: Any, *, device: str, model: str, network_device: str,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Project canonical family paths into the established public dictionary."""
    resolved = {}
    for measurement_type, suffix in (
        (MeasurementType.CUDA_EVENT, ""),
        (MeasurementType.DEVICE_EVENT, "_device_event"),
        (MeasurementType.KERNEL_ONLY, "_kernel_only"),
    ):
        paths = resolve_measurement_input_paths(
            config, measurement_type, device=device, model=model,
            network_device=network_device, overrides=overrides,
        )
        for name in ("compute", "attention", "moe"):
            resolved[f"{name}{suffix}_input_file"] = getattr(paths, name)
        if measurement_type == MeasurementType.CUDA_EVENT:
            resolved.update(all_reduce_input_file=paths.all_reduce,
                            send_recv_input_file=paths.send_recv,
                            cpu_overhead_input_file=paths.cpu_overhead)
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            resolved["cpu_overhead_kernel_only_input_file"] = paths.cpu_overhead
    for name in ("pp_stage_boundary", "pp_receiver_head", "pp_producer_send_path", "pp_prefill_consumer_active"):
        key = f"{name}_input_file"
        value = (overrides or {}).get(key, getattr(config, key, ""))
        resolved[key] = substitute_input_path(value, device=device, model=model, network_device=network_device)
    return resolved


def resolve_event_measurement_type(replica_config: Any) -> MeasurementType:
    """Resolve a supplied device; malformed metadata never selects a fallback."""
    from frontier.config.device_sku_config import BaseDeviceSKUConfig

    device_config = getattr(replica_config, "device_config", None)
    if device_config is None:
        device_config = BaseDeviceSKUConfig.create_from_type_string(replica_config.device)
    platform = device_config.gpu_platform
    families = {"cuda": MeasurementType.CUDA_EVENT, "rocm": MeasurementType.DEVICE_EVENT}
    if platform not in families:
        raise ValueError(f"Unsupported configured GPU platform: {platform!r}")
    return families[platform]
