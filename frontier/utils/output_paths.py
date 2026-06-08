"""Output path helpers for simulator-generated artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path


_OUTPUT_COMPONENT_PATTERN = re.compile(r"[^a-z0-9]+")


def sanitize_output_component(value: str, field_name: str = "output component") -> str:
    """Return a stable lowercase path component for metrics output taxonomy."""
    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError(f"{field_name} must not be empty")

    sanitized = _OUTPUT_COMPONENT_PATTERN.sub("_", raw_value.lower()).strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    if not sanitized:
        raise ValueError(f"{field_name} must contain at least one alphanumeric character")
    return sanitized


def validate_output_filename(value: str, field_name: str) -> str:
    """Validate that an output filename is a single local file component."""
    filename = str(value).strip()
    if not filename:
        raise ValueError(f"{field_name} must not be empty")
    if os.path.isabs(filename) or "/" in filename or "\\" in filename:
        raise ValueError(f"{field_name} must be a filename, not a path: {value!r}")
    if ".." in filename:
        raise ValueError(f"{field_name} must not contain path traversal: {value!r}")
    return filename


def validate_run_id(value: str) -> str:
    """Validate and sanitize the run-id path component."""
    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("run_id must not be empty")
    if os.path.isabs(raw_value) or "/" in raw_value or "\\" in raw_value:
        raise ValueError(f"run_id must be a single path component: {value!r}")
    if ".." in raw_value:
        raise ValueError(f"run_id must not contain path traversal: {value!r}")
    return sanitize_output_component(raw_value, "run_id")


def build_metrics_run_output_dir(
    *,
    output_root: str,
    model_type: str,
    workload_type: str,
    run_id: str,
) -> str:
    """Build the canonical metrics output directory for one simulation run."""
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root

    return str(
        root
        / sanitize_output_component(model_type, "model_type")
        / sanitize_output_component(workload_type, "workload_type")
        / validate_run_id(run_id)
    )
