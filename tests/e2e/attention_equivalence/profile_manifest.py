#!/usr/bin/env python3
"""Manifest helpers for deterministic attention equivalence checks."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ENVIRONMENT_VARIABLES = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "WANDB_DISABLED",
    "VIDUR_DISABLE_WANDB",
)


def require_file(path: str | Path) -> Path:
    """Return a required file path or fail fast with a clear error."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required file missing: {resolved}")
    return resolved


def file_sha256(path: str | Path) -> str:
    """Compute SHA256 for a required file."""

    resolved = require_file(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: str | Path) -> dict[str, Any]:
    """Build a deterministic manifest entry for one file."""

    resolved = require_file(path)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def files_manifest(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Build named file manifests, failing fast if any file is absent."""

    return {name: file_manifest(path) for name, path in sorted(paths.items())}


def environment_manifest(
    package_names: Iterable[str] = (),
    env_var_names: Iterable[str] = DEFAULT_ENVIRONMENT_VARIABLES,
) -> dict[str, Any]:
    """Record the Python/runtime environment used for an equivalence run."""

    packages: dict[str, str] = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            packages[package_name] = "MISSING"

    return {
        "python_bin": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "environment_variables": {
            name: os.environ.get(name) for name in env_var_names if name in os.environ
        },
    }


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON report."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
