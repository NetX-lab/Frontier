"""Resolve the writable scratch root shared by Frontier's heavyweight test harnesses.

The wall-time scaling sweep, the MoE-EP baseline replay, and the MoE-EP
non-dummy matrix all write large intermediate outputs under one scratch root.
Historically that root was hard-coded to a developer-specific path. Set
``FRONTIER_TMP_ROOT`` to an absolute, writable directory to relocate it; the
historical path remains the fallback so existing deployments keep working.
"""

from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT_ENV = "FRONTIER_TMP_ROOT"
FALLBACK_SCRATCH_ROOT = Path("/data/ycfeng/tmp")


def resolve_scratch_root() -> Path:
    """Return the configured scratch root without touching the filesystem.

    Reads ``FRONTIER_TMP_ROOT`` on every call so tests and callers that set the
    variable after import are honored. Callers remain responsible for
    resolving symlinks and creating the directory, matching their existing
    behavior for the fallback path.
    """

    configured_value = os.environ.get(SCRATCH_ROOT_ENV)
    if configured_value is None:
        return FALLBACK_SCRATCH_ROOT
    scratch_root = Path(configured_value)
    if not configured_value or not scratch_root.is_absolute():
        raise ValueError(
            f"{SCRATCH_ROOT_ENV} must be an absolute path, got {configured_value!r}"
        )
    return scratch_root
