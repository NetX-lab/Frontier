"""Resolve the pinned Reference repo root used by the PD-AF parity harness.

The parity harness compares the current branch against a pinned, read-only
Reference checkout whose git HEAD and source hashes are asserted at runtime.
Historically the checkout location was hard-coded to a developer-specific
path. Set ``FRONTIER_PDAF_REFERENCE_REPO_ROOT`` to an absolute path to relocate
it; the historical path remains the fallback so existing deployments keep
working. Only the location is configurable: the pinned identity checks are
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

REFERENCE_REPO_ROOT_ENV = "FRONTIER_PDAF_REFERENCE_REPO_ROOT"
FALLBACK_REFERENCE_REPO_ROOT = Path(
    "/data/ycfeng/stepfun-performance-optimization/Frontier/"
    "worktrees/ref-afd-readonly"
)


def resolve_reference_repo_root() -> Path:
    """Return the configured Reference repo root without touching the filesystem."""

    configured_value = os.environ.get(REFERENCE_REPO_ROOT_ENV)
    if configured_value is None:
        return FALLBACK_REFERENCE_REPO_ROOT
    reference_repo_root = Path(configured_value)
    if not configured_value or not reference_repo_root.is_absolute():
        raise ValueError(
            f"{REFERENCE_REPO_ROOT_ENV} must be an absolute path, "
            f"got {configured_value!r}"
        )
    return reference_repo_root
