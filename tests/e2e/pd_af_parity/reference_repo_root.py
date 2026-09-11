"""Resolve the pinned Reference repo root used by the PD-AF parity harness.

The parity harness compares the current branch against a pinned, read-only
Reference checkout whose git HEAD and source hashes are asserted at runtime.
Historically the checkout location was hard-coded to a developer-specific
path. Set ``FRONTIER_PDAF_REFERENCE_REPO_ROOT`` to an absolute path to relocate
it; the historical path remains the fallback so existing deployments keep
working. Only the location is configurable: the pinned identity checks are
unchanged.

The returned path is canonical (symlinks and ``..`` resolved) so that the
sidecar written by the bootstrap, the integration driver, and the harness
validator all compare the same string.

``reference_observer_bootstrap.py`` deliberately carries its own copy of this
logic because it runs under a Reference-only ``PYTHONPATH`` and cannot import
this module; ``tests/unit/test_pdaf_reference_repo_root.py`` keeps them in sync.
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
    """Return the canonical configured Reference repo root.

    Uses ``resolve(strict=False)`` so a missing fallback path does not raise; the
    callers that need existence still call ``resolve(strict=True)`` themselves.
    """

    configured_value = os.environ.get(REFERENCE_REPO_ROOT_ENV)
    if configured_value is None:
        return FALLBACK_REFERENCE_REPO_ROOT.resolve(strict=False)
    reference_repo_root = Path(configured_value)
    if not configured_value or not reference_repo_root.is_absolute():
        raise ValueError(
            f"{REFERENCE_REPO_ROOT_ENV} must be an absolute path, "
            f"got {configured_value!r}"
        )
    return reference_repo_root.resolve(strict=False)
