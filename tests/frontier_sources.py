"""Enumerate the Python sources Frontier owns, for repository-governance tests.

`frontier/cc_backend/backends/collective-sim` is an optional vendored submodule.
A default checkout leaves it empty and a developer populates it only to run
`--cc_backend_config_type collective_sim`, so any scan that walks `frontier/`
sees a different file set on the two sides, and one vendored file does not parse
under Python 3 at all. These scans govern Frontier's own code, so they skip it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


VENDORED_SUBTREES = ("frontier/cc_backend/backends/collective-sim",)


def iter_frontier_sources(repo_root: Path) -> Iterator[Path]:
    """Yield every Python file under `frontier/` that Frontier itself owns."""
    vendored = tuple((repo_root / subtree).resolve() for subtree in VENDORED_SUBTREES)
    for path in (repo_root / "frontier").rglob("*.py"):
        resolved = path.resolve()
        if any(resolved.is_relative_to(root) for root in vendored):
            continue
        yield path
