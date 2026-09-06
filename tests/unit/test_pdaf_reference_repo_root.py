"""Contract tests for the FRONTIER_PDAF_REFERENCE_REPO_ROOT override."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.pd_af_parity.reference_repo_root import (
    FALLBACK_REFERENCE_REPO_ROOT,
    REFERENCE_REPO_ROOT_ENV,
    resolve_reference_repo_root,
)


def test_env_name_and_fallback_are_stable() -> None:
    assert REFERENCE_REPO_ROOT_ENV == "FRONTIER_PDAF_REFERENCE_REPO_ROOT"
    assert FALLBACK_REFERENCE_REPO_ROOT == Path(
        "/data/ycfeng/stepfun-performance-optimization/Frontier/"
        "worktrees/ref-afd-readonly"
    )


def test_unset_variable_returns_historical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REFERENCE_REPO_ROOT_ENV, raising=False)

    assert resolve_reference_repo_root() == FALLBACK_REFERENCE_REPO_ROOT


def test_absolute_override_is_returned_without_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = tmp_path / "ref-afd-readonly"
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, str(configured))

    assert resolve_reference_repo_root() == configured
    assert not configured.exists()


@pytest.mark.parametrize("configured_value", ["relative/path", "", "."])
def test_non_absolute_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, configured_value)

    with pytest.raises(ValueError, match="absolute"):
        resolve_reference_repo_root()


def test_harness_and_bootstrap_share_the_resolved_root() -> None:
    from tests.e2e.pd_af_parity import harness, reference_observer_bootstrap

    assert harness.REFERENCE_REPO_ROOT == reference_observer_bootstrap.REFERENCE_REPO_ROOT
    assert harness.REFERENCE_REPO_ROOT.is_absolute()
