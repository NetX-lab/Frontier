"""Contract tests for the shared FRONTIER_TMP_ROOT scratch-root override."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scratch_root import (
    FALLBACK_SCRATCH_ROOT,
    SCRATCH_ROOT_ENV,
    resolve_scratch_root,
)


def test_env_name_and_fallback_are_stable() -> None:
    assert SCRATCH_ROOT_ENV == "FRONTIER_TMP_ROOT"
    assert FALLBACK_SCRATCH_ROOT == Path("/data/ycfeng/tmp")


def test_unset_variable_returns_historical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCRATCH_ROOT_ENV, raising=False)

    assert resolve_scratch_root() == FALLBACK_SCRATCH_ROOT


def test_absolute_override_is_returned_without_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = tmp_path / "does-not-exist-yet"
    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(configured))

    assert resolve_scratch_root() == configured
    assert not configured.exists()


def test_override_is_reread_on_every_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(tmp_path / "first"))
    assert resolve_scratch_root() == tmp_path / "first"

    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(tmp_path / "second"))
    assert resolve_scratch_root() == tmp_path / "second"


@pytest.mark.parametrize("configured_value", ["relative/path", "", "."])
def test_non_absolute_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    monkeypatch.setenv(SCRATCH_ROOT_ENV, configured_value)

    with pytest.raises(ValueError, match="absolute"):
        resolve_scratch_root()
