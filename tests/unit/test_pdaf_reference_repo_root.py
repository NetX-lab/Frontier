"""Contract tests for the FRONTIER_PDAF_REFERENCE_REPO_ROOT override."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.pd_af_parity import harness, reference_observer_bootstrap
from tests.e2e.pd_af_parity.reference_repo_root import (
    FALLBACK_REFERENCE_REPO_ROOT,
    REFERENCE_REPO_ROOT_ENV,
    resolve_reference_repo_root,
)


BOOTSTRAP_SOURCE = Path(reference_observer_bootstrap.__file__).resolve(strict=True)


def test_env_name_and_fallback_are_stable() -> None:
    assert REFERENCE_REPO_ROOT_ENV == "FRONTIER_PDAF_REFERENCE_REPO_ROOT"
    assert FALLBACK_REFERENCE_REPO_ROOT == Path(
        "/data/ycfeng/stepfun-performance-optimization/Frontier/"
        "worktrees/ref-afd-readonly"
    )


def test_unset_variable_returns_canonical_historical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REFERENCE_REPO_ROOT_ENV, raising=False)

    assert resolve_reference_repo_root() == FALLBACK_REFERENCE_REPO_ROOT.resolve(
        strict=False
    )


def test_absolute_override_to_missing_path_does_not_require_existence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = tmp_path / "ref-afd-readonly"
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, str(configured))

    assert resolve_reference_repo_root() == configured.resolve(strict=False)
    assert not configured.exists()


@pytest.mark.parametrize("configured_value", ["relative/path", "", "."])
def test_non_absolute_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, configured_value)

    with pytest.raises(ValueError, match="absolute"):
        resolve_reference_repo_root()


def test_override_containing_parent_segments_is_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_root = tmp_path / "worktrees" / "ref-afd-readonly"
    real_root.mkdir(parents=True)
    dotted = tmp_path / "worktrees" / "other" / ".." / "ref-afd-readonly"
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, str(dotted))

    resolved = resolve_reference_repo_root()

    assert resolved == real_root.resolve(strict=True)
    assert ".." not in resolved.parts


def test_override_through_symlink_is_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_root = tmp_path / "real" / "ref-afd-readonly"
    real_root.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, str(link))

    assert resolve_reference_repo_root() == real_root.resolve(strict=True)


def test_sidecar_root_equals_harness_validation_root_under_symlink_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap writes ``str(_require_reference_root(...))`` into the sidecar
    and the harness compares it against ``str(REFERENCE_REPO_ROOT)``. Both must
    agree even when the configured path goes through a symlink and ``..``."""

    real_root = tmp_path / "real" / "ref-afd-readonly"
    (real_root / "sub").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real_root, target_is_directory=True)
    configured = link / "sub" / ".."
    monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, str(configured))

    harness_root = resolve_reference_repo_root()
    monkeypatch.setattr(
        reference_observer_bootstrap, "REFERENCE_REPO_ROOT", harness_root
    )

    sidecar_root = reference_observer_bootstrap._require_reference_root(str(link))

    assert str(sidecar_root) == str(harness_root)
    assert str(sidecar_root) == str(real_root.resolve(strict=True))


def test_harness_and_bootstrap_share_the_resolved_root() -> None:
    assert harness.REFERENCE_REPO_ROOT == reference_observer_bootstrap.REFERENCE_REPO_ROOT
    assert harness.REFERENCE_REPO_ROOT.is_absolute()


def test_bootstrap_resolver_mirrors_shared_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap carries its own copy of the resolver; keep them in lockstep."""

    assert (
        reference_observer_bootstrap.REFERENCE_REPO_ROOT_ENV
        == REFERENCE_REPO_ROOT_ENV
    )
    assert (
        reference_observer_bootstrap.FALLBACK_REFERENCE_REPO_ROOT
        == FALLBACK_REFERENCE_REPO_ROOT
    )

    real_root = tmp_path / "real"
    real_root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_root, target_is_directory=True)

    for configured in (None, str(link), str(tmp_path / "x" / ".." / "real")):
        if configured is None:
            monkeypatch.delenv(REFERENCE_REPO_ROOT_ENV, raising=False)
        else:
            monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, configured)
        assert (
            reference_observer_bootstrap._resolve_pinned_reference_repo_root()
            == resolve_reference_repo_root()
        ), configured

    for bad in ("relative", ""):
        monkeypatch.setenv(REFERENCE_REPO_ROOT_ENV, bad)
        with pytest.raises(ValueError, match="absolute"):
            reference_observer_bootstrap._resolve_pinned_reference_repo_root()


def test_bootstrap_imports_only_the_standard_library() -> None:
    """The bootstrap runs under a Reference-only PYTHONPATH; importing the
    current repository's ``tests`` package would fail before argument parsing."""

    tree = ast.parse(BOOTSTRAP_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    stdlib = set(sys.stdlib_module_names)
    non_stdlib = sorted(name for name in imported if name not in stdlib)
    assert non_stdlib == [], non_stdlib


def test_bootstrap_reaches_argument_parsing_under_reference_only_pythonpath(
    tmp_path: Path,
) -> None:
    """Mirror the integration test launch: script by absolute path, cwd and
    PYTHONPATH pointing at a tree that does not contain this repository."""

    fake_reference = tmp_path / "ref-afd-readonly"
    fake_reference.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(fake_reference)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop(REFERENCE_REPO_ROOT_ENV, None)

    completed = subprocess.run(
        [sys.executable, "-B", str(BOOTSTRAP_SOURCE)],
        cwd=fake_reference,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # No arguments: the bootstrap's own argument parser must be the thing that
    # rejects the invocation, which proves module import succeeded.
    assert completed.returncode == 2, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "--reference-repo-root" in completed.stderr
    assert "exactly one -- delimiter is required" in completed.stderr


def test_bootstrap_honors_override_under_reference_only_pythonpath(
    tmp_path: Path,
) -> None:
    """With the override set, a mismatching --reference-repo-root must be
    rejected by the bootstrap's own check, proving the env read happened."""

    pinned = tmp_path / "pinned"
    pinned.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(pinned)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[REFERENCE_REPO_ROOT_ENV] = str(pinned)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(BOOTSTRAP_SOURCE),
            "--reference-repo-root",
            str(other),
            "--sidecar-path",
            str(tmp_path / "sidecar.json"),
            "--expected-request-count",
            "1",
            "--",
            "--simulation_mode",
            "offline",
        ],
        cwd=pinned,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "must equal the pinned Reference repo root" in completed.stderr
    assert str(pinned.resolve()) in completed.stderr
    assert REFERENCE_REPO_ROOT_ENV in completed.stderr
