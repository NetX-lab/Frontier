"""Contract tests for the MoE-EP non-dummy matrix shell wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "tests" / "e2e" / "run_moe_ep_non_dummy_matrix.sh"


def _run_wrapper(tmp_path: Path, extra_env: dict[str, str]) -> list[str]:
    """Run the wrapper with a stub interpreter that echoes its arguments."""

    stub = tmp_path / "stub-python"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    stub.chmod(0o755)

    environment = os.environ.copy()
    for key in ("MATRIX_OUTPUT_ROOT", "MATRIX_MODE", "FRONTIER_TMP_ROOT"):
        environment.pop(key, None)
    environment["PYTHON_BIN"] = str(stub)
    environment.update(extra_env)

    completed = subprocess.run(
        ["bash", str(WRAPPER), "--continue-on-failure"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return completed.stdout.splitlines()


def test_wrapper_passes_bash_syntax_check() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True, timeout=30)


def test_wrapper_defers_output_root_to_python_when_matrix_output_root_unset(
    tmp_path: Path,
) -> None:
    argv = _run_wrapper(tmp_path, {"FRONTIER_TMP_ROOT": str(tmp_path / "scratch")})

    assert argv[0] == str(REPO_ROOT / "tests" / "e2e" / "moe_ep_non_dummy_matrix.py")
    assert "--output-root" not in argv
    assert "/data/ycfeng" not in " ".join(argv)
    assert argv[argv.index("--repo-root") + 1] == str(REPO_ROOT)
    assert argv[argv.index("--mode") + 1] == "run"
    assert argv[-1] == "--continue-on-failure"


def test_wrapper_preserves_explicit_matrix_output_root(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-output"

    argv = _run_wrapper(
        tmp_path,
        {"MATRIX_OUTPUT_ROOT": str(explicit), "MATRIX_MODE": "preflight"},
    )

    assert argv[argv.index("--output-root") + 1] == str(explicit)
    assert argv[argv.index("--mode") + 1] == "preflight"
    assert argv[-1] == "--continue-on-failure"
