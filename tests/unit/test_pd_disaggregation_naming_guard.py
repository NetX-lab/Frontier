"""Guard against reintroducing legacy prefill/decode split naming."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("frontier", "examples", "tests")
FORBIDDEN_TOKENS_LOWERCASE = (
    "PD" + "-only",
    "pd" + "-only",
    "pd" + "_only",
    "pd" + "only",
    "PD" + " only",
    "pd" + " only",
)


def _scanned_paths() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SCANNED_ROOTS,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def test_frontier_examples_tests_use_pd_disaggregation_naming() -> None:
    violations: list[str] = []

    for path in _scanned_paths():
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        normalized_relative_path = relative_path.lower()
        if any(token.lower() in normalized_relative_path for token in FORBIDDEN_TOKENS_LOWERCASE):
            violations.append(f"{relative_path}: path contains legacy naming")

        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            normalized_line = line.lower()
            for token in FORBIDDEN_TOKENS_LOWERCASE:
                if token.lower() in normalized_line:
                    violations.append(
                        f"{relative_path}:{line_number}: contains case-insensitive legacy naming"
                    )
                    break

    assert violations == []
