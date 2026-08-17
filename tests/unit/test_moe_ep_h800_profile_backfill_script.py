from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "tests/performance/run_moe_ep_h800_profile_backfill.sh"


def test_profile_backfill_dry_run_snapshots_explicit_git_base_ref(
    tmp_path: Path,
) -> None:
    base_commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    stage_root = tmp_path / "profile-stage"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage-root",
            str(stage_root),
            "--base-ref",
            "HEAD",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"base_ref=HEAD base_commit={base_commit}" in result.stdout
    assert (
        f"git -C {REPO_ROOT} archive --format=tar {base_commit} -- "
        "data/profiling/compute/h800/Phi-tiny-MoE-instruct "
        "data/profiling/compute/h800/step-moe-noquant-small "
        "data/profiling/compute/h800/"
        "qwen3-next-80b-a3b-instruct-reduced-l2"
    ) in result.stdout
    assert f"cp -a {REPO_ROOT}/data/profiling/compute/h800/" not in result.stdout
