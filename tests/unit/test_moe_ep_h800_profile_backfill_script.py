from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "tests/performance/run_moe_ep_h800_profile_backfill.sh"


def test_profile_backfill_enriches_all_target_embedded_mtp_tp4_timings() -> None:
    from frontier.spec_decode.mtp_registry import (
        get_target_embedded_mtp_linear_ops,
        get_target_embedded_mtp_same_tp_linear_ops,
    )

    source = SCRIPT.read_text(encoding="utf-8")
    timing_stats = ("min", "max", "mean", "median", "std", "count")
    required_ops = (
        *get_target_embedded_mtp_same_tp_linear_ops(),
        *get_target_embedded_mtp_linear_ops(),
    )

    for op_name in required_ops:
        for statistic in timing_stats:
            assert f"time_stats.{op_name}.{statistic}" in source
    assert source.count('--enrich-columns "${MTP_ENRICH_COLUMNS[@]}"') == 1
    assert "Qwen non-TP4 same-TP timing changed" in source


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


def test_profile_backfill_step_tp1_scope_profiles_only_missing_rows(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "step-tp1-profile-stage"
    env = os.environ.copy()
    env.update(
        {
            "NUM_GPUS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage-root",
            str(stage_root),
            "--base-ref",
            "HEAD",
            "--scope",
            "step-tp1-standard",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    profile_commands = [
        line
        for line in result.stdout.splitlines()
        if "frontier.profiling.moe.main" in line
    ]
    assert len(profile_commands) == 2
    profile_output = "\n".join(profile_commands)
    assert profile_output.count("--models step-moe-noquant-small") == 2
    assert profile_output.count("--num_gpus 1") == 2
    assert profile_output.count("--num_tensor_parallel_workers 1") == 2
    assert profile_output.count("--expert_parallel_sizes 1") == 2
    assert profile_output.count("--gating_runtime_context standalone_legacy") == 2
    assert "--profile_method cuda_event" in profile_output
    assert "--profile_method record_function" in profile_output
    assert "Phi-tiny-MoE-instruct" not in result.stdout
    assert "qwen3-next-80b-a3b-instruct-reduced-l2" not in result.stdout
    assert "prefill_hot" not in result.stdout
    assert result.stdout.count("merge_profile_csv_contexts.py") == 2
    assert (
        "data/profiling/compute/h800/step-moe-noquant-small"
        in result.stdout
    )
    assert "step-tp1-standard" in result.stdout


def test_profile_backfill_audit_only_skips_collection_and_merge(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "existing-step-tp1-profile-stage"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage-root",
            str(stage_root),
            "--base-ref",
            "HEAD",
            "--scope",
            "step-tp1-standard",
            "--audit-only",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "frontier.profiling.moe.main" not in result.stdout
    assert "merge_profile_csv_contexts.py" not in result.stdout
    assert " archive --format=tar " not in result.stdout
    assert "step-tp1-standard" in result.stdout
    assert "profile_audit.json" in result.stdout
