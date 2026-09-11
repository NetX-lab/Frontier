"""Contract tests for the read-only old-version replay harness."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.e2e import moe_ep_baseline_replay as baseline
from tests.e2e.moe_ep_non_dummy_matrix import build_matrix, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _scratch_root_under_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the replay scratch root at pytest's tmp_path for every test."""

    monkeypatch.setenv(baseline.SCRATCH_ROOT_ENV, str(tmp_path))


def test_baseline_replay_harness_exists_under_e2e_tests() -> None:
    assert (REPO_ROOT / "tests/e2e/moe_ep_baseline_replay.py").is_file()


@pytest.mark.parametrize(
    ("distribution", "expected_mode"),
    [
        ("random", "uniform_random"),
        ("balanced", "simulation"),
        ("skewed", "simulation"),
        ("zipf", "simulation"),
    ],
)
def test_routing_translation_uses_only_old_owned_fields(
    distribution: str,
    expected_mode: str,
) -> None:
    assert baseline.translate_routing_selector(distribution) == (
        expected_mode,
        distribution,
    )


def test_routing_translation_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError, match="unsupported routing distribution"):
        baseline.translate_routing_selector("legacy-uniform")


def _make_fake_wrapper(repo_root: Path, relative_path: str) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n", encoding="utf-8")


def test_baseline_command_uses_old_selector_and_baseline_only_pythonpath(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.routing_distribution == "random"
    )
    baseline_root = tmp_path / "baseline"
    _make_fake_wrapper(
        baseline_root,
        "examples/architecture/co-location/offline/moe_model_basic.sh",
    )
    output_root = tmp_path / "output"

    command, env = baseline.build_baseline_shell_command(
        case,
        baseline_root,
        output_root,
        python_executable=Path("/opt/frontier/bin/python"),
        cache_token="attempt-1",
    )

    assert env["PYTHONPATH"] == str(baseline_root.resolve())
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TMPDIR"] == str(tmp_path)
    assert env["TEMP"] == str(tmp_path)
    assert env["TMP"] == str(tmp_path)
    assert env["ENABLE_DUMMY_MODE"] == "false"
    assert env["MOE_ROUTING_MODE"] == "uniform_random"
    assert env["MOE_ROUTING_DISTRIBUTION_TYPE"] == "random"
    assert env["DP"] == "1"
    assert "--replica_config_moe_routing_distribution_type random" in command
    assert "--replica_config_device h800" in command
    assert str((output_root / case.case_id / "predictor_cache_attempt-1").resolve()) in command
    assert str(REPO_ROOT) not in env["PYTHONPATH"]


@pytest.mark.parametrize(
    ("architecture", "model_kind", "wrapper"),
    [
        (
            "pd-disaggregation",
            "moe",
            "examples/architecture/pdd/offline/moe_model_basic.sh",
        ),
        (
            "pd-af-disaggregation",
            "moe",
            "examples/architecture/pd-af-disagg/offline/moe_model_ep.sh",
        ),
    ],
)
def test_disaggregated_baseline_command_preserves_historical_attention_dp_fields(
    tmp_path: Path,
    architecture: str,
    model_kind: str,
    wrapper: str,
) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == architecture and case.model_kind == model_kind
    )
    baseline_root = tmp_path / "baseline"
    _make_fake_wrapper(baseline_root, wrapper)

    _, env = baseline.build_baseline_shell_command(
        case,
        baseline_root,
        tmp_path / "output",
        python_executable=Path("/opt/frontier/bin/python"),
        cache_token="attempt-1",
    )

    assert env["PREFILL_ATTN_DP"] == "1"
    assert env["DECODE_ATTN_DP"] == "1"


def test_data_only_cwd_exposes_only_current_data_tree(tmp_path: Path) -> None:
    current_data = tmp_path / "current-repo" / "data"
    current_data.mkdir(parents=True)
    data_cwd = tmp_path / "baseline-data-cwd"

    baseline.prepare_data_only_cwd(current_data, data_cwd)

    assert (data_cwd / "data").is_symlink()
    assert (data_cwd / "data").resolve() == current_data.resolve()
    baseline.validate_data_only_cwd(current_data, data_cwd)
    (data_cwd / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must contain only"):
        baseline.validate_data_only_cwd(current_data, data_cwd)


def _init_detached_repo(
    path: Path,
    *,
    files: dict[str, str] | None = None,
) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Frontier Tests"], cwd=path, check=True)
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    for relative_path, content in (files or {}).items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    subprocess.run(["git", "checkout", "--detach", "-q", head], cwd=path, check=True)
    return head


def test_baseline_worktree_validation_requires_exact_clean_detached_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "baseline-repo"
    head = _init_detached_repo(repo)

    assert baseline.validate_baseline_worktree(repo, expected_commit=head) == head

    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        baseline.validate_baseline_worktree(repo, expected_commit=head)


def test_baseline_log_checker_separates_execution_from_workflow_evidence(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    log_path = tmp_path / "baseline.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] batch_id=1, layer_id=0, predicted_time_ms=0.1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2",
            ]
        ),
        encoding="utf-8",
    )
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.25},
                "tpot_statistics": {"mean": 0.5},
                "request_e2e_time_statistics": {"mean": 2.75},
            }
        ),
        encoding="utf-8",
    )

    result = baseline.check_baseline_case(case, log_path, metrics_dir)

    assert result["execution_status"] == "PASS"
    assert result["workflow_evidence_status"] == "MISSING_CURRENT_SCHEMA"
    assert result["old_moe_op_trace_count"] == 2
    assert result["ep_workload_records"] == 0
    assert result["dispatch_barrier_records"] == 0
    assert result["combine_barrier_records"] == 0
    assert result["ep_conservation_records"] == 0
    assert result["ttft_mean_ms"] == 1.25
    assert result["tpot_mean_ms"] == 0.5
    assert result["e2e_mean_ms"] == 2.75


def test_baseline_log_checker_does_not_accept_stale_metrics_when_no_fresh_dir_exists(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "dense"
    )
    log_path = tmp_path / "baseline.log"
    log_path.write_text(
        "Dummy Mode: false\nSimulation completed successfully.\n",
        encoding="utf-8",
    )

    result = baseline.check_baseline_case(case, log_path, None)

    assert result["execution_status"] == "FAIL"
    assert "missing fresh metrics" in result["execution_errors"]
    assert result["ttft_mean_ms"] is None
    assert result["e2e_mean_ms"] is None


def test_manifest_validation_requires_exact_matrix_and_card_bound(tmp_path: Path) -> None:
    cases = build_matrix(REPO_ROOT)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, cases)

    loaded = baseline.load_and_validate_manifest(manifest_path)

    assert len(loaded) == 110
    invalid = list(loaded)
    invalid[0] = replace(invalid[0], total_cards=33)
    with pytest.raises(ValueError, match="exceeds 32 cards"):
        baseline.validate_manifest_cases(invalid)


def test_baseline_ledger_provenance_rejects_paths_from_another_campaign(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    results_path = tmp_path / "results.jsonl"
    baseline_root = tmp_path / "baseline"
    data_cwd = tmp_path / "data-cwd"
    case_id = "case-0"
    valid_row = {
        "case_id": case_id,
        "baseline_repo_root": str(baseline_root.resolve()),
        "baseline_commit": "abc123",
        "data_cwd": str(data_cwd.resolve()),
        "output_root": str(output_root.resolve()),
        "results_path": str(results_path.resolve()),
        "log_path": str((output_root / case_id / "case.log").resolve()),
        "metrics_path": str((output_root / case_id / "metrics").resolve()),
        "execution_status": "PASS",
    }

    baseline.validate_baseline_ledger_provenance(
        [valid_row],
        baseline_repo_root=baseline_root,
        baseline_commit="abc123",
        data_cwd=data_cwd,
        output_root=output_root,
        results_path=results_path,
    )

    invalid_row = dict(valid_row)
    invalid_row["log_path"] = str((tmp_path / "other" / "case.log").resolve())
    with pytest.raises(ValueError, match="outside its canonical case directory"):
        baseline.validate_baseline_ledger_provenance(
            [invalid_row],
            baseline_repo_root=baseline_root,
            baseline_commit="abc123",
            data_cwd=data_cwd,
            output_root=output_root,
            results_path=results_path,
        )


def test_baseline_runner_records_failure_and_continues_to_next_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_time_ns = baseline.time.time_ns
    monkeypatch.setattr(
        baseline.time,
        "time_ns",
        lambda: real_time_ns() + 1_000_000_000,
    )
    dense_cases = [
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "dense"
    ][:2]
    first_case_id = dense_cases[0].case_id
    wrapper = f"""#!/bin/bash
set -euo pipefail
echo "Dummy Mode: false"
if [ "$RUN_ID" = "{first_case_id}" ]; then
  echo "intentional old-runtime failure"
  exit 7
fi
mkdir -p "$METRICS_OUTPUT_DIR"
cat > "$METRICS_OUTPUT_DIR/system_metrics.json" <<'JSON'
{{"ttft_statistics": {{"mean": 1.0}}, "request_e2e_time_statistics": {{"mean": 2.0}}}}
JSON
echo "layer_id=0"
echo "Simulation completed successfully."
"""
    baseline_repo = tmp_path / "baseline-repo"
    head = _init_detached_repo(
        baseline_repo,
        files={
            "examples/architecture/co-location/offline/dense_model_basic.sh": wrapper
        },
    )
    profile_repo = tmp_path / "profile-repo"
    model_dir = (
        profile_repo
        / "data/profiling/compute"
        / dense_cases[0].device
        / dense_cases[0].model_name
    )
    model_dir.mkdir(parents=True)
    profile_csv = (
        "profiling_precision,model_arch,model_architecture_profile,"
        "quant_signature,measurement_type,num_tensor_parallel_workers\n"
        "BF16,generic,generic,none,CUDA_EVENT,1\n"
    )
    (model_dir / "attention.csv").write_text(profile_csv, encoding="utf-8")
    (model_dir / "linear_op.csv").write_text(profile_csv, encoding="utf-8")
    output_root = tmp_path / "output"
    results_path = tmp_path / "results.jsonl"
    data_cwd = tmp_path / "data-cwd"

    results = baseline.run_baseline_cases(
        dense_cases,
        baseline_repo,
        profile_repo,
        data_cwd,
        output_root,
        results_path,
        python_executable=Path(sys.executable),
        expected_commit=head,
        timeout_seconds=30,
    )

    assert [result["execution_status"] for result in results] == ["FAIL", "PASS"]
    assert [result["exit_code"] for result in results] == [7, 0]
    assert results[0]["metrics_path"] == ""
    assert results[1]["check"]["ttft_mean_ms"] == 1.0
    persisted = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["case_id"] for row in persisted] == [
        dense_cases[0].case_id,
        dense_cases[1].case_id,
    ]
    assert baseline.validate_baseline_worktree(
        baseline_repo, expected_commit=head
    ) == head


def test_baseline_timeout_signals_the_entire_process_group(tmp_path: Path) -> None:
    marker_path = tmp_path / "child-terminated.txt"
    log_path = tmp_path / "timeout.log"
    child = (
        "trap 'printf terminated > "
        + shlex.quote(str(marker_path))
        + "' TERM; while true; do sleep 1; done"
    )
    command = shlex.join(["bash", "-c", child])

    with log_path.open("w", encoding="utf-8") as stream:
        exit_code = baseline._run_baseline_process(
            command,
            cwd=tmp_path,
            env=dict(os.environ),
            stream=stream,
            timeout_seconds=1,
        )

    assert exit_code == 124
    assert marker_path.read_text(encoding="utf-8") == "terminated"


def test_baseline_cli_loads_the_exact_manifest_and_reports_runtime_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    cases = build_matrix(REPO_ROOT)
    write_manifest(manifest_path, cases)
    recorded: dict[str, object] = {}

    def fake_run(
        loaded_cases: list[object],
        baseline_repo_root: Path,
        profile_repo_root: Path,
        data_cwd: Path,
        output_root: Path,
        results_path: Path,
        **kwargs: object,
    ) -> list[dict[str, str]]:
        recorded.update(
            {
                "case_count": len(loaded_cases),
                "baseline_repo_root": baseline_repo_root,
                "profile_repo_root": profile_repo_root,
                "data_cwd": data_cwd,
                "output_root": output_root,
                "results_path": results_path,
                **kwargs,
            }
        )
        return [
            {"execution_status": "FAIL"},
            {"execution_status": "PASS"},
        ]

    monkeypatch.setattr(baseline, "run_baseline_cases", fake_run)
    baseline_repo = tmp_path / "baseline"
    profile_repo = tmp_path / "profile"
    data_cwd = tmp_path / "data-cwd"
    output_root = tmp_path / "output"
    results_path = tmp_path / "results.jsonl"

    exit_code = baseline.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--baseline-repo-root",
            str(baseline_repo),
            "--profile-repo-root",
            str(profile_repo),
            "--data-cwd",
            str(data_cwd),
            "--output-root",
            str(output_root),
            "--results-path",
            str(results_path),
            "--python-executable",
            sys.executable,
            "--start",
            "4",
            "--limit",
            "2",
            "--timeout-seconds",
            "30",
        ]
    )

    assert exit_code == 1
    assert recorded == {
        "case_count": 110,
        "baseline_repo_root": baseline_repo.resolve(),
        "profile_repo_root": profile_repo.resolve(),
        "data_cwd": data_cwd.resolve(),
        "output_root": output_root.resolve(),
        "results_path": results_path.resolve(),
        "python_executable": Path(sys.executable).resolve(),
        "expected_commit": baseline.EXPECTED_BASELINE_COMMIT,
        "start": 4,
        "limit": 2,
        "timeout_seconds": 30,
    }
