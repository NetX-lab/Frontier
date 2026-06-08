import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH_ENV = "FRONTIER_PROFILING_DATASET_TOOLS"
SCRIPT_PATH = (
    Path(os.environ[SCRIPT_PATH_ENV])
    if SCRIPT_PATH_ENV in os.environ
    else None
)
MODEL_CONFIG_PATH = REPO_ROOT / "data/config/models/Llama-3.2-1B-Instruct.json"


def _run_tool(args: list[str]) -> subprocess.CompletedProcess[str]:
    if SCRIPT_PATH is None:
        pytest.skip(f"Set {SCRIPT_PATH_ENV} to run external profiling dataset tool tests")
    if not SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"{SCRIPT_PATH_ENV} does not point to a file: {SCRIPT_PATH}")
    command = [sys.executable, str(SCRIPT_PATH), *args]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_coverage_diff_reads_llama32_model_config(tmp_path: Path) -> None:
    existing_csv = tmp_path / "existing_linear_op.csv"
    required_csv = tmp_path / "required_linear_op.csv"
    missing_csv = tmp_path / "missing.csv"
    summary_json = tmp_path / "summary.json"

    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1, "time_stats.add.mean": 0.1},
            {"num_tensor_parallel_workers": 1, "num_tokens": 2, "time_stats.add.mean": 0.2},
        ]
    ).to_csv(existing_csv, index=False)
    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1},
            {"num_tensor_parallel_workers": 1, "num_tokens": 2},
            {"num_tensor_parallel_workers": 2, "num_tokens": 1},
        ]
    ).to_csv(required_csv, index=False)

    result = _run_tool(
        [
            "coverage-diff",
            "--module",
            "linear_op",
            "--existing-csv",
            str(existing_csv),
            "--required-csv",
            str(required_csv),
            "--model-config",
            str(MODEL_CONFIG_PATH),
            "--missing-output-csv",
            str(missing_csv),
            "--summary-output-json",
            str(summary_json),
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert summary["missing_tuple_count"] == 1

    missing_df = pd.read_csv(missing_csv)
    assert len(missing_df) == 1
    assert tuple(missing_df.iloc[0][["num_tensor_parallel_workers", "num_tokens"]]) == (2, 1)


def test_diff_merge_merges_patch_with_keep_last(tmp_path: Path) -> None:
    base_csv = tmp_path / "base_linear_op.csv"
    required_csv = tmp_path / "required_linear_op.csv"
    supplement_csv = tmp_path / "supplement_linear_op.csv"
    output_csv = tmp_path / "merged_linear_op.csv"
    summary_json = tmp_path / "summary.json"

    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1, "time_stats.add.mean": 0.1},
        ]
    ).to_csv(base_csv, index=False)
    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1},
            {"num_tensor_parallel_workers": 1, "num_tokens": 2},
        ]
    ).to_csv(required_csv, index=False)
    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1, "time_stats.add.mean": 9.9},
            {"num_tensor_parallel_workers": 1, "num_tokens": 2, "time_stats.add.mean": 2.2},
        ]
    ).to_csv(supplement_csv, index=False)

    result = _run_tool(
        [
            "diff-merge",
            "--module",
            "linear_op",
            "--base-csv",
            str(base_csv),
            "--required-csv",
            str(required_csv),
            "--supplement-csv",
            str(supplement_csv),
            "--output-csv",
            str(output_csv),
            "--model-config",
            str(MODEL_CONFIG_PATH),
            "--strict-supplement-coverage",
            "--summary-output-json",
            str(summary_json),
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr

    merged_df = pd.read_csv(output_csv).sort_values(["num_tensor_parallel_workers", "num_tokens"])
    assert len(merged_df) == 2
    row_token_1 = merged_df[merged_df["num_tokens"] == 1].iloc[0]
    assert row_token_1["time_stats.add.mean"] == pytest.approx(9.9)

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["missing_tuple_count_before_merge"] == 1
    assert summary["uncovered_missing_tuple_count"] == 0


def test_diff_merge_fails_when_supplement_cannot_cover_missing(tmp_path: Path) -> None:
    base_csv = tmp_path / "base_linear_op.csv"
    required_csv = tmp_path / "required_linear_op.csv"
    supplement_csv = tmp_path / "supplement_linear_op.csv"
    output_csv = tmp_path / "merged_linear_op.csv"

    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1, "time_stats.add.mean": 0.1},
        ]
    ).to_csv(base_csv, index=False)
    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 1},
            {"num_tensor_parallel_workers": 1, "num_tokens": 2},
            {"num_tensor_parallel_workers": 1, "num_tokens": 3},
        ]
    ).to_csv(required_csv, index=False)
    pd.DataFrame(
        [
            {"num_tensor_parallel_workers": 1, "num_tokens": 2, "time_stats.add.mean": 2.2},
        ]
    ).to_csv(supplement_csv, index=False)

    result = _run_tool(
        [
            "diff-merge",
            "--module",
            "linear_op",
            "--base-csv",
            str(base_csv),
            "--required-csv",
            str(required_csv),
            "--supplement-csv",
            str(supplement_csv),
            "--output-csv",
            str(output_csv),
            "--model-config",
            str(MODEL_CONFIG_PATH),
            "--strict-supplement-coverage",
        ]
    )

    assert result.returncode != 0
    assert "cannot cover all missing tuples" in result.stdout
