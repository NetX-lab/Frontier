from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.e2e.attention_equivalence.run_equivalence_matrix import run_case
from tests.e2e.attention_equivalence.run_equivalence_matrix import run_matrix
from tests.e2e.attention_equivalence.simulation_output_equivalence import (
    compare_simulation_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_request_metrics(root: Path, rows: list[dict[str, object]]) -> None:
    _write_csv(
        root / "request_metrics.csv",
        rows,
        [
            "Request Id",
            "request_e2e_time",
            "ttft",
            "tpot",
            "request_waiting_time_total",
            "request_num_prefill_tokens",
            "request_num_decode_tokens",
            "transfer_kv_cache",
            "request_spec_accepted_drafts",
            "request_spec_rejected_drafts",
            "request_spec_committed_tokens",
        ],
    )


def test_simulation_output_equivalence_compares_request_metrics_by_request_id(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference_rows = [
        {
            "Request Id": "0",
            "request_e2e_time": "10.0",
            "ttft": "4.0",
            "tpot": "3.0",
            "request_waiting_time_total": "0.0",
            "request_num_prefill_tokens": "8",
            "request_num_decode_tokens": "2",
            "transfer_kv_cache": "0.0",
            "request_spec_accepted_drafts": "0",
            "request_spec_rejected_drafts": "0",
            "request_spec_committed_tokens": "0",
        },
        {
            "Request Id": "1",
            "request_e2e_time": "20.0",
            "ttft": "8.0",
            "tpot": "6.0",
            "request_waiting_time_total": "1.0",
            "request_num_prefill_tokens": "16",
            "request_num_decode_tokens": "2",
            "transfer_kv_cache": "2.0",
            "request_spec_accepted_drafts": "1",
            "request_spec_rejected_drafts": "0",
            "request_spec_committed_tokens": "1",
        },
    ]
    _write_request_metrics(reference, reference_rows)
    _write_request_metrics(candidate, list(reversed(reference_rows)))

    report = compare_simulation_outputs(
        reference,
        candidate,
        case_manifest={"workload_sha256": "same-input", "seed": 1234},
    )

    assert report["status"] == "PASS"
    assert report["mismatch_count"] == 0
    assert report["artifacts"]["request_metrics.csv"]["row_count"] == {
        "reference": 2,
        "candidate": 2,
        "delta": 0,
    }
    assert report["summary"]["request_metrics.csv"]["total_completed_requests"] == {
        "reference": 2,
        "candidate": 2,
        "delta": 0,
    }
    assert report["summary"]["request_metrics.csv"]["aggregates"]["request_e2e_time"]["reference_sum"] == pytest.approx(30.0)


def test_simulation_output_equivalence_reports_numeric_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_request_metrics(
        reference,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "10.0",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )
    _write_request_metrics(
        candidate,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "10.5",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )

    report = compare_simulation_outputs(reference, candidate)

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 1
    mismatch = report["artifacts"]["request_metrics.csv"]["numeric_comparisons"][0]
    assert mismatch["column"] == "request_e2e_time"
    assert mismatch["reference_value"] == 10.0
    assert mismatch["candidate_value"] == 10.5
    assert mismatch["absolute_delta"] == pytest.approx(0.5)
    assert mismatch["relative_delta_pct"] == pytest.approx(5.0)


def test_simulation_output_equivalence_fails_fast_on_schema_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_csv(reference / "request_metrics.csv", [{"Request Id": "0", "ttft": "1.0"}], ["Request Id", "ttft"])
    _write_csv(candidate / "request_metrics.csv", [{"Request Id": "0", "tpot": "1.0"}], ["Request Id", "tpot"])

    with pytest.raises(ValueError, match="schema mismatch"):
        compare_simulation_outputs(reference, candidate)


def test_matrix_runner_classifies_missing_reference_as_reference_baseline_failure(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    _write_request_metrics(
        candidate,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "10.0",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )

    result = run_case(
        {
            "name": "missing-reference",
            "reference_dir": str(tmp_path / "missing_reference"),
            "candidate_dir": str(candidate),
            "reference_input_manifest": {"workload_sha256": "same", "seed": 1},
            "candidate_input_manifest": {"workload_sha256": "same", "seed": 1},
        }
    )

    assert result["status"] == "REFERENCE_BASELINE_FAILS"
    assert result["exit_code"] != 0
    assert "request_metrics.csv" in result["error"]


def test_matrix_runner_classifies_missing_candidate_as_candidate_artifact_failure(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "run"
    _write_request_metrics(
        reference,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "10.0",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )

    result = run_case(
        {
            "name": "missing-candidate",
            "reference_dir": str(reference),
            "candidate_dir": str(tmp_path / "run_candidate"),
            "reference_input_manifest": {"workload_sha256": "same", "seed": 1},
            "candidate_input_manifest": {"workload_sha256": "same", "seed": 1},
        }
    )

    assert result["status"] == "CANDIDATE_ARTIFACT_FAILS"
    assert result["exit_code"] != 0
    assert "request_metrics.csv" in result["error"]


def test_matrix_runner_rejects_manifest_mismatch_before_comparison(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    rows = [
        {
            "Request Id": "0",
            "request_e2e_time": "10.0",
            "ttft": "4.0",
            "tpot": "3.0",
            "request_waiting_time_total": "0.0",
            "request_num_prefill_tokens": "8",
            "request_num_decode_tokens": "2",
            "transfer_kv_cache": "0.0",
            "request_spec_accepted_drafts": "0",
            "request_spec_rejected_drafts": "0",
            "request_spec_committed_tokens": "0",
        }
    ]
    _write_request_metrics(reference, rows)
    _write_request_metrics(candidate, rows)

    result = run_case(
        {
            "name": "manifest-mismatch",
            "reference_dir": str(reference),
            "candidate_dir": str(candidate),
            "reference_input_manifest": {"workload_sha256": "reference", "seed": 1},
            "candidate_input_manifest": {"workload_sha256": "candidate", "seed": 1},
        }
    )

    assert result["status"] == "STRUCTURAL_FAIL"
    assert result["exit_code"] != 0
    assert "manifest mismatch" in result["error"]


def test_matrix_runner_rejects_empty_matrix() -> None:
    report = run_matrix([])

    assert report["status"] == "STRUCTURAL_FAIL"
    assert report["exit_code"] != 0
    assert report["case_count"] == 0
    assert report["pass_count"] == 0
    assert "at least one case" in report["error"]


def test_matrix_runner_uses_highest_severity_status(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_request_metrics(
        reference,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "10.0",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )
    _write_request_metrics(
        candidate,
        [
            {
                "Request Id": "0",
                "request_e2e_time": "11.0",
                "ttft": "4.0",
                "tpot": "3.0",
                "request_waiting_time_total": "0.0",
                "request_num_prefill_tokens": "8",
                "request_num_decode_tokens": "2",
                "transfer_kv_cache": "0.0",
                "request_spec_accepted_drafts": "0",
                "request_spec_rejected_drafts": "0",
                "request_spec_committed_tokens": "0",
            }
        ],
    )

    report = run_matrix(
        [
            {
                "name": "candidate-mismatch",
                "reference_dir": str(reference),
                "candidate_dir": str(candidate),
                "reference_input_manifest": {"workload_sha256": "same", "seed": 1},
                "candidate_input_manifest": {"workload_sha256": "same", "seed": 1},
            },
            {
                "name": "structural-fail",
                "reference_dir": str(reference),
                "candidate_dir": str(candidate),
                "reference_input_manifest": {"workload_sha256": "reference", "seed": 1},
                "candidate_input_manifest": {"workload_sha256": "candidate", "seed": 1},
            },
        ]
    )

    assert report["status"] == "STRUCTURAL_FAIL"
    assert report["exit_code"] != 0
    assert report["case_count"] == 2
    assert [result["status"] for result in report["results"]] == [
        "CANDIDATE_MISMATCH",
        "STRUCTURAL_FAIL",
    ]
