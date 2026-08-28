"""Contract tests for current-versus-baseline MoE EP campaign comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.e2e.moe_ep_non_dummy_matrix import build_matrix, write_manifest
from tests.performance import moe_ep_baseline_comparison as comparison


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_metrics(
    root: Path,
    case_id: str,
    *,
    ttft_ms: float,
    tpot_ms: float,
    e2e_ms: float,
) -> Path:
    metrics_dir = root / case_id / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": ttft_ms, "unit": "ms"},
                "tpot_statistics": {"mean": tpot_ms, "unit": "ms"},
                "request_e2e_time_statistics": {
                    "mean": e2e_ms,
                    "unit": "ms",
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics_dir


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_case_metadata(
    root: Path,
    *,
    case: object,
    filename: str,
) -> Path:
    case_id = str(getattr(case, "case_id"))
    case_root = root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    log_path = case_root / f"{case_id}.log"
    log_path.write_text("", encoding="utf-8")
    (case_root / filename).write_text(
        json.dumps({"case": asdict(case)}, sort_keys=True),
        encoding="utf-8",
    )
    return log_path


def _build_complete_campaigns(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    cases = build_matrix(REPO_ROOT)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, cases)
    current_metrics_root = tmp_path / "current"
    baseline_metrics_root = tmp_path / "baseline"
    current_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        current_log = _write_case_metadata(
            current_metrics_root,
            case=case,
            filename="case_metadata.json",
        )
        current_metrics = _write_metrics(
            current_metrics_root,
            case.case_id,
            ttft_ms=12.0,
            tpot_ms=2.0,
            e2e_ms=30.0,
        )
        current_rows.append(
            {
                "case_id": case.case_id,
                "architecture": case.architecture,
                "model_kind": case.model_kind,
                "total_cards": case.total_cards,
                "status": "PASS",
                "log_path": str(current_log),
                "metrics_path": str(current_metrics),
                "check": {
                    "ep_workload_records": 4 if case.is_moe else 0,
                    "ep_barrier_records": 8 if case.is_moe else 0,
                    "ep_conservation_records": 4 if case.is_moe else 0,
                },
            }
        )
        baseline_log = _write_case_metadata(
            baseline_metrics_root,
            case=case,
            filename="baseline_case_metadata.json",
        )
        if index == len(cases) - 1:
            baseline_rows.append(
                {
                    "case_id": case.case_id,
                    "architecture": case.architecture,
                    "model_kind": case.model_kind,
                    "total_cards": case.total_cards,
                    "execution_status": "FAIL",
                    "workflow_evidence_status": "MISSING_CURRENT_SCHEMA",
                    "log_path": str(baseline_log),
                    "metrics_path": "",
                    "exit_code": 7,
                    "check": {
                        "execution_errors": "intentional old-runtime failure",
                        "ep_workload_records": 0,
                        "dispatch_barrier_records": 0,
                        "combine_barrier_records": 0,
                        "ep_conservation_records": 0,
                    },
                }
            )
        else:
            baseline_metrics = _write_metrics(
                baseline_metrics_root,
                case.case_id,
                ttft_ms=10.0,
                tpot_ms=1.0,
                e2e_ms=25.0,
            )
            baseline_rows.append(
                {
                    "case_id": case.case_id,
                    "architecture": case.architecture,
                    "model_kind": case.model_kind,
                    "total_cards": case.total_cards,
                    "execution_status": "PASS",
                    "workflow_evidence_status": (
                        "MISSING_CURRENT_SCHEMA"
                        if case.is_moe
                        else "NOT_APPLICABLE_DENSE"
                    ),
                    "log_path": str(baseline_log),
                    "metrics_path": str(baseline_metrics),
                    "exit_code": 0,
                    "check": {
                        "execution_errors": "",
                        "ep_workload_records": 0,
                        "dispatch_barrier_records": 0,
                        "combine_barrier_records": 0,
                        "ep_conservation_records": 0,
                    },
                }
            )
    current_path = tmp_path / "current.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(current_path, current_rows)
    _write_jsonl(baseline_path, baseline_rows)
    return (
        manifest_path,
        current_path,
        baseline_path,
        current_rows,
        baseline_rows,
    )


def test_comparison_aligns_all_cases_and_reports_numeric_and_workflow_gaps(
    tmp_path: Path,
) -> None:
    manifest, current, baseline, _, _ = _build_complete_campaigns(tmp_path)

    result = comparison.build_comparison(manifest, current, baseline)

    assert result["report_date"] == "2026-08-17"
    assert result["case_count"] == 110
    assert result["current_execution_counts"] == {"PASS": 110}
    assert result["baseline_execution_counts"] == {"FAIL": 1, "PASS": 109}
    assert result["paired_execution_pass_count"] == 109
    assert result["baseline_workflow_counts"]["MISSING_CURRENT_SCHEMA"] > 0
    ttft = result["metric_summaries"]["ttft_mean_ms"]
    assert ttft["paired_count"] == 109
    assert ttft["baseline_mean_ms"] == 10.0
    assert ttft["current_mean_ms"] == 12.0
    assert ttft["paired_median_relative_gap_percent"] == pytest.approx(20.0)
    first_case = result["cases"][0]
    assert first_case["metrics"]["ttft_mean_ms"] == {
        "baseline_ms": 10.0,
        "current_ms": 12.0,
        "delta_ms": 2.0,
        "absolute_gap_ms": 2.0,
        "relative_gap_percent": 20.0,
    }
    assert first_case["current_ep_barrier_records"] == 0


def test_comparison_writes_json_csv_and_markdown_artifacts(tmp_path: Path) -> None:
    manifest, current, baseline, _, _ = _build_complete_campaigns(tmp_path)
    result = comparison.build_comparison(manifest, current, baseline)
    json_path = tmp_path / "comparison.json"
    csv_path = tmp_path / "comparison.csv"
    markdown_path = tmp_path / "comparison.md"

    comparison.write_comparison_artifacts(
        result,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["case_count"] == 110
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 111
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Execution Summary" in markdown
    assert "## Metric Gaps" in markdown
    assert "intentional old-runtime failure" in markdown


def test_comparison_rejects_duplicate_or_incomplete_campaign_rows(
    tmp_path: Path,
) -> None:
    manifest, current, baseline, _, baseline_rows = _build_complete_campaigns(tmp_path)
    _write_jsonl(baseline, baseline_rows[:-1] + [baseline_rows[0]])

    with pytest.raises(ValueError, match="duplicate case_id"):
        comparison.build_comparison(manifest, current, baseline)


def test_comparison_rejects_non_millisecond_metric_units(
    tmp_path: Path,
) -> None:
    manifest, current, baseline, _, baseline_rows = _build_complete_campaigns(
        tmp_path
    )
    metric_file = (
        Path(str(baseline_rows[0]["metrics_path"])) / "system_metrics.json"
    )
    metrics = json.loads(metric_file.read_text(encoding="utf-8"))
    metrics["ttft_statistics"]["unit"] = "s"
    metric_file.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="unit='s'"):
        comparison.build_comparison(manifest, current, baseline)


@pytest.mark.parametrize(
    ("campaign", "metadata_filename", "field", "invalid_value"),
    (
        ("current", "case_metadata.json", "model_name", "wrong-model"),
        (
            "baseline",
            "baseline_case_metadata.json",
            "decode_ffn_replicas",
            999,
        ),
    ),
)
def test_comparison_rejects_case_metadata_that_differs_from_manifest(
    tmp_path: Path,
    campaign: str,
    metadata_filename: str,
    field: str,
    invalid_value: object,
) -> None:
    manifest, current, baseline, current_rows, baseline_rows = (
        _build_complete_campaigns(tmp_path)
    )
    rows = current_rows if campaign == "current" else baseline_rows
    metadata_path = (
        Path(str(rows[0]["log_path"])).parent / metadata_filename
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["case"][field] = invalid_value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"{campaign} case metadata mismatch.*{field}",
    ):
        comparison.build_comparison(manifest, current, baseline)


def test_comparison_rejects_result_row_metadata_that_differs_from_manifest(
    tmp_path: Path,
) -> None:
    manifest, current, baseline, current_rows, _ = _build_complete_campaigns(
        tmp_path
    )
    current_rows[0]["total_cards"] = 999
    _write_jsonl(current, current_rows)

    with pytest.raises(
        ValueError,
        match="current result metadata mismatch.*total_cards",
    ):
        comparison.build_comparison(manifest, current, baseline)
