#!/usr/bin/env python3
"""Compare the current non-dummy MoE EP campaign with the old-version replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.e2e.moe_ep_baseline_replay import load_and_validate_manifest


REPORT_DATE = "2026-08-17"
METRIC_FIELDS = {
    "ttft_mean_ms": ("ttft_statistics", True),
    "tpot_mean_ms": ("tpot_statistics", False),
    "e2e_mean_ms": ("request_e2e_time_statistics", True),
}


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid {label} JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} row is not an object at {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _index_complete_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} row has no non-empty case_id")
        if case_id in indexed:
            raise ValueError(f"{label} rows contain duplicate case_id={case_id!r}")
        indexed[case_id] = row
    expected = set(expected_case_ids)
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} rows do not match the manifest: missing={missing}, extra={extra}"
        )
    return indexed


def _metric_means(
    row: Mapping[str, Any],
    *,
    status: str,
    label: str,
) -> dict[str, float | None]:
    if status != "PASS":
        return {field: None for field in METRIC_FIELDS}
    metrics_path = row.get("metrics_path")
    if not isinstance(metrics_path, str) or not metrics_path:
        raise ValueError(f"{label} PASS row has no metrics_path")
    metric_file = Path(metrics_path) / "system_metrics.json"
    if not metric_file.is_file():
        raise FileNotFoundError(metric_file)
    try:
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON: {metric_file}") from exc
    if not isinstance(metrics, Mapping):
        raise ValueError(f"metrics root must be an object: {metric_file}")

    values: dict[str, float | None] = {}
    for output_field, (metric_name, required) in METRIC_FIELDS.items():
        stats = metrics.get(metric_name)
        candidate = stats.get("mean") if isinstance(stats, Mapping) else None
        if candidate is None and not required:
            values[output_field] = None
            continue
        unit = stats.get("unit") if isinstance(stats, Mapping) else None
        if unit != "ms":
            raise ValueError(
                f"{label} has invalid {metric_name}.unit={unit!r} "
                f"in {metric_file}"
            )
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
        ):
            raise ValueError(
                f"{label} has no finite {metric_name}.mean in {metric_file}"
            )
        value = float(candidate)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{label} has invalid {metric_name}.mean={candidate!r}"
            )
        values[output_field] = value
    return values


def _metric_gap(
    baseline_ms: float | None,
    current_ms: float | None,
) -> dict[str, float | None]:
    if baseline_ms is None or current_ms is None:
        return {
            "baseline_ms": baseline_ms,
            "current_ms": current_ms,
            "delta_ms": None,
            "absolute_gap_ms": None,
            "relative_gap_percent": None,
        }
    delta = current_ms - baseline_ms
    relative = None if baseline_ms == 0 else delta / baseline_ms * 100.0
    return {
        "baseline_ms": baseline_ms,
        "current_ms": current_ms,
        "delta_ms": delta,
        "absolute_gap_ms": abs(delta),
        "relative_gap_percent": relative,
    }


def _metric_summaries(
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    summaries: dict[str, dict[str, float | int | None]] = {}
    for metric_field in METRIC_FIELDS:
        paired = [
            row["metrics"][metric_field]
            for row in case_rows
            if row["metrics"][metric_field]["baseline_ms"] is not None
            and row["metrics"][metric_field]["current_ms"] is not None
        ]
        baseline_values = [float(item["baseline_ms"]) for item in paired]
        current_values = [float(item["current_ms"]) for item in paired]
        deltas = [float(item["delta_ms"]) for item in paired]
        absolute_gaps = [float(item["absolute_gap_ms"]) for item in paired]
        relative_gaps = [
            float(item["relative_gap_percent"])
            for item in paired
            if item["relative_gap_percent"] is not None
        ]
        summaries[metric_field] = {
            "paired_count": len(paired),
            "baseline_mean_ms": (
                statistics.fmean(baseline_values) if baseline_values else None
            ),
            "current_mean_ms": (
                statistics.fmean(current_values) if current_values else None
            ),
            "paired_mean_delta_ms": (
                statistics.fmean(deltas) if deltas else None
            ),
            "paired_median_delta_ms": (
                statistics.median(deltas) if deltas else None
            ),
            "paired_mean_absolute_gap_ms": (
                statistics.fmean(absolute_gaps) if absolute_gaps else None
            ),
            "paired_median_absolute_gap_ms": (
                statistics.median(absolute_gaps) if absolute_gaps else None
            ),
            "paired_mean_relative_gap_percent": (
                statistics.fmean(relative_gaps) if relative_gaps else None
            ),
            "paired_median_relative_gap_percent": (
                statistics.median(relative_gaps) if relative_gaps else None
            ),
        }
    return summaries


def _count_status(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _group_summaries(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in case_rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        group: {
            "case_count": len(rows),
            "current_execution_counts": _count_status(rows, "current_status"),
            "baseline_execution_counts": _count_status(rows, "baseline_status"),
            "paired_execution_pass_count": sum(
                row["current_status"] == "PASS"
                and row["baseline_status"] == "PASS"
                for row in rows
            ),
            "baseline_workflow_counts": _count_status(
                rows,
                "baseline_workflow_status",
            ),
            "metric_summaries": _metric_summaries(rows),
        }
        for group, rows in sorted(groups.items())
    }


def build_comparison(
    manifest_path: Path,
    current_results_path: Path,
    baseline_results_path: Path,
) -> dict[str, Any]:
    """Build a manifest-aligned comparison with no missing or duplicate rows."""

    cases = load_and_validate_manifest(manifest_path)
    expected_case_ids = [case.case_id for case in cases]
    current_rows = _load_jsonl(current_results_path, label="current result")
    baseline_rows = _load_jsonl(baseline_results_path, label="baseline result")
    current_by_id = _index_complete_rows(
        current_rows,
        expected_case_ids=expected_case_ids,
        label="current result",
    )
    baseline_by_id = _index_complete_rows(
        baseline_rows,
        expected_case_ids=expected_case_ids,
        label="baseline result",
    )

    case_comparisons: list[dict[str, Any]] = []
    for case in cases:
        current = current_by_id[case.case_id]
        baseline = baseline_by_id[case.case_id]
        if current.get("architecture") != case.architecture:
            raise ValueError(
                f"current architecture mismatch for {case.case_id}"
            )
        if baseline.get("architecture") != case.architecture:
            raise ValueError(
                f"baseline architecture mismatch for {case.case_id}"
            )
        current_status = current.get("status")
        baseline_status = baseline.get("execution_status")
        if current_status not in {"PASS", "FAIL"}:
            raise ValueError(
                f"invalid current status for {case.case_id}: {current_status!r}"
            )
        if baseline_status not in {"PASS", "FAIL"}:
            raise ValueError(
                f"invalid baseline status for {case.case_id}: {baseline_status!r}"
            )
        baseline_workflow = baseline.get("workflow_evidence_status")
        if not isinstance(baseline_workflow, str) or not baseline_workflow:
            raise ValueError(
                f"missing baseline workflow status for {case.case_id}"
            )

        current_metrics = _metric_means(
            current,
            status=str(current_status),
            label=f"current case {case.case_id}",
        )
        baseline_metrics = _metric_means(
            baseline,
            status=str(baseline_status),
            label=f"baseline case {case.case_id}",
        )
        current_check = current.get("check", {})
        baseline_check = baseline.get("check", {})
        if not isinstance(current_check, Mapping) or not isinstance(
            baseline_check,
            Mapping,
        ):
            raise ValueError(f"missing check object for {case.case_id}")

        case_comparisons.append(
            {
                "case_id": case.case_id,
                "architecture": case.architecture,
                "model_kind": case.model_kind,
                "model_name": case.model_name,
                "routing_distribution": case.routing_distribution,
                "ep_size": case.ep_size,
                "total_cards": case.total_cards,
                "current_status": current_status,
                "baseline_status": baseline_status,
                "baseline_exit_code": baseline.get("exit_code"),
                "baseline_workflow_status": baseline_workflow,
                "baseline_execution_errors": str(
                    baseline_check.get("execution_errors", "")
                ),
                "current_ep_workload_records": int(
                    current_check.get("ep_workload_records", 0)
                ),
                "current_ep_barrier_records": int(
                    current_check.get("ep_barrier_records", 0)
                ),
                "current_ep_conservation_records": int(
                    current_check.get("ep_conservation_records", 0)
                ),
                "baseline_ep_workload_records": int(
                    baseline_check.get("ep_workload_records", 0)
                ),
                "baseline_ep_barrier_records": int(
                    baseline_check.get("dispatch_barrier_records", 0)
                )
                + int(baseline_check.get("combine_barrier_records", 0)),
                "baseline_ep_conservation_records": int(
                    baseline_check.get("ep_conservation_records", 0)
                ),
                "baseline_old_op_trace_count": int(
                    baseline_check.get("old_op_trace_count", 0)
                ),
                "baseline_old_moe_op_trace_count": int(
                    baseline_check.get("old_moe_op_trace_count", 0)
                ),
                "metrics": {
                    metric_field: _metric_gap(
                        baseline_metrics[metric_field],
                        current_metrics[metric_field],
                    )
                    for metric_field in METRIC_FIELDS
                },
            }
        )

    current_execution_counts = _count_status(
        case_comparisons,
        "current_status",
    )
    baseline_execution_counts = _count_status(
        case_comparisons,
        "baseline_status",
    )
    baseline_workflow_counts = _count_status(
        case_comparisons,
        "baseline_workflow_status",
    )
    return {
        "schema_version": 1,
        "report_date": REPORT_DATE,
        "manifest_path": str(manifest_path.resolve()),
        "current_results_path": str(current_results_path.resolve()),
        "baseline_results_path": str(baseline_results_path.resolve()),
        "case_count": len(case_comparisons),
        "current_execution_counts": current_execution_counts,
        "baseline_execution_counts": baseline_execution_counts,
        "paired_execution_pass_count": sum(
            row["current_status"] == "PASS"
            and row["baseline_status"] == "PASS"
            for row in case_comparisons
        ),
        "baseline_workflow_counts": baseline_workflow_counts,
        "current_workflow_record_totals": {
            "ep_workload_records": sum(
                row["current_ep_workload_records"] for row in case_comparisons
            ),
            "ep_barrier_records": sum(
                row["current_ep_barrier_records"] for row in case_comparisons
            ),
            "ep_conservation_records": sum(
                row["current_ep_conservation_records"]
                for row in case_comparisons
            ),
        },
        "baseline_workflow_record_totals": {
            "ep_workload_records": sum(
                row["baseline_ep_workload_records"] for row in case_comparisons
            ),
            "ep_barrier_records": sum(
                row["baseline_ep_barrier_records"] for row in case_comparisons
            ),
            "ep_conservation_records": sum(
                row["baseline_ep_conservation_records"]
                for row in case_comparisons
            ),
            "old_op_trace_records": sum(
                row["baseline_old_op_trace_count"] for row in case_comparisons
            ),
            "old_moe_op_trace_records": sum(
                row["baseline_old_moe_op_trace_count"]
                for row in case_comparisons
            ),
        },
        "metric_summaries": _metric_summaries(case_comparisons),
        "architecture_summaries": _group_summaries(
            case_comparisons,
            field="architecture",
        ),
        "model_kind_summaries": _group_summaries(
            case_comparisons,
            field="model_kind",
        ),
        "routing_distribution_summaries": _group_summaries(
            case_comparisons,
            field="routing_distribution",
        ),
        "cases": case_comparisons,
    }


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "## Modification History",
        "",
        "| Date | Summary of Changes |",
        "|------|--------------------|",
        f"| {REPORT_DATE} | Generated the current-versus-old non-dummy MoE EP comparison. |",
        "",
        "# MoE EP Old-Version Comparison",
        "",
        "## Execution Summary",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Cases | {result['case_count']} |",
        f"| Current execution | {json.dumps(result['current_execution_counts'], sort_keys=True)} |",
        f"| Baseline execution | {json.dumps(result['baseline_execution_counts'], sort_keys=True)} |",
        f"| Paired PASS | {result['paired_execution_pass_count']} |",
        f"| Baseline workflow | {json.dumps(result['baseline_workflow_counts'], sort_keys=True)} |",
        "",
        "## Workflow Evidence",
        "",
        "| Campaign | EP workload | EP barrier | EP conservation | Old OP-TRACE |",
        "|----------|-------------|------------|-----------------|--------------|",
        (
            "| Current | "
            f"{result['current_workflow_record_totals']['ep_workload_records']} | "
            f"{result['current_workflow_record_totals']['ep_barrier_records']} | "
            f"{result['current_workflow_record_totals']['ep_conservation_records']} | "
            "N/A |"
        ),
        (
            "| Baseline | "
            f"{result['baseline_workflow_record_totals']['ep_workload_records']} | "
            f"{result['baseline_workflow_record_totals']['ep_barrier_records']} | "
            f"{result['baseline_workflow_record_totals']['ep_conservation_records']} | "
            f"{result['baseline_workflow_record_totals']['old_op_trace_records']} |"
        ),
        "",
        "## Metric Gaps",
        "",
        "| Metric | Pairs | Baseline mean (ms) | Current mean (ms) | Median delta (ms) | Median relative gap (%) |",
        "|--------|-------|--------------------|-------------------|-------------------|-------------------------|",
    ]
    for metric, summary in result["metric_summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(summary["paired_count"]),
                    _format_number(summary["baseline_mean_ms"]),
                    _format_number(summary["current_mean_ms"]),
                    _format_number(summary["paired_median_delta_ms"]),
                    _format_number(
                        summary["paired_median_relative_gap_percent"]
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Architecture Summary",
            "",
            "| Architecture | Cases | Baseline execution | Paired PASS | TTFT median relative gap (%) | E2E median relative gap (%) |",
            "|--------------|-------|--------------------|-------------|------------------------------|-----------------------------|",
        ]
    )
    for architecture, summary in result["architecture_summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    architecture,
                    str(summary["case_count"]),
                    json.dumps(
                        summary["baseline_execution_counts"],
                        sort_keys=True,
                    ),
                    str(summary["paired_execution_pass_count"]),
                    _format_number(
                        summary["metric_summaries"]["ttft_mean_ms"][
                            "paired_median_relative_gap_percent"
                        ]
                    ),
                    _format_number(
                        summary["metric_summaries"]["e2e_mean_ms"][
                            "paired_median_relative_gap_percent"
                        ]
                    ),
                ]
            )
            + " |"
        )

    failures = [
        row for row in result["cases"] if row["baseline_status"] != "PASS"
    ]
    lines.extend(
        [
            "",
            "## Baseline Failures",
            "",
        ]
    )
    if failures:
        lines.extend(
            [
                "| Case | Architecture | Exit code | Error |",
                "|------|--------------|-----------|-------|",
            ]
        )
        for row in failures:
            error = str(row["baseline_execution_errors"]).replace("|", "\\|")
            lines.append(
                f"| {row['case_id']} | {row['architecture']} | "
                f"{row['baseline_exit_code']} | {error} |"
            )
    else:
        lines.append("No baseline runtime failures.")

    lines.extend(
        [
            "",
            "## Per-Case Metrics",
            "",
            "| Case | Architecture | Kind | Routing | Baseline status | Workflow | TTFT baseline/current/delta (ms) | TPOT baseline/current/delta (ms) | E2E baseline/current/delta (ms) |",
            "|------|--------------|------|---------|-----------------|----------|----------------------------------|----------------------------------|---------------------------------|",
        ]
    )
    for row in result["cases"]:
        metric_text: list[str] = []
        for metric in ("ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms"):
            values = row["metrics"][metric]
            metric_text.append(
                "/".join(
                    _format_number(values[field], digits=4)
                    for field in ("baseline_ms", "current_ms", "delta_ms")
                )
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["architecture"]),
                    str(row["model_kind"]),
                    str(row["routing_distribution"]),
                    str(row["baseline_status"]),
                    str(row["baseline_workflow_status"]),
                    *metric_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_comparison_artifacts(
    result: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Write machine-readable, tabular, and concise human-readable evidence."""

    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "case_id",
        "architecture",
        "model_kind",
        "routing_distribution",
        "ep_size",
        "total_cards",
        "current_status",
        "baseline_status",
        "baseline_exit_code",
        "baseline_workflow_status",
        "current_ep_workload_records",
        "current_ep_barrier_records",
        "current_ep_conservation_records",
        "baseline_ep_workload_records",
        "baseline_ep_barrier_records",
        "baseline_ep_conservation_records",
        "baseline_old_op_trace_count",
        "baseline_old_moe_op_trace_count",
        "baseline_execution_errors",
    ]
    for metric in METRIC_FIELDS:
        fieldnames.extend(
            [
                f"{metric}_baseline_ms",
                f"{metric}_current_ms",
                f"{metric}_delta_ms",
                f"{metric}_absolute_gap_ms",
                f"{metric}_relative_gap_percent",
            ]
        )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["cases"]:
            flat = {field: row.get(field) for field in fieldnames}
            for metric in METRIC_FIELDS:
                values = row["metrics"][metric]
                for value_name, value in values.items():
                    flat[f"{metric}_{value_name}"] = value
            writer.writerow(flat)

    markdown_path.write_text(_markdown_report(result), encoding="utf-8")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--current-results-path", type=Path, required=True)
    parser.add_argument("--baseline-results-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--markdown-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = build_comparison(
        args.manifest_path,
        args.current_results_path,
        args.baseline_results_path,
    )
    write_comparison_artifacts(
        result,
        json_path=args.json_path,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
    )
    print(
        f"cases={result['case_count']} "
        f"paired_pass={result['paired_execution_pass_count']} "
        f"json={args.json_path} markdown={args.markdown_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
