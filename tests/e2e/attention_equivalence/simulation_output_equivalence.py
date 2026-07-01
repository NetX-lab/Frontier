#!/usr/bin/env python3
"""Strict equivalence checks for Frontier simulation output artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from frontier.metrics.constants import RequestMetricsTimeDistributions
from tests.e2e.attention_equivalence.measurement_csv_equivalence import (
    FAIL_STATUS,
    MISMATCH_EXIT_CODE,
    PASS_EXIT_CODE,
    PASS_STATUS,
    compare_csv_files,
)
from tests.e2e.attention_equivalence.profile_manifest import write_json_report

REQUEST_METRICS = "request_metrics.csv"
DEFAULT_REQUEST_KEY_COLUMNS = ("Request Id",)
SUMMARY_COLUMNS = (
    RequestMetricsTimeDistributions.REQUEST_E2E_TIME.value,
    RequestMetricsTimeDistributions.TTFT.value,
    RequestMetricsTimeDistributions.TPOT.value,
    RequestMetricsTimeDistributions.REQUEST_WAITING_TIME_TOTAL.value,
    RequestMetricsTimeDistributions.TRANSFER_KV_CACHE.value,
    "request_num_prefill_tokens",
    "request_num_decode_tokens",
    "request_spec_accepted_drafts",
    "request_spec_rejected_drafts",
    "request_spec_committed_tokens",
)


def _read_numeric_values(report: Mapping[str, Any], column: str, side: str) -> list[float]:
    key = "reference_value" if side == "reference" else "candidate_value"
    return [
        comparison[key]
        for comparison in report["numeric_comparisons"]
        if comparison["column"] == column
    ]


def _aggregate_column(report: Mapping[str, Any], column: str) -> dict[str, float]:
    reference_sum = sum(_read_numeric_values(report, column, "reference"))
    candidate_sum = sum(_read_numeric_values(report, column, "candidate"))
    return {
        "reference_sum": reference_sum,
        "candidate_sum": candidate_sum,
        "absolute_delta": abs(candidate_sum - reference_sum),
    }


def _request_metrics_summary(artifact_report: Mapping[str, Any]) -> dict[str, Any]:
    aggregates = {
        column: _aggregate_column(artifact_report, column)
        for column in SUMMARY_COLUMNS
        if any(
            comparison["column"] == column
            for comparison in artifact_report["numeric_comparisons"]
        )
    }
    row_count = artifact_report["row_count"]
    summary: dict[str, Any] = {
        "total_completed_requests": dict(row_count),
        "aggregates": aggregates,
    }
    e2e_values = _read_numeric_values(
        artifact_report,
        RequestMetricsTimeDistributions.REQUEST_E2E_TIME.value,
        "reference",
    )
    decode_token_values = _read_numeric_values(
        artifact_report,
        "request_num_decode_tokens",
        "reference",
    )
    if e2e_values:
        duration_seconds = max(e2e_values) / 1000.0
        if duration_seconds > 0:
            summary["reference_throughput"] = {
                "completed_requests_per_second": row_count["reference"] / duration_seconds,
                "decode_tokens_per_second": sum(decode_token_values) / duration_seconds,
            }
    return summary


def compare_simulation_outputs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    artifact_names: Sequence[str] = (REQUEST_METRICS,),
    key_columns_by_artifact: Mapping[str, Sequence[str]] | None = None,
    tolerance_allowlist_by_artifact: Mapping[str, Mapping[str, Mapping[str, float] | float]] | None = None,
    case_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare simulation output directories with required artifact declarations."""

    reference_root = Path(reference_dir)
    candidate_root = Path(candidate_dir)
    artifacts: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    mismatch_count = 0
    key_columns_by_artifact = key_columns_by_artifact or {}
    tolerance_allowlist_by_artifact = tolerance_allowlist_by_artifact or {}

    for artifact_name in artifact_names:
        key_columns = tuple(key_columns_by_artifact.get(artifact_name, DEFAULT_REQUEST_KEY_COLUMNS))
        artifact_report = compare_csv_files(
            reference_root / artifact_name,
            candidate_root / artifact_name,
            key_columns=key_columns,
            tolerance_allowlist=tolerance_allowlist_by_artifact.get(artifact_name),
            case_manifest=case_manifest,
            include_per_op_statistics=artifact_name.endswith("op_traces.csv"),
            artifact_label=artifact_name,
        )
        artifacts[artifact_name] = artifact_report
        mismatch_count += int(artifact_report["mismatch_count"])
        if artifact_name == REQUEST_METRICS:
            summary[artifact_name] = _request_metrics_summary(artifact_report)

    return {
        "status": PASS_STATUS if mismatch_count == 0 else FAIL_STATUS,
        "mismatch_count": mismatch_count,
        "reference_dir": str(reference_root),
        "candidate_dir": str(candidate_root),
        "case_manifest": dict(case_manifest or {}),
        "artifacts": artifacts,
        "summary": summary,
    }


def _parse_artifacts(raw: str) -> tuple[str, ...]:
    artifacts = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not artifacts:
        raise ValueError("--artifacts must contain at least one artifact name")
    return artifacts


def _load_json_mapping(raw: str | None) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    return json.loads(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--artifacts", default=REQUEST_METRICS)
    parser.add_argument("--case-manifest-json", default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    report = compare_simulation_outputs(
        args.reference_dir,
        args.candidate_dir,
        artifact_names=_parse_artifacts(args.artifacts),
        case_manifest=_load_json_mapping(args.case_manifest_json),
    )
    write_json_report(args.output_json, report)
    return PASS_EXIT_CODE if report["status"] == PASS_STATUS else MISMATCH_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
