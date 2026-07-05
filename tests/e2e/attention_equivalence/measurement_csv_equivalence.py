#!/usr/bin/env python3
"""Strict keyed equivalence checks for attention measurement CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.e2e.attention_equivalence.profile_manifest import file_manifest, write_json_report

PASS_STATUS = "PASS"
FAIL_STATUS = "FAIL"
STRUCTURAL_EXIT_CODE = 2
MISMATCH_EXIT_CODE = 1
PASS_EXIT_CODE = 0

_TIME_STATS_RE = re.compile(r"^time_stats\.(?P<op>.+)\.(?P<stat>min|max|mean|median|std)$")
WILDCARD_TOLERANCE_COLUMN = "*"


@dataclass(frozen=True)
class NumericTolerance:
    """Allowed numeric tolerance for one column."""

    absolute: float = 0.0
    relative_pct: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative_pct": self.relative_pct}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"required CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def _normalize_tolerances(
    tolerance_allowlist: Mapping[str, Mapping[str, float] | float] | None,
) -> dict[str, NumericTolerance]:
    normalized: dict[str, NumericTolerance] = {}
    for column, raw in (tolerance_allowlist or {}).items():
        if isinstance(raw, (int, float)):
            tolerance = NumericTolerance(absolute=float(raw), relative_pct=0.0)
        else:
            tolerance = NumericTolerance(
                absolute=float(raw.get("absolute", 0.0)),
                relative_pct=float(raw.get("relative_pct", 0.0)),
            )
        if tolerance.absolute < 0 or tolerance.relative_pct < 0:
            raise ValueError(f"negative tolerance is not allowed for column {column!r}")
        normalized[column] = tolerance
    return normalized


def _ensure_schema_equal(reference_columns: Sequence[str], candidate_columns: Sequence[str]) -> None:
    if list(reference_columns) != list(candidate_columns):
        reference_set = set(reference_columns)
        candidate_set = set(candidate_columns)
        raise ValueError(
            "schema mismatch: "
            f"missing_in_candidate={sorted(reference_set - candidate_set)}, "
            f"extra_in_candidate={sorted(candidate_set - reference_set)}, "
            f"same_columns_different_order={reference_set == candidate_set and list(reference_columns) != list(candidate_columns)}"
        )


def _ensure_key_columns(columns: Sequence[str], key_columns: Sequence[str]) -> None:
    if not key_columns:
        raise ValueError("key_columns must be explicitly provided")
    missing = [column for column in key_columns if column not in columns]
    if missing:
        raise ValueError(f"key column missing from schema: {missing}")


def _ensure_non_empty_rows(
    reference_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
) -> None:
    if not reference_rows or not candidate_rows:
        raise ValueError(
            "CSV must contain at least one data row on both sides: "
            f"reference_rows={len(reference_rows)}, candidate_rows={len(candidate_rows)}"
        )


def _key_for_row(row: Mapping[str, str], key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in key_columns)


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    key_columns: Sequence[str],
    side_name: str,
) -> dict[tuple[str, ...], Mapping[str, str]]:
    indexed: dict[tuple[str, ...], Mapping[str, str]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = _key_for_row(row, key_columns)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        raise ValueError(f"duplicate key in {side_name}: {duplicates[:5]}")
    return indexed


def _parse_float(value: Any) -> tuple[bool, float | None]:
    if value is None:
        return False, None
    text = str(value).strip()
    if text == "":
        return False, None
    try:
        return True, float(text)
    except ValueError:
        return False, None


def _relative_delta_pct(reference_value: float, candidate_value: float) -> float:
    if math.isnan(reference_value) or math.isnan(candidate_value):
        return math.nan
    if math.isinf(reference_value) or math.isinf(candidate_value):
        return 0.0 if reference_value == candidate_value else math.inf
    if reference_value == 0:
        return 0.0 if candidate_value == 0 else math.inf
    return abs(candidate_value - reference_value) / abs(reference_value) * 100.0


def _numeric_values_equal(
    reference_value: float,
    candidate_value: float,
    tolerance: NumericTolerance,
) -> tuple[bool, float, float]:
    if math.isnan(reference_value) or math.isnan(candidate_value):
        both_nan = math.isnan(reference_value) and math.isnan(candidate_value)
        return both_nan, 0.0 if both_nan else math.nan, 0.0 if both_nan else math.nan
    if math.isinf(reference_value) or math.isinf(candidate_value):
        same_infinity = reference_value == candidate_value
        return same_infinity, 0.0 if same_infinity else math.inf, 0.0 if same_infinity else math.inf

    absolute_delta = abs(candidate_value - reference_value)
    relative_delta = _relative_delta_pct(reference_value, candidate_value)
    if tolerance.absolute == 0.0 and tolerance.relative_pct == 0.0:
        return reference_value == candidate_value, absolute_delta, relative_delta
    return (
        absolute_delta <= tolerance.absolute
        or relative_delta <= tolerance.relative_pct
    ), absolute_delta, relative_delta


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    sorted_values = sorted(values)
    index = max(math.ceil(percentile / 100.0 * len(sorted_values)) - 1, 0)
    return sorted_values[min(index, len(sorted_values) - 1)]


def _wide_time_stats(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    op_to_mean_column = {
        match.group("op"): column
        for column in columns
        for match in [_TIME_STATS_RE.match(column)]
        if match is not None and match.group("stat") == "mean"
    }
    result: dict[str, dict[str, Any]] = {}
    for op_name, column in sorted(op_to_mean_column.items()):
        values: list[float] = []
        for row in rows:
            parsed, value = _parse_float(row.get(column))
            if parsed and value is not None and not math.isnan(value) and not math.isinf(value):
                values.append(value)
        if values:
            result[op_name] = {
                "sample_count": len(values),
                "mean_of_means_ms": sum(values) / len(values),
                "median_of_means_ms": _percentile_nearest_rank(values, 50.0),
                "p95_of_means_ms": _percentile_nearest_rank(values, 95.0),
            }
        else:
            result[op_name] = {
                "sample_count": 0,
                "mean_of_means_ms": None,
                "median_of_means_ms": None,
                "p95_of_means_ms": None,
            }
    return result


def _comparison_key_dict(key_columns: Sequence[str], key: Sequence[str]) -> dict[str, str]:
    return dict(zip(key_columns, key, strict=True))


def compare_csv_files(
    reference_csv: str | Path,
    candidate_csv: str | Path,
    *,
    key_columns: Sequence[str],
    tolerance_allowlist: Mapping[str, Mapping[str, float] | float] | None = None,
    case_manifest: Mapping[str, Any] | None = None,
    include_per_op_statistics: bool = False,
    artifact_label: str = "csv",
) -> dict[str, Any]:
    """Compare two CSV files with explicit keyed rows and strict default equality."""

    reference_path = Path(reference_csv)
    candidate_path = Path(candidate_csv)
    reference_columns, reference_rows = _read_csv(reference_path)
    candidate_columns, candidate_rows = _read_csv(candidate_path)
    _ensure_schema_equal(reference_columns, candidate_columns)
    _ensure_key_columns(reference_columns, key_columns)
    _ensure_non_empty_rows(reference_rows, candidate_rows)
    tolerances = _normalize_tolerances(tolerance_allowlist)

    reference_index = _index_rows(reference_rows, key_columns, "reference")
    candidate_index = _index_rows(candidate_rows, key_columns, "candidate")
    reference_keys = set(reference_index)
    candidate_keys = set(candidate_index)
    missing_keys = sorted(reference_keys - candidate_keys)
    extra_keys = sorted(candidate_keys - reference_keys)

    numeric_comparisons: list[dict[str, Any]] = []
    non_numeric_mismatches: list[dict[str, Any]] = []
    type_mismatches: list[dict[str, Any]] = []
    mismatch_count = len(missing_keys) + len(extra_keys)

    compared_columns = [column for column in reference_columns if column not in key_columns]
    for key in sorted(reference_keys & candidate_keys):
        reference_row = reference_index[key]
        candidate_row = candidate_index[key]
        for column in compared_columns:
            reference_raw = reference_row.get(column, "")
            candidate_raw = candidate_row.get(column, "")
            reference_is_number, reference_number = _parse_float(reference_raw)
            candidate_is_number, candidate_number = _parse_float(candidate_raw)
            key_dict = _comparison_key_dict(key_columns, key)
            if reference_is_number and candidate_is_number:
                assert reference_number is not None
                assert candidate_number is not None
                tolerance = tolerances.get(
                    column,
                    tolerances.get(WILDCARD_TOLERANCE_COLUMN, NumericTolerance()),
                )
                passed, absolute_delta, relative_delta = _numeric_values_equal(
                    reference_number,
                    candidate_number,
                    tolerance,
                )
                comparison = {
                    "key": key_dict,
                    "column": column,
                    "reference_value": reference_number,
                    "candidate_value": candidate_number,
                    "absolute_delta": absolute_delta,
                    "relative_delta_pct": relative_delta,
                    "tolerance": tolerance.as_dict(),
                    "passed": passed,
                }
                numeric_comparisons.append(comparison)
                if not passed:
                    mismatch_count += 1
            elif reference_is_number != candidate_is_number:
                mismatch_count += 1
                type_mismatches.append(
                    {
                        "key": key_dict,
                        "column": column,
                        "reference_value": reference_raw,
                        "candidate_value": candidate_raw,
                    }
                )
            elif str(reference_raw) != str(candidate_raw):
                mismatch_count += 1
                non_numeric_mismatches.append(
                    {
                        "key": key_dict,
                        "column": column,
                        "reference_value": reference_raw,
                        "candidate_value": candidate_raw,
                    }
                )

    report: dict[str, Any] = {
        "artifact_label": artifact_label,
        "status": PASS_STATUS if mismatch_count == 0 else FAIL_STATUS,
        "mismatch_count": mismatch_count,
        "reference": file_manifest(reference_path),
        "candidate": file_manifest(candidate_path),
        "row_count": {
            "reference": len(reference_rows),
            "candidate": len(candidate_rows),
            "delta": len(candidate_rows) - len(reference_rows),
        },
        "column_count": {
            "reference": len(reference_columns),
            "candidate": len(candidate_columns),
            "delta": len(candidate_columns) - len(reference_columns),
        },
        "key_columns": list(key_columns),
        "case_manifest": dict(case_manifest or {}),
        "tolerance_allowlist": {
            column: tolerance.as_dict() for column, tolerance in sorted(tolerances.items())
        },
        "missing_keys": [_comparison_key_dict(key_columns, key) for key in missing_keys],
        "extra_keys": [_comparison_key_dict(key_columns, key) for key in extra_keys],
        "numeric_comparisons": numeric_comparisons,
        "non_numeric_mismatches": non_numeric_mismatches,
        "type_mismatches": type_mismatches,
    }
    if include_per_op_statistics:
        reference_stats = _wide_time_stats(reference_rows, reference_columns)
        candidate_stats = _wide_time_stats(candidate_rows, candidate_columns)
        report["per_op_statistics"] = {
            op_name: {
                "reference": reference_stats.get(op_name, {}),
                "candidate": candidate_stats.get(op_name, {}),
            }
            for op_name in sorted(set(reference_stats) | set(candidate_stats))
        }
    return report


def compare_measurement_csv(
    reference_csv: str | Path,
    candidate_csv: str | Path,
    *,
    key_columns: Sequence[str],
    tolerance_allowlist: Mapping[str, Mapping[str, float] | float] | None = None,
    case_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare reference and candidate attention measurement CSVs."""

    return compare_csv_files(
        reference_csv,
        candidate_csv,
        key_columns=key_columns,
        tolerance_allowlist=tolerance_allowlist,
        case_manifest=case_manifest,
        include_per_op_statistics=True,
        artifact_label="measurement_csv",
    )


def _parse_key_columns(raw: str) -> tuple[str, ...]:
    columns = tuple(column.strip() for column in raw.split(",") if column.strip())
    if not columns:
        raise ValueError("--key-columns must contain at least one column")
    return columns


def _load_json_mapping(raw: str | None) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    return json.loads(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--key-columns", required=True, help="Comma-separated stable key columns")
    parser.add_argument("--tolerance-json", default=None, help="JSON tolerance allowlist")
    parser.add_argument("--case-manifest-json", default=None, help="JSON manifest for pinned inputs")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    report = compare_measurement_csv(
        args.reference_csv,
        args.candidate_csv,
        key_columns=_parse_key_columns(args.key_columns),
        tolerance_allowlist=_load_json_mapping(args.tolerance_json),
        case_manifest=_load_json_mapping(args.case_manifest_json),
    )
    write_json_report(args.output_json, report)
    return PASS_EXIT_CODE if report["status"] == PASS_STATUS else MISMATCH_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
