#!/usr/bin/env python3
"""Audit staged true-mixed attention profiling CSVs before canonical merge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODELS: tuple[str, ...] = (
    "llama2_7b_dense_example",
    "Phi-tiny-MoE-instruct",
    "Step2Mini-tiny",
    "step-moe-noquant-small",
    "Qwen3-30B-A3B-tiny",
    "qwen3-next-80b-a3b-instruct-reduced-l2",
)

TRUE_MIXED_STAGE_FILES: tuple[str, ...] = (
    "attention_true_mixed.csv",
    "attention_true_mixed_kernel_only.csv",
)

POSITIVE_INTEGER_COLUMNS: tuple[str, ...] = (
    "decode_batch_size",
    "num_prefill_seqs",
    "total_batch_size",
    "total_prefill_tokens",
    "total_tokens",
)
POSITIVE_FLOAT_COLUMNS: tuple[str, ...] = (
    "decode_avg_kv_cache_size",
    "time_stats.attn_decode.median",
)
RATIO_COLUMNS: tuple[str, ...] = ("batch_composition_ratio",)
REQUIRED_NUMERIC_COLUMNS: tuple[str, ...] = (
    *RATIO_COLUMNS,
    "decode_avg_kv_cache_size",
    "decode_batch_size",
    "num_prefill_seqs",
    "total_batch_size",
    "total_prefill_tokens",
    "total_tokens",
)
REQUIRED_VALIDATION_COLUMNS: tuple[str, ...] = (
    *REQUIRED_NUMERIC_COLUMNS,
    "time_stats.attn_decode.median",
)
INVALID_NUMERIC_VALUES = {"", "nan", "none", "null"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy_csv_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_finite_float(value: object) -> float | None:
    raw_value = str(value).strip().lower()
    if raw_value in INVALID_NUMERIC_VALUES:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _valid_positive_number(value: object) -> bool:
    parsed = _parse_finite_float(value)
    return parsed is not None and parsed > 0.0


def _valid_positive_integer(value: object) -> bool:
    parsed = _parse_finite_float(value)
    return parsed is not None and parsed > 0.0 and parsed.is_integer()


def _valid_ratio(value: object) -> bool:
    parsed = _parse_finite_float(value)
    return parsed is not None and 0.0 < parsed < 1.0


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _increment_count(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _validate_true_mixed_numeric_row(
    row: dict[str, str],
) -> tuple[dict[str, int], list[str]]:
    invalid_columns: dict[str, int] = {}
    for column in POSITIVE_INTEGER_COLUMNS:
        if not _valid_positive_integer(row.get(column, "")):
            _increment_count(invalid_columns, column)
    for column in POSITIVE_FLOAT_COLUMNS:
        if not _valid_positive_number(row.get(column, "")):
            _increment_count(invalid_columns, column)
    for column in RATIO_COLUMNS:
        if not _valid_ratio(row.get(column, "")):
            _increment_count(invalid_columns, column)

    semantic_errors: list[str] = []
    decode_batch_size = _parse_finite_float(row.get("decode_batch_size", ""))
    num_prefill_seqs = _parse_finite_float(row.get("num_prefill_seqs", ""))
    total_batch_size = _parse_finite_float(row.get("total_batch_size", ""))
    total_prefill_tokens = _parse_finite_float(row.get("total_prefill_tokens", ""))
    total_tokens = _parse_finite_float(row.get("total_tokens", ""))
    batch_composition_ratio = _parse_finite_float(
        row.get("batch_composition_ratio", "")
    )

    if (
        decode_batch_size is not None
        and num_prefill_seqs is not None
        and total_batch_size is not None
        and not math.isclose(
            total_batch_size,
            decode_batch_size + num_prefill_seqs,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        semantic_errors.append(
            "total_batch_size != decode_batch_size + num_prefill_seqs"
        )
    if (
        num_prefill_seqs is not None
        and total_batch_size is not None
        and total_batch_size > 0.0
        and batch_composition_ratio is not None
        and not math.isclose(
            batch_composition_ratio,
            num_prefill_seqs / total_batch_size,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        semantic_errors.append(
            "batch_composition_ratio != num_prefill_seqs / total_batch_size"
        )
    if (
        decode_batch_size is not None
        and total_prefill_tokens is not None
        and total_tokens is not None
        and not math.isclose(
            total_tokens,
            total_prefill_tokens + decode_batch_size,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        semantic_errors.append("total_tokens != total_prefill_tokens + decode_batch_size")
    return invalid_columns, semantic_errors


def _get_tp_values_and_invalid_count(rows: Sequence[dict[str, str]]) -> tuple[list[int], int]:
    tp_values: set[int] = set()
    invalid_count = 0
    for row in rows:
        parsed = _parse_finite_float(row.get("num_tensor_parallel_workers", ""))
        if parsed is None or not parsed.is_integer():
            invalid_count += 1
        else:
            tp_values.add(int(parsed))
    return sorted(tp_values), invalid_count


def _count_duplicate_profile_keys(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
) -> int:
    key_columns = [
        fieldname for fieldname in fieldnames if not fieldname.startswith("time_stats.")
    ]
    rows_by_key: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row.get(column, "") for column in key_columns)
        rows_by_key[key] = rows_by_key.get(key, 0) + 1
    return sum(count - 1 for count in rows_by_key.values() if count > 1)


def _audit_file(
    path: Path,
    *,
    expected_true_mixed_rows_per_file: int | None,
    expected_tp_values: Sequence[int],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        report.update(
            {
                "status": "FAIL",
                "reason": "missing file",
                "row_count": 0,
                "true_mixed_row_count": 0,
                "true_mixed_attn_decode_valid_count": 0,
                "true_mixed_required_numeric_valid_row_count": 0,
                "true_mixed_required_numeric_invalid_cell_count": 0,
                "true_mixed_required_numeric_invalid_columns": {},
                "true_mixed_semantic_invalid_row_count": 0,
                "true_mixed_semantic_error_counts": {},
                "duplicate_true_mixed_profile_key_count": 0,
                "tp_values": [],
                "invalid_tp_row_count": 0,
            }
        )
        return report

    fieldnames, rows = _read_rows(path)
    time_stats_columns = [
        column for column in fieldnames if column.startswith("time_stats.")
    ]
    true_mixed_rows = [
        row for row in rows if _truthy_csv_value(row.get("is_true_mixed_batch", ""))
    ]
    valid_decode_rows = [
        row
        for row in true_mixed_rows
        if _valid_positive_number(row.get("time_stats.attn_decode.median", ""))
    ]

    invalid_numeric_columns: dict[str, int] = {}
    semantic_error_counts: dict[str, int] = {}
    required_numeric_valid_rows = 0
    required_numeric_invalid_cells = 0
    semantic_invalid_rows = 0
    for row in true_mixed_rows:
        row_invalid_columns, row_semantic_errors = _validate_true_mixed_numeric_row(row)
        for column, count in row_invalid_columns.items():
            invalid_numeric_columns[column] = invalid_numeric_columns.get(column, 0) + count
        for error in row_semantic_errors:
            semantic_error_counts[error] = semantic_error_counts.get(error, 0) + 1
        required_numeric_invalid_cells += sum(row_invalid_columns.values())
        if row_semantic_errors:
            semantic_invalid_rows += 1
        if not row_invalid_columns and not row_semantic_errors:
            required_numeric_valid_rows += 1

    tp_values, invalid_tp_row_count = _get_tp_values_and_invalid_count(rows)
    duplicate_true_mixed_profile_key_count = _count_duplicate_profile_keys(
        true_mixed_rows,
        fieldnames,
    )

    reasons: list[str] = []
    if not rows:
        reasons.append("row count is 0")
    if (
        expected_true_mixed_rows_per_file is not None
        and len(rows) != expected_true_mixed_rows_per_file
    ):
        reasons.append(
            "row count "
            f"{len(rows)} != expected {expected_true_mixed_rows_per_file}"
        )
    if len(true_mixed_rows) != len(rows):
        reasons.append(
            f"true-mixed rows {len(true_mixed_rows)} != row count {len(rows)}"
        )
    if not time_stats_columns:
        reasons.append("no time_stats columns")
    missing_required_columns = [
        column for column in REQUIRED_VALIDATION_COLUMNS if column not in fieldnames
    ]
    if missing_required_columns:
        reasons.append(f"missing required columns {missing_required_columns}")
    if len(valid_decode_rows) != len(true_mixed_rows):
        reasons.append(
            "valid attn_decode rows "
            f"{len(valid_decode_rows)} != true-mixed rows {len(true_mixed_rows)}"
        )
    if required_numeric_valid_rows != len(true_mixed_rows):
        reasons.append(
            "valid required numeric rows "
            f"{required_numeric_valid_rows} != true-mixed rows {len(true_mixed_rows)}"
        )
    if required_numeric_invalid_cells:
        details = ", ".join(
            f"{column}={count}"
            for column, count in sorted(invalid_numeric_columns.items())
        )
        reasons.append(
            f"invalid required numeric cells {required_numeric_invalid_cells}: {details}"
        )
    if semantic_invalid_rows:
        details = ", ".join(
            f"{error}={count}" for error, count in sorted(semantic_error_counts.items())
        )
        reasons.append(f"inconsistent true-mixed rows: {details}")
    if duplicate_true_mixed_profile_key_count:
        reasons.append(
            "duplicate true-mixed profile keys "
            f"{duplicate_true_mixed_profile_key_count}"
        )
    expected_tp_set = set(expected_tp_values)
    if invalid_tp_row_count:
        reasons.append(f"invalid num_tensor_parallel_workers rows {invalid_tp_row_count}")
    if expected_tp_set and set(tp_values) != expected_tp_set:
        reasons.append(
            f"tp values {tp_values} != expected {sorted(expected_tp_set)}"
        )

    report.update(
        {
            "status": "FAIL" if reasons else "PASS",
            "reasons": reasons,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "column_count": len(fieldnames),
            "row_count": len(rows),
            "true_mixed_row_count": len(true_mixed_rows),
            "time_stats_column_count": len(time_stats_columns),
            "true_mixed_attn_decode_valid_count": len(valid_decode_rows),
            "true_mixed_required_numeric_valid_row_count": (
                required_numeric_valid_rows
            ),
            "true_mixed_required_numeric_invalid_cell_count": (
                required_numeric_invalid_cells
            ),
            "true_mixed_required_numeric_invalid_columns": invalid_numeric_columns,
            "true_mixed_semantic_invalid_row_count": semantic_invalid_rows,
            "true_mixed_semantic_error_counts": semantic_error_counts,
            "duplicate_true_mixed_profile_key_count": (
                duplicate_true_mixed_profile_key_count
            ),
            "tp_values": tp_values,
            "invalid_tp_row_count": invalid_tp_row_count,
        }
    )
    return report


def audit_stage(
    *,
    stage_root: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    expected_true_mixed_rows_per_file: int | None = None,
    expected_tp_values: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for model in models:
        for filename in TRUE_MIXED_STAGE_FILES:
            path = stage_root / model / filename
            report = _audit_file(
                path,
                expected_true_mixed_rows_per_file=expected_true_mixed_rows_per_file,
                expected_tp_values=expected_tp_values,
            )
            report.update({"model": model, "filename": filename})
            reports.append(report)

    return {
        "status": "PASS"
        if reports and all(report["status"] == "PASS" for report in reports)
        else "FAIL",
        "stage_root": str(stage_root),
        "expected_model_count": len(models),
        "expected_file_count": len(models) * len(TRUE_MIXED_STAGE_FILES),
        "observed_file_count": sum(1 for report in reports if report["exists"]),
        "report_count": len(reports),
        "pass_count": sum(1 for report in reports if report["status"] == "PASS"),
        "fail_count": sum(1 for report in reports if report["status"] != "PASS"),
        "total_row_count": sum(int(report.get("row_count", 0)) for report in reports),
        "total_true_mixed_row_count": sum(
            int(report.get("true_mixed_row_count", 0)) for report in reports
        ),
        "reports": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit staged attention_true_mixed*.csv files before building a "
            "canonical H800 profiling supplement."
        )
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help=(
            "Staged device profiling root, e.g. "
            "<stage>/profiling/compute/h800."
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--expected-true-mixed-rows-per-file",
        type=int,
        default=None,
        help="Strict expected row count for each true-mixed staged file.",
    )
    parser.add_argument(
        "--expected-tp-values",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8),
        help="Strict expected TP values in every staged file.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = audit_stage(
        stage_root=args.stage_root,
        models=tuple(args.models),
        expected_true_mixed_rows_per_file=args.expected_true_mixed_rows_per_file,
        expected_tp_values=tuple(args.expected_tp_values),
    )
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
