#!/usr/bin/env python3
"""Strict profiling CSV equivalence checks for operator-parity golden profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from tests.e2e.attention_equivalence.measurement_csv_equivalence import (
    FAIL_STATUS,
    MISMATCH_EXIT_CODE,
    PASS_EXIT_CODE,
    PASS_STATUS,
)
from tests.e2e.attention_equivalence.profile_manifest import write_json_report
from tests.e2e.operator_parity.profile_prerequisite_audit import (
    GOLDEN_CONFIG_FILENAMES,
    build_requirements,
)

_METADATA_COLUMNS_ALLOWED_FOR_EXPLICIT_MIGRATION = frozenset({"model_architecture_profile"})


def _required_relative_paths(*, config_root: Path, config_filenames: Sequence[str]) -> tuple[str, ...]:
    requirements = build_requirements(config_root=config_root, config_filenames=config_filenames)
    paths = [
        f"{requirement.model_name}/{filename}"
        for requirement in requirements
        for filename in requirement.required_files
    ]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate required profiling CSV paths: {paths}")
    return tuple(sorted(paths))


def _actual_csv_relative_paths(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"profiling CSV root missing: {root}")
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*.csv")))


def _inspect_required_file_set(root: Path, required_paths: Sequence[str]) -> tuple[str, ...]:
    actual_paths = set(_actual_csv_relative_paths(root))
    required_path_set = set(required_paths)
    missing = sorted(required_path_set - actual_paths)
    if missing:
        first_missing = root / missing[0]
        raise FileNotFoundError(
            "required profiling CSV missing: "
            f"root={root}, missing_required_files={missing}, first_missing={first_missing}"
        )
    return tuple(sorted(actual_paths - required_path_set))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"required profiling CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return list(reader.fieldnames), rows


def _normalize_candidate_schema(
    reference_columns: Sequence[str],
    candidate_columns: Sequence[str],
    *,
    ignored_candidate_extra_metadata_columns: frozenset[str],
) -> list[str]:
    candidate_extra = [column for column in candidate_columns if column not in reference_columns]
    if candidate_extra and set(candidate_extra).issubset(ignored_candidate_extra_metadata_columns):
        normalized_candidate_columns = [
            column for column in candidate_columns if column not in ignored_candidate_extra_metadata_columns
        ]
    else:
        normalized_candidate_columns = list(candidate_columns)

    if list(reference_columns) != normalized_candidate_columns:
        reference_set = set(reference_columns)
        normalized_candidate_set = set(normalized_candidate_columns)
        raise ValueError(
            "schema mismatch: "
            f"missing_in_candidate={sorted(reference_set - normalized_candidate_set)}, "
            f"extra_in_candidate={sorted(normalized_candidate_set - reference_set)}, "
            f"ignored_candidate_extra_metadata_columns={sorted(ignored_candidate_extra_metadata_columns)}, "
            f"same_columns_different_order={reference_set == normalized_candidate_set and list(reference_columns) != normalized_candidate_columns}"
        )
    return normalized_candidate_columns


def _validate_ignored_candidate_metadata_values(
    *,
    relative_path: str,
    candidate_path: Path,
    candidate_rows: Sequence[dict[str, str]],
    candidate_extra_columns: Sequence[str],
    ignored_candidate_extra_metadata_columns: frozenset[str],
    expected_model_architecture_profile: str,
) -> dict[str, dict[str, Any]]:
    ignored_extra_columns = sorted(
        set(candidate_extra_columns) & ignored_candidate_extra_metadata_columns
    )
    reports: dict[str, dict[str, Any]] = {}
    for column in ignored_extra_columns:
        values: set[str] = set()
        for row_index, row in enumerate(candidate_rows):
            raw_value = str(row.get(column, "")).strip()
            if not raw_value:
                raise ValueError(
                    "ignored candidate metadata mismatch: "
                    f"file={relative_path}, path={candidate_path}, row_index={row_index}, "
                    f"column={column}, expected_non_empty=true, observed={raw_value!r}"
                )
            normalized_value = raw_value.lower()
            if (
                column == "model_architecture_profile"
                and normalized_value != expected_model_architecture_profile
            ):
                raise ValueError(
                    "ignored candidate metadata mismatch: "
                    f"file={relative_path}, path={candidate_path}, row_index={row_index}, "
                    f"column={column}, expected={expected_model_architecture_profile!r}, "
                    f"observed={raw_value!r}"
                )
            values.add(normalized_value)
        reports[column] = {
            "column": column,
            "cell_count": len(candidate_rows),
            "values": sorted(values),
        }
    return reports


def _decimal_cell(value: str, *, path: Path, column: str, row_index: int) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            "time_stats numeric cell is not parseable: "
            f"path={path}, row_index={row_index}, column={column}, value={value!r}"
        ) from exc


def _relative_delta_pct(reference_value: Decimal, candidate_value: Decimal) -> float:
    if reference_value == 0:
        return 0.0 if candidate_value == 0 else math.inf
    return float(abs(candidate_value - reference_value) / abs(reference_value) * Decimal(100))


def _compare_csv_file(
    reference_path: Path,
    candidate_path: Path,
    *,
    relative_path: str,
    ignored_candidate_extra_metadata_columns: frozenset[str],
    expected_model_architecture_profile: str,
) -> dict[str, Any]:
    reference_columns, reference_rows = _read_csv(reference_path)
    candidate_columns, candidate_rows = _read_csv(candidate_path)
    candidate_extra_columns = [
        column for column in candidate_columns if column not in reference_columns
    ]
    ignored_metadata_value_reports = _validate_ignored_candidate_metadata_values(
        relative_path=relative_path,
        candidate_path=candidate_path,
        candidate_rows=candidate_rows,
        candidate_extra_columns=candidate_extra_columns,
        ignored_candidate_extra_metadata_columns=ignored_candidate_extra_metadata_columns,
        expected_model_architecture_profile=expected_model_architecture_profile,
    )
    columns = _normalize_candidate_schema(
        reference_columns,
        candidate_columns,
        ignored_candidate_extra_metadata_columns=ignored_candidate_extra_metadata_columns,
    )
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            "row-count mismatch: "
            f"file={relative_path}, reference_rows={len(reference_rows)}, candidate_rows={len(candidate_rows)}"
        )

    mismatches: list[dict[str, Any]] = []
    numeric_cells_compared = 0
    numeric_mismatch_count = 0
    text_cells_compared = 0
    text_mismatch_count = 0
    time_stats_blank_cells_matched = 0
    time_stats_blank_mismatch_count = 0
    max_abs_delta = 0.0
    max_relative_delta_pct = 0.0

    for row_index, (reference_row, candidate_row) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        for column in columns:
            reference_value = reference_row[column]
            candidate_value = candidate_row[column]
            if column.startswith("time_stats."):
                numeric_cells_compared += 1
                if reference_value == "" or candidate_value == "":
                    if reference_value == candidate_value:
                        time_stats_blank_cells_matched += 1
                    else:
                        time_stats_blank_mismatch_count += 1
                        mismatches.append(
                            {
                                "kind": "time_stats_blank_mismatch",
                                "file": relative_path,
                                "row_index": row_index,
                                "column": column,
                                "reference_value": reference_value,
                                "candidate_value": candidate_value,
                            }
                        )
                    continue
                reference_number = _decimal_cell(
                    reference_value, path=reference_path, column=column, row_index=row_index
                )
                candidate_number = _decimal_cell(
                    candidate_value, path=candidate_path, column=column, row_index=row_index
                )
                abs_delta_decimal = abs(candidate_number - reference_number)
                abs_delta = float(abs_delta_decimal)
                relative_delta_pct = _relative_delta_pct(reference_number, candidate_number)
                max_abs_delta = max(max_abs_delta, abs_delta)
                max_relative_delta_pct = max(max_relative_delta_pct, relative_delta_pct)
                if reference_number != candidate_number:
                    numeric_mismatch_count += 1
                    mismatches.append(
                        {
                            "kind": "numeric_mismatch",
                            "file": relative_path,
                            "row_index": row_index,
                            "column": column,
                            "reference_value": reference_value,
                            "candidate_value": candidate_value,
                            "absolute_delta": abs_delta,
                            "relative_delta_pct": relative_delta_pct,
                        }
                    )
            else:
                text_cells_compared += 1
                if reference_value != candidate_value:
                    text_mismatch_count += 1
                    mismatches.append(
                        {
                            "kind": "text_mismatch",
                            "file": relative_path,
                            "row_index": row_index,
                            "column": column,
                            "reference_value": reference_value,
                            "candidate_value": candidate_value,
                        }
                    )

    return {
        "status": PASS_STATUS if not mismatches else FAIL_STATUS,
        "file": relative_path,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "row_count": len(reference_rows),
        "column_count": len(columns),
        "numeric_cells_compared": numeric_cells_compared,
        "text_cells_compared": text_cells_compared,
        "mismatch_count": len(mismatches),
        "numeric_mismatch_count": numeric_mismatch_count,
        "text_mismatch_count": text_mismatch_count,
        "time_stats_blank_cells_matched": time_stats_blank_cells_matched,
        "time_stats_blank_mismatch_count": time_stats_blank_mismatch_count,
        "max_abs_delta": max_abs_delta,
        "max_relative_delta_pct": max_relative_delta_pct,
        "ignored_candidate_extra_metadata": ignored_metadata_value_reports,
        "ignored_candidate_extra_metadata_cell_count": sum(
            int(report["cell_count"]) for report in ignored_metadata_value_reports.values()
        ),
        "mismatches": mismatches,
    }


def _validate_ignored_metadata_columns(columns: Sequence[str]) -> frozenset[str]:
    requested = frozenset(columns)
    unsupported = sorted(requested - _METADATA_COLUMNS_ALLOWED_FOR_EXPLICIT_MIGRATION)
    if unsupported:
        raise ValueError(
            "unsupported ignored candidate metadata columns: "
            f"{unsupported}; allowed={sorted(_METADATA_COLUMNS_ALLOWED_FOR_EXPLICIT_MIGRATION)}"
        )
    return requested


def compare_profiling_csv_roots(
    *,
    config_root: str | Path,
    reference_profile_root: str | Path,
    candidate_profile_root: str | Path,
    config_filenames: Sequence[str] = GOLDEN_CONFIG_FILENAMES,
    ignore_candidate_extra_metadata_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare all required golden profiling CSV files under two roots."""

    config_root_path = Path(config_root)
    reference_root_path = Path(reference_profile_root)
    candidate_root_path = Path(candidate_profile_root)
    ignored_metadata_columns = _validate_ignored_metadata_columns(
        ignore_candidate_extra_metadata_columns
    )
    requirements = build_requirements(
        config_root=config_root_path, config_filenames=config_filenames
    )
    required_paths = tuple(
        sorted(
            f"{requirement.model_name}/{filename}"
            for requirement in requirements
            for filename in requirement.required_files
        )
    )
    if len(required_paths) != len(set(required_paths)):
        raise ValueError(f"duplicate required profiling CSV paths: {required_paths}")
    expected_profiles_by_path = {
        f"{requirement.model_name}/{filename}": (
            requirement.expected_model_architecture_profile
        )
        for requirement in requirements
        for filename in requirement.required_files
    }
    if required_paths != _required_relative_paths(
        config_root=config_root_path, config_filenames=config_filenames
    ):
        raise AssertionError("internal required profiling CSV path builder mismatch")
    reference_extra_csv_files = _inspect_required_file_set(reference_root_path, required_paths)
    candidate_extra_csv_files = _inspect_required_file_set(candidate_root_path, required_paths)

    file_reports = [
        _compare_csv_file(
            reference_root_path / relative_path,
            candidate_root_path / relative_path,
            relative_path=relative_path,
            ignored_candidate_extra_metadata_columns=ignored_metadata_columns,
            expected_model_architecture_profile=expected_profiles_by_path[relative_path],
        )
        for relative_path in required_paths
    ]
    mismatch_count = sum(int(file_report["mismatch_count"]) for file_report in file_reports)
    numeric_mismatch_count = sum(
        int(file_report["numeric_mismatch_count"]) for file_report in file_reports
    )
    text_mismatch_count = sum(int(file_report["text_mismatch_count"]) for file_report in file_reports)
    time_stats_blank_cells_matched = sum(
        int(file_report["time_stats_blank_cells_matched"]) for file_report in file_reports
    )
    time_stats_blank_mismatch_count = sum(
        int(file_report["time_stats_blank_mismatch_count"]) for file_report in file_reports
    )
    ignored_candidate_extra_metadata: dict[str, dict[str, Any]] = {}
    for file_report in file_reports:
        for column, column_report in file_report[
            "ignored_candidate_extra_metadata"
        ].items():
            aggregate = ignored_candidate_extra_metadata.setdefault(
                column,
                {"column": column, "cell_count": 0, "values": set()},
            )
            aggregate["cell_count"] += int(column_report["cell_count"])
            aggregate["values"].update(column_report["values"])
    ignored_candidate_extra_metadata_json = {
        column: {
            "column": column,
            "cell_count": int(report["cell_count"]),
            "values": sorted(report["values"]),
        }
        for column, report in sorted(ignored_candidate_extra_metadata.items())
    }
    return {
        "status": PASS_STATUS if mismatch_count == 0 else FAIL_STATUS,
        "config_root": str(config_root_path),
        "reference_profile_root": str(reference_root_path),
        "candidate_profile_root": str(candidate_root_path),
        "required_files": list(required_paths),
        "required_file_count": len(required_paths),
        "reference_extra_csv_files": list(reference_extra_csv_files),
        "candidate_extra_csv_files": list(candidate_extra_csv_files),
        "reference_extra_csv_count": len(reference_extra_csv_files),
        "candidate_extra_csv_count": len(candidate_extra_csv_files),
        "file_count": len(file_reports),
        "row_count": sum(int(file_report["row_count"]) for file_report in file_reports),
        "numeric_cells_compared": sum(
            int(file_report["numeric_cells_compared"]) for file_report in file_reports
        ),
        "text_cells_compared": sum(
            int(file_report["text_cells_compared"]) for file_report in file_reports
        ),
        "mismatch_count": mismatch_count,
        "numeric_mismatch_count": numeric_mismatch_count,
        "text_mismatch_count": text_mismatch_count,
        "time_stats_blank_cells_matched": time_stats_blank_cells_matched,
        "time_stats_blank_mismatch_count": time_stats_blank_mismatch_count,
        "max_abs_delta": max(float(file_report["max_abs_delta"]) for file_report in file_reports)
        if file_reports
        else 0.0,
        "max_relative_delta_pct": max(
            float(file_report["max_relative_delta_pct"]) for file_report in file_reports
        )
        if file_reports
        else 0.0,
        "ignored_candidate_extra_metadata_columns": sorted(ignored_metadata_columns),
        "ignored_candidate_extra_metadata": ignored_candidate_extra_metadata_json,
        "ignored_candidate_extra_metadata_cell_count": sum(
            int(report["cell_count"])
            for report in ignored_candidate_extra_metadata_json.values()
        ),
        "file_reports": file_reports,
        "mismatches": [
            mismatch for file_report in file_reports for mismatch in file_report["mismatches"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--reference-profile-root", required=True, type=Path)
    parser.add_argument("--candidate-profile-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--config-filename",
        action="append",
        dest="config_filenames",
        help="Golden config filename to include. Defaults to the full H800 golden set.",
    )
    parser.add_argument(
        "--ignore-candidate-extra-metadata-column",
        action="append",
        default=(),
        choices=sorted(_METADATA_COLUMNS_ALLOWED_FOR_EXPLICIT_MIGRATION),
        dest="ignore_candidate_extra_metadata_columns",
        help="Explicit one-off metadata migration column allowed only when present in candidate.",
    )
    args = parser.parse_args(argv)

    report = compare_profiling_csv_roots(
        config_root=args.config_root,
        reference_profile_root=args.reference_profile_root,
        candidate_profile_root=args.candidate_profile_root,
        config_filenames=tuple(args.config_filenames or GOLDEN_CONFIG_FILENAMES),
        ignore_candidate_extra_metadata_columns=tuple(
            args.ignore_candidate_extra_metadata_columns
        ),
    )
    write_json_report(args.output_json, report)
    return PASS_EXIT_CODE if report["status"] == PASS_STATUS else MISMATCH_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
