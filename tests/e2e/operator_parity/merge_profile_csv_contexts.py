"""Deterministically merge staged profiling CSV rows into canonical CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing profiling CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f"Profiling CSV has no header: {path}")
        return fieldnames, [dict(row) for row in reader]


def _merge_fieldnames(*fieldname_sets: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for fieldnames in fieldname_sets:
        for fieldname in fieldnames:
            if fieldname not in merged:
                merged.append(fieldname)
    return merged


def _normalized_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    return {fieldname: row.get(fieldname, "") for fieldname in fieldnames}


def _row_key(row: dict[str, str], key_columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in key_columns)


def _row_identity(row: dict[str, str], fieldnames: list[str]) -> tuple[str, ...]:
    return tuple(row.get(fieldname, "") for fieldname in fieldnames)


def _normalize_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[dict[str, str]]:
    return [_normalized_row(row, fieldnames) for row in rows]


def _group_rows_by_key(
    rows: list[dict[str, str]],
    key_columns: list[str],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    rows_by_key: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        rows_by_key.setdefault(_row_key(row, key_columns), []).append(row)
    return rows_by_key


def merge_profile_csvs(
    *,
    canonical_csv: Path,
    supplement_csv: Path,
    output_csv: Path,
) -> dict[str, object]:
    """Merge by all non-time_stats columns and fail on conflicting duplicates."""

    base_fieldnames, base_rows = _read_csv(canonical_csv)
    supplement_fieldnames, supplement_rows = _read_csv(supplement_csv)
    fieldnames = _merge_fieldnames(base_fieldnames, supplement_fieldnames)
    key_columns = [
        fieldname for fieldname in fieldnames if not fieldname.startswith("time_stats.")
    ]
    if not key_columns:
        raise ValueError(
            "Cannot merge profiling CSVs without at least one non-time_stats key column: "
            f"{canonical_csv}, {supplement_csv}"
        )

    normalized_base_rows = _normalize_rows(base_rows, fieldnames)
    normalized_supplement_rows = _normalize_rows(supplement_rows, fieldnames)
    accepted_rows_by_key = _group_rows_by_key(normalized_base_rows, key_columns)
    base_row_identities = {
        _row_identity(row, fieldnames) for row in normalized_base_rows
    }
    accepted_row_identities = set(base_row_identities)

    supplement_rows_to_append: list[dict[str, str]] = []
    duplicate_identical_count = 0
    supplement_duplicate_identical_count = 0
    for supplement_row in normalized_supplement_rows:
        row_identity = _row_identity(supplement_row, fieldnames)
        if row_identity in accepted_row_identities:
            duplicate_identical_count += 1
            if row_identity not in base_row_identities:
                supplement_duplicate_identical_count += 1
            continue
        key = _row_key(supplement_row, key_columns)
        matching_accepted_rows = accepted_rows_by_key.get(key, [])
        if not matching_accepted_rows:
            supplement_rows_to_append.append(supplement_row)
            accepted_rows_by_key.setdefault(key, []).append(supplement_row)
            accepted_row_identities.add(row_identity)
            continue
        raise ValueError(
            "Conflicting duplicate profiling row for key "
            f"{dict(zip(key_columns, key, strict=True))} from supplement: "
            f"{canonical_csv} vs {supplement_csv}"
        )

    merged_rows = sorted(
        [*normalized_base_rows, *supplement_rows_to_append],
        key=lambda row: _row_key(row, key_columns),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    return {
        "canonical_csv": str(canonical_csv),
        "supplement_csv": str(supplement_csv),
        "output_csv": str(output_csv),
        "base_row_count": len(base_rows),
        "supplement_row_count": len(supplement_rows),
        "merged_row_count": len(merged_rows),
        "key_column_count": len(key_columns),
        "key_columns": key_columns,
        "duplicate_identical_count": duplicate_identical_count,
        "supplement_duplicate_identical_count": supplement_duplicate_identical_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge staged profiling CSV rows into canonical CSVs using all "
            "non-time_stats columns as the deterministic row key."
        )
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Canonical device profiling root, e.g. data/profiling/compute/h800.",
    )
    parser.add_argument(
        "--supplement-root",
        type=Path,
        required=True,
        help="Staged device profiling root, e.g. <stage>/compute/h800.",
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=("moe.csv", "moe_kernel_only.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output device profiling root. By default the CLI refuses in-place "
            "writes; pass this for a safe non-mutating merge."
        ),
    )
    parser.add_argument(
        "--allow-in-place",
        action="store_true",
        help="Explicitly allow writing merged CSVs back into --canonical-root.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_root is None and not args.allow_in_place:
        raise ValueError(
            "Refusing in-place merge without --allow-in-place. "
            "Pass --output-root for a safe non-mutating merge."
        )
    if args.output_root is not None and args.allow_in_place:
        raise ValueError("Use either --output-root or --allow-in-place, not both.")

    reports: list[dict[str, object]] = []
    for model in args.models:
        for filename in args.filenames:
            canonical_csv = args.canonical_root / model / filename
            supplement_csv = args.supplement_root / model / filename
            output_csv = (
                args.output_root / model / filename
                if args.output_root is not None
                else canonical_csv
            )
            if output_csv.resolve() == canonical_csv.resolve() and not args.allow_in_place:
                raise ValueError(
                    "Refusing in-place merge without --allow-in-place. "
                    "Pass --output-root for a safe non-mutating merge."
                )
            reports.append(
                merge_profile_csvs(
                    canonical_csv=canonical_csv,
                    supplement_csv=supplement_csv,
                    output_csv=output_csv,
                )
            )

    output = json.dumps({"merged_files": reports}, indent=2, sort_keys=True)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
