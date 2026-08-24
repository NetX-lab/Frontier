"""Deterministically merge staged profiling CSV rows into canonical CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


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
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
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


def _index_unique_rows(
    rows: list[dict[str, str]],
    key_columns: list[str],
    *,
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _row_key(row, key_columns)
        if key in rows_by_key:
            raise ValueError(
                f"Duplicate {label} profiling key: "
                f"{dict(zip(key_columns, key, strict=True))}"
            )
        rows_by_key[key] = row
    return rows_by_key


def _finite_profile_value(
    raw_value: str,
    *,
    column: str,
    key_columns: list[str],
    key: tuple[str, ...],
    source: str,
) -> float:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(
            f"Empty profiling column {column!r} in {source} for key "
            f"{dict(zip(key_columns, key, strict=True))}"
        )
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric profiling column {column!r} in {source} for key "
            f"{dict(zip(key_columns, key, strict=True))}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite profiling column {column!r} in {source} for key "
            f"{dict(zip(key_columns, key, strict=True))}: {value!r}"
        )
    return parsed


def enrich_profile_csv_columns(
    *,
    canonical_csv: Path,
    supplement_csv: Path,
    output_csv: Path,
    target_columns: Iterable[str],
    drop_canonical_key_values: Mapping[str, str] | None = None,
    supplement_key_values: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Copy selected timing columns onto exact canonical keys without replacing rows."""

    base_fieldnames, base_rows = _read_csv(canonical_csv)
    supplement_fieldnames, supplement_rows = _read_csv(supplement_csv)
    columns = list(target_columns)
    if not columns:
        raise ValueError("Column enrichment requires at least one target column.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Column enrichment target columns must be unique: {columns}")
    invalid_columns = [
        column for column in columns if not column.startswith("time_stats.")
    ]
    if invalid_columns:
        raise ValueError(
            "Column enrichment accepts only time_stats.* columns; "
            f"got {invalid_columns}"
        )
    missing_supplement_columns = [
        column for column in columns if column not in supplement_fieldnames
    ]
    if missing_supplement_columns:
        raise ValueError(
            f"{supplement_csv} missing enrichment columns: "
            f"{missing_supplement_columns}"
        )

    selected_supplement_key_values = {
        str(column): str(expected_value).strip()
        for column, expected_value in (supplement_key_values or {}).items()
    }
    for column in selected_supplement_key_values:
        if column.startswith("time_stats."):
            raise ValueError(
                f"Cannot filter supplement by timing column: {column!r}"
            )
        if column not in supplement_fieldnames:
            raise ValueError(
                f"Supplement profiling CSV missing supplement filter key: {column!r}"
            )
    selected_supplement_rows = [
        row
        for row in supplement_rows
        if all(
            str(row.get(column, "")).strip() == expected_value
            for column, expected_value in selected_supplement_key_values.items()
        )
    ]
    if selected_supplement_key_values and not selected_supplement_rows:
        raise ValueError(
            "Supplement key filters matched no supplement rows: "
            f"{selected_supplement_key_values}"
        )

    dropped_key_values = dict(drop_canonical_key_values or {})
    for column, expected_value in dropped_key_values.items():
        if column.startswith("time_stats."):
            raise ValueError(
                f"Cannot drop timing column as a canonical key: {column!r}"
            )
        if column not in base_fieldnames:
            raise ValueError(
                f"Canonical profiling CSV missing key selected for removal: {column!r}"
            )
        if column in supplement_fieldnames:
            raise ValueError(
                f"Canonical key selected for removal is still emitted by supplement: "
                f"{column!r}"
            )
        observed_values = sorted(
            {str(row.get(column, "")).strip() for row in base_rows}
        )
        if observed_values != [str(expected_value)]:
            raise ValueError(
                f"Cannot remove canonical key {column!r}: expected canonical key "
                f"value {expected_value!r}, observed {observed_values}"
            )

    if dropped_key_values:
        base_fieldnames = [
            fieldname
            for fieldname in base_fieldnames
            if fieldname not in dropped_key_values
        ]
        base_rows = [
            {
                fieldname: value
                for fieldname, value in row.items()
                if fieldname not in dropped_key_values
            }
            for row in base_rows
        ]

    key_columns = [
        fieldname
        for fieldname in base_fieldnames
        if not fieldname.startswith("time_stats.")
    ]
    if not key_columns:
        raise ValueError(
            f"Cannot enrich profiling CSV without non-time_stats keys: {canonical_csv}"
        )
    supplement_key_columns = [
        fieldname
        for fieldname in supplement_fieldnames
        if not fieldname.startswith("time_stats.")
    ]
    if set(supplement_key_columns) != set(key_columns):
        raise ValueError(
            "Canonical and supplement profiling key columns differ: "
            f"canonical={key_columns}, supplement={supplement_key_columns}"
        )

    output_fieldnames = _merge_fieldnames(base_fieldnames, columns)
    normalized_base_rows = _normalize_rows(base_rows, output_fieldnames)
    normalized_supplement_rows = _normalize_rows(
        selected_supplement_rows,
        supplement_fieldnames,
    )
    base_rows_by_key = _index_unique_rows(
        normalized_base_rows,
        key_columns,
        label="canonical",
    )
    supplement_rows_by_key = _index_unique_rows(
        normalized_supplement_rows,
        key_columns,
        label="supplement",
    )

    populated_cell_count = 0
    identical_existing_cell_count = 0
    for key, supplement_row in supplement_rows_by_key.items():
        canonical_row = base_rows_by_key.get(key)
        if canonical_row is None:
            raise ValueError(
                f"Supplement profiling key has no canonical row: "
                f"{dict(zip(key_columns, key, strict=True))}"
            )
        for column in columns:
            supplement_value = str(supplement_row.get(column, "")).strip()
            supplement_numeric = _finite_profile_value(
                supplement_value,
                column=column,
                key_columns=key_columns,
                key=key,
                source=str(supplement_csv),
            )
            canonical_value = str(canonical_row.get(column, "")).strip()
            if canonical_value:
                canonical_numeric = _finite_profile_value(
                    canonical_value,
                    column=column,
                    key_columns=key_columns,
                    key=key,
                    source=str(canonical_csv),
                )
                if canonical_numeric != supplement_numeric:
                    raise ValueError(
                        f"Conflicting profiling column {column!r} for key "
                        f"{dict(zip(key_columns, key, strict=True))}: "
                        f"canonical={canonical_value!r}, "
                        f"supplement={supplement_value!r}"
                    )
                identical_existing_cell_count += 1
                continue
            canonical_row[column] = supplement_value
            populated_cell_count += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(normalized_base_rows)

    return {
        "canonical_csv": str(canonical_csv),
        "supplement_csv": str(supplement_csv),
        "output_csv": str(output_csv),
        "base_row_count": len(base_rows),
        "supplement_row_count": len(supplement_rows),
        "selected_supplement_row_count": len(selected_supplement_rows),
        "excluded_supplement_row_count": (
            len(supplement_rows) - len(selected_supplement_rows)
        ),
        "supplement_key_values": selected_supplement_key_values,
        "enriched_row_count": len(supplement_rows_by_key),
        "target_columns": columns,
        "added_column_count": sum(
            column not in base_fieldnames for column in columns
        ),
        "populated_cell_count": populated_cell_count,
        "identical_existing_cell_count": identical_existing_cell_count,
        "dropped_canonical_key_values": dropped_key_values,
        "key_columns": key_columns,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge staged profiling rows or enrich selected timing columns "
            "using all non-time_stats columns as the deterministic row key."
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
    parser.add_argument(
        "--enrich-columns",
        nargs="+",
        default=None,
        help=(
            "Copy only these time_stats.* columns onto exact canonical keys. "
            "All existing canonical values and non-target timing columns are preserved."
        ),
    )
    parser.add_argument(
        "--drop-canonical-key",
        action="append",
        default=[],
        metavar="COLUMN=EXPECTED_VALUE",
        help=(
            "During --enrich-columns, explicitly remove a constant canonical-only "
            "key after validating its exact value. Repeat for multiple keys."
        ),
    )
    parser.add_argument(
        "--supplement-key",
        action="append",
        default=[],
        metavar="COLUMN=EXPECTED_VALUE",
        help=(
            "During --enrich-columns, select supplement rows by an exact "
            "non-time_stats key value. Repeat for multiple keys."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _parse_key_value_specs(
    specs: Iterable[str],
    *,
    option_name: str,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        column, separator, expected_value = str(spec).partition("=")
        if not separator or not column or column in parsed:
            raise ValueError(
                f"{option_name} must use unique COLUMN=EXPECTED_VALUE "
                f"specifications; got {spec!r}"
            )
        parsed[column] = expected_value
    return parsed


def main() -> int:
    args = _parse_args()
    if args.output_root is None and not args.allow_in_place:
        raise ValueError(
            "Refusing in-place merge without --allow-in-place. "
            "Pass --output-root for a safe non-mutating merge."
        )
    if args.output_root is not None and args.allow_in_place:
        raise ValueError("Use either --output-root or --allow-in-place, not both.")
    drop_canonical_key_values = _parse_key_value_specs(
        args.drop_canonical_key,
        option_name="--drop-canonical-key",
    )
    supplement_key_values = _parse_key_value_specs(
        args.supplement_key,
        option_name="--supplement-key",
    )
    if drop_canonical_key_values and args.enrich_columns is None:
        raise ValueError("--drop-canonical-key requires --enrich-columns.")
    if supplement_key_values and args.enrich_columns is None:
        raise ValueError("--supplement-key requires --enrich-columns.")

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
            if args.enrich_columns is None:
                reports.append(
                    merge_profile_csvs(
                        canonical_csv=canonical_csv,
                        supplement_csv=supplement_csv,
                        output_csv=output_csv,
                    )
                )
            else:
                reports.append(
                    enrich_profile_csv_columns(
                        canonical_csv=canonical_csv,
                        supplement_csv=supplement_csv,
                        output_csv=output_csv,
                        target_columns=args.enrich_columns,
                        drop_canonical_key_values=drop_canonical_key_values,
                        supplement_key_values=supplement_key_values,
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
