"""Deterministically merge staged profiling CSV rows into canonical CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from frontier.profiling.attention.provenance import (
    publish_attention_merge,
    validate_attention_run_sidecar,
)


_ATTENTION_ALIASES = {
    "attention.csv": "attention_combined.csv",
    "attention_kernel_only.csv": "attention_combined_kernel_only.csv",
}
_ATTENTION_MEASUREMENT_TYPES = {
    "attention.csv": "CUDA_EVENT",
    "attention_kernel_only.csv": "KERNEL_ONLY",
}
_ATTENTION_OUTPUT_ONLY_FILENAMES = frozenset(_ATTENTION_ALIASES.values())
_ATTENTION_PARTITION_FILENAMES = frozenset(
    {
        "attention_mixed.csv",
        "attention_mixed_kernel_only.csv",
        "attention_true_mixed.csv",
        "attention_true_mixed_kernel_only.csv",
    }
)
_ATTENTION_BOUND_PATH_FIELDS = (
    "artifact_csv",
    "source_run_csv",
    "source_run_sidecar",
)


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


def _validate_csv_filename(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or Path(filename).suffix != ".csv"
    ):
        raise ValueError(
            "--filenames entries must be plain CSV basenames without directories: "
            f"{filename!r}."
        )
    if filename in _ATTENTION_OUTPUT_ONLY_FILENAMES:
        raise ValueError(
            "Attention compatibility aliases are output-only and cannot be merged "
            f"as inputs: {filename!r}."
        )
    if filename in _ATTENTION_PARTITION_FILENAMES:
        raise ValueError(
            "Attention partition CSVs cannot use the generic merge surface. "
            "Map the partition into attention.csv or attention_kernel_only.csv "
            f"and publish it with run sidecars: {filename!r}."
        )
    return filename


def _load_attention_source_payload(
    *,
    csv_path: Path,
    sidecar_path: Path,
    expected_model: str,
    expected_measurement_type: str,
    label: str,
) -> dict[str, Any]:
    validate_attention_run_sidecar(
        csv_path=csv_path,
        sidecar_path=sidecar_path,
    )
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid attention {label} run sidecar: {sidecar_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Attention {label} run sidecar must contain a JSON object: "
            f"{sidecar_path}."
        )
    if payload.get("model") != expected_model:
        raise ValueError(
            f"Attention {label} source model identity mismatch: "
            f"expected={expected_model!r}, actual={payload.get('model')!r}."
        )
    if payload.get("measurement_type") != expected_measurement_type:
        raise ValueError(
            f"Attention {label} source measurement family mismatch: "
            f"expected={expected_measurement_type!r}, "
            f"actual={payload.get('measurement_type')!r}."
        )
    if payload.get("is_native_profile_allocation", True) is not True:
        raise ValueError(
            f"Attention {label} source requires native allocation provenance."
        )
    return dict(payload)


def _attention_bound_artifact_paths(payload: Mapping[str, Any]) -> set[Path]:
    paths: set[Path] = set()
    for field in _ATTENTION_BOUND_PATH_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Attention provenance field {field!r} must be a non-empty path."
            )
        paths.add(Path(value))
    return paths


def _validate_json_out_path(
    *,
    json_out: Path | None,
    artifact_paths: Iterable[Path],
    directory_paths: Iterable[Path] = (),
) -> None:
    if json_out is None:
        return
    resolved_json_out = json_out.resolve()
    if json_out.exists() and json_out.is_dir():
        raise ValueError(
            f"--json-out must be a file path, not a directory: {json_out}."
        )
    artifact_collisions = {
        path.resolve()
        for path in artifact_paths
        if path is not None
    }
    reserved_directories = {
        path.resolve()
        for path in directory_paths
        if path is not None
    }
    if (
        any(
            resolved_json_out == artifact
            or resolved_json_out in artifact.parents
            or artifact in resolved_json_out.parents
            for artifact in artifact_collisions
        )
        or any(
            resolved_json_out == directory
            or resolved_json_out in directory.parents
            or directory in resolved_json_out.parents
            for directory in reserved_directories
        )
    ):
        raise ValueError(
            "--json-out must not overwrite a merge source, source sidecar, "
            f"publication artifact, or publication directory: {json_out}."
        )


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
    parser.add_argument(
        "--canonical-sidecar",
        type=Path,
        default=None,
        help=(
            "Run sidecar for the canonical attention source. Attention publication "
            "accepts exactly one model/file per invocation."
        ),
    )
    parser.add_argument(
        "--supplement-sidecar",
        type=Path,
        default=None,
        help=(
            "Run sidecar for the supplement attention source. Attention publication "
            "accepts exactly one model/file per invocation."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    filenames = tuple(_validate_csv_filename(name) for name in args.filenames)
    if args.output_root is None and not args.allow_in_place:
        raise ValueError(
            "Refusing in-place merge without --allow-in-place. "
            "Pass --output-root for a safe non-mutating merge."
        )
    if args.output_root is not None and args.allow_in_place:
        raise ValueError("Use either --output-root or --allow-in-place, not both.")
    attention_filenames = [
        filename for filename in filenames if filename in _ATTENTION_ALIASES
    ]
    has_any_sidecar = (
        args.canonical_sidecar is not None or args.supplement_sidecar is not None
    )
    if attention_filenames and not has_any_sidecar:
        raise ValueError(
            "Attention merge publication requires --canonical-sidecar and "
            "--supplement-sidecar."
        )
    if has_any_sidecar:
        if args.canonical_sidecar is None or args.supplement_sidecar is None:
            raise ValueError(
                "Use --canonical-sidecar and --supplement-sidecar together."
            )
        if (
            len(args.models) != 1
            or len(filenames) != 1
            or filenames[0] not in _ATTENTION_ALIASES
        ):
            raise ValueError(
                "Attention sidecar publication accepts exactly one model and one "
                "attention filename per invocation."
            )
        if args.output_root is None or args.allow_in_place:
            raise ValueError(
                "Attention sidecar publication requires --output-root and does not "
                "allow in-place source replacement."
            )

    plans: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    publication_directories: list[Path] = []
    if args.output_root is not None:
        publication_directories.append(args.output_root)
    for model in args.models:
        for filename in filenames:
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
            plan: dict[str, Any] = {
                "model": model,
                "filename": filename,
                "canonical_csv": canonical_csv,
                "supplement_csv": supplement_csv,
                "output_csv": output_csv,
            }
            artifact_paths.extend((canonical_csv, supplement_csv, output_csv))
            publication_directories.append(output_csv.parent)
            if filename in _ATTENTION_ALIASES:
                expected_measurement_type = _ATTENTION_MEASUREMENT_TYPES[filename]
                alias_csv = output_csv.with_name(_ATTENTION_ALIASES[filename])
                sidecar_path = output_csv.with_name(
                    f"{Path(filename).stem}.merge_provenance.json"
                )
                canonical_sidecar = Path(args.canonical_sidecar)
                supplement_sidecar = Path(args.supplement_sidecar)
                canonical_payload = _load_attention_source_payload(
                    csv_path=canonical_csv,
                    sidecar_path=canonical_sidecar,
                    expected_model=model,
                    expected_measurement_type=expected_measurement_type,
                    label="canonical",
                )
                supplement_payload = _load_attention_source_payload(
                    csv_path=supplement_csv,
                    sidecar_path=supplement_sidecar,
                    expected_model=model,
                    expected_measurement_type=expected_measurement_type,
                    label="supplement",
                )
                plan.update(
                    {
                        "alias_csv": alias_csv,
                        "sidecar_path": sidecar_path,
                        "canonical_sidecar": canonical_sidecar,
                        "supplement_sidecar": supplement_sidecar,
                    }
                )
                artifact_paths.extend(
                    (
                        canonical_sidecar,
                        supplement_sidecar,
                        alias_csv,
                        sidecar_path,
                        *_attention_bound_artifact_paths(canonical_payload),
                        *_attention_bound_artifact_paths(supplement_payload),
                    )
                )
            plans.append(plan)

    _validate_json_out_path(
        json_out=args.json_out,
        artifact_paths=artifact_paths,
        directory_paths=publication_directories,
    )

    reports: list[dict[str, object]] = []
    for plan in plans:
        filename = str(plan["filename"])
        canonical_csv = Path(plan["canonical_csv"])
        supplement_csv = Path(plan["supplement_csv"])
        output_csv = Path(plan["output_csv"])
        if filename in _ATTENTION_ALIASES:
            alias_csv = Path(plan["alias_csv"])
            sidecar_path = Path(plan["sidecar_path"])
            published = publish_attention_merge(
                output_csv=output_csv,
                alias_csv=alias_csv,
                sidecar_path=sidecar_path,
                sources=[
                    {
                        "label": "base",
                        "csv_path": canonical_csv,
                        "sidecar_path": plan["canonical_sidecar"],
                    },
                    {
                        "label": "supplement",
                        "csv_path": supplement_csv,
                        "sidecar_path": plan["supplement_sidecar"],
                    },
                ],
            )
            report = dict(published["report"])
            report.update(
                {
                    "canonical_csv": str(canonical_csv),
                    "supplement_csv": str(supplement_csv),
                    "output_csv": str(published["canonical"]),
                    "alias_csv": str(published["alias"]),
                    "sidecar_path": str(published["sidecar"]),
                }
            )
            reports.append(report)
        else:
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
