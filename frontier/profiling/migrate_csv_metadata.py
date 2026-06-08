"""Utility to migrate legacy profiling CSV metadata columns."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from frontier.config.precision_type import PrecisionType
from frontier.types import MeasurementType

REQUIRED_METADATA_COLUMNS = (
    "profiling_precision",
    "model_arch",
    "quant_signature",
    "measurement_type",
)


def _normalize_precision(precision: str) -> str:
    return PrecisionType.from_string(precision).name


def _normalize_measurement_type(measurement_type: str) -> str:
    return MeasurementType.from_string(measurement_type).value


def _compute_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_no_conflict(
    df: pd.DataFrame,
    column: str,
    expected_value: str,
    overwrite: bool,
) -> None:
    if column not in df.columns or overwrite:
        return

    existing_values = df[column].dropna().astype(str).str.strip()
    if existing_values.empty:
        return

    unique_values = sorted(set(existing_values.tolist()))
    if len(unique_values) == 1 and unique_values[0] == expected_value:
        return

    raise ValueError(
        f"Cannot migrate column '{column}' without --overwrite. "
        f"Existing values={unique_values}, expected={expected_value!r}."
    )


def migrate_csv_metadata(
    input_csv: str,
    output_csv: str | None,
    profiling_precision: str,
    model_arch: str,
    quant_signature: str,
    measurement_type: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add required metadata columns to a profiling CSV file.

    The utility is intentionally strict:
    - missing input file => fail-fast
    - empty CSV => fail-fast
    - conflicting existing metadata => fail-fast unless --overwrite
    - measurement_type must be provided explicitly
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    normalized_precision = _normalize_precision(profiling_precision)
    normalized_model_arch = str(model_arch).strip()
    normalized_quant_signature = str(quant_signature).strip()
    normalized_measurement_type = _normalize_measurement_type(measurement_type)

    if not normalized_model_arch:
        raise ValueError("model_arch must be non-empty.")
    if not normalized_quant_signature:
        raise ValueError("quant_signature must be non-empty.")

    metadata_values: Dict[str, str] = {
        "profiling_precision": normalized_precision,
        "model_arch": normalized_model_arch,
        "quant_signature": normalized_quant_signature,
        "measurement_type": normalized_measurement_type,
    }

    for column in REQUIRED_METADATA_COLUMNS:
        _ensure_no_conflict(
            df=df,
            column=column,
            expected_value=metadata_values[column],
            overwrite=overwrite,
        )
        df[column] = metadata_values[column]

    output_path = output_csv if output_csv else input_csv
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _iter_csv_files(input_dir: str | Path) -> Iterable[Path]:
    for path in sorted(Path(input_dir).rglob("*.csv")):
        if path.is_file():
            yield path


def migrate_csv_metadata_directory(
    input_dir: str,
    output_dir: str | None,
    profiling_precision: str,
    model_arch: str,
    quant_signature: str,
    measurement_type: str,
    overwrite: bool = False,
) -> List[Dict[str, object]]:
    """Migrate an entire directory tree of CSV files and emit manifest records."""
    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    csv_files = list(_iter_csv_files(input_root))
    if not csv_files:
        raise ValueError(f"No CSV files found under directory: {input_dir}")

    records: List[Dict[str, object]] = []
    for csv_path in csv_files:
        relative_path = csv_path.relative_to(input_root)
        if output_dir is None:
            target_path = csv_path
        else:
            target_path = Path(output_dir) / relative_path

        input_checksum = _compute_checksum(csv_path)
        input_columns = list(pd.read_csv(csv_path, nrows=0).columns)
        migrated = migrate_csv_metadata(
            input_csv=str(csv_path),
            output_csv=str(target_path),
            profiling_precision=profiling_precision,
            model_arch=model_arch,
            quant_signature=quant_signature,
            measurement_type=measurement_type,
            overwrite=overwrite,
        )
        output_checksum = _compute_checksum(target_path)
        added_columns = [column for column in migrated.columns if column not in input_columns]
        records.append(
            {
                "input_csv": str(csv_path),
                "output_csv": str(target_path),
                "row_count": int(len(migrated)),
                "added_columns": ",".join(added_columns),
                "input_checksum": input_checksum,
                "output_checksum": output_checksum,
            }
        )

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy profiling CSVs by adding required metadata columns."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_csv",
        type=str,
        help="Input CSV path to migrate.",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing CSV files to migrate recursively.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. If omitted, input CSV will be updated in-place.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for directory migration. If omitted, files are updated in-place.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional manifest CSV path for directory migration.",
    )
    parser.add_argument(
        "--profiling_precision",
        type=str,
        required=True,
        choices=[p.name for p in PrecisionType],
        help="Profiling precision to write into metadata columns.",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        required=True,
        help="Model architecture tag (for example: generic, step2_mini).",
    )
    parser.add_argument(
        "--quant_signature",
        type=str,
        default="none",
        help="Quantization signature (default: none).",
    )
    parser.add_argument(
        "--measurement_type",
        type=str,
        required=True,
        choices=[measurement.value for measurement in MeasurementType],
        help="Measurement type to write into metadata columns.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata columns even if values conflict.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input_csv is not None:
        output_path = args.output_csv if args.output_csv else args.input_csv
        migrate_csv_metadata(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            profiling_precision=args.profiling_precision,
            model_arch=args.model_arch,
            quant_signature=args.quant_signature,
            measurement_type=args.measurement_type,
            overwrite=args.overwrite,
        )
        print(
            "CSV metadata migration completed. "
            f"input={args.input_csv}, output={output_path}, "
            f"profiling_precision={args.profiling_precision}, "
            f"model_arch={args.model_arch}, quant_signature={args.quant_signature}, "
            f"measurement_type={args.measurement_type}"
        )
        return

    records = migrate_csv_metadata_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        profiling_precision=args.profiling_precision,
        model_arch=args.model_arch,
        quant_signature=args.quant_signature,
        measurement_type=args.measurement_type,
        overwrite=args.overwrite,
    )
    manifest_path = args.manifest_path
    if manifest_path is not None:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(
        "CSV metadata directory migration completed. "
        f"input_dir={args.input_dir}, output_dir={args.output_dir or args.input_dir}, "
        f"files={len(records)}, measurement_type={args.measurement_type}, "
        f"manifest_path={manifest_path or 'NONE'}"
    )


if __name__ == "__main__":
    main()
