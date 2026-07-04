#!/usr/bin/env python3
"""Build a safe overlay/supplement root for true-mixed attention profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from tests.e2e.operator_parity.audit_true_mixed_attention_stage import (
    DEFAULT_MODELS,
    audit_stage,
    _truthy_csv_value,
)


SOURCE_TO_CANONICAL_FILENAME: dict[str, str] = {
    "attention_true_mixed.csv": "attention.csv",
    "attention_true_mixed_kernel_only.csv": "attention_kernel_only.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_source_rows(path: Path) -> tuple[int, int]:
    rows = _read_rows(path)
    true_mixed_rows = [
        row for row in rows if _truthy_csv_value(row.get("is_true_mixed_batch", ""))
    ]
    if not rows:
        raise ValueError(f"true-mixed supplement source is empty: {path}")
    if len(true_mixed_rows) != len(rows):
        raise ValueError(
            "true-mixed supplement source must contain only true-mixed rows: "
            f"{path}, rows={len(rows)}, true_mixed_rows={len(true_mixed_rows)}"
        )
    return len(rows), len(true_mixed_rows)


def build_overlay(
    *,
    canonical_root: Path,
    stage_root: Path,
    overlay_root: Path,
    supplement_root: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    expected_true_mixed_rows_per_file: int | None = None,
    expected_tp_values: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    if overlay_root.exists():
        raise FileExistsError(f"overlay root already exists: {overlay_root}")
    if supplement_root.exists():
        raise FileExistsError(f"supplement root already exists: {supplement_root}")
    if not canonical_root.is_dir():
        raise FileNotFoundError(f"canonical root missing: {canonical_root}")
    if not stage_root.is_dir():
        raise FileNotFoundError(f"stage root missing: {stage_root}")

    audit_summary = audit_stage(
        stage_root=stage_root,
        models=models,
        expected_true_mixed_rows_per_file=expected_true_mixed_rows_per_file,
        expected_tp_values=expected_tp_values,
    )
    if audit_summary["status"] != "PASS":
        raise ValueError(
            "stage audit failed; refusing to build true-mixed overlay: "
            f"fail_count={audit_summary['fail_count']}"
        )

    shutil.copytree(canonical_root, overlay_root)

    reports: list[dict[str, Any]] = []
    for model in models:
        for source_filename, target_filename in SOURCE_TO_CANONICAL_FILENAME.items():
            source = stage_root / model / source_filename
            target = supplement_root / model / target_filename
            if not source.is_file():
                raise FileNotFoundError(source)
            row_count, true_mixed_row_count = _validate_source_rows(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            reports.append(
                {
                    "model": model,
                    "source": str(source),
                    "target": str(target),
                    "source_filename": source_filename,
                    "target_filename": target_filename,
                    "row_count": row_count,
                    "true_mixed_row_count": true_mixed_row_count,
                    "source_sha256": _sha256(source),
                    "target_sha256": _sha256(target),
                    "target_size_bytes": target.stat().st_size,
                }
            )

    return {
        "status": "PASS",
        "canonical_root": str(canonical_root),
        "stage_root": str(stage_root),
        "overlay_root": str(overlay_root),
        "supplement_root": str(supplement_root),
        "model_count": len(models),
        "mapped_file_count": len(reports),
        "total_supplement_rows": sum(
            int(report["row_count"]) for report in reports
        ),
        "stage_audit": audit_summary,
        "reports": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy canonical profiles into an overlay root and map staged "
            "attention_true_mixed*.csv files into a same-filename supplement "
            "root for merge_profile_csv_contexts.py."
        )
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help=(
            "Staged device profiling root, e.g. "
            "<stage>/profiling/compute/h800."
        ),
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
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
    summary = build_overlay(
        canonical_root=args.canonical_root,
        stage_root=args.stage_root,
        overlay_root=args.overlay_root,
        supplement_root=args.supplement_root,
        models=tuple(args.models),
        expected_true_mixed_rows_per_file=args.expected_true_mixed_rows_per_file,
        expected_tp_values=tuple(args.expected_tp_values),
    )
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
