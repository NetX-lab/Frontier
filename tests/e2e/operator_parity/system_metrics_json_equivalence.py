#!/usr/bin/env python3
"""Strict system_metrics.json equivalence checks for operator-parity simulations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.e2e.attention_equivalence.measurement_csv_equivalence import (
    FAIL_STATUS,
    MISMATCH_EXIT_CODE,
    PASS_EXIT_CODE,
    PASS_STATUS,
)
from tests.e2e.attention_equivalence.profile_manifest import write_json_report

SYSTEM_METRICS_JSON = "system_metrics.json"
ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE_PCT = 1e-7


def _case_paths(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"system metrics root missing: {root}")
    paths = sorted(root.rglob(SYSTEM_METRICS_JSON))
    return {str(path.parent.relative_to(root)): path for path in paths}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required system_metrics.json missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _relative_delta_pct(reference_value: float, candidate_value: float) -> float:
    if math.isnan(reference_value) or math.isnan(candidate_value):
        return math.nan
    if math.isinf(reference_value) or math.isinf(candidate_value):
        return 0.0 if reference_value == candidate_value else math.inf
    if reference_value == 0.0:
        return 0.0 if candidate_value == 0.0 else math.inf
    return abs(candidate_value - reference_value) / abs(reference_value) * 100.0


def _path_join(parent: str, child: str) -> str:
    if parent == "":
        return child
    if child.startswith("["):
        return f"{parent}{child}"
    return f"{parent}.{child}"


def _compare_values(
    reference_value: Any,
    candidate_value: Any,
    *,
    path: str,
    mismatches: list[dict[str, Any]],
    numeric_comparisons: list[dict[str, Any]],
) -> None:
    if isinstance(reference_value, dict) and isinstance(candidate_value, dict):
        reference_keys = set(reference_value)
        candidate_keys = set(candidate_value)
        missing_keys = sorted(reference_keys - candidate_keys)
        extra_keys = sorted(candidate_keys - reference_keys)
        if missing_keys or extra_keys:
            mismatches.append(
                {
                    "kind": "object_key_mismatch",
                    "path": path or "$",
                    "missing_in_candidate": missing_keys,
                    "extra_in_candidate": extra_keys,
                }
            )
            return
        for key in sorted(reference_keys):
            _compare_values(
                reference_value[key],
                candidate_value[key],
                path=_path_join(path, str(key)),
                mismatches=mismatches,
                numeric_comparisons=numeric_comparisons,
            )
        return

    if isinstance(reference_value, list) and isinstance(candidate_value, list):
        if len(reference_value) != len(candidate_value):
            mismatches.append(
                {
                    "kind": "array_length_mismatch",
                    "path": path or "$",
                    "reference_length": len(reference_value),
                    "candidate_length": len(candidate_value),
                }
            )
            return
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference_value, candidate_value, strict=True)
        ):
            _compare_values(
                reference_item,
                candidate_item,
                path=_path_join(path, f"[{index}]"),
                mismatches=mismatches,
                numeric_comparisons=numeric_comparisons,
            )
        return

    if _is_number(reference_value) and _is_number(candidate_value):
        reference_number = float(reference_value)
        candidate_number = float(candidate_value)
        if math.isnan(reference_number) or math.isnan(candidate_number):
            passed = math.isnan(reference_number) and math.isnan(candidate_number)
            absolute_delta = 0.0 if passed else math.nan
            relative_delta_pct = 0.0 if passed else math.nan
        elif math.isinf(reference_number) or math.isinf(candidate_number):
            passed = reference_number == candidate_number
            absolute_delta = 0.0 if passed else math.inf
            relative_delta_pct = 0.0 if passed else math.inf
        else:
            absolute_delta = abs(candidate_number - reference_number)
            relative_delta_pct = _relative_delta_pct(reference_number, candidate_number)
            passed = (
                absolute_delta <= ABSOLUTE_TOLERANCE
                or relative_delta_pct <= RELATIVE_TOLERANCE_PCT
            )
        comparison = {
            "path": path or "$",
            "reference_value": reference_number,
            "candidate_value": candidate_number,
            "absolute_delta": absolute_delta,
            "relative_delta_pct": relative_delta_pct,
            "tolerance": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative_pct": RELATIVE_TOLERANCE_PCT,
            },
            "passed": passed,
        }
        numeric_comparisons.append(comparison)
        if not passed:
            mismatches.append({"kind": "numeric_mismatch", **comparison})
        return

    if _is_number(reference_value) != _is_number(candidate_value):
        mismatches.append(
            {
                "kind": "type_mismatch",
                "path": path or "$",
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "reference_type": type(reference_value).__name__,
                "candidate_type": type(candidate_value).__name__,
            }
        )
        return

    if type(reference_value) is not type(candidate_value) or reference_value != candidate_value:
        mismatches.append(
            {
                "kind": "value_mismatch",
                "path": path or "$",
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "reference_type": type(reference_value).__name__,
                "candidate_type": type(candidate_value).__name__,
            }
        )


def _max_finite(values: Sequence[float]) -> float:
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return 0.0
    return max(finite_values)


def compare_system_metrics_files(
    reference_json: str | Path,
    candidate_json: str | Path,
    *,
    case: str | None = None,
) -> dict[str, Any]:
    """Compare one pair of system_metrics.json files."""

    reference_path = Path(reference_json)
    candidate_path = Path(candidate_json)
    mismatches: list[dict[str, Any]] = []
    numeric_comparisons: list[dict[str, Any]] = []
    _compare_values(
        _read_json(reference_path),
        _read_json(candidate_path),
        path="",
        mismatches=mismatches,
        numeric_comparisons=numeric_comparisons,
    )
    return {
        "status": PASS_STATUS if not mismatches else FAIL_STATUS,
        "case": case,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "numeric_fields_compared": len(numeric_comparisons),
        "mismatch_count": len(mismatches),
        "max_abs_delta": _max_finite(
            [float(comparison["absolute_delta"]) for comparison in numeric_comparisons]
        ),
        "max_relative_delta_pct": _max_finite(
            [float(comparison["relative_delta_pct"]) for comparison in numeric_comparisons]
        ),
        "numeric_comparisons": numeric_comparisons,
        "mismatches": mismatches,
    }


def compare_system_metrics_roots(reference_root: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    """Compare every system_metrics.json case under two metrics roots."""

    reference_root_path = Path(reference_root)
    candidate_root_path = Path(candidate_root)
    reference_cases = _case_paths(reference_root_path)
    candidate_cases = _case_paths(candidate_root_path)
    reference_case_names = set(reference_cases)
    candidate_case_names = set(candidate_cases)
    if reference_case_names != candidate_case_names:
        raise ValueError(
            "system_metrics case set mismatch: "
            f"missing_in_candidate={sorted(reference_case_names - candidate_case_names)}, "
            f"extra_in_candidate={sorted(candidate_case_names - reference_case_names)}"
        )

    case_reports = [
        compare_system_metrics_files(
            reference_cases[case],
            candidate_cases[case],
            case=case,
        )
        for case in sorted(reference_cases)
    ]
    mismatch_count = sum(int(case_report["mismatch_count"]) for case_report in case_reports)
    numeric_fields_compared = sum(
        int(case_report["numeric_fields_compared"]) for case_report in case_reports
    )
    return {
        "status": PASS_STATUS if mismatch_count == 0 else FAIL_STATUS,
        "reference_root": str(reference_root_path),
        "candidate_root": str(candidate_root_path),
        "case_count": len(case_reports),
        "cases": [case_report["case"] for case_report in case_reports],
        "numeric_fields_compared": numeric_fields_compared,
        "mismatch_count": mismatch_count,
        "max_abs_delta": _max_finite(
            [float(case_report["max_abs_delta"]) for case_report in case_reports]
        ),
        "max_relative_delta_pct": _max_finite(
            [float(case_report["max_relative_delta_pct"]) for case_report in case_reports]
        ),
        "case_reports": case_reports,
        "mismatches": [
            {"case": case_report["case"], **mismatch}
            for case_report in case_reports
            for mismatch in case_report["mismatches"]
        ],
        "tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative_pct": RELATIVE_TOLERANCE_PCT,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    report = compare_system_metrics_roots(args.reference_root, args.candidate_root)
    write_json_report(args.output_json, report)
    return PASS_EXIT_CODE if report["status"] == PASS_STATUS else MISMATCH_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
