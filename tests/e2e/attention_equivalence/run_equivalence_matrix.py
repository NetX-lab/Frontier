#!/usr/bin/env python3
"""Run declared attention equivalence matrix cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from tests.e2e.attention_equivalence.measurement_csv_equivalence import (
    compare_measurement_csv,
)
from tests.e2e.attention_equivalence.profile_manifest import write_json_report
from tests.e2e.attention_equivalence.simulation_output_equivalence import (
    compare_simulation_outputs,
)

STATUS_PASS = "PASS"
STATUS_CANDIDATE_MISMATCH = "CANDIDATE_MISMATCH"
STATUS_REFERENCE_BASELINE_FAILS = "REFERENCE_BASELINE_FAILS"
STATUS_CANDIDATE_ARTIFACT_FAILS = "CANDIDATE_ARTIFACT_FAILS"
STATUS_STRUCTURAL_FAIL = "STRUCTURAL_FAIL"

EXIT_CODES = {
    STATUS_PASS: 0,
    STATUS_CANDIDATE_MISMATCH: 1,
    STATUS_STRUCTURAL_FAIL: 2,
    STATUS_REFERENCE_BASELINE_FAILS: 3,
    STATUS_CANDIDATE_ARTIFACT_FAILS: 4,
}

STATUS_SEVERITY = {
    STATUS_PASS: 0,
    STATUS_CANDIDATE_MISMATCH: 1,
    STATUS_REFERENCE_BASELINE_FAILS: 2,
    STATUS_CANDIDATE_ARTIFACT_FAILS: 2,
    STATUS_STRUCTURAL_FAIL: 3,
}


def _require_matching_input_manifests(case: Mapping[str, Any]) -> dict[str, Any]:
    reference_manifest = case.get("reference_input_manifest")
    candidate_manifest = case.get("candidate_input_manifest")
    if not isinstance(reference_manifest, dict) or not isinstance(candidate_manifest, dict):
        raise ValueError(
            "reference_input_manifest and candidate_input_manifest must both be provided"
        )
    if reference_manifest != candidate_manifest:
        raise ValueError(
            "manifest mismatch: "
            f"reference_input_manifest={reference_manifest}, "
            f"candidate_input_manifest={candidate_manifest}"
        )
    return dict(reference_manifest)


def _declared_artifact_paths(case: Mapping[str, Any], side: str) -> list[Path]:
    paths: list[Path] = []
    measurement_key = f"{side}_measurement_csv"
    if case.get(measurement_key):
        paths.append(Path(str(case[measurement_key])))

    dir_key = f"{side}_dir"
    if case.get(dir_key):
        root = Path(str(case[dir_key]))
        for artifact_name in case.get("simulation_artifacts", ("request_metrics.csv",)):
            paths.append(root / str(artifact_name))
    return paths


def _missing_path_from_error(error: Exception) -> Path | None:
    message = str(error)
    for prefix in ("required CSV missing: ", "required file missing: "):
        if message.startswith(prefix):
            return Path(message.removeprefix(prefix))
    return None


def _same_declared_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)


def _is_reference_missing(error: Exception, case: Mapping[str, Any]) -> bool:
    missing_path = _missing_path_from_error(error)
    if missing_path is None:
        return False
    return any(_same_declared_path(missing_path, path) for path in _declared_artifact_paths(case, "reference"))


def _is_candidate_missing(error: Exception, case: Mapping[str, Any]) -> bool:
    missing_path = _missing_path_from_error(error)
    if missing_path is None:
        return False
    return any(_same_declared_path(missing_path, path) for path in _declared_artifact_paths(case, "candidate"))


def run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Run one matrix case and return a non-pass status for any problem."""

    case_name = str(case.get("name") or "unnamed-case")
    try:
        case_manifest = _require_matching_input_manifests(case)
    except Exception as exc:
        return {
            "name": case_name,
            "status": STATUS_STRUCTURAL_FAIL,
            "exit_code": EXIT_CODES[STATUS_STRUCTURAL_FAIL],
            "error": str(exc),
        }

    reports: dict[str, Any] = {}
    try:
        if case.get("reference_measurement_csv") or case.get("candidate_measurement_csv"):
            if not (case.get("reference_measurement_csv") and case.get("candidate_measurement_csv")):
                raise ValueError(
                    "measurement comparison requires both reference_measurement_csv and candidate_measurement_csv"
                )
            reports["measurement"] = compare_measurement_csv(
                case["reference_measurement_csv"],
                case["candidate_measurement_csv"],
                key_columns=tuple(case.get("measurement_key_columns", ())),
                tolerance_allowlist=case.get("measurement_tolerance_allowlist"),
                case_manifest=case_manifest,
            )
        if case.get("reference_dir") or case.get("candidate_dir"):
            if not (case.get("reference_dir") and case.get("candidate_dir")):
                raise ValueError(
                    "simulation comparison requires both reference_dir and candidate_dir"
                )
            reports["simulation"] = compare_simulation_outputs(
                case["reference_dir"],
                case["candidate_dir"],
                artifact_names=tuple(case.get("simulation_artifacts", ("request_metrics.csv",))),
                key_columns_by_artifact=case.get("simulation_key_columns", {}),
                tolerance_allowlist_by_artifact=case.get("simulation_tolerance_allowlist", {}),
                case_manifest=case_manifest,
            )
        if not reports:
            raise ValueError("case declares no measurement or simulation comparison")
    except FileNotFoundError as exc:
        if _is_reference_missing(exc, case):
            status = STATUS_REFERENCE_BASELINE_FAILS
        elif _is_candidate_missing(exc, case):
            status = STATUS_CANDIDATE_ARTIFACT_FAILS
        else:
            status = STATUS_STRUCTURAL_FAIL
        return {
            "name": case_name,
            "status": status,
            "exit_code": EXIT_CODES[status],
            "error": str(exc),
        }
    except ValueError as exc:
        return {
            "name": case_name,
            "status": STATUS_STRUCTURAL_FAIL,
            "exit_code": EXIT_CODES[STATUS_STRUCTURAL_FAIL],
            "error": str(exc),
        }

    mismatch_count = sum(int(report.get("mismatch_count", 0)) for report in reports.values())
    status = STATUS_PASS if mismatch_count == 0 else STATUS_CANDIDATE_MISMATCH
    return {
        "name": case_name,
        "status": status,
        "exit_code": EXIT_CODES[status],
        "mismatch_count": mismatch_count,
        "case_manifest": case_manifest,
        "reports": reports,
    }


def run_matrix(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "status": STATUS_STRUCTURAL_FAIL,
            "exit_code": EXIT_CODES[STATUS_STRUCTURAL_FAIL],
            "case_count": 0,
            "pass_count": 0,
            "error": "equivalence matrix must contain at least one case",
            "results": [],
        }

    results = [run_case(case) for case in cases]
    status = max(
        (str(result["status"]) for result in results),
        key=lambda result_status: STATUS_SEVERITY[result_status],
    )
    return {
        "status": status,
        "exit_code": EXIT_CODES[status],
        "case_count": len(results),
        "pass_count": sum(1 for result in results if result["status"] == STATUS_PASS),
        "results": results,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("matrix YAML must contain a top-level cases list")
    return list(payload["cases"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-cases", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    report = run_matrix(_load_cases(args.matrix_cases))
    write_json_report(args.output_json, report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
