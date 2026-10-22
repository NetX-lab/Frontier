"""Read-only evidence audit for the requested merged Step fidelity run."""

import hashlib
import json
import subprocess
from pathlib import Path

from tests.e2e.refactor_fidelity.cases import build_cases
from tests.e2e.refactor_fidelity.run_matrix import (
    missing_evidence_for,
    side_provenance_findings,
)


ROOT = Path("/data/ycfeng/tmp/step5_workflow_fidelity_20261022")
BASELINE = Path("/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate")
CANDIDATE = ROOT / "candidate"
REPO = Path("/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2")
REVISION = "fa5bae2980366da1d075125d6c26248a737fc419"
BASELINE_REVISION = "7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_side(directory, revision):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_id = {record["case_id"]: record for record in records}
    expected = {case.case_id for case in build_cases()}
    failures = [
        record["case_id"]
        for record in records
        if record["returncode"] != 0 or record["error"]
    ]
    evidence_errors = []
    verified = 0
    for record in records:
        problem = missing_evidence_for(record, directory)
        if problem:
            evidence_errors.append({"case_id": record["case_id"], "problem": problem})
        for artifact in record.get("artifacts", []):
            path = directory / record["artifact_dir"] / artifact["name"]
            if not path.is_file():
                evidence_errors.append({"path": str(path), "problem": "missing file"})
            elif path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
                evidence_errors.append({"path": str(path), "problem": "size/hash changed"})
            else:
                verified += 1
    definitions = [
        case.case_id
        for case in build_cases()
        if case.case_id not in by_id
        or case.definition_digest() != by_id[case.case_id].get("case_digest")
    ]
    record_provenance_errors = [
        record["case_id"]
        for record in records
        if record.get("source_revision") != revision
        or record.get("harness_revision") != revision
        or record.get("source_dirty") is not False
    ]
    cache_root = Path(manifest["cache_dir"])
    actual_cache = sorted(str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file())
    expected_cache = sorted(manifest["cache_files"])
    summary = {
        "label_path": str(directory),
        "source_revision": manifest["source_revision"],
        "harness_revision": manifest["harness_revision"],
        "source_dirty": manifest["source_dirty"],
        "manifest_dirty_paths": manifest["git_dirty_paths"],
        "case_records": len(records),
        "unique_case_records": len(by_id),
        "case_failures": failures,
        "missing_cases": sorted(expected - set(by_id)),
        "unexpected_cases": sorted(set(by_id) - expected),
        "definition_mismatches": definitions,
        "provenance_findings": side_provenance_findings(str(directory), by_id),
        "record_provenance_errors": record_provenance_errors,
        "artifacts_rehashed": verified,
        "evidence_errors": evidence_errors,
        "cache_filename_count": len(expected_cache),
        "cache_inventory_changed": actual_cache != expected_cache,
        "full_clean_capture": bool(
            manifest["cache_clean_before_run"]
            and manifest["case_filter"] is None
            and set(manifest["cases_executed_in_last_run"]) == expected
            and manifest["case_count"] == len(expected)
        ),
        "manifest_sha256": digest(directory / "manifest.json"),
        "results_sha256": digest(directory / "results.jsonl"),
    }
    return summary


def main():
    output = ROOT / "postrun_receipt.json"
    if output.exists():
        raise SystemExit(f"Refusing existing audit receipt: {output}")
    report = json.loads((ROOT / "comparison.json").read_text())
    baseline = load_side(BASELINE, BASELINE_REVISION)
    candidate = load_side(CANDIDATE, REVISION)
    tracked_status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip()
    current_revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    difference_fields = [
        "mismatched_cases",
        "baseline_failures",
        "candidate_only_failures",
        "cases_missing_from_one_side",
        "cases_with_missing_evidence",
        "cases_with_differing_definitions",
        "cases_not_compared",
        "cases_not_compared_without_explanation",
        "cases_absent_from_both_sides",
        "provenance_findings",
        "predictor_cache_files_only_in_baseline",
        "predictor_cache_files_only_in_candidate",
        "predictor_cache_findings",
    ]
    counts = {field: len(report[field]) for field in difference_fields}
    summaries_clean = all(
        side["case_records"] == side["unique_case_records"] == 77
        and not side["source_dirty"]
        and not side["manifest_dirty_paths"]
        and side["full_clean_capture"]
        and not side["cache_inventory_changed"]
        and all(
            not side[field]
            for field in [
                "case_failures", "missing_cases", "unexpected_cases", "definition_mismatches",
                "provenance_findings", "record_provenance_errors", "evidence_errors",
            ]
        )
        for side in (baseline, candidate)
    )
    passed = bool(
        summaries_clean
        and current_revision == REVISION
        and not tracked_status
        and report["baseline"]["git_head"] == BASELINE_REVISION
        and report["candidate"]["git_head"] == REVISION
        and report["cases_compared"] == report["case_table_size"] == len(report["identical_cases"]) == 77
        and report["complete_comparison"]
        and report["predictor_cache_compared"]
        and report["predictor_cache_populated_cleanly"]
        and not any(counts.values())
    )
    first_difference = None
    for field in difference_fields:
        if report[field]:
            first_difference = {"field": field, "first": report[field][0]}
            break
    receipt = {
        "task_id": "D26-D27-D28-MERGED-FIDELITY",
        "verdict": "PASS" if passed else "FAIL",
        "tested_revision": REVISION,
        "actual_revision_after_run": current_revision,
        "candidate_status_after_run": tracked_status,
        "baseline": baseline,
        "candidate": candidate,
        "comparison_counts": counts,
        "cases_compared": report["cases_compared"],
        "identical_cases": len(report["identical_cases"]),
        "complete_comparison": report["complete_comparison"],
        "cache_compared": report["predictor_cache_compared"],
        "cache_populated_cleanly": report["predictor_cache_populated_cleanly"],
        "comparison_sha256": digest(ROOT / "comparison.json"),
        "first_comparison_difference": first_difference,
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
