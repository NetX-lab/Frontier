#!/usr/bin/env python3
"""Driver for the refactor fidelity matrix.

Capture one side, then the other, then compare:

    PYTHONPATH=$PWD python tests/e2e/refactor_fidelity/run_matrix.py run \\
        --repo-root /path/to/baseline/worktree --label baseline \\
        --output-root "$FRONTIER_TMP_ROOT/refactor-fidelity" --clean-cache
    PYTHONPATH=$PWD python tests/e2e/refactor_fidelity/run_matrix.py run \\
        --repo-root "$PWD" --label candidate \\
        --output-root "$FRONTIER_TMP_ROOT/refactor-fidelity" --clean-cache
    PYTHONPATH=$PWD python tests/e2e/refactor_fidelity/run_matrix.py compare \\
        --output-root "$FRONTIER_TMP_ROOT/refactor-fidelity"

The driver always comes from the checkout it is invoked in, while ``--repo-root``
selects the simulator under test, so both sides are measured by one case table
and one comparator.

Each case runs a checked-in example wrapper with the working directory set to
the repository under test, which is what makes the relative profiling-data and
``cache`` paths in the shipped configuration resolve.  Each side therefore keeps
its own predictor cache inside its own checkout.  The names of the cache files a
side produces are recorded and compared as well, because a changed training
identity or cache key would otherwise be invisible: retraining from the same CSV
yields the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tests.e2e.refactor_fidelity.cases import FidelityCase, build_cases, group_counts
from tests.e2e.refactor_fidelity.compare import (
    MAX_REPORTED_CACHE_FINDINGS_PER_KIND,
    classify_cache_differences,
    compare_artifact_directories,
    file_digest,
    list_artifacts,
    substitutions_for,
)


DEFAULT_TIMEOUT_SECONDS = 1800


@dataclass
class CaseOutcome:
    case: FidelityCase
    returncode: int
    duration_seconds: float
    artifact_dir: str | None
    artifacts: list[dict]
    error: str | None
    log_tail: str | None = None

    def as_record(self) -> dict:
        record = self.case.as_record()
        record.update({
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "artifact_dir": self.artifact_dir,
            "artifacts": self.artifacts,
            "error": self.error,
            "log_tail": self.log_tail,
        })
        return record


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


def _package_versions(python_bin: str) -> dict[str, str]:
    script = (
        "import json,platform;"
        "mods=['numpy','pandas','sklearn','scipy','plotly','fasteners','ddsketch'];"
        "out={'python':platform.python_version()};"
        "\nfor m in mods:\n"
        "    try:\n"
        "        out[m]=__import__(m).__version__\n"
        "    except Exception as e:\n"
        "        out[m]='unavailable: %s' % type(e).__name__\n"
        "print(json.dumps(out))"
    )
    result = subprocess.run([python_bin, "-c", script], capture_output=True, text=True, check=False)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": result.stderr.strip()[:400]}


ARTIFACT_DISCOVERY_RETRY_SECONDS = 10.0


def _find_artifact_dir(metrics_root: Path, wait_seconds: float = 0.0) -> Path | None:
    """Locate the single normalized metrics directory a run produced.

    A successful run that appears to have written nothing is retried for a
    bounded window: the directory listing can lag the child process on a
    networked filesystem, and reporting a spurious failure would be worse than
    waiting.  A genuinely empty run still fails, only later.
    """

    deadline = time.monotonic() + wait_seconds
    while True:
        candidates = sorted(
            path.parent for path in metrics_root.rglob("system_metrics.json")
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"expected one metrics directory under {metrics_root}, "
                f"found {len(candidates)}: "
                + ", ".join(str(path) for path in candidates)
            )
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)


def _log_tail(log_path: Path, max_lines: int = 20) -> str:
    """Return the last lines of a run log, for a failure record."""

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return f"<log unreadable: {error}>"
    return "\n".join(lines[-max_lines:])


def _run_case(
    case: FidelityCase,
    repo_root: Path,
    label_root: Path,
    python_bin: str,
    timeout_seconds: int,
) -> CaseOutcome:
    case_root = label_root / "cases" / case.case_id
    metrics_root = case_root / "metrics"
    if case_root.exists():
        shutil.rmtree(case_root)
    metrics_root.mkdir(parents=True)

    script_path = repo_root / case.script
    if not script_path.is_file():
        return CaseOutcome(case, 127, 0.0, None, [], f"missing script: {script_path}")

    env = os.environ.copy()
    env.pop("PYTHONSTARTUP", None)
    env.update({
        "PYTHONPATH": str(repo_root),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_BIN": python_bin,
        "WANDB_DISABLED": "true",
        "VIDUR_DISABLE_WANDB": "1",
        "METRICS_OUTPUT_DIR": str(metrics_root),
        "RUN_ID": case.case_id,
    })
    env.update(case.env)

    command = ["bash", str(script_path)]
    if case.extra_args:
        command.append("--")
        command.extend(case.extra_args)

    log_path = case_root / "run.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {' '.join(command)}\n")
        log.write(f"# cwd: {repo_root}\n")
        log.write("# env overrides: " + json.dumps(dict(case.env), sort_keys=True) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command, cwd=str(repo_root), env=env,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout_seconds, check=False,
            )
            returncode = completed.returncode
            error = None
        except subprocess.TimeoutExpired:
            returncode = 124
            error = f"timed out after {timeout_seconds}s"
    duration = time.monotonic() - started

    artifact_dir: Path | None = None
    artifacts: list[dict] = []
    try:
        artifact_dir = _find_artifact_dir(
            metrics_root,
            wait_seconds=ARTIFACT_DISCOVERY_RETRY_SECONDS if returncode == 0 else 0.0,
        )
    except RuntimeError as failure:
        error = str(failure)
    if artifact_dir is not None:
        for name in list_artifacts(artifact_dir):
            path = artifact_dir / name
            artifacts.append({
                "name": name,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            })
    elif returncode == 0 and error is None:
        error = "run reported success but produced no system_metrics.json"

    return CaseOutcome(
        case=case,
        returncode=returncode,
        duration_seconds=duration,
        artifact_dir=(
            str(artifact_dir.relative_to(label_root)) if artifact_dir is not None else None
        ),
        artifacts=artifacts,
        error=error,
        # Keep the evidence with the record: a later re-run of the same case
        # overwrites run.log, which would otherwise erase why it failed.
        log_tail=_log_tail(log_path) if (returncode != 0 or error) else None,
    )


def _select_cases(
    cases: Sequence[FidelityCase], case_filter: str | None, start: int, limit: int | None
) -> list[FidelityCase]:
    selected = [case for case in cases if not case_filter or case_filter in case.case_id]
    selected = selected[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def source_provenance(repo_root: Path) -> dict:
    """Identify the source a case was measured against.

    Every case record carries this, not just the label-wide manifest, because
    a filtered run rewrites the manifest with its own revision while keeping
    the records of the cases it did not execute.  Without a per-case stamp
    there is nothing left to show that those retained cases ran on the same
    source.
    """

    return {
        "source_revision": _git(repo_root, "rev-parse", "HEAD"),
        "source_dirty": bool(_git(repo_root, "status", "--porcelain")),
        "harness_revision": _git(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"),
    }


def check_retained_records(
    results_path: Path, provenance: dict, executed_ids: set[str], known_ids: set[str]
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Decide which previously recorded cases this run may keep.

    Returns the retained records, the case ids whose provenance conflicts with
    this run, and the case ids that are no longer in the case table.  A
    conflicting record is never silently dropped or silently kept: the caller
    refuses the run so that one label always describes one source.
    """

    retained: dict[str, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                retained[record["case_id"]] = record

    conflicts = [
        case_id
        for case_id, record in sorted(retained.items())
        if case_id not in executed_ids
        and any(record.get(key) != value for key, value in provenance.items())
    ]
    stale = sorted(case_id for case_id in retained if case_id not in known_ids)
    return retained, conflicts, stale


def run_label(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    label_root = output_root / args.label
    label_root.mkdir(parents=True, exist_ok=True)

    cases = _select_cases(build_cases(), args.case_filter, args.start, args.limit)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    provenance = source_provenance(repo_root)
    results_path = label_root / "results.jsonl"
    all_cases = build_cases()
    retained, conflicts, stale = check_retained_records(
        results_path,
        provenance,
        {case.case_id for case in cases},
        {case.case_id for case in all_cases},
    )
    # Refuse before running anything rather than after, so a rejected label
    # costs a second instead of a full matrix.
    if conflicts:
        print(
            f"refusing to add to label {args.label!r}: {len(conflicts)} retained "
            "case records describe a different source, a different harness, or "
            "carry no provenance at all, and this run would leave the label "
            "describing a mixture.",
            file=sys.stderr,
        )
        for case_id in conflicts[:10]:
            recorded = {key: retained[case_id].get(key) for key in provenance}
            print(f"  {case_id}: recorded {recorded}", file=sys.stderr)
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more", file=sys.stderr)
        print(f"  this run: {provenance}", file=sys.stderr)
        print("  write to a new label, or re-run the whole matrix.", file=sys.stderr)
        return 2
    if stale:
        print(
            f"dropping {len(stale)} retained record(s) for case ids that are no "
            f"longer in the case table: {', '.join(stale)}"
        )

    cache_dir = repo_root / "cache"
    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"label={args.label} repo_root={repo_root} cases={len(cases)}")
    print(f"source {provenance['source_revision'][:12]} "
          f"dirty={provenance['source_dirty']} "
          f"harness {provenance['harness_revision'][:12]}")
    print(f"groups: {json.dumps(group_counts(cases), sort_keys=True)}")

    serial_cases = [case for case in cases if case.uses_trained_predictor]
    parallel_cases = [case for case in cases if not case.uses_trained_predictor]

    outcomes: dict[str, CaseOutcome] = {}

    def record(outcome: CaseOutcome) -> None:
        outcomes[outcome.case.case_id] = outcome
        status = "ok" if outcome.returncode == 0 and outcome.error is None else "FAIL"
        detail = f" ({outcome.error})" if outcome.error else ""
        print(
            f"  [{status:>4}] {outcome.case.case_id} rc={outcome.returncode} "
            f"{outcome.duration_seconds:.1f}s artifacts={len(outcome.artifacts)}{detail}",
            flush=True,
        )

    for case in serial_cases:
        record(_run_case(case, repo_root, label_root, args.python_bin, args.timeout_seconds))

    if parallel_cases:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [
                pool.submit(_run_case, case, repo_root, label_root, args.python_bin, args.timeout_seconds)
                for case in parallel_cases
            ]
            for future in futures:
                record(future.result())

    ordered = [outcomes[case.case_id] for case in cases]
    # A filtered or partial run refreshes only the cases it executed and keeps
    # the records of the cases it skipped, so the two sides stay comparable.
    # The guard above has already established that the retained records
    # describe this same source and harness.
    merged = dict(retained)
    for outcome in ordered:
        merged[outcome.case.case_id] = {**outcome.as_record(), **provenance}
    case_order = [case.case_id for case in all_cases]
    written = [case_id for case_id in case_order if case_id in merged]
    with results_path.open("w", encoding="utf-8") as handle:
        for case_id in written:
            handle.write(json.dumps(merged[case_id], sort_keys=True) + "\n")

    cache_files = sorted(
        str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*") if path.is_file()
    ) if cache_dir.is_dir() else []

    manifest = {
        "label": args.label,
        "repo_root": str(repo_root),
        "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "git_describe_branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_paths": _git(repo_root, "status", "--short"),
        "python_bin": args.python_bin,
        "package_versions": _package_versions(args.python_bin),
        # What the results file holds, not what the merge dictionary held: a
        # retained record for a case id that has since left the table is
        # dropped, and counting it here is what made an earlier label report
        # 72 cases over 71 result lines.
        "case_count": len(written),
        "cases_executed_in_last_run": [case.case_id for case in cases],
        "cases_dropped_as_stale": stale,
        **provenance,
        "case_filter": args.case_filter,
        "cache_dir": str(cache_dir),
        "cache_clean_before_run": bool(args.clean_cache),
        "cache_files": cache_files,
        "driver_checkout": str(Path(__file__).resolve().parents[3]),
    }
    (label_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failures = [o for o in ordered if o.returncode != 0 or o.error is not None]
    print(f"\n{len(ordered) - len(failures)}/{len(ordered)} cases produced artifacts")
    if failures:
        print("cases that did not produce artifacts:")
        for outcome in failures:
            print(f"  {outcome.case.case_id}: rc={outcome.returncode} {outcome.error or ''}")
    print(f"results: {results_path}")
    return 0 if not failures or args.continue_on_failure else 1


def _load_side(output_root: Path, label: str) -> tuple[dict, dict[str, dict]]:
    label_root = output_root / label
    manifest = json.loads((label_root / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    for line in (label_root / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            results[record["case_id"]] = record
    return manifest, results


PROVENANCE_KEYS = ("source_revision", "source_dirty", "harness_revision")


def side_provenance_findings(label: str, results: dict[str, dict]) -> list[str]:
    """Report anything that stops this label from describing one measurement."""

    findings: list[str] = []
    without = sorted(
        case_id for case_id, record in results.items()
        if any(key not in record for key in PROVENANCE_KEYS)
    )
    if without:
        findings.append(
            f"{label}: {len(without)} case record(s) carry no source provenance, "
            f"so the label cannot be shown to describe one revision "
            f"(first: {', '.join(without[:3])}). Recapture this side."
        )
    stamped = {
        tuple(record[key] for key in PROVENANCE_KEYS)
        for record in results.values()
        if all(key in record for key in PROVENANCE_KEYS)
    }
    if len(stamped) > 1:
        findings.append(
            f"{label}: case records describe {len(stamped)} different "
            f"source/harness combinations: {sorted(stamped)}"
        )
    for revision, dirty, _harness in sorted(stamped):
        if dirty:
            findings.append(
                f"{label}: measured against a modified working tree at "
                f"{revision[:12]}, which is not a commit-specific result"
            )
    return findings


def missing_evidence_for(record: dict, label_root: Path) -> str | None:
    """Return why a successful case's recorded artifacts cannot be compared.

    A record that says a case succeeded is not evidence on its own.  The files
    it named have to still be on disk, because the comparison reads the
    directory and an absent directory otherwise reads as an empty one.
    """

    if not record.get("artifact_dir"):
        return "the record names no artifact directory"
    recorded = {entry["name"] for entry in record.get("artifacts", [])}
    if not recorded:
        return "the record lists no artifacts"
    directory = label_root / record["artifact_dir"]
    if not directory.is_dir():
        return f"the recorded artifact directory is gone: {directory}"
    on_disk = set(list_artifacts(directory))
    if on_disk != recorded:
        lost = sorted(recorded - on_disk)
        extra = sorted(on_disk - recorded)
        return (
            f"the directory no longer matches the record "
            f"({len(lost)} missing, {len(extra)} unexpected)"
            + (f"; missing {lost[:3]}" if lost else "")
        )
    return None


def compare_labels(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    baseline_manifest, baseline_results = _load_side(output_root, args.baseline_label)
    candidate_manifest, candidate_results = _load_side(output_root, args.candidate_label)

    baseline_root = output_root / args.baseline_label
    candidate_root = output_root / args.candidate_label
    baseline_subs = substitutions_for(
        Path(baseline_manifest["repo_root"]), output_root, args.baseline_label
    )
    candidate_subs = substitutions_for(
        Path(candidate_manifest["repo_root"]), output_root, args.candidate_label
    )

    identical: list[str] = []
    mismatched: list[dict] = []
    baseline_failures: list[dict] = []
    candidate_only_failures: list[dict] = []
    missing: list[str] = []
    missing_evidence: list[dict] = []
    definition_mismatches: list[dict] = []

    for case_id in sorted(set(baseline_results) | set(candidate_results)):
        baseline_record = baseline_results.get(case_id)
        candidate_record = candidate_results.get(case_id)
        if baseline_record is None or candidate_record is None:
            missing.append(case_id)
            continue

        # The two sides are joined by case id, so the id has to have meant the
        # same run on both of them.  Without this an edited case definition
        # compares two different experiments under one name.
        baseline_digest = baseline_record.get("case_digest")
        candidate_digest = candidate_record.get("case_digest")
        if baseline_digest != candidate_digest:
            definition_mismatches.append({
                "case_id": case_id,
                "baseline_case_digest": baseline_digest,
                "candidate_case_digest": candidate_digest,
            })
            continue

        baseline_ok = baseline_record["returncode"] == 0 and not baseline_record["error"]
        candidate_ok = candidate_record["returncode"] == 0 and not candidate_record["error"]
        if not baseline_ok:
            baseline_failures.append({
                "case_id": case_id,
                "baseline_returncode": baseline_record["returncode"],
                "baseline_error": baseline_record["error"],
                "candidate_returncode": candidate_record["returncode"],
                "candidate_error": candidate_record["error"],
                "candidate_also_failed": not candidate_ok,
            })
            continue
        if not candidate_ok:
            candidate_only_failures.append({
                "case_id": case_id,
                "candidate_returncode": candidate_record["returncode"],
                "candidate_error": candidate_record["error"],
                "candidate_log_tail": candidate_record.get("log_tail"),
            })
            continue

        evidence_problems = {
            side: reason
            for side, reason in (
                ("baseline", missing_evidence_for(baseline_record, baseline_root)),
                ("candidate", missing_evidence_for(candidate_record, candidate_root)),
            )
            if reason is not None
        }
        if evidence_problems:
            missing_evidence.append({"case_id": case_id, **evidence_problems})
            continue

        differences = compare_artifact_directories(
            baseline_root / baseline_record["artifact_dir"],
            candidate_root / candidate_record["artifact_dir"],
            baseline_subs,
            candidate_subs,
        )
        if differences:
            mismatched.append({
                "case_id": case_id,
                "differences": [difference.as_record() for difference in differences],
            })
        else:
            identical.append(case_id)

    # The predictor cache is populated by the cases that actually ran, so the
    # cache comparison only means something when both sides ran the same full
    # matrix. A filtered or partial run would otherwise report every model the
    # other side trained as a difference.
    full_case_set = {case.case_id for case in build_cases()}
    # A cache listing is only a fair comparison when each side's cache was
    # populated by one clean run of the whole table.  A label assembled from a
    # filtered continuation without --clean-cache carries models trained by an
    # earlier case selection, which is not what the other side has.  The
    # filter alone does not settle it: --start and --limit also narrow the
    # executed selection and leave no filter in the manifest, so the executed
    # case ids are checked by name.  A manifest without that list cannot show
    # a full run and is not compared.
    def populated_by_one_clean_full_run(manifest: dict) -> bool:
        executed = manifest.get("cases_executed_in_last_run")
        return bool(
            manifest.get("cache_clean_before_run")
            and not manifest.get("case_filter")
            and executed is not None
            and set(executed) == full_case_set
        )

    cache_populated_cleanly = all(
        populated_by_one_clean_full_run(manifest)
        for manifest in (baseline_manifest, candidate_manifest)
    )
    cache_comparable = (
        set(baseline_results) == full_case_set
        and set(candidate_results) == full_case_set
        and cache_populated_cleanly
    )
    baseline_cache = baseline_manifest.get("cache_files", [])
    candidate_cache = candidate_manifest.get("cache_files", [])
    if cache_comparable:
        cache_only_in_baseline = sorted(set(baseline_cache) - set(candidate_cache))
        cache_only_in_candidate = sorted(set(candidate_cache) - set(baseline_cache))
        cache_findings = classify_cache_differences(baseline_cache, candidate_cache)
    else:
        cache_only_in_baseline = []
        cache_only_in_candidate = []
        cache_findings = []

    # Completeness is about what was compared, not about which case ids appear
    # in the two result tables.  A table can be full of records that all
    # describe failed runs, and comparing none of them is not agreement.
    compared_ids = set(identical) | {entry["case_id"] for entry in mismatched}
    compared = len(compared_ids)
    not_compared = sorted(full_case_set - compared_ids)
    complete = not not_compared
    provenance_findings = (
        side_provenance_findings(args.baseline_label, baseline_results)
        + side_provenance_findings(args.candidate_label, candidate_results)
    )
    # Every case that was not compared must be accounted for by one of the
    # specific findings above.  Anything left over means a path through this
    # function dropped a case silently, and the gate must not pass on it.
    explained = (
        set(missing)
        | {entry["case_id"] for entry in baseline_failures}
        | {entry["case_id"] for entry in candidate_only_failures}
        | {entry["case_id"] for entry in missing_evidence}
        | {entry["case_id"] for entry in definition_mismatches}
    )
    absent_from_both = full_case_set - set(baseline_results) - set(candidate_results)
    unexplained = sorted(set(not_compared) - explained - absent_from_both)

    report = {
        "baseline": {
            "label": args.baseline_label,
            "repo_root": baseline_manifest["repo_root"],
            "git_head": baseline_manifest["git_head"],
        },
        "candidate": {
            "label": args.candidate_label,
            "repo_root": candidate_manifest["repo_root"],
            "git_head": candidate_manifest["git_head"],
        },
        "identical_cases": identical,
        "mismatched_cases": mismatched,
        "baseline_failures": baseline_failures,
        "candidate_only_failures": candidate_only_failures,
        "cases_missing_from_one_side": missing,
        "cases_with_missing_evidence": missing_evidence,
        "cases_with_differing_definitions": definition_mismatches,
        "cases_not_compared": not_compared,
        "cases_not_compared_without_explanation": unexplained,
        "cases_absent_from_both_sides": sorted(absent_from_both),
        "provenance_findings": provenance_findings,
        "predictor_cache_populated_cleanly": cache_populated_cleanly,
        "predictor_cache_files_only_in_baseline": cache_only_in_baseline,
        "predictor_cache_files_only_in_candidate": cache_only_in_candidate,
        "predictor_cache_findings": [f.as_record() for f in cache_findings],
        "predictor_cache_compared": cache_comparable,
        "cases_compared": compared,
        "case_table_size": len(full_case_set),
        "complete_comparison": complete,
    }
    report_path = output_root / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"baseline {args.baseline_label} @ {baseline_manifest['git_head'][:12]}")
    print(f"candidate {args.candidate_label} @ {candidate_manifest['git_head'][:12]}")
    print(f"cases compared: {compared} of {len(full_case_set)} in the case table")
    print(f"identical: {len(identical)}")
    print(f"mismatched: {len(mismatched)}")
    print(f"baseline failures: {len(baseline_failures)}")
    print(f"candidate-only failures: {len(candidate_only_failures)}")
    print(f"cases missing from one side: {len(missing)}")
    print(f"cases with missing evidence: {len(missing_evidence)}")
    print(f"cases with differing definitions: {len(definition_mismatches)}")
    print(f"cases not compared: {len(not_compared)}")
    if not cache_comparable:
        print(
            "predictor cache file names: not compared, because at least one "
            "side's cache was not populated by one clean run of the complete "
            "case set"
        )
    else:
        print(
            "predictor cache file names differ: "
            f"{len(cache_only_in_baseline)} baseline-only, "
            f"{len(cache_only_in_candidate)} candidate-only"
        )
    if cache_findings:
        rekeyed = [f for f in cache_findings if f.kind == "rekeyed"]
        if rekeyed:
            print(
                f"  {len(rekeyed)} artifacts kept their model name but changed hash, "
                "which means a training identity or cache key changed:"
            )
            for finding in rekeyed[:MAX_REPORTED_CACHE_FINDINGS_PER_KIND]:
                print(f"    {finding.stem}: {finding.detail}")
            if len(rekeyed) > MAX_REPORTED_CACHE_FINDINGS_PER_KIND:
                print(f"    ... and {len(rekeyed) - MAX_REPORTED_CACHE_FINDINGS_PER_KIND} more")
        for kind, label in (("only_in_baseline", "baseline"),
                            ("only_in_candidate", "candidate")):
            entries = [f for f in cache_findings if f.kind == kind]
            if entries:
                print(f"  {len(entries)} artifacts exist only on the {label} side:")
                for finding in entries[:MAX_REPORTED_CACHE_FINDINGS_PER_KIND]:
                    print(f"    {finding.stem}: {finding.detail}")
                if len(entries) > MAX_REPORTED_CACHE_FINDINGS_PER_KIND:
                    print(
                        f"    ... and {len(entries) - MAX_REPORTED_CACHE_FINDINGS_PER_KIND} more"
                    )
    for entry in mismatched:
        print(f"\nMISMATCH {entry['case_id']}")
        for difference in entry["differences"]:
            print(f"  {difference['artifact']} [{difference['kind']}] {difference['detail']}")
    for entry in candidate_only_failures:
        print(f"\nCANDIDATE-ONLY FAILURE {entry['case_id']}: rc={entry['candidate_returncode']} "
              f"{entry['candidate_error'] or ''}")
        if entry.get("candidate_log_tail"):
            print("  last log lines:")
            for line in entry["candidate_log_tail"].splitlines():
                print(f"    {line}")
    for entry in baseline_failures:
        print(f"\nBASELINE FAILURE {entry['case_id']}: rc={entry['baseline_returncode']} "
              f"{entry['baseline_error'] or ''}"
              + (" (the candidate failed too)" if entry["candidate_also_failed"] else ""))
    for entry in missing_evidence:
        sides = ", ".join(f"{side}: {reason}" for side, reason in entry.items()
                          if side != "case_id")
        print(f"\nMISSING EVIDENCE {entry['case_id']}: {sides}")
    for entry in definition_mismatches:
        print(f"\nDIFFERENT DEFINITION {entry['case_id']}: "
              f"baseline {entry['baseline_case_digest']} != "
              f"candidate {entry['candidate_case_digest']}")
    for finding in provenance_findings:
        print(f"\nPROVENANCE {finding}")
    if unexplained:
        print(f"\nUNEXPLAINED: {len(unexplained)} case(s) were neither compared "
              f"nor reported as a specific finding: {', '.join(unexplained[:10])}")
    print(f"\nreport: {report_path}")

    # A comparison passes only when it actually compared the whole case table.
    # Two sides that both ran nothing agree trivially, and two sides whose runs
    # all failed have no artifacts to disagree about; neither is evidence about
    # the branch. Every reason a case was not compared is therefore a failure
    # in its own right, and --allow-partial waives exactly one of them: cases
    # nobody attempted on either side.
    incomplete = bool(absent_from_both) and not args.allow_partial
    if incomplete:
        print(
            "\nINCOMPLETE: this comparison did not cover the whole case table, "
            "so it is not evidence that the branch is unchanged. "
            "Re-run both sides without a filter, or pass --allow-partial to "
            "accept a deliberate subset."
        )
    failed = bool(
        mismatched
        or baseline_failures
        or candidate_only_failures
        or missing
        or missing_evidence
        or definition_mismatches
        or unexplained
        or provenance_findings
        or cache_only_in_baseline
        or cache_only_in_candidate
        or incomplete
    )
    return 1 if failed else 0


def list_cases(_: argparse.Namespace) -> int:
    cases = build_cases()
    print(f"{len(cases)} cases")
    for group, count in sorted(group_counts(cases).items()):
        print(f"  {group}: {count}")
    print()
    for case in cases:
        marker = "T" if case.uses_trained_predictor else " "
        print(f"{marker} {case.case_id:<44} {case.script}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="capture one side of the matrix")
    run_parser.add_argument("--repo-root", required=True, help="checkout of the simulator under test")
    run_parser.add_argument("--label", required=True, help="name of this side, e.g. baseline or candidate")
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--python-bin", default=sys.executable)
    run_parser.add_argument("--jobs", type=int, default=4)
    run_parser.add_argument("--case-filter", default=None, help="substring of case_id")
    run_parser.add_argument("--start", type=int, default=0)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    run_parser.add_argument("--clean-cache", action="store_true",
                            help="remove <repo-root>/cache before running")
    run_parser.add_argument("--continue-on-failure", action="store_true",
                            help="exit 0 even when some cases produced no artifacts")
    run_parser.set_defaults(func=run_label)

    compare_parser = subparsers.add_parser("compare", help="compare two captured sides")
    compare_parser.add_argument("--output-root", required=True)
    compare_parser.add_argument("--baseline-label", default="baseline")
    compare_parser.add_argument("--candidate-label", default="candidate")
    compare_parser.add_argument(
        "--allow-partial", action="store_true",
        help="accept a comparison that does not cover the whole case table",
    )
    compare_parser.set_defaults(func=compare_labels)

    list_parser = subparsers.add_parser("list", help="print the case table")
    list_parser.set_defaults(func=list_cases)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
