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


def run_label(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    label_root = output_root / args.label
    label_root.mkdir(parents=True, exist_ok=True)

    cases = _select_cases(build_cases(), args.case_filter, args.start, args.limit)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    cache_dir = repo_root / "cache"
    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"label={args.label} repo_root={repo_root} cases={len(cases)}")
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
    results_path = label_root / "results.jsonl"
    # A filtered or partial run refreshes only the cases it executed and keeps
    # the records of the cases it skipped, so the two sides stay comparable.
    merged: dict[str, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                merged[record["case_id"]] = record
    for outcome in ordered:
        merged[outcome.case.case_id] = outcome.as_record()
    case_order = [case.case_id for case in build_cases()]
    with results_path.open("w", encoding="utf-8") as handle:
        for case_id in case_order:
            if case_id in merged:
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
        "case_count": len(merged),
        "cases_executed_in_last_run": [case.case_id for case in cases],
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

    for case_id in sorted(set(baseline_results) | set(candidate_results)):
        baseline_record = baseline_results.get(case_id)
        candidate_record = candidate_results.get(case_id)
        if baseline_record is None or candidate_record is None:
            missing.append(case_id)
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

    baseline_cache = baseline_manifest.get("cache_files", [])
    candidate_cache = candidate_manifest.get("cache_files", [])
    cache_only_in_baseline = sorted(set(baseline_cache) - set(candidate_cache))
    cache_only_in_candidate = sorted(set(candidate_cache) - set(baseline_cache))

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
        "predictor_cache_files_only_in_baseline": cache_only_in_baseline,
        "predictor_cache_files_only_in_candidate": cache_only_in_candidate,
    }
    report_path = output_root / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"baseline {args.baseline_label} @ {baseline_manifest['git_head'][:12]}")
    print(f"candidate {args.candidate_label} @ {candidate_manifest['git_head'][:12]}")
    print(f"identical: {len(identical)}")
    print(f"mismatched: {len(mismatched)}")
    print(f"baseline failures (excluded): {len(baseline_failures)}")
    print(f"candidate-only failures: {len(candidate_only_failures)}")
    print(f"cases missing from one side: {len(missing)}")
    print(
        "predictor cache file names differ: "
        f"{len(cache_only_in_baseline)} baseline-only, {len(cache_only_in_candidate)} candidate-only"
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
    print(f"\nreport: {report_path}")

    failed = bool(
        mismatched
        or candidate_only_failures
        or missing
        or cache_only_in_baseline
        or cache_only_in_candidate
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
    compare_parser.set_defaults(func=compare_labels)

    list_parser = subparsers.add_parser("list", help="print the case table")
    list_parser.set_defaults(func=list_cases)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
