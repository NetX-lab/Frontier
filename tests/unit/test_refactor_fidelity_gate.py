"""The fidelity gate must not report success without successful comparisons.

These tests exist because it did.  `compare_labels` counted a case as covered
when its id appeared in both result tables, which is not the same as the case
having run, produced artifacts, and been compared.  A matrix whose executions
all failed therefore satisfied every condition the gate checked and exited
zero, which `measure_commit.py` prints as ``VERDICT: IDENTICAL``.

The second group covers provenance: one label has to describe one measurement,
because a filtered run keeps the records of the cases it did not execute while
rewriting the label-wide manifest with its own revision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e.refactor_fidelity.cases import build_cases
from tests.e2e.refactor_fidelity.measure_commit import reuse_blocked_reason
from tests.e2e.refactor_fidelity.run_matrix import check_retained_records, compare_labels


ALL_CASES = build_cases()
BASELINE_REVISION = "a" * 40
CANDIDATE_REVISION = "c" * 40
HARNESS_REVISION = "h" * 40


def _write_side(
    output_root: Path,
    label: str,
    *,
    revision: str,
    failed: frozenset[str] = frozenset(),
    content: str = '{"value": 1}',
    content_overrides: dict[str, str] | None = None,
    delete_artifacts_for: frozenset[str] = frozenset(),
    digest_overrides: dict[str, str] | None = None,
    dirty: bool = False,
    stamped: bool = True,
    clean_cache: bool = True,
    case_filter: str | None = None,
) -> Path:
    """Write one synthetic label directory shaped like a real matrix run."""

    label_root = output_root / label
    label_root.mkdir(parents=True, exist_ok=True)
    records = []
    for case in ALL_CASES:
        artifact_dir = f"cases/{case.case_id}/metrics/run"
        record = {
            **case.as_record(),
            "returncode": 1 if case.case_id in failed else 0,
            "duration_seconds": 0.1,
            "artifact_dir": artifact_dir,
            "error": "boom" if case.case_id in failed else None,
            "log_tail": "traceback" if case.case_id in failed else None,
        }
        if stamped:
            record.update({
                "source_revision": revision,
                "source_dirty": dirty,
                "harness_revision": HARNESS_REVISION,
            })
        if digest_overrides and case.case_id in digest_overrides:
            record["case_digest"] = digest_overrides[case.case_id]

        if case.case_id in failed:
            record["artifact_dir"] = None
            record["artifacts"] = []
        else:
            body = (content_overrides or {}).get(case.case_id, content)
            record["artifacts"] = [{"name": "system_metrics.json",
                                    "sha256": "0" * 64, "bytes": len(body)}]
            if case.case_id not in delete_artifacts_for:
                target = label_root / artifact_dir
                target.mkdir(parents=True, exist_ok=True)
                (target / "system_metrics.json").write_text(body, encoding="utf-8")
        records.append(record)

    (label_root / "results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (label_root / "manifest.json").write_text(json.dumps({
        "label": label,
        "repo_root": str(output_root / f"{label}-checkout"),
        "git_head": revision,
        "git_dirty_paths": "M frontier/config/config.py" if dirty else "",
        "case_count": len(records),
        "case_filter": case_filter,
        "cases_executed_in_last_run": [case.case_id for case in ALL_CASES],
        "cache_clean_before_run": clean_cache,
        "cache_files": ["model_abc.pkl"],
        "source_revision": revision,
        "source_dirty": dirty,
        "harness_revision": HARNESS_REVISION,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return label_root


def _compare(output_root: Path, allow_partial: bool = False) -> tuple[int, dict]:
    exit_code = compare_labels(argparse.Namespace(
        output_root=str(output_root),
        baseline_label="baseline",
        candidate_label="candidate",
        allow_partial=allow_partial,
    ))
    report = json.loads((output_root / "comparison.json").read_text(encoding="utf-8"))
    return exit_code, report


# --- R34-01: the gate cannot pass without successful comparisons -------------


def test_healthy_identical_sides_pass(tmp_path: Path) -> None:
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 0
    assert report["cases_compared"] == len(ALL_CASES)
    assert report["mismatched_cases"] == []
    assert report["complete_comparison"] is True


def test_healthy_content_mismatch_fails(tmp_path: Path) -> None:
    changed = ALL_CASES[0].case_id
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION,
                content_overrides={changed: '{"value": 2}'})

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert [entry["case_id"] for entry in report["mismatched_cases"]] == [changed]
    assert report["cases_compared"] == len(ALL_CASES)


def test_every_case_failing_on_both_sides_is_not_a_pass(tmp_path: Path) -> None:
    """The exact false success this gate was missing."""

    everything = frozenset(case.case_id for case in ALL_CASES)
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION, failed=everything)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION, failed=everything)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert report["cases_compared"] == 0
    assert report["complete_comparison"] is False
    assert len(report["baseline_failures"]) == len(ALL_CASES)


def test_baseline_failure_hiding_a_different_candidate_failure_fails(tmp_path: Path) -> None:
    broken = frozenset({ALL_CASES[0].case_id})
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION, failed=broken)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION, failed=broken)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert report["cases_compared"] == len(ALL_CASES) - 1
    assert report["baseline_failures"][0]["candidate_also_failed"] is True


def test_baseline_failure_with_candidate_success_is_not_full_equality(tmp_path: Path) -> None:
    broken = frozenset({ALL_CASES[0].case_id})
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION, failed=broken)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert report["cases_not_compared"] == [ALL_CASES[0].case_id]
    assert report["baseline_failures"][0]["candidate_also_failed"] is False


@pytest.mark.parametrize("sides", [("baseline",), ("candidate",), ("baseline", "candidate")])
def test_missing_artifact_directory_is_missing_evidence(tmp_path: Path, sides) -> None:
    gone = frozenset({ALL_CASES[0].case_id})
    for label, revision in (("baseline", BASELINE_REVISION), ("candidate", CANDIDATE_REVISION)):
        _write_side(tmp_path, label, revision=revision,
                    delete_artifacts_for=gone if label in sides else frozenset())

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert [entry["case_id"] for entry in report["cases_with_missing_evidence"]] == list(gone)
    assert report["cases_compared"] == len(ALL_CASES) - 1
    assert report["cases_not_compared_without_explanation"] == []


def test_a_case_absent_from_both_sides_needs_allow_partial(tmp_path: Path) -> None:
    for label, revision in (("baseline", BASELINE_REVISION), ("candidate", CANDIDATE_REVISION)):
        label_root = _write_side(tmp_path, label, revision=revision)
        results = label_root / "results.jsonl"
        kept = [line for line in results.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["case_id"] != ALL_CASES[0].case_id]
        results.write_text("\n".join(kept) + "\n", encoding="utf-8")

    assert _compare(tmp_path)[0] == 1
    exit_code, report = _compare(tmp_path, allow_partial=True)
    assert exit_code == 0
    assert report["cases_absent_from_both_sides"] == [ALL_CASES[0].case_id]


# --- R34-02: one label describes one measurement -----------------------------


def test_same_case_id_with_a_different_definition_is_not_compared(tmp_path: Path) -> None:
    renamed = ALL_CASES[0].case_id
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION,
                digest_overrides={renamed: "deadbeefdeadbeef"})

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert [entry["case_id"] for entry in report["cases_with_differing_definitions"]] == [renamed]
    assert renamed not in report["identical_cases"]


def test_a_side_measured_dirty_is_reported(tmp_path: Path) -> None:
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION, dirty=True)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert any("modified working tree" in finding for finding in report["provenance_findings"])


def test_a_side_without_provenance_is_reported(tmp_path: Path) -> None:
    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION)
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION, stamped=False)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 1
    assert any("no source provenance" in finding for finding in report["provenance_findings"])


def test_an_assembled_cache_is_not_compared(tmp_path: Path) -> None:
    """A filtered continuation without a cache clean cannot be compared by name."""

    _write_side(tmp_path, "baseline", revision=BASELINE_REVISION,
                clean_cache=False, case_filter="dp_")
    _write_side(tmp_path, "candidate", revision=CANDIDATE_REVISION)

    exit_code, report = _compare(tmp_path)

    assert exit_code == 0, "an assembled cache is not by itself a fidelity failure"
    assert report["predictor_cache_compared"] is False
    assert report["predictor_cache_populated_cleanly"] is False


def _retained(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
                    encoding="utf-8")
    return path


def _record(case_id: str, **overrides) -> dict:
    record = {
        "case_id": case_id,
        "source_revision": BASELINE_REVISION,
        "source_dirty": False,
        "harness_revision": HARNESS_REVISION,
    }
    record.update(overrides)
    return record


def test_filtered_continuation_on_the_same_source_is_allowed(tmp_path: Path) -> None:
    known = {case.case_id for case in ALL_CASES}
    first, second = ALL_CASES[0].case_id, ALL_CASES[1].case_id
    path = _retained(tmp_path, [_record(first), _record(second)])

    retained, conflicts, stale = check_retained_records(
        path,
        {"source_revision": BASELINE_REVISION, "source_dirty": False,
         "harness_revision": HARNESS_REVISION},
        {second},
        known,
    )

    assert conflicts == []
    assert stale == []
    assert set(retained) == {first, second}


def test_continuation_on_a_different_revision_conflicts(tmp_path: Path) -> None:
    known = {case.case_id for case in ALL_CASES}
    first, second = ALL_CASES[0].case_id, ALL_CASES[1].case_id
    path = _retained(tmp_path, [_record(first), _record(second)])

    _, conflicts, _ = check_retained_records(
        path,
        {"source_revision": CANDIDATE_REVISION, "source_dirty": False,
         "harness_revision": HARNESS_REVISION},
        {second},
        known,
    )

    assert conflicts == [first], "the case this run did not re-execute is the one at risk"


def test_continuation_after_the_source_became_dirty_conflicts(tmp_path: Path) -> None:
    known = {case.case_id for case in ALL_CASES}
    first, second = ALL_CASES[0].case_id, ALL_CASES[1].case_id
    path = _retained(tmp_path, [_record(first), _record(second)])

    _, conflicts, _ = check_retained_records(
        path,
        {"source_revision": BASELINE_REVISION, "source_dirty": True,
         "harness_revision": HARNESS_REVISION},
        {second},
        known,
    )

    assert conflicts == [first]


def test_unstamped_retained_records_conflict(tmp_path: Path) -> None:
    known = {case.case_id for case in ALL_CASES}
    first = ALL_CASES[0].case_id
    path = _retained(tmp_path, [{"case_id": first}])

    _, conflicts, _ = check_retained_records(
        path,
        {"source_revision": BASELINE_REVISION, "source_dirty": False,
         "harness_revision": HARNESS_REVISION},
        set(),
        known,
    )

    assert conflicts == [first]


def test_a_retained_record_outside_the_case_table_is_stale(tmp_path: Path) -> None:
    known = {case.case_id for case in ALL_CASES}
    path = _retained(tmp_path, [_record("coloc_dense_offline_renamed_away")])

    _, conflicts, stale = check_retained_records(
        path,
        {"source_revision": BASELINE_REVISION, "source_dirty": False,
         "harness_revision": HARNESS_REVISION},
        set(),
        known,
    )

    assert stale == ["coloc_dense_offline_renamed_away"]
    assert conflicts == []


# --- R34-02: a reused detached checkout must be clean ------------------------


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "source.py").write_text("value = 1\n", encoding="utf-8")
    git("add", "source.py")
    git("commit", "-q", "-m", "initial")
    return repo, git("rev-parse", "HEAD")


def test_a_clean_checkout_at_the_right_commit_may_be_reused(tiny_repo) -> None:
    repo, head = tiny_repo
    assert reuse_blocked_reason(repo, head) is None


def test_a_checkout_at_another_commit_is_refused(tiny_repo) -> None:
    repo, _ = tiny_repo
    reason = reuse_blocked_reason(repo, "f" * 40)
    assert reason is not None and "not " + "f" * 40 in reason


def test_a_modified_checkout_is_refused_even_at_the_right_commit(tiny_repo) -> None:
    repo, head = tiny_repo
    (repo / "source.py").write_text("value = 2\n", encoding="utf-8")

    reason = reuse_blocked_reason(repo, head)

    assert reason is not None
    assert "working tree has been modified" in reason
    assert "source.py" in reason
    assert (repo / "source.py").read_text(encoding="utf-8") == "value = 2\n", (
        "the check must not clean the checkout to make itself pass"
    )


def test_an_untracked_source_file_also_refuses_reuse(tiny_repo) -> None:
    repo, head = tiny_repo
    (repo / "extra_module.py").write_text("value = 3\n", encoding="utf-8")

    reason = reuse_blocked_reason(repo, head)

    assert reason is not None and "extra_module.py" in reason
