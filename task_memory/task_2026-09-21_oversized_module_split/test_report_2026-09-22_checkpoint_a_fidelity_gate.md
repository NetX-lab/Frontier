# Test Report 2026-09-22 — Checkpoint A: the fidelity gate's false successes

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Initial report: R34-01 and R34-02 corrections, with the before/after reproduction. |

## What this closes

Maintainer review comments **R34-01** (the gate can report success without
successful comparisons) and **R34-02** (partial reruns and reused worktrees can
mix source identities), from
`.local-draft/Frontier_PR34_PR35_Review_and_D1_D2_Decisions_2026-09-22.md`.

No file under `frontier/` was touched. The change is confined to
`tests/e2e/refactor_fidelity/` plus one new test module.

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2`, CPU only |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6 |
| Worktree | `/data/ycfeng/Frontier/.worktrees/oversized-module-split` |
| Pre-fix harness | extracted from `5ef96b5` with `git show` into a scratch directory, so no worktree was created or deleted for the comparison |

## 1. The defects, as found in source

### R34-01, first path: a full result table of failed runs

`compare_labels` skipped a case whose baseline execution failed, appending it to
`baseline_failures` — and `baseline_failures` was absent from the `failed`
predicate (`run_matrix.py:535` before the change). `complete` (`:437`) tested
only whether both result tables contained every case id, which a table of
failure records satisfies. A matrix in which every case failed on both sides
therefore compared nothing and returned 0, which `measure_commit.py:139` prints
as `VERDICT: IDENTICAL`.

The same branch hid a *new* candidate failure whenever the baseline had failed
on the same case.

### R34-01, second path: deleted artifact directories

`list_artifacts` returns `[]` for a path that does not exist
(`compare.py:83-84`), so `compare_artifact_directories` found no differences
between two absent directories and recorded the case as identical.

### R34-02: one label could describe several measurements

`run_label` merged previously recorded cases into `results.jsonl` and then
rewrote the label-wide manifest with the current run's `git_head`, environment
and cache listing. Nothing checked that the retained records came from the same
source. `measure_commit.py` reused an existing checkout after comparing `HEAD`
only; a detached worktree is not read-only.

## 2. Corrections

| Area | Change |
| --- | --- |
| Completeness | Completeness now means *compared*, not *present in the table*: `compared_ids` is built from the cases that reached a content comparison, and `not_compared = full_case_set - compared_ids`. |
| Failure predicate | `baseline_failures`, `missing_evidence`, `definition_mismatches`, `provenance_findings` and `unexplained` were added. `--allow-partial` now waives exactly one thing: cases nobody attempted on either side. |
| Structural guard | `unexplained` lists any case that was neither compared nor reported under a specific finding, so a future code path cannot drop a case silently and still pass. |
| Missing evidence | `missing_evidence_for` checks that a successful record still names an artifact directory, that the directory exists, and that the files on disk match the recorded inventory. `compare_artifact_directories` reports an absent directory instead of treating it as empty. |
| Case identity | `FidelityCase.definition_digest()` digests script, env, extra args and the serial/parallel flag. A case id whose digest differs between the two sides is not compared. |
| Source provenance | Every case record carries `source_revision`, `source_dirty` and `harness_revision`. `check_retained_records` refuses a continuation whose retained records disagree, **before** running anything, and reports retained ids that have left the case table. |
| Manifest | `case_count` counts the lines actually written. An earlier label reported 72 over 71 result lines because a stale retained id was counted but not written. |
| Cache comparison | Cache file names are compared only when both sides ran the whole table **and** each side's manifest records a clean cache with no case filter. |
| Reused checkouts | `measure_commit.reuse_blocked_reason` refuses a reused checkout with tracked modifications or untracked files. It reports and stops; it does not clean the checkout. |

## 3. Before/after reproduction

Driver: `scratchpad/driver/negative_control.py`, run once with the pre-fix
harness on `PYTHONPATH` and once with the corrected harness. Identical synthetic
inputs in both runs; the driver lives outside both trees so the script directory
cannot shadow the import.

| Scenario | Expected | Before (`5ef96b5`) | After |
| --- | --- | --- | --- |
| All 71 cases failed on both sides | fail | **exit 0**, compared 0 | exit 1, compared 0 |
| Baseline failure hiding a candidate failure | fail | **exit 0**, compared 70 | exit 1, compared 70 |
| Baseline failure, candidate succeeded | fail | **exit 0**, compared 70 | exit 1, compared 70 |
| Artifact directories deleted on both sides | fail | **exit 0**, compared 71 | exit 1, compared 70 |
| Healthy identical sides (control) | pass | exit 0, compared 71 | exit 0, compared 71 |
| Healthy content mismatch (control) | fail | exit 1, compared 71 | exit 1, compared 71 |

Four false successes before; none after; both controls unchanged. The fourth row
is the clearest: before the fix the deleted case was *counted as compared and
identical*; after it is excluded as missing evidence, which is why `compared`
drops to 70.

## 4. Committed regression tests

`tests/unit/test_refactor_fidelity_gate.py`, 22 tests, all passing:

```bash
PYTHONPATH=$PWD python -m pytest tests/unit/test_refactor_fidelity_gate.py -q -p no:cacheprovider
# 22 passed in 0.85s
```

They cover the six R34-01 scenarios the review listed, the five R34-02
scenarios, and the three reused-checkout states. One asserts that the dirty
check leaves the modified file untouched, so the check cannot start "fixing"
the checkout to pass itself.

Every test that mentions the harness:

```bash
PYTHONPATH=$PWD python -m pytest $(grep -rln --include="*.py" refactor_fidelity tests/ | sort) \
  -q -p no:cacheprovider
# 30 passed in 25.77s
```

## 5. Live verification of the run side

One real case, then a continuation, then a tampered continuation, against this
worktree (which was deliberately dirty at the time, and was recorded as such):

| Step | Command | Result |
| --- | --- | --- |
| Run one case | `run --label smoke --case-filter coloc_dense_offline_small` | `ok`, 5 artifacts; record stamped `source_revision=5ef96b5…`, `source_dirty=True`, `harness_revision=5ef96b5…`, `case_digest=a1e285a457d62d46` |
| Continue, same source | `run --label smoke --case-filter coloc_dense_offline_default` | allowed; 2 records |
| Continue after rewriting the first record's `source_revision` to `000…` | same command | **refused, exit 2**, naming the conflicting case and both provenance tuples |

The smoke output directory was removed afterwards.

## Limits

- This report is about the gate, not about the refactor. It does not re-establish
  that the production tree is unchanged; that is Checkpoint B.
- The provenance stamp is new, so **every label captured before this change now
  fails the gate for want of provenance**. That is intended: those labels cannot
  be shown to describe one revision, and the baseline label in particular was an
  assembled partial run. Both sides are recaptured in Checkpoint B.
- `missing_evidence_for` compares the recorded file *names* against disk, not
  their digests. The content comparison that follows reads the same files, so a
  changed file is caught there; a file replaced between recording and comparison
  with one of the same name is not separately flagged.
