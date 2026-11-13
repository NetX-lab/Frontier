# Progress: merged Step legacy fidelity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Complete strict clean77, evidence audit and records-only commit/file-ownership verification. |

## Scope

Task ID: `D26-D27-D28-MERGED-FIDELITY`. Raw request: [requirements.md](requirements.md).
The tested source is `fa5bae2980366da1d075125d6c26248a737fc419`.

## Status and evidence

- **completed:** read root instructions and current main requirements/plan;
  created the fresh verification worktree/branch at the exact SHA. Candidate
  `git status --porcelain` was empty. Parent untracked directories were untouched.
- **completed:** audited the prior label
  `/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate`, revision
  `7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`. Its 77 unique case records all
  have consistent clean source/harness provenance; all 451 artifacts still
  match recorded size and SHA-256; all 77 definitions match current cases.
  Runner/matrix, examples and data have no tracked differences. Python and
  dependency versions match. Prior strict proof has 77 identical cases and
  clean/full cache-name comparison. Baseline has 519 cache filenames.
- **completed:** one candidate matrix run with jobs 4 and clean cache produced
  artifacts for 77/77 cases; every returncode is 0 and error is null.
- **completed:** existing strict comparator returned 0 and reports 77/77
  identical cases, complete comparison and clean/full cache comparison. There
  are zero metric/artifact, failure, missing, definition, provenance or cache
  differences; first mismatch is null.
- **completed:** post-run audit rehashed all 451 artifacts on each side
  (902/902 total); both actual cache inventories equal their 519-name manifests.
  Candidate HEAD/source/harness remain the exact tested SHA and status is empty
  before task records are installed. Baseline manifests/results are unchanged.
- **completed:** mainline self-check after preflight and after verified
  comparison found no drift; the full Step CLI remains parent-owned/pending.
- **completed:** installed the final scoped records and selected receipts only
  after the capture/comparison/audit finished. No source/test/fixture changed.
- **completed:** records commit `90307f215a4078dea42a88184f75e283d2db389b`
  contains exactly 19 additions in this task directory; parent is the tested
  SHA, source/tests/examples/data are unchanged and worktree status is clean.
  A records-only closure update records this verified commit outcome. Final
  record commits are returned separately from the tested SHA, not retested code.

Preflight receipt: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022/preflight_receipt.json`.
Raw candidate log: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022/candidate_run.log`.
Strict comparison: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022/comparison.json`.
Post-run receipt: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022/postrun_receipt.json`.
Exact commands/environment and results: [test report](test_report_2026-10-22_merged_clean77.md).

Reason for baseline reuse: the existing full clean run is intact and uses
identical matrix, wrappers and inputs; rerunning it would add cost without
changing the comparison. No expectations or simulator behavior were changed.

Verification limits: legacy matrix only; no full Step CLI, new D26 fixture
execution, native workflow/state acceptance, GPU/operator-gap or CPU-overhead
measurement. Prior parent reviews/check counts receive no test credit here.

Unresolved issues within this verification: none. Parent-only full CLI and
native/calibration prerequisites remain unchanged. The final records commit
is discoverable as the tip of `verify/fidelity-candidate-fa5bae2`; its complete
records-only range starts immediately after tested commit
`fa5bae2980366da1d075125d6c26248a737fc419`.
Commit/file-ownership receipt:
`/data/ycfeng/tmp/step5_workflow_fidelity_20261022/record_commit_receipt.json`.
