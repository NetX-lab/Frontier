# Summary: merged Step legacy fidelity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archive strict 77/77 legacy fidelity PASS and verification limitations. |

## Overview

Task ID: `D26-D27-D28-MERGED-FIDELITY`.
Raw request: [requirements.md](requirements.md).
Tested source: `fa5bae2980366da1d075125d6c26248a737fc419`.
Worktree: `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2`.
Branch: `verify/fidelity-candidate-fa5bae2`.

The intact prior clean77 label at
`/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate`, source/harness
`7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`, is reused after manifest/provenance,
77-case definitions, 451 artifact hashes, runner/examples/data and dependency
identity checks. The baseline was not rerun or changed.

## Deliverables

Persistent record directory:
`task_memory/task_2026-10-22_step5_workflow_fidelity/`.
Required records: `requirements.md`, `plan.md`, `progress.md`, this `summary.md`
and `test_report_2026-10-22_merged_clean77.md`.
Selected evidence: `evidence/{preflight_receipt,postrun_receipt,scope_receipt}.json`,
`evidence/{baseline_manifest,candidate_manifest,comparison}.json`,
`evidence/{candidate_run,comparison,evidence_audit}.log` and
`checkpoint_2026-10-22_merged_fidelity.md` in the record directory.
Raw artifacts: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022/`.

## Observed validation

Strict comparison **PASS**: candidate produced artifacts for 77/77 cases; all
77 cases were actually compared and identical. Metric/artifact, failure,
missing-case/evidence, definition, provenance and cache-filename differences
are all zero. Clean/full cache comparison is enabled; each side has 519 cache
filenames. All 451 artifacts per side were rehashed against recorded sizes and
SHA-256 (902/902 total). Both cache inventories match the manifests. Candidate
source/harness remained clean at the exact requested SHA throughout capture;
records were installed only afterward. No first mismatch exists.

Comparison SHA-256:
`e1f4f94047f24889db7109f5de02b364e8ab033df8d36421da6978b496e4d677`.
Exact commands, environment and PASS conditions are in the test report.

## Open/deferred work

This task changes no source/tests/fixtures and runs no native/platform/GPU,
authentication/cloud/SSH or performance suite. Full Step CLI and native
workflow/calibration acceptance remain with the parent. The legacy matrix is
CPU simulator execution only, not native CPU-overhead measurement. Parent
reviews and 121/191/299/81 checks are not counted as this task's evidence.
