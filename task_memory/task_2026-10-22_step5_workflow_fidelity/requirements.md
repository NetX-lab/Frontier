# Requirements: merged Step legacy fidelity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record the scoped parent request and verification boundaries. |

## [Original Request] D26-D27-D28-MERGED-FIDELITY

The parent delegated strict clean77 verification of final merged Step source
`fa5bae2980366da1d075125d6c26248a737fc419`. Create a fresh checkout at
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2`; preserve all
existing worktrees, artifacts and the parent's untracked `.cortexkit/` and `.pi/`.
Read the root `AGENTS.md` and the current Step main requirements and plan.

Reuse a proven baseline from `step5_d23_perf_fidelity_20261022` or
`step5_d23_root_fidelity_20261022` only after checking the actual manifest,
provenance and identical matrix, runner and configuration inputs. Run the existing
matrix with label `candidate`, output root
`/data/ycfeng/tmp/step5_workflow_fidelity_20261022`, jobs 4, clean cache and
continue-on-failure. Use `/data/ycfeng/envs/frontier-py310/bin/python`, the current
checkout's `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`,
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.

Use the existing artifact comparator, including cache filename and provenance
checks. Report the exact baseline identity, commands/environment, coverage,
failures, missing evidence, cache differences, metric differences and receipts.
If comparison fails, retain the evidence and identify the first concrete
mismatch; do not fix it. Write scoped records in
`task_memory/task_2026-10-22_step5_workflow_fidelity`, commit only those records,
and return the record commit separately from the tested SHA.

## [Original Request] Limits

- This is CPU simulator execution, not native CPU-overhead measurement.
- Source edits and the unit suite belong to the parent. Do not edit source,
  expectations, fixture02, D26 fixtures or tests.
- No GPU, platform, authentication, cloud, SSH, performance suites or nested agents.
- The full Step CLI remains the parent's pending item; do not run it here.
- Prior narrow reviews and the parent's 121/191/299/81 checks are historical
  parent evidence, not this verification's results.

## References and acceptance

Main request/decisions: `../task_2026-10-15_step5_preview_h800/requirements.md`,
D26, D27, D28 and D29. Active dependency path: that task's `plan.md`,
"Immediate parallel work and verification". This slice supplies the required
unchanged-legacy verification only; it does not close Step actual-main,
native workflow, measured profiles or calibration acceptance.

Relevant ticket-owned criteria: `17-single-request-pd-handoff.md` (legacy
fidelity for the final native-matched change), `22-predictor-bundle-cost-replay.md`
(same-source legacy fidelity) and `08-native-pd-state-qualification.md`
(qualification-tooling fidelity). None of their native, full-request or
performance-budget criteria is closed by this scoped run.

Acceptance: 77/77 cases actually compared with no artifact, metric, definition,
provenance or cache-filename differences; clean, exact-SHA candidate provenance;
complete recorded commands and receipts; records-only commit. Report failure
honestly instead of altering expectations.
