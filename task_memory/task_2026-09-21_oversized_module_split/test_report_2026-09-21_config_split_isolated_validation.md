# Test Report 2026-09-21 — Isolated Validation of the Config Split (`99922d2`)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Initial report. |

## Why this run exists

The branch worktree was being edited while the config split's own matrix run was in flight, so that run measured a mixture of the committed config split and an in-progress scheduler split and had to be discarded. This report records a run that cannot have that problem: the simulator under test is a **read-only detached checkout of the commit itself**, not the shared working tree.

It is an independent confirmation of `99922d2`, not a replacement for the gate that commit already passed.

## Method

| Field | Value |
| --- | --- |
| Commit under test | `99922d26365e5639bfdbcc2fe59e94d0750a6d27` (`refactor(config): split config.py into one module per configuration family`) |
| Checkout under test | `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-99922d2`, detached at that commit, no source edits |
| Only local change | `tests/e2e/refactor_fidelity/run_matrix.py`, the harness hardening described below. No file under `frontier/` differs from the commit. |
| Baseline | `/data/ycfeng/tmp/issue26-correctness-pr/refactor-fidelity/baseline`, captured earlier from `.worktrees/fidelity-baseline-main` at `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6 |
| Command | `run_matrix.py run --repo-root <isolated checkout> --label candidate_99922d2 --output-root <scratch> --jobs 4 --clean-cache --continue-on-failure`, then `compare --baseline-label baseline --candidate-label candidate_99922d2` |

## Result

| Measure | Expected | Actual |
| --- | --- | --- |
| Cases producing artifacts | 67 of 67 | 67 of 67 |
| Identical cases | 67 | **67** |
| Mismatched cases | 0 | **0** |
| Baseline failures excluded | 0 | 0 |
| Candidate-only failures | 0 | **0** |
| Predictor cache file names differing | 0 | **0** |

**Result: PASS.** Every compared artifact matched exactly: `request_metrics.csv`, `system_metrics.json`, `frontier_stage_batch_ledger.jsonl`, `op_precision_metadata.csv` and `config.json`, across all 67 cases, with only run-specific absolute paths normalized.

The predictor cache result is worth stating separately. All 426 cache file names matched, which means the split changed no training identity and no model cache key. Output equality alone would not have shown that, because retraining from the same CSV reproduces the same numbers.

## Harness hardening applied for this run

Two changes to `tests/e2e/refactor_fidelity/run_matrix.py`, both about diagnosability rather than about what counts as a pass:

1. **Bounded retry on artifact discovery.** A case that exits zero but appears to have written nothing is retried for up to ten seconds before being recorded as a failure. A directory listing can lag the child process, and a spurious failure is worse than waiting. A genuinely empty run still fails, only later.
2. **Failure log capture.** The last twenty lines of a failing case's log are stored in `results.jsonl` and printed by `compare`. Without this the evidence is lost as soon as the case is re-run, because a re-run overwrites `run.log`.

Both were motivated by a concrete incident: two cases were reported as failures in an earlier run, passed immediately when re-run individually, and by then their logs had been overwritten, so the cause could not be established from the record.

## Limits

- This confirms that 67 supported configurations produce identical output at `99922d2`. It says nothing about configurations outside the matrix, and it makes no accuracy claim.
- The isolated checkout remains on disk at `.worktrees/fidelity-candidate-99922d2` as a pre-scheduler-split reference. It is detached and ignored by Git; delete it once the branch's later splits are validated.
