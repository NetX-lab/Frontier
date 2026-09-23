# Test Report 2026-09-22 — C34-01: predictor-cache comparison eligibility

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created: defect, correction, gate tests, re-derivation of the Checkpoint B verdict. |

## Finding (external review of PR #34 / PR #35, 2026-09-22, C34-01, P1)

`compare_labels` in `tests/e2e/refactor_fidelity/run_matrix.py` compared the two
sides' predictor-cache file names when both result sets covered the full case
table and both manifests had `cache_clean_before_run: true` with an empty
`case_filter`. `_select_cases` also honors `--start` and `--limit`, which narrow
the executed selection without writing a filter. A run that cleaned the cache,
executed a slice, and merged its records onto full retained results therefore
carried a partially populated cache while satisfying the predicate, and a
matching partial cache on the other side would have compared equal for the
wrong reason. The manifest already records `cases_executed_in_last_run`; the
predicate did not read it.

## Correction

| File | Change |
| --- | --- |
| `tests/e2e/refactor_fidelity/run_matrix.py` | `populated_by_one_clean_full_run(manifest)` requires `cache_clean_before_run`, no `case_filter`, and `set(cases_executed_in_last_run) == full case table`; a manifest without the field is ineligible. The console line names the reason. Report keys unchanged (`predictor_cache_populated_cleanly`, `predictor_cache_compared`). |
| `tests/unit/test_refactor_fidelity_gate.py` | `_write_side` gains `cases_executed` and `record_cases_executed`; the healthy case asserts both cache keys true; four new tests: `--limit`-narrowed, `--start`-narrowed, symmetric partial caches, manifest without the executed list. |

No production module changed. The manifest schema is unchanged.

## Verification

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10),
`PYTHONPATH` at the worktree root, run from
`/data/ycfeng/Frontier/.worktrees/oversized-module-split`.

| # | Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | Gate unit tests | `python -m pytest tests/unit/test_refactor_fidelity_gate.py -q -p no:cacheprovider` | all pass, including the four new cases | 26 passed in 1.05 s | PASS |
| 2 | New cases fail on the old predicate | same file imported against the unmodified runner (observed when the tests were first run with the PR35 copy of `run_matrix.py` on `sys.path`) | the four new cases fail, the 22 existing pass | 4 failed, 22 passed | PASS (negative control) |
| 3 | Checkpoint B verdict re-derived | `python tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/issue26-correctness-pr/refactor-fidelity --baseline-label baseline_v2 --candidate-label candidate_bb582a4` | same verdict as the Checkpoint B report | `cases compared: 71 of 71`, `identical: 71`, `mismatched: 0`, `predictor cache file names differ: 0 baseline-only, 0 candidate-only`, exit 0; `comparison.json`: `predictor_cache_populated_cleanly: true`, `predictor_cache_compared: true`, `complete_comparison: true` | PASS |

Both Checkpoint B manifests record `cache_clean_before_run: true`, `case_filter:
null`, 71 entries in `cases_executed_in_last_run` and 426 cache files, so the
corrected rule admits them; the previous `comparison.json` was saved before the
rerun and the rerun rewrote it with the same verdict.

## Limits

- Observation, not inference: the retained output root is scratch and not
  committed; the rerun's `comparison.json` lives there.
- The rule reads the manifest the runner writes. A manifest edited by hand can
  still claim a full run; the gate protects against the harness's own `--start`
  and `--limit` paths, which is the reviewed gap.
