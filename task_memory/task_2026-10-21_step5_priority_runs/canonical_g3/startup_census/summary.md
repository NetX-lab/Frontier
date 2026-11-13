# D23 startup census completion

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archive the verified observer lifecycle correction. |

The native compile/warmup path invokes the same worker `execute_model` used during serving. The observer previously classified those calls as serving, contaminating its all-rank census. The correction scopes startup to the existing `Worker.compile_or_warm_up_model` lifetime, restores state on exceptions and preserves later useful/dummy observations without classifying request names.

Deliverables:

- `tests/integration/step5_pd/r2_observation.py`: lifecycle marker and startup business-capture exclusion.
- `tests/integration/step5_pd/r2_runtime.py`: startup runtime-census exclusion.
- `tests/unit/test_step5_native_forward_census.py`: nested-worker RED/GREEN, exception cleanup, later serving retention and locked-source callback coverage.
- `test_report_2026-10-22_startup_census.md`: exact commands, evidence and limits.

Parent validation: expected RED02 2 FAIL / 1 PASS; affected suite 181 PASS; final census fixture 19 PASS; diff check PASS. Original capture01 verdicts remain unchanged. A future native capture is still required to validate this observer fix on hardware; G3 and calibration are not qualified by CPU results.
