# A6 report recovery

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded complete recovery and four attention / seven collective curve failures. |
| 2026-11-01 | Recorded the observed report failure, scoped source-reference correction and affected tests. |

## Scope and cause

Tickets 05/09 require a fresh A6 held-out evaluation. The October 31 evaluation completed its five training folds but failed before writing `result.json`: `family_classification()` referenced the removed `MeasuredCCBackend._predict_unequal`. The current runtime routes unequal SP collective rows through `_predict_exact_sp`; the report now cites that method and names the family accordingly. This changes report metadata only. Model fitting, admission, folds, thresholds and simulator behavior are unchanged.

Python: `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6. Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Environment: `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Tests use no pytest cache and private temporary directories under `/data/ycfeng/tmp`.

## Verification

1. RED: `timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_v4_attention_heldout.py -k family_report -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/a6_report_red_20261101`. Expected: the real family-report call reproduces the missing-method error. Actual: **1 failed, 7 deselected**, exact AttributeError at line 427. Reproduction PASS.
2. GREEN: `timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_v4_attention_heldout.py tests/unit/test_measured_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/a6_report_green_20261101 --junitxml=/data/ycfeng/tmp/a6_report_green_20261101.xml`. Expected: report source resolves, real curve held-out statistics remain correct, both affected suites pass. Actual: **82 passed in 5.63 s**, no failures/errors/skips. PASS.
3. Full A6 recovery: PASS for execution, FAIL for full accuracy qualification. Recovery completed at 2026-10-31T16:51:13Z, exit0 in308.673018s. The five independent fold caches (720/831/831/831/711 files) were reused; the log has no CACHE MISS. Results are in `h3o_operator_costs/a6_codex_20261101/{command.json,result.json,summary.md}`; the failed October31 generation remains unchanged.

Evidence: `h3o_operator_costs/a6_codex_20261101/`. The original run receipt remains `h3o_operator_costs/a6_codex_20261031/command.json` (exit 1, 2348.880659 s). Unit success does not qualify the measured attention or communication tables.

## Observed accuracy and limits

The supplemented attention table has 7,105 layer rows, 1,421 complete workloads and five folds (seed1234). Four scopes exceed p90 <=10%: q_projection10.225%, index_fused_query_prepare12.267%, compressor_fused_compress_store13.992%, index_compressor_fused_compress_store17.506%. The two short-context-fill rows are not evaluable. All17,060 legally-zero contributions have correct selection;94,982 shared-admission predictions have no changed value. Seven of eight uniform collective curves fail the existing median/p90 rule. Tickets05/09 remain open.

The launch receipt binds source3e906fa3. The result's later HEAD/dirty state reflects concurrent F2 analysis and profiling-only edits; the attention table, fold/training/admission logic and thresholds were unchanged throughout recovery. Preserve both provenance records. No threshold, training grid or formal input is changed because of these results.
