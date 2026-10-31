# A6 report recovery and error attribution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Replayed recorded exact-feature lookups and isolated all four attention p90 failures from interpolation accuracy. |
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

## Exact-feature error attribution

`tests/analysis/v4_attention_exact_errors.py` reuses the original 1,421-workload fold assignment, the unchanged input SHA-256, the production loader, `_build_exact_feature_lookup` and `DeepSeekV4AttentionPredictor` admission. The runtime exact lookup averages the measured training values with the same feature tuple. This analysis constructs those means without estimator fitting. If an admitted row would require interpolation, it counts the request and emits no predicted value for it. Complete workloads remain held out; an exact feature tuple does not imply an identical held-out workload entered training.

Command from this worktree:

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028:/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_attention_exact_errors.py --result task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_codex_20261101/result.json --output /data/ycfeng/tmp/glm53_dsv4/a6_exact_config_bound_20261101.json --scratch-dir /data/ycfeng/tmp/glm53_dsv4/a6_exact_config_bound_20261101
```

Actual: exit 0 in 15.366596 seconds with `PASS: counts match A6; exact-only scope statistics match recorded results`, Python 3.10.6. The final tool reuses the A6 case builder for model/device/TP/block-size inputs and requires its binding to equal the recorded configuration. Its output is byte-identical to the initial exact-error analysis at the retained task path. Exact commands and source hashes are in `exact_errors_command.json` and `exact_errors_config_bound_command.json`; the final log is `exact_errors_config_bound.log`. All five folds complete. Each of the four scopes has exactly the original rejection, exact-lookup and interpolation counts. For the two scopes with no interpolation, median and p90 match the original A6 values at relative tolerance 1e-10 and absolute tolerance 1e-12. This is a consistency check of the analysis; A6 accuracy remains FAIL. Original warnings about mixed CSV metadata types are retained; they do not alter admission or values.

| Scope (dsv4_ prefix omitted) | Evaluable | Exact lookup | Interpolated | Exact rows with >10% error | Fraction of all evaluable rows | Exact-only p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_projection | 6,995 | 6,995 | 0 | 734 | 10.493209% | 10.225411% |
| index_fused_query_prepare | 2,798 | 2,798 | 0 | 480 | 17.155111% | 12.267273% |
| compressor_fused_compress_store | 5,576 | 5,402 | 174 | 890 | 15.961263% | 14.194680% |
| index_compressor_fused_compress_store | 2,794 | 2,706 | 88 | 540 | 19.327130% | 17.587952% |

Each scope already has more than 10% of its entire evaluable population above the 10% error threshold from exact lookups alone. Consequently, even perfect predictions for every interpolated row cannot make its p90 pass while these exact values and the population stay unchanged. Random Forest hyperparameter changes alone do not address these failures. This conclusion does not establish measurement corruption: measured time can differ between workloads mapped to the same feature tuple because of unrepresented state, measurement variation, or both.

The retained rows provide absolute values and per-mode/phase/width groups. Selected largest exact errors are:

| Scope | Table row / case | Measured | Predicted | Absolute error | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| q_projection | 5450 / g1hs-e0-0101-full | 7.169922 us | 10.870530 us | 3.700609 us | 51.612955% |
| index_fused_query_prepare | 6215 / g1hs-e0-0254-piecewise | 4.255859 us | 2.709420 us | 1.546439 us | 36.336709% |
| compressor_fused_compress_store | 3505 / g1-e1-0729-first_tail | 11.681152 us | 20.664307 us | 8.983154 us | 76.902980% |
| index_compressor_fused_compress_store | 1240 / g1-e0-0263-full | 7.583984 us | 11.195801 us | 3.611816 us | 47.624260% |

Every actual value is positive; no undefined relative-error denominator is replaced. `h3o_operator_costs/a6_codex_20261101/exact_errors.json` retains all exact rows and the recorded table/result hashes. For example, CSA/FULL/decode widths 8 and 16 account for 329 of the 734 q_projection errors and 380 of the 540 index-compressor errors. These strata identify where to inspect existing measurements; they are not exclusions or new fit partitions.

Next action for 05/09: inspect the retained measurements and native operator inputs for the same feature tuples, beginning with those CSA groups, and distinguish unrepresented operation state from measurement variation before changing features or collecting more data. No table, model, fold, population, criterion or formal input changes in this analysis. F1a/F2 replays continue independently.
