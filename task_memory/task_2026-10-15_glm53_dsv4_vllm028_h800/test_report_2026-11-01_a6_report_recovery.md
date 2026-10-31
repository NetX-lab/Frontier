# A6 report recovery and error attribution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Reconstructed selected CSA medians from raw calls and began independent-run persistence checks. |
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


## CSA measurement groups and raw reconstruction

The unchanged table contains 1,421 CSA workload pairs. The `plans_tile_schedule=True` row pools layer 2; the `False` row pools layers 4, 6, 8 and 10. These are two measurement groups, not two individual layers. `frontier/profiling/attention/dsv4_replay_rows.py` groups the measured calls by this field and takes the median across each group. For each of the four failing scopes, every pair has equal kernel counts.

The following differences describe paired measurements, not prediction errors. The denominator is the smaller positive member of each pair.

| Scope (dsv4_ prefix omitted) | Available pairs | Median absolute paired difference / minimum | Pairs above 10% |
| --- | ---: | ---: | ---: |
| q_projection | 1,421 | 3.526190% | 455 |
| index_fused_query_prepare | 1,420 | 6.324645% | 456 |
| compressor_fused_compress_store | 1,421 | 5.987726% | 478 |
| index_compressor_fused_compress_store | 1,421 | 3.683595% | 325 |

Verification used `timeout 60s python3 /data/ycfeng/tmp/glm53_dsv4/inspect_csa_groups_20261101.py` from the worktree. It streamed the retained G1h-02 e0/e1 JSONL files with `resource.RLIMIT_AS=30 GiB`. The attempted `systemd-run --scope -p MemoryMax=30G` required interactive authentication and started no raw read; this failure is retained in the command receipt. The stream-based check exited 0 in 3.292388 seconds. All 40 selected raw calls reconstruct all 16 scope/group medians, with absolute reconstruction difference below 1e-12 ms: **PASS_SELECTED_RAW_MEDIANS_RECONSTRUCT**.

For `g1-e0-0263-full`, compressor medians are 11.808105 us for layer 2 and 19.759766 us for the four-layer reuse pool. For `g1-e1-0729-first_tail`, they are 11.681152 and 20.304443 us. Each case has one sampled step. The difference already exists in the raw calls; aggregation is not its origin in these cases. This does not prove measurement corruption or a causal effect of tile planning. Planner status is associated with layer position and execution state, so introducing it as a cost feature requires further evidence.

Evidence: `h3o_operator_costs/a6_codex_20261101/csa_measurement_groups.json`, `csa_measurement_groups_command.json`, and the exact script copy `inspect_csa_groups.py`. The JSON records input hashes and every selected actual/reconstructed value.

### Next hypothesis test, recorded before execution

Hypothesis: the signed first-CSA-layer versus reuse-layer difference represents persistent layer execution state rather than variation specific to G1h-02. Reuse the existing independent G2h runner attention measurements with the same workload and layer identities. Join exact non-timing identities, report missing/conflicting identities, then compare signed paired differences and per-group duration ratios. Frequent sign reversal or failure of large G1h contrasts to recur would argue against a stable planner/layer split. Agreement would support persistence but would not establish causality, because run/source/instrumentation differences remain confounders. Keep all measurements and folds unchanged. This is an input-diagnosis check for tickets 05/09, not a new A6 gate or permission to refit.


## Independent-run CSA repeatability result

`tests/analysis/v4_attention_repeats.py` compared the retained G1h-02 rows with G2h runner 01 and 04. The final identity check removes only the routing descriptor's top-level source `origin`; all nested native-source paths and hashes remain identity fields, consistent with decision 40. All 4,945 row identities and all scope sample/kernel counts match in each comparison. All 67,244 G1h scope medians match the published A6 input at tolerance 1e-12 ms. The final command exits 0 in 6.149610 seconds with Python 3.10.6 and a 30-GiB address-space limit. Its output is byte-identical to the preliminary comparison, whose broader source-path normalization was narrowed before acceptance. Both command receipts are retained.

Command: `timeout 60s /data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_attention_repeats.py --reference /data/ycfeng/tmp/b13r_accept/after/rows.json --comparison /data/ycfeng/tmp/g2h_accept/rows_01/rows.json --comparison /data/ycfeng/tmp/g2h_accept/rows_04/rows.json --published-table data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv --output <fresh-output.json>`. Working directory is the task worktree. The exact absolute command and timestamps are in `csa_repeats_strict_command.json`; `csa_repeats.json` retains all per-pair values and input hashes.

| Scope (dsv4_ prefix omitted) | Matched CSA pairs | G1h differences above 10% | Same sign in G2h-01 / G2h-04 | Above-10% difference recurring with same sign, G2h-01 / G2h-04 |
| --- | ---: | ---: | ---: | ---: |
| q_projection | 989 | 289 | 512 / 538 | 73 / 76 |
| index_fused_query_prepare | 988 | 304 | 506 / 496 | 73 / 61 |
| compressor_fused_compress_store | 989 | 244 | 532 / 531 | 51 / 47 |
| index_compressor_fused_compress_store | 989 | 239 | 510 / 547 | 69 / 80 |

Only 50.2–55.3% of all matched contrasts retain their direction, and 19.3–33.5% of the large G1h contrasts recur above 10% in the same direction. The pre-recorded hypothesis of a persistent first-layer/reuse-layer contrast is not supported by these data.

| Compressor case | G1h first / reuse | G2h-01 first / reuse | G2h-04 first / reuse |
| --- | ---: | ---: | ---: |
| g1-e0-0263-full | 11.808105 / 19.759766 us | 17.695801 / 19.055908 us | 18.305176 / 18.240479 us |
| g1-e1-0729-first_tail | 11.681152 / 20.304443 us | 19.136230 / 20.352051 us | 20.895996 / 20.400391 us |

The corresponding G1h signed differences, -7.951660 and -8.623291 us, become +0.064697 and +0.495605 us in G2h-04. For q_projection, the G2h/G1h ratio of the single first-layer sample spans 0.720–1.408 (p05–p95) in G2h-01 and 0.735–1.409 in G2h-04, despite median ratios near 1.00. Group-wide median agreement from the earlier G2h reports therefore does not establish individual-workload precision.

**Decision under D-NO-INTERRUPT:** do not add a planner/layer feature or tune Random Forest to these contrasts. First measure repeatability of freshly reconstructed, identical workload geometry. Decision 35 and `dsa_vllm_runner.run_case` explain why simply increasing consecutive `measured` steps is inappropriate: contexts and compression writers advance. Independent reconstruction followed by one measured step preserves the intended case. Prepare a limited one-H200 probe through the existing frozen G1h package mechanism, within the remaining G1h–G4h budget, before changing measurement inputs. The retained budget record is approximately 4.52 of 7 GPU-hours; the exact new plan and timeout exposure must be recorded before submission.

Limits: G2h source/sampling instrumentation differs from G1h, even though attention row identities and native hashes match. The comparison supports run-dependent variation and rejects the simple persistent layer-role explanation; it does not isolate GPU cache, concurrency, clock, or measurement-instrumentation causes. It covers the original 989 workloads, not the 432 supplement-only workloads. No new measurement is promoted, no A6 population is dropped, and the formal metric/qualification gates remain unchanged.
