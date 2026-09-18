## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded final-source review-remediation verification and comparison scope. |

# PR33 review remediation — final verification

## Source and environment

- Reviewed source: `d43ae93240444bd4eff9bd99f296d2e751370514`.
- Tested production/test source: `35ac95eb231aa3990a25cd49e24b62b034bda3a3`. Subsequent edits are documentation only; source equality is checked before archive.
- Frozen comparison candidate: `c288a19f59bec09529ee18d782fa57218da2c781`, checkout `../quality-baseline-c288a19f`.
- Frozen main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Exact candidate preservation is distinct from earlier accepted main-to-candidate D01 differences.
- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Python: `/data/ycfeng/tmp/quality-review-env/bin/python`, 3.12.3, uv environment, no conda activation. No GPU execution.

Prefix each command below with:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl /data/ycfeng/tmp/quality-review-env/bin/python
```

```bash
-m pytest tests/unit -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-final-unit --junitxml=/data/ycfeng/tmp/pr33-final-unit.xml
-m pytest tests/integration/test_pr33_nondummy_acceptance.py tests/integration/test_gdn_phase_admission.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-final-nondummy
tests/integration/run_scheduler_refactor_fidelity.py --baseline ../quality-baseline-c288a19f --candidate . --output /data/ycfeng/tmp/pr33-final-fidelity --workers 2
```

Full integration script paths are the working directory above joined with the exact `tests/integration/...` paths. Logs use `/data/ycfeng/tmp/pr33-final-{unit,nondummy,fidelity}.log`. Correctness campaigns run concurrently; isolated timing starts only after all exit.

## Full unit — no new failed or skipped nodes

Observed **3928 PASS / 18 FAIL / 25 SKIP**, 576 warnings, 167.43 s; exit code 1. This is not an all-green suite. Parsed both final JUnit and `/data/ycfeng/tmp/quality-p5-unit.xml`: the failed-node sets are identical (18 each), error sets empty, and skipped-node sets identical (25 each). No tests were removed or expected outputs weakened. Passing count increases by 115 from the prior 3813-pass cleanup checkpoint.

The retained failures are ten legacy optimizer/debug contracts, five public documentation contracts, two missing-FlashInfer stage2 CLI paths and one legacy MLA trace-builder dependency path. Prior reports reproduced these against frozen candidate and main. The current run matches that exact node set. All 18 failure messages are also identical after substituting only the per-run scratch-directory name. Classification evidence: `/data/ycfeng/tmp/pr33-final-unit-classification.{log,json}`; durable node lists are retained in the final evidence JSON.

Reproduce the classification by parsing `xml.etree.ElementTree`, selecting every `testcase` with a `failure`, `error`, or `skipped` child, and comparing sets of `(classname, name)` between those two XML files. Require equality for each status; observed additions/removals are all empty.

## Non-dummy — nine cases PASS and enabled artifacts preserved

Observed **9 PASS**, 81.28 s. Eight existing real Simulator cases use synthetic trained profiles and reporting enabled. The additional R01 case uses capacity 3 and four staggered requests: it observes 14 batches, two mixed batches, a one-token final prefill chunk, stable slot ownership, slot reuse and zero final KV/GDN ownership. The second batch is exactly request 1 prefill 31 plus request 0 decode 1. RuntimeWarning is asserted; native numerical equivalence is not.

Compared the eight existing cases to the retained frozen-candidate run `/data/ycfeng/tmp/quality-pre-nondummy/test_nondummy_simulator_accept<N>/run`, N=0..7, using the existing `compare_baseline` helper. Require every result PASS, equal bidirectional metrics-file path sets, and exactly one nonempty acceptance/ledger-summary pair per case. Observed **90/90 stable artifacts PASS**, **106/106 symmetric metrics files**, **16/16 nonempty supplemental JSON pairs PASS**. Per-case metric-file counts are `[10,14,17,10,11,15,18,11]`. Trace wall-clock headers retain the existing comparator treatment; event rows are compared. No comparator tolerance changed.

Evidence: `/data/ycfeng/tmp/pr33-final-artifact-comparison.log`; mixed-batch observations at `/data/ycfeng/tmp/pr33-final-nondummy/test_staggered_gdn_admission0/phase_admission_evidence.json`. Durable summaries are copied into the final evidence JSON.

## Fidelity and isolated timing

Fidelity: **58 PASS / 0 FAIL**, 304 compared artifacts, 122 completed requests on each side. The unchanged comparator uses rel_tol=1e-12 and abs_tol=1e-9. A separate traversal of all compared artifacts found **902,628 finite numeric leaf pairs, zero unequal pairs, maximum absolute difference 0.0 and maximum relative difference 0.0**. Numeric leaves include metadata and identities as well as timing; this is preservation against a simulator baseline, not error against hardware ground truth.

For reproduction, load results.json and require exactly 58 PASS entries. For each baseline/candidate run locate its single request_metrics.csv parent, load every listed compared_artifact with the existing load_artifact helper, recursively check key sets/list lengths, and compute absolute/relative differences for finite float-convertible leaf pairs. Require zero difference when the reference is zero; retain matching non-finite values without converting them. Raw numeric check: `/data/ycfeng/tmp/pr33-final-fidelity-numeric.log`; durable outcomes: `review_final_evidence.json`.

Isolated paired timing: **18/18 successful measurements; 9/9 pairs preserve event and completed-request counts**. Same source/baseline as above. No concurrent task-owned test, benchmark, profiler or GPU job ran; documentation editing continued. This does not imply exclusive host ownership. The manifest records documentation-only dirty files; `git diff --exit-code 35ac95eb -- frontier tests` confirms source equality.

Exact command from the working directory:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python tests/performance/measure_pr33_paired.py --baseline ../quality-baseline-c288a19f --candidate . --output /data/ycfeng/tmp/pr33-final-paired --repetitions 3
```

Full driver: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/performance/measure_pr33_paired.py`. Per-revision runner: `<checkout>/tests/performance/sim_walltime_scaling/run_case.py`. The driver uses sequential online PDD, dummy predictor 1 ms, seed 42, reporting disabled identically, fresh subprocesses and alternating pair order. Small dense: 2 requests, QPS 2, prefill/decode 16/8, TP4 DP1 EP1 PP2; longer dense: 4 requests, QPS 10, 64/16, same topology; MoE: 2 requests, QPS 5, 32/8, attention TP8 DP1, MoE TP1 EP8 PP2. h800/h800_dgx are simulated identities, not physical GPU measurements.

Changes are medians of the three paired percentages `100*(candidate/baseline-1)`, not ratios of marginal medians.

| Workload | Median paired Simulator.run change | Median paired subprocess change |
| --- | ---: | ---: |
| small_dense | -12.07% | -3.62% |
| longer_dense | -8.84% | -6.90% |
| representative_moe | -0.62% | -2.92% |

All nine pairs are retained below; times are seconds, rounded only for display.

| Workload | Attempt | Baseline run | Candidate run | Baseline subprocess | Candidate subprocess | Events each | Completed each |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_dense | 0 | 0.030495455 | 0.025645521 | 6.229321949 | 6.003915065 | 176 | 2 |
| small_dense | 1 | 0.029758692 | 0.026166176 | 6.236097936 | 6.025903242 | 176 | 2 |
| small_dense | 2 | 0.030057975 | 0.035053768 | 6.291508305 | 6.045921459 | 176 | 2 |
| longer_dense | 0 | 0.055141710 | 0.050586566 | 6.315919811 | 5.879805706 | 352 | 4 |
| longer_dense | 1 | 0.055671859 | 0.050421174 | 6.455649538 | 5.946381642 | 352 | 4 |
| longer_dense | 2 | 0.057474560 | 0.052395720 | 6.485158784 | 6.082760675 | 352 | 4 |
| representative_moe | 0 | 9.784404630 | 9.526202969 | 16.152272770 | 15.422850012 | 3560 | 2 |
| representative_moe | 1 | 9.536195810 | 9.499557681 | 15.726461716 | 15.438890610 | 3560 | 2 |
| representative_moe | 2 | 9.657511835 | 9.597793765 | 15.931673119 | 15.466162300 | 3560 | 2 |

The small-dense attempt-2 Simulator.run sample is **16.62% slower** (about 5 ms); the other two are 15.90% and 12.07% faster. It is retained, not replaced. Three pairs with this variability do not establish universal or statistically reliable speedup. MoE's median improvement is only 0.62%. These comparisons include the complete cleanup versus c288a19f and do not isolate C02 causality. No universal performance threshold was authorized. The measured allocation/call reduction is separately established by C01/C02 regressions.

Raw evidence: `/data/ycfeng/tmp/pr33-final-paired.log`, `/data/ycfeng/tmp/pr33-final-paired/{manifest,results,paired_summary}.json` and per-case measurement/run artifacts. All 18 measurements and pair summaries are retained in `review_final_evidence.json`. Reporting-enabled artifact preservation was verified separately above. No production-profile or native speed claim is made.

## Scope and limits

R01 follows the user's explicit temporary approximation, superseding the review's scheduler phase-purity proposal. GDN uses prefill estimators for the complete mixed workload, with real features and state ownership preserved. Co-location is explicitly documented. Native profiling/training mixed rows remain rejected, and full-attention mixed-profile requirements remain intact.

R02/R08 native attention validation, TP8 native execution, SGLang native replay and production-profile GDN accuracy remain unexecuted on hardware. CPU doubles exercise orchestration, not native kernels. Model dtype identity validation does not establish output-gate variant equivalence. Immutable estimator publication intentionally retains old generations. These are explicit limits, not silently reported passes.

No external PR edit, push, merge or native worker provisioning occurred. C03 is delivered as a local reviewable PR description.
