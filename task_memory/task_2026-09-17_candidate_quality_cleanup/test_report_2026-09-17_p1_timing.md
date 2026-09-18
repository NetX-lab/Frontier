## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded isolated paired runtime measurements. |

# P1 paired runtime verification

Environment: dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda; CPU execution, BLAS/OpenMP threads fixed to one. Frozen baseline `c288a19f59bec09529ee18d782fa57218da2c781`, candidate production `b0b9683b8fe6d03a6465b6e8fd7ac5662175d415`. Only the unexecuted metrics regression test was dirty. No tests ran concurrently and production remained unchanged.

Command from the feature worktree:

```bash
PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python tests/performance/measure_pr33_paired.py --baseline ../quality-baseline-c288a19f --candidate . --output /data/ycfeng/tmp/quality-p1-paired --repetitions 3
```

Harness: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/performance/measure_pr33_paired.py`, delegating to each revision's `tests/performance/sim_walltime_scaling/run_case.py`. Three cases, three interleaved repetitions, eighteen total successful measurements. Inputs/topology and sources are recorded in `/data/ycfeng/tmp/quality-p1-paired/manifest.json`; individual rows in `results.json` and each case's `measurement.json`/`run.log`.

Criteria: matching events and completions, successful execution, separate measured Simulator.run overhead and process elapsed time. No universal performance budget was assumed.

| Case | Baseline run median (s) | Candidate run median (s) | Median paired run change | Median paired process change | Events / completed requests |
| --- | ---: | ---: | ---: | ---: | --- |
| small_dense | 0.031171945 | 0.025772507 | -17.3215% | -8.5028% | 176 / 2, all equal |
| longer_dense | 0.055568096 | 0.050301037 | -10.1226% | -7.5923% | 352 / 4, all equal |
| representative_moe | 9.792026157 | 9.417455228 | -2.9555% | -5.4726% | 3560 / 2, all equal |

PASS for execution and event/completion preservation. Paired percentage is the median of `(candidate_i / baseline_i - 1) * 100`, not the ratio of independent medians. Reporting is disabled and timings are dummy predictions; these results do not establish metrics-enabled speedup, trained numerical accuracy, or native GPU parity. Three samples, especially millisecond dense cases, are limited evidence rather than a general performance guarantee. The separate 58-case fidelity and eight non-dummy artifact checks establish their recorded numerical preservation, not this timing check alone.
