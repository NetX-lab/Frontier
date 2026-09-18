## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded the completed final isolated 18-run paired campaign, exact workload preservation and measured performance limits. |

# Final paired wall-clock verification

## Source and execution

- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Frozen baseline: `c288a19f59bec09529ee18d782fa57218da2c781`, checkout `../quality-baseline-c288a19f`.
- Measured candidate: `567742f3df4cdc818400d00c0575426d66d0dee3`. Its production source is identical to `5cb8794f`; its tests are identical to `ff2b6cdf`. Both launch trees are clean in the manifest. Later archive-only edits do not change measured source.
- Python: `/data/ycfeng/tmp/quality-review-env/bin/python`, 3.12.3, uv venv with same-interpreter system packages, no conda activation.
- Full driver: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/performance/measure_pr33_paired.py`.
- Per-revision runner: `<revision checkout>/tests/performance/sim_walltime_scaling/run_case.py`.
- All correctness jobs and review workers had exited before launch. No concurrent task-owned test or benchmark load, source edit, profiler or GPU execution occurred during the campaign. This does not establish exclusive host ownership.

Exact command, run from the working directory above:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
 TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /data/ycfeng/tmp/quality-review-env/bin/python \
 tests/performance/measure_pr33_paired.py \
 --baseline ../quality-baseline-c288a19f --candidate . \
 --output /data/ycfeng/tmp/quality-p5-paired --repetitions 3 \
 > /data/ycfeng/tmp/quality-p5-paired.log 2>&1
```

The existing runner sets `NUMEXPR_NUM_THREADS=1`, disables W&B and reporting, uses the dummy predictor at 1 ms, seed 42, online arrivals and sequential PDD execution. Device/network identities `h800` / `h800_dgx` are simulated configuration, not physical GPU measurements. Attempts 0/2 run baseline then candidate; attempt 1 reverses order. Every invocation starts a fresh process and output directory. Reproduction requires a new output path because the driver rejects an existing directory.

## Criteria and workload

Require exactly 18 successful measurements, one baseline/candidate pair for each of three cases and three attempts, identical event counts and completed requests in each pair. Inspect Simulator.run time separately from subprocess wall time so initialization is not mistaken for simulation overhead. No fixed speedup threshold was authorized; small or noisy gains must not be promoted to statistically established improvements.

| Case | Requests / QPS | Prefill / decode tokens | Attention TP / DP | MoE TP / EP | PP | Simulated GPUs |
| --- | --- | --- | --- | --- | --- | --- |
| small_dense | 2 / 2 | 16 / 8 | 4 / 1 | 1 / 1 | 2 | 16 |
| longer_dense | 4 / 10 | 64 / 16 | 4 / 1 | 1 / 1 | 2 | 16 |
| representative_moe | 2 / 5 | 32 / 8 | 8 / 1 | 1 / 8 | 2 | 32 |

Observed: **18/18 successful measurements; 9/9 event/completion-preserving pairs PASS**. Driver exit code: 0. Its final line is `completed 18 measurements; output=/data/ycfeng/tmp/quality-p5-paired`.

## Results

Changes below are the **median of the three paired percentages**, each computed as `100 * (candidate_i / baseline_i - 1)`. They are not the percentage calculated from the two marginal medians. Negative means lower measured wall time.

| Case | Baseline Simulator.run median (s) | Candidate Simulator.run median (s) | Median paired run change | Baseline subprocess median (s) | Candidate subprocess median (s) | Median paired subprocess change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small_dense | 0.030698377 | 0.027128791 | -13.46% | 6.151787587 | 5.850992820 | -4.89% |
| longer_dense | 0.054797340 | 0.053700651 | -2.00% | 6.170368183 | 5.939244965 | -3.12% |
| representative_moe | 9.667771658 | 9.642512048 | -1.46% | 15.858523788 | 15.458201239 | -2.52% |

All 18 measurements are preserved below as nine pairs. Times are seconds, rounded only for display. Events and completions are equal on both sides of each row.

| Case | Attempt | Baseline run | Candidate run | Baseline subprocess | Candidate subprocess | Events per side | Completed per side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_dense | 0 | 0.030698377 | 0.026238187 | 6.151787587 | 5.850992820 | 176 | 2 |
| small_dense | 1 | 0.031347906 | 0.027128791 | 6.066424991 | 5.777540652 | 176 | 2 |
| small_dense | 2 | 0.030235467 | 0.028235758 | 6.757396852 | 6.422888210 | 176 | 2 |
| longer_dense | 0 | 0.053477788 | 0.052682776 | 6.081949613 | 5.939244965 | 352 | 4 |
| longer_dense | 1 | 0.055984340 | 0.053943754 | 6.170368183 | 5.977714164 | 352 | 4 |
| longer_dense | 2 | 0.054797340 | 0.053700651 | 6.297149913 | 5.882357306 | 352 | 4 |
| representative_moe | 0 | 9.667771658 | 9.658895702 | 15.858523788 | 15.458201239 | 3560 | 2 |
| representative_moe | 1 | 10.780148279 | 9.642512048 | 17.539393942 | 15.581294084 | 3560 | 2 |
| representative_moe | 2 | 9.617936392 | 9.477600404 | 15.836630076 | 15.450871536 | 3560 | 2 |

Per-attempt Simulator.run changes are small_dense `[-14.5291%, -13.4590%, -6.6138%]`, longer_dense `[-1.4866%, -3.6449%, -2.0014%]`, representative_moe `[-0.0918%, -10.5531%, -1.4591%]`. The slower attempt-1 MoE baseline is visible, not excluded or replaced.

## Interpretation and evidence limits

The measured small dense path shows lower overhead in all three pairs. Longer dense and MoE improvements are small; three pairs with visible host/run variability do not establish a material general speedup. In particular, the MoE marginal medians differ by only about 0.0253 s. No paired sample regressed, but this is not a universal performance guarantee or causal attribution to a single cleanup.

Subprocess time includes imports and initialization; these changes are not a measurement of Simulator.run alone. Earlier P1 timing is historical and is not substituted for this final-source campaign.

Reporting was disabled identically by the pre-existing benchmark on both revisions; no metrics were removed to produce a speedup. Separately, [final correctness evidence](test_report_2026-09-17_p5_final.md) verifies reporting-enabled non-dummy artifacts and 58-scenario fidelity. This benchmark does not establish metrics-enabled performance, trained-predictor runtime performance, native CUDA/ROCm/SGLang kernel accuracy or live vLLM parity.

Raw evidence: `/data/ycfeng/tmp/quality-p5-paired/{manifest,results}.json`, `/data/ycfeng/tmp/quality-p5-paired.log`, and `<case>/<side>/attempt-<N>/{case,result,measurement}.json` plus `run.log` under that output directory. The durable tables above preserve all measured run/subprocess values needed to recompute the reported paired percentages.
