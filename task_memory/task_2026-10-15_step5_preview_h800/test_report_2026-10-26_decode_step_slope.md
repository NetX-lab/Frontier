# Test report: decode step growth and native decode stalls (H200 32 GPUs)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Added the h200_32 length-sweep rerun (section 6) and the step E comparison against it; updated the limits. |
| 2026-10-27 | Created: native decode device timeline, Frontier step comparison, kernel components, just-in-time compile stalls, step E rerun. |

## Purpose

Calibration item 2b (`plan.md`): the native decode step grows about 0.7 ms per request per DP
engine, faster than Frontier's step. This report records where the native step time goes and
why the native TTFT/TPOT tails exist. All native step times and kernel times here are
diagnostic only (D31.10): they explain differences, they are never a Frontier prediction input.

## Environment

- Native: Step-5-Preview P/D, H200, P16 and D16 per role TP8/DP2/EP16, image
  `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`
  (`position_cyclic` routing), `FULL_AND_PIECEWISE` CUDA graphs, chunked prefill and prefix
  caching off, 6,144/1,024-token requests, 128 formal requests at QPS 1, 16 warmups, three
  repeats.
- Device-timeline run: job `exp-1027-062510-683917` (4 × 8 H200, `step_main`, non-preemptible,
  creator `i-fengyicheng`), stage `/data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01`
  built from `26d51a30` (worktree `.worktrees/step5-d31-native-26d51a3`). Succeeded at
  22:44:26 UTC, run time 1,156 s, replay status `CLIENT_WORKLOAD_VERIFIED`. Profiler windows
  on the decode servers during the first formal repeat: `load_t016`, `load_t040`, `load_t088`
  (seconds after the first formal dispatch), 24 traces per decode node.
- Analyzer: Python 3.10 (`/data/ycfeng/envs/frontier-runtime-py310-d29-20261025`), analyzer
  `tests/integration/step5_pd/native_decode_timeline.py` at `87cb0d58`.
- Frontier step curve: `/data/ycfeng/tmp/d31_diag/step_curve/` (Frontier `1ef82e4d`, merged04
  tables, measured H200 DP2 collectives, 84 requests at 0.2 s spacing, stage-batch and EP-wave
  ledgers on), reduced by `analyze_window.py` to `analyze_window.txt`.

## Commands

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/native_decode_timeline.py \
  /data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01/stage/evidence/nodes/*/torch_trace \
  --output /data/ycfeng/tmp/d31_diag/devtl/timeline_categories.json
# Per-kernel medians of one trace (diagnostic script, not in Git):
python3 /data/ycfeng/tmp/d31_diag/devtl/kernels_joined.py <trace.pt.trace.json.gz>
# Stall events per repeat (diagnostic script, not in Git):
python3 /data/ycfeng/tmp/d31_diag/native_stall_events.py <stage>/evidence/replay/client_requests.jsonl
# Analyzer unit test:
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest \
  tests/unit/test_native_decode_timeline.py -q -p no:cacheprovider
```

The analyzer test passed (16 passed). Its output files are
`/data/ycfeng/tmp/d31_diag/devtl/timeline_categories.json` and `summary_categories.txt`.

## Results

### 1. Native decode iterations against Frontier's step

Medians of decode-only iterations (no context request), both TP ranks of each DP engine. The
profiler adds host time to the native period (about 50 ms at 8–9 requests against 39.8 ms
without the profiler, from the vLLM CUDA-graph statistics), so the device busy time is the
comparable number.

| Requests per engine | Native period | Native device busy | Host-wait idle | Other idle | Frontier FULL step |
| --- | --- | --- | --- | --- | --- |
| 8–9 | 45–51 ms | 40.8–41.0 ms | 1.5–2.1 ms | 1.8–2.4 ms | 38.4 ms (8), 43.1 ms (9) |
| 20–21 | 55 ms | 45.5–47.9 ms | 5.8–6.7 ms | 1.9–2.2 ms | 46.4 ms |
| 44–46 | 62–65 ms | 52.8–53.7 ms | 7.1–8.1 ms | 2.3–2.5 ms | 49.9 ms at 32–36 (the curve ends at 36) |

Frontier pads to CUDA Graph capture sizes, so its step rises in stairs (9–16 requests share
43.1 ms). Expectation: Frontier's step is within 10% of the native device busy time. Observed:
+5% at 9 requests, −3% to +2% at 20–21, −6% to −7% at 44–46 against Frontier's last point. PASS for
the device part. The native host-wait idle time grows from about 2 ms to 8 ms and is not
modelled (item 3, CPU overhead).

### 2. Native device time per kernel category

Median ms per decode step (`87cb0d58` categories), DP engine 0 / 1:

| Requests | Routed expert | EP collective | TP all-reduce | Dense GEMM | Attention | DSA indexer | Router | FFN activation | Norm | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 10.91 / 11.05 | 14.41 / 13.79 | 6.63 / 6.86 | 3.20 / 3.17 | 1.89 / 1.87 | 0.57 / 0.86 | 1.10 / 1.10 | 0.76 / 0.76 | 0.39 / 0.38 | 2.09 / 2.10 |
| 20 | 13.04 / 12.57 | 16.10 / 16.84 | 7.13 / 7.15 | 3.20 / 3.19 | 2.29 / 2.27 | 0.62 / 0.90 | 0.81 / 0.84 | 0.83 / 0.83 | 0.41 / 0.40 | 2.41 / 2.40 |
| 45 | 15.06 / 16.13 | 16.42 / 16.04 | 8.51 / 8.30 | 4.99 / 4.95 | 3.50 / 3.47 | 0.79 / 1.04 | 0.00 / 0.00 | 0.93 / 0.93 | 0.43 / 0.42 | 2.81 / 2.78 |

At 44–46 requests no `fp32_router_gemm_kernel` runs, and the dense GEMM column grows by about
1.8 ms; probably the router GEMM runs as a cuBLAS kernel there (not checked kernel by kernel). Every `deep_gemm` call in these traces is a grouped GEMM over the 22
local experts.

### 3. Components at about 9 requests per engine

| Component | Native | Frontier (EP-wave ledger, 9 requests) | Difference |
| --- | --- | --- | --- |
| EP dispatch (all-gather) | 9.1 ms | 9.18 ms | +1% |
| EP combine (reduce-scatter) | 4.9 ms | 5.84 ms | +19% |
| Routed expert compute | 10.9–11.1 ms | 16.9 ms | about +53% |
| TP all-reduce | 6.1 ms (fusion kernel), 6.6–6.9 ms (category) | about 2.6–3.1 ms | about −55% |

Frontier's routed compute stays at 16.9 ms from 2 to 16 requests and 18.5 ms from 17 to 36,
while native grows from 10.9 to 15–16 ms. Inference, not yet measured: the MoE profile routes
tokens so that every local expert is active, while a small native batch activates only a few
of the 22 local experts. The two errors (routed compute too high, TP all-reduce too low) partly
cancel. Both are op-profiling fixes (item 2).

### 4. Native decode stalls from just-in-time kernel compiles

A decode step that also carries a new request's 16-token prompt tail is planned by
`Step5DSAMetadataBuilder._plan_prefill`, which launches the DSA scoring kernel
`_paged_scores_sm90` with `REGIONS = (longest sequence + 7) // 8`. `REGIONS` is a compile-time
constant, so decode lengths 6,144–7,168 need 129 variants (768–896). The native warm-up compiles
only 513, and the first mixed step at each new longest length compiles one variant (about
0.43 s) while the engine's requests wait.

Stall events: gaps over 150 ms between two tokens of a formal request that is already decoding,
merged across requests.

| Run | Repeat | Events | Ending within 50 ms of a first token | Median | Total |
| --- | --- | --- | --- | --- | --- |
| h200_32 workload02 | 1 | 84 | 84 | 424 ms | 38.2 s |
| h200_32 workload02 | 2 | 54 | 52 | 430 ms | 22.8 s |
| h200_32 workload02 | 3 | 34 | 32 | 424 ms | 14.7 s |
| h200_64 workload01 | 1 | 106 | 106 | 436 ms | 47.1 s |
| h200_64 workload01 | 2 | 77 | 75 | 435 ms | 33.3 s |
| h200_64 workload01 | 3 | 53 | 53 | 431 ms | 23.2 s |
| h200_32 device timeline | 1 | 86 | 83 | 406 ms | 81.5 s |
| h200_32 device timeline | 2 | 58 | 55 | 385 ms | 22.1 s |
| h200_32 device timeline | 3 | 32 | 30 | 381 ms | 12.3 s |

The count falls from repeat to repeat because each repeat compiles the variants it reaches and
later repeats reach fewer new ones. The per-repeat native means fall with it: h200_32 TPOT
67.69 → 60.38 → 57.16 ms and TTFT 1,171 → 1,071 → 1,008 ms; h200_64 TPOT 74.97 → 68.84 →
65.1 ms and TTFT 1,462 → 1,376 → 1,306 ms.

Inference, not verified: a linear fit of the per-repeat means against the stall time gives
about 50.6 ms TPOT and 906 ms TTFT for h200_32 without stalls, and 55.5 ms and 1,154 ms for
h200_64. The native rerun with the length sweep (`320c2082`) measures this directly.

### 5. Step E rerun against the current native reference

`h200_32_comb_h200` (workflow port, measured H200 DP2 collectives, merged04), output
`/data/ycfeng/tmp/d31_cvfix_run/h200_32_comb_h200/compare.json`, gate 30%:

| Metric | Frontier | Native | Relative error | Result |
| --- | --- | --- | --- | --- |
| TTFT mean | 863.9 ms | 1,083.7 ms | −20.3% | PASS |
| TTFT p50 | 868.7 ms | 1,042.3 ms | −16.7% | PASS |
| TTFT p90 | 892.9 ms | 1,307.2 ms | −31.7% | FAIL |
| TTFT p95 | 895.8 ms | 1,359.9 ms | −34.1% | FAIL |
| TPOT mean | 45.91 ms | 61.75 ms | −25.7% | PASS |
| TPOT p50 | 46.67 ms | 61.70 ms | −24.4% | PASS |
| TPOT p90 | 46.81 ms | 71.02 ms | −34.1% | FAIL |
| TPOT p95 | 46.85 ms | 71.40 ms | −34.4% | FAIL |
| E2E mean | 47.83 s | 64.25 s | −25.6% | PASS |
| E2E p90 | 48.78 s | 73.81 s | −33.9% | FAIL |
| Total throughput | 5,372.1 tok/s | 5,214.6 tok/s | +3.0% | PASS |
| Output throughput | 767.4 tok/s | 744.9 tok/s | +3.0% | PASS |

All four means pass; the p90/p95 rows fail. The native tails include the compile stalls above.

### 6. Native rerun with the length sweep (h200_32)

Run: job `exp-1027-074903-556239` (4 × 8 H200, `step_main`, non-preemptible, creator
`i-fengyicheng`), stage `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_sweep02` built
from `320c2082` (worktree `.worktrees/step5-d31-native-320c208`). Length sweep settings in the run
definition (`canonical_replay.length_sweep`): two long requests per DP lane, 0.1 s between lanes,
short 64/16-token requests, `duration_s` 240. Succeeded at 00:14:20 UTC, run time 1,517 s, replay
`CLIENT_WORKLOAD_VERIFIED`, 3 × (16 + 128) requests recorded, none failed.

```bash
python3 /data/ycfeng/tmp/d31_diag/native_stall_events.py \
  /data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_sweep02/stage/evidence/replay/client_requests.jsonl
cd /data/ycfeng/Frontier/.worktrees/step5-preview
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/workload_metrics.py \
  --stage /data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_sweep02/stage \
  --output /data/ycfeng/tmp/d31_sweep02/native_metrics.json
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/frontier_workload.py compare \
  --frontier /data/ycfeng/tmp/d31_cvfix_run/h200_32_comb_h200/frontier_summary.json \
  --native /data/ycfeng/tmp/d31_sweep02/native_metrics.json --gate 0.3 \
  --output /data/ycfeng/tmp/d31_sweep02/compare_E.json
```

Length sweep: 4,804 requests recorded, one short request failed (HTTP 502). The sweep stopped at
its 240 s limit while the two long requests per lane were still running (they ended at 252–253 s),
so `coverage_verified` is false. The long requests were slow because almost every early step
compiled a variant (median token gap about 410 ms for the first 256 tokens). They reached
lengths 6,803–6,849 (region counts 851–857) before the short requests stopped.

| Repeat | Stall events (before the sweep) | Total stall time | TTFT mean | TPOT mean |
| --- | --- | --- | --- | --- |
| 1 | 38 (84) | 15.0 s | 1,005.8 ms | 58.74 ms (67.69) |
| 2 | 21 (54) | 8.4 s | 961.9 ms | 54.98 ms (60.38) |
| 3 | 22 (34) | 9.1 s | 968.3 ms | 55.22 ms (57.16) |

79 of the 81 remaining stall events have region counts 857–896, the range the sweep did not
reach. Repeats 2 and 3 agree; repeat 1 is still 3.7 ms higher in TPOT.

Expectations and results:
- `workload_metrics.py` passes: PASS.
- No or only a few stall events in the formal repeats: FAIL (81 events, almost all above the
  length the sweep reached).
- Sweep coverage verified: FAIL (the sweep ended before the long requests).

Step E (`h200_32_comb_h200`, unchanged) against this run, gate 30%,
`/data/ycfeng/tmp/d31_sweep02/compare_E.json`:

| Metric | Frontier | Native | Relative error | Result |
| --- | --- | --- | --- | --- |
| TTFT mean | 863.9 ms | 978.7 ms | −11.7% | PASS |
| TTFT p50 | 868.7 ms | 929.7 ms | −6.6% | PASS |
| TTFT p90 | 892.9 ms | 1,249.2 ms | −28.5% | PASS |
| TTFT p95 | 895.8 ms | 1,269.5 ms | −29.4% | PASS |
| TPOT mean | 45.91 ms | 56.31 ms | −18.5% | PASS |
| TPOT p50 | 46.67 ms | 56.23 ms | −17.0% | PASS |
| TPOT p90 | 46.81 ms | 61.99 ms | −24.5% | PASS |
| TPOT p95 | 46.85 ms | 62.71 ms | −25.3% | PASS |
| E2E mean | 47.83 s | 58.59 s | −18.4% | PASS |
| E2E p50 | 48.61 s | 58.45 s | −16.8% | PASS |
| E2E p90 | 48.78 s | 64.49 s | −24.4% | PASS |
| E2E p95 | 48.80 s | 65.19 s | −25.1% | PASS |
| Total throughput | 5,372.1 tok/s | 5,236.6 tok/s | +2.6% | PASS |
| Output throughput | 767.4 tok/s | 748.1 tok/s | +2.6% | PASS |

All 14 rows pass. The native reference still contains the residual stalls, so these errors are
not final. Next: `duration_s` 420 s, h200_64 first (job `exp-1027-082338-716026`), then h200_32.

## Limits

- The torch profiler perturbs host timing (period about 50 ms against 39.8 ms without it), so
  native periods and host-wait times in the profiled windows are upper bounds; device busy times
  are the comparable numbers.
- Profiled windows cover the first formal repeat only, which has the most compile stalls; mixed
  iterations are excluded from the decode-only medians.
- Category assignment is by kernel-name fragments checked against the Step decode traces; the
  `other` column holds generic PyTorch elementwise kernels (router top-k selection, gating
  arithmetic, sampling) that cannot be assigned by name.
- The length-sweep rerun (section 6) still has 81 stall events, so a native run without
  compile stalls has not been measured yet. The linear fit in section 4 remains an inference.
