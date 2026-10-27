# Test report: decode step growth and native decode stalls (H200 32 and 64 GPUs)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Added section 10.1: the critical path without double-counted overlapping kernels (router matches; idle shortfall about 2.4 ms; position-routing kernels, shuffling counted twice, no L2 flush in the linear-op and MoE profilers). |
| 2026-10-27 | Added the exact critical path of native decode steps (section 10): the step prefix before the first TP all-reduce is the KV side-storage reset (item 3); Frontier's step is within −7% to +4% of the native forward body; corrected section 3's TP all-reduce direction (Frontier is about 1.3 ms high, not low). |
| 2026-10-27 | Corrected the table version of `h200_32_comb_h200` to merged03 (its `command.json`). |
| 2026-10-27 | Added the 780 s h200_64 rerun with step G against it (section 9); corrected the reset count to 23 sparse-layer hooks with 46 indexed writes (section 8). |
| 2026-10-27 | Added the h200_64 length-sweep run with step G against it (section 7) and the KV side-storage reset sync as the host-time cause (section 8); updated the limits. |
| 2026-10-27 | Corrected section 3 with the per-rank check: the median-rank values overstated both operator errors. |
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
| Routed expert compute | 10.9–11.1 ms (median rank); 13.8–14.4 ms with SwiGLU on the slowest routed rank | 16.9 ms | about +2.5–3 ms per step against the slowest routed rank |
| TP all-reduce | 6.1 ms (fusion kernel), 6.6–6.9 ms (category), median rank; 3.4–4.4 ms smallest rank | about 2.6–3.1 ms | about 1–2 ms per step low |

Frontier's routed compute stays at 16.9 ms from 2 to 16 requests and 18.5 ms from 17 to 36,
while native grows from 10.9 to 15–16 ms. Inference, not yet measured: the MoE profile routes
tokens so that every local expert is active, while a small native batch activates only a few
of the 22 local experts. Both are op-profiling fixes (item 2).

Correction 2 (2026-10-27, exact critical path, section 10): the per-rank TP all-reduce times
above include waiting for the other ranks. The all-reduce cost on the critical path is 1.85 ms
at 9 requests (273 calls per step, about 6.8 µs each), so Frontier's 3.14 ms (272 calls, about
11.5 µs each) is about 1.3 ms high, not low.

Correction (2026-10-27, per-rank check in `progress.md`): within one DP engine the fast ranks
wait inside the TP all-reduce and EP collectives for the slowest routed rank, so the per-rank
medians first used here understate the routed time that gates the step and overstate the
all-reduce. Against the slowest routed rank, Frontier's routed compute is about 2.5–3 ms per
step high and the TP all-reduce about 1–2 ms low. Device busy time on that rank is 41.0 ms at
9 requests, 45.1–48.0 ms at 20–21 and 52.6–52.9 ms at 44–45, against Frontier's 43.1, 46.4 and
49.9 ms (36 requests): within about 5%.

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

`h200_32_comb_h200` (workflow port, measured H200 DP2 collectives, merged03 tables), output
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

### 7. Native rerun with the length sweep (h200_64) and step G

Run: job `exp-1027-082338-716026` (8 × 8 H200, `step_main`, non-preemptible, creator
`i-fengyicheng`), stage `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_sweep01`, built
from `320c2082`, length sweep as in section 6 with `duration_s` 420 (two long requests per lane
on four lanes, short requests every 0.025 s). Succeeded at 00:53:49 UTC, run time 1,811 s,
replay `CLIENT_WORKLOAD_VERIFIED`, 3 × (16 + 128) requests recorded, none failed.

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
WS=/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_sweep01
python3 /data/ycfeng/tmp/d31_diag/stall_regions.py $WS/stage/evidence/replay/client_requests.jsonl
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/workload_metrics.py --stage $WS/stage \
  --output /data/ycfeng/tmp/d31_sweep_h200_64/native_metrics.json
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/frontier_workload.py compare \
  --frontier /data/ycfeng/tmp/d31_merged04_run/h200_64_g01/frontier_summary.json \
  --native /data/ycfeng/tmp/d31_sweep_h200_64/native_metrics.json --gate 0.3 \
  --output /data/ycfeng/tmp/d31_sweep_h200_64/compare_G_sweep.json
```

Length sweep: 16,808 requests recorded, 9 short requests failed (HTTP 502), `coverage_verified`
false. The long requests ran to 448 s; at the last short dispatch (420 s) they had length
6,695–6,712 (region count 837–839). Tokens 256–640 took about 220 s (about 0.57 s per token,
compiling); tokens 640–1,024 took 14 s, after the short requests had stopped.

| Repeat | Stall events (before the sweep) | Total stall time (before) | TTFT mean | TPOT mean (before) |
| --- | --- | --- | --- | --- |
| 1 | 69 (106) | 27.3 s (47.1 s) | 1,336.2 ms | 69.3 ms (75.0) |
| 2 | 37 (77) | 14.9 s (33.3 s) | 1,257.6 ms | 64.1 ms (68.8) |
| 3 | 27 (53) | 11.1 s (23.2 s) | 1,242.4 ms | 63.2 ms (65.1) |

Every remaining stall event has a region count of 847–896, above the sweep's reach. On the
h200_32 sweep (section 6) the same script finds 80 of 81 events above that sweep's reach.

Expectations and results:
- `workload_metrics.py` passes: PASS.
- No or only a few stall events in the formal repeats: FAIL (133 events, all above the length
  the sweep reached).
- Sweep coverage verified: FAIL.

Step G (`h200_64_g01`, merged04, measured H200 DP4 collectives) against this run, gate 30%:

| Metric | Frontier | Native | Relative error | Result |
| --- | --- | --- | --- | --- |
| TTFT mean | 973.3 ms | 1,278.7 ms | −23.9% | PASS |
| TTFT p50 | 984.5 ms | 1,269.3 ms | −22.4% | PASS |
| TTFT p90 | 1,013.0 ms | 1,573.5 ms | −35.6% | FAIL |
| TTFT p95 | 1,016.3 ms | 1,629.4 ms | −37.6% | FAIL |
| TPOT mean | 49.51 ms | 65.54 ms | −24.5% | PASS |
| TPOT p50 | 51.12 ms | 64.57 ms | −20.8% | PASS |
| TPOT p90 | 51.36 ms | 72.76 ms | −29.4% | PASS |
| TPOT p95 | 51.36 ms | 73.48 ms | −30.1% | FAIL |
| E2E mean | 51.62 s | 68.32 s | −24.5% | PASS |
| E2E p50 | 53.26 s | 67.33 s | −20.9% | PASS |
| E2E p90 | 53.53 s | 75.74 s | −29.3% | PASS |
| E2E p95 | 53.54 s | 76.44 s | −30.0% | PASS |
| Total throughput | 5,316.6 tok/s | 4,983.6 tok/s | +6.7% | PASS |
| Output throughput | 759.5 tok/s | 711.9 tok/s | +6.7% | PASS |

11 of 14 rows pass. Against the pre-sweep h200_64 reference 8 of 14 pass (TPOT mean −28.9%,
`compare_G_pre.json`). The G run log has the BF16 → FP8 "approximation factor=0.500" warnings;
they change no Step-5 time (the MoE predictor never applies the factor, see `progress.md`).

The sweep needs short requests until the long requests end. At sweep01's pace the remaining 48
region counts take about 220 s more, so the rerun (`h200_64_sweep02`, job submitted on
2026-10-27 01:14 UTC) uses `duration_s` 780.

### 8. Host time between decode steps: KV side-storage reset sync (diagnostic)

Scripts (not in Git): `/data/ycfeng/tmp/d31_diag/hostgap/zeroing_split.py`, `critical_path.py`,
`patterns.py`, `loop_count.py`; input: the h200_32 device-timeline traces of section 1.

```bash
python3 /data/ycfeng/tmp/d31_diag/hostgap/zeroing_split.py \
  /data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01/stage/evidence/nodes/*/torch_trace/load_t0*_tp0_*.pt.trace.json.gz
```

In the native source (`/data/ycfeng/tmp/d29_published_sources01/source`), every step that
allocates a new KV block sets `new_block_ids_to_zero` (`vllm/v1/core/sched/scheduler.py:1622-1650`).
The model runner (`vllm/v1/worker/gpu/model_runner.py:1250-1297`) then calls
`KVBlockZeroer.zero_block_ids` (`vllm/v1/worker/utils.py:519-598`), which calls one reset hook for
each of the 23 sparse-attention layers (the model has 69 sliding-window and 23 sparse layers).
Each hook (`vllm/models/step5/nvidia/sparse_summary_cache.py:486-516`) copies the GPU block-ID
tensor to the host with `.tolist()`, a blocking copy, and then writes two side-storage tensors
with Python-list indexing (`:353-355`), so a reset step runs 46 indexed writes (the 46
`aten::index_put_` calls in the trace, each with one `cudaStreamSynchronize`). The first copy
waits for the previous step's device work on the main stream. Under async scheduling the host otherwise prepares the next
step while the device runs, so a reset step leaves the device idle while the host runs the
resets, the rest of the input preparation, the gloo DP all-reduce and the graph launch.

Decode-only profiled steps, split into steps with that sync (a long `aten::to` before the DP
all-reduce) and without:

| Requests per engine | Class | Steps | Period (ms) | Device busy (ms) | Host wait (ms) | Sync → all-reduce (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | async | 36 | 39.87 | 40.75 | 0.00 | – |
| 5 | sync | 16 | 54.99 | 41.91 | 7.05 | 4.36 |
| 20 | async | 23 | 46.57 | 44.87 | 1.39 | – |
| 20 | sync | 33 | 58.75 | 48.90 | 6.98 | 5.02 |
| 40 | sync | 12 | 63.67 | 52.56 | 9.64 | 6.02 |
| 45 | sync | 36 | 62.70 | 52.58 | 8.03 | 5.29 |

From the end of the sync to the first graph kernel: about 8.7–9.7 ms, nearly constant with the
request count (`critical_path.py`). Frontier's step at 20 requests (46.4 ms) matches the async
steps. The profiled mix at 20 requests (about 53.8 ms) matches the unprofiled native token gaps
(52.6–53.6 ms, `/data/ycfeng/tmp/d31_diag/native_decode_gaps_h200_32.json`).

Expectations and results: the remaining TPOT gap should come from host time on the step's
critical path: CONFIRMED for a mechanism (the reset sync), with profiled host times as upper
bounds. An unprofiled measurement of the host terms is still needed before Frontier can price
them (item 3 design).

### 9. Native rerun with the 780 s length sweep (h200_64) and step G

Run: job `exp-1027-091511-064923` (8 × 8 H200, `step_main`, non-preemptible), stage
`/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_sweep02`, built from `320c2082`; the run
definition differs from section 7 only in `duration_s` 780. Succeeded at 01:49:49 UTC; replay
`CLIENT_WORKLOAD_VERIFIED`, 3 × (16 + 128) requests recorded, none failed.

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
WS=/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_sweep02
python3 /data/ycfeng/tmp/d31_diag/stall_regions.py $WS/stage/evidence/replay/client_requests.jsonl
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/workload_metrics.py --stage $WS/stage \
  --output /data/ycfeng/tmp/d31_sweep_h200_64_02/native_metrics.json
PYTHONPATH=$PWD:$PWD/tests/comparison/dp_placement_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/frontier_workload.py compare \
  --frontier /data/ycfeng/tmp/d31_merged04_run/h200_64_g01/frontier_summary.json \
  --native /data/ycfeng/tmp/d31_sweep_h200_64_02/native_metrics.json --gate 0.3 \
  --output /data/ycfeng/tmp/d31_sweep_h200_64_02/compare_G.json
```

- Length sweep: 31,208 requests recorded, 36 short requests failed, `coverage_verified=true`.
  The long requests reached their full length (7,167 tokens, region count 896) before the last
  short dispatch.
- Formal stall events: 1 / 1 / 1 per repeat (section 7: 69 / 37 / 27), at region counts 887,
  803 and 838, all within the sweep's reach.
- Per-repeat means: TPOT 61.45 / 60.95 / 61.37 ms, TTFT 1,185 / 1,176 / 1,178 ms. The repeats
  agree within 1%, with no falling trend.

Step G (`h200_64_g01`, unchanged) against this reference
(`evidence/d31_h200_64_sweep02_compare_G.json`, gate 30%):

| Metric | Frontier | Native | Abs. error | Rel. error | Status |
| --- | --- | --- | --- | --- | --- |
| TTFT mean (ms) | 973.3 | 1,179.9 | −206.6 | −17.5% | PASS |
| TTFT p95 (ms) | 1,016.3 | 1,367.0 | −350.7 | −25.7% | PASS |
| TPOT mean (ms) | 49.51 | 61.26 | −11.75 | −19.2% | PASS |
| TPOT p95 (ms) | 51.36 | 63.75 | −12.39 | −19.4% | PASS |
| E2E mean (ms) | 51,618 | 63,847 | −12,229 | −19.2% | PASS |
| Total throughput (tokens/s) | 5,316.6 | 4,994.5 | +322.1 | +6.4% | PASS |

All 14 rows pass (the JSON holds p50/p90 and output throughput as well). Frontier is low on
both TTFT and TPOT. Expectation: with compile stalls removed, the remaining TPOT gap should be
the host time of section 8 plus the operator errors of item 2. CONFIRMED in direction; the split
between the two needs the item 3 probe.

### 10. Exact critical path of native decode steps (diagnostic)

Diagnostic only (D31.10). Scripts, not in Git: `/data/ycfeng/tmp/d31_diag/critpath/`
(`extract.py`, `critpath4.py`, `prefix.py`, `segments.py`, `gap_sites.py`), input the
device-timeline traces of section 1 (`load_t016`, `load_t040`, `load_t088`).

Method. Within one DP engine the 8 TP ranks leave each TP all-reduce at nearly the same time
(median end spread 0.8–1.3 µs). The rank that arrives last at a call (the shortest call duration)
gates it, so its kernels and idle time between the previous call's end and its own start are the
critical segment. EP dispatch (all-gather) and combine (reduce-scatter) pair each rank with the
same TP rank of the other engine; their time is split into own cost (the shorter duration of the
pair) and waiting for the peer. The parts add up exactly to the step span. The window before the
first TP all-reduce (the embedding all-reduce) is reported separately as the step prefix.

Step prefix (rank 0 of each engine, one profiled window each):

| Window | Steps that reset KV side storage | Prefix of a resetting step | Other steps |
| --- | --- | --- | --- |
| `load_t016`, 9 requests (`dp0`, `dp1`) | 11 of 26, 8 of 27 | 6.1–7.6 ms (medians 7.5 and 6.4 ms), 177 kernels including 23 device-to-host copies | 0.07 ms, 13 kernels; after a peer engine's reset: 4.6–7.6 ms idle before the step plus 1.0–1.3 ms prefix |
| `load_t040`, 20 requests (`dp1`) | 15 of 27 | 7.8 ms (median) | as above |
| `load_t088`, 45 requests (`dp0`) | 24 of 26 | 7.8 ms (median) | as above |

The 23 device-to-host copies are the 23 `.tolist()` calls of the sparse-layer reset hooks
(item 3, `design.md` section 2). The resetting step's first kernel starts right after the
previous step's last kernel (median −5 to −9 µs), as the first `.tolist()` waits for the previous
step's device work. The other engine idles until its gloo all-reduce returns, then runs its own
input preparation (1.0–1.3 ms). This is the item 3 host time; the torch profiler inflates it.

Forward body (step span minus prefix), per engine, against Frontier's step curve
(`/data/ycfeng/tmp/d31_diag/step_curve/analyze_window.txt`, merged04, H200 DP2 curves):

| Requests per engine | Native body | Frontier step | Difference |
| --- | --- | --- | --- |
| 9 | 41.5–42.3 ms | 43.14 ms | +2% to +4% |
| 20–21 | 46.2–49.8 ms (three of four lanes 46.2–46.8) | 46.32–46.37 ms | −7% to 0% |
| 44–46 | 53.1–53.8 ms | 49.82 ms at 36 (curve ends there) | about −7% |

Components at 9 requests per engine (native critical path of `dp0`/`dp1`; Frontier from the
EP-wave and stage ledgers):

| Component | Native | Frontier | Difference |
| --- | --- | --- | --- |
| Routed experts (grouped GEMM and SwiGLU, plus waiting for the peer in combine) | 12.2–13.4 ms | 16.90 ms | +3.5 to +4.7 ms |
| TP all-reduce (own cost) | 1.85–1.88 ms, 273 calls | 3.14 ms, 272 calls | +1.3 ms |
| EP dispatch and combine with peer waits | 14.6–15.3 ms | 15.02 ms | about 0 |
| Idle time inside the body | 1.8–1.85 ms (plus 0.3 ms tail and end skew) | 0.62 ms (kernel-gap proxy) | −1.2 to −1.5 ms |
| Other compute (attention, DSA indexer, dense GEMMs, norms, router, elementwise) | 9.3 ms | about 7.5 ms (residual) | about −1.8 to −2.6 ms |

Idle time inside the body per segment (rank `dp*/tp0`, per step; identical counts at 9, 20 and
45 requests): attention segments 92 × 12.3 kernels, 0.41–0.63 ms idle (0.36–0.56 µs per
kernel); MoE segments (gate to combine) 88 × 29–31 kernels, 1.5–1.8 ms idle (0.6–0.7 µs per
kernel, including about 6 µs before each EP collective launch); other segments 92 × 5 kernels,
0.14–0.18 ms idle. In total about 4,100–4,300 kernels per body. The proxy's 0.43 µs per graph
kernel matches the native median gap (0.42 µs); Frontier's 0.62 ms of gap corresponds to about
1,400 counted kernels, so the shortfall is the kernel count, not the gap size.

Inference (not yet tested by a Frontier change): the errors cancel to within −7% to +4% of the
native body, so the remaining TPOT gap is the step prefix (item 3). Fixing only the routed-expert
overpricing would leave Frontier about 8% low at 9 requests; the item 2 fixes (routed experts
from the native position routing, the TP all-reduce own cost, the kernel count of graph steps,
the other-compute shortfall) have to land together.

#### 10.1 Correction: overlapping kernels (critpath5)

`critpath4.py` added the full duration of every kernel on the critical rank, but some kernels
start before the previous kernel on the same stream has ended (the router's kernels and some
attention kernels). The overlapped part was counted twice, once in each kernel. `critpath5.py`
counts only the part of a kernel after the stream's previous end and reports the overlap in
`overlap_*` columns (`/data/ycfeng/tmp/d31_diag/critpath/crit5_load_t016.txt`, `load_t016`
window, 9 requests per engine, `dp0` 20 steps and `dp1` 14 steps):

| Component | Native (critpath5) | Frontier | Difference |
| --- | --- | --- | --- |
| Routed experts (grouped GEMM, plus FFN activation 0.77 ms) | 11.38–12.58 ms + 0.77 ms | 16.90 ms (includes `moe_shuffling`) | about +3 to +4 ms after the shuffling row below |
| `moe_shuffling` | inside the routed-expert kernels | about 0.65 ms | +0.65 ms, counted twice |
| TP all-reduce (own cost) | 1.70–1.72 ms | 3.14 ms | about +1.4 ms |
| Router | 0.52–0.53 ms (overlap 0.57–0.59 ms not counted) | | |
| Position routing (inside "other" 1.99 ms) | about 1.5 ms | about 0.6 ms (`topkGating` rows) | about −0.9 ms |
| Attention, DSA indexer, dense GEMMs, norms | 1.75–1.77, 0.57–0.86, 3.27, 0.39 ms | | about −0.9 ms in total |
| EP dispatch with peer wait | 8.80–8.99 + 0.30–0.71 ms | | about 0 together with combine |
| EP combine with peer wait | 4.67–4.74 + 0.38–0.83 ms | | |
| Idle inside the body (gap, tail, end skew) | 2.72 + 0.33 ms | 0.62 ms | about −2.4 ms |

What changes against the table above:

- The earlier reading "Frontier underprices the router" came from the overlap; the router
  itself matches.
- Native idle time inside the body is 2.7–3.1 ms, not 1.8 ms; Frontier's shortfall is about
  2.4 ms. The per-kernel gap (0.42 µs native, 0.43 µs in the proxy) is right; Frontier counts
  about 1,400 of the 4,100–4,300 native kernels. Open question: whether the about 6 µs idle
  before each EP collective launch is already inside the measured collective cost.
- Native routing for this workload is `position_cyclic`: 14 kernels and about 17 µs per layer,
  against 6.5–8 µs for the `topkGating` kernels that Frontier prices. No H200 `position_cyclic`
  routing rows exist yet.
- `moe_shuffling` is priced on top of a routed-expert measurement (`moe_native_experts_class`)
  whose kernels already include the alignment (`moe_align_block_size`, count and sort;
  `/data/ycfeng/tmp/d31_diag/kmap/profile_shuffle_dec03.txt`, about 8.6 µs per layer).
- The linear-op and MoE profilers do not flush the L2 cache between repeats; only the attention
  profiler does (`frontier/profiling/attention/attention_wrapper.py:116-119,210`). At 16 tokens
  the cached rows are 3–20% low (for example `mlp_up_proj` 10.74 µs against 12.1–13.4 µs native).

Estimate (inference, not yet a Frontier run): with all item 2b fixes Frontier's body at 9
requests would be about 41.0 ms against 41.5–42.3 ms native; without the kernel-count fix about
9% low. The fixes and their order are in `plan.md` item 2b.

## Limits

- The torch profiler perturbs host timing (period about 50 ms against 39.8 ms without it), so
  native periods and host-wait times in the profiled windows are upper bounds; device busy times
  are the comparable numbers.
- Profiled windows cover the first formal repeat only, which has the most compile stalls; mixed
  iterations are excluded from the decode-only medians.
- Category assignment is by kernel-name fragments checked against the Step decode traces; the
  `other` column holds generic PyTorch elementwise kernels (router top-k selection, gating
  arithmetic, sampling) that cannot be assigned by name.
- The length-sweep reruns of sections 6 and 7 still had 81 and 133 stall events. The 780 s
  h200_64 rerun (section 9) has one per repeat; the h200_32 rerun with 420 s is pending. The
  linear fit in section 4 remains an inference.
- Section 8's host times come from profiled steps; the profiler inflates host time. They show
  the mechanism, not the size Frontier should price.
