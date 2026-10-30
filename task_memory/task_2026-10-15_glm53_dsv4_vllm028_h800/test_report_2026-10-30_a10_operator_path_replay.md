# Test report: A10 operator-path E2E replay of P2-32 and P2-64 on H200 (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Section 2.2: P2-32 cold finished; A9 cold against warm is EQUAL. |
| 2026-10-30 | Section 2.2: P2-32 warm reading (PASS; TTFT −8.50%, TPOT, E2E and throughput within 0.13%). Section 3.6: observation 8 (P2-32 remainder) and the evidence for `nixl_read_completion=engine_step` in observation 5; the ingress reading now includes its P2-32 effect. |
| 2026-10-30 | Added section 3.6: after the prefill side, the native P2-64 remainder is 165.2 ms once the first-contact NIXL handshakes are left out (Frontier 138.6 ms); without the five handshake requests the reference TTFT is 720.6 ms instead of 741.2 ms. Sections 3.1 and 3.5 point to the revised values. |
| 2026-10-30 | Added section 3.5 from the instrumented wave run `exp-1030-144848-164721`: the native host issues DP dummy forwards ahead of the GPU, up to about 19 unfinished forwards, so a DP wave ends with about 18 dummies (about 440 ms of GPU work) still queued and the next prefill waits behind them. Section 3.3's reading is replaced. |
| 2026-10-30 | Added section 3: the P2-64 TTFT gap is on the native prefill side, where a prefill that arrives after the previous DP wave ended finishes about 450-540 ms after that wave end (mechanism not found in the code). |
| 2026-10-30 | Created: P2-64 client reading on the pre-supplement table (probe `p2_64_c`), the loss of the promoted-table runs to the full disk (I-003), and their restart. |

## Scope

A10 replays the two H200 P2 cases on the V4 operator path (D-OPERATOR-LEVEL-ROUTE, ADR 0002: no whole-step cost table) and reads the client metrics against the persisted native vLLM reference with `tests/comparison/e2e_gate/run_gate.py`. Each case runs twice: cold (empty predictor cache, the simulator trains every model) and warm (cache from a cold run, no fitting).

- Driver: `h3o_operator_costs/a10/a10_driver.py <case> <root>`, launched by `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh <case> <name>` (root `stage4/a10/<name>`, `ulimit -v 62914560`, `timeout 86400`).
- Gate: `h3o_operator_costs/a10/a10_gate.py <case> <frontier run root> <output dir>` (the run root is the directory that contains `metrics/`; the output directory must not exist), run from the `a10/` directory with `PYTHONPATH=$REPO:$A10:$REPO/tests/analysis`.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1.
- Gate rule: client gate, relative error limit 0.10 per metric. TTFT semantic `request_issue_to_first_output_token`. Native reference: three repeats of 128 requests (`vllm/dsv4-h200-p2-64-canonical-20261025-01-repeat01..03`). Routing status MATCH (R0 and A13 (b)).

## 1. P2-64 on the pre-supplement table (probe `p2_64_c`)

This run is not the official A10 reading. It used the attention table committed before the G1h supplement (main `5561d762`) and ran without the B21 observer. P2-64 had no replay coverage misses on that table, so the run completed.

- Command: `stage4/run_probe_c.sh p2_64 p2_64_c` (root `stage4/probe/p2_64_c`).
- Receipt: `status FINISHED`, `init_wall_s 606.4`, `run_wall_s 20,391.7` (5.66 h).
- Gate output: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10_gate/p2_64_c_old_table/` (`matrix.json`, `reading.json`, `rows/`); copied from the original location under `/tmp`.

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 741.216 ms | 417.061 ms | −324.155 ms | 43.73% | 10% | FAIL |
| TPOT mean | 25.871 ms | 25.299 ms | −0.572 ms | 2.21% | 10% | PASS |
| Request E2E mean | 27,206.770 ms | 26,297.824 ms | −908.946 ms | 3.34% | 10% | PASS |
| Token throughput (window rate) | 5,963.206 tps | 5,985.814 tps | +22.608 tps | 0.38% | 10% | PASS |

Per native repeat (relative error of the same Frontier value):

| Repeat | TTFT native | TTFT error | TPOT native | TPOT error | E2E native | E2E error | Throughput native | Throughput error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat01 | 779.89 ms | 46.52% | 25.43 ms | 0.52% | 26,794.64 ms | 1.85% | 5,948.03 tps | 0.64% |
| repeat02 | 725.49 ms | 42.51% | 26.26 ms | 3.64% | 27,584.86 ms | 4.67% | 5,976.59 tps | 0.15% |
| repeat03 | 718.27 ms | 41.94% | 25.93 ms | 2.42% | 27,240.80 ms | 3.46% | 5,965.07 tps | 0.35% |

Reading: overall FAIL, on TTFT only. The native TTFT spread across repeats is 718-780 ms, so the Frontier value of 417 ms lies far outside the native run-to-run variation. Frontier predicts the first token too early by about 324 ms on average; the decode-side metrics agree within 5%. Section 3 diagnoses the cause.

## 2. Official runs on the promoted table

The promoted table is `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv` (7,105 rows, SHA-256 `01a9d51b4389db15c0d61eaa7347482e6a99df421540929abc7f0c55052365eb`, commit `fdb6e032`).

### 2.1 First attempt (lost)

Started on main `b8f6f506` on 2026-10-29: the cold runs (`stage4/a10/p2_32_cold_g1hs`, `p2_64_cold_g1hs`) started at 20:12, trained 1,758 cache files and started simulating at 20:23. The warm runs (`p2_32_warm_g1hs`, `p2_64_warm_g1hs`, caches hard-linked from the cold caches) started simulating at 20:27. All four failed with `OSError: [Errno 28] No space left on device` at 20:58-21:01, when `/data/ycfeng` filled (I-003). No metrics were written. The trained cold caches are complete (the cold runs had finished initialization) and are kept.

### 2.2 Restart (running)

Started 2026-10-30 04:39 on main `9a4f3471`; the simulator code and the promoted table are the same as in the first attempt.

| Run | Root | Cache |
| --- | --- | --- |
| P2-32 cold | `stage4/a10/p2_32_cold_g1hs_r2` | empty; trains |
| P2-64 cold | `stage4/a10/p2_64_cold_g1hs_r2` | empty; trains |
| P2-32 warm | `stage4/a10/p2_32_warm_g1hs_r2` | hard-link copy of `p2_32_cold_g1hs/cache` |
| P2-64 warm | `stage4/a10/p2_64_warm_g1hs_r2` | hard-link copy of `p2_64_cold_g1hs/cache` |

Deviation: each warm cache comes from the first-attempt cold run, not from the restarted cold run. Both cold runs train from the same commit's code and the same table, so the cache files are expected to be the same; the cold-warm comparison checks this through the receipts' cache inventories.

A mid-run cProfile window for the speed work (`stage4/run_profile3.sh p2_64 p2_64_mid2 60 1200`, cache hard-linked from `p2_64_cold_g1hs`) runs beside them; its first attempt was also lost to the full disk.

Results (gate outputs under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10_gate/`):

| Run | Status | Wall time (init / run) | Gate output |
| --- | --- | --- | --- |
| P2-32 warm | FINISHED | 85.9 s / 16,584.3 s (4.61 h) | `p2_32_warm_r2_run/` |
| P2-32 cold | FINISHED | 680.6 s / 16,494.8 s (4.58 h); trained 1,758 cache files | same metrics as P2-32 warm (A9 below) |
| P2-64 cold | running | | |
| P2-64 warm | running | | |

**P2-32 warm** (native reference: three repeats of 128 requests, `dsv4-h200-p2-canonical-20261025-01-repeat01..03`):

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 364.030 ms | 333.087 ms | −30.943 ms | 8.50% | 10% | PASS |
| TPOT mean | 19.919 ms | 19.932 ms | +0.013 ms | 0.07% | 10% | PASS |
| Request E2E mean | 20,741.132 ms | 20,723.610 ms | −17.522 ms | 0.08% | 10% | PASS |
| Token throughput (window rate) | 6,231.415 tps | 6,223.381 tps | −8.034 tps | 0.13% | 10% | PASS |

Per native repeat, the TTFT error is 7.98% / 8.68% / 8.83% (native 361.97 / 364.75 / 365.37 ms); TPOT, E2E and throughput stay within 0.30%. Overall PASS with routing MATCH. The TTFT margin is small: Frontier is 30.9 ms early, and section 3.6 observation 8 locates the shortfall.

**A9, cold against warm (P2-32):** `compare_cache_runs.py stage4/a10/p2_32_cold_g1hs_r2 stage4/a10/p2_32_warm_g1hs_r2`, run from the `a10/` directory with the gate's `PYTHONPATH`, prints "EQUAL: metrics and cache identities match; warm cache file set equals cold at run end" (`h3o_operator_costs/a10/a9_p2_32_r2_stdout.txt`). The metrics of the cold run are therefore the same as those of the warm run, and the gate reading above holds for both. This also settles the deviation of this section: the warm cache from the first-attempt cold run equals the cache that the restarted cold run trains.

The receipt reports `kv_admission` status DIFFERS: each Frontier decode lane admits 2,940 blocks at the maximum model length (maximum concurrency 37.39), native 2,944 (37.34). The difference is 0.14% and is not explored further here. Inference: it does not bind in this case, because QPS 1 with a 20.7 s request E2E time keeps about 21 requests in flight over the two decode lanes, far below 37 per lane.

## 3. TTFT diagnosis of P2-64: the native prefill side waits after a DP wave ends

Inputs:

- Frontier: the attribution of probe `p2_64_c` (section 1) by `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/h4a5_attribution.py`.
- Native: the scheduler lifecycle records of the instrumented P2-64 run `exp-1026-125836-702505` (`h200_mainline/diagnostic/inputs/p2_64l/nodes/<host>/scheduler_lifecycle/dp<k>_pid*.jsonl`; report `test_report_2026-10-26_h200_p2_64_lifecycle.md`). Its client TTFT (796.15 / 740.94 / 736.07 ms) matches the three reference repeats (779.89 / 725.49 / 718.27 ms) within 2.3%, so its prefill timing stands for the reference. The P2-32 counterpart is `p2_32l`.
- Scripts (copies of the scratch scripts in `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/ttft/`): `h3o_operator_costs/a10/ttft/p_side.py <nodes>`, `p_blackout.py <nodes> <ungated compute ms>`, `p_rows.py <nodes> <min duration ms> <index>`. Run with `/data/ycfeng/envs/frontier-py310/bin/python`, `ulimit -v 31457280`.

### 3.1 Where the 324 ms go

| Part (mean per formal request) | Native | Frontier | Difference |
| --- | --- | --- | --- |
| Arrival to the first token on the prefill side | 568.3 ms | 278.5 ms (wait 3.27 + compute 275.22) | 289.8 ms |
| After the prefill side (KV transfer, first decode output, client), as the remainder | about 173 ms | about 139 ms | about 34 ms |
| Client TTFT | 741.2 ms | 417.1 ms | 324.2 ms |

The native prefill-side value comes from the lifecycle run and the native TTFT from the three reference repeats. The lifecycle run's own TTFT mean is 757.7 ms, so its remainder is about 189 ms; the split of the 324 ms between the two parts is accurate to about 16 ms. Section 3.6 revises the remainder: without the requests that wait for a one-time NIXL handshake, the native remainder is 165.2 ms in both instrumented runs, and the reference TTFT is 720.6 ms.

On the native prefill side, the request reaches the engine 6.2 ms after its client arrival, the scheduler schedules it within 1 ms (mean 0.4-0.5 ms per lane), and the whole prompt runs in one 6,144-token step. All of the excess is inside that step: its duration (schedule to first-token update) has mean 562.1 ms, minimum 296.5 ms and maximum 813.8 ms over the 384 formal requests. Requests arrive every 1,000 ms in rotation over the four prefill lanes, so the prefill side never queues and two lanes never run real prefills at the same time.

P2-32 does not show this: its step has mean 244.6 ms (minimum 216.8, maximum 336.6), and arrival to first token is 249.4 ms.

### 3.2 The wait depends on the time since the previous prefill ended

`p_blackout.py` (P2-64, ungated compute 300 ms) groups each formal prefill by the time between the end of the previous formal prefill on any lane and its own schedule time (schedule follows arrival within about 7 ms). The effective start is the first-token update minus 300 ms, measured from the previous end.

| Scheduled after the previous end | n | Step duration mean (min-max) | Effective start after the previous end |
| --- | --- | --- | --- |
| 196-254 ms, served at once | 43 | 300.6 ms | equal to the schedule time |
| 188-259 ms, held | 36 | 750-790 ms | mean 699.5 ms, sd 19.3 |
| 260-760 ms | 302 | 577.6 ms (343.8-813.8) | mean 771.0 ms, sd 24.8 |

Three of the 384 formal prefills have no previous prefill within 2 s and are not grouped. Taking each step's wait as its duration minus 300 ms (zero when shorter), the mean wait is 264.2 ms. The P-side difference of 289.8 ms therefore splits into about 264 ms of native waiting and about 25 ms of compute (native ungated 300 ms against Frontier 275 ms, 8% lower).

The anchor is the previous prefill end on any lane (sd 19-25 ms). The latest KV `finished_sending` report is a worse anchor (sd 108 ms).

For P2-32 the same grouping (ungated compute 245 ms) finds requests scheduled about 750 ms after the previous end, with flat steps (243.2 and 246.0 ms means) and a mean wait of 5.5 ms. Its requests arrive after the window in which P2-64 requests wait, so P2-32 cannot show whether the same wait exists there.

### 3.3 Reading (inference, not verified)

Superseded by section 3.5: the about 5 ms dummy steps below are host time, not GPU time, and the late part is GPU work still queued from the ended wave.

In vLLM-BS `2cf0a691` the prefill engines run one data-parallel wave together: each engine keeps running, with dummy batches when it has no request, until the all-reduce every 32 steps (`ParallelConfig.sync_dp_state`) finds no unfinished request on any engine; then the wave ends and the engines go idle. The groups above fit this picture:

- The previous prefill's KV `finished_sending` arrives about 48 ms after its end. The last request leaves the engines then, and up to 32 more dummy steps of about 5 ms run before the next sync ends the wave, about 200-250 ms after the previous end.
- A request that arrives while the wave still runs is served at once (the 43 prefills of about 300 ms).
- A request that arrives after the wave ended cannot finish its forward until about 450-540 ms after the wave end (the 36 held prefills and the 302 prefills of the 260-760 ms group).

The wake-up path in the code looks immediate: the front end sends `FIRST_REQ` when no wave runs, the coordinator broadcasts `START_DP_WAVE`, an engine that receives a request for a finished wave sends `start_wave`, and the idle engines start dummy steps as soon as `engines_running` is set. The late part is therefore not explained by the source reading. Candidates are the inter-process messaging, the coordinator, the worker processes, or the first collective of a new wave on the GPU side. Which one it is needs a measurement.

Frontier already models the wave mechanics (`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`: `VLLM_DP_SYNC_INTERVAL = 32`, dummy steps on idle MoE / `attn_dp > 1` lanes, the wave end at the sync step), but it has no cost for restarting a wave, so its prefill compute stays flat at 275 ms. The modeling rule (no empirical scaling factors) means the gap can be closed only after the mechanism is known.

### 3.4 Warnings that do not apply

The runs log approximation factors for `attn_pre_proj` and `attn_post_proj` (0.5, BF16 to FP8) and `moe_gating_routing_topk` (2.0, BF16 to FP32) from `frontier/config/quantization_manager.py`, and a missing `attention_kernel_only.csv`. The V4 path prices its attention from `attention_graph_replay_kernel_only.csv` through `dsv4_predictor.py` and does not use those three operator predictions, so these warnings do not change the A10 results.

### 3.5 Measured mechanism: the native host issues DP dummy forwards ahead of the GPU

Input: the instrumented native P2-64 run `exp-1030-144848-164721` (requirements item 42, D-P2-64-WAVE-DIAGNOSTIC; 64 H200, succeeded), downloaded to `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-wave-20261030-01/artifacts/` (`nodes/<host>/scheduler_lifecycle/`, `nodes/<host>/worker_dp_wave/`). Prefill hosts gpu-h200-0142, 0229, 0556 and 0710; decode hosts 0828, 0882, 0938 and 1191. The START_DP_WAVE records were lost to an observer bug (I-005); the forward-level records are complete.

Scripts: copies in `h3o_operator_costs/a10/wave/`, each with its output beside it. Run each with `/data/ycfeng/envs/frontier-py310/bin/python <script> <artifacts>` from that directory under `ulimit -v 31457280`; `fifo.py` reads `explore_out.json` (written by `explore.py`) from its working directory.

#### Observations

1. **The four prefill lanes advance in lockstep.** Every forward, real or dummy, runs a CPU all-reduce over the DP group (`dp_cpu_all_reduce`, numbered forwards; `vllm/v1/worker/gpu/model_runner.py:1408`, `dispatch_cg_and_sync_dp`). All four prefill workers record forwards 1-3,872 (`explore_stdout.txt`).
2. **Forward modes** (`modes.py`): on each prefill lane, 3,738 serving dummies are 1-token FULL CUDA graph replays, and 97-99 run without a graph (the forwards that pair with a peer's eager prefill). Every formal prefill is one 6,144-token eager forward. The decode lanes run mostly real batches.
3. **Dummy host and GPU times** (`depth.py`, 122 prefill waves). After a prefill, about 23 dummies return from the host at a median of 4.68 ms each. The host then becomes paced at a median of 23.81 ms per dummy, which is the dummy's GPU time; Frontier prices the V4 dummy at 24.38 ms (+2.4%).
4. **How deep the host runs ahead** (`depth.py`, `depth_summary.py`). At the switch from fast to paced, the GPU holds a median of 18.56 issued and unfinished dummies. 117 of the 122 waves fall in 18.04-18.90; the other five (-37.8, -2.0, 14.7, 17.8, 17.8) are waves that a peer request interrupted. A depth between 18 and 19, with the oldest dummy partly executed, means the host's launch blocks while 19 forwards are unfinished.
5. **The prefill step** (`explore.py`, 128 formal requests). From schedule to first-token update, the step takes a mean of 536.5 ms (285.6-750.8). Its parts:

   | Part | Mean | Range |
   | --- | --- | --- |
   | Schedule to the DP all-reduce | 4.1 ms | 3.1-6.1 ms |
   | The DP all-reduce itself | 1.2 ms | 0.7-4.0 ms |
   | All-reduce end to the host's return from `execute_model` | 401.2 ms | 178.9-613.5 ms |
   | Host return to the update | 130.0 ms | 44.0-152.7 ms |

   The eager prefill launches its kernels behind the queued dummies, so the host returns only when its last kernels are queued. The GPU then still runs the prefill's tail: the lane's next forward starts a median of 135.4 ms after the host returns (`post2.py`, p10 99.8 ms, p90 136.3 ms, n 144).
6. **Decode side** (`dside.py`). 6,674 of 8,192 decode forwards have a real batch on all four lanes, and only 214 are dummies on all four. Those 214 show the same pattern: about 22 dummies at about 4-5 ms each, then paced at about 24.8 ms.
7. **No explicit wait in the dummy path.** In the native source, the dummy path (`_dummy_run`, `model_runner.py:633`) issues one FULL graph replay without `synchronize`. The runner synchronizes only for profiling and warmup (`model_runner.py:808`, `825`, `1921`), and `UvaBufferPool` (`buffer_utils.py`) cycles its buffers without waiting on an event.

#### Reading (inference with quantitative support)

- **The wave counter counts host steps, not GPU work.** At 4.68 ms of host time against 23.81 ms of GPU time per dummy, the host reaches the 32-step DP sync (`VLLM_DP_SYNC_INTERVAL`) long before the GPU finishes. When the sync ends the wave, about 18 dummies (about 440 ms of GPU work) are still queued. A prefill that arrives next queues behind them on the GPU. This explains the waits of section 3.2 and replaces the reading of section 3.3.
- **The GPU cycle per request is slightly longer than the arrival interval.** One prefill plus 31 dummies is about 290 + 31 × 23.8 ≈ 1,030 ms of GPU time, against 1,000 ms between arrivals. The queue therefore grows by about 30 ms per request until a wave drains, which produces the sawtooth of TTFT from about 290 to about 745 ms.
- **The cause of the 19-forward limit is not identified.** No call in the source sets it. Hypothesis: the capacity of the CUDA launch queue. Not verified.
- **Why P2-32 is flat.** P2-32 requests arrive about 750 ms after the previous end (section 3.2), when the queued dummies have drained.

#### Check of the mechanism: a FIFO model of the prefill side (`fifo.py`)

The model replays the native prefill-side add times through one lockstep host and one GPU in order. In a wave, a forward is the prefill when a request is waiting, otherwise a dummy. A dummy is launched once forward n−K has finished (so at most K forwards are unfinished after a launch). It costs c of host time and g of GPU time. A prefill holds the host until its output. The wave ends at a 32-step sync when no request is waiting.

Arguments `g c K P d_sched d_post` (ms, except K). Result: the mean add-to-update time of the 128 formal requests, against the actual 536.7 ms.

| Parameters | K | Predicted mean | Output file |
| --- | --- | --- | --- |
| native: g 23.8, c 4.68, P 300, d_sched 6.5, d_post 3 | 1 | 318.7 ms | `fifo_1.txt` |
| native | 18 | 528.2 ms | `fifo_18.txt` |
| native | 19 | 544.9 ms | `fifo_19.txt` |
| native with P 290 | 18 | 521.4 ms | `fifo_P290.txt` |
| native with P 283 | 18 | 504.5 ms | `fifo_P283.txt` |
| Frontier: g 24.38, c 0, P 275.2, d_sched 3.3, d_post 0 | 1 | 300.0 ms | `fifo_frontier_K1.txt` |
| Frontier | 18 | 508.5 ms | `fifo_frontier_K18.txt` |
| Frontier | 19 | 514.9 ms | `fifo_frontier_K19.txt` |

- With run-ahead, the model reproduces the native mean within 1.6% (K 18) and 1.5% (K 19). Without it (K 1), the model falls 41% short.
- Per request, the model does not follow the native sequence (correlation -0.33 to 0.24, mean absolute error 135-238 ms). The sawtooth has the right shape and period, but its phase drifts with small errors in g and P. The model checks the mean and the range of the TTFT, not its per-request order.
- Frontier at K 1, at 300.0 ms, matches Frontier's own prefill side (278.5 ms, section 3.1) within about 22 ms. Because c is below g, host time matters little; the number of queued dummies at a wave end does.

#### Consequence for Frontier

Frontier's host waits for each dummy forward to end: `on_schedule` returns no batch while `_dummy_forwards_in_flight` is non-zero (`vllm_v1_engine_replica_scheduler.py`). A wave therefore never ends with queued dummies, and a prefill starts at once. The fix models the native launch rule (assignment B26): the host does not wait for a dummy forward's end, and launches its next forward once fewer than K forwards are unfinished on the device; the lanes keep the per-forward lockstep of observation 1. The value K = 19 comes from observation 4.

Expected effect on P2-64, from the model: the prefill side rises from about 300 ms to about 509-515 ms. With Frontier's current remainder of about 139 ms, TTFT would be about 650 ms, against native 741.2 ms (about -12%). Two known gaps remain: the prefill compute (275.2 ms against about 290-300 ms) and the remainder after the prefill side (138.6 ms against 165.2 ms without the first-contact requests, section 3.6).

### 3.6 After the prefill side: one-time NIXL handshakes and the decode-side handoff

Inputs:

- Native: the wave run of section 3.5 (`W`, its `artifacts/` directory) and the lifecycle run `exp-1026-125836-702505` of section 3, read from its download `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-lifecycle-20261026-01/artifacts/` (`L`); the proxy and client records of the reference runs, `h200_mainline/diagnostic/inputs/{p2_64r,p2_32r}/run/`.
- Frontier: probe `p2_64_c` (section 1), `metrics_ground_truth.jsonl` and `frontier_transfer_ledger.jsonl` in `stage4/probe/p2_64_c/metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_p2_64_operator_path/` (`F`).

Scripts: copies in `h3o_operator_costs/a10/remainder/`, each with its output beside it; outputs for the lifecycle run are in `lifecycle/`. Run from that directory with `/data/ycfeng/envs/frontier-py310/bin/python` under `ulimit -v 31457280`.

| Step | Command | Output |
| --- | --- | --- |
| Native per-request join (proxy, prefill and decode lifecycle) | `remainder.py <W> remainder_out.json`; the same with `<L>` and `lifecycle/` | `remainder_stdout.txt`, `lifecycle/remainder_stdout.txt` |
| KV receive detail per request | `nixl_detail.py <W> remainder_out.json nixl_detail_out.json`; the same with `<L>` | `nixl_detail_stdout.txt`, `lifecycle/nixl_detail_stdout.txt` |
| Lane pairs, first-contact requests, ingress split | `pairs.py <W> remainder_out.json nixl_detail_out.json first_contact.json`; the same with `<L>` | `pairs_stdout.txt`, `lifecycle/pairs_stdout.txt` |
| Frontier join | `frontier_remainder.py <F> frontier_remainder_p2_64_c.json` | `frontier_remainder_p2_64_c.txt` |
| Span comparison | `remainder_compare.py frontier_remainder_p2_64_c.json remainder_out.json first_contact.json lifecycle/remainder_out.json lifecycle/first_contact.json` | `remainder_compare_stdout.txt` |
| Slow requests per repeat | `handoff_slow.py <run>/proxy_requests.jsonl <run>/client_requests.jsonl` for P2-64r, the lifecycle run and the wave run | `handoff_slow_stdout.txt` |
| Reference with and without first-contact requests | `reference_without_handshake.py <run> [first_contact.json]` for P2-64r, P2-32r, the wave run and the lifecycle run | `reference_without_handshake_stdout.txt` |
| Reference split on the proxy clock | `reference_split.py <run>` for P2-64r and P2-32r | `reference_split_stdout.txt` |

On the lifecycle run, `pairs.py` writes `first_contact.json` and then stops with "no add_enqueue records: ingress split not available", because that run's observer did not record `add_enqueue`. Rerunning `remainder.py` and `nixl_detail.py` on the lifecycle run from the copies reproduces both JSON outputs byte for byte (`cmp`).

#### Observations

1. **The first request between a prefill lane and a decode lane waits for a NIXL handshake.** A decode worker connects to a prefill engine when the first request from that engine arrives: `start_load_kv` starts a background handshake and posts the READ only after the handshake ends (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:65-77`, native source `2cf0a6915`). Four prefill lanes and four decode lanes make 16 pairs. The warmup requests open 13 of them in the wave run and 11 in the lifecycle run (`pairs_stdout.txt`). The formal requests that open the remaining pairs wait 1,438.8-1,620.5 ms (wave run: r005, r017, r037) and 1,457.9-1,800.4 ms (lifecycle run: r002, r004, r006, r009, r053) from decode add to `finished_recving`, against medians of 78.0 and 74.5 ms for all formal requests. All of them are in repeat 1.
2. **These requests are the ones with a proxy handoff above 1,000 ms.** In both instrumented runs, the formal requests whose handoff (prefill response to the first decode byte, on the proxy clock) exceeds 1,000 ms are exactly the first-contact requests ("same set as the first-contact requests: True", `reference_without_handshake_stdout.txt`). The other formal requests have handoff medians of 152.6-168.2 ms per repeat (`handoff_slow_stdout.txt`). The sawtooth tops just above 1,000 ms TTFT (r119-repeat02 in P2-64r, 1,012.7 ms, and six requests of the lifecycle run) have handoffs of 190.8-213.4 ms and long prefill sides of 796.2-824.8 ms, so they are not first contacts.
3. **The reference without first contacts.** P2-64r has five such requests, all in repeat 1 (r006, r017, r027, r035 and r036, handoff 1,455.3-1,923.9 ms). Without them, the reference TTFT mean is 720.6 ms (n 379) instead of 741.2 ms (n 384). P2-32r has none (364.0 ms, n 384). Split on the proxy clock (`reference_split_stdout.txt`):

   | Part (mean) | P2-64r without first contacts (n 379) | P2-32r (n 384) |
   | --- | --- | --- |
   | Client dispatch to proxy arrival | 0.6 ms | 0.5 ms |
   | Proxy arrival to prefill response (prefill side) | 552.0 ms | 235.5 ms |
   | Prefill response to first decode byte (handoff) | 167.8 ms | 127.9 ms |
   | First decode byte to client | 0.2 ms | 0.2 ms |
   | TTFT | 720.6 ms | 364.0 ms |

4. **Span comparison with Frontier** (`remainder_compare_stdout.txt`). Native means leave out the first-contact requests, and native parts are merged into the spans that Frontier models. Frontier's KV transfer starts at the prefill end.

   | Span (mean) | Native wave run (n 125) | Native lifecycle run (n 379) | Frontier `p2_64_c` (n 128) |
   | --- | --- | --- | --- |
   | Prefill first-token update to decode add (ingress) | 21.7 ms | 22.7 ms | 0 (not modeled) |
   | Decode add to `finished_recving` | 86.9 ms | 84.8 ms | 88.2 ms (KV transfer start to end) |
   | `finished_recving` to decode first-token update | 55.2 ms | 53.0 ms | 50.4 ms |
   | Decode first-token update to client | 0.9 ms | 4.3 ms | 0 (not modeled) |
   | Total after the prefill side | 165.2 ms | 165.2 ms | 138.6 ms |

5. **The ingress** (wave run, `pairs_stdout.txt`, `remainder_stdout.txt`). The 21.7 ms consist of the prefill response (2.3 ms), prefill response to the decode `add_enqueue` (9.1 ms, 6.3-12.8 ms: the proxy and the decode API server) and `add_enqueue` to the decode add (10.4 ms, 0.2-24.8 ms: the wait for the decode engine loop). Frontier posts its READ at a decode step start (`nixl_read_completion=engine_step` in `F/config.json`; the receipt records it as the `h4a4` override of the input's default `transfer_end`), so its 88.2 ms already contain the equivalent of the engine-loop wait. Compared like for like, native `add_enqueue` to `finished_recving` is 10.4 + 86.9 = 97.3 ms against Frontier's 88.2 ms (−9.1 ms). The parts outside the engine, 2.3 + 9.1 = 11.4 ms, are not modeled.
6. **KV handles** (`nixl_detail_stdout.txt`, `lifecycle/nixl_detail_stdout.txt`). Each request reads 8 handles of 46,103,040 bytes. The slowest handle of a request has a median of 27.9 ms (wave run) and 26.2 ms (lifecycle run) and a mean of 37.0-37.1 ms. NIXL's `transfer_duration` is recorded only when the per-step poll finds the handle DONE (`base_worker.py:2224-2241`, `stats.py:45-47`). In the rows printed for the slowest requests, its values lie near multiples of the decode step (about 25, 50, 75 and 96 ms) or in a separate group at 1.6-2.6 ms. Frontier's ledger records 39,521,664 bytes per request transfer; how that figure relates to the 8 native handles is not checked here.
7. **The first decode step after the receive** takes 53.0-55.2 ms natively against Frontier's 50.4 ms (2.6-4.8 ms longer).
8. **P2-32 has the same shortfall after the prefill side.** Frontier's P2-32 warm run (section 2.2; `frontier_remainder.py <P2-32 warm metrics dir> frontier_remainder_p2_32_warm_r2.json`, output `frontier_remainder_p2_32_warm_r2.txt`) has a prefill side of 222.3 ms and a remainder of 110.8 ms (KV transfer 70.6 ms, transfer end to first decode token 40.2 ms). The reference P2-32r has 235.5 ms and 127.9 ms (observation 3; its handoff starts at the prefill response, so it holds the 2.3 ms of the response that Frontier's remainder does not). Frontier is 13.2 ms short on the prefill side and 17.1 ms short after it, against 26.6 ms after it on P2-64.

#### Reading (inference)

- **The handshake.** The first-contact wait is a one-time connection cost. How many formal requests pay it depends on how many pairs the warmup reached (13 and 11 of 16 in the instrumented runs; five first contacts in P2-64r). It adds 20.6 ms to the reference mean. Recommendation: do not model it in Frontier; keep the gate on the full reference and report a diagnostic reading against 720.6 ms beside it. A native warmup that reaches all 16 pairs would remove it from future references; that needs a GPU run and YC's approval.
- **The handle durations.** The 1.6-2.6 ms group suggests that the transfer of one 46 MB handle takes at most about 2 ms on the wire, and that the larger values mostly count the decode steps until the poll sees the handle done. This is not verified.
- **The remainder after the prefill side.** Without first contacts the native value is 165.2 ms, not about 173-189 ms as estimated in section 3.1. Frontier's 138.6 ms is 26.6 ms short: the ingress outside the engine (11.4 ms; not modeled, deferred in `h4b_d_side_model/design.md` section 6.3), the late READ handles (about 9 ms; open, section 6.2 of that design), the first decode step (3-5 ms) and the client egress (1-4 ms).
- **Expected TTFT after B26.** With a prefill side of about 509-515 ms (section 3.5) and the remainder of 138.6 ms, Frontier TTFT would be about 648-654 ms: −11.8% to −12.6% against 741.2 ms and −9.3% to −10.1% against 720.6 ms.
- **The prefill side after B26.** Against the reference prefill side of 552.0 ms, Frontier would be about 37-43 ms short. The known parts are the prefill compute (275.2 ms against about 290-300 ms, section 3.5), the prefill ingress (proxy arrival to prefill add 9.1-11.3 ms natively against Frontier's 3.27 ms wait) and the prefill response (2.3-4.3 ms).
- **Ingress modeling needs approval.** Modeling the ingress outside the engine needs a new delay field with measured values per platform, which is a configuration schema change. On P2-32 it would move Frontier toward the reference, not past it: Frontier is 17.1 ms short after the prefill side there (observation 8), and adding the 11.4 ms measured on P2-64 would give a TTFT of about 344.5 ms (−5.4% instead of −8.5%). The P2-32 ingress itself is not measured; no P2-32 run recorded `add_enqueue`, so the P2-64 value stands in for it.
