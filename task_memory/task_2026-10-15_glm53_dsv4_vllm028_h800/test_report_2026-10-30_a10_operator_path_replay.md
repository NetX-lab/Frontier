# Test report: A10 operator-path E2E replay of P2-32 and P2-64 on H200 (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Added section 5.7: the B28 replays on main `ee29d548`. P2-32 TTFT −11.22% (FAIL; B24 −7.94%) and P2-64 −10.60% (FAIL; −8.05% against 720.6 ms; B24 −11.62%); the other metrics stay within 2.45%. The per-forward EP change depends only on the DP lane of the 6,144-token prefill; the dp0 values equal the B28 price check. P2-32's earlier PASS rested partly on an EP16 price about 12 ms per forward too high. |
| 2026-10-30 | Added section 5.6: Frontier's 6,144-token prefill against the native kernels of the P2-64 profiled run. Only the EP combine differs much (−7.08 ms of the about 8.2 ms device gap). Its back-to-back benchmark timing is the period of pipelined calls (rank 0 2,011 µs) and not the time of a call started on aligned ranks (2,176 µs). On EP16 the error has the opposite sign. The host part, about 7.5 ms, is not modeled for V4. |
| 2026-10-30 | Added section 5.5: every repeat of the lifecycle runs of 2026-10-25/26 is slower on the prefill side than every canonical repeat (by 13.8-34.4 ms), so the 300 ms native prefill step of sections 3.2 and 3.5 is overstated; on the wave run, whose timing matches the canonical run, the step is 289.6 ms. FIFO parameter swaps attribute the P2-64 in-engine gap to the prefill step. |
| 2026-10-30 | Added section 5: the K = 19 replays on main `901bc091` (B26 + B24). P2-32 PASS (TTFT −7.94%); P2-64 TTFT −11.62% (FAIL; −9.09% against 720.6 ms), prefill side 513.3 ms as predicted; A9 EQUAL for both cases. New split of the native prefill ingress; section 3.1's 6.2 ms corrected. Section 4 notes B24's lane reuse. |
| 2026-10-30 | Section 2.2: P2-64 cold and warm finished (FAIL on TTFT, 416.2 against 741.2 ms; the other metrics within 3.4%); A9 EQUAL; section 4 gains the P2-64 census. |
| 2026-10-30 | Added section 4: A7 and A8 mechanism and the A8 census of the P2-32 runs (every collective on a measured point, no violation). |
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

### 2.2 Restart

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
| P2-64 cold | FINISHED | 683.5 s / 20,295.8 s (5.64 h); trained 1,758 cache files | same metrics as P2-64 warm (A9 below) |
| P2-64 warm | FINISHED | 82.7 s / 20,343.9 s (5.65 h) | `p2_64_warm_r2_run/` (cold: `p2_64_cold_r2_run/`) |

**P2-32 warm** (native reference: three repeats of 128 requests, `dsv4-h200-p2-canonical-20261025-01-repeat01..03`):

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 364.030 ms | 333.087 ms | −30.943 ms | 8.50% | 10% | PASS |
| TPOT mean | 19.919 ms | 19.932 ms | +0.013 ms | 0.07% | 10% | PASS |
| Request E2E mean | 20,741.132 ms | 20,723.610 ms | −17.522 ms | 0.08% | 10% | PASS |
| Token throughput (window rate) | 6,231.415 tps | 6,223.381 tps | −8.034 tps | 0.13% | 10% | PASS |

Per native repeat, the TTFT error is 7.98% / 8.68% / 8.83% (native 361.97 / 364.75 / 365.37 ms); TPOT, E2E and throughput stay within 0.30%. Overall PASS with routing MATCH. The TTFT margin is small: Frontier is 30.9 ms early, and section 3.6 observation 8 locates the shortfall.

**A9, cold against warm (P2-32):** `compare_cache_runs.py stage4/a10/p2_32_cold_g1hs_r2 stage4/a10/p2_32_warm_g1hs_r2`, run from the `a10/` directory with the gate's `PYTHONPATH`, prints "EQUAL: metrics and cache identities match; warm cache file set equals cold at run end" (`h3o_operator_costs/a10/a9_p2_32_r2_stdout.txt`). The metrics of the cold run are therefore the same as those of the warm run, and the gate reading above holds for both. This also settles the deviation of this section: the warm cache from the first-attempt cold run equals the cache that the restarted cold run trains.

**P2-64 warm** (native reference of section 1):

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 741.216 ms | 416.227 ms | −324.989 ms | 43.85% | 10% | FAIL |
| TPOT mean | 25.871 ms | 25.291 ms | −0.580 ms | 2.24% | 10% | PASS |
| Request E2E mean | 27,206.770 ms | 26,288.929 ms | −917.841 ms | 3.37% | 10% | PASS |
| Token throughput (window rate) | 5,963.206 tps | 5,986.314 tps | +23.107 tps | 0.39% | 10% | PASS |

Per native repeat, the TTFT error is 46.63% / 42.63% / 42.05%; TPOT 0.55% / 3.69% / 2.46%; E2E 1.89% / 4.70% / 3.49%; throughput 0.64% / 0.16% / 0.36%. Overall FAIL on TTFT only, with routing MATCH. The gate on the cold run prints the same values. Against the diagnostic reference without the five first-contact requests (720.6 ms, section 3.6), the TTFT error is −42.2%.

The reading agrees with probe `p2_64_c` (section 1: TTFT 417.061 ms) within 0.9 ms: the promoted table changes TTFT by −0.83 ms and the other metrics by less than 0.2%. `frontier_remainder.py` on the warm run (`frontier_remainder_p2_64_warm_r2.txt`) gives a prefill side of 278.5 ms and a remainder of 137.7 ms (KV transfer 87.3 ms, transfer end to first decode token 50.4 ms), against 278.5 and 138.6 ms in the probe. This run is on main `9a4f3471`, without B26; section 3.5 explains the prefill-side gap, and B26 models it.

**A9, cold against warm (P2-64):** `compare_cache_runs.py stage4/a10/p2_64_cold_g1hs_r2 stage4/a10/p2_64_warm_g1hs_r2` prints "EQUAL: metrics and cache identities match; warm cache file set equals cold at run end" (`h3o_operator_costs/a10/a9_p2_64_r2_stdout.txt`).

The receipt reports `kv_admission` status DIFFERS: each Frontier decode lane admits 2,940 blocks at the maximum model length (maximum concurrency 37.39), native 2,944 (37.34). The difference is 0.14% and is not explored further here. Inference: it does not bind in this case, because QPS 1 with a 20.7 s request E2E time keeps about 21 requests in flight over the two decode lanes, far below 37 per lane. P2-64 shows the same per-lane admission (2,940 against 2,944 blocks; maximum concurrency 38.41 against native 38.35, status DIFFERS); about 27 requests are in flight there over four decode lanes, so it does not bind either.

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

On the native prefill side, the request reaches the engine 9.1 ms after its proxy arrival (2.96 ms to the API server, then 6.16 ms to the engine add; section 5.4 corrects the 6.2 ms that this sentence first gave as the whole ingress), the scheduler schedules it within 1 ms (mean 0.4-0.5 ms per lane), and the whole prompt runs in one 6,144-token step. All of the excess is inside that step: its duration (schedule to first-token update) has mean 562.1 ms, minimum 296.5 ms and maximum 813.8 ms over the 384 formal requests. Requests arrive every 1,000 ms in rotation over the four prefill lanes, so the prefill side never queues and two lanes never run real prefills at the same time.

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

## 4. A7 and A8 on the finished runs

A7 (EP rank admission) and A8 (collective coverage) are gate items of H3o (`h3o_operator_costs/design.md` section 1). On the V4 operator path the simulator enforces both while it runs, so a finished run satisfies them; this section records the mechanism and the census of the finished runs.

**A7, mechanism (from the code).** Each routed-MoE layer wave predicts every participating EP lane, not only rank 0 (`frontier/scheduler/utils/expert_parallel.py:108-170`), and the wave takes the maximum over the lanes for each phase: pre-dispatch, dispatch, routed compute, combine and post-combine (`expert_parallel.py:252-272`). The grouped GEMM of a lane is looked up at the lane's balanced rank class: the lane's token counts must equal those of the measured balanced workload, and the coordinate (EP size, routed population, rank class) must be a measured point (`frontier/execution_time_predictor/dsv4_operator_predictor.py:359-374`, exact lookup at lines 309-316). A lane that is not a measured class stops the run with an error. There is no interpolation and no fallback.

**A8, mechanism (from the code).** `MeasuredCCBackend.predict_operator_collective` (`frontier/cc_backend/backends/measured_cc_backend.py:250-281`) rejects a request outside the measured size range of its curve, and prices an unequal EP combine (`reduce_scatterv` with unequal SP rows) only from an exact measurement of its ordered rows and bytes; otherwise it raises `MeasuredCCOutOfRangeError`. An unequal combine therefore never uses the scalar curve.

**A8, census.** `h3o_operator_costs/a10/a8_collectives.py <run root>` reads the observer's `collectives.json`, checks that its call total equals the receipt's, and checks the recorded bounds of every request: an exact point has equal lower and upper bounds, an interpolated point lies strictly inside its bracket, and every unequal combine is exact.

| Run | Distinct requests | Calls | Exact | Interpolated | Unequal combines (all exact) | Violations | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2-32 warm | 43 | 19,438,100 | 19,438,100 | 0 | 88,064 (prefill) | 0 | `a8_p2_32_stdout.txt` |
| P2-32 cold | 43 | 19,438,100 | 19,438,100 | 0 | 88,064 (prefill) | 0 | `a8_p2_32_stdout.txt` |
| P2-64 cold | 35 | 27,711,291 | 27,711,291 | 0 | 176,128 (prefill) | 0 | `a8_p2_64_stdout.txt` |
| P2-64 warm | 35 | 27,711,291 | 27,711,291 | 0 | 176,128 (prefill) | 0 | `a8_p2_64_stdout.txt` |

The two P2-32 `collectives.json` files are byte-identical (`cmp`). Every P2-32 collective falls on a measured point: 12 curves (operation, domain, devices) in the two roles, from `sp_all_gather` and `sp_reduce_scatter` at TP 8 to `all_gatherv` and `reduce_scatterv` at EP 16. The two P2-64 files are also byte-identical; P2-64 uses 12 curves as well, with the EP collectives at EP 32.

Since B24 (main `901bc091`), lanes of one wave with the same prediction key share the first lane's prediction instead of calling the predictor again ([B24 test report](test_report_2026-10-30_b24_v4_ep_lane_reuse.md)). The phase maxima over the lanes are unchanged, because the key covers every input of the prediction; the census of section 5.3 therefore counts fewer calls.

Reading: A7 and A8 hold for P2-32 and P2-64. The census shows how A8 is met; A7 has no per-lane census, because the observer records rank-0 scopes only, so its evidence is the code path above and the fact that the runs finished.

## 5. Run-ahead replays (B26, K = 19)

Code: main `901bc091`, which holds B26 (the DP host forms forwards ahead of the GPU, up to `max_forwards_in_flight` unfinished forwards; `a10_driver.py` sets 19 from section 3.5, observation 4) and B24 (one prediction per distinct EP lane key, identity-preserving). Detached worktree `.worktrees/a10-k19-901bc091`; launcher `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10_k19_b24.sh`; roots `stage4/a10/p2_{32,64}_{cold,warm}_k19b24`. Each warm cache is a `cp -a` copy of its case's section 2.2 cold cache (`stage4/a10/p2_{32,64}_cold_g1hs_r2/cache`, 1,758 files). Post-processing: `h3o_operator_costs/a10/k19/post_k19.sh <run name>` (gate, A8 census and `frontier_remainder.py`) and `k19/per_repeat.py <reading.json>`; copies of the outputs are in `h3o_operator_costs/a10/k19/`. Before use, `post_k19.sh` was run on the section 2.2 P2-32 warm run (`stage4/a10/p2_32_warm_g1hs_r2`, output `stage4/a10_gate/p2_32_warm_g1hs_r2_post/`) and reproduced its reading (TTFT 333.087 ms, 8.50%).

| Run | Status | Wall time (init / run) | Source dirty paths |
| --- | --- | --- | --- |
| P2-32 warm | FINISHED | 81.6 s / 9,701.1 s (2.69 h; 4.61 h before B24) | none |
| P2-32 cold | FINISHED | 755.3 s (1,758 cache files trained) / 9,575.8 s | none |
| P2-64 warm | FINISHED | 82.6 s / 12,035.8 s (3.34 h; 5.65 h before B24) | none |
| P2-64 cold | FINISHED | 755.4 s (1,758 cache files trained) / 11,924.6 s | none |

### 5.1 P2-32 (gate output `stage4/a10_gate/p2_32_warm_k19b24_run/`)

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result | K unset (section 2.2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 364.030 ms | 335.115 ms | −28.915 ms | 7.94% | 10% | PASS | 8.50% |
| TPOT mean | 19.919 ms | 19.927 ms | +0.008 ms | 0.04% | 10% | PASS | 0.07% |
| Request E2E mean | 20,741.132 ms | 20,720.305 ms | −20.827 ms | 0.10% | 10% | PASS | 0.08% |
| Token throughput (window rate) | 6,231.415 tps | 6,226.140 tps | −5.276 tps | 0.08% | 10% | PASS | 0.13% |

Per native repeat, the TTFT error is 7.42% / 8.12% / 8.28%; the other metrics stay within 0.32%. Overall PASS with routing MATCH. As expected, P2-32 barely moves: its prefills arrive after the previous wave has drained (section 3.5).

Request 0, the first request of the trace, has a TTFT of 635.2 ms against 279.4 ms with K unset. Its KV transfer ends while the idle decode DP group runs a dummy wave, and its first decode forward queues behind the dummies that the host has formed ahead: 376.2 ms from the KV end to the first decode token, against 20.4 ms with K unset. Native shows the same run-ahead on its all-dummy decode forwards (section 3.5, observation 6), and its warmup requests take this cold start before the formal requests. Its rise of 355.8 ms raises the mean by 2.8 ms. Without request 0 in both runs, the TTFT mean is 332.75 ms with K = 19 against 333.51 ms with K unset.

### 5.2 P2-64 (gate output `stage4/a10_gate/p2_64_warm_k19b24_run/`)

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result | K unset (section 2.2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 741.216 ms | 655.102 ms | −86.114 ms | 11.62% | 10% | FAIL | 43.85% |
| TPOT mean | 25.871 ms | 25.295 ms | −0.576 ms | 2.23% | 10% | PASS | 2.24% |
| Request E2E mean | 27,206.770 ms | 26,531.441 ms | −675.328 ms | 2.48% | 10% | PASS | 3.37% |
| Token throughput (window rate) | 5,963.206 tps | 5,976.585 tps | +13.379 tps | 0.22% | 10% | PASS | 0.39% |

Per native repeat, the TTFT error is 16.00% / 9.70% / 8.79% (native 779.89 / 725.49 / 718.27 ms); repeat 1 holds the five first-contact requests (section 3.6, observation 3). TPOT 0.53% / 3.66% / 2.44%; E2E 0.98% / 3.82% / 2.60%; throughput 0.48% / 0.00% / 0.19%. Overall FAIL on TTFT only, with routing MATCH. Against the diagnostic reference without the first-contact requests (720.6 ms), the TTFT error is −9.09%.

The prefill side is 513.3 ms (range 273.8-748.5 ms), within the 509-515 ms that the FIFO model of section 3.5 predicted, against 278.5 ms with K unset. The remainder after it is 141.8 ms (137.7 ms with K unset, section 2.2). Request 0 again waits behind the decode dummies (489.3 ms from the KV end to the first decode token); without it, the TTFT mean is 653.9 ms, the prefill side 515.2 ms and the remainder 138.7 ms (K unset: 416.8, 278.5 and 138.2 ms).

### 5.3 A9 and A8

- **A9.** `compare_cache_runs.py <cold root> <warm root>`, run from the `a10/` directory with the gate's `PYTHONPATH`, prints "EQUAL: metrics and cache identities match; warm cache file set equals cold at run end" for both cases (`k19/a9_p2_32_k19b24_stdout.txt`, `k19/a9_p2_64_k19b24_stdout.txt`). The cold and warm `collectives.json` files are byte-identical in both cases (`cmp`). The readings above therefore hold for the cold runs too.
- **A8** (`k19/a8_p2_32_warm_k19b24_stdout.txt`, `k19/a8_p2_64_warm_k19b24_stdout.txt`):

| Run | Distinct requests | Calls | Exact | Interpolated | Unequal combines (all exact) | Violations |
| --- | --- | --- | --- | --- | --- | --- |
| P2-32 warm K19 | 43 | 6,310,644 | 6,310,644 | 0 | 16,512 (prefill) | 0 |
| P2-64 warm K19 | 35 | 9,024,624 | 9,024,624 | 0 | 27,520 (prefill) | 0 |

The distinct requests are the same as in section 4. The call count falls (19,438,100 and 27,711,291 in section 4). Inference, not yet checked: most of the fall is B24's lane reuse, which skips the backend for lanes that share a key; run-ahead can also change the number of dummy forwards. The `_k19` runs on main `50ed2061` (B26 without B24, still running) separate the two.

### 5.4 Where the remaining TTFT gap is

Native values leave out the first-contact requests. The native prefill side runs from the proxy arrival to the prefill response; Frontier's from the arrival to the prefill end.

| Part (mean) | P2-64 native (P2-64r, n 379) | P2-64 Frontier K19 | Difference | P2-32 native (P2-32r, n 384) | P2-32 Frontier K19 | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| Client dispatch to proxy arrival | 0.6 ms | 0 | −0.6 ms | 0.5 ms | 0 | −0.5 ms |
| Prefill side | 552.0 ms | 513.3 ms | −38.7 ms | 235.5 ms | 222.3 ms | −13.2 ms |
| After the prefill side, to the client | 168.0 ms | 141.8 ms | −26.2 ms | 128.1 ms | 112.8 ms | −15.3 ms |
| TTFT | 720.6 ms | 655.1 ms | −65.5 ms | 364.0 ms | 335.1 ms | −28.9 ms |

**New measurement: the native prefill ingress.** `k19/ingress_split.py <artifacts>` (outputs `k19/ingress_split_wave-20261030-01.txt`, `k19/ingress_split_lifecycle-20261026-01.txt`) splits the path from the proxy arrival (proxy host clock) through the API server's `arrival_time` and the `add_enqueue` record (prefill host clock) to the engine add:

| Span (mean, median) | Wave run (n 128) | Lifecycle run (n 384) |
| --- | --- | --- |
| Proxy arrival to API-server arrival | 3.11 ms, 2.86 ms | 2.96 ms, 2.93 ms |
| API-server arrival to `add_enqueue` | 6.59 ms, 6.74 ms | not recorded |
| `add_enqueue` to engine add | 1.63 ms, 0.15 ms | not recorded |
| Proxy arrival to engine add | 11.33 ms, 10.11 ms | 9.12 ms, 7.27 ms |

The first span crosses hosts, so it includes any offset between the proxy and prefill host clocks; the two runs agree within 0.15 ms. Section 3.1's "6.2 ms after its client arrival" was the API-server arrival to the engine add (6.16 ms) and left out the proxy hop; it is corrected there.

**Reading (inference).**

- **Outside the engine, about 22 ms per request (wave-run values) are not modeled:** the prefill ingress (proxy to API server to engine, 9.1-11.3 ms), the prefill response back to the proxy (2.3 ms, section 3.6 observation 5), the decode ingress (prefill response to the decode `add_enqueue`, 9.1 ms; deferred in `h4b_d_side_model/design.md` section 6.3), the client dispatch (0.5-0.6 ms) and the egress after the decode first-token update (0.9 ms in the wave run, 4.3 ms in the lifecycle run). They explain the whole P2-32 prefill-side gap (13.2 ms) and about 9 ms of its remaining 15.3 ms.
- **Inside the engine, P2-64 has about 25-27 ms more on the prefill side.** Section 5.5 attributes it: the prefill step, which the earlier 300 ms overstated because it came from a slower instrumented run.
- **After the prefill side**, beyond the decode ingress: the late READ handles (about 9 ms; open in `h4b_d_side_model/design.md` section 6.2) and the first decode step (3-5 ms, section 3.6).
- **Expected effect of modeling the path outside the engine** (arithmetic, not a run): about +22 ms on both cases, giving P2-64 about 677 ms (−8.7% against 741.2 ms) and P2-32 about 357 ms (−1.9%). It needs new delay fields with measured values per platform, a configuration schema change that needs YC's approval. P2-32's own ingress is not measured; no P2-32 run recorded `add_enqueue`.

### 5.5 The instrumented runs and the native prefill step

**Observation 1: the lifecycle runs are slower on the prefill side.** `k19/p_side_per_repeat.py` (output `k19/p_side_per_repeat_stdout.txt`) gives the prefill side (proxy arrival to prefill response, proxy clock, without first-contact requests) per repeat:

| Run | Instrumentation | Prefill side per repeat |
| --- | --- | --- |
| P2-32r `dsv4-h200-p2-canonical-20261025-01` | none | 234.4 / 236.0 / 236.1 ms |
| `dsv4-h200-p2-lifecycle-20261025-01` (P2-32) | scheduler lifecycle | 260.7 / 255.4 / 256.4 ms |
| P2-64r `dsv4-h200-p2-64-canonical-20261025-01` | none | 556.5 / 552.4 / 547.2 ms |
| `dsv4-h200-p2-64-lifecycle-20261026-01` | scheduler lifecycle | 570.3 / 581.6 / 574.0 ms |
| `dsv4-h200-p2-64-wave-20261030-01` | scheduler lifecycle with `add_enqueue`, DP wave records | 551.5 ms (one repeat) |

Every lifecycle repeat is above every canonical repeat of its case (P2-32 by 19.3-26.3 ms, P2-64 by 13.8-34.4 ms); the wave run falls inside the canonical range. On P2-32, which does not wait, the lifecycle run's step (schedule to first-token update) averages 244.6 ms, while the canonical prefill side leaves about 222 ms for the step (235.5 ms less the ingress and the response, section 5.4). The added time therefore sits inside the engine step. Its cause in the lifecycle observer is not identified. The handoff after the prefill side is not slower in the lifecycle runs (section 3.6).

**Observation 2: the prefill step on the wave run.** `ttft/p_blackout.py` on the three instrumented runs (output `k19/p_blackout_instrumented_stdout.txt`): the prefills served at once (no queued dummies) take 300.6 ms on the P2-64 lifecycle run (n 43) and 289.6 ms on the wave run (n 6, minimum 285.6 ms). Frontier's time for such a prefill is 275.2 ms of compute, or 278.5 ms with the 3.27 ms scheduling wait from its arrival (section 3.1, K unset), about 11-14 ms (4-5%) shorter than the wave run. The wave-run sample is small.

**Observation 3: FIFO parameter swaps** (`k19/fifo_swaps/run_swaps.sh`, outputs `k19/fifo_swaps/fifo_<name>.txt`; `a10/wave/fifo.py` on the wave run, K = 19, actual mean 536.7 ms). Starting from the native set (g 23.8, c 4.68, P 300, d_sched 6.5, d_post 3; predicted 544.9 ms), one parameter at a time is set to Frontier's value:

| Swap | Predicted mean | Change |
| --- | --- | --- |
| none (native set) | 544.9 ms | 0 |
| g 24.38 (Frontier's dummy price) | 561.2 ms | +16.3 ms |
| c 0 | 545.5 ms | +0.6 ms |
| P 275.2 | 512.1 ms | −32.8 ms |
| d_sched 3.3 | 542.3 ms | −2.6 ms |
| d_post 0 | 545.7 ms | +0.8 ms |
| all (Frontier set) | 514.9 ms | −30.0 ms |
| Frontier set with P 300 | 548.8 ms | +3.9 ms |
| Frontier set with P 290 | 537.0 ms | −7.9 ms |

**Reading (inference).**

- The 300 ms native compute of sections 3.2 and 3.5, and the "about 25 ms of compute" difference of section 3.2, were measured on the slower lifecycle run. On a run with canonical timing, the step is about 286-290 ms.
- The prefill step is the parameter that matters: in the model, each ms of P moves the P2-64 prefill side by about 1.3-1.5 ms, because a later prefill end also delays the queue behind it. With Frontier's other parameters, P = 290 ms reproduces the wave-run mean (537.0 against 536.7 ms). The host time per dummy, the scheduling delay and the post-processing delay move the mean by at most 3 ms each.
- Frontier's dummy, priced 2.4% above native (24.38 against 23.81 ms, section 3.5 observation 3), raises the prefill side by about 16 ms in the model. It works against the short prefill step, so the in-engine gap of about 23 ms is smaller than the short step alone would give. The two errors are reported separately, not netted.
- Next check on the prefill compute, CPU only: compare Frontier's operator prices for the 6,144-token prefill forward with the native kernels. The profiled run cannot give the step directly: its prefill forwards on DP lanes 1-3 span a median of about 338 ms on the device, because they wait for peers in each MoE collective, while lane 0's single prefill spans 279.1 ms (busy 274.5 ms; `evidence/runtime_census/p2_64.txt`).
- The wave run's ingress (11.33 ms, section 5.4) is from the run whose timing matches the canonical runs. The lifecycle run's 9.12 ms is not larger despite that run's slowdown, which places the slowdown in the engine step, not in the ingress.

### 5.6 The prefill step on the device: Frontier's operator prices against the native kernels

**Data.** Native: the P2-64 profiled run `dsv4-h200-p2-64-profile-20261025-01`, prefill DP lane 0, TP rank 0, forward 64. It is the lane's only 6,144-token prefill in NONE mode, while the peer lanes run NONE dummies (`num_tokens_across_dp` 6144/1/1/1). Frontier: the first real prefill forward of `stage4/a10/p2_64_warm_k19b24/forwards.jsonl.gz`, which has the same shape and the same peers. Scripts and outputs are in `h3o_operator_costs/a10/k19/prefill_compute/`. They run with `PYTHONPATH=<repo>/tests/analysis` and `ulimit -v 31457280`, and they reuse `forward_kernel_events` of `tests/analysis/v4_runtime_census.py`.

**Observation 1: segment totals** (`native_kernels.py` → `p2_64_dp0_tp0.txt`, `around.py` → `around.txt`).

| Segment | Native, device | Frontier |
| --- | --- | --- |
| KV block zero (native: `_zero_kv_blocks_kernel` in the previous forward's post kernels) | 2.913 ms | 2.911 ms |
| Prepare | 67 kernels, 0.37 ms span | 0.237 ms |
| Dispatch | 2,313 kernels: sum 276.82 ms, union 274.03 ms, span 278.65 ms | 270.67 ms (scope prices 274.68 ms in total; the composition overlaps 0.74 ms of them) |
| Post | 11 kernels, 0.11 ms | 0.10 ms |
| From the end of the previous dispatch to the end of post | 282.1 ms | 273.94 ms (composed) |

**Observation 2: dispatch by scope.**

| Native kernels | Native | Frontier scope | Frontier | Frontier − native |
| --- | --- | --- | --- | --- |
| `ncclDevKernel_Reduce_Sum_bf16_RING_LL` (EP combine), 43 | 93.58 ms | `dsv4_ep_output_reduce_scatter` | 86.50 ms | −7.08 ms |
| `ncclDevKernel_AllGatherV_RING_LL` (EP dispatch), 129 | 45.08 ms | `dsv4_ep_dispatch_allgather` | 46.29 ms | +1.21 ms |
| `ncclDevKernel_AllGather_RING_LL` (SP), 44 | 7.27 ms | `dsv4_sp_attention_allgather` + `dsv4_sp_final_mhc_allgather` | 7.53 ms | +0.26 ms |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` (SP), 43 | 6.66 ms | `dsv4_sp_attention_reduce_scatter` | 6.76 ms | +0.10 ms |
| All other kernels (sparse attention 29.79, index top-k 28.61, routed MoE about 20.1 ms) | 124.23 ms | All other dispatch scopes (sparse attention 30.80, index top-k 28.68, `moe_grouped_gemm` 20.44 ms) | 124.40 ms | +0.17 ms |
| Native idle time in the span, less the multi-stream overlap | +1.83 ms | Overlap in the composition | −0.74 ms | |

Only the EP combine is far from native. The other collectives and the compute differ by +1.74 ms in total.

**Observation 3: why the native combine takes 2,176 µs per layer and Frontier prices 2,012 µs.**

- **Peer lanes are not late** (`peer_align.py` → `peer_align_f64.txt`). The host clocks of the four lanes (TP rank 0 on four hosts) are aligned by the median end time of the EP dispatch `AllGatherV` kernels; the residual standard deviation is 16-158 µs. Lane 0 waits a median of 3.5 µs per layer (0.20 ms per forward) for the last lane to reach the combine. From that last arrival to its end, lane 0's Reduce takes 2,171.2 µs (median; 93.38 ms in total).
- **TP ranks are not skewed** (`collective_calls.py` → `calls_*.json`, `align.py` → `align.txt`, `b2b_vs_model.py` → `b2b_vs_model.txt`). The 8 TP ranks of lane 0 share one host clock and start each Reduce within a median of 21.4 µs. Rank 0 ends last in all 43 layers, 2,176.5 µs after the last start, and ranks 1-7 end 1,435-1,652 µs after rank 0's start. The SP `AllGather` that follows waits for rank 0, so the Reduce and that `AllGather` together take about 101 ms on every rank.
- **No contention on the GPU** (`reduce_overlap.py` → `reduce_overlap_dp0_tp0_f64.txt`). No kernel on another stream overlaps any of the 43 Reduce kernels (0.0 ms). Each Reduce starts a median of 4.4 µs after `moe_sum`.
- **The benchmark's timing mode is the cause** (`b2b_vs_model.txt`). G5h-32 `_01` (`eager_t6144_1_1_1`, the same kernel, grid 16, 50,528,256 input bytes) times 10 back-to-back calls. From the second call on, ranks 1-7 start call k 450-720 µs before rank 0, because they finished call k−1 earlier. Rank 0's kernel then takes about 2,000 µs, and its per-call median of 2,011.1 µs is the value in `summary.csv`. In the forward, all ranks start together, and rank 0 takes 2,176.5 µs. The back-to-back value is therefore the period of a pipelined sequence of calls, not the time of one call that starts on aligned ranks. Frontier's price is 2,011.6 µs per call, 164.7 µs per layer below the native mean.
- **On EP16 the sign is reversed** (`bench_ranks.py` → `bench_ranks_ep16.txt`; native groups 7 and 8 of `evidence/collective_kernels/p2_32_native.txt`). In G5h-16 `_02` (`eager_t6144_1`), ranks 1-7 start only 60-260 µs before rank 0, and rank 0 takes 743-795 µs per call. Frontier prices the P2-32 combine at 764.4 µs per layer. Native takes 718.8 µs on lane 0 and 747.8 µs on lane 1 (rank 0, median), so Frontier is 0.7-2.0 ms per forward high. The back-to-back time is therefore below the forward's time on EP32 and above it on EP16. A correction factor would not fit both, and correction factors are not allowed. The isolated run's single traced call cannot give the aligned-start time either: its ranks start up to 6.9 ms apart (`bench_ranks_ep16.txt`, first block).

**Observation 4: the host part of the step** (`step_split.py` → `step_split.txt`; wave run). The worker's `execute_model` returns before the device finishes, so these host spans do not bound the device time. For the 32 prefill steps under 400 ms (schedule to update):

- schedule to the worker's `execute_model` start: 2.14 ms;
- `execute_model` start to the DP all-reduce: 2.08 ms;
- DP all-reduce wait: 1.23 ms.

Frontier models a 3.27 ms scheduling wait (section 3.1) and no other host time for V4. `skip_cpu_overhead_modeling` is `True`, and `data/profiling/cpu_overhead/h200_dgx/` has tables only for Qwen3-235B-A22B, Qwen3-30B-A3B-tiny and llama2_7b. The existing CPU-overhead path (`frontier.profiling.cpu_overhead.vllm_cpu_probe`) builds its table from the CPU-probe logs of Frontier's instrumented vLLM (`VLLM_FRONTIER_CPU_PROBE_LOG_PATH`). The V4 serving image does not have that probe.

**Reading.**

- Frontier's prefill step (273.94 ms) is about 15.7 ms shorter than the wave run's (289.6 ms). About 8.2 ms of that is on the device and about 7.5 ms on the host. This split is an inference: the wave run has no device trace, so its device time is taken from the profiled run (282.1 ms).
- On the device, 7.08 ms comes from the EP combine price. Its back-to-back timing does not match a call that starts on aligned ranks. The native idle time less overlap adds 1.83 ms. The other prices are within +1.74 ms in total.
- **Effect of pricing the combine at the forward's time** (arithmetic with the FIFO model of section 5.5, not a run):
  - P2-64: P rises from 275.2 to 282.3 ms. At the FIFO slope of 1.49 ms per ms (Frontier's set: P 275.2 → 514.9 ms, P 290 → 537.0 ms), the prefill side rises about 10.6 ms. TTFT becomes about 665.7 ms, −10.2% against 741.2 ms.
  - P2-32: there is little queueing, so P2-32 moves by the step change itself, about 1-2 ms down (−8.2% to −8.5%).
  - Together with the ingress modeling of section 5.4 (about +22 ms on both cases), P2-64 would be about 688 ms (−7.2%) and P2-32 about 355 ms (−2.4%).
- **Next measurement** (needs a GPU run and YC's approval): time each EP collective from an aligned start, on TP8 DP4 EP32 and TP8 DP2 EP16.
  - Each timed call follows a barrier on the group's stream.
  - The kernel trace gives each rank's end time, and the price is the latest end in the timed lane.
  - Compare the result with the native kernels of every group, decode included, before the consumer changes.

### 5.7 B28 replays (isolated eager EP rows, exact uneven dispatch)

Code: main `ee29d548`, which adds B28 (`issues.md` I-008) to the code of sections 5.1-5.6. The H200 network table takes the EP rows of every eager scenario with a 6,144-token lane from the isolated single-call runs, and `MeasuredCCBackend` prices an uneven EP dispatch from its exact `AllGatherV` row. Detached worktree `.worktrees/a10-k19-ee29d548`; launcher `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10_k19_b28.sh`; roots `stage4/a10/p2_{32,64}_warm_k19b28`. Each warm cache is a `cp -a` copy of `stage4/a10/p2_{32,64}_cold_g1hs_r2/cache` (1,758 files). There are no cold runs: the network table is read into the backend and never cached (`h3o_operator_costs/b28/cache_audit.md`), so a cold run would test nothing that B28 changes. Post-processing as in section 5 (`post_k19.sh`, `per_repeat.py`); copies of the outputs are in `h3o_operator_costs/a10/k19/` (`*_k19b28*`). The comparisons with the B24 runs are in `k19/b28/` (`interim.py`, `wave_prices.py`, `scope_diff.py`, `wave_ep_by_dp.py`; outputs `post_p2_32.txt`, `post_p2_64.txt`, `scope_diff.txt`, `wave_ep_by_dp.txt`).

| Run | Status | Wall time (init / run) | Source commit, dirty paths |
| --- | --- | --- | --- |
| P2-32 warm | FINISHED | 83.7 s / 9,576.2 s (2.66 h) | `ee29d548`, none |
| P2-64 warm | FINISHED | 87.0 s / 12,012.0 s (3.34 h) | `ee29d548`, none |

**P2-32** (gate output `stage4/a10_gate/p2_32_warm_k19b28_run/`):

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result | B24 (section 5.1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 364.030 ms | 323.179 ms | −40.851 ms | 11.22% | 10% | FAIL | 7.94% |
| TPOT mean | 19.919 ms | 19.927 ms | +0.008 ms | 0.04% | 10% | PASS | 0.04% |
| Request E2E mean | 20,741.132 ms | 20,708.281 ms | −32.851 ms | 0.16% | 10% | PASS | 0.10% |
| Token throughput (window rate) | 6,231.415 tps | 6,226.653 tps | −4.763 tps | 0.08% | 10% | PASS | 0.08% |

Per native repeat, the TTFT error is 10.72% / 11.40% / 11.55%; TPOT 0.10% / 0.19% / 0.17%; E2E 0.09% / 0.01% / 0.37%; throughput 0.12% / 0.08% / 0.03%. Overall FAIL on TTFT only, with routing MATCH.

**P2-64** (gate output `stage4/a10_gate/p2_64_warm_k19b28_run/`):

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result | B24 (section 5.2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 741.216 ms | 662.614 ms | −78.603 ms | 10.60% | 10% | FAIL | 11.62% |
| TPOT mean | 25.871 ms | 25.295 ms | −0.575 ms | 2.22% | 10% | PASS | 2.23% |
| Request E2E mean | 27,206.770 ms | 26,539.743 ms | −667.026 ms | 2.45% | 10% | PASS | 2.48% |
| Token throughput (window rate) | 5,963.206 tps | 5,976.096 tps | +12.890 tps | 0.22% | 10% | PASS | 0.22% |

Per native repeat, the TTFT error is 15.04% / 8.67% / 7.75%; TPOT 0.53% / 3.66% / 2.43%; E2E 0.95% / 3.79% / 2.57%; throughput 0.47% / 0.01% / 0.18%. Overall FAIL on TTFT only, with routing MATCH. Against the diagnostic reference without the first-contact requests (720.6 ms, section 3.6), the TTFT error is −8.05%.

**The price change per prefill forward** (`wave_ep_by_dp.py`, every wave with a 6,144-token lane, paired by order with the B24 run). In both cases, every such wave is (6,144, 1) on P2-32 and (6,144, 1, 1, 1) on P2-64 (`wave_prices.py`: 128 waves each). No A10 wave has more than one prefill lane, so the uniform rows that B28 also replaced (I-008, "Uniform rows") are never used here. The change depends only on which DP lane holds the 6,144 tokens, and it is the same for every wave of that lane:

| Case, lane of the 6,144 tokens | Waves | Dispatch (B24 → B28) | Combine (B24 → B28) | Forward change |
| --- | --- | --- | --- | --- |
| P2-32, dp0 | 64 | 34.495 → 24.112 ms | 32.869 → 31.610 ms | −11.642 ms |
| P2-32, dp1 | 64 | 34.495 → 24.448 ms | 34.369 → 31.340 ms | −13.076 ms |
| P2-64, dp0 | 32 | 46.292 → 44.873 ms | 86.502 → 93.419 ms | +5.498 ms |
| P2-64, dp1 | 32 | 46.292 → 45.523 ms | 89.342 → 93.624 ms | +3.512 ms |
| P2-64, dp2 | 32 | 46.292 → 45.103 ms | 88.884 → 94.459 ms | +4.385 ms |
| P2-64, dp3 | 32 | 46.292 → 45.606 ms | 86.410 → 92.275 ms | +5.179 ms |

- The dp0 rows equal the expected change of the B28 price check (`h3o_operator_costs/evidence/collective_kernels/expected_prices.txt`: P2-32 −10.38 / −1.26 / −11.64 ms, P2-64 −1.42 / +6.92 / +5.50 ms). That check covered only the dp0 arrangement; the other lanes use their own measured rows. In B24 the dispatch price did not depend on the lane, because the uneven dispatch was priced from the uniform curve at the largest rank's size. `scope_diff.txt` shows that only the two EP scopes change in the first wave of each case.
- Mean over the waves: P2-32 −12.359 ms, P2-64 +4.644 ms. P2-32's prefill side falls by the same 12.4 ms (222.3 → 210.0 ms): its prefills do not queue. P2-64's prefill side rises by 6.80 ms (513.3 → 520.1 ms, range 279.0-755.4 ms), 1.46 times the forward change, because a later prefill end also delays the queue behind it (section 5.5). The time after the prefill side barely moves: 113.2 ms (112.8) and 142.5 ms (141.8). Request 0 is again the slowest (P2-32 TTFT 623.5 ms).
- Against the native critical paths of I-008 (per forward of 43 layers on the dp0 arrangement): P2-32 dispatch 24.112 against 23.327 ms and combine 31.610 against 30.847 ms, together 1.55 ms high; P2-64 dispatch 44.873 against about 45.06 ms and combine 93.419 against 93.36 ms, within 0.2 ms. Section 5.6's P2-64 forward becomes 279.44 ms against the native 282.1 ms, so its device gap falls from 8.2 ms to about 2.7 ms.

**A8** (`k19/a8_p2_32_warm_k19b28_stdout.txt`, `k19/a8_p2_64_warm_k19b28_stdout.txt`):

| Run | Distinct requests | Calls | Exact | Interpolated | Unequal prefill dispatches / combines (all exact) | Violations |
| --- | --- | --- | --- | --- | --- | --- |
| P2-32 warm K19 B28 | 44 (B24 43) | 6,310,644 | 6,310,644 | 0 | 16,512 / 16,512 | 0 |
| P2-64 warm K19 B28 | 38 (B24 35) | 9,002,176 | 9,002,176 | 0 | 27,520 / 27,520 | 0 |

The uneven prefill dispatches are now priced from exact measured rows (B24: 0 unequal dispatches), which adds the distinct requests. As section 3 of the B24 report explains, these counts are predictions, not simulated calls. P2-32's count is unchanged. P2-64's falls by 22,448, and its prefill TP all-reduce predictions fall from 15,488 to 15,360. Inference, not checked: with the changed prefill timing, the prefill DP group formed fewer dummy forwards. The decode counts differ by fewer than 100 predictions per row, and B28 did not change the decode rows of the table.

**Reading.**

- B28 does what it was built for. Every EP price of a 6,144-token prefill forward now comes from a single aligned call, and on the dp0 arrangement it matches the native critical path within 1.55 ms (P2-32) and 0.2 ms (P2-64) per forward. On P2-64, the TTFT error falls from −11.62% to −10.60%.
- P2-32 passed before (−7.94%) partly because its EP16 calls were priced about 12 ms per forward too high. That error offset some of what Frontier does not model. Without it, P2-32 fails at −11.22%.
- What remains (against 364.0 and 720.6 ms; the native parts are those of section 5.4):

| Part (mean) | P2-64 native | P2-64 Frontier B28 | Difference | P2-32 native | P2-32 Frontier B28 | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| Client dispatch to proxy arrival | 0.6 ms | 0 | −0.6 ms | 0.5 ms | 0 | −0.5 ms |
| Prefill side | 552.0 ms | 520.1 ms | −31.9 ms | 235.5 ms | 210.0 ms | −25.5 ms |
| After the prefill side, to the client | 168.0 ms | 142.5 ms | −25.5 ms | 128.1 ms | 113.2 ms | −14.9 ms |
| TTFT | 720.6 ms | 662.6 ms | −58.0 ms | 364.0 ms | 323.2 ms | −40.9 ms |

- Inference, from sections 5.4-5.6:
  - The path outside the engine (about 22 ms per request; section 5.4) is still not modeled.
  - The host part of the prefill step (about 7.5 ms on P2-64, inferred in section 5.6) is still not modeled for V4 (D2 (c): explicit zero).
  - On P2-32, the step itself is now about 12 ms shorter than native: about 222 ms native (section 5.5) against 210.0 ms.
- **Expected effect of ingress option A** (section 5.4; arithmetic, not a run). Adding about 22 ms per request outside the engine shifts every engine arrival by the same amount, so the queues do not change:
  - P2-32: about 345 ms (−5.2%);
  - P2-64: about 685 ms (−7.6% against 741.2 ms).
  - Both would PASS. This needs new delay fields, a configuration schema change, and YC's approval (D-NO-NEW-MODELING).
- **Host part.** Modeling the host part would revisit D2 (c), which keeps V4 host time at zero. It needs host-time data for V4: the CPU probe of `frontier.profiling.cpu_overhead.vllm_cpu_probe` would have to be ported to the V4 serving image, followed by a GPU run.
