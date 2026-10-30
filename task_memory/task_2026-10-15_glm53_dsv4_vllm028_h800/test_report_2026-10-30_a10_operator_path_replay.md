# Test report: A10 operator-path E2E replay of P2-32 and P2-64 on H200 (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Added section 3: the P2-64 TTFT gap is on the native prefill side, where a prefill that arrives after the previous DP wave ended finishes about 450-540 ms after that wave end (mechanism not found in the code). |
| 2026-10-30 | Created: P2-64 client reading on the pre-supplement table (probe `p2_64_c`), the loss of the promoted-table runs to the full disk (I-003), and their restart. |

## Scope

A10 replays the two H200 P2 cases on the V4 operator path (D-OPERATOR-LEVEL-ROUTE, ADR 0002: no whole-step cost table) and reads the client metrics against the persisted native vLLM reference with `tests/comparison/e2e_gate/run_gate.py`. Each case runs twice: cold (empty predictor cache, the simulator trains every model) and warm (cache from a cold run, no fitting).

- Driver: `h3o_operator_costs/a10/a10_driver.py <case> <root>`, launched by `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh <case> <name>` (root `stage4/a10/<name>`, `ulimit -v 62914560`, `timeout 86400`).
- Gate: `h3o_operator_costs/a10/a10_gate.py <case> <frontier run dir> <output dir>`, run from the `a10/` directory with `PYTHONPATH=$REPO:$A10:$REPO/tests/analysis`.
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

Reading: overall FAIL, on TTFT only. The native TTFT spread across repeats is 718-780 ms, so the Frontier value of 417 ms lies far outside the native run-to-run variation. Frontier predicts the first token too early by about 324 ms on average; the decode-side metrics agree within 5%. The cause is not yet diagnosed.

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

Results: pending.

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

The native prefill-side value comes from the lifecycle run and the native TTFT from the three reference repeats. The lifecycle run's own TTFT mean is 757.7 ms, so its remainder is about 189 ms; the split of the 324 ms between the two parts is accurate to about 16 ms.

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

In vLLM-BS `2cf0a691` the prefill engines run one data-parallel wave together: each engine keeps running, with dummy batches when it has no request, until the all-reduce every 32 steps (`ParallelConfig.sync_dp_state`) finds no unfinished request on any engine; then the wave ends and the engines go idle. The groups above fit this picture:

- The previous prefill's KV `finished_sending` arrives about 48 ms after its end. The last request leaves the engines then, and up to 32 more dummy steps of about 5 ms run before the next sync ends the wave, about 200-250 ms after the previous end.
- A request that arrives while the wave still runs is served at once (the 43 prefills of about 300 ms).
- A request that arrives after the wave ended cannot finish its forward until about 450-540 ms after the wave end (the 36 held prefills and the 302 prefills of the 260-760 ms group).

The wake-up path in the code looks immediate: the front end sends `FIRST_REQ` when no wave runs, the coordinator broadcasts `START_DP_WAVE`, an engine that receives a request for a finished wave sends `start_wave`, and the idle engines start dummy steps as soon as `engines_running` is set. The late part is therefore not explained by the source reading. Candidates are the inter-process messaging, the coordinator, the worker processes, or the first collective of a new wave on the GPU side. Which one it is needs a measurement.

Frontier already models the wave mechanics (`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`: `VLLM_DP_SYNC_INTERVAL = 32`, dummy steps on idle MoE / `attn_dp > 1` lanes, the wave end at the sync step), but it has no cost for restarting a wave, so its prefill compute stays flat at 275 ms. The modeling rule (no empirical scaling factors) means the gap can be closed only after the mechanism is known.

### 3.4 Warnings that do not apply

The runs log approximation factors for `attn_pre_proj` and `attn_post_proj` (0.5, BF16 to FP8) and `moe_gating_routing_topk` (2.0, BF16 to FP32) from `frontier/config/quantization_manager.py`, and a missing `attention_kernel_only.csv`. The V4 path prices its attention from `attention_graph_replay_kernel_only.csv` through `dsv4_predictor.py` and does not use those three operator predictions, so these warnings do not change the A10 results.
