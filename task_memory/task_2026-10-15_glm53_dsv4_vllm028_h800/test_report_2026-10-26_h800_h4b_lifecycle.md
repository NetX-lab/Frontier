# Test report 2026-10-26: H4b-(a) D-side handoff lifecycle measurement on H800-32 (native06 lifecycle)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: run, copy, R4 PASS, observer acceptance, client metrics against native06, parts a-g with the d split, comparison with the H200 run, persistence and the H4b-(b) inputs. |

**Authority.** D-H4A-CHECKPOINT (requirements item 14, H4b-(a)); D-NO-INTERRUPT (item 19); D-H200-MAINLINE-CONTINUE (item 21, Pi route restored). Label DIAGNOSTIC: this is the H800 evidence for the H4b-(b) design. It never qualifies an H200 cell. Design: [h4b_lifecycle_measurement/design.md](h4b_lifecycle_measurement/design.md) sections 4 and 8 (acceptance items 1-5).

## 1. Run and copy

| Item | Value | Source |
| --- | --- | --- |
| Job | `exp-1025-121318-827142`, submitted 12:13 UTC, queued about 8 h 14 min for H800 quota, containers ran 20:27-20:52 UTC, Succeeded | `h4b_lifecycle_measurement/inputs/h800l/platform_terminal_receipt.json` |
| Platform | creator `i-fengyicheng`, `codesign`, preemptible `no`, 4 replicas Succeeded on `gpu-h800-0133/0179/0209/0339`, every container exit 0, restart 0, the locked image `vllm-openai@sha256:2286e853…` | same receipt |
| Node roles | 0133 prefill lane 0 (head), 0179 prefill lane 1 (headless), 0209 decode lane 0 (head), 0339 decode lane 1 (headless) | `nodes/*/drained.json`, `nodes/*/result.json` |
| Launcher | pid 348050, `launcher_exit.json` exit 0 at 20:52:27 UTC | `h4b_lifecycle_measurement/package_01/launcher_exit.json` |
| Copy | Pi label `h800-lifecycle-copy-1025d` turn 4 (prompt `/data/ycfeng/tmp/ds4_runs/prompts/h800-lifecycle-copy-1025d.md`, after YC restored the Pi budget; turns 1-3 had ended on the model API's 402): terminal receipt, read-only `ssh shai-ycfeng` inventory plus `sudo tar` stream, local rehash `HASH_IDENTICAL_DOWNLOAD`, 70 files, 48,588,133 B, inventory `a37e4136…7ba0` | `/data/ycfeng/tmp/glm53_dsv4/h4b/downloads/dsv4-h800-p0-lifecycle-20261025-01/download_receipt.json` |
| Cloud origin | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_lifecycle_20261025/case/runs/groundtruth_clean/dsv4-h800-p0-lifecycle-20261025-01` | same receipt |
| Evaluation case | `/data/ycfeng/tmp/glm53_dsv4/h4b/cases/h800_lifecycle_case_20261025/` (`package_01/case` plus the run copy; 70 of 70 copied files hash-equal to the download) | — |

## 2. Acceptance results

| # | Item | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Platform Succeeded and R4 PASS on the copy | all R4 checks PASS | platform Succeeded; R4 status PASS, all 8 checks PASS (section 3) | PASS |
| 2 | Observer records | 4 engine cores, a `start` and no `error` record each, full timeline for every formal D request | 4 cores (prefill dp0/dp1, decode dp0/dp1), 0 `error`, 4 `stop` records; 384 of 384 formal requests complete; every D request id matches; every READ in WAITING_FOR_REMOTE_KVS; every promotion step k=1 | PASS |
| 3 | Client metrics against native06 r1-r3 | reported, not gated | TTFT, TPOT, E2E and THPT outside the native06 range in all three repetitions; the proxy handoff in two (section 4) | reported |
| 4 | Parts a-g and iterations per lane state | reported | section 5 | reported |
| 5 | Results in the task directory with cloud paths and hashes | copies verified | section 7 | PASS |

## 3. R4 result

Command (`/data/ycfeng/envs/asyncopd-py312/bin/python`, Python 3.12):

```bash
C=/data/ycfeng/tmp/glm53_dsv4/h4b/cases/h800_lifecycle_case_20261025; TAG=dsv4-h800-p0-lifecycle-20261025-01
(ulimit -v 31457280; env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 TMPDIR=/data/ycfeng/tmp \
  timeout 1800 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py $C $TAG clean \
  > /data/ycfeng/tmp/glm53_dsv4/h4b/r4/h800_lifecycle.out 2> /data/ycfeng/tmp/glm53_dsv4/h4b/r4/h800_lifecycle.err)
```

Output: `{"run_tag": "dsv4-h800-p0-lifecycle-20261025-01", "status": "PASS", "failed": [], ...}`. Status file `h4b_lifecycle_measurement/inputs/h800l/run_check_status.groundtruth_clean.json` (checked 04:39:14 UTC).

| Check | Verdict |
| --- | --- |
| `checkout`, `inputs_unchanged` | PASS |
| `harness_pinned_and_client_rows` | PASS: 128 formal client rows in each of the 3 repetitions |
| `requests_and_idle_lanes` | PASS: 7 boundaries, both roles, both DP lanes |
| `native_workers` | PASS: 4 distinct native nodes, every role and DP lane; source, model and GPU verified |
| `native_lifecycle` | PASS: every node drained, heads stopped after all drains, headless nodes after their heads, return codes 0 |
| `rank_runtime_graph` | PASS: 32 physical ranks, 6 phases each |
| `terminal_status` | PASS: Succeeded, all containers exit 0, no restarts |

Shutdown order, from the Pi fact sheet (UTC; `drained.json` mtime, first `[shutdown]` line, `native_stopped.json` mtime): 0133 20:51:23 / 20:51:35 / 20:51:50; 0209 20:51:23 / 20:51:41 / 20:51:56; 0179 20:51:24 / 20:51:51 / 20:52:05; 0339 20:51:24 / 20:51:57 / 20:52:12. Each head began its shutdown 12-18 s after the drains, and each headless node was signalled 1 s after its head returned. The H200 run's late decode head (120 s) does not occur here, so its shutdown-order failure is not a defect of the harness on every cluster.

## 4. Client metrics against native06 (acceptance 3)

Tools: the canonical client metrics inside `analyze_lifecycle.py` (`client_metrics.native_client_rows`). Reference: `h4a_p0_diagnostic/inputs/native06` (R4 PASS).

| Metric | Run r1 / r2 / r3 | native06 range r1-r3 | Run mean against reference mean |
| --- | --- | --- | --- |
| TTFT (ms) | 378.82 / 374.31 / 374.01 | 393.22-403.31 | −5.8% |
| TPOT (ms) | 19.998 / 20.017 / 20.028 | 18.981-19.031 | +5.3% |
| E2E (ms) | 20836.7 / 20851.7 / 20862.3 | 19817.2-19872.1 | +5.1% |
| Total THPT (tok/s) | 6231.1 / 6227.0 / 6222.4 | 6269.8-6273.3 | −0.7% |
| Proxy handoff (ms) | 126.28 / 126.44 / 124.57 | 121.14-124.90 | +1.8% |

TPOT is 5.3% higher, while TTFT is 5.8% lower, so the run is not uniformly slower. The run used different hosts (0133/0179/0209/0339 against 0214/0310/0346/0399), and there is one run of each. Decode-step intervals while 4 or more requests are cached (`results/h800l/step_gaps.txt`): p50 19.83 / 19.83 ms, p90 20.27 / 20.31 ms, p99 24.92 / 24.95 ms on decode dp0 / dp1. 88 and 90 of 23,453 and 23,356 gaps exceed 30 ms (maximum 57.0 / 56.5 ms), clustered around the same wall-clock seconds on both lanes. There is no periodic stall that would point to observer flushing. The wrapped `schedule()` call takes 0.26 / 0.25 ms on average and `update_from_output()` 0.09 / 0.10 ms. As on H200, the observer effect is not separated from the host effect.

## 5. Parts a-g (acceptance 4)

Command (`/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6):

```bash
O=/data/ycfeng/tmp/glm53_dsv4/h4b/lifecycle/analysis_h800
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 timeout 900 /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement/analyze_lifecycle.py \
  $C/runs/groundtruth_clean/$TAG task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/inputs/native06 $O > $O.out 2> $O.err)
(ulimit -v 31457280; timeout 600 /usr/bin/python3 \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/results/p2_32l/step_gaps.py \
  $C/runs/groundtruth_clean/$TAG > /data/ycfeng/tmp/glm53_dsv4/h4b/lifecycle/step_gaps_h800.txt)
```

Parts as in the H200 report: a proxy P response → D API server; b → engine `add`; c → READ schedule; d → `finished_recving` update; e → promotion schedule; f → first-token update return; g → proxy first D byte. The remainder a + b + g is clock-free.

Means on decoding lanes (378 requests), with the H200 run (P2-32 lifecycle) beside them:

| Part | H800 mean (ms) | p50 | p95 | H200 mean (ms) |
| --- | --- | --- | --- | --- |
| a | 4.00 | 3.91 | 4.94 | 0.96 |
| b | 13.31 | 12.71 | 24.26 | 13.61 |
| c | 0.058 | 0.050 | 0.090 | 0.056 |
| d | 65.61 | 60.99 | 99.15 | 77.94 |
| e | 0.22 | 0.17 | 0.33 | 0.19 |
| f | 41.80 | 41.47 | 43.37 | 42.05 |
| g | 0.90 | 0.90 | 1.60 | 3.67 |
| engine c..f | 107.70 | 102.84 | 140.82 | 120.24 |
| remainder a+b+g | 18.21 | 17.55 | 29.20 | 18.23 |
| proxy handoff | 125.91 | 122.50 | 156.02 | 138.47 |
| engine iterations c..e | 4.26 | 4 | 6 | 4.81 |
| d iterations | 2.26 | 2 | 4 | 2.81 |
| READ transfer max over the 8 handles | 27.25 | 22.68 | 60.25 | 38.32 |

Idle lanes (6 requests): d 26.3 ms, f 67.3 ms, 4.33 iterations c..e. By decode DP rank (decoding lanes): b 12.53 / 13.91 ms, g 1.07 / 0.71 ms, remainder 17.57 / 18.64 ms on dp0 (0209) / dp1 (0339). The a and g values differ from H200 in opposite directions (a +3.0 ms, g −2.8 ms); they cross hosts and depend on clock synchronization. Their sum with b, the remainder, is 18.2 ms on both clusters.

**d split by the number of D iterations up to `finished_recving`** (from `results/h800l/lifecycle_requests.jsonl`, decoding lanes):

| d iterations | Requests | d (ms) | READ transfer max (ms) | d − transfer max (ms) | Spread over the 8 handles (ms) |
| --- | --- | --- | --- | --- | --- |
| 2 | 307 | 60.6 | 22.3 | 38.4 | 1.4 |
| 3 | 46 | 79.7 | 41.7 | 38.1 | 20.8 |
| 4 | 24 | 100.3 | 61.4 | 38.9 | 40.4 |
| 5 | 1 | 119.5 | 80.4 | 39.1 | 58.7 |

Observed:
- As on H200, d is a constant of about 38.4 ms, about 2 decode steps of 19.9 ms, plus the READ transfer max. The transfer max grows by about 20 ms, one step, per further iteration.
- On H800 no request falls in the 1-iteration bucket that H200 had (45 requests, transfer max 1.9 ms). 81% of the H800 requests have 2 iterations; their 8 handles all report about 22 ms, within 1.4 ms of each other.
- f is one engine iteration and about 2 decode steps (41.8 ms), as on H200.
- The READ windows of different requests never overlap (`read_window_shared` false for all 384). Every request moves 8 handles and 368,824,320 B, the same as on H200.

Inference: NIXL READ completion is detected only at step boundaries on both clusters. On H800 the common case is completion detected one step after the post; on H200 it was split between zero and one step. Handles that finish late add whole steps on both. Why some handles finish late is not determined by these runs.

**Expectation written before the run** (H4b-(a) design section 8), compared:

| Expectation | Measured | Verdict |
| --- | --- | --- |
| c at most 1 iteration | c 0.058 ms, inside the READ step | met |
| c + d + e about 3 iterations | 3.26 iterations (4.26 c..e minus the f iteration) | met |
| f about 30-40 ms | 41.80 ms | slightly above |
| c to f about 75-105 ms | 107.70 ms | slightly above |
| remainder a + b + g about 10-25 ms | 18.21 ms | met |

## 6. Inputs to H4b-(b)

Frontier today (H4a run5 on native06, `h4a_p0_diagnostic/results/h4a5_attribution.json`): the D-side handoff is the KV transfer, 2.57 ms, plus the first D step, 39.71 ms, so 42.28 ms in total, against the native handoff of 123.49 ms (gap −81.22 ms). On this run the handoff is 125.91 ms on decoding lanes:
- f (41.8 ms) matches Frontier's first D step.
- d holds about 2 steps (38.4 ms) that Frontier does not model, plus a step-quantized READ completion of 27.3 ms on average, where Frontier models 2.57 ms. Item 1 (D-H4B-ITEM1) models the 2 steps and the quantization of the wire time at step starts; it does not model the late handles.
- The remainder a + b + g (18.2 ms) is outside the engine and missing in Frontier. It is the same on both clusters, which supports a measured per-platform ingress input later (the deferred ingress field).

The design section 7 expectation for item 1 alone on H800 P0 (TTFT 366.03 ms, −8.2%) used S_F and the analytical T; the engine_step diagnostic run checks it.

## 7. Persistence (acceptance 5, D-INPUT-PERSISTENCE)

- Inputs: `h4a_p0_diagnostic/copy_inputs.py h800_lifecycle` writes the set `h800l` into `h4b_lifecycle_measurement/inputs/h800l/`, with its own `manifest.json` (`3d2866d5…35e1`), so the H4a manifest stays unchanged. 20 files, 17,170,207 B: the 7 run files, the two receipts, the 4 lifecycle files, the 4 `native_stopped.json`, the R4 status, the download receipt and the cloud inventory.
- 15 of the 15 run and node files match the cloud inventory digests. The other 5 (`launcher_receipt.json`, `platform_terminal_receipt.json`, the R4 status, the download receipt and the cloud inventory) were written locally and are not in the cloud inventory.
- Lifecycle file digests: 0209 decode dp0 starts `4b3e4cbfb04966a5`, 0339 decode dp1 `590016bc0c7c8e10`, 0133 prefill dp0 `1f67f4c55179b7a9`, 0179 prefill dp1 `eeb5c56f284d42d9`; full digests in the manifest. The 4 lifecycle files (16.2 MB) stay in the task directory but out of Git, as on H200.
- Committed in `3eee44b7`. Results `h4b_lifecycle_measurement/results/h800l/` (SHA-256):

  | File | SHA-256 |
  | --- | --- |
  | `lifecycle_summary.json` | `1b30f02ae838f0b595dd0ab96f05a9295519d0d26a22ba622e0f080686f46582` |
  | `lifecycle_requests.jsonl` | `a822c0b7572bd426fc1f4907781227d78bd3a5a2d823216fa46316d7a1dafaad` |
  | `analyze_lifecycle.out` | `eaae478f99aa026ae320c3f1d0a3671d184bb0767ac7d5eaf1840286cd7c509a` |
  | `step_gaps.txt` | `99d5d44234e0c45822927709facd7ef0b9688c9866ebf8c2ca576e1f1cdb6cf9` |

## 8. Limitations

- One run on H800. It never qualifies an H200 cell, and the H200 run never qualifies an H800 cell.
- The client metrics differ from native06 by −5.8% (TTFT) to +5.3% (TPOT) on different hosts. The observer effect is not separated from the host effect.
- The cross-host parts a, b and g depend on unmeasured clock synchronization; only their sum is clock-free.
- The NIXL `transfer_duration` is the worker's handle duration, quantized at step boundaries. The wire time of a late handle is not observed.
