# Test report 2026-10-25: H4b-(a) D-side handoff lifecycle measurement on H200-32 (P2-32 lifecycle)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: run, copy, R4 result and its shutdown-order failure, observer acceptance, client metrics against P2-32r, parts a-g, the analysis-script fix, persistence and the H4b-(b) inputs. |

**Authority.** D-H200-H4B-LIFECYCLE (requirements item 18) under D-H200-TEMP-MAINLINE; D-NO-INTERRUPT (item 19). Label DIAGNOSTIC: this H200 evidence feeds the H4b-(b) design and never qualifies an H800 cell. Design: [h200_mainline/design.md](h200_mainline/design.md) section 8 and [h4b_lifecycle_measurement/design.md](h4b_lifecycle_measurement/design.md) sections 4 and 8 (acceptance items 1-5, with P2-32r in the place of native06).

## 1. Run and copy

| Item | Value | Source |
| --- | --- | --- |
| Job | `exp-1026-031327-540424`, submitted 19:13:27 UTC, Succeeded | `h200_mainline/p2_32_lifecycle/package_02/case/runs/groundtruth_clean/dsv4-h200-p2-lifecycle-20261025-01/platform_terminal_receipt.json` |
| Platform | creator `i-fengyicheng`, `step_main`, preemptible `no`, 4 replicas Succeeded on `gpu-h200-0043/0628/0705/1191`, every container exit 0, restart 0, the locked image `vllm-openai@sha256:2286e853…` | same receipt |
| Launcher | pid 3710657, `launcher_exit.json` exit 0 | `p2_32_lifecycle/package_02/launcher_exit.json` |
| Copy | Pi label `h200-lifecycle-submit-1025b` turn 3 (prompt `/data/ycfeng/tmp/ds4_runs/prompts/h200-lifecycle-submit-1025b-turn3.md`), read-only `ssh shai-ycfeng` inventory plus `sudo tar` stream, local rehash: `HASH_IDENTICAL_DOWNLOAD`, 70 files | `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-lifecycle-20261025-01/download_receipt.json` |
| Cloud origin | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_lifecycle_20261025_01/case/runs/groundtruth_clean/dsv4-h200-p2-lifecycle-20261025-01` | same receipt |
| Evaluation case | `/data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_32l_case_20261025/` (package case files plus the run copy, as for P2-32r) | — |

## 2. Acceptance results

| # | Item | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Platform Succeeded and R4 PASS on the copy | all R4 checks PASS | platform Succeeded; R4 status FAIL, `failed` = `native_workers`, `native_lifecycle`, both on `gpu-h200-1191`; all serving-window checks PASS (section 3) | FAIL (shutdown order only) |
| 2 | Observer records | 4 engine cores, a `start` and no `error` record each, full timeline for every formal D request | 4 cores (prefill dp0/dp1, decode dp0/dp1), 0 `error`, 4 `stop` records; 384 of 384 formal requests complete; every D request id matches; every READ in WAITING_FOR_REMOTE_KVS; every promotion step k=1 | PASS |
| 3 | Client metrics against P2-32r r1-r3 | reported, not gated | all five metrics outside the P2-32r range in all three repetitions (section 4) | reported |
| 4 | Parts a-g and iterations per lane state | reported | section 5 | reported |
| 5 | Results in the task directory with cloud paths and hashes | copies verified | section 7 | PASS |

## 3. R4 result and the shutdown-order failure

Command (`/data/ycfeng/envs/asyncopd-py312/bin/python`, Python 3.12):

```bash
(ulimit -v 31457280; env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 TMPDIR=/data/ycfeng/tmp \
  timeout 1800 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py \
  /data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_32l_case_20261025 dsv4-h200-p2-lifecycle-20261025-01 clean \
  > /data/ycfeng/tmp/glm53_dsv4/h200/r4/p2_32l.out 2> /data/ycfeng/tmp/glm53_dsv4/h200/r4/p2_32l.err)
```

Status file: `h200_mainline/diagnostic/inputs/p2_32l/run_check_status.groundtruth_clean.json` (checked 19:44:15 UTC).

| Check | Verdict |
| --- | --- |
| `checkout`, `inputs_unchanged` | PASS |
| `harness_pinned_and_client_rows` | PASS: 128 formal client rows in each of the 3 repetitions |
| `requests_and_idle_lanes` | PASS: 7 boundaries, both roles, both DP lanes |
| `rank_runtime_graph` | PASS: 32 physical ranks, 6 phases each |
| `terminal_status` | PASS: Succeeded, all containers exit 0, no restarts |
| `native_workers` | FAIL: `UNVERIFIED: ValueError: Native failure, observation error, weight files or runtime/source mismatch: gpu-h200-1191.lgcm.sh.istep.fun` |
| `native_lifecycle` | FAIL: `UNVERIFIED: ValueError: Headless node stopped before its head: gpu-h200-1191.lgcm.sh.istep.fun` |

Evidence (persisted copies under `h200_mainline/diagnostic/inputs/p2_32l/nodes/`):
- `gpu-h200-1191/result.json`: `observation_errors` = `["Head shutdown marker missing; native stop order unverified"]`, `server_stop` = SIGINT, return code 0.
- `native_stopped.json`: every node SIGINT, return code 0, `drained_before_stop` true. Both heads (0043 prefill, 0705 decode) list all 4 nodes as drained before the head stop. Prefill headless 0628: `head_stopped_before_headless_stop` true. Decode headless 1191: false.
- Times from the run copy (`drained.json` file mtime, whole seconds after the tar stream; first `Waiting up to 30 seconds for processes to exit` line in each `server.log`):

  | Host | Role | `drained.json` | First shutdown line | `native_stopped.json` |
  | --- | --- | --- | --- | --- |
  | 0043 | prefill head | 19:38:00 | 19:38:01 | 19:38:14 |
  | 0628 | prefill headless | 19:38:00 | 19:38:14 | 19:38:28 |
  | 0705 | decode head | 19:38:00 | 19:40:00 (`server.log:1144`) | 19:40:15 |
  | 1191 | decode headless | 19:38:00 | 19:39:31 (`server.log:394`) | 19:39:44 |

- The same reading on the four earlier H200 runs (P2-32r and P2-64r R4 PASS; the profile runs were not R4-checked): every node logs its first shutdown line 2-40 s after its drain, and the decode heads 2, 19, 6 and 7 s after it (heads identified by a non-empty `all_nodes_drained_before_head_stop`). The 120 s of decode head 0705 occur in this run only.

Mechanism (`package_02/native_pd_worker.py`, `vllm_replay.py:601-610`): the lane-0 head waits for every `drained.json`, then calls `stop_server`; a headless node waits `stop_timeout_s` = 90 s for the head's `native_stopped.json` and otherwise stops itself and records the observation error. The decode head's stop came later than this 90 s window, so 1191 stopped first.

On 0705 the 120 s lie between its own `drained.json` and SIGINT. In that interval the head runs only `await_condition` on `len(glob('nodes/*/drained.json')) == 4`, polling the cloud volume once per second (`native_pd_worker.py:356-358`), and then calls `stop_server`. The worker log has stage names but no timestamps (`worker.log`: `drain_returned`, then `finished`).

Inference, not verified: 0705 saw the last peer's `drained.json` late through the JuiceFS mount, which would be a metadata-visibility delay on that host. The observer runs inside the engine core process, not in this worker loop; its last decode-head records are updates at 19:37:54 and its `stop` record at 19:40:11, after SIGTERM. Every check of the serving window passes; the failure is limited to the shutdown order after the workload ended.

## 4. Client metrics against P2-32r (acceptance 3)

Tools: the canonical client metrics inside `analyze_lifecycle.py` (`client_metrics.native_client_rows`). Reference: `h200_mainline/diagnostic/inputs/p2_32r` (R4 PASS).

| Metric | Run r1 / r2 / r3 | P2-32r range r1-r3 | Run mean against reference mean |
| --- | --- | --- | --- |
| TTFT (ms) | 399.09 / 395.27 / 397.01 | 361.97-365.37 | +9.1% |
| TPOT (ms) | 20.427 / 20.563 / 20.355 | 19.889-19.961 | +2.7% |
| E2E (ms) | 21295.6 / 21430.8 / 21219.9 | 20710.8-20785.9 | +2.8% |
| Total THPT (tok/s) | 6218.6 / 6207.4 / 6206.5 | 6228.7-6234.1 | −0.3% |
| Proxy handoff (ms) | 137.7 / 139.3 / 140.0 | 127.0-128.6 | +8.6% |

Every metric is outside the reference range in every repetition. Per the H4b design, this is reported as a possible observer effect. Two other differences limit that reading: the run used different hosts (0043/0628/0705/1191 against 0082/0146/0710/0938), and there is one run of each. Decode-step intervals while 4 or more requests are cached (`results/p2_32l/step_gaps.txt`): p50 20.28 / 20.26 ms, p99 23.68 / 22.95 ms on decode dp0 / dp1. Each lane has one gap of about 500 ms (516 / 483 ms, at the same wall-clock time on both lanes). 65 and 56 gaps exceed 30 ms, out of 22,858 and 23,162. There is no periodic stall that would point to observer flushing. The wrapped `schedule()` call takes 0.30 / 0.22 ms on average and `update_from_output()` 0.11 / 0.07 ms.

## 5. Parts a-g (acceptance 4)

Command (`/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6):

```bash
O=/data/ycfeng/tmp/glm53_dsv4/h200/lifecycle/analysis_v2
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 timeout 900 /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement/analyze_lifecycle.py \
  /data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_32l_case_20261025/runs/groundtruth_clean/dsv4-h200-p2-lifecycle-20261025-01 \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/inputs/p2_32r $O > $O.out 2> $O.err)
```

Parts (H4b design section 4): a proxy P response → D API server; b → engine `add`; c → READ schedule; d → `finished_recving` update; e → promotion schedule; f → first-token update return; g → proxy first D byte. The clock-free remainder is the proxy handoff minus the engine span c..f, that is a + b + g.

Means on decoding lanes (378 requests); all 384 requests in brackets where different:

| Part | Mean (ms) | p50 | p95 |
| --- | --- | --- | --- |
| a | 0.96 | 0.79 | 2.19 |
| b | 13.61 | 13.45 | 23.58 |
| c | 0.056 | 0.050 | 0.088 |
| d | 77.94 (77.51) | 77.55 | 122.74 |
| e | 0.19 (0.37) | 0.16 | 0.28 |
| f | 42.05 (42.88) | 42.25 | 43.81 |
| g | 3.67 | 3.67 | 4.43 |
| engine c..f | 120.24 | 118.51 | 164.97 |
| remainder a+b+g | 18.23 | 17.65 | 28.21 |
| proxy handoff | 138.47 | 134.62 | 181.38 |
| engine iterations c..e | 4.81 | 5 | 7 |
| d iterations (after READ, up to `finished_recving`) | 2.81 | 3 | 5 |
| READ transfer max over the 8 handles | 38.32 | 39.46 | 83.20 |

Idle lanes (6 requests): d 50.4 ms, f 95.3 ms, 7.17 iterations c..e.

By decode DP rank (decoding lanes): b 13.45 / 13.76 ms, g 4.07 / 3.28 ms, remainder 18.45 / 18.03 ms on dp0 (host 0705) / dp1 (host 1191). The g difference crosses hosts and depends on clock synchronization; the remainder does not.

**d split by the number of D iterations up to `finished_recving`** (computed from `results/p2_32l/lifecycle_requests.jsonl`):

| d iterations | Requests | d (ms) | READ transfer max (ms) | d − transfer max (ms) | Spread over the 8 handles (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 45 | 41.8 | 1.9 | 39.9 | 0.5 |
| 2 | 140 | 61.6 | 22.0 | 39.6 | 13.1 |
| 3 | 73 | 81.9 | 42.3 | 39.7 | 21.8 |
| 4 | 82 | 101.6 | 62.1 | 39.5 | 41.5 |
| 5 | 36 | 121.1 | 81.7 | 39.3 | 61.0 |
| 6 | 2 | 137.6 | 99.4 | 38.2 | 78.6 |

Observed:
- d is a constant of about 39.6 ms, about 2 decode steps of 20.35 ms (p50 20.52 ms), plus the READ transfer max.
- The transfer max is itself quantized at the step size. It is 1.9 ms when all 8 handles finish within the first step, which is close to Frontier's analytical wire time of 2.57 ms. It grows by about 20 ms for each further step.
- The spread across the 8 handles grows with the step count.
- f is always one engine iteration and about 2 decode steps (42 ms).
- The READ windows of different requests never overlap (`read_window_shared` false for all 384). Every request moves 8 handles and 368,824,320 B.

Inference: the NIXL READ completion is detected only at step boundaries. The extra steps come from handles that finish late, not from the wire time of one handle. Why some handles finish late is not determined by this run.

**Expectation written before the run** (H4b design section 8), compared:

| Expectation | Measured | Verdict |
| --- | --- | --- |
| c at most 1 iteration | c 0.056 ms, inside the READ step | met |
| c + d + e about 3 iterations | 3.81 iterations (4.81 c..e minus the f iteration) | above |
| f about 30-40 ms | 42.05 ms | slightly above |
| c to f about 75-105 ms | 120.24 ms | above |
| remainder a + b + g about 10-25 ms | 18.23 ms | met |

## 6. Analysis-script fix

`h4b_lifecycle_measurement/analyze_lifecycle.py` took the NIXL `kv_stats` only from the `finished_recving` update. The worker-side stats are not keyed by request, and each update carries the stats of the handles that finished in that step. Design section 4 asks for the READ transfer time of each request, so the script now collects the stats of every update in the window (READ schedule, `finished_recving` update]. It marks `read_window_shared` when the window overlaps another READ on the same lane, and adds `d_iterations` (engine iterations scheduled after the READ step, up to the `finished_recving` update) and `read_transfer_max_ms`.

Checks:
- Synthetic fixture from the H4b CPU check, rerun into `/data/ycfeng/tmp/glm53_dsv4/h4b/analysis_check/fixture2_out3` against native06: all earlier keys unchanged; new values d_iterations 1, shared false, transfer max 12.3 ms, bytes 4096. PASS.
- Hand-built overlap case (two requests with overlapping READ windows): the expected stats per request, shared true, d_iterations 2 and 3, iterations 4 and 5. PASS.
- H200 run: every part of the earlier output (`/data/ycfeng/tmp/glm53_dsv4/h200/lifecycle/analysis/`) is identical in the new output; only the transfer fields changed. PASS.

## 7. Persistence (acceptance 5, D-INPUT-PERSISTENCE)

- Inputs: `h4a_p0_diagnostic/copy_inputs.py h200_p2` added the set `p2_32l`: 21 files, 18,333,938 B, in `h200_mainline/diagnostic/inputs/p2_32l/`. They are the 7 run files, the two receipts, the 4 lifecycle files, the 4 `native_stopped.json`, decode headless `result.json`, the R4 status, the download receipt and the cloud inventory. The earlier sets in `manifest.json` are unchanged.
- 16 of the 18 run files match the cloud inventory digests. `launcher_receipt.json` and `platform_terminal_receipt.json` are not in the cloud inventory, as for P2-32r: the launcher and Pi wrote them locally into the package.
- Lifecycle file digests: 0705 decode dp0 starts `aae1a3aee5c2db06`, 1191 decode dp1 `33282073621dfd02`; full digests in `manifest.json`. The 4 lifecycle files (17.3 MB) stay in the task directory but out of Git, as the decode torch traces do; their hashes and cloud origin are in the committed manifest.
- Results `h200_mainline/diagnostic/results/p2_32l/` (SHA-256):

  | File | SHA-256 |
  | --- | --- |
  | `lifecycle_summary.json` | `f24b72614dc17f31372870c042ee3a8153d6e40edf470f4be8b7a19f0bb6dab1` |
  | `lifecycle_requests.jsonl` | `a174f15ba0269ae5e0e7ca50de51bbb7d80bf0e15af685a3fb69c06af64b15a3` |
  | `analyze_lifecycle.out` | `5af05709d3fefabc5096ea889699e816d2ecf899ae157369af5c70927a939904` |
  | `step_gaps.py` | `7c6e338c2603bdbf3e9929e60b89afb86d10e735fb9e50f0532d247c4b6f1d04` |
  | `step_gaps.txt` | `c66165e88a750fdebd85bf10606aa6b5b64f9ab955ec7eb921e1d984bf3ba2eb` |

## 8. Inputs to H4b-(b)

Frontier today (P2-32d): the D-side handoff is the KV transfer, 2.57 ms, plus the first D step, 43.24 ms, so 45.81 ms in total. `KVCacheTransferEndEvent` → `handle_reserved_decode_arrival` (`frontier/scheduler/utils/kv_arrival.py`) adds no engine-iteration delay. The native handoff is 127.92 ms on P2-32r and 138.98 ms on this run (138.47 ms on decoding lanes).

The parts locate the missing time:
- f (42 ms, 2 steps) matches Frontier's first D step.
- d holds a constant of about 2 steps (39.6 ms) that Frontier does not model. It also holds a step-quantized READ completion of 1.81 extra steps on average (about 37 ms), where Frontier models a 2.57 ms wire time.
- The remainder a + b + g (18.2 ms) is outside the engine and missing in Frontier; b, from the D API server to the engine `add`, is the largest part at 13.6 ms.

Together, about 94 ms on this run. The P2-32d handoff error against P2-32r is −82.1 ms; P2-32r has no observer, and its handoff is 11.1 ms shorter than this run's.

## 9. Limitations

- One run on H200. The H800 H4b-(a) run (`exp-1025-121318-827142`) is the H800 evidence; this run never qualifies an H800 cell.
- R4 FAIL on the shutdown order. The serving window passes every check, but the run is not R4-accepted. Its parts are used as DIAGNOSTIC evidence only.
- The client metrics differ from P2-32r by −0.3% to +9.1% (TTFT +9.1%), on different hosts. The observer effect is not separated from the host effect.
- The cross-host parts a, b and g depend on unmeasured clock synchronization; only their sum is clock-free.
- The NIXL `transfer_duration` is the worker's handle duration, and its value is quantized at step boundaries. The wire time of a late handle is not observed.
