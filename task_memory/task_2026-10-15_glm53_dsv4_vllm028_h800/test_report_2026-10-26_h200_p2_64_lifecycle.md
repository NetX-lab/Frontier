# Test report 2026-10-26: H4b-(a) D-side handoff lifecycle measurement on H200-64 (P2-64 lifecycle)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: run, copy, R4 PASS, observer acceptance, client metrics against P2-64r, parts a-g with the d split, the first-contact READs on all three lifecycle runs, comparison with P2-32l and H800, persistence and the H4b-(b) inputs. |

**Authority.** D-H200-MAINLINE-CONTINUE (requirements item 21): DSV4 on H200-32 and H200-64 stays the mainline, with non-preemptible `step_main` RJobs of this session within 64 H200. D-NO-INTERRUPT (item 19): the lead chose one DIAGNOSTIC lifecycle run on H200-64 at DP4, the H200-64 counterpart of the P2-32 lifecycle run. Label DIAGNOSTIC: this H200 evidence feeds the H4b-(b) design and never qualifies an H800 cell. Method: [test_report_2026-10-25_h200_h4b_lifecycle.md](test_report_2026-10-25_h200_h4b_lifecycle.md), with P2-64r in the place of P2-32r.

## 1. Run and copy

| Item | Value | Source |
| --- | --- | --- |
| Package | `h200_mainline/p2_64_lifecycle/package_01` (commit `4e5d7f99`), `manifest.json` `ff3a3d3e…a0f`, run tag `dsv4-h200-p2-64-lifecycle-20261026-01` | `package_01_build.json` |
| Submission | Pi label `h200-p64-lifecycle-submit-1026a` turn 1 (prompt `/data/ycfeng/tmp/ds4_runs/prompts/h200-p64-lifecycle-submit-1026a.md`): quota before submission 59 free H200 and 7 nodes with 8 usable GPUs in `step_main`, so the job queued about 4 minutes | Pi turn-1 report, `/data/ycfeng/tmp/glm53_dsv4/h200/monitor/` |
| Job | `exp-1026-125836-702505`, launcher start 04:58:33 UTC, containers 05:01:19-05:03:36 to 05:26:34-05:26:51 UTC, Succeeded at 05:26:53 UTC; 26.3 GPU-hours | `package_01/case/runs/groundtruth_clean/<tag>/platform_terminal_receipt.json` |
| Platform | creator `i-fengyicheng`, `step_main`, preemptible `no`, 8 replicas Succeeded, every container exit 0, restart 0, the locked image `vllm-openai@sha256:2286e853…` | same receipt |
| Node roles | prefill dp0-dp3 on `gpu-h200-0040/0645/0705/0710` (0040 head), decode dp0-dp3 on `gpu-h200-0714/0716/0828/1087` (0714 head); 8 H200 on each host | `nodes/*/result.json` (`nvidia_smi`), `nodes/*/serving/rank*.json` |
| Launcher | pid 3751642, `launcher_exit.json` exit 0 at 05:27:02 UTC | `package_01/launcher_exit.json` |
| Copy | Pi turn 2 (prompt `...-turn2.md`): terminal receipt, read-only `ssh shai-ycfeng` inventory plus `sudo tar` stream, local rehash `HASH_IDENTICAL_DOWNLOAD`, 130 files, 102,020,803 B, inventory `2366eee2…f2be` (the lead rehashed `cloud_inventory.json`: equal) | `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-lifecycle-20261026-01/download_receipt.json` |
| Cloud origin | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_lifecycle_20261026_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-lifecycle-20261026-01` | same receipt |
| Evaluation case | `/data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_64l_case_20261026/`: `package_01/case/manifest.yaml` and `case/inputs/`, plus the run copy and the two package receipts under `runs/groundtruth_clean/<tag>/` (`run_manifest.json` byte-identical to the package copy) | — |
| KV cache | `num_blocks` 112,914 on all 64 ranks, maximum concurrency 38.35, both equal to P2-64r | `nodes/*/serving/rank*.json`, `nodes/*/server.log` |
| Replay | 3 × (16 warmup + 128 formal), 0 failures; error-like log lines (startup chat-template warnings, shutdown `EngineDeadError`, NCCL teardown) also appear in P2-64r; 0 CUDA, NCCL or NIXL failures | Pi facts `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/<tag>/facts_report.md` |

## 2. Acceptance results

| # | Item | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Platform Succeeded and R4 PASS on the copy | all R4 checks PASS | platform Succeeded; R4 status PASS, all 8 checks PASS (section 3) | PASS |
| 2 | Observer records | 8 engine cores, a `start` and no `error` record each, full timeline for every formal D request | 8 cores (prefill dp0-dp3, decode dp0-dp3), 8 `stop`, 0 `error`; 384 of 384 formal requests complete; every D request id matches; every READ in WAITING_FOR_REMOTE_KVS; every promotion step k=1 | PASS |
| 3 | Client metrics against P2-64r r1-r3 | reported, not gated | section 4 | reported |
| 4 | Parts a-g and iterations per lane state | reported | section 5 | reported |
| 5 | Results in the task directory with cloud paths and hashes | copies verified | section 7 | PASS |

## 3. R4 result

Command (`/data/ycfeng/envs/asyncopd-py312/bin/python`, Python 3.12), from the repository root:

```bash
(ulimit -v 31457280; env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 TMPDIR=/data/ycfeng/tmp \
  timeout 1800 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py \
  /data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_64l_case_20261026 dsv4-h200-p2-64-lifecycle-20261026-01 clean \
  > /data/ycfeng/tmp/glm53_dsv4/h200/r4/p2_64l.out 2> /data/ycfeng/tmp/glm53_dsv4/h200/r4/p2_64l.err)
```

Status file: `h200_mainline/diagnostic/inputs/p2_64l/run_check_status.groundtruth_clean.json` (checked 06:11:33 UTC), status PASS:

| Check | Verdict |
| --- | --- |
| `checkout`, `inputs_unchanged` | PASS (vLLM-BS `2cf0a691…`, clean) |
| `harness_pinned_and_client_rows` | PASS: 128 formal client rows in each of the 3 repetitions |
| `requests_and_idle_lanes` | PASS: 7 boundaries, both roles, all 4 DP lanes |
| `native_workers` | PASS: 8 distinct native nodes, every DP lane, source, model and GPU verified |
| `native_lifecycle` | PASS: every node drained, heads stopped after all drains, headless nodes after the heads, return codes 0 |
| `rank_runtime_graph` | PASS: 64 physical ranks, 6 phases each |
| `terminal_status` | PASS: Succeeded, all containers exit 0, no restarts |

The shutdown-order failure of the P2-32 lifecycle run (decode head late by 120 s) did not occur here.

## 4. Client metrics against P2-64r (acceptance 3)

Tool: the canonical client metrics inside `analyze_lifecycle.py`. Reference: `h200_mainline/diagnostic/inputs/p2_64r` (R4 PASS).

| Metric | Run r1 / r2 / r3 | P2-64r r1 / r2 / r3 | Run mean against reference mean |
| --- | --- | --- | --- |
| TTFT (ms) | 796.15 / 740.94 / 736.07 | 779.89 / 725.49 / 718.27 | +2.23% |
| TPOT (ms) | 25.032 / 24.867 / 24.990 | 25.430 / 26.255 / 25.926 | −3.51% |
| E2E (ms) | 26403.6 / 26179.4 / 26300.4 | 26794.6 / 27584.9 / 27240.8 | −3.35% |
| Total THPT (tok/s) | 5982.5 / 5989.8 / 5990.9 | 5948.0 / 5976.6 / 5965.1 | +0.41% |
| Proxy handoff (ms) | 223.38 / 158.73 / 161.48 | 220.75 / 172.35 / 170.19 | −3.50% |

Observed:
- TTFT is above the P2-64r range in r1 only; r2 and r3 are inside it. TPOT, E2E, THPT and the handoff are outside the range in all three repetitions, but on the faster side: TPOT and the handoff are lower, THPT is higher.
- On H200-32 the observer run was slower than its reference on every metric (TTFT +9.1%). On H200-64 it is faster on TPOT, E2E, THPT and the handoff. The observer therefore shows no effect in a consistent direction. Both runs used other hosts than their references, and there is one run of each, so the observer effect is still not separated from the host effect.
- Decode-step intervals while 4 or more requests are cached (`results/p2_64l/step_gaps.txt`): p50 24.45 ms on all 4 decode lanes, p99 30.85-31.27 ms, max 101-102 ms; 194-218 of 16,219-17,263 gaps exceed 30 ms. There is no periodic stall. The wrapped `schedule()` call takes 0.17-0.24 ms on average and `update_from_output()` 0.07-0.10 ms.

## 5. Parts a-g (acceptance 4)

Commands (`/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, and `/usr/bin/python3` for the scratch scripts), from the repository root:

```bash
C=/data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_64l_case_20261026; TAG=dsv4-h200-p2-64-lifecycle-20261026-01
O=/data/ycfeng/tmp/glm53_dsv4/h200/lifecycle/p2_64l_analysis
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 timeout 900 /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement/analyze_lifecycle.py \
  $C/runs/groundtruth_clean/$TAG task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/inputs/p2_64r $O > $O.out 2> $O.err)
(ulimit -v 31457280; timeout 600 /usr/bin/python3 \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/results/p2_32l/step_gaps.py \
  $C/runs/groundtruth_clean/$TAG > /data/ycfeng/tmp/glm53_dsv4/h200/lifecycle/step_gaps_p2_64l.txt)
R=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/results/p2_64l
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 timeout 300 /usr/bin/python3 $R/first_contact.py \
  $C/runs/groundtruth_clean/$TAG $R/lifecycle_requests.jsonl > $R/first_contact.txt)
T=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 timeout 300 /usr/bin/python3 $R/first_contact_waits.py \
  $T/h200_mainline/diagnostic/inputs/p2_32l $T/h4b_lifecycle_measurement/inputs/h800l $T/h200_mainline/diagnostic/inputs/p2_64l \
  > $R/first_contact_waits.txt)
```

Parts as in the H200-32 report: a proxy P response → D API server; b → engine `add`; c → READ schedule; d → `finished_recving` update; e → promotion schedule; f → first-token update return; g → proxy first D byte. The remainder a + b + g is clock-free.

Means on decoding lanes, all repetitions (372 requests) and repetitions 2-3 (248 requests), with the other two lifecycle runs (decoding lanes, all repetitions):

| Part | P2-64l all, mean (p50) | P2-64l r2-r3, mean (p50) | P2-32l mean | H800 mean |
| --- | --- | --- | --- | --- |
| a | 0.26 (0.07) | 0.28 (0.07) | 0.96 | 4.00 |
| b | 18.20 (17.89) | 18.04 (17.67) | 13.61 | 13.31 |
| c | 0.067 (0.058) | 0.064 (0.058) | 0.056 | 0.058 |
| d | 102.81 (74.66) | 85.88 (74.21) | 77.94 | 65.61 |
| e | 0.19 (0.15) | 0.19 (0.15) | 0.19 | 0.22 |
| f | 51.78 (50.71) | 51.53 (50.68) | 42.05 | 41.80 |
| g | 3.71 (3.65) | 3.67 (3.65) | 3.67 | 0.90 |
| engine c..f | 154.85 (127.28) | 137.66 (127.07) | 120.24 | 107.70 |
| remainder a+b+g | 22.17 (22.03) | 22.00 (22.04) | 18.23 | 18.21 |
| proxy handoff | 177.02 (153.36) | 159.66 (153.24) | 138.47 | 125.91 |
| engine iterations c..e | 5.12 (4) | 4.46 (4) | 4.81 | 4.26 |
| d iterations | 3.12 (2) | 2.46 (2) | 2.81 | 2.26 |
| READ transfer max over the 8 handles | 37.22 (26.25) | 37.49 (26.15) | 38.32 | 27.25 |
| decode step p50 (`step_gaps.txt`) | 24.45 | 24.45 | 20.27 | 19.83 |

Idle lanes (12 requests, of which one is a first-contact READ, below): d 185.2 ms (p50 48.2), f 81.0 ms, 10.25 iterations c..e (p50 4.5). By decode DP rank (decoding lanes): b 17.72 / 15.74 / 18.00 / 21.42 ms and remainder 23.69 / 20.69 / 20.74 / 23.59 ms on dp0-dp3; g 5.67 / 4.65 / 2.42 / 2.04 ms crosses hosts and depends on clock synchronization.

**d split by the number of D iterations up to `finished_recving`** (`results/p2_64l/first_contact.txt`, decoding lanes):

| d iterations | Requests | d (ms) | READ transfer max (ms) | d − transfer max (ms) | Spread over the 8 handles (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 49.7 | 2.0 | 47.8 | 0.6 |
| 2 | 252 | 74.6 | 26.4 | 48.2 | 2.1 |
| 3 | 66 | 99.5 | 51.0 | 48.5 | 25.6 |
| 4 | 48 | 124.8 | 75.9 | 48.9 | 51.2 |
| 5 | 1 | 148.6 | 96.3 | 52.3 | 73.4 |
| 56-70 | 4 | 1457.8-1800.3 | 2.2-32.1 | 1432-1775 | 0.5-23.9 |

Observed:
- As on H200-32 and H800, d is a constant of 48.2-48.9 ms in the 2-4 iteration groups, about 2 decode steps of 24.45 ms, plus the READ transfer max. The transfer max grows by about one step (25 ms) per further iteration.
- The common case is 2 iterations (252 of 372 requests). There the 8 handles all report about 26 ms, slightly above one decode step, within 2.1 ms of each other. Only 1 request completes within the first step; on H200-32 that bucket held 45 requests.
- f is one engine iteration and about 2 decode steps (51.5 ms = 2.1 steps), as on the other two runs.
- The remainder a + b + g is 22.2 ms, about 4 ms above the 18.2 ms of both 32-GPU runs. b, from the D API server to the engine `add`, is 4.6-4.9 ms above theirs.
- The READ windows of different requests never overlap (`read_window_shared` false for all 384). Every request moves 8 handles and 368,824,320 B, as on the other two runs.

**First-contact READs in repetition 1.** Five formal requests of repetition 1 have d of 1.46-1.80 s and 56-70 D iterations, while their 8 handles report only 2.2-32.1 ms. The D engine keeps stepping at the normal 25 ms rate while these requests wait in WAITING_FOR_REMOTE_KVS. Each of the five is the first request of its prefill-to-decode DP lane pair (`first_contact.txt`):

| Pair | Request | d (ms) | d iterations |
| --- | --- | --- | --- |
| P3→D0 | formal r002 (idle lane) | 1665.5 | 68 |
| P1→D3 | formal r004 | 1457.8 | 56 |
| P3→D1 | formal r006 | 1724.5 | 67 |
| P2→D0 | formal r009 | 1800.3 | 70 |
| P1→D2 | formal r053 | 1719.1 | 68 |

The READ wait of every D request, warmup included, from the READ schedule to the `finished_recving` update, on all three lifecycle runs (`results/p2_64l/first_contact_waits.py`, output `first_contact_waits.txt`; the "behind" requests are later requests on the same lane pair whose READ was scheduled while the pair's first contact was pending):

| Run | Lane pairs | First contacts inside the warmup of repetition 1 | First-contact wait (s) | Requests behind a first contact, wait (s) | Other D requests: mean / p50 / max (ms) |
| --- | --- | --- | --- | --- | --- |
| P2-32l | 2 × 2 = 4 | 4 of 4 | 1.62-4.74 | 8, 0.35-3.91 | 78.5 / 73.3 / 406.8 (420 requests) |
| H800 | 2 × 2 = 4 | 4 of 4 | 1.04-1.35 | 5, 0.04-1.04 | 65.2 / 60.9 / 261.7 (423 requests) |
| P2-64l | 4 × 4 = 16 | 11 of 16 | 1.46-5.00 | 4, 0.63-3.33 | 83.8 / 73.9 / 148.6 (412 requests) |

Observed:
- The first READ of every lane pair waits 1.0-5.0 s on all three runs. On P2-64l, each request behind a first contact finishes in the same update as it.
- On the 32-GPU runs, the 16 warmup requests of repetition 1 reach all 4 lane pairs, so no formal request meets a first contact: the largest formal d is 143.2 ms on P2-32l and 119.5 ms on H800. On H200-64 the same 16 warmup requests reach only 11 of the 16 pairs, and the first request of each of the other 5 pairs is a formal request.
- In this run, the r1 handoff mean is 63.3 ms above the r2-r3 mean. The five formal first contacts, at about 1.6 s above the usual 75 ms each, account for about 62 ms of it when spread over the 128 requests.
- P2-64r, without the observer, has a handoff maximum of 1923.9 ms in r1 and of 343.7 / 329.5 ms in r2 / r3, and its r1 handoff mean is 49.5 ms above the r2-r3 mean. It has no lifecycle files, so its first contacts are not observed directly.

Inference, not verified: the first READ between a D engine and a P engine waits for the NIXL connection setup between them (the remote agent metadata exchange), and the handle durations start only after it. This is a one-time cost per lane pair. In the H200-64 formal window it comes from incomplete warmup coverage in the measurement harness, not from steady-state behavior. If P2-64r has the same cause, it raises the P2-64r native handoff mean of repeats (187.76 ms, the P2-64d comparison target) by about 16.5 ms over its r2-r3 value of 171.27 ms.

**Expectation written before the run** (H4b-(a) design section 8, 32-GPU values), compared on repetitions 2-3:

| Expectation | Measured | Verdict |
| --- | --- | --- |
| c at most 1 iteration | c 0.064 ms, inside the READ step | met |
| c + d + e about 3 iterations | 3.46 iterations (4.46 c..e minus the f iteration) | slightly above |
| f about 30-40 ms | 51.53 ms (2.1 steps of 24.45 ms) | above; in steps, as on the other runs |
| c to f about 75-105 ms | 137.66 ms | above, with the longer step |
| remainder a + b + g about 10-25 ms | 22.00 ms | met |

## 6. Inputs to H4b-(b)

Frontier today (P2-64d, `h200_mainline/diagnostic/results/p2_64d/h4a5_attribution.json`): the D-side handoff is the KV transfer, 2.57 ms, plus the first D step, 38.98 ms, so 41.55 ms in total. The native handoff target is 187.76 ms (P2-64r mean of repeats), gap −146.22 ms.

In steady state (repetitions 2-3 of this run, handoff 159.66 ms):
- f (51.5 ms, about 2 steps) is 12.6 ms above Frontier's first D step.
- d holds about 2 steps (48.4 ms) that Frontier does not model, plus a step-quantized READ completion of 37.5 ms on average, where Frontier models 2.57 ms. Item 1 (D-H4B-ITEM1, `nixl_read_completion='engine_step'`) models both; the engine_step diagnostic run on P2-64 checks the result.
- The remainder a + b + g (22.0 ms) is outside the engine and missing in Frontier.
- The first-contact READs probably add about 16.5 ms to the P2-64 native target (inference above). No Frontier change models them, because they are a gap in the harness warmup. A later P2-64 measurement can send at least one warmup request on each of the 16 lane pairs to remove them. The comparisons can also use repetitions 2-3. That is a choice of comparison target, and it is recorded here as an option, not applied.

## 7. Persistence (acceptance 5, D-INPUT-PERSISTENCE)

- Inputs: `h4a_p0_diagnostic/copy_inputs.py h200_p2` added the set `p2_64l`: 28 files, 28,929,215 B, in `h200_mainline/diagnostic/inputs/p2_64l/`. They are the 7 run files, the two receipts, the 8 lifecycle files, the 8 `native_stopped.json`, the R4 status, the download receipt and the cloud inventory. The earlier sets in `manifest.json` are unchanged (checked against a copy taken before the run of `copy_inputs.py`). `manifest.json` SHA-256 `f42986dedb49c5cef208b1fdbbfdb27b77de811a05e5394a2c29d2500d237596`.
- 23 of the 23 run and node files match the cloud inventory digests. The other 5 (`launcher_receipt.json`, `platform_terminal_receipt.json`, the R4 status, the download receipt and the cloud inventory) were written locally and are not in the cloud inventory.
- Lifecycle file digests (first 16 hex digits): prefill 0040 dp0 `f463769506bd4fe2`, 0645 dp1 `abf760a618c3009a`, 0705 dp2 `529b1b9d6ae07ff3`, 0710 dp3 `972646cc64dabe42`; decode 0714 dp0 `3cc4b53d1913a895`, 0716 dp1 `68edaa1aac89c37c`, 0828 dp2 `6a119ce76b3de1cb`, 1087 dp3 `121fa8d7214dd7de`; full digests in the manifest. The 8 lifecycle files (27.9 MB) stay in the task directory but out of Git, as for P2-32l; their hashes and cloud origin are in the committed manifest.
- Results `h200_mainline/diagnostic/results/p2_64l/` (SHA-256):

  | File | SHA-256 |
  | --- | --- |
  | `lifecycle_summary.json` | `d1ac9400357bf78ab1cd939f9176e5d3a64ea0bc73cfad098301bfd6e765f9fe` |
  | `lifecycle_requests.jsonl` | `6503618637bf4ecd6fdd2bfde12f4cc5fb84ee9dac71d066997fa5f022eb0fbc` |
  | `analyze_lifecycle.out` | `b56545ac7c6995d0fd19fe3449e64c5b2cb9c70e3363e030f055be5ed08b7f2a` |
  | `step_gaps.txt` | `7ce4242226f755b3161079a426bbc6764a5f80e034a594aecf51c1255f3eacc7` |
  | `first_contact.py` | `6f308b6fa4ce7bc3bba2e2d06186ee4c1ef991269c01eeb6a61a0a090e370c4f` |
  | `first_contact.txt` | `5c49a95053d8d3225b18ea9d584b802fc6e97766da4b2a6a11ca1c7c5760a4e3` |
  | `first_contact_waits.py` | `57c695a793e945e9d19677891de05c3d7e339883fb6e31c218c2fde27549bfaa` |
  | `first_contact_waits.txt` | `7d65e1b79fc34b90544c9ce4ca6c1d185bc0af51090fdb68a2152b27a65286ba` |

## 8. Limitations

- One run on H200-64, on hosts other than P2-64r's. The observer effect is not separated from the host effect; here the observer run is the faster one.
- The first-contact explanation is an inference from timing; the NIXL connection setup is not observed directly.
- The cross-host parts a, b and g depend on unmeasured clock synchronization; only their sum is clock-free.
- The NIXL `transfer_duration` is the worker's handle duration and is quantized at step boundaries. The wire time of a late handle is not observed.
- H200 evidence never qualifies an H800 cell.
