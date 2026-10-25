# Test report: H200 P2-32d and P2-64d diagnostic predictions (2026-10-25)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: run copies, R4, H200 step-cost tables, the P2-64 substitution, both Frontier predictions, client gates, attributions and regressions on native06. |

**Label.** DIAGNOSTIC. These readings count toward neither the 30% promotion gate nor the 10% final gate (D-H4A-CHECKPOINT). H200 evidence does not qualify any H800 cell.

**Authority.** D-H200-TEMP-MAINLINE (H200 as the temporary mainline, reusing the H800 suites), D-H200-STEP-COST-SUBSTITUTION (YC chose the same-table substitution for the P2-64 gap) and YC's approval of the read-only SSH copy route, all in [requirements.md](requirements.md). Design: [h200_mainline/design.md](h200_mainline/design.md).

## 1. Cases

| Case | Topology (per role) | Native reference (client gate side) | Native profile (step-cost source) | Frontier run |
| --- | --- | --- | --- | --- |
| P2-32d | P16 + D16, TP8 / `attn_dp` 2 / EP16 / PP1 | P2-32r `dsv4-h200-p2-canonical-20261025-01` | P2-32p `dsv4-h200-p2-profile-20261025-01` | `/data/ycfeng/tmp/glm53_dsv4/h200/diag/p2_32d_run1` |
| P2-64d | P32 + D32, TP8 / `attn_dp` 4 / EP32 / PP1 | P2-64r `dsv4-h200-p2-64-canonical-20261025-01` | P2-64p `dsv4-h200-p2-64-profile-20261025-01` | `/data/ycfeng/tmp/glm53_dsv4/h200/diag/p2_64d_run2` |

Workload for all four native runs: the frozen 68f891b trace, 3 repeats of 16 warm-up plus 128 formal requests, 6144 input and 1024 output tokens, QPS 1. The client gate compares Frontier with each native repeat and with the mean of the three.

## 2. Environment and inputs

- Local host. Simulation, gate and attribution: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6). R4: `/data/ycfeng/envs/asyncopd-py312/bin/python` (Python 3.12.13). Copy: Pi on HOST, `/usr/bin/python3`. Every run used `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp` and a `ulimit -v` cap.
- Source: the Frontier runs ran at `890ab80d`. `frontier/` is unchanged from `890ab80d` to HEAD. The receipts list one untracked path, `data/profiling/compute/h200/deepseek-ai/` (the P2-32 table); it was committed in `2e9e8c61` with the same SHA-256 the receipt records.
- Run copies: `h200_mainline/diagnostic/inputs/{p2_32r,p2_32p,p2_64r,p2_64p}/` with `manifest.json` (SHA-256 `dce1c9aa…183c`; 12, 11, 12 and 13 files), committed in `b4eb1efa`. The profile traces stay in the local download directory and are not tracked.
- Retained Frontier inputs (`case/inputs/frontier.json` of each reference package): P2-32 `5ff79a51…8b43`, P2-64 `3aac0a3f…c4f4`.
- R4 results pinned by the gate: P2-32r `8cc662e6…04a1`, P2-64r `e1da1a98…3b00`.
- Step-cost tables, committed in `2e9e8c61`:
  - P2-32: `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json`, `cdf7d116…850d`.
  - P2-64: `h200_mainline/diagnostic/step_cost_attn_dp4/dsv4_step_cost.json`, `b452351a…ac9d`. The loader reads the table beside the resolved attention input and accepts one topology per directory, so the driver points `random_forrest_execution_time_predictor_config_atten_input_file` into this directory. No `attention.csv` is read on the step-cost path.
  - The H800 table `ea817ce5…f8c` is unchanged.
- Scripts (case names `native06`, `p2_32`, `p2_64`), committed in `22bf1732`: `h4a_p0_diagnostic/h4a4_driver.py`, `h4a4_gate.py`, `h4a5_attribution.py`.
- `replica_config_network_device` stays `h800_dgx` (design section 4). On the step-cost path it prices only the analytical KV transfer, 2.57 ms per request.

## 3. Commands

```bash
# Copy (Pi label h200-ssh-copy-1025c, prompt /data/ycfeng/tmp/ds4_runs/prompts/h200-ssh-copy-1025c.md,
# model custom/deepseek-v4.1-flash-sys, thinking high, exit 0). Per run, read-only on shai-ycfeng:
ssh -o BatchMode=yes -o ConnectTimeout=15 shai-ycfeng 'sudo -n timeout 900s python3 - <remote root>' < <hash script> > cloud_inventory.json
ssh -o BatchMode=yes -o ConnectTimeout=15 shai-ycfeng 'sudo -n tar -C <remote root> -cf - .' \
  | tar --no-same-owner --no-same-permissions -xf - -C <local>/artifacts
# then every local file is rehashed against the inventory.

# R4 on each local reference copy
(ulimit -v 31457280; env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=<repo> \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 TMPDIR=/data/ycfeng/tmp \
  timeout 1800 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py \
  /data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_32r_case_20261025 dsv4-h200-p2-canonical-20261025-01 clean)
# and the same for p2_64r with dsv4-h200-p2-64-canonical-20261025-01

# Step-cost table, built twice per profile (s1, s2)
(ulimit -v 31457280; PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 timeout 1200 $PY tests/comparison/calibration/prepare_dsv4_step_cost.py \
  --run <inputs/p2_32p | inputs/p2_64p> \
  --engine-config <h200_mainline/p2_32_profile/package_02 | h200_mainline/p2_64_profile/package_01>/case/inputs/engine.json \
  --device h200 --output /data/ycfeng/tmp/glm53_dsv4/h200/step_cost_build/<name>.s1.json)

# Frontier prediction
nohup bash -c "(ulimit -v 62914560; env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true \
  VIDUR_DISABLE_WANDB=1 timeout 21600 $PY task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/h4a4_driver.py \
  <p2_32|p2_64> /data/ycfeng/tmp/glm53_dsv4/h200/diag/<run>) > <run>.log 2>&1; echo EXIT \$? >> <run>.log" &

# Client gate and attribution
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 timeout 1200 $PY .../h4a_p0_diagnostic/h4a4_gate.py \
  <p2_32|p2_64> /data/ycfeng/tmp/glm53_dsv4/h200/diag/<run> /data/ycfeng/tmp/glm53_dsv4/h200/gate/<p2_32d|p2_64d>
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY .../h4a_p0_diagnostic/h4a5_attribution.py \
  <p2_32|p2_64> /data/ycfeng/tmp/glm53_dsv4/h200/diag/<run> /data/ycfeng/tmp/glm53_dsv4/h200/attribution/<p2_32|p2_64>.json
```

`$PY` is `/data/ycfeng/envs/frontier-py310/bin/python`. Logs: `/data/ycfeng/tmp/glm53_dsv4/h200/{r4,step_cost_build,diag,gate}/`. Persisted results: [h200_mainline/diagnostic/results/](h200_mainline/diagnostic/results/) (`p2_32d/`, `p2_64d/`, `p2_64d_run1_coverage_failure/`).

## 4. Checks

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Copy receipts (`/data/ycfeng/tmp/glm53_dsv4/h200/downloads/<tag>/download_receipt.json`) | `HASH_IDENTICAL_DOWNLOAD`, `mismatched` empty, inventory digest equal | P2-32r 66 files / 32,365,082 B; P2-32p 226 / 4,544,895,900; P2-64r 122 / 76,931,719; P2-64p 442 / 9,146,245,823; all as expected, 17:52-17:57 UTC | PASS |
| 2 | Fact sheet (`downloads/facts.json`) | 8 × H200 per host; 432 of 432 requests per run; KV binding read from the runs; no failure during the workload | as expected; decode `num_blocks` 109934 (P2-32) and 112914 (P2-64), logged maximum concurrency 37.34 and 38.35 (H800: 54740, 18.59). The logged errors are the same kinds H800 profile02 logs: the `Tokenizer not available when skip_tokenizer_init=True` warning before the first request (8 per 32-GPU run, 16 per 64-GPU run; profile02 8), `EngineDeadError` from the API servers after the worker's SIGTERM shutdown (e.g. P2-32r `server.log:954-973`), NCCL shutdown warnings, and in the profile runs `NATIVE_PROFILE_UNVERIFIED` from the optional CUDA stream observer, which cannot load `libcupti.so` (48 per host, as on profile02; the step-cost builder does not use it). No restarts. | PASS |
| 3 | R4 on P2-32r and P2-64r (`r4/*.out`) | `status` PASS, `failed` [] | both PASS, `failed` [], checked 17:57:54 UTC | PASS |
| 4 | Persisted inputs | copies equal to the downloads; `manifest.json` lists every file | equal by SHA-256 (Python `rglob`); `dce1c9aa…183c` | PASS |
| 5 | Table builds | two builds per profile byte-identical; H800 rebuild unchanged under the substitution rule | `h800.s1/s2` `ea817ce5…`, `p2_32p.s1/s2` `cdf7d116…`, `p2_64p.s1/s2` `b452351a…`; the H800 and P2-32 tables contain no substituted entry | PASS |
| 6 | Driver regression on native06 (`diag/native06_regression`, final script) | the three metric files and the transfer ledger byte-identical to H4c `replay_b1` | `request_metrics.csv` `977c4d5a…`, `system_metrics.json` `1845fe17…`, `metrics_ground_truth.jsonl` `4fee9488…`, `frontier_transfer_ledger.jsonl` `693d6812…`, all equal | PASS |
| 7 | Gate regression on native06 (`gate_regression_native06b/reading.json`) | equal to `h4a_p0_diagnostic/results/gate/reading.json` | equal | PASS |
| 8 | Attribution regression on native06 (`attribution/native06_regression.json`) | equal to `h4a_p0_diagnostic/results/h4a5_attribution.json` apart from key names | equal after the renames `native06` → `native`, `profile02` → `profile`; new fields `case` and `unassigned_k1_steps` (0) | PASS |
| 9 | P2-32d run (`results/p2_32d/receipt.json`) | FINISHED; 128 of 128 formal requests; 0 preemptions | as expected; 780.3 s wall time | PASS |
| 10 | P2-64d run 1 (`results/p2_64d_run1_coverage_failure/`) | — | `DeepSeekV4CoverageError`, exit 1: P2-64p's profiler window has the decode compositions k = 1 and k = 9 newly arrived requests only with 0 decode requests | FAIL, resolved by D-H200-STEP-COST-SUBSTITUTION |
| 11 | P2-64d run 2 (`results/p2_64d/receipt.json`) | FINISHED; 128 of 128; 0 preemptions | as expected; 1103.5 s wall time | PASS |
| 12 | KV admission (receipts, `kv_admission`) | Frontier blocks per request in the range that gives the logged native maximum concurrency | Frontier 2940 blocks per request; native 2944 on both cases | DIFFERS (non-binding, same as H800) |

## 5. Client gate results

The gate's relative error limit is 10% per metric (`relative_error_limit` 0.1). The verdict is FAIL on both cases because of TTFT. Frontier has one value per metric because its run is deterministic.

**P2-32d** (`results/p2_32d/gate/reading.json`):

| Metric | Native r1 | Native r2 | Native r3 | Native mean | Frontier | Error vs mean | Per-repeat error | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean (ms) | 361.97 | 364.75 | 365.37 | 364.03 | 327.23 | −36.80 ms, 10.11% | 9.60 / 10.29 / 10.44% | FAIL |
| TPOT mean (ms) | 19.907 | 19.889 | 19.962 | 19.919 | 18.642 | −1.277 ms, 6.41% | 6.36 / 6.27 / 6.61% | PASS |
| E2E mean (ms) | 20726.7 | 20710.8 | 20785.9 | 20741.1 | 19397.5 | −1343.6 ms, 6.48% | 6.41 / 6.34 / 6.68% | PASS |
| Throughput (tok/s) | 6234.1 | 6231.4 | 6228.7 | 6231.4 | 6279.3 | +47.9, 0.77% | 0.72 / 0.77 / 0.81% | PASS |

**P2-64d** (`results/p2_64d/gate/reading.json`):

| Metric | Native r1 | Native r2 | Native r3 | Native mean | Frontier | Error vs mean | Per-repeat error | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTFT mean (ms) | 779.89 | 725.49 | 718.27 | 741.22 | 601.00 | −140.21 ms, 18.92% | 22.94 / 17.16 / 16.33% | FAIL |
| TPOT mean (ms) | 25.430 | 26.256 | 25.926 | 25.871 | 24.768 | −1.103 ms, 4.26% | 2.60 / 5.67 / 4.47% | PASS |
| E2E mean (ms) | 26794.6 | 27584.9 | 27240.8 | 27206.8 | 25938.4 | −1268.4 ms, 4.66% | 3.20 / 5.97 / 4.78% | PASS |
| Throughput (tok/s) | 5948.0 | 5976.6 | 5965.1 | 5963.2 | 6000.9 | +37.7, 0.63% | 0.89 / 0.41 / 0.60% | PASS |

For comparison, H4a on H800 (native06): TTFT 17.35%, TPOT 2.50%, E2E 2.10%, throughput 0.54%.

## 6. Attribution

Source: `results/<case>/h4a5_attribution.json`. Native parts are means over the three repeats of the reference run. Native P service runs from proxy arrival to the prefill response; the handoff runs from the prefill response to the first decode byte at the proxy. Frontier's handoff is the KV transfer plus the time from KV arrival to the first decode token.

**TTFT split (ms).**

| Part | P2-32 native | P2-32 Frontier | Gap | P2-64 native | P2-64 Frontier | Gap | H800 gap (H4a) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Client to proxy plus decode byte to client | 0.61 | 0 | −0.61 | 0.82 | 0 | −0.82 | — |
| P side (P service / P compute) | 235.50 | 281.42 | +45.92 | 552.63 | 559.46 | +6.82 | — |
| Handoff (KV transfer + first decode step) | 127.92 | 2.57 + 43.24 | **−82.11** | 187.76 | 2.57 + 38.98 | **−146.22** | **−81.2** |
| TTFT | 364.03 | 327.23 | −36.80 | 741.22 | 601.00 | −140.21 | — |

Observations:
- The handoff is the largest TTFT error on all three cases. Frontier does not model the time between the prefill response and the first D-side step: the remote-KV lifecycle that H4b-(a) measures and H4b-(b) is to model.
- The P2-32 P-side gap (+45.9 ms) partly cancels the handoff gap. The table's PREFILL entry is the median P service of P2-32p's repeats 2 and 3 (281.4 ms), about 20% above P2-32r's 235.5 ms. The two runs share the serving configuration apart from the profiler, which is off in repeats 2 and 3, and ran on different hosts. Inference: this is run-to-run spread, not a Frontier pricing error. Without it, P2-32's TTFT error would be larger.
- P2-64r's P service spreads evenly from about 300 to 800 ms (P2-32r: 383 of 384 requests between 200 and 300 ms). The cause is not established. The PREFILL entry reproduces the mean (+6.8 ms) but Frontier gives every request the same P compute, so it does not reproduce the spread.

**Handoff samples in the profile traces** (repeat 1, the only profiled repeat). Each case has three requests whose first D-side step is a k = 1 step, with no unassigned steps. In each case one of the three waited about 18.5 s behind a profiler pause (P2-32 `r002`, P2-64 `r002`) and is left out below.

| Case, request | Handoff | Prefill response → k=1 step CPU start | CPU start → GPU start | k=1 GPU step | Next step start → first byte |
| --- | --- | --- | --- | --- | --- |
| P2-32 `r001` dp0 | 144.4 | 82.6 | 26.3 | 29.6 | 5.9 |
| P2-32 `r000` dp1 | 120.5 | 58.5 | 36.6 | 20.9 | 4.5 |
| P2-64 `r000` dp2 | 133.4 | 60.7 | 41.3 | 26.8 | 4.6 |
| P2-64 `r001` dp3 | 170.8 | 102.6 | 38.4 | 26.5 | 3.4 |

The span from the prefill response to the k = 1 step's CPU start (58-103 ms) has no counterpart in Frontier apart from its 2.57 ms KV transfer. The CPU-to-GPU delay (26-41 ms) is also longer than the wait for the running lane step that Frontier includes in its first-step time (Frontier's mean first-step time minus its k = 1 entries: about 9-18 ms on P2-32, 12 ms on P2-64). Together these two account for about the size of the handoff gap. These are four samples from profiled steps, so they show the shape of the gap, not its distribution.

**TPOT.**

| | P2-32 | P2-64 |
| --- | --- | --- |
| Reference run TPOT (mean of 3 repeats) | 19.919 | 25.871 |
| Profile run TPOT, repeats 2-3 (repeat 1 is profiled: 22.84 / 62.23) | 19.083 | 25.076 |
| Profile run faster than reference run | 4.2% | 3.1% |
| Frontier TPOT | 18.642 | 24.768 |
| Frontier against the profile run | −2.3% | −1.2% |
| Decode-only table entry and the decode batch sizes it was measured at | 18.370 ms at 1-2 requests | 24.723 ms at 1, 9 and 13 requests |
| Native decode requests at once (whole pod, per lane) | 17.8, about 8.9 per lane | 21.7, about 5.4 per lane |

Inference: most of the TPOT error comes from the reference and profile runs differing on different hosts (3-4%). The rest comes from pricing decode steps at the batch sizes the profiler window saw. P2-32's window saw only 1-2 decode requests while native lanes held about 9. P2-64's window included 9 and 13, and its remaining gap is smaller.

**Effect of the substitution (estimate).** On P2-32 the measured step with one newly arrived request and at least one decode request costs 34.25 ms, 8.96 ms above the same step with no decode requests (25.28 ms). P2-64's table prices that composition at its 0-decode value, 26.76 ms. If P2-64's real step carried the same +8.96 ms, Frontier's P2-64 TPOT would rise by at most about 0.20 ms (at most about 22.5 such steps per request, from Frontier's TPOT excess over the decode-only entry, which also includes the k = 4, 8 and 9 steps), moving the TPOT error from 4.26% to about 3.5%. TTFT would rise by at most about 9 ms, leaving the TTFT error above 17%. Neither verdict changes.

## 7. Limitations

- DIAGNOSTIC only; no formal gate counts these numbers.
- NIXL `native_transfer_verification` remains UNVERIFIED, as on H800.
- KV admission DIFFERS by 4 blocks per request (2940 against 2944). It does not bind here: no preemptions, and native concurrency stays below the 37-38 request limit.
- P2-64's table holds two substituted compositions (section 6 gives their estimated effect).
- The profile run's repeat 1 includes profiler overhead. Only repeats 2-3 price the prefill step, and only the 64-iteration profiler window at the start of repeat 1 prices the decode steps.
- The reference and profile runs ran on different host sets. Their run-to-run spread is about ±4% on TPOT and up to about 20% on P service, and it enters these errors directly.
- `replica_config_network_device` stays `h800_dgx`.
- The handoff breakdown rests on four samples.
- The R4 checker and harness were accepted on H800 at DP2; this is their first H200 and DP4 use. R4 passed on both references.
