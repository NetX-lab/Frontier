# Test report: Step 9 ground-truth comparison (G3, G4, G5)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-25 | Admission times in the observations marked as upper bounds; intervals in WG05/WG12. |
| 2026-09-23 | Created. Covers G3 (native PP1 smoke), G4 (native PP2 ground truth) and G5 (Frontier runs, pre-change rejection, T1/T2 comparison and workflow-gap analysis). |

The case directory is `calibration/dp_pp_case_001/`, written `$C` below. The
design and verdicts are in plan §18.20 and §18.21.

## Environment

| Item | Value |
| --- | --- |
| Host | `kun-workspace-vgen2`, submission and NFS source |
| Frontier | `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`, branch `fix/issue26-correctness-pr`. G3 comparison at `e425ec9` plus the multi-burst harness, committed in `de41cd9`. G5 at `47d9190`. |
| CPU interpreter | `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10); `PYTHONPATH=<worktree> WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp` |
| Ground truth | `vllm/vllm-openai:v0.10.2` image with the overlay from `.real-engine/vLLM-BS` `63ac6c6b9` plus `inputs/groundtruth_overlay.patch`; Python 3.12, torch 2.8.0+cu128; tiny Qwen3-30B-A3B, dummy weights |
| GPU | StepMind `RJobBackend`, `i-fengyicheng`, codesign, H800, `code_mount_point=/data/ycfeng/Frontier`, cloud volume `/mnt/codesign-exp/ycfeng` |
| Launcher | `$C/launcher/{run_dp_pp_job.sh,submit_dp_pp_run.py,poll_replica_logs.py}`, the G4 versions copied from the session scratchpad. They contain no credential values. |

## 1. G3: native PP1 smoke (`dpp-g3-20260923a`)

Command: `RUN_TAG=dpp-g3-20260923a ENGINE_INPUT=inputs/engine_g3.json TRACE_INPUT=inputs/trace_g3 NUM_GPUS=2 NUM_CPUS=16 MEM_GB=96 bash run_dp_pp_job.sh`

| Check | Expected | Actual | Outcome |
| --- | --- | --- | --- |
| Run-check | `PASS` before launch | `PASS` | PASS |
| Platform | job succeeded; creator, source and devices verified | `exp-0923-221233-009652`, 14:12:33Z to 14:16:29Z, succeeded; 2 H800 | PASS |
| Worker | overlay import; 38/38 HTTP 200 | `vllm` and `frontier_trace` import from the overlay; 38/38; 140985 blocks per engine | PASS |
| Extraction | one placement per request | 565 iterations, 69/69 reports paired, 50 publications, 38 placements, 0 out-of-order | PASS |
| T1 replay | every route matches in engine and counts | 38/38 (30/30 formal) | PASS |

## 2. G4: native PP2 ground truth (`dpp-g4-20260923a`)

Command: `RUN_TAG=dpp-g4-20260923a ENGINE_INPUT=engine_g4.json TRACE_INPUT=trace_g4 NUM_GPUS=4 NUM_CPUS=16 MEM_GB=128 bash run_dp_pp_job.sh`

The server ran `vllm serve` with DP2, PP2, TP1 and EP, `max_num_batched_tokens`
20448, `max_num_seqs` 16, block size 16, eager mode, chunked prefill on and
prefix caching off (`run/replay_summary.json`).

| Check | Expected | Actual | Outcome |
| --- | --- | --- | --- |
| Run-check | `PASS` before launch | `PASS`; manifest committed in `47d9190` before submission | PASS |
| Platform | job succeeded; creator `i-fengyicheng`; current-host NFS source; 4 H800 | `exp-0923-230103-591735`, created 15:01:03Z, succeeded 15:07:04Z; creator and source verified; 4 H800 | PASS |
| Launcher | exit 0 and a wrapper-written receipt | exit 0, 15:00:59Z to 15:14:32Z, `launcher_receipt.json` | PASS |
| Worker | `WORKER_STATUS=0`; 51/51 HTTP 200 | 0; 51/51 | PASS |
| KV budget | recorded per engine | `[4566896, 4556400]` tokens per PP worker; minimum 284775 blocks | PASS |
| Extraction | one placement per request; receipts paired | 1620 iterations, 92/92 reports paired, 83 publications, 81 applications, 51 placements, 0 out-of-order | PASS |
| First-chunk duration | recorded (§18.5 G4 row) | 108.73, 116.13, 110.76, 108.39 ms; mean 111.00 ms | PASS |

## 3. G5: Frontier runs and comparison

### 3.1 Pre-change rejection

Command: `cd /data/ycfeng/tmp/issue26-correctness-pr/pre_change_d1a2a06 && PYTHONPATH=$PWD python pre_change_rejection.py <worktree> <case> $PWD`
(tree from `git archive d1a2a06 frontier data/config`).

Expected: the constructor rejects PP2. Actual: `ValueError vllm_load_balancing
supports one co-location Replica with vllm_v1 and PP1, got
cluster_type=MONOLITHIC, num_replicas=1, num_pipeline_stages=2,
replica_scheduler=vllm_v1`. **PASS.**

### 3.2 Simulator runs

Command: `python tests/comparison/dp_placement_pp/run_frontier_case.py --engine-config $C/inputs/engine_g4.json --frontier-config $C/inputs/frontier_g4.json --trace-dir $C/inputs/trace_g4 --output-dir /data/ycfeng/tmp/issue26-correctness-pr/calibration/dp_pp_case_001/runs/frontier_g4`

Inputs set from G4: `num_blocks` 284775 and `dummy_execution_time_ms` 0.8943
(2.0 × 111.00 / 248.25).

| Check | Expected | Actual | Outcome |
| --- | --- | --- | --- |
| Completion | 51/51 with tokens conserved in all three runs | 51/51 in all three runs | PASS |
| Dummy calibration | first completion near the native 111.00 ms | 111.14 ms (a–c), 116.89 ms (d) | PASS |
| Repeatability | same output as the check run | `summary.json` byte-identical (`cmp`) | PASS |

### 3.3 Comparison and workflow gap

Command: `python tests/comparison/dp_placement_pp/compare_placement.py --vllm-chain $C/runs/groundtruth_clean/dpp-g4-20260923a/extraction/chain.json --request-ids $C/inputs/trace_g4/request_ids.json --frontier-dir <frontier_g4 output> --output-dir $C/analysis/g5_comparison`

| Criterion | Expected | Actual | Outcome |
| --- | --- | --- | --- |
| C3 T1 replay | every native route matches in engine and counts | 51/51 (48/48 formal), 0 mismatches | PASS |
| C3 labeling | each natural-history difference carries a first cause | WG01–WG11: 4 MATCH, 7 MISMATCH, each with its cause | PASS |
| C4 qualification | fixed matches native and the control fails, on a qualified burst; otherwise `SCENARIO_NOT_REACHED` with checks named | all four bursts `SCENARIO_NOT_REACHED` (see below) | SCENARIO_NOT_REACHED |

Per-burst probe placement (engine or lane):

| Burst | Native | Fixed | Control | Round-robin | Failed checks |
| --- | --- | --- | --- | --- | --- |
| a | 0 | 0 | 1 | 0 | trace order; output before probe |
| b | 0 | 0 | 1 | 0 | trace order; output before probe |
| c | 0 | 0 | 1 | 0 | trace order; output before probe |
| d | 0 | 0 | 1 | 0 | one snapshot; schedule-time provenance; output before probe |

`workflow_gap_status.json` is `COMPLETE`/`PASS` with `correction_state=pending`
for S43 and S42.

## 4. Observations and inferences

Observed:

- In bursts a–c, the 40896-token body routed after three later 32-token
  requests, about 7 ms after dispatch.
- The first output was applied 19.7–33.8 ms after the first route of each
  burst, before every probe.
- Engine 1 has no step-0 record in burst d.
- Later burst requests were admitted 21.3–47.4 ms (a–c) and 129.5–311.8 ms (d)
  after the first route. **Corrected 2026-09-25:** these are iteration-record times, which bound an admission only from above; the admission intervals and the narrowed S43 evidence are in `calibration/dp_pp_case_001/analysis/workflow_gap_summary.md` (WG05, and WG12 for burst d engine 1).

Inferred from source:

- The missing burst-d step 0 is the wave-start dummy forward
  (`core.py:1170-1216`), which writes no record.
- The delayed admissions follow from the engine blocking on its oldest batch
  after an empty schedule (`core.py:385-424`).

## 5. Limits

- Frontier timing is dummy and token-independent (D-e). No E2E numeric gate is
  run, and `e2e-metrics-gap` is not applicable (plan §18.5 G5).
- The probe lane agreement in T2 is not C4 evidence, because no burst
  qualified.
- One of the three authorized GPU jobs was not used (plan §18.21).
- The G3 launcher receipt was written by hand from `submit.log`; the G4 one was
  written by the wrapper. `exec_capture` is UNKNOWN for both.
