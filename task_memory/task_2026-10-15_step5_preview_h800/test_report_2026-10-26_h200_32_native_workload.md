# Test report: H200 32-GPU native workload (D31 step C) — 2026-10-26

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | First version: job terminal state, routing check, KV capacity and native metrics. |

## Scope

D31 step C acceptance (plan.md): the job succeeds and all requests complete in each repetition;
the log shows `position_cyclic` active; the KV capacity of each P and D lane is recorded; metrics
come from `tests/integration/step5_pd/workload_metrics.py`.

## Run

- Job `exp-1026-120141-290449`, submitted 2026-10-26 04:01:41 UTC through StepMind
  `RJobBackend` (Pi label `d31-h200-32-submit-1026a`), creator `i-fengyicheng`, 4 replicas × 8
  H200, `step_main`, non-preemptible, routing image `vllm-openai@sha256:f037d958…`.
- Source: commit `e275c83a`. Stage:
  `/data/ycfeng/tmp/step5_d31_workload_20261025/h200_32_workload02/stage/`.
- Cloud output: `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d31_h200_32_workload02/`,
  copied back by the launcher to `stage/evidence/`.
- Topology: P on nodes 0057 (DP lane 0, API server, replay client) and 0229 (lane 1); D on 0938
  (lane 0, API server) and 1161 (lane 1). Each role TP8/DP2/EP16, 22 local of 352 experts.
- Workload: 6144 input / 1024 output tokens, 16 warmups plus 128 formal requests at QPS 1, three
  repetitions; max_model_len 7168, max_num_batched_tokens 8192, max_num_seqs 128,
  FULL_AND_PIECEWISE CUDA graphs.

Metrics command (local host, Python 3.10.6):

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
PYTHONPATH=$PWD:$PWD/tests/comparison/stage_admission_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/workload_metrics.py \
  --stage /data/ycfeng/tmp/step5_d31_workload_20261025/h200_32_workload02/stage \
  --output /data/ycfeng/tmp/d31_native/h200_32_workload02_metrics.json
```

The output is retained as `evidence/d31_h200_32_native_metrics.json`.

## Results

| Check | Expectation | Actual | Result |
| --- | --- | --- | --- |
| Terminal state | Job succeeded | `launcher.log`: `FINAL_STATUS succeeded` at 04:23:17 UTC; every node `result.json` has `stage: finished`, `workload_returncode: 0`, empty `observation_errors` | PASS |
| Requests | 16 warmups + 128 formal per repetition, no failures | 432 requests, 48 warmups, 384 formal; `workload_metrics.py` strict checks pass (prompt hashes, token deltas, P and D status 200, per-repetition metrics recomputed equal) | PASS |
| Routing | `position_cyclic` active on every node | Every `server.log` has `routing_simulator_router.py:335 Simulating MoE routing using a position_cyclic strategy`; `server_env_set` has `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic` | PASS |
| KV capacity per lane | Recorded for each P and D lane | All four lanes: `Available KV cache memory: 81.98 GiB`, `GPU KV cache size: 1,028,438 tokens`, 143.48× concurrency at 7168 tokens, four KV cache groups of block size 16. Model weights 40.74 GiB per GPU; graph capture 2.30 GiB | recorded |
| Errors | No error during the workload | One `EngineDeadError` traceback on the D API server at 04:23:03, after the workload finished, during the SIGINT shutdown (`server_stop.returncode: 0`) | PASS |

This vLLM build does not log `num_gpu_blocks`; the token capacity above is the logged value.
The peak number of client requests in flight is 85, so at most 85 × 7168 = 609,280 tokens are
needed across the whole system, below one lane's 1,028,438. KV capacity does not limit this
workload, and no preemption is expected on either side.

Native metrics (milliseconds, and tokens per second):

| Repetition | TTFT mean / p50 / p90 / p95 | TPOT mean / p50 / p90 / p95 | E2E mean / p90 | Total / output throughput |
| --- | --- | --- | --- | --- |
| 1 | 1171.2 / 1214.2 / 1355.9 / 1445.2 | 67.7 / 69.2 / 80.6 / 81.3 | 70422.8 / 83730.1 | 5211.5 / 744.5 |
| 2 | 1071.4 / 969.9 / 1285.3 / 1323.5 | 60.4 / 59.4 / 69.1 / 69.3 | 62844.7 / 71864.7 | 5204.7 / 743.5 |
| 3 | 1008.4 / 942.9 / 1280.5 / 1311.0 | 57.2 / 56.5 / 63.3 / 63.6 | 59483.5 / 65849.0 | 5227.6 / 746.8 |
| Mean of repetitions | 1083.7 / — / — / — | 61.7 / — / — / — | 64250.3 / — | 5214.6 / 744.9 |

TTFT split (mean over formal requests): client to proxy 0.9 ms, through prefill 591.9 ms, after
prefill response to first token 490.9 ms.

## Observations and limits

- TPOT falls from 67.7 ms to 57.2 ms across the three repetitions. The spread between
  repetitions (about 17% of the mean) is a reference noise level for the 30% gate.
- Throughput is set mainly by the arrival window: 128 arrivals at QPS 1 plus the last request's
  E2E time give a window of about 176 s in every repetition, so total throughput varies by less
  than 0.5% while TPOT varies by 17%. Throughput error therefore says little about decode speed.
- `runtime_qualification` in the canonical summary is `UNVERIFIED`; D31.4 keeps native
  qualification out of scope.
- The platform's node-to-node ping check again reported failures between all node pairs, while
  NIXL transfers and all requests succeeded. It is a platform check outside this workload.
