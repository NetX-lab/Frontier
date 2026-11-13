# Test report: H200 64-GPU native workload (D31 step F) — 2026-10-26

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | First version: job terminal state, routing check, KV capacity, native metrics and the NIXL READ cost. |

## Scope

D31 step F acceptance (plan.md, "same checks as C"):
- the job succeeds and all requests complete in each repetition;
- the log shows `position_cyclic` active;
- the KV capacity of each P and D lane is recorded;
- metrics come from `tests/integration/step5_pd/workload_metrics.py`.

## Run

- Job `exp-1026-201417-791460`, submitted 2026-10-26 12:14:17 UTC through StepMind
  `RJobBackend` (Pi run `d31-h200-64-submit-1026c`, turn 2). Creator `i-fengyicheng`; 8 replicas
  × 8 H200; `step_main`; non-preemptible; routing image `vllm-openai@sha256:f037d958…`.
- Source: commit `e275c83a`, the same as step C. Stage:
  `/data/ycfeng/tmp/step5_d31_workload_20261025/h200_64_workload01/stage/`.
- Cloud output: `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d31_h200_64_workload01/`.
  The launcher copied it back to `stage/evidence/`.
- Topology: P32 and D32, each TP8/DP4/EP32 (11 local of 352 experts). Workload and engine
  settings are the same as step C.
- Submission condition: D31.2 limits this session to 64 non-preemptible H200 GPUs. At submission
  this session held none. Another session's 4 stepcast jobs held 32 non-preemptible H200 GPUs, so
  the account held 96 while this job ran.

Metrics command (local host, Python 3.10.6):

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
PYTHONPATH=$PWD:$PWD/tests/comparison/stage_admission_pp \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/workload_metrics.py \
  --stage /data/ycfeng/tmp/step5_d31_workload_20261025/h200_64_workload01/stage \
  --output /data/ycfeng/tmp/d31_native/h200_64_workload01_metrics.json
```

Exit 0. The output is kept as `evidence/d31_h200_64_native_metrics.json`.

## Results

| Check | Expectation | Actual | Result |
| --- | --- | --- | --- |
| Terminal state | Job succeeded | `launcher.log`: `Job succeeded` and `FINAL_STATUS succeeded` at 12:36:41 UTC. All 8 node `result.json` files have `stage: finished`, `workload_returncode: 0` and empty `observation_errors`. | PASS |
| Requests | 16 warmups + 128 formal requests per repetition, no failures | 432 requests: 48 warmups, 384 formal. The `workload_metrics.py` strict checks pass (6 verified repetition boundaries). | PASS |
| Routing | `position_cyclic` active on every node | Every node's `server.log` has `Simulating MoE routing using a position_cyclic strategy`. | PASS |
| KV capacity per lane | Recorded | Available KV memory 97.94–97.98 GiB per node; KV cache 1,228,667–1,229,158 tokens. | recorded |
| Errors | No error during the workload | One `EngineDeadError` traceback, on node 0028 at 12:36:32, during the SIGINT shutdown after the workload (`server_stop.returncode: 0`). This is the same shutdown message as in step C. | PASS |

Peak client in-flight requests: 95. KV capacity does not limit the workload.

Native metrics (milliseconds; throughput in tokens per second):

| Repetition | TTFT mean / p50 / p90 / p95 | TPOT mean / p50 / p90 / p95 | E2E mean / p90 | Total / output throughput |
| --- | --- | --- | --- | --- |
| 1 | 1462.3 / 1467.1 / 1676.1 / 1711.0 | 75.0 / 75.0 / 89.4 / 90.4 | 78159.6 / 92865.1 | 5010.2 / 715.7 |
| 2 | 1375.7 / 1385.5 / 1631.9 / 1689.3 | 68.8 / 67.0 / 81.1 / 81.4 | 71798.3 / 84359.5 | 5002.4 / 714.6 |
| 3 | 1305.8 / 1288.1 / 1619.0 / 1653.1 | 65.1 / 61.7 / 75.0 / 75.3 | 67907.2 / 78089.9 | 5003.9 / 714.8 |
| Pooled formal requests | 1381.3 mean, 1646.3 p90 | 69.6 mean, 84.3 p90 | 72621.7 mean, 87896.2 p90 | 5005.5 / 715.1 (mean of repetitions) |

TTFT split, as the mean over formal requests:
- client to proxy: 0.9 ms;
- through prefill: 808.9 ms;
- after the prefill response, to the first token: 571.5 ms.

Comparison with the 32-GPU run (step C), which had 1083.7 ms TTFT and 61.7 ms TPOT:
- The 64-GPU run is slower: TTFT is 27% higher and TPOT is 13% higher.
- The prefill part grows by 217 ms. The part after prefill grows by 81 ms.
- At QPS 1, doubling the DP lanes halves the batch per lane but doubles the EP all-to-all group
  size. Calibration at 64 GPUs therefore depends on EP32 communication and DP4 lockstep, not only
  on per-lane compute.

## NIXL READ cost at 64 GPUs

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/nixl_read_profile.py --case h200_64 \
  --server-logs /data/ycfeng/tmp/step5_d31_workload_20261025/h200_64_workload01/stage/evidence/nodes \
  --prompt-tokens 6144 --profile-identity "<H200 TP8 P/D native h200_64 workload01 ...>" \
  --output /data/ycfeng/tmp/d31_native/h200_64_nixl_read_cost.json
```

The output is kept as `evidence/d31_h200_64_nixl_read_cost.json`.

| Item | Value |
| --- | --- |
| Logging intervals | 182 |
| Per-rank READs | 3,456 (3 × 144 × 8, all logged) |
| Payload per rank | 139,887,104 B, equal to Frontier's reserved payload for the 6,128-token prefix |
| Transfer-weighted mean | 56.55 ms (32 GPUs: 51.37 ms) |

Step G uses this file instead of the 32-GPU value.

## Observations and limits

- TPOT falls from 75.0 to 65.1 ms over the three repetitions; step C showed the same downward
  trend.
- `runtime_qualification` is `UNVERIFIED`. D31.4 keeps native qualification out of scope.
