## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded clean replay-02 validation and fresh snapshot-policy CPU execution. |

# Snapshot-policy replay of the 4096/1024 case

## Execution and criteria

Source root: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907. Frontier commit a4a0c496d370c84d83a5b098d3fdd8561af1c74e. CPU Python: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, conda dev-vidur-v03-hopper-e2e, Python3.13.13. GPU Python: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python3.10.16. H200 step_main node gpu-h200-1091; fixed image digest and complete mode/checkout tuples are in runs/h200-historical-replay-02/{clean,batch}/run_manifest.json.

Expected: clean400client/server records,100unique formal4096/1024 requests,300excluded warmups across three drained phases. Frontier must complete the same100formal requests using this run's observed engine arrivals. Retain collective_sim/htsim plus nvlink_analytic and the new vllm_load_balancing policy. All predictor and collective caches must be new. Compare the two clean request sets by exact IDs; report mean TTFT, TPOT, E2E, and formal-window throughput separately. Official vLLM server TTFT is the D006 primary reference; no residual CPU constant is fitted.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_official_ttft_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/clean/runtime/clean --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-clean
```

CPU launch: user service yc26-cpu-dp-snapshot-numerical-20260908-02-r1.service runs `/bin/bash /data/ycfeng/tmp/issue26-h200-network/launch-cpu-dp-snapshot-numerical-02.sh`. The script calls `tests/e2e/issue26_cpu_frontier_worker.py` with `config/frontier_dp_snapshot_candidate.json`; full expanded command, fresh cache paths, runtime check, and provenance are preserved in analysis/dp_snapshot_numerical_02_run_manifest.json. Raw logs and metrics reside at `/data/ycfeng/tmp/issue26-dp-snapshot-numerical-02/runtime` during execution.

## Observed clean evidence

PASS:400client/server rows, no missing, duplicate, or extra request identities;100formal requests satisfy4096/1024; all four phases drain in order. Runtime H200 compute checks and24uniform-router checksPASS. Mean official TTFT131.26863718032837ms; client135.40103293ms. First formal officialTTFT121.48427963256836ms.

A premature automatic validation failed `assert len(server) == 400` after the client completion marker became visible. It produced no trace and launched no simulation. On subsequent inspection, the final server artifact contained400unique matching rows; the unchanged full validator passed. The temporary followup readiness condition now also requires400complete server lines. Original failure and receipt remain preserved. Exact visibility-delay cause (producer timing versus NFS visibility) is not established.

After the formal phase, server shutdown emitted TCPStore Broken pipe warnings. Final metrics are complete and both engines report zero running/waiting before application shutdown completion. These post-measurement warnings do not establish a runtime timing defect.

## Remaining validation

CPU numerical execution and isolated batch diagnostics are in progress. No Frontier prediction, absolute error, relative error, or numerical PASS is claimed yet.

## Subsequent observed result

GPU batch validationPASS:400client rows,100formal4096/1024requests,8DP/TP/PPworkers, exact per-request prefill coverage. Full replay completion marker observed.

CPU numerical executionFAIL:exit1, event queue empty while scheduler holds unfinished requests. Exact same-commit diagnostic reproduction and first3request reduction both fail at0.29563780122719624s. No predicted mean/absolute error/relative error is available. See analysis/mixed_phase_forward_stall_rca.md for source and phase-control evidence; shared-protocol repair awaits YC decision.
