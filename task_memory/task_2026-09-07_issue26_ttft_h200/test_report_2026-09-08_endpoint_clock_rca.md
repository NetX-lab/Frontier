## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the exact-workload endpoint failure and independent long-run clock RCA. |

# Endpoint clock RCA and validation

## Execution

Exact-case worker: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_endpoint_ab_worker.sh`.
Launcher preserving the full allocation/image/proxy command: `/data/ycfeng/tmp/issue26-h200-network/launch-endpoint-ab-02.sh`.
Evidence: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-endpoint-ab-02`.

Independent probe: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_prefill_clock_drift_probe.py`.
Worker command after sourcing the recorded environment:
```bash
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_prefill_clock_drift_probe.py /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-clock-drift-02/clock_drift.jsonl
```
Launcher: `/data/ycfeng/tmp/issue26-h200-network/launch-clock-drift-02.sh`.
Wrapper: `/data/ycfeng/tmp/issue26-h200-network/clock-drift-worker.sh`.
Job: `yc26-h200-clock-drift-20260908-0052`, H200 x8, step_main, image digest `sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.
Python 3.10.16 in `vllm-bs-0.10.2`; Torch 2.8.0+cu128; FlashInfer 0.3.0. Full environment/NVML observations are retained in `runs/h200-clock-drift-02/environment.log`.

## Criteria

The client must deliver 400 unique full-length requests in each A/B mode, with three drained warmup replays before formal IDs. Every canonical endpoint must occur after queue arrival and no later than natural synchronization return within the measured anchor bracket.

The independent probe must produce eight GPUs x six elapsed-time checkpoints (0/30/60/90/120/150 seconds). Each GPU's retained startup anchor is compared against a new idle CUDA event bracketed by host monotonic reads. The existing short-probe clock precision criterion was at most 100 us non-overlap. Complete process execution is distinct from satisfying that clock criterion.

## Evidence

**Client correction PASS:** baseline 400/400 unique requests, all 4096 input / 1024 observed output tokens. Four complete, sequential replay intervals; no repeated HTTP transport failure.

**Endpoint integration FAIL:** after 100 warmup requests completed, rank 4 failed near 130 seconds from its clock anchor:
```text
frontier_prefill.py:56, finish_after_sync
RuntimeError: CUDA/host clock mapping exceeds observed completion time
```
The client subsequently saw no streamed token because the engine had failed. No endpoint-mode formal TTFT value is admissible.

**Independent probe execution PASS; fixed-anchor clock invariance FAIL:** 48/48 finite records, six checkpoints per GPU, `CLOCK_DRIFT_PROBE_COMPLETE`, launcher ExecMainStatus=0. Final checkpoint:

| GPU | Host elapsed (s) | CUDA elapsed (s) | CUDA − host midpoint (us) | Non-overlap (us) |
| --- | --- | --- | --- | --- |
| 0 | 151.185566174 | 151.185843750 | 275.139 | 255.998 |
| 1 | 151.021166383 | 151.021234375 | 66.442 | 49.028 |
| 2 | 150.857386224 | 150.857640625 | 253.883 | 238.287 |
| 3 | 150.699253898 | 150.699765625 | 511.392 | 495.805 |
| 4 | 150.538217452 | 150.538859375 | 641.458 | 625.640 |
| 5 | 150.375429227 | 150.375625000 | 195.313 | 179.328 |
| 6 | 150.205248695 | 150.205484375 | 235.579 | 219.879 |
| 7 | 150.024990801 | 150.024937500 | -53.488 | 37.707 |

The largest final bracket separation is 625.640 us, exceeding the 100 us diagnostic precision criterion. At about 60 seconds, rank 4 already showed +265.57 us midpoint offset and 249.75 us non-overlap. This is independent of the model, request transport, or scheduler. Different devices have different offsets and signs.

## RCA and limits

Observed cause: elapsed CUDA event time and host monotonic time are not interchangeable through one fixed initial offset over a minutes-long run. The recorder exported only the initial host bracket and therefore understated subsequent mapping uncertainty. The causal-order guard correctly rejected a mapped completion later than the already observed natural synchronization return.

The probe establishes drift between clock domains; it does not identify oscillator calibration, host time slewing, or driver behavior as the underlying physical source. No such unverified attribution is needed to reject the fixed-anchor conversion. A uniform scaling factor or wider guard would not establish a valid per-request timestamp.

Proposed D005: refresh the clock anchor for each prefill batch before forward execution and measure the added local event synchronization overhead. Preserve the original queue-visible arrival -> prefill forward completion metric. User decision is pending; no such path change has been applied.

Simulation predicted TTFT / actual canonical TTFT / absolute error / relative error: **not available**. This report diagnoses the measurement infrastructure, not a Frontier latency gap.
