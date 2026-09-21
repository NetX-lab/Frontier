## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Recorded platform admission failure for H800 normal diagnostic job and absence of timing evidence. |

# H800 normal diagnostic failure RCA

## Question

Determine whether `yc26-h800-ar-normal-20260909-01` produced usable normal-mode timing evidence and identify the failure class.

## Ranked synthesis

| Rank | Explanation | Confidence | Basis |
|------|-------------|------------|-------|
| 1 | Platform admission rejected the worker because the requested Mellanox RDMA device was unhealthy. | High | Replica status contains the exact `UnexpectedAdmissionError` message. |
| 2 | Worker script or model initialization failed. | Not supported | Worker never reached `Running`; no container restart or pod IP, and no runtime artifacts were produced. |
| 3 | CUDA/NCCL/model timing failure. | Not supported | No worker process or timing log exists. |

## Evidence

- `brainctl get rjob yc26-h800-ar-normal-20260909-01 -n shai-core -o yaml` reports `phase: Failed`, `reason: RJobFailed`, and `message: Tasks with failed status: [worker]`.
- Replica `yc26-h800-ar-normal-20260909-01-cfcd2084` reports:
  - `phase: Failed`, `reason: UnexpectedAdmissionError`;
  - `nodeName: gpu-h800-0233.host.platform.shaipower.com`;
  - `startTime: 2026-09-09T07:53:29Z`, `finishTimestamp: 2026-09-09T08:02:49Z`;
  - exact message: `Pod was rejected: Allocate failed due to no healthy devices present; cannot allocate unhealthy devices mellanox.com/mlnx_rdma, which is unexpected`.
- The replica had `READY 0/1`, no pod IP, and zero restarts, confirming the worker container did not execute.
- The requested container resources include `nvidia.com/gpu: "8"` and `mellanox.com/mlnx_rdma: "1"`; the admission error names the latter as unhealthy.
- The requested persistent output directory `analysis/h800-ar-normal-01/` is absent. Consequently there are no `environment.log`, `client.log`, `server.log`, `mode_manifest.json`, `server.batch.*.jsonl`, `server.ops.*.jsonl`, `moe_boundary.rank*.jsonl`, or timing rows for this job.

## Conclusion

This run has no timing evidence and must be excluded from normal/scalar/skip reconciliation. The failure occurred during platform resource admission before the diagnostic worker command, vLLM startup, CUDA initialization, or NCCL execution. A valid comparison requires a fresh run on a healthy H800 allocation (or a platform recipe that omits/fixes the unhealthy RDMA resource only if that recipe is verified for this topology).

## Limits

The failed job cannot establish anything about post-MoE AR timing, host submission gaps, payload size, or outer batch span. No CUDA, NCCL, or script root-cause claim can be derived from this artifact.
