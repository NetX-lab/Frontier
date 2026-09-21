## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded two PPLX clean-boundary RJobs that failed before worker start because H200 scheduling/preemption was unavailable. |

# PPLX clean-boundary platform failures

## Execution

The approved H200 recipe was used for both attempts:

- `step_main`, `h200`, 8 GPUs, 64 CPUs, 409600 MiB
- `num_gpu_blocks_override=310809`
- Qwen3-30B-A3B dummy model, TP4/DP2/EP8/PP1, BF16, eager, FLASHINFER
- uniform routing, prefix caching and chunked prefill disabled
- 4096-prefill / 1024-output, ten drained 100-request warmups, 100 formal requests
- PPLX source `448f2b65e7679ae7490114ad382b6ba79becb3c3`
- PPLX overlay `/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03`
- NCCL/NVSHMEM library overlay from `/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01`
- controlled `VLLM_MOE_DP_CHUNK_SIZE=4096`

Commands:

```bash
bash analysis/pplx-clean-20260913-run01/launch.sh
bash analysis/pplx-clean-20260913-run02/launch.sh
```

Both launch manifests use the fixed image and `/data` mount. `predict-only` passed before the retry submission.

## Criteria and evidence

| RJob | Platform result | Worker started | Client/boundary artifacts | Cause |
| --- | --- | --- | --- | --- |
| `yc26-h200-pplx-clean-20260913-01` | `Failed` | No | None; only `launch.sh` | `FailedScheduling`; nominated H200 node had terminating pod |
| `yc26-h200-pplx-clean-20260913-02` | `Failed` | No | None; only `launch.sh` | `FailedScheduling`; no eligible H200 node / preemption unavailable |

`brainctl describe rjob ...` reported 0/4372 eligible nodes, with insufficient GPU/memory/CPU on the H200 pool and a terminating pod on the nominated node. No `preflight/environment.log`, `runtime/client.jsonl`, `server.boundary*.jsonl`, or `boundary_analysis.json` was created. Therefore no PPLX latency or timing conclusion is made.

## Limits and next action

These attempts establish a platform scheduling failure only. They do not test the PPLX source, chunk-size setting, or clean boundary hook. Retain both RJobs and submit at most one fresh retry after the active native Nsight run releases the H200 worker, with the explicit overlay variables preserved.
