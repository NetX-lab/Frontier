## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Integrated reviewed corrected MoE data and existing TP4 AR parameter; observed remaining first-forward gate failure. |

# Direct verification: integrated first-forward correction

Result: scoped correction integration PASS; first-forward CUDA gate FAIL. Actual Frontier request0 boundary is **59.190354384 ms**, compared with vLLM batch-only before79.307357788ms and after78.118782043ms. Error is **−25.365872682% / −24.230315891%** (absolute20.117003404ms /18.928427660ms). No official TTFT or fullcase result is claimed.

## What changed and why

The independently reviewed existing `nvlink_allreduce_launch_overhead_us` case field is4.384788772964477us/step, replacing50us. Bandwidth450GB/s, efficiency.8, latency.5us and ideal EP dispatch/combine are unchanged. Four primitive training sizes with heldout16MiB support this fixed-TP4, BF16, PyNCCL RING_LL correction; it is an effective collective floor, not an independent CPU-overhead measurement. Full evidence and source scope are in `../d019-communication-sweep.md` and `../d019-communication-fit-review.md`.

Corrected MoE inputs/cache are those already freshly trained in the current task. All compute query features, values, paths and counts exactly match the MoE-only prior query. No fitting occurred in this execution. Fresh communication cache and htsim output paths prevent reusing old communication costs.

| Checkpoint | Actual DES first-forward ms | Interpretation |
| --- | ---: | --- |
| Original current-task profiles | 65.790076904 | Incomplete GG and overestimated AR |
| Corrected GG only | 72.327535217 | Verified new measured expert path; AR error still present |
| Corrected GG + calibrated existing AR | 59.190354384 | Scoped integrated result; CUDA gate fails |
| vLLM batch-only before | 79.307357788 | Reference, no per-op event probes |
| vLLM batch-only after | 78.118782043 | Reference, no per-op event probes |

AR48-layer prediction falls17.899443200→4.762262367ms. Independent arithmetic72.327535217−17.899443200+4.762262367 matches actual59.190354384 within5.69e−14ms. The previously sub10% MoE-only total cannot establish closure: a known AR overestimate was offsetting other omissions/differences.

## Execution and verification

CPU master; conda dev-vidur-v03-hopper-e2e, Python3.13.13. Exact command from `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_predictor_query_audit.py \
  --config task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-first-forward-integrated/config.json \
  --output /data/ycfeng/tmp/issue26-d019-first-forward-integrated-01/query \
  > /data/ycfeng/tmp/issue26-d019-first-forward-integrated-01.log 2>&1
```

`--allow-fit` was omitted: zero RF.fit calls;2592 actual query returns,13 unique entries. Harness stops before handling first stage-end event after its completion time has been scheduled, so it validates that modeled boundary without claiming a completed100-request run. During startup, process briefly waited in `wait_on_page_bit_common` while importing a Python cache file, then resumed without intervention; no configuration workaround or cleanup was applied. Process exit0 and actual receipt `PASS_BOUNDED_PREDICTOR_QUERY_AUDIT` are preserved with `query_receipt.json`, `run.log`, `input_receipt.json` and `validation.json`.

## Remaining causes and next boundary

- Linear profiling has a demonstrated context effect; the same kernels in the existing synthetic hot control approximate vLLM's event scopes, while precreating Event objects has only a small effect. Production profile/context selection is not changed by this result; see `../d019-linear-timing-context.md`.
- Primitive calibrated AR still differs from in-context events; waits are not fitted into bandwidth/launch values.
- Ideal-versus-naive communication remains the explicit D020 approximation. Its separate measurements do not prove a precise additive share of this18.9–20.1ms residual. Do not silently implement the optional protocol or compensate it with a constant.
- Physical MoE populations F4096/V4097, missing embedding/output-init/final-norm model terms, instrumentation perturbation, and surrounding forward execution still require qualified attribution. Existing inventory distinguishes these from RF fit errors.
- CUDA closure precedes workflow CPU integration. No CPU overhead has been added; CUDA-marker internal host gaps must not also be charged outside the forward.
