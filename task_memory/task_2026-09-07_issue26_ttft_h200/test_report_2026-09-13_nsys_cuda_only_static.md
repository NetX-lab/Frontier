## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Added the execution update after the CUDA-only worker completed. |
| 2026-09-13 | Prepared a reduced-overhead native/naive Nsight CUDA-only capture worker and H200 launch manifest; no GPU RJob submitted while the PPLX allocation remains active. |

# Native/naive CUDA-only profiling preparation

## Outcome

A dedicated H200 worker is ready for a lower-perturbation diagnostic capture. It keeps the standard first-formal replay and identity contract while removing Nsight's explicit NCCL API/GPU tracing hooks. CUDA GPU activity tracing remains enabled, so NCCL kernel names remain available for communication classification. This is intended to reduce CUPTI callback overhead relative to the previous `--trace=cuda,nvtx,nccl --nccl-trace=api-coll,gpu` run.

The worker is prepared but not submitted because `yc26-h200-pplx-clean-20260913-01` is still in `Starting` and owns the H200 allocation queue. The new launch must run only after central scheduling permits it.

## Files and exact paths

- Worker: `tests/e2e/issue26_h200_nsys_cuda_only_worker.sh`
- Launch manifest: `analysis/nsys-cuda-only-20260913-run04/launch.sh`
- Expected output: `analysis/nsys-cuda-only-20260913-run04/`
- Existing parser: `tests/e2e/issue26_nsys_breakdown.py`
- Source: `/data/ycfeng/tmp/vLLM-BS@0f34fb271fd66d7dd84201ebdd4722781f829390` (the clean source plus the independent Issue26 Nsight hook; no working-tree edits)

## Configuration

- H200 / `step_main` / `--positive-tags=h200` / 8 GPUs / 64 CPUs / 409600 MiB
- `num_gpu_blocks_override=310809`
- Qwen3-30B-A3B-Instruct-2507 dummy weights, TP4/DP2/PP1/EP8
- BF16, FLASHINFER, eager execution, uniform routing
- Prefix caching and chunked prefill disabled
- 4096-prefill / 1024-output, 100 requests per replay
- 10 complete drained warmup replays, then 100 formal requests (1100 rows)
- `VLLM_ALL2ALL_BACKEND=naive`
- `VLLM_FRONTIER_INSTRUMENTATION=0` and all Frontier batch/operator/routing loggers unset

## Capture design

The worker invokes:

```text
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --cuda-event-trace=false \
  --cuda-trace-scope=process-tree \
  --trace-fork-before-exec=false
```

The Issue26 hook calls `cudaProfilerStart` immediately before the first formal batch's model forward and `cudaProfilerStop` immediately after that forward returns. The accepted clean native reference remains `78.118782043--79.307357788 ms`; a profiler window outside this scale is diagnostic and cannot be used as a clean replacement.

The reduced trace omits `--trace=nvtx`, `--trace=nccl`, and `--nccl-trace=api-coll,gpu`. Communication classification will use CUDA GPU kernel names (for example `ncclDevKernel_*`, `cross_device_reduce_*`) from `cuda_gpu_trace.csv`. No NCCL API timing or CUPTI NCCL callback population will be available in this variant.

## Static verification

Commands:

```bash
bash -n tests/e2e/issue26_h200_nsys_cuda_only_worker.sh \
  analysis/nsys-cuda-only-20260913-run04/launch.sh
git diff --check
git -C /data/ycfeng/tmp/vLLM-BS status --porcelain
git -C /data/ycfeng/tmp/vLLM-BS rev-parse HEAD
```

Observed:

- `bash -n`: PASS
- `git diff --check`: PASS
- vLLM working tree: clean
- vLLM HEAD: `0f34fb271fd66d7dd84201ebdd4722781f829390`
- New worker tracked status: pending commit by coordinator

The launch script has not been run. No GPU timing or decomposition result exists from this variant.

## Execution update

The launch was subsequently submitted as RJob
`yc26-h200-nsys-cuda-only-20260913-04` and completed with platform status
`Succeeded`. Its runtime and postprocess artifacts are recorded in
`analysis/nsys-cuda-only-20260913-run04/` and
`analysis/nsys-cuda-only-20260913-run04-postprocess/`. The formal execution
report is `test_report_2026-09-13_nsys_cuda_only_run04.md`; that report
supersedes the preparation-only status above and records the diagnostic
`99.196101--106.824398 ms` rank-aware windows and their limitations.

## Acceptance and limits

After submission, the run must independently pass: ten warmup phases with 100 completions and full drain each; 100 formal completions; 1100 unique client rows; first formal identity `pf4096_dc1024:0`; complete DP0 TP0--TP3 marker or device traces; and capture start/stop artifacts. The resulting interval unions will be labeled a profiler diagnostic. They can explain category populations and rank-local idle placement but cannot close the clean 78--79 ms gate or authorize Frontier corrections.
