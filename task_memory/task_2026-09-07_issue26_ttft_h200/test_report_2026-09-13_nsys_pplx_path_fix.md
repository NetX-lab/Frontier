## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the Nsight worker fix that preserves optional PPLX Python and library overlays. |

# Nsight PPLX overlay propagation fix

## Execution

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`
- Worker: `tests/e2e/issue26_h200_nsys_profiler_worker.sh`
- Verification environment: shell `bash`; no GPU command was needed for this harness-only correction.
- Verification commands:

```bash
bash -n tests/e2e/issue26_h200_nsys_profiler_worker.sh
git diff --check
```

## Criteria

The Nsight worker must preserve the PPLX runtime supplied by an RJob through the target process and the Nsight child process:

- `ISSUE26_OPTIONAL_PYTHONPATH` is prepended to `PYTHONPATH` after the source checkout is added.
- `ISSUE26_EXTRA_LD_LIBRARY_PATH` is prepended to `LD_LIBRARY_PATH` before `TARGET_LD_LIBRARY_PATH` is captured for `nsys profile --env-var`.
- `ISSUE26_LD_PRELOAD`, when supplied, is propagated.
- Default native/naive behavior remains unchanged when the variables are unset.

## Evidence

PASS. The worker now appends the optional Python overlay and merges optional libraries before constructing the Nsight environment. The change is committed as `cd592796` (`Preserve optional PPLX paths in Nsight worker`).

The corrected PPLX Nsight RJob `yc26-h200-pplx-nsys-profiler-20260913-02` was submitted with:

```text
ISSUE26_OPTIONAL_PYTHONPATH=/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03
ISSUE26_EXTRA_LD_LIBRARY_PATH=/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nccl/lib:/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nvshmem/lib
```

At report creation the RJob remained `Starting`; no PPLX timing or Nsight report is claimed from it yet. The previous PPLX Nsight run failed before vLLM execution because the worker discarded these overlays, and its `server.log` contained only `No reports were generated`.
