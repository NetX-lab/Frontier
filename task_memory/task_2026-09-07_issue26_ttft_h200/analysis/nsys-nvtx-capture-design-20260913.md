## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Added an NVTX-triggered Nsight design and bounded H200 worker. |

# NVTX capture design for a native clean-span diagnostic

The existing native Nsight run uses `Issue26NsightCapture` in
`/data/ycfeng/tmp/vLLM-BS/vllm/v1/issue26_nsys_capture.py`.  That hook calls
`torch.cuda.cudart().cudaProfilerStart()` before the selected first formal
model forward and `cudaProfilerStop()` after the forward returns.  The
corresponding worker uses `--capture-range=cudaProfilerApi` and captures only
CUDA activity.  The run completed the ten-warmup/1100-row gate, but its
process-local capture windows were 99--107 ms while the accepted uninstrumented
native batch reference is 78.119--79.308 ms.  One rank also lacked a local
`cudaProfilerStop` API row and required an earliest-stop fallback.  The result
is therefore diagnostic and cannot be used as a clean 79 ms decomposition.

The host marker anomaly (`formal-start` to client first token of roughly
11.9 s) is outside the Nsight process-local CUDA window.  It shows that
wall-clock client markers cannot be used as Nsight timestamp boundaries.  It
does not establish an 11.9 s GPU interval.  A separate concern remains that a
`cudaProfilerStop` API call can trigger profiler control/flush work in the
serving thread even when CUPTI flush is disabled.

## Controlled NVTX alternative

`nsys profile --help` in the pinned runtime (`issue26-nsys-runtime-20260913-01`)
accepts:

```text
--capture-range=nvtx
--nvtx-capture=range[@domain]
--capture-range-end=stop
```

The `torch.cuda.nvtx.range_push()` and `range_pop()` calls are host marker
operations.  They do not invoke CUDA profiler start/stop or request a profiler
flush.  The proposed range name is `issue26_formal_forward`; the range is
opened and closed at exactly the same first-formal boundary as the existing
hook.  Nsight is configured with `--trace=cuda,nvtx` and the same reduced
sampling, memory, and CPU context options as the CUDA-only worker.  `nsys
stats` is additionally invoked with `--filter-nvtx=issue26_formal_forward` so
post-processing can clip GPU rows to the NVTX range without importing external
wall-clock markers.

Because the current pinned vLLM hook has no NVTX mode, the bounded worker
`tests/e2e/issue26_h200_nsys_nvtx_worker.sh` creates a run-root-only
`sitecustomize.py`.  At Python startup that shim imports
`vllm.v1.issue26_nsys_capture`, subclasses `Issue26NsightCapture`, and replaces
only `maybe_start`/`maybe_stop` with NVTX push/pop.  It reuses the hook's
private first-formal request and token predicates, shared start marker, strict
failure behavior, and one-shot state.  The source checkout remains clean and
its commit is verified before launch.  A status file is checked after health
readiness, so a silently skipped `sitecustomize` import cannot produce an
apparently valid trace.

This design removes one potential perturbation (the CUDA profiler API and its
stop-side control path) while preserving the capture boundary.  It cannot
guarantee a clean 78--79 ms result: Nsight CUDA/CUPTI collection itself can
still perturb kernel scheduling, CUPTI buffers can still be flushed at session
end, and the selected model forward contains asynchronous device work.  The
range is a diagnostic candidate only until a repeated run demonstrates both a
native-scale window and consistent request/batch identity.

## Required validation and limits

The worker is H200-only (`step_main + h200`, eight GPUs, and
`num_gpu_blocks_override=310809`) and retains the standard workload and ten
complete drained warmup replays.  It requires 100 formal requests, 1100 client
rows, exact formal request identity, and a nonempty Nsight report.  No GPU job
was submitted for this design artifact.

The existing `tests/e2e/issue26_nsys_sqlite_breakdown.py` reads
`cuProfilerStart`/`cudaProfilerStop` API rows and cannot consume an NVTX-only
capture window.  A future, separately scoped post-processing change must read
the `NVTX_EVENTS` push/pop rows (or use Nsight's NVTX filter) and map the
selected range to each CUDA context.  Until that parser is added and verified,
an NVTX `.nsys-rep` can establish capture presence and timing diagnostics but
cannot produce the authoritative per-rank compute/communication/memory/idle
table.  This note therefore proposes a measurement design and records the
blocker; it does not authorize Frontier predictor, communication backend,
operator mapping, CPU accounting, or clean/diagnostic reconciliation changes.

## Local checks performed

The new worker passed `bash -n` and `git diff --check`.  The embedded
`sitecustomize.py` extracted from the worker passed `python -m py_compile` and
an isolated fake-module test verified one push followed by one pop, exact range
name, and a `status=PASS` file.  The pinned Nsight CLI accepted the NVTX
capture switches in a no-GPU `/bin/true` dry invocation (the expected local
failure was only missing the host importer binary, not an argument error).
