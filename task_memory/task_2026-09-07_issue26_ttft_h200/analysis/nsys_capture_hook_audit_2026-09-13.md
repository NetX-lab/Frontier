## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the run-02 Nsight no-report root cause and temporary vLLM capture-hook implementation. |

# Issue 26 Nsight Capture Hook Audit

## Root cause

The worker invokes Nsight Systems with `--capture-range=cudaProfilerApi` and
`--capture-range-end=stop` (`tests/e2e/issue26_h200_nsys_profiler_worker.sh:90-92`).
That mode requires the target process to call the CUDA profiler API. The
client only writes JSON files named `start` and `stop`
(`tests/e2e/issue26_naive_profiler_client.py:198-207` and `210-222`); creating
those files does not call `cudaProfilerStart` or `cudaProfilerStop`.

The repository contains `tests/e2e/issue26_nsys_cuda_trigger.py`, but the
worker only adds `tests/e2e` to `PYTHONPATH`. No vLLM module imports
`issue26_nsys_cuda_trigger`, and Python does not execute arbitrary modules
merely because their directory is on `PYTHONPATH`. A source search of the
native vLLM checkout at commit `46f7b179fd3bf42b9616dc4670cba419afdb2085`
found no `cudaProfilerStart` or `cudaProfilerStop` call under `vllm/`.

Run-02 therefore completed the client workload and wrote both control files,
but no process entered a CUDA profiler range. Its server log ends with:

```text
Processing events...
Generated:
	No reports were generated
```

Run-01 additionally had an Nsight host-importer packaging failure. Run-02
included the host importer and did not report that error, while still
generating no reports. This separates the run-02 root cause (missing target
API calls) from the independent run-01 importer problem.

## Temporary source implementation

The temporary checkout `/data/ycfeng/tmp/vLLM-BS` now has local commit
`db29dc406709b7ba6db420a31b56dfe650e0c476` with two files:

- `vllm/v1/issue26_nsys_capture.py`: process-local `Issue26NsightCapture`.
- `vllm/v1/worker/gpu_model_runner.py`: creates the hook in the runner, calls
  `maybe_start(scheduler_output)` immediately before `self.model(...)`, and
  calls `maybe_stop()` immediately after the model forward returns.

The hook is independent of Frontier logging. It calls
`torch.cuda.cudart().cudaProfilerStart()` once when all of the following hold:

1. `ISSUE26_NSYS_CONTROL_DIR/start` exists, proving the warmup drain and
   client-side formal phase transition have occurred.
2. The worker batch contains the configured first formal request ID.
3. That request has the configured prefill token count (4096 by default).

After the model forward returns, before sampling and bookkeeping, the same
process calls `torch.cuda.cudart().cudaProfilerStop()` exactly once. The
boundary therefore matches the model-forward-only diagnostic span. The state
is process-local, so the selected DP lane's TP workers each create their own
CUDA profiler range. No operation logger, batch logger, routing logger, CUDA
timing event, synchronization, or record-function instrumentation is enabled
by this hook.

The default request identity is `cmpl-pf4096_dc1024:0-0`, matching the worker
batch identity observed in existing diagnostics. The suffix-normalization
fallback also accepts a worker ID whose `-0` suffix has already been removed.

## Diagnostic environment

Use the following variables for the temporary diagnostic run:

```bash
export ISSUE26_NSYS_CONTROL_DIR="$RUN_ROOT/nsys-control"
export ISSUE26_NSYS_FORMAL_REQUEST_ID="cmpl-pf4096_dc1024:0-0"
export ISSUE26_NSYS_FORMAL_PREFILL_TOKENS=4096
export ISSUE26_NSYS_CAPTURE_STRICT=1
export VLLM_FRONTIER_INSTRUMENTATION=0
unset VLLM_FRONTIER_BATCH_LOG_PATH
unset VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
```

`ISSUE26_NSYS_CAPTURE_STRICT=1` makes a CUDA profiler API failure terminate
the diagnostic process instead of silently disabling the hook. For a normal
vLLM process with `ISSUE26_NSYS_CONTROL_DIR` unset, the hook is a no-op and
does not touch the CUDA profiler API.

Because the source checkout's HEAD changed, the worker's source guard must
use the temporary commit explicitly:

```bash
export ISSUE26_DIAGNOSTIC_VLLM_COMMIT=db29dc406709b7ba6db420a31b56dfe650e0c476
```

The Frontier worker script itself was not modified in this audit.

## Verification

Source syntax and whitespace checks passed:

```bash
git -C /data/ycfeng/tmp/vLLM-BS diff --check
python - <<'PY'
from pathlib import Path
for path in (
    Path('/data/ycfeng/tmp/vLLM-BS/vllm/v1/issue26_nsys_capture.py'),
    Path('/data/ycfeng/tmp/vLLM-BS/vllm/v1/worker/gpu_model_runner.py'),
):
    compile(path.read_text(), str(path), 'exec')
PY
```

A fake CUDART smoke test passed: a warmup request did not start capture, the
configured first-formal 4096-token request produced exactly one `start` call,
and `maybe_stop()` produced exactly one `stop` call. A separate no-op smoke
test with `ISSUE26_NSYS_CONTROL_DIR` unset verified that no CUDA API was
touched.

No GPU job was launched as part of this source audit. A future run must still
verify a complete 10-warmup/100-formal client artifact, the source hook's
selected-batch identity, a `.nsys-rep`, and all required stats exports before
using the breakdown parser.
