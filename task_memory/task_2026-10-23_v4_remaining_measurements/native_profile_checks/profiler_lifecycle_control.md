# CPU-only torch profiler lifecycle control

Bounded control for `native_profile_seam.md` §6. No CUDA initialisation, no GPU job, no
native/Docker/platform action. Run 2026-10-23 on the primary host.

## Command

```bash
python3 - <<'PY'
import torch, tempfile, glob, os
from torch.profiler import profile, ProfilerActivity
d = tempfile.mkdtemp(dir=".../native_profile_checks", prefix="trace_")
p = profile(activities=[ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(d, worker_name="w0"))
for start in (1, 2, 3):
    p.start()
    with torch.profiler.record_function(f"window{start}"):
        sum(range(1000))
    p.stop()
print(sorted(os.path.basename(x) for x in glob.glob(d + "/*")))
PY
```

## Environment

- Interpreter: `/usr/bin/python3`, CPython 3.10.6 (the only interpreter on this host; no
  Python 3.12 and no torch 2.13 install is present locally).
- `torch.__version__ == 2.1.0+cu121`. The accepted native06 runtime is
  `torch 2.13.0+cu130`, `vllm 0.28.0` (from `nodes/*/result.json` `versions`), which cannot be
  imported here — so this control is *design-level*, not the runtime version.
- `ProfilerActivity.CPU` only; expected and observed
  `cudaGetDeviceCount failed with code 35` from `CudaDeviceProperties.cpp:27`, i.e. no CUDA
  context was created.

## Result: PASS (API level)

```
torch 2.1.0+cu121 profiler obj profile
start/stop cycle 1 OK; kineto obj profile
start/stop cycle 2 OK; kineto obj profile
start/stop cycle 3 OK; kineto obj profile
['w0.1792726948207158520.pt.trace.json',
 'w0.1792726948208277889.pt.trace.json',
 'w0.1792726948208998658.pt.trace.json']
```

Three start/stop cycles on a single `torch.profiler.profile` object completed without error and
produced three distinct, non-colliding trace files (`trace_56x11l33/`). Source basis:
`_KinetoProfile.start` → `prepare_trace()` constructs a new kineto `prof.profile(...)` per
start, so the object is reusable.

## What this does and does not settle

- Settles: "the profiler object is created once, therefore only one profiling window per
  process" is false at the API level; restart is supported and trace filenames are timestamped.
- Does **not** settle: CUPTI behaviour under CUDA graphs across restarts. torch itself carries a
  `FIXME: CUPTI Lazy Re-init and CUDA Graph crashes` note and sets
  `DISABLE_CUPTI_LAZY_REINIT=1` when inductor triton cudagraphs are enabled
  (`torch/profiler/profiler.py`, `_KinetoProfile.start_trace`). Whether vLLM's breakable/FULL
  capture hits this must be answered at runtime, not here.
- Artifact constraint, independent of lifecycle: `TorchProfilerWrapper._write_profiler_table`
  writes a fixed `profiler_out_{rank}.txt` (`vllm/profiler/wrapper.py:267-275`), so that file is
  overwritten on restart; trace files are not.
