"""Coordinate a low-perturbation Nsight Systems CUDA profiler window.

The module is loaded through ``PYTHONPATH`` in the profiling worker.  Each
vLLM Python process watches a shared directory and calls the CUDA profiler API
after warmup and at the first formal token.  It contains no operator logging or
timing instrumentation.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def _watch_cuda_profiler() -> None:
    control = os.environ.get("ISSUE26_NSYS_CONTROL_DIR")
    if not control:
        return
    control_dir = Path(control)
    start_marker = control_dir / "start"
    stop_marker = control_dir / "stop"
    deadline = time.monotonic() + 1800.0
    try:
        import torch
    except Exception:
        return

    while time.monotonic() < deadline and not start_marker.exists():
        time.sleep(0.02)
    if not start_marker.exists():
        return
    while time.monotonic() < deadline:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.profiler.start()
                break
        except Exception:
            return
        time.sleep(0.02)
    else:
        return

    while time.monotonic() < deadline and not stop_marker.exists():
        time.sleep(0.02)
    try:
        torch.cuda.profiler.stop()
    except Exception:
        pass


if os.environ.get("ISSUE26_NSYS_CONTROL_DIR"):
    threading.Thread(target=_watch_cuda_profiler, daemon=True,
                     name="issue26-nsys-cuda-trigger").start()
