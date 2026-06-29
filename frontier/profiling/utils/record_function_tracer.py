import json
import uuid
from typing import Optional, Set

import numpy as np
import torch


_CUDA_EXECUTION_EVENT_CATEGORIES = frozenset({"kernel"})


class RecordFunctionTracer:
    def __init__(
        self,
        output_path: str,
        allow_zero_cuda_ops: Optional[Set[str]] = None,
        fail_on_zero_cuda_time: bool = True,
    ):
        trace_id = str(uuid.uuid4())[:8]
        self.trace_path = (
            f"{output_path}/profiler_traces/profiler_trace_{trace_id}.json"
        )
        self.allow_zero_cuda_ops = set(allow_zero_cuda_ops or [])
        self.fail_on_zero_cuda_time = fail_on_zero_cuda_time

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush async CUDA work before stopping the profiler so the final launch
        # still exports its correlated kernel event into the trace.
        torch.cuda.synchronize()
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        self.profiler.export_chrome_trace(self.trace_path)

    def find_children(self, trace, event):
        if not ("dur" in event and "ts" in event):
            return

        children = []
        for e in trace:
            if not ("dur" in e and "ts" in e):
                continue

            # if the ts of the child is completely within the ts of the parent
            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(
        self,
        trace,
        event,
        allowed_categories: Optional[Set[str]] = None,
    ):
        if not ("args" in event and "correlation" in event["args"]):
            return

        for e in trace:
            if not ("args" in e and "correlation" in e["args"]):
                continue

            if e == event:
                continue

            if allowed_categories is not None and e.get("cat") not in allowed_categories:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e

    def get_operation_time_stats(self, debug=False):
        per_operation_times_ms: dict[str, list[float]] = {}

        with open(self.trace_path, "r", encoding="utf-8") as trace_file:
            trace = json.load(trace_file)["traceEvents"]

        for event in trace:
            if not ("cat" in event and event["cat"] == "user_annotation"):
                continue
            event_name = event.get("name", "")
            if not isinstance(event_name, str) or not event_name.startswith("vidur_"):
                # Ignore unrelated annotations emitted by libraries/framework internals.
                continue
            children = self.find_children(trace, event) or []
            cuda_time = 0
            for child in children:
                # Check for both cuda_runtime (cudaLaunchKernel) and cuda_driver (cuLaunchKernel)
                if not ("cat" in child and child["cat"] in ("cuda_runtime", "cuda_driver")):
                    continue
                correlated_event = self.find_correlated_event(
                    trace,
                    child,
                    allowed_categories=_CUDA_EXECUTION_EVENT_CATEGORIES,
                )
                if not correlated_event:
                    continue
                cuda_time += correlated_event["dur"]

            name = event_name.replace("vidur_", "")

            if name not in per_operation_times_ms:
                per_operation_times_ms[name] = []

            per_operation_times_ms[name].append(cuda_time * 1e-3)  # to convert to ms

        for name, times_ms in per_operation_times_ms.items():
            if name in self.allow_zero_cuda_ops:
                continue

            if any(time_ms == 0.0 for time_ms in times_ms) and self.fail_on_zero_cuda_time:
                raise ValueError(
                    f"RecordFunctionTracer: operation '{name}' has zero CUDA kernel time. "
                    f"This means no CUDA kernel was found via correlation ID matching "
                    f"for this user_annotation scope. Possible causes: "
                    f"(1) the operation launched no GPU kernels, "
                    f"(2) the profiler trace is incomplete or corrupted, "
                    f"(3) correlation ID chain is broken. "
                    f"Trace path: {self.trace_path}"
                )

        if debug:
            print(f"[DEBUG] Collected operations: {list(per_operation_times_ms.keys())}")

        return {
            operation: {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
                "count": len(times),
            }
            for operation, times in per_operation_times_ms.items()
        }
