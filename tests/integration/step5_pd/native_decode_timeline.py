"""Reduce the decode torch-profiler traces of a native Step P/D run to per-iteration device timelines.

Diagnostic only: the native step times read here explain where Frontier and vLLM differ;
they are never a Frontier prediction input (D31.10).

The ``device_timeline`` run kind (canonical_stage.py) profiles the decode servers during short
windows of the first formal repeat. Every decode worker writes one trace per window, named
``<window>_dp<d>_pp<p>_tp<t>_dcp<c>_ep<e>_rank<r>.<time>.pt.trace.json.gz``. A worker iteration
starts at the worker's ``execute_context_<n>(<tokens>)_generation_<n>(<tokens>)`` annotation and
ends where the next one starts on the same thread. For each iteration:

- ``period_ms``: annotation start to the next annotation start, the worker's step time;
- ``execute_ms``: the annotation itself, the host time of ``execute_model``;
- the device work launched on that thread during the iteration, joined to the CUDA runtime or
  driver call by correlation id (the kernels of a CUDA graph replay carry the
  ``cudaGraphLaunch`` call's id): ``busy_ms`` (union of the intervals), ``span_ms`` (first start
  to last end), and the idle time inside the span split at each gap into the part before the
  call that launched the next device work returned (``idle_host_wait_ms``: the device waited
  for the host) and the rest (``idle_other_ms``), as in glm53's forward_device_timeline.py;
- ``device_ms_by_category``: summed device time per category of ``KERNEL_CATEGORIES``;
- ``wall_busy_ms``: union of all device activity of the trace inside the period, whichever
  thread launched it.

The summary groups the decode-only iterations (no context request) of each window and DP
engine by their number of decode requests, with medians over iterations and TP ranks, and lists
the kernels left in the ``other`` category.

Usage: native_decode_timeline.py <trace_dir>... --output timeline.json
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from pathlib import Path
import re
import statistics

TRACE_NAME = re.compile(r"^(?P<window>.+)_dp(?P<dp>\d+)_pp\d+_tp(?P<tp>\d+)_dcp\d+_ep(?P<ep>\d+)"
                        r"_rank(?P<rank>\d+)\..*\.pt\.trace\.json\.gz$")
STEP_ANNOTATION = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
# The first category whose name fragment occurs in a kernel name wins.
KERNEL_CATEGORIES = (
    ("tp_all_reduce", ("AllReduce", "cross_device_reduce", "allreduce", "all_reduce")),
    ("ep_collective", ("AllToAll", "all_to_all", "AllGather", "all_gather", "ReduceScatter",
                       "reduce_scatter", "deep_ep", "deepep", "nvshmem", "SendRecv")),
    ("moe_compute", ("moe", "MoE", "Moe", "expert", "grouped", "topk", "routing")),
    ("attention", ("attn", "Attn", "attention", "Attention", "flash", "fmha", "mla", "Mla",
                   "indexer", "Indexer", "sparse", "paged", "cache")),
    ("gemm", ("gemm", "Gemm", "GEMM", "nvjet", "xmma", "cutlass", "matmul")),
)


def kernel_category(event: dict) -> str:
    if event["cat"] != "kernel":
        return "copy"
    for category, fragments in KERNEL_CATEGORIES:
        if any(fragment in event["name"] for fragment in fragments):
            return category
    return "other"


def merged(intervals: list[tuple[float, float]]) -> list[list[float]]:
    """The disjoint union of the intervals, in time order."""
    union: list[list[float]] = []
    for start, end in sorted(intervals):
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    return union


def union_ms(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merged(intervals)) / 1000.0


def iterations(trace: dict) -> list[dict]:
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    steps = sorted((event for event in events if event.get("cat") == "user_annotation"
                    and STEP_ANNOTATION.match(event["name"])), key=lambda event: event["ts"])
    device = sorted((event for event in events if event.get("cat") in DEVICE_CATEGORIES),
                    key=lambda event: event["ts"])
    device_union = merged([(event["ts"], event["ts"] + event["dur"]) for event in device])
    union_starts = [start for start, _ in device_union]
    device_by_correlation: dict[int, list[dict]] = {}
    for event in device:
        device_by_correlation.setdefault(event["args"]["correlation"], []).append(event)
    launches_by_thread: dict[tuple, list[dict]] = {}
    for event in events:
        if event.get("cat") in LAUNCH_CATEGORIES:
            launches_by_thread.setdefault((event["pid"], event["tid"]), []).append(event)
    for launches in launches_by_thread.values():
        launches.sort(key=lambda event: event["ts"])
    starts_by_thread = {thread: [event["ts"] for event in launches]
                        for thread, launches in launches_by_thread.items()}

    steps_by_thread: dict[tuple, list[dict]] = {}
    for step in steps:
        steps_by_thread.setdefault((step["pid"], step["tid"]), []).append(step)
    records = []
    for thread, thread_steps in steps_by_thread.items():
        launches, starts = launches_by_thread.get(thread, []), starts_by_thread.get(thread, [])
        for step, following in zip(thread_steps, thread_steps[1:]):
            begin, end = step["ts"], following["ts"]
            context, context_tokens, generation, generation_tokens = map(
                int, STEP_ANNOTATION.match(step["name"]).groups())
            inside = launches[bisect.bisect_left(starts, begin):bisect.bisect_left(starts, end)]
            work = sorted(((activity, launch["ts"] + launch["dur"]) for launch in inside
                           for activity in device_by_correlation.get(launch["args"]["correlation"], [])),
                          key=lambda pair: pair[0]["ts"])
            record = {"step_start_us": begin, "context_requests": context, "context_tokens": context_tokens,
                      "decode_requests": generation, "decode_tokens": generation_tokens,
                      "period_ms": (end - begin) / 1000.0, "execute_ms": step["dur"] / 1000.0,
                      "device_activities": len(work)}
            overlapping = device_union[max(0, bisect.bisect_right(union_starts, begin) - 1):
                                       bisect.bisect_left(union_starts, end)]
            record["wall_busy_ms"] = sum(max(0.0, min(stop, end) - max(start, begin))
                                         for start, stop in overlapping) / 1000.0
            if work:
                first = work[0][0]
                covered_end = first["ts"] + first["dur"]
                host_wait = other = 0.0
                by_category: dict[str, float] = {}
                for activity, launch_end in work:
                    category = kernel_category(activity)
                    by_category[category] = by_category.get(category, 0.0) + activity["dur"] / 1000.0
                    if activity is not first and activity["ts"] > covered_end:
                        gap = activity["ts"] - covered_end
                        waited = min(gap, max(0.0, launch_end - covered_end))
                        host_wait += waited
                        other += gap - waited
                    covered_end = max(covered_end, activity["ts"] + activity["dur"])
                busy = union_ms([(activity["ts"], activity["ts"] + activity["dur"]) for activity, _ in work])
                record.update(busy_ms=busy, span_ms=(covered_end - first["ts"]) / 1000.0,
                              idle_host_wait_ms=host_wait / 1000.0, idle_other_ms=other / 1000.0,
                              device_ms_by_category=by_category)
            records.append(record)
    return sorted(records, key=lambda record: record["step_start_us"])


def other_kernels(trace: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "kernel" and kernel_category(event) == "other":
            totals[event["name"]] = totals.get(event["name"], 0.0) + event["dur"] / 1000.0
    return totals


def summarize(traces: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for trace in traces:
        for record in trace["iterations"]:
            if record["context_requests"] == 0 and "busy_ms" in record:
                key = (trace["window"], trace["dp"], record["decode_requests"])
                groups.setdefault(key, []).append(record)
    summary: dict = {}
    for (window, dp, decode_requests), records in sorted(groups.items()):
        categories = sorted({name for record in records for name in record["device_ms_by_category"]})
        summary.setdefault(window, {}).setdefault(f"dp{dp}", {})[str(decode_requests)] = {
            "iterations": len(records),
            **{field: statistics.median(record[field] for record in records)
               for field in ("period_ms", "execute_ms", "wall_busy_ms", "busy_ms", "span_ms",
                             "idle_host_wait_ms", "idle_other_ms")},
            "device_ms_by_category": {name: statistics.median(record["device_ms_by_category"].get(name, 0.0)
                                                              for record in records)
                                      for name in categories},
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("trace_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    traces, others = [], {}
    for path in sorted(path for directory in args.trace_dirs for path in directory.glob("*.pt.trace.json.gz")):
        name = TRACE_NAME.match(path.name)
        if name is None:
            raise SystemExit(f"Unexpected trace name: {path}")
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        traces.append({"file": str(path), "window": name["window"],
                       **{key: int(name[key]) for key in ("dp", "tp", "ep", "rank")},
                       "iterations": iterations(trace)})
        for kernel, duration in other_kernels(trace).items():
            others[kernel] = others.get(kernel, 0.0) + duration
        del trace
    result = {"traces": traces, "summary": summarize(traces),
              "other_kernels_ms": dict(sorted(others.items(), key=lambda item: -item[1])[:40])}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
