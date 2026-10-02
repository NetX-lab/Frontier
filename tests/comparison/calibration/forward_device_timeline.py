#!/usr/bin/env python3
"""Reduce the torch-profiler trace of a vLLM run to one record per eager forward.

  python forward_device_timeline.py --trace-dir <dir> --output <device_timeline.json>

The ``device_timeline`` mode of vllm_replay.py makes every process of the server
write a ``*.pt.trace.json.gz`` file to one directory. This script reads the
trace of global rank 0, the TP rank whose steps the CPU probe records; the
trace header's ``distributedInfo`` names the rank (the API server's CPU-only
trace has none). Each ``Forward`` range of the model runner becomes one record
of an eager forward, or is counted as a CUDA graph replay when its thread
called ``cudaGraphLaunch``:

- the host range of the forward call;
- its device work: the kernels, copies and sets that the CUDA runtime and
  driver calls made on the forward's thread inside that range launched,
  joined by correlation id;
- the device span, from the first device start to the last device end; the
  busy time, the union of the device intervals; the summed device time; and
  the idle time, the span minus the busy time;
- the idle time split at each gap into the part before the call that launched
  the next device work returned (the device waited for the host) and the rest;
- device time by kernel, copy or set name.

Timestamps are microseconds on the host's real-time clock: kineto writes each
``ts`` relative to the header's ``baseTimeNanoseconds``. Runs on the standard
library only, inside the vLLM image.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from pathlib import Path

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
GRAPH_LAUNCHES = ("cudaGraphLaunch", "cuGraphLaunch")


def trace_header(path: Path) -> dict:
    """The JSON members kineto writes before ``traceEvents``."""
    lines = []
    with gzip.open(path, "rt") as trace:
        for line in trace:
            if line.lstrip().startswith('"traceEvents"'):
                break
            lines.append(line)
    return json.loads("".join(lines).rstrip().rstrip(",") + "}")


def rank0_trace(trace_dir: Path) -> Path:
    ranks = {path: trace_header(path).get("distributedInfo", {}).get("rank")
             for path in sorted(trace_dir.glob("*.pt.trace.json.gz"))}
    rank0 = [path for path, rank in ranks.items() if rank == 0]
    if len(rank0) != 1:
        raise SystemExit(f"expected one rank-0 trace in {trace_dir}, found ranks {ranks}")
    return rank0[0]


def device_timeline(trace: dict) -> dict:
    base_us = trace["baseTimeNanoseconds"] / 1000.0
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    forwards = sorted((event for event in events
                       if event.get("cat") == "user_annotation" and event["name"] == "Forward"),
                      key=lambda event: event["ts"])
    device_by_correlation: dict[int, list[dict]] = {}
    for event in events:
        if event.get("cat") in DEVICE_CATEGORIES:
            device_by_correlation.setdefault(event["args"]["correlation"], []).append(event)
    launches_by_thread: dict[tuple, list[dict]] = {}
    for event in events:
        if event.get("cat") in LAUNCH_CATEGORIES:
            launches_by_thread.setdefault((event["pid"], event["tid"]), []).append(event)
    for launches in launches_by_thread.values():
        launches.sort(key=lambda event: event["ts"])
    starts_by_thread = {thread: [event["ts"] for event in launches]
                        for thread, launches in launches_by_thread.items()}

    records, graph_forwards = [], 0
    for forward in forwards:
        thread = (forward["pid"], forward["tid"])
        launches = launches_by_thread.get(thread, [])
        starts = starts_by_thread.get(thread, [])
        end = forward["ts"] + forward["dur"]
        inside = launches[bisect.bisect_left(starts, forward["ts"]):bisect.bisect_right(starts, end)]
        if any(launch["name"].startswith(GRAPH_LAUNCHES) for launch in inside):
            graph_forwards += 1
            continue
        work = sorted(((activity, launch["ts"] + launch["dur"]) for launch in inside
                       for activity in device_by_correlation.get(launch["args"]["correlation"], [])),
                      key=lambda pair: pair[0]["ts"])
        if not work:
            raise ValueError(f"eager Forward at ts {forward['ts']} launched no device work")
        first = work[0][0]
        covered_end = first["ts"] + first["dur"]
        busy, host_wait, other = first["dur"], 0.0, 0.0
        by_name: dict[str, float] = {}
        for activity, launch_end in work:
            by_name[activity["name"]] = by_name.get(activity["name"], 0.0) + activity["dur"]
            activity_end = activity["ts"] + activity["dur"]
            if activity is first:
                continue
            if activity["ts"] > covered_end:
                gap = activity["ts"] - covered_end
                waited = min(gap, max(0.0, launch_end - covered_end))
                host_wait += waited
                other += gap - waited
                busy += activity["dur"]
            else:
                busy += max(0.0, activity_end - covered_end)
            covered_end = max(covered_end, activity_end)
        span = covered_end - first["ts"]
        records.append({
            "host_start_us": base_us + forward["ts"],
            "host_end_us": base_us + end,
            "host_ms": forward["dur"] / 1000.0,
            "launch_calls": len(inside),
            "device_activities": len(work),
            "device_start_us": base_us + first["ts"],
            "device_end_us": base_us + covered_end,
            "device_span_ms": span / 1000.0,
            "device_sum_ms": sum(activity["dur"] for activity, _ in work) / 1000.0,
            "busy_ms": busy / 1000.0,
            "idle_ms": (span - busy) / 1000.0,
            "idle_host_wait_ms": host_wait / 1000.0,
            "idle_other_ms": other / 1000.0,
            "device_ms_by_name": {name: duration / 1000.0 for name, duration in
                                  sorted(by_name.items(), key=lambda item: -item[1])},
        })
    return {"base_time_ns": trace["baseTimeNanoseconds"], "graph_forwards": graph_forwards,
            "forwards": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    path = rank0_trace(args.trace_dir)
    with gzip.open(path, "rt") as trace_file:
        trace = json.load(trace_file)
    info = trace["distributedInfo"]
    result = {"trace_file": path.name,
              "distributed_info": {key: info[key] for key in ("backend", "rank", "world_size")},
              **device_timeline(trace)}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("DEVICE_TIMELINE", json.dumps({"trace_file": path.name, "eager_forwards": len(result["forwards"]),
                                         "graph_forwards": result["graph_forwards"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
