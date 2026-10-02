#!/usr/bin/env python3
"""Reduce the torch-profiler traces of a vLLM run to one record per forward.

  python forward_device_timeline.py --trace-dir <dir> --output <device_timeline.json>

The ``device_timeline`` mode of vllm_replay.py makes every process of the server
write a ``*.pt.trace.json.gz`` file to one directory; the trace header's
``distributedInfo`` names each worker's global rank (the API server's CPU-only
trace has none). The records come from global rank 0, the TP rank whose steps
the CPU probe records. Each ``Forward`` range of the model runner becomes one
record, under ``forwards`` for an eager forward and under ``graph_forwards``
for a CUDA graph replay (its thread called ``cudaGraphLaunch``, and the graph's
kernels carry that call's correlation id):

- the host range of the forward call;
- its device work: the kernels, copies and sets that the CUDA runtime and
  driver calls made on the forward's thread inside that range launched,
  joined by correlation id;
- the device span, from the first device start to the last device end; the
  busy time, the union of the device intervals; the summed device time; and
  the idle time, the span minus the busy time;
- the idle time split at each gap into the part before the call that launched
  the next device work returned (the device waited for the host) and the rest;
- device time by kernel, copy or set name;
- the duration of each TP all-reduce kernel in device order (``all_reduce_us``)
  and, per all-reduce, the shortest duration across all ranks
  (``all_reduce_min_us``). Every rank leaves an all-reduce kernel together, so
  the shortest kernel is the one that waited least for its peers, and a rank's
  excess over it is time spent waiting for slower ranks. A rank's forward is
  the one of the same kind whose device end lies nearest rank 0's, within half
  of rank 0's device span, with as many all-reduces; without one on every rank,
  ``all_reduce_min_us`` is null.

Timestamps are microseconds on the host's real-time clock: kineto writes each
``ts`` relative to the header's ``baseTimeNanoseconds``. All ranks of one run
share the host clock. Runs on the standard library only, inside the vLLM image.
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
# Name substrings of vLLM's TP all-reduce kernels: pynccl and the custom all-reduce
# (cuda_communicator.py all_reduce), whose kernels are named "void vllm::cross_device_reduce_1stage<...>(...)".
ALL_REDUCE_KERNELS = ("ncclDevKernel_AllReduce", "vllm::cross_device_reduce_")


def trace_header(path: Path) -> dict:
    """The JSON members kineto writes before ``traceEvents``."""
    lines = []
    with gzip.open(path, "rt") as trace:
        for line in trace:
            if line.lstrip().startswith('"traceEvents"'):
                break
            lines.append(line)
    return json.loads("".join(lines).rstrip().rstrip(",") + "}")


def rank_traces(trace_dir: Path) -> list[Path]:
    """The worker traces ordered by global rank, one per rank of the world."""
    infos = {path: trace_header(path).get("distributedInfo")
             for path in sorted(trace_dir.glob("*.pt.trace.json.gz"))}
    workers = {path: info for path, info in infos.items() if info is not None}
    ranks = sorted(info["rank"] for info in workers.values())
    world_sizes = {info["world_size"] for info in workers.values()}
    if len(world_sizes) != 1 or ranks != list(range(world_sizes.pop())):
        found = {path.name: info["rank"] if info else None for path, info in infos.items()}
        raise SystemExit(f"expected one trace per rank in {trace_dir}, found ranks {found}")
    return sorted(workers, key=lambda path: workers[path]["rank"])


def load_trace(path: Path) -> dict:
    with gzip.open(path, "rt") as trace_file:
        return json.load(trace_file)


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

    records = {"forwards": [], "graph_forwards": []}
    for forward in forwards:
        thread = (forward["pid"], forward["tid"])
        launches = launches_by_thread.get(thread, [])
        starts = starts_by_thread.get(thread, [])
        end = forward["ts"] + forward["dur"]
        inside = launches[bisect.bisect_left(starts, forward["ts"]):bisect.bisect_right(starts, end)]
        graph = any(launch["name"].startswith(GRAPH_LAUNCHES) for launch in inside)
        work = sorted(((activity, launch["ts"] + launch["dur"]) for launch in inside
                       for activity in device_by_correlation.get(launch["args"]["correlation"], [])),
                      key=lambda pair: pair[0]["ts"])
        if not work:
            raise ValueError(f"Forward at ts {forward['ts']} launched no device work")
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
        records["graph_forwards" if graph else "forwards"].append({
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
            "all_reduce_us": [activity["dur"] for activity, _ in work
                              if any(kernel in activity["name"] for kernel in ALL_REDUCE_KERNELS)],
        })
    return {"base_time_ns": trace["baseTimeNanoseconds"], **records}


def matching_forward(record: dict, peer_records: list[dict], peer_ends: list[float]) -> dict | None:
    """The peer forward whose device end lies nearest the record's, if it is the same forward."""
    index = bisect.bisect_left(peer_ends, record["device_end_us"])
    candidates = peer_records[max(index - 1, 0):index + 1]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda peer: abs(peer["device_end_us"] - record["device_end_us"]))
    same_forward = (abs(nearest["device_end_us"] - record["device_end_us"]) <= 0.5 * record["device_span_ms"] * 1000
                    and len(nearest["all_reduce_us"]) == len(record["all_reduce_us"]))
    return nearest if same_forward else None


def add_all_reduce_minimum(timeline: dict, peers: list[dict]) -> None:
    """Set each rank-0 record's per-all-reduce minimum over its matching forward on every rank."""
    for kind in ("forwards", "graph_forwards"):
        peer_records = [sorted(peer[kind], key=lambda record: record["device_end_us"]) for peer in peers]
        peer_ends = [[record["device_end_us"] for record in records] for records in peer_records]
        for record in timeline[kind]:
            durations = [record["all_reduce_us"]]
            for records, ends in zip(peer_records, peer_ends):
                match = matching_forward(record, records, ends)
                if match is None:
                    break
                durations.append(match["all_reduce_us"])
            record["all_reduce_min_us"] = ([min(column) for column in zip(*durations)]
                                           if len(durations) == len(peers) + 1 else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rank0, *peers = rank_traces(args.trace_dir)
    trace = load_trace(rank0)
    info = trace["distributedInfo"]
    timeline = device_timeline(trace)
    del trace
    add_all_reduce_minimum(timeline, [device_timeline(load_trace(path)) for path in peers])
    result = {"trace_file": rank0.name,
              "distributed_info": {key: info[key] for key in ("backend", "rank", "world_size")},
              **timeline}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("DEVICE_TIMELINE", json.dumps({
        "trace_file": rank0.name, "eager_forwards": len(result["forwards"]),
        "graph_forwards": len(result["graph_forwards"]),
        "all_reduce_minimum_matched": sum(record["all_reduce_min_us"] is not None
                                          for kind in ("forwards", "graph_forwards") for record in result[kind])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
