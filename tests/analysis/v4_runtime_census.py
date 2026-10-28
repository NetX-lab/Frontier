#!/usr/bin/env python3
"""Runtime census (H3o stage 0, R1) of native DeepSeek-V4 torch-profiler traces.

  python tests/analysis/v4_runtime_census.py --trace-dir <run>/torch_traces \
      --output <census.json> --summary <census.txt>

Reads the TP-rank-0 trace of every DP lane of each role
(``<role>/dp<N>/dp<N>_pp0_tp0_*.pt.trace.json.gz``), one trace at a time, and
keeps one record per forward. The native runner annotates each forward with a
``frontier_native_metadata|<id>|attention_metadata`` range followed by one
dispatch range whose label gives the graph mode: ``model_dispatch_run_fullgraph``
(FULL), ``model_dispatch_run_pw_graph`` (PIECEWISE) or ``model_forward``
(NONE, eager). The ``gloo:all_reduce`` range before the metadata is the DP
synchronization of ``dp_utils.py`` (recorded on a gloo worker thread of the
same process). A forward has three segments, by the host
time of the launch calls on the forward's thread:

- ``prepare``: from the end of the DP synchronization to the dispatch range
  (input and metadata preparation after the rendezvous);
- ``dispatch``: inside the dispatch range (graph launches, eager segments,
  eager layers);
- ``post``: from the dispatch end to the next forward's DP synchronization
  (logits, sampling and output copies, then the next step's work before its
  DP synchronization, such as KV-block zeroing).

Device work joins the launch calls by correlation id. For each forward the
record gives, per segment, the launch calls by kind, the device activities by
launch kind and stream, and their summed and union device time; over the
forward's device work, the span and the idle gaps, each gap attributed to the
segment of the activity after it and split into the part before the call that
launched that activity returned (the device waited for the host) and the
rest; and the device idle time since the previous forward's last activity,
split the same way after the host began the forward's preparation (idle
before that is time without work or the DP rendezvous). A forward is real when it computes logits
(``frontier_native_ownership|<id>|logits``) and a dummy otherwise; the
``execute_context_<n>(<tokens>)_generation_<n>(<tokens>)`` step range that
contains a real forward gives its step class. Profiler overhead inflates host
times, so host durations and host waits are upper bounds.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
DISPATCH_MODES = {
    "model_dispatch_run_fullgraph": "FULL",
    "model_dispatch_run_pw_graph": "PIECEWISE",
    "model_forward": "NONE",
}
SEGMENTS = ("prepare", "dispatch", "post")
_KERNEL_CATEGORY = "kernel"
_GRID, _BLOCK = "grid", "block"
_LAUNCH_DIMENSIONS = 3
_DISPATCH_SEGMENT = "dispatch"
STEP_RANGE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
LANE_TRACE = re.compile(r"(?P<role>prefill|decode)/dp(?P<dp>\d+)/dp(?P=dp)_pp0_tp0_.*\.pt\.trace\.json\.gz$")


def launch_kind(name: str) -> str:
    if name in ("cudaGraphLaunch", "cuGraphLaunch"):
        return "graph"
    if name.startswith(("cudaLaunch", "cuLaunch")):
        return "kernel"
    if name.startswith(("cudaMemcpy", "cuMemcpy")):
        return "memcpy"
    if name.startswith(("cudaMemset", "cuMemset")):
        return "memset"
    if name.startswith("cudaEventRecord"):
        return "event_record"
    if name == "cudaStreamWaitEvent":
        return "stream_wait"
    if name.endswith("Synchronize"):
        return "synchronize"
    return "other"


def annotation_label(name: str) -> str:
    parts = name.split("|")
    return parts[2] if len(parts) == 3 and parts[0].startswith("frontier_native_") else name


def union_us(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for start, stop in sorted(spans):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def lane_traces(trace_dir: Path) -> list[tuple[str, int, Path]]:
    lanes = []
    for path in sorted(trace_dir.rglob("*.pt.trace.json.gz")):
        match = LANE_TRACE.search(path.relative_to(trace_dir).as_posix())
        if match:
            lanes.append((match["role"], int(match["dp"]), path))
    if not lanes:
        raise SystemExit(f"no <role>/dp<N>/dp<N>_pp0_tp0_* traces under {trace_dir}")
    return sorted(lanes, key=lambda lane: (lane[0], lane[1]))


def _forward_segments(role: str, path: Path):
    with gzip.open(path, "rt") as trace_file:
        events = [event for event in json.load(trace_file)["traceEvents"] if event.get("ph") == "X"]

    annotations = sorted((event for event in events if event.get("cat") == "user_annotation"),
                         key=lambda event: event["ts"])
    metadata = [event for event in annotations if annotation_label(event["name"]) == "attention_metadata"]
    threads = {(event["pid"], event["tid"]) for event in metadata}
    if len(threads) != 1:
        raise ValueError(f"{path}: attention metadata ranges on {len(threads)} threads, expected one")
    thread = threads.pop()
    on_thread = [event for event in annotations if (event["pid"], event["tid"]) == thread]
    dispatches = [event for event in on_thread if annotation_label(event["name"]) in DISPATCH_MODES]
    # Gloo runs the DP all-reduce on its own worker threads of the same process.
    syncs = [event for event in annotations if event["name"] == "gloo:all_reduce" and event["pid"] == thread[0]]
    logits = [event["ts"] for event in on_thread if annotation_label(event["name"]) == "logits"]
    steps = [event for event in on_thread if STEP_RANGE.fullmatch(re.sub(r"\s", "", event["name"]))]

    launches = sorted((event for event in events
                       if event.get("cat") in LAUNCH_CATEGORIES and (event["pid"], event["tid"]) == thread),
                      key=lambda event: event["ts"])
    launch_starts = [event["ts"] for event in launches]
    device_by_correlation: dict[int, list[dict]] = defaultdict(list)
    foreign_device = Counter()
    thread_correlations = {event["args"].get("correlation") for event in launches}
    for event in events:
        if event.get("cat") in DEVICE_CATEGORIES:
            correlation = event["args"].get("correlation")
            if correlation in thread_correlations:
                device_by_correlation[correlation].append(event)
            else:
                foreign_device["activities"] += 1
                foreign_device["sum_us"] += event["dur"]
    eager_streams = Counter(activity["args"].get("stream") for launch in launches
                            if launch_kind(launch["name"]) != "graph"
                            for activity in device_by_correlation.get(launch["args"].get("correlation"), []))
    main_stream = eager_streams.most_common(1)[0][0] if eager_streams else None

    if len(dispatches) != len(metadata):
        raise ValueError(f"{path}: {len(metadata)} metadata ranges but {len(dispatches)} dispatch ranges")
    forwards = []
    for index, (meta, dispatch) in enumerate(zip(metadata, dispatches)):
        dispatch_end = dispatch["ts"] + dispatch["dur"]
        next_meta = metadata[index + 1]["ts"] if index + 1 < len(metadata) else float("inf")
        if not meta["ts"] <= dispatch["ts"] < next_meta:
            raise ValueError(f"{path}: forward {index} dispatch range is not between its metadata ranges")
        floor = forwards[-1][0]["_dispatch_end"] if forwards else float("-inf")
        sync = next((event for event in reversed(syncs) if floor <= event["ts"] < meta["ts"]), None)
        next_sync = next((event for event in syncs if dispatch_end <= event["ts"] < next_meta), None)
        bounds = {
            "prepare": (sync["ts"] + sync["dur"] if sync else meta["ts"], dispatch["ts"]),
            "dispatch": (dispatch["ts"], dispatch_end),
            "post": (dispatch_end, next_sync["ts"] if next_sync else next_meta),
        }
        record = {
            "index": index,
            "mode": DISPATCH_MODES[annotation_label(dispatch["name"])],
            "real": any(dispatch["ts"] <= ts < bounds["post"][1] for ts in logits),
            "sync_host_us": sync["dur"] if sync else None,
            "host_us": {name: (stop - start if stop != float("inf") else None)
                        for name, (start, stop) in bounds.items()},
            "_dispatch_end": dispatch_end,
        }
        step = next((event for event in steps if event["ts"] <= meta["ts"] <= event["ts"] + event["dur"]), None)
        if step is not None:
            numbers = [int(value) for value in STEP_RANGE.fullmatch(re.sub(r"\s", "", step["name"])).groups()]
            record["step"] = dict(zip(("context_requests", "context_tokens",
                                       "generation_requests", "generation_tokens"), numbers))
        record["class"] = step_class(role, record)
        forwards.append((record, bounds))

    return thread, main_stream, eager_streams, foreign_device, launches, launch_starts, device_by_correlation, forwards


@dataclass(frozen=True)
class ForwardKernels:
    index: int
    mode: str
    step_class: str
    kernels: Counter
    scheduled_tokens: int | None = None


@dataclass(frozen=True)
class ForwardKernelEvents:
    record: dict
    segment_bounds: dict[str, tuple[float, float]]
    kernels_by_segment: dict[str, tuple[dict, ...]]


def forward_kernel_events(role: str, path: Path) -> Iterator[ForwardKernelEvents]:
    """Yield each forward's record, host segment bounds and correlated kernels.

    Segment membership follows the host launch, including every kernel attached
    to a graph launch. The iterator holds only this trace's parsed events.
    """
    _, _, _, _, launches, starts, device, forwards = _forward_segments(role, path)
    for record, bounds in forwards:
        kernels = {}
        for segment, (start, stop) in bounds.items():
            inside = launches[bisect.bisect_left(starts, start):bisect.bisect_left(starts, stop)]
            kernels[segment] = tuple(event for launch in inside
                                     for event in device.get(launch["args"].get("correlation"), [])
                                     if event.get("cat") == _KERNEL_CATEGORY)
        yield ForwardKernelEvents(record, bounds, kernels)


def dispatch_kernel_multisets(role: str, path: Path) -> tuple[ForwardKernels, ...]:
    """Return launch-shaped dispatch kernel multisets, sharing census segmentation."""
    result = []
    for forward in forward_kernel_events(role, path):
        record = forward.record
        kernels = Counter()
        for event in forward.kernels_by_segment[_DISPATCH_SEGMENT]:
            dimensions = []
            for field in (_GRID, _BLOCK):
                value = event["args"].get(field)
                if (not isinstance(value, (list, tuple)) or len(value) != _LAUNCH_DIMENSIONS
                        or any(type(item) is not int for item in value)):
                    raise ValueError(f"{path}: kernel {event['name']!r} requires {field} with three ints")
                dimensions.append(tuple(value))
            kernels[(event["name"], *dimensions)] += 1
        step = record.get("step")
        scheduled_tokens = None if step is None else step["context_tokens"] + step["generation_tokens"]
        result.append(ForwardKernels(record["index"], record["mode"], record["class"], kernels, scheduled_tokens))
    return tuple(result)


def forward_census(role: str, path: Path) -> dict:
    thread, main_stream, eager_streams, foreign_device, launches, launch_starts, device_by_correlation, segments_by_forward = _forward_segments(role, path)
    forwards = []
    eager_names: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    previous_device_end = None
    for record, bounds in segments_by_forward:
        work = []
        segments = {}
        for name, (start, stop) in bounds.items():
            inside = launches[bisect.bisect_left(launch_starts, start):bisect.bisect_left(launch_starts, stop)]
            calls = Counter(launch_kind(launch["name"]) for launch in inside)
            activities = Counter()
            streams = Counter()
            spans = []
            for launch in inside:
                kind = launch_kind(launch["name"])
                for activity in device_by_correlation.get(launch["args"].get("correlation"), []):
                    via = "graph" if kind == "graph" else activity["cat"]
                    activities[via] += 1
                    activities[via + "_sum_us"] += activity["dur"]
                    if via != "graph":
                        streams["main" if activity["args"].get("stream") == main_stream else "other"] += 1
                        named = eager_names[f"{record['mode']}|{record['class']}|{name}"][activity["name"]]
                        named[0] += 1
                        named[1] += activity["dur"]
                    spans.append((activity["ts"], activity["ts"] + activity["dur"]))
                    work.append((activity["ts"], activity["ts"] + activity["dur"],
                                 launch["ts"] + launch["dur"], launch["ts"], name, via))
            segments[name] = {
                "calls": dict(calls),
                "activities": dict(activities),
                "eager_streams": dict(streams),
                "device_union_us": union_us(spans),
            }
            if name == "dispatch":
                # Summed over union device time: above 1 where activities of the graph replays,
                # or of the eager launches, overlap on several streams.
                for via in ("graph", "eager"):
                    via_spans = [(start, stop) for start, stop, _, _, segment, kind in work
                                 if segment == name and (kind == "graph") == (via == "graph")]
                    if via_spans:
                        segments[name][via + "_concurrency"] = (sum(stop - start for start, stop in via_spans)
                                                                / union_us(via_spans))
        record["segments"] = segments

        work.sort()
        if work:
            first_start, covered_end, first_launch_end, first_launch_start, *_ = work[0]
            idle = {name: {"host_wait_us": 0.0, "other_us": 0.0} for name in SEGMENTS}
            # The kernel_gap.csv definition: device idle before each activity but the first.
            idle_by_via = defaultdict(lambda: {"host_wait_us": 0.0, "other_us": 0.0, "activities": 0})
            for start, stop, launch_end, _, segment, via in work[1:]:
                kind = "graph" if via == "graph" else "eager"
                idle_by_via[kind]["activities"] += 1
                if start > covered_end:
                    gap = start - covered_end
                    waited = min(gap, max(0.0, launch_end - covered_end))
                    idle[segment]["host_wait_us"] += waited
                    idle[segment]["other_us"] += gap - waited
                    idle_by_via[kind]["host_wait_us"] += waited
                    idle_by_via[kind]["other_us"] += gap - waited
                covered_end = max(covered_end, stop)
            record["device"] = {
                "span_us": covered_end - first_start,
                "busy_us": union_us([(start, stop) for start, stop, *_ in work]),
                "idle_by_next_segment": idle,
                "idle_by_next_activity": dict(idle_by_via),
                "launch_lag_us": first_start - first_launch_start,
            }
            if previous_device_end is not None:
                # Idle before the host began this forward's preparation is time without work
                # or the DP rendezvous; after it, the device waits for the preparation.
                gap = max(0.0, first_start - previous_device_end)
                host_start = bounds["prepare"][0]
                before = min(gap, max(0.0, host_start - previous_device_end))
                waited = min(gap - before, max(0.0, first_launch_end - max(previous_device_end, host_start)))
                record["device"]["since_previous_forward"] = {
                    "before_host_us": before, "host_wait_us": waited, "other_us": gap - before - waited}
            previous_device_end = covered_end if previous_device_end is None else max(previous_device_end,
                                                                                      covered_end)
        forwards.append(record)
    for record in forwards:
        del record["_dispatch_end"]
    return {
        "trace_file": path.name,
        "thread": list(thread),
        "main_stream": main_stream,
        "eager_streams": {str(stream): count for stream, count in eager_streams.most_common()},
        "foreign_thread_device": dict(foreign_device),
        "forwards": forwards,
        "eager_names": {group: dict(names) for group, names in eager_names.items()},
    }


def step_class(role: str, record: dict) -> str:
    if not record["real"]:
        return "dummy"
    step = record.get("step")
    if step is None:
        return "real_without_step_range"
    if role == "prefill":
        return "prefill"
    if step["context_requests"] == 0:
        return "decode"
    return "first_tail_with_decodes" if step["generation_requests"] else "first_tail_alone"


def summarize(lanes: list[dict]) -> list[str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for lane in lanes:
        for record in lane["forwards"]:
            groups[(lane["role"], lane["dp"], record["mode"], record["class"])].append(record)

    def stat(records, getter):
        values = [value for value in (getter(record) for record in records) if value is not None]
        if not values:
            return "-"
        return f"{median(values):.0f}/{percentile(values, 0.9):.0f}"

    def device(record, *keys):
        value = record.get("device")
        for key in keys:
            if value is None:
                return None
            value = value.get(key)
        return value

    lines = ["Medians/p90 in microseconds per forward; counts are medians/p90 per forward.",
             "role dp mode class n | host: sync prepare dispatch post | device: span busy, "
             "busy by segment prepare dispatch post, lag | "
             "idle wait/other by next segment: prepare dispatch post | since previous forward before-host/wait/other | "
             "calls: graph kernel memcpy event_record stream_wait sync | eager activities main/other | "
             "graph activities, dispatch concurrency graph/eager | "
             "device-side idle per activity (kernel_gap.csv definition) graph/eager"]
    for key in sorted(groups, key=lambda key: (key[0], key[1], key[2], key[3])):
        records = groups[key]
        segment = lambda record, name, part, field: record["segments"][name][part].get(field, 0)
        calls = " ".join(stat(records, lambda r, kind=kind: sum(segment(r, name, "calls", kind) for name in SEGMENTS))
                         for kind in ("graph", "kernel", "memcpy", "event_record", "stream_wait", "synchronize"))
        idle = " ".join(
            stat(records, lambda r, name=name: device(r, "idle_by_next_segment", name, "host_wait_us")) + "+"
            + stat(records, lambda r, name=name: device(r, "idle_by_next_segment", name, "other_us"))
            for name in SEGMENTS)
        eager = (stat(records, lambda r: sum(segment(r, name, "eager_streams", "main") for name in SEGMENTS)) + " "
                 + stat(records, lambda r: sum(segment(r, name, "eager_streams", "other") for name in SEGMENTS)))
        concurrency = " ".join(
            f"{median(values):.2f}" if (values := [r["segments"]["dispatch"][via + "_concurrency"] for r in records
                                                   if via + "_concurrency" in r["segments"]["dispatch"]]) else "-"
            for via in ("graph", "eager"))
        graph = stat(records, lambda r: segment(r, "dispatch", "activities", "graph")) + " " + concurrency
        per_activity = []
        for via in ("graph", "eager"):
            parts = [r["device"]["idle_by_next_activity"][via] for r in records
                     if "device" in r and via in r["device"]["idle_by_next_activity"]]
            transitions = sum(part["activities"] for part in parts)
            per_activity.append(f"{sum(part['other_us'] for part in parts) / transitions:.2f}" if transitions else "-")
        lines.append(
            f"{key[0]} dp{key[1]} {key[2]} {key[3]} n={len(records)} | "
            f"host: {stat(records, lambda r: r['sync_host_us'])} "
            + " ".join(stat(records, lambda r, name=name: r["host_us"][name]) for name in SEGMENTS)
            + f" | device: {stat(records, lambda r: device(r, 'span_us'))} {stat(records, lambda r: device(r, 'busy_us'))} "
            + " ".join(stat(records, lambda r, name=name: r["segments"][name]["device_union_us"]) for name in SEGMENTS)
            + " "
            f"{stat(records, lambda r: device(r, 'launch_lag_us'))} | idle: {idle} | since previous: "
            f"{stat(records, lambda r: device(r, 'since_previous_forward', 'before_host_us'))}+"
            f"{stat(records, lambda r: device(r, 'since_previous_forward', 'host_wait_us'))}+"
            f"{stat(records, lambda r: device(r, 'since_previous_forward', 'other_us'))} | calls: {calls} | "
            f"eager: {eager} | graph: {graph} | gap: {'/'.join(per_activity)}")

    lines += ["", "Device idle while the host had not yet returned the next launch, as a share of the device span "
              "plus the idle time since the previous forward after the host began the forward's preparation; "
              "forwards whose share exceeds 1%:"]
    shares: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for (role, _, mode, kind), records in groups.items():
        for record in records:
            device_record = record.get("device")
            if device_record is None:
                continue
            since = device_record.get("since_previous_forward",
                                      {"before_host_us": 0.0, "host_wait_us": 0.0, "other_us": 0.0})
            waited = since["host_wait_us"] + sum(part["host_wait_us"]
                                                 for part in device_record["idle_by_next_segment"].values())
            shares[(role, mode, "dummy" if kind == "dummy" else "real")].append(
                (waited, device_record["span_us"] + since["host_wait_us"] + since["other_us"]))
    for key in sorted(shares):
        pairs = shares[key]
        lines.append(f"{key[0]} {key[1]} {key[2]} n={len(pairs)}: "
                     f"{100 * sum(w for w, _ in pairs) / sum(t for _, t in pairs):.2f}% overall, "
                     f"{sum(w > 0.01 * t for w, t in pairs)} forwards above 1%")

    lines += ["", "Eager device activities by name, all lanes: count and device time per forward (us), top 12 by time:"]
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    counts = Counter()
    for lane in lanes:
        for record in lane["forwards"]:
            counts[(lane["role"], record["mode"], record["class"])] += 1
        for group, names in lane["eager_names"].items():
            for name, (count, duration) in names.items():
                totals[f"{lane['role']}|{group}"][name][0] += count
                totals[f"{lane['role']}|{group}"][name][1] += duration
    for group in sorted(totals):
        role, mode, kind, segment = group.split("|")
        forwards_in_group = counts[(role, mode, kind)]
        lines.append(f"{role} {mode} {kind} {segment} (forwards {forwards_in_group}):")
        for name, (count, duration) in sorted(totals[group].items(), key=lambda item: -item[1][1])[:12]:
            lines.append(f"    {count / forwards_in_group:7.1f} {duration / forwards_in_group:9.1f}  {name[:110]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    lanes = []
    for role, dp, path in lane_traces(args.trace_dir):
        census = forward_census(role, path)
        lanes.append({"role": role, "dp": dp, **census})
        print("LANE", role, f"dp{dp}", path.name, "forwards", len(census["forwards"]),
              dict(Counter((record["mode"], record["class"]) for record in census["forwards"])),
              flush=True)
    args.output.write_text(json.dumps({"trace_dir": str(args.trace_dir), "lanes": lanes}, indent=1) + "\n")
    args.summary.write_text("\n".join([str(args.trace_dir), *summarize(lanes)]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
