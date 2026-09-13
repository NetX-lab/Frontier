"""Build a rank-aware CUDA activity breakdown from an Nsight SQLite export.

The normal ``cuda_gpu_trace`` CSV does not preserve the process/globalPid
association needed to select DP0 when multiple data-parallel workers share a
physical device label.  This parser uses CUPTI's ``globalPid`` and the CUDA
context table to select the four processes that actually entered the Issue26
first-formal capture.  It reports inclusive kernel durations and per-category
interval unions; unions are the pure device-time accounting used by the
Frontier comparison and must not be summed across ranks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COMMUNICATION_RE = re.compile(
    r"(?:nccl|all.?reduce|all.?to.?all|alltoall|broadcast|reduce.?scatter|"
    r"allgather|cross.?device.?reduce|\bsend\b|\brecv\b|\bscatter\b|"
    r"\bgather\b|collective)", re.IGNORECASE)
MEMORY_RE = re.compile(
    r"(?:memcpy|memset|copy_|copykernel|\bcast\b|\bconvert\b|"
    r"\btranspose\b|index_select)", re.IGNORECASE)


def union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def finite_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if number < 0:
        raise ValueError(f"negative {field}: {number}")
    return number


def classify(name: str) -> str:
    if MEMORY_RE.search(name):
        return "memory"
    if COMMUNICATION_RE.search(name):
        return "communication"
    return "compute"


def read_api_windows(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    starts: dict[int, int] = {}
    stops: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("Name") or "").strip()
            if name not in {"cuProfilerStart", "cudaProfilerStop"}:
                continue
            try:
                pid = int(row["Pid"])
                timestamp = finite_int(row["Start (ns)"], "API timestamp")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid API row: {row}") from exc
            target = starts if name == "cuProfilerStart" else stops
            target.setdefault(pid, timestamp)
    if not starts:
        raise ValueError("no cuProfilerStart rows found")
    return starts, stops


def kernel_name(strings: dict[int, str], value: Any) -> str:
    try:
        return strings[int(value)]
    except (KeyError, TypeError, ValueError):
        return f"<string:{value}>"


def analyze(sqlite_path: Path, api_trace: Path, output: Path) -> dict[str, Any]:
    starts, stops = read_api_windows(api_trace)
    db = sqlite3.connect(sqlite_path)
    try:
        strings = {
            int(row[0]): str(row[1])
            for row in db.execute("SELECT id, value FROM StringIds")
        }
        processes = {
            int(row[1]): {"global_pid": int(row[0]), "name": str(row[2])}
            for row in db.execute("SELECT globalPid, pid, name FROM PROCESSES")
        }
        contexts = list(db.execute(
            "SELECT deviceId, processId FROM TARGET_INFO_CUDA_CONTEXT_INFO"))
        device_pids: dict[int, list[int]] = defaultdict(list)
        for device_id, pid in contexts:
            if int(pid) in starts:
                device_pids[int(device_id)].append(int(pid))
        if not device_pids:
            raise ValueError("no CUDA contexts match profiler-start PIDs")

        # The first rank that calls cudaProfilerStop ends the Nsight capture for
        # all processes.  Use that earliest stop as a common fallback for a
        # rank whose stop row is absent from the API trace.
        common_stop = min(stops.values()) if stops else None
        if common_stop is None:
            raise ValueError("no cudaProfilerStop rows found")

        devices: list[dict[str, Any]] = []
        for device_id in sorted(device_pids):
            # One started process per device corresponds to DP0.  The second
            # process on each device is the idle DP1 worker.
            pid = min(device_pids[device_id], key=lambda item: starts[item])
            info = processes.get(pid, {"global_pid": None, "name": "<unknown>"})
            global_pid = info["global_pid"]
            window_start = starts[pid]
            window_stop = stops.get(pid, common_stop)
            intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
            inclusive: dict[str, int] = defaultdict(int)
            counts: dict[str, int] = defaultdict(int)
            operations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

            if global_pid is None:
                raise ValueError(f"missing globalPid for process {pid}")
            kernel_query = (
                "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid = ? AND start < ? AND end > ?")
            for start, end, name_id in db.execute(
                    kernel_query, (global_pid, window_stop, window_start)):
                start = max(finite_int(start, "kernel start"), window_start)
                end = min(finite_int(end, "kernel end"), window_stop)
                if start >= end:
                    continue
                name = kernel_name(strings, name_id)
                category = classify(name)
                intervals[category].append((start, end))
                inclusive[category] += end - start
                counts[category] += 1
                operations[category][name] += end - start

            # CUDA memcpy activities have no operation name but are memory
            # device work and must be included in the memory category.
            memcpy_query = (
                "SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE globalPid = ? AND start < ? AND end > ?")
            for start, end, copy_kind in db.execute(
                    memcpy_query, (global_pid, window_stop, window_start)):
                start = max(finite_int(start, "memcpy start"), window_start)
                end = min(finite_int(end, "memcpy end"), window_stop)
                if start >= end:
                    continue
                name = f"[CUDA memcpy kind={copy_kind}]"
                intervals["memory"].append((start, end))
                inclusive["memory"] += end - start
                counts["memory"] += 1
                operations["memory"][name] += end - start

            all_intervals = [item for values in intervals.values() for item in values]
            envelope_start = min((item[0] for item in all_intervals), default=None)
            envelope_end = max((item[1] for item in all_intervals), default=None)
            envelope = (envelope_end - envelope_start
                        if envelope_start is not None and envelope_end is not None
                        else 0)
            unions = {category: union_duration(intervals.get(category, []))
                      for category in ("compute", "communication", "memory")}
            all_union = union_duration(all_intervals)
            category_union_total = sum(unions.values())
            devices.append({
                "device_id": device_id,
                "process_id": pid,
                "global_pid": global_pid,
                "process_name": info["name"],
                "window": {
                    "start_ns": window_start,
                    "stop_ns": window_stop,
                    "duration_ns": window_stop - window_start,
                    "stop_source": "process" if pid in stops else "earliest_process_stop",
                },
                "inclusive_duration_ns_by_category": {
                    category: inclusive.get(category, 0)
                    for category in ("compute", "communication", "memory")
                },
                "category_union_ns": unions,
                "event_count_by_category": {
                    category: counts.get(category, 0)
                    for category in ("compute", "communication", "memory")
                },
                "all_kernel_or_activity_union_ns": all_union,
                "activity_envelope_ns": envelope,
                "idle_or_non_activity_ns": max(0, envelope - all_union),
                "category_overlap_ns": max(0, category_union_total - all_union),
                "top_operations": {
                    category: [
                        {"name": name, "inclusive_duration_ns": duration}
                        for name, duration in sorted(
                            values.items(), key=lambda item: (-item[1], item[0]))[:20]
                    ]
                    for category, values in operations.items()
                },
            })
    finally:
        db.close()

    report = {
        "status": "PASS",
        "sqlite": str(sqlite_path),
        "api_trace": str(api_trace),
        "capture": {
            "started_pids": sorted(starts),
            "stopped_pids": sorted(stops),
            "missing_stop_pids": sorted(set(starts) - set(stops)),
            "common_stop_ns": common_stop,
        },
        "devices": devices,
        "limits": [
            "Per-device process selection uses CUDA context globalPid and profiler-start membership.",
            "A missing cudaProfilerStop uses the earliest observed process stop as a bounded fallback.",
            "Category unions are per-device pure interval unions and cannot be summed across ranks.",
            "Kernel-name communication/memory classification is heuristic; inclusive durations are reported separately.",
            "Nsight capture remains diagnostic and may perturb the accepted clean 78--79 ms span.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--api-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.api_trace, args.output)
    for device in result["devices"]:
        unions = device["category_union_ns"]
        print(
            f"device={device['device_id']} pid={device['process_id']} "
            f"window_ms={device['window']['duration_ns'] / 1e6:.6f} "
            f"compute_ms={unions['compute'] / 1e6:.6f} "
            f"communication_ms={unions['communication'] / 1e6:.6f} "
            f"memory_ms={unions['memory'] / 1e6:.6f} "
            f"idle_ms={device['idle_or_non_activity_ns'] / 1e6:.6f}")


if __name__ == "__main__":
    main()
