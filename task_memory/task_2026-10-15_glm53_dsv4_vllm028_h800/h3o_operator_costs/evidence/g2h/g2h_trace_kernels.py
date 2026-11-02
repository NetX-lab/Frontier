"""Summarize G2h RECORD_FUNCTION traces: kernels per annotated scope.

Each kernel is attributed to the innermost GPU user annotation whose range
contains it. Output per trace: scope, calls, then kernel name prefix, grid,
launches per call and median duration (us).

Usage: g2h_trace_kernels.py <trace file or directory> ...
"""

import collections
import json
import statistics
import sys
from pathlib import Path

WIDTH = 110


def files(arguments):
    for argument in arguments:
        path = Path(argument)
        yield from sorted(path.rglob("*.json")) if path.is_dir() else [path]


def summarize(path):
    events = json.load(open(path))["traceEvents"]
    scopes = [e for e in events if e.get("cat") == "gpu_user_annotation"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    calls = collections.Counter(e["name"] for e in scopes)
    grouped = collections.defaultdict(list)
    for kernel in kernels:
        start, end = kernel["ts"], kernel["ts"] + kernel["dur"]
        owners = [s for s in scopes if s["ts"] <= start and end <= s["ts"] + s["dur"]]
        owner = min(owners, key=lambda s: s["dur"])["name"] if owners else "<unowned>"
        key = (owner, kernel["name"][:WIDTH], tuple(kernel.get("args", {}).get("grid", ())))
        grouped[key].append(kernel["dur"])
    print(f"== {path.parent.name}/{path.name} scopes {dict(calls)}")
    for (owner, name, grid), durations in sorted(grouped.items()):
        per_call = len(durations) / calls[owner] if calls.get(owner) else float(len(durations))
        print(f"   {owner:32s} x{per_call:5.2f} grid={list(grid)} med={statistics.median(durations):8.2f} {name}")


for trace in files(sys.argv[1:]):
    summarize(trace)
