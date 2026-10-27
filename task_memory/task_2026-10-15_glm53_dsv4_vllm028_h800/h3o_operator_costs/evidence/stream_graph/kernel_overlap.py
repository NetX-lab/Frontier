"""Measure how much device time of a kernel set overlaps other kernels, per dispatch mode.

  python kernel_overlap.py <trace.json.gz> <kernel name substrings>...

For every FULL / PIECEWISE dispatch range of the trace's forward thread: the
summed device time of the kernels whose names contain one of the substrings,
and the part of it during which a kernel outside that set also ran; and the
median device span of a forward (first kernel start to last kernel end).
"""
import gzip
import json
import sys
from collections import defaultdict

MODES = {"model_dispatch_run_fullgraph": "FULL", "model_dispatch_run_pw_graph": "PIECEWISE",
         "model_forward": "NONE"}


def label(name):
    parts = name.split("|")
    return parts[2] if len(parts) == 3 and parts[0].startswith("frontier_native_") else name


path, patterns = sys.argv[1], sys.argv[2:]
events = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"] if e.get("ph") == "X"]
launches = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
dispatches = [e for e in events if e.get("cat") == "user_annotation" and label(e["name"]) in MODES]
totals = defaultdict(lambda: [0, 0.0, 0.0, 0])
spans = defaultdict(list)
for r in dispatches:
    start, end = r["ts"], r["ts"] + r["dur"]
    owned = []
    for k in kernels:
        launch = launches.get(k["args"].get("correlation"))
        if launch is not None and launch["tid"] == r["tid"] and start <= launch["ts"] <= end:
            owned.append(k)
    shared = [k for k in owned if any(p in k["name"] for p in patterns)]
    others = [(k["ts"], k["ts"] + k["dur"]) for k in owned if not any(p in k["name"] for p in patterns)]
    others.sort()
    busy, overlap = 0.0, 0.0
    for k in shared:
        a, b = k["ts"], k["ts"] + k["dur"]
        busy += b - a
        covered, cursor = 0.0, a
        for s, t in others:
            if t <= cursor:
                continue
            if s >= b:
                break
            lo, hi = max(s, cursor), min(t, b)
            if hi > lo:
                covered += hi - lo
                cursor = hi
        overlap += covered
    if owned:
        spans[MODES[label(r["name"])]].append(max(k["ts"] + k["dur"] for k in owned) - min(k["ts"] for k in owned))
    entry = totals[MODES[label(r["name"])]]
    entry[0] += 1
    entry[1] += busy
    entry[2] += overlap
    entry[3] += len(shared)
for mode, (n, busy, overlap, count) in sorted(totals.items()):
    print(f"{mode}: forwards {n}, matching kernels {count / n:.1f} per forward, "
          f"busy {busy / n:.1f} us per forward, overlapped {overlap / n:.1f} us "
          f"({100 * overlap / busy if busy else 0:.1f}%), "
          f"median forward device span {sorted(spans[mode])[len(spans[mode]) // 2]:.1f} us")
