"""Count the eager kernels launched inside PIECEWISE dispatch ranges (the eager breaks).

  python piecewise_breaks.py <trace.json.gz>
"""
import gzip
import json
import sys
from collections import Counter


def label(name):
    parts = name.split("|")
    return parts[2] if len(parts) == 3 and parts[0].startswith("frontier_native_") else name


events = [e for e in json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"] if e.get("ph") == "X"]
device = {}
for e in events:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "correlation" in e.get("args", {}):
        device.setdefault(e["args"]["correlation"], []).append(e)
dispatches = [e for e in events if e.get("cat") == "user_annotation"
              and label(e["name"]) == "model_dispatch_run_pw_graph"]
for r in dispatches:
    start, end = r["ts"], r["ts"] + r["dur"]
    calls = sorted((e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and e["tid"] == r["tid"]
                    and start <= e["ts"] <= end), key=lambda e: e["ts"])
    names, streams, graphs, sequence = Counter(), Counter(), 0, []
    for c in calls:
        if c["name"] in ("cudaGraphLaunch", "cuGraphLaunch"):
            graphs += 1
            sequence.append("G")
            continue
        for d in device.get(c.get("args", {}).get("correlation"), []):
            names[d["name"][:80]] += 1
            streams[d["args"].get("stream")] += 1
            sequence.append(d["name"][:50])
        if c["name"].startswith("cudaEventRecord") or c["name"] == "cudaStreamWaitEvent":
            sequence.append(c["name"])
    print(f"=== PIECEWISE dispatch: {graphs} graph launches, eager activities {sum(names.values())}, "
          f"streams {streams.most_common()}")
    for name, count in names.most_common():
        print(f"{count:5d} {name}")
    print("sequence of the first 3 breaks:")
    shown = 0
    for item in sequence:
        print("   ", item)
        shown += item == "G"
        if shown > 4:
            break
    break
