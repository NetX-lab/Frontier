"""Print the ordered host launch sequence of eager (NONE) forwards of one native V4 trace.

  python peek_sequence.py <trace.json.gz> <limit>

For each eager dispatch range: real or dummy, the device stream counts, the
runtime-call counts, then the first <limit> launch calls in host order, each
kernel with its device stream (M = the forward's most used stream).
"""
import bisect
import gzip
import json
import sys
from collections import Counter


def label(name):
    parts = name.split("|")
    return parts[2] if len(parts) == 3 and parts[0].startswith("frontier_native_") else name


path, limit = sys.argv[1], int(sys.argv[2])
events = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"] if e.get("ph") == "X"]
device = {e["args"]["correlation"]: e for e in events
          if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "correlation" in e.get("args", {})}
notes = sorted((e for e in events if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
dispatches = [e for e in notes if label(e["name"]) in ("model_forward", "model_dispatch_run_fullgraph",
                                                       "model_dispatch_run_pw_graph")]
logits = [e["ts"] for e in notes if label(e["name"]) == "logits"]
for index, r in enumerate(dispatches):
    if label(r["name"]) != "model_forward":
        continue
    start, end = r["ts"], r["ts"] + r["dur"]
    nxt = dispatches[index + 1]["ts"] if index + 1 < len(dispatches) else float("inf")
    i = bisect.bisect_left(logits, end)
    real = i < len(logits) and logits[i] < nxt
    calls = sorted((e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
                    and (e["pid"], e["tid"]) == (r["pid"], r["tid"]) and start <= e["ts"] <= end),
                   key=lambda e: e["ts"])
    streams = Counter(device[c["args"]["correlation"]]["args"].get("stream") for c in calls
                      if c.get("args", {}).get("correlation") in device)
    main = streams.most_common(1)[0][0]
    print(f"=== dispatch {index} {'real' if real else 'dummy'} streams {streams.most_common()}")
    print("calls", Counter(c["name"] for c in calls).most_common(10))
    for c in calls[:limit]:
        d = device.get(c.get("args", {}).get("correlation"))
        if d is not None:
            s = d["args"].get("stream")
            print(f"K s{'M' if s == main else s} {d['name'][:100]}")
        elif c["name"].startswith("cudaEventRecord") or c["name"] == "cudaStreamWaitEvent":
            print("  ", c["name"])
