"""Compare the device kernel names of two eager (NONE) dispatches of one native V4 trace.

  python dispatch_kernel_diff.py <trace.json.gz> <dispatch index A> <dispatch index B>

Dispatch indices are those printed by peek_sequence.py. Prints, per stream, the
kernel names whose counts differ, and the logical row count of each dispatch's
first `fused_q_kv_rmsnorm` kernel grid when the trace records it.
"""
import gzip
import json
import sys
from collections import Counter


def label(name):
    parts = name.split("|")
    return parts[2] if len(parts) == 3 and parts[0].startswith("frontier_native_") else name


path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
events = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"] if e.get("ph") == "X"]
device = {e["args"]["correlation"]: e for e in events
          if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "correlation" in e.get("args", {})}
notes = sorted((e for e in events if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
dispatches = [e for e in notes if label(e["name"]) in ("model_forward", "model_dispatch_run_fullgraph",
                                                       "model_dispatch_run_pw_graph")]


def kernels(index):
    r = dispatches[index]
    start, end = r["ts"], r["ts"] + r["dur"]
    found = [device[e["args"]["correlation"]] for e in events
             if e.get("cat") in ("cuda_runtime", "cuda_driver") and (e["pid"], e["tid"]) == (r["pid"], r["tid"])
             and start <= e["ts"] <= end and e.get("args", {}).get("correlation") in device]
    return Counter((k["args"].get("stream"), k["name"][:110]) for k in found), found


ca, ka = kernels(a)
cb, kb = kernels(b)
for key in sorted(set(ca) | set(cb), key=lambda k: (k[0], k[1])):
    if ca[key] != cb[key]:
        print(f"s{key[0]} {ca[key]:4d} {cb[key]:4d}  {key[1]}")
for index, found in ((a, ka), (b, kb)):
    norm = next((k for k in found if "fused_q_kv_rmsnorm" in k["name"]), None)
    print(f"dispatch {index}: first fused_q_kv_rmsnorm grid {norm['args'].get('grid') if norm else None}")
