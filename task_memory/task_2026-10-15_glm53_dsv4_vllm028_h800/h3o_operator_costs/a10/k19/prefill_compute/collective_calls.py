"""Per-call device durations of the EP collectives in the native real prefill forward.

    python collective_calls.py <trace.json.gz> <out.json>
"""
import json
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events

NAMES = ("ncclDevKernel_Reduce_Sum_bf16_RING_LL", "ncclDevKernel_AllGatherV_RING_LL",
         "ncclDevKernel_AllGather_RING_LL", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL")
path, out = Path(sys.argv[1]), Path(sys.argv[2])
result = []
for fw in forward_kernel_events("prefill", path):
    r = fw.record
    if not (r["real"] and r["mode"] == "NONE"):
        continue
    ks = sorted(fw.kernels_by_segment["dispatch"], key=lambda k: k["ts"])
    calls = {n: [(k["ts"], k["dur"], k["args"].get("grid")) for k in ks if k["name"].startswith(n + "(")] for n in NAMES}
    spans = [(k["ts"], k["ts"] + k["dur"]) for k in ks]
    result.append({"index": r["index"], "step": r.get("step"), "first": spans[0][0],
                   "last": max(e for _, e in spans), "calls": calls})
out.write_text(json.dumps(result))
for f in result:
    print(f["index"], f["step"], {n: (len(c), round(sum(d for _, d, _ in c) / 1e3, 3)) for n, c in f["calls"].items()},
          "span", round((f["last"] - f["first"]) / 1e3, 3))
