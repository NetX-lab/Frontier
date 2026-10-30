"""Absolute start and per-layer EP combine Reduce durations of every NONE forward of one trace.

    python peers.py <trace.json.gz> <out.json>
"""
import gzip
import json
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events

R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL("
path, out = Path(sys.argv[1]), Path(sys.argv[2])
with gzip.open(path, "rt") as f:
    base = json.load(f).get("baseTimeNanoseconds")
result = []
for fw in forward_kernel_events("prefill", path):
    r = fw.record
    if r["mode"] != "NONE":
        continue
    ks = sorted(fw.kernels_by_segment["dispatch"], key=lambda k: k["ts"])
    red = [(k["ts"], k["dur"]) for k in ks if k["name"].startswith(R)]
    marlin = [k["dur"] for k in ks if k["name"].startswith("void marlin_moe_wna16")]
    result.append({"index": r["index"], "class": r["class"], "real": r["real"], "first_ts": ks[0]["ts"],
                   "abs_first_s": (base + ks[0]["ts"] * 1e3) / 1e9 if base else None,
                   "reduce": red, "marlin_sum_us": sum(marlin)})
out.write_text(json.dumps({"base": base, "forwards": result}))
for f in result:
    d = [x for _, x in f["reduce"]]
    print(f["index"], f["class"], f["real"], f"abs {f['abs_first_s']:.6f}" if f["abs_first_s"] else "", "reduce n", len(d),
          "sum ms", round(sum(d) / 1e3, 3), "min us", round(min(d), 1) if d else None, "marlin ms", round(f["marlin_sum_us"] / 1e3, 3))
