"""Align forward <index> across the 4 prefill lanes (TP rank 0 each, different hosts).

The host clock offset of each lane is the median difference of the EP dispatch AllGatherV end
times against lane 0; an all-gather completes on every rank at about the same time.

    python peer_align.py <index> <trace dp0> <trace dp1> <trace dp2> <trace dp3>
"""
import gzip
import json
import statistics
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events

index = int(sys.argv[1])
R, V = "ncclDevKernel_Reduce_Sum_bf16_RING_LL(", "ncclDevKernel_AllGatherV_RING_LL("
lanes = []
for p in sys.argv[2:]:
    path = Path(p)
    with gzip.open(path, "rt") as f:
        base = json.load(f)["baseTimeNanoseconds"]
    for fw in forward_kernel_events("prefill", path):
        if fw.record["index"] == index:
            ks = sorted(fw.kernels_by_segment["dispatch"], key=lambda k: k["ts"])
            to_abs = lambda ts: base / 1e3 + ts  # microseconds
            lanes.append({
                "class": fw.record["class"],
                "reduce": [(to_abs(k["ts"]), to_abs(k["ts"]) + k["dur"]) for k in ks if k["name"].startswith(R)],
                "agv": [(to_abs(k["ts"]), to_abs(k["ts"]) + k["dur"]) for k in ks if k["name"].startswith(V)],
                "first": to_abs(ks[0]["ts"]), "last": max(to_abs(k["ts"]) + k["dur"] for k in ks)})
            break
offsets = [statistics.median(b[1] - a[1] for a, b in zip(lanes[0]["agv"], lane["agv"])) for lane in lanes]
spread = [statistics.pstdev([(b[1] - a[1]) - off for a, b in zip(lanes[0]["agv"], lane["agv"])]) for lane, off in zip(lanes, offsets)]
print("classes", [l["class"] for l in lanes], "offsets us", [round(o, 1) for o in offsets], "residual sd us", [round(s, 1) for s in spread])
layers = len(lanes[0]["reduce"])
late = []
for i in range(layers):
    starts = [lane["reduce"][i][0] - off for lane, off in zip(lanes, offsets)]
    ends = [lane["reduce"][i][1] - off for lane, off in zip(lanes, offsets)]
    late.append((starts.index(max(starts)), max(starts) - starts[0], max(ends) - max(starts), ends[0] - max(starts),
                 [round(s - starts[0], 1) for s in starts]))
from collections import Counter
print("last arriving lane per layer", Counter(l[0] for l in late))
print("lane 0 wait for the last arrival: sum ms", round(sum(l[1] for l in late) / 1e3, 3), "per layer median us", round(statistics.median(l[1] for l in late), 1))
print("last arrival to lane 0 end: sum ms", round(sum(l[3] for l in late) / 1e3, 3), "per layer median us", round(statistics.median(l[3] for l in late), 1))
for i in (0, 1, 10, 20, 42):
    print("layer", i, "arrival relative to lane 0 (us)", late[i][4])
print("forward first kernel relative to lane 0 (us)", [round(l["first"] - off - lanes[0]["first"], 1) for l, off in zip(lanes, offsets)],
      "last end", [round(l["last"] - off - lanes[0]["last"], 1) for l, off in zip(lanes, offsets)])
