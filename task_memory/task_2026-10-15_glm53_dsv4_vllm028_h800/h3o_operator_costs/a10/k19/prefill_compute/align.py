"""Align the EP combine Reduce and the next SP AllGather across the 8 TP ranks of lane 0 (same host clock)."""
import glob
import json
import statistics

ranks = {}
for p in sorted(glob.glob("calls_dp0_pp0_tp*_dcp0_ep*_rank*.json")):
    tp = int(p.split("_tp")[1].split("_")[0])
    ranks[tp] = json.load(open(p))[0]["calls"]
R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL"
G = "ncclDevKernel_AllGather_RING_LL"
V = "ncclDevKernel_AllGatherV_RING_LL"
layers = len(ranks[0][R])
rows = []
for i in range(layers):
    starts = [ranks[t][R][i][0] for t in ranks]
    ends = [ranks[t][R][i][0] + ranks[t][R][i][1] for t in ranks]
    rows.append({"spread_start": max(starts) - min(starts), "spread_end": max(ends) - min(ends),
                 "last_start_to_last_end": max(ends) - max(starts),
                 "first_start_to_last_end": max(ends) - min(starts),
                 "tp0": ranks[0][R][i][1], "last_arriving_tp": max(ranks, key=lambda t: ranks[t][R][i][0]),
                 "last_ending_tp": max(ranks, key=lambda t: ranks[t][R][i][0] + ranks[t][R][i][1])})
for k in ("spread_start", "spread_end", "last_start_to_last_end", "first_start_to_last_end", "tp0"):
    v = [r[k] for r in rows]
    print(f"{k:24s} per layer mean {statistics.mean(v):8.1f} us median {statistics.median(v):8.1f} min {min(v):8.1f} max {max(v):8.1f} sum {sum(v)/1e3:8.3f} ms")
from collections import Counter
print("last arriving tp", Counter(r["last_arriving_tp"] for r in rows))
print("last ending tp", Counter(r["last_ending_tp"] for r in rows))
# AllGatherV (EP dispatch, 3 per layer) and the SP AllGather (44: one per layer plus the final one)
for name in (V, G):
    n = len(ranks[0][name])
    d = [max(ranks[t][name][i][0] + ranks[t][name][i][1] for t in ranks) - max(ranks[t][name][i][0] for t in ranks) for i in range(n)]
    print(f"{name}: calls {n}, last start to last end sum {sum(d)/1e3:.3f} ms; tp0 sum {sum(c[1] for c in ranks[0][name])/1e3:.3f} ms")
