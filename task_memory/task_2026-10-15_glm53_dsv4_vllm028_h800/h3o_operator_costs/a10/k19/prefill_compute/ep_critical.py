"""Per-layer critical-path time of the EP dispatch and combine in one native prefill forward.

For each layer, a collective's time is the latest end over all EP ranks less the latest
start (the last arrival). Ranks on one host share a clock; the lanes' host clocks are
aligned by the median end time of the dispatch AllGatherV kernels against lane 0.

    python ep_critical.py <index> <trace dir of lane 0> <trace dir of lane 1> [...]
"""
import gzip
import json
import statistics
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events

index = int(sys.argv[1])
V = "ncclDevKernel_AllGatherV_RING_LL("
R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL("
S = "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL("


def rank_calls(path):
    with gzip.open(path, "rt") as f:
        base = json.load(f)["baseTimeNanoseconds"] / 1e3
    for fw in forward_kernel_events("prefill", path):
        if fw.record["index"] == index:
            ks = sorted(fw.kernels_by_segment["dispatch"], key=lambda k: k["ts"])
            pick = lambda n: [(base + k["ts"], base + k["ts"] + k["dur"]) for k in ks if k["name"].startswith(n)]
            return fw.record["class"], pick(V), pick(R), pick(S)
    raise SystemExit(f"forward {index} not in {path}")


cache = Path(f"ep_critical_calls_f{index}_" + "_".join(Path(d).name for d in sys.argv[2:]) + ".json")
if cache.exists():
    lanes = json.loads(cache.read_text())
else:
    lanes = [[rank_calls(p) for p in sorted(Path(d).glob("*.pt.trace.json.gz"))] for d in sys.argv[2:]]
    cache.write_text(json.dumps(lanes))
for d, ranks in zip(sys.argv[2:], lanes):
    print(d, "classes", sorted({r[0] for r in ranks}), "ranks", len(ranks))
ref = lanes[0][0][1]
offsets = [statistics.median(b[1] - a[1] for a, b in zip(ref, lane[0][1])) for lane in lanes]
ranks = [(li, r) for li, lane in enumerate(lanes) for r in lane]
shift = lambda li, spans: [(s - offsets[li], e - offsets[li]) for s, e in spans]
layers = len(ref) // 3
rows = []
for i in range(layers):
    disp = [shift(li, r[1][3 * i:3 * i + 3]) for li, r in ranks]
    comb = [shift(li, r[2][i:i + 1])[0] for li, r in ranks]
    sp = [shift(0, r[3][i:i + 1])[0] for r in lanes[0]]
    d_last = max(c[0][0] for c in disp)
    d_end = max(c[2][1] for c in disp)
    c_last = max(c[0] for c in comb)
    c_end = max(c[1] for c in comb)
    lane0_r0 = disp[0]
    rows.append({
        "dispatch_critical": d_end - d_last,
        "dispatch_lane0_rank0_sum": sum(e - s for s, e in lane0_r0),
        "dispatch_lane0_rank0_first_start_to_last_end": lane0_r0[2][1] - lane0_r0[0][0],
        "dispatch_arrival_spread": d_last - min(c[0][0] for c in disp),
        "combine_critical": c_end - c_last,
        "combine_lane0_rank0": comb[0][1] - comb[0][0],
        "combine_arrival_spread": c_last - min(c[0] for c in comb),
        "combine_end_spread": c_end - min(c[1] for c in comb),
        "sp_rs_lane0_rank0": sp[0][1] - sp[0][0],
        "combine_end_to_lane0_sp_rs_end_max": max(s[1] for s in sp) - c_end,
    })
print("host offsets us", [round(o, 1) for o in offsets], "layers", layers)
for key in rows[0]:
    values = [r[key] for r in rows]
    print(f"{key:48s} median {statistics.median(values):8.1f} min {min(values):8.1f} max {max(values):8.1f} sum ms {sum(values) / 1e3:8.3f}")
