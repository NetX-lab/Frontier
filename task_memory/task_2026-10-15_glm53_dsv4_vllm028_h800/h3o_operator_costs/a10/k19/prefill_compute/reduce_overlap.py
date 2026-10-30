"""List the kernels on other streams that overlap each EP combine Reduce in one NONE prefill forward.

    python reduce_overlap.py <trace.json.gz> <forward index>
"""
import collections
import statistics
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events

R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL("
path, index = Path(sys.argv[1]), int(sys.argv[2])
for fw in forward_kernel_events("prefill", path):
    if fw.record["index"] != index:
        continue
    ks = sorted(fw.kernels_by_segment["dispatch"], key=lambda k: k["ts"])
    reduces = [k for k in ks if k["name"].startswith(R)]
    stream = reduces[0]["args"]["stream"]
    overlap = collections.defaultdict(lambda: [0, 0.0])
    per_layer = []
    for r in reduces:
        a, b = r["ts"], r["ts"] + r["dur"]
        o = 0.0
        for k in ks:
            if k is r or k["args"]["stream"] == stream:
                continue
            x = min(b, k["ts"] + k["dur"]) - max(a, k["ts"])
            if x > 0:
                overlap[k["name"][:110]][0] += 1
                overlap[k["name"][:110]][1] += x
                o += x
        prev = [k for k in ks if k["args"]["stream"] == stream and k["ts"] + k["dur"] <= a]
        nxt = [k for k in ks if k["args"]["stream"] == stream and k["ts"] >= b]
        per_layer.append((r["dur"], o, a - (prev[-1]["ts"] + prev[-1]["dur"]) if prev else None,
                          prev[-1]["name"][:60] if prev else None, nxt[0]["name"][:60] if nxt else None))
    print("reduce stream", stream, "count", len(reduces), "sum ms", round(sum(r["dur"] for r in reduces) / 1e3, 3),
          "median us", round(statistics.median(r["dur"] for r in reduces), 1), "min us", round(min(r["dur"] for r in reduces), 1),
          "max us", round(max(r["dur"] for r in reduces), 1))
    print("grid/block", reduces[0]["args"].get("grid"), reduces[0]["args"].get("block"))
    print("overlap with other streams: total ms", round(sum(v[1] for v in overlap.values()) / 1e3, 3))
    for name, (n, us) in sorted(overlap.items(), key=lambda x: -x[1][1])[:15]:
        print(f"  {n:5d} {us/1e3:8.3f} ms  {name}")
    no = [d for d, o, *_ in per_layer if o == 0]
    yes = [d for d, o, *_ in per_layer if o > 0]
    print("layers without overlap", len(no), "median dur", round(statistics.median(no), 1) if no else None,
          "| with overlap", len(yes), "median dur", round(statistics.median(yes), 1) if yes else None)
    print("predecessor on the reduce stream", collections.Counter(p for *_, p, _ in per_layer).most_common(3))
    print("successor on the reduce stream", collections.Counter(n for *_, n in per_layer).most_common(3))
    print("gap before reduce us median", round(statistics.median(g for _, _, g, *_ in per_layer if g is not None), 1))
    for i in (0, 1, 2, 21, 42):
        print("layer", i, [round(v, 1) if isinstance(v, float) else v for v in per_layer[i][:3]])
    break
