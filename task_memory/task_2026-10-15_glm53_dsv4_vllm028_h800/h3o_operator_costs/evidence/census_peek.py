"""Quick census of one native decode trace: graph-launched vs eager kernels and gaps by class."""
import gzip, json, sys
from collections import Counter, defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    events = json.load(f)["traceEvents"]
graph_corr = set()
runtime_names = Counter()
for e in events:
    if e.get("cat") in ("cuda_runtime", "cuda_driver"):
        runtime_names[e["name"]] += 1
        if e["name"].startswith(("cudaGraphLaunch", "cuGraphLaunch")):
            graph_corr.add(e["args"]["correlation"])
kernels = [e for e in events if e.get("cat") == "kernel"]
streams = Counter(k["args"].get("stream") for k in kernels)
main_stream = streams.most_common(1)[0][0]
print("kernels", len(kernels), "graph-launched", sum(k["args"]["correlation"] in graph_corr for k in kernels))
print("streams (top 6)", streams.most_common(6))
print("runtime calls (top 12)", runtime_names.most_common(12))
# gaps on the main stream, classified by the kinds of the two neighbouring kernels
ks = sorted((k for k in kernels if k["args"].get("stream") == main_stream), key=lambda k: k["ts"])
gap = defaultdict(list)
for a, b in zip(ks, ks[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g < 0 or g > 2000:  # >2 ms treated as a step boundary, not an intra-forward gap
        continue
    ca = a["args"]["correlation"] in graph_corr
    cb = b["args"]["correlation"] in graph_corr
    gap[("graph" if ca else "eager") + "->" + ("graph" if cb else "eager")].append(g)
for key, values in sorted(gap.items()):
    values.sort()
    n = len(values)
    print(key, "n", n, "mean_us %.2f" % (sum(values) / n), "p50_us %.2f" % values[n // 2], "sum_ms %.1f" % (sum(values) / 1000))
