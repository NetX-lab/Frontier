import gzip, json, sys
from collections import Counter
with gzip.open(sys.argv[1], "rt") as f:
    events = json.load(f)["traceEvents"]
launch = Counter(e["name"] for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Graph" in e["name"])
print("graph APIs", launch.most_common(8))
graph_corr = {e["args"]["correlation"] for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e["name"]}
kernels = [e for e in events if e.get("cat") == "kernel"]
eager = [k for k in kernels if k["args"]["correlation"] not in graph_corr]
print("eager kernel names", Counter(k["name"][:70] for k in eager).most_common(12))
print("eager streams", Counter(k["args"].get("stream") for k in eager).most_common(4))
annos = Counter(e["name"][:60] for e in events if e.get("cat") in ("user_annotation", "gpu_user_annotation"))
print("annotations", annos.most_common(15))
