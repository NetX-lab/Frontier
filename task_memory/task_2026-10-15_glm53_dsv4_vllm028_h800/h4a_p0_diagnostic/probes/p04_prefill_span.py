"""Cross-check probe: GPU annotation length and contiguous kernel span of the profile04 P1 (dp1, TP0) prefill steps."""
import bisect
import gzip
import json
import os

ROOT = "/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04"
TRACE = "torch_traces/prefill/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792842747939163438.pt.trace.json.gz"
GAP_US = 2000  # a kernel gap above 2 ms ends the step's contiguous span

events = json.load(gzip.open(os.path.join(ROOT, TRACE)))["traceEvents"]
steps = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context")),
               key=lambda e: e["ts"])
kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") == "kernel")
starts = [k[0] for k in kernels]
host = [(e["name"], round(e["dur"] / 1e3, 3)) for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context_1")]
print("cpu annotations of prefill steps (ms)", host)
for i, step in enumerate(steps):
    j = bisect.bisect_left(starts, step["ts"])
    limit = steps[i + 1]["ts"] if i + 1 < len(steps) else float("inf")
    last, k = kernels[j][1], j
    while k < len(kernels) and kernels[k][0] < limit and kernels[k][0] - last < GAP_US:
        last = max(last, kernels[k][1])
        k += 1
    print(step["name"], "annotation", round(step["dur"] / 1e3, 3), "contiguous kernel span",
          round((last - kernels[j][0]) / 1e3, 3), "kernels", k - j)
