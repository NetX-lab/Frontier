"""Print the collective kernel sequence of one P2-64 DP1 TP0 eager forward per class.

Shows the per-layer EP dispatch/combine kernels for an unequal-lane prefill
(3 AllGatherV + Reduce), a dummy forward and an equal-lane prefill (forward 98:
1 AllGather + ReduceScatter). Run from the worktree root; output is
per_layer_sequence_stdout.txt.
"""
import collections, sys
from pathlib import Path
sys.path.insert(0, "tests/analysis")
from v4_runtime_census import forward_kernel_events
D = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces")
trace = next((D / "prefill" / "dp1").glob("dp1_*_tp0_*.json.gz"))
shown = collections.Counter()
for fw in forward_kernel_events("prefill", trace):
    r = fw.record
    events = sorted((e for ev in fw.kernels_by_segment.values() for e in ev), key=lambda e: e["ts"])
    comm = [e for e in events if "nccl" in e["name"].lower() or "gather" in e["name"].lower() or "reduce" in e["name"].lower()]
    has_v = any(e["name"].startswith("ncclDevKernel_AllGatherV") for e in comm)
    tag = (r["mode"], r.get("class"), has_v)
    if r["mode"] != "NONE" or shown[tag]:
        continue
    shown[tag] += 1
    print(tag, "index", r["index"], {k: r[k] for k in r if k not in ("index",)})
    print("  comm kernel count", len(comm), collections.Counter(e["name"].split("(")[0][:48] for e in comm))
    for e in comm[:14]:
        print("    %-60s dur=%8.2f us" % (e["name"].split("(")[0][:60], e["dur"]))
    print("    ... last:")
    for e in comm[-6:]:
        print("    %-60s dur=%8.2f us" % (e["name"].split("(")[0][:60], e["dur"]))
