"""Frontier P2-64 prefill dp0 timeline: queue-ahead sequence and run-ahead dummy host/GPU spacing (read-only)."""
import gzip, json, statistics, sys
from pathlib import Path
run = Path(sys.argv[1])
fs = [json.loads(l) for l in gzip.open(run / "forwards.jsonl.gz", "rt")]
lane = [f for f in fs if f["role"] == "prefill" and f["replica"] == 0 and f["dp"] == 0 and f["phase"] == "serving"]
lane.sort(key=lambda f: f["formation_index"])
pre = [f for f in lane if f["step_class"] == "prefill" or (f["step_class"] == "dummy" and f["mode"] == "NONE")]
print("dp0 forwards", len(lane), "prefill-width forwards", len(pre))
allpre = sorted([f for f in fs if f["role"] == "prefill" and f["step_class"] == "prefill"], key=lambda f: f["start_s"])
print("queue-ahead sequence by start:", [f["queued_ahead_at_formation"] for f in allpre])
# spacing of dummies on dp0
form_gap, gpu = [], []
for a, b in zip(lane, lane[1:]):
    if a["step_class"] == "dummy" and b["step_class"] == "dummy" and a["mode"] == "FULL" and b["mode"] == "FULL":
        form_gap.append((b["formed_s"] - a["formed_s"]) * 1e3)
        gpu.append((b["end_s"] - b["start_s"]) * 1e3)
print("dummy formed gaps quantiles", [round(q, 3) for q in statistics.quantiles(form_gap, n=10)], "gpu", round(statistics.median(gpu), 4))
# wave lengths: count dummies between consecutive prefill-width forwards
idx = [i for i, f in enumerate(lane) if f in pre]
lens = [b - a - 1 for a, b in zip(idx, idx[1:])]
print("dummies between prefill-width forwards (first 40):", lens[:40])
print("keys", sorted(lane[0].keys()))
