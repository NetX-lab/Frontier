"""Relate each D-side NIXL READ's step count to prefill activity on the P nodes (H200 P2-32 lifecycle run)."""
import bisect
import collections
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("nodes/*/scheduler_lifecycle/*.jsonl"))
lanes = {}
for f in files:
    rows = [json.loads(line) for line in f.open()]
    lanes[(rows[0]["role"], rows[0]["dp_rank"])] = rows

# P busy intervals: wall time from a schedule with tokens to the update of the same step.
busy = []
for (role, dp), rows in lanes.items():
    if role != "prefill":
        continue
    sched = {r["step"]: r["t"] for r in rows if r["event"] == "schedule" and r.get("tokens")}
    for r in rows:
        if r["event"] == "update" and r["step"] in sched:
            busy.append((sched[r["step"]], r["t"], dp))
busy.sort()
print("prefill steps with tokens", len(busy), "mean ms", round(statistics.mean((b - a) * 1e3 for a, b, _ in busy), 1))


def busy_overlap(t0, t1):
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b, _ in busy if a < t1 and b > t0)


out = []
for (role, dp), rows in lanes.items():
    if role != "decode":
        continue
    sched_t = [r["t"] for r in rows if r["event"] == "schedule"]
    read_at = {}
    for r in rows:
        if r["event"] == "schedule":
            for rid in r.get("recv", []):
                read_at[rid] = r["t"]
        if r["event"] == "update":
            for rid in r.get("finished_recving", []):
                if rid in read_at:
                    t0 = read_at.pop(rid)
                    k = bisect.bisect_right(sched_t, r["t"]) - bisect.bisect_right(sched_t, t0)
                    out.append(dict(dp=dp, t0=t0, t1=r["t"], k=k, d=(r["t"] - t0) * 1e3))
out.sort(key=lambda x: x["t0"])
for i, o in enumerate(out):
    o["busy_ms"] = busy_overlap(o["t0"], o["t1"]) * 1e3
    o["p_busy_at_start"] = any(a <= o["t0"] + 0.02 < b for a, b, _ in busy)
    prev = [x for x in out[:i] if x["t1"] > o["t0"]]
    o["other_lane_read"] = bool(prev)
print("reads", len(out))
by = collections.defaultdict(list)
for o in out:
    by[o["k"]].append(o)
for k in sorted(by):
    g = by[k]
    print(f"k={k} n={len(g)} d={statistics.mean(o['d'] for o in g):.1f} "
          f"P-busy-at-start={sum(o['p_busy_at_start'] for o in g)}/{len(g)} "
          f"P-busy-ms-in-window={statistics.mean(o['busy_ms'] for o in g):.1f} "
          f"other-lane-read-overlap={sum(o['other_lane_read'] for o in g)}")
for flag in (True, False):
    g = [o for o in out if o["p_busy_at_start"] == flag]
    print("P busy at start" if flag else "P idle at start", len(g), "mean k", round(statistics.mean(o["k"] for o in g), 2),
          collections.Counter(o["k"] for o in g))

# Source P lane per READ: the D add record names the P request; the P add record carries the P lane.
p_lane_of = {}
for (role, dp), rows in lanes.items():
    if role == "prefill":
        for r in rows:
            if r["event"] == "add":
                p_lane_of[r["request_id"]] = dp
src = {}
for (role, dp), rows in lanes.items():
    if role == "decode":
        for r in rows:
            if r["event"] == "add":
                src[r["request_id"]] = p_lane_of.get(r.get("remote_request_id"))
# Rebuild with request ids for the route split.
route = collections.defaultdict(list)
for (role, dp), rows in lanes.items():
    if role != "decode":
        continue
    sched_t = [r["t"] for r in rows if r["event"] == "schedule"]
    read_at = {}
    for r in rows:
        if r["event"] == "schedule":
            for rid in r.get("recv", []):
                read_at[rid] = r["t"]
        if r["event"] == "update":
            for rid in r.get("finished_recving", []):
                if rid in read_at and "formal" in rid:
                    t0 = read_at.pop(rid)
                    k = bisect.bisect_right(sched_t, r["t"]) - bisect.bisect_right(sched_t, t0)
                    route[(src.get(rid), dp)].append(k)
for key in sorted(route, key=str):
    ks = route[key]
    print("P dp%s -> D dp%s" % key, len(ks), "mean k", round(statistics.mean(ks), 2), sorted(collections.Counter(ks).items()))
