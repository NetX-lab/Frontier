"""Native decode-side KV receive detail per formal request (lead scratch).

    python nixl_detail.py <artifacts> <remainder_out.json> <output json>

For each formal request on its decode lane: the kv_stats entries of the updates
between its add and its finished_recving (lanes receive one request at a time),
the decode forwards in that interval, and whether the lane was in a dummy wave.
"""
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
remainder = {r["request"]: r for r in json.load(open(sys.argv[2]))["rows"]}


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


out = []
for path in sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
    records = rows(path)
    start = next(r for r in records if r["event"] == "start")
    if start["role"] != "decode":
        continue
    adds = {r["request_id"]: r["t"] for r in records if r["event"] == "add" and "formal" in r["request_id"]}
    recv = {}
    for r in records:
        if r["event"] == "update":
            for rid in r.get("finished_recving", []) or []:
                recv[rid] = r["t"]
    updates = [r for r in records if r["event"] == "update"]
    busy = [r for r in records if r["event"] == "busy_iteration"]
    for rid, t_add in adds.items():
        t_recv = recv[rid]
        stats = [r["kv_stats"] for r in updates if t_add < r["t"] <= t_recv and "kv_stats" in r]
        durations = [d * 1000 for s in stats for d in s["transfer_duration"]]
        nbytes = sorted({b for s in stats for b in s["bytes_transferred"]})
        window = [b for b in busy if t_add <= b["t"] < t_recv]
        dummies = sum(1 for b in window if "engine_dummy_batch" in b.get("calls", {}))
        key = rid.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
        out.append(dict(request=key, d_lane=start["dp_rank"], wait_ms=(t_recv - t_add) * 1000,
                        transfers=len(durations), max_transfer_ms=max(durations, default=None),
                        transfer_ms=sorted(round(d, 1) for d in durations), bytes=nbytes,
                        forwards=len(window), dummy_forwards=dummies,
                        forward_ms=[round((b["end_t"] - b["t"]) * 1000, 1) for b in window],
                        remainder_ms=remainder.get(key, {}).get("remainder_ms")))
out.sort(key=lambda o: o["request"])
print("requests", len(out))
print("transfers per request", sorted({o["transfers"] for o in out}))
print("bytes", sorted({b for o in out for b in o["bytes"]}))
for name, sel in (("all", out), ("wait<400", [o for o in out if o["wait_ms"] < 400])):
    waits = [o["wait_ms"] for o in sel]
    maxes = [o["max_transfer_ms"] for o in sel if o["max_transfer_ms"] is not None]
    allt = [d for o in sel for d in o["transfer_ms"]]
    print(f"{name}: n {len(sel)} wait mean {statistics.mean(waits):.1f} median {statistics.median(waits):.1f}; "
          f"max transfer mean {statistics.mean(maxes):.1f} median {statistics.median(maxes):.1f}; "
          f"all transfers mean {statistics.mean(allt):.1f} median {statistics.median(allt):.1f} p90 {sorted(allt)[int(0.9*len(allt))]:.1f}")
dummy = [o for o in out if o["forwards"] and o["dummy_forwards"] == o["forwards"]]
real = [o for o in out if o["dummy_forwards"] == 0]
for name, sel in (("lane idle (all dummies)", dummy), ("lane busy (no dummies)", real)):
    if sel:
        print(f"{name}: n {len(sel)} wait median {statistics.median(o['wait_ms'] for o in sel):.1f} "
              f"max transfer median {statistics.median(o['max_transfer_ms'] for o in sel):.1f}")
for o in sorted(out, key=lambda o: -o["wait_ms"])[:6]:
    print("slowest", json.dumps({k: o[k] for k in ("request", "d_lane", "wait_ms", "transfers", "transfer_ms", "forwards", "dummy_forwards")})[:600])
json.dump(out, open(sys.argv[3], "w"), indent=1)
