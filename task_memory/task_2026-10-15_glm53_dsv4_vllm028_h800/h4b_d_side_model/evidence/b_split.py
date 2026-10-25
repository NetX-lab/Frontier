"""Split part b (D API arrival -> engine add) into the wait for the engine loop and the rest (H200 P2-32 lifecycle run)."""
import bisect
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
waits, rests, bs = [], [], []
for f in sorted(root.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
    rows = [json.loads(line) for line in f.open()]
    if rows[0]["role"] != "decode":
        continue
    upd = [r["t"] for r in rows if r["event"] == "update"]
    for r in rows:
        if r["event"] != "add" or "formal" not in r["request_id"]:
            continue
        i = bisect.bisect_right(upd, r["t"]) - 1
        last_update = upd[i]
        b = (r["t"] - r["arrival_time"]) * 1e3
        bs.append(b)
        # Only lanes that were stepping: the add follows an update within 2 ms.
        if (r["t"] - last_update) * 1e3 < 2 and last_update > r["arrival_time"]:
            waits.append((last_update - r["arrival_time"]) * 1e3)
            rests.append((r["t"] - last_update) * 1e3)
q = lambda xs: (round(statistics.mean(xs), 2), round(statistics.median(xs), 2), round(min(xs), 2), round(max(xs), 2))
print("b all n", len(bs), "mean/median/min/max", q(bs))
print("adds right after an update", len(waits))
print("arrival -> preceding update (engine loop wait)", q(waits))
print("update -> add", q(rests))
