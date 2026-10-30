"""Lead scratch: all prefill-lane rows between the previous formal prefill end and a gated prefill's end."""
import json
import sys
from pathlib import Path

nodes, min_dur_ms, which = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
lanes = {}
for path in sorted(nodes.glob("*/scheduler_lifecycle/*.jsonl")):
    rows = [json.loads(line) for line in path.open()]
    if rows[0].get("role") == "prefill":
        lanes[rows[0]["dp_rank"]] = rows
steps = []
for lane, rows in lanes.items():
    pending = {}
    for r in rows:
        if r["event"] == "schedule" and r.get("tokens") == 6144 and "warmup" not in r["new"][0][0]:
            pending[r["step"]] = r
        elif r["event"] == "update" and r.get("step") in pending:
            s = pending.pop(r["step"])
            steps.append((s["t"], r["t"], lane, s["new"][0][0]))
steps.sort()
gated = [i for i in range(1, len(steps)) if (steps[i][1] - steps[i][0]) * 1e3 > min_dur_ms]
i = gated[which]
prev_end, (start, end, lane, rid) = steps[i - 1][1], steps[i]
print(f"request {rid} lane {lane} dur {(end - start) * 1e3:.1f} ms; previous end {(prev_end - start) * 1e3:.1f} ms (lane {steps[i - 1][2]})")
for lane_id, rows in sorted(lanes.items()):
    for r in rows:
        if prev_end - 0.05 <= r["t"] <= end + 0.05 or r.get("request_id") == rid:
            extra = {k: v for k, v in r.items() if k not in ("t", "mono", "event", "new")}
            if r.get("new"):
                extra["new"] = [n[0][-30:] for n in r["new"]]
            if "arrival_time" in r:
                extra["arrival_rel_ms"] = round((r["arrival_time"] - start) * 1e3, 1)
            print(f"lane{lane_id} {(r['t'] - start) * 1e3:9.1f} {r['event']:9s} {json.dumps(extra)[:220]}")
