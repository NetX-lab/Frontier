"""Lead scratch: split the native P-side engine time of each formal request on the prefill lanes."""
import collections
import json
import statistics
import sys
from pathlib import Path

nodes = Path(sys.argv[1])
def q(values):
    values = sorted(values)
    if not values:
        return "n=0"
    pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]
    return f"n={len(values)} mean={statistics.mean(values):.1f} p10={pick(.1):.1f} p50={pick(.5):.1f} p90={pick(.9):.1f} max={values[-1]:.1f}"
for path in sorted(nodes.glob("*/scheduler_lifecycle/*.jsonl")):
    rows = [json.loads(line) for line in path.open()]
    if rows[0].get("role") != "prefill":
        continue
    adds, first_schedule, first_update = {}, {}, {}
    step_rows = collections.defaultdict(dict)
    for r in rows:
        if r["event"] == "add":
            adds[r["request_id"]] = r
        elif r["event"] == "schedule":
            step_rows[r["step"]]["schedule"] = r
            for rid, *_ in r.get("new", []):
                first_schedule.setdefault(rid, r)
        elif r["event"] == "update":
            step_rows[r["step"]]["update"] = r
            for rid in r.get("first_token", []):
                first_update.setdefault(rid, r)
    formal = [rid for rid in adds if "warmup" not in rid]
    wait = [(first_schedule[r]["t"] - adds[r]["t"]) * 1e3 for r in formal]
    step = [(first_update[r]["t"] - first_schedule[r]["t"]) * 1e3 for r in formal]
    nsteps = [first_update[r]["step"] - first_schedule[r]["step"] + 1 for r in formal]
    engine = [(first_update[r]["t"] - adds[r]["t"]) * 1e3 for r in formal]
    spans = collections.Counter()
    durations = collections.defaultdict(list)
    for s, pair in step_rows.items():
        if "schedule" in pair and "update" in pair:
            durations[pair["schedule"].get("tokens", -1)].append((pair["update"]["t"] - pair["schedule"]["t"]) * 1e3)
    print(path.parent.parent.name, rows[0]["dp_rank"], "formal", len(formal))
    print("  add->first schedule", q(wait))
    print("  first schedule->first-token update", q(step))
    print("  steps spanned", collections.Counter(nsteps))
    print("  add->first-token update", q(engine))
    for tokens in sorted(durations):
        print("  step tokens", tokens, q(durations[tokens]))
