"""Lead scratch: classify native formal prefills by the wait before their forward completes.

    python p_blackout.py <nodes dir> <ungated compute ms>

A formal prefill is one 6144-token step (schedule -> first-token update). The
effective forward start is the update time minus the ungated compute time.
"""
import json
import statistics
import sys
from pathlib import Path

nodes, compute_ms = Path(sys.argv[1]), float(sys.argv[2])
steps = []
for path in sorted(nodes.glob("*/scheduler_lifecycle/*.jsonl")):
    rows = [json.loads(line) for line in path.open()]
    if rows[0].get("role") != "prefill":
        continue
    pending = {}
    for r in rows:
        if r["event"] == "schedule" and r.get("tokens") == 6144:
            pending[r["step"]] = r
        elif r["event"] == "update" and r.get("step") in pending:
            s = pending.pop(r["step"])
            steps.append((s["t"], r["t"], "warmup" in s["new"][0][0]))
steps.sort()
formal = [(i, s, e) for i, (s, e, w) in enumerate(steps) if not w]
durations = [(e - s) * 1e3 for _, s, e in formal]
print("formal", len(formal), "duration mean %.1f min %.1f max %.1f" % (statistics.mean(durations), min(durations), max(durations)))
rows = []
for i, s, e in formal:
    if i == 0 or s - steps[i - 1][1] > 2.0:
        continue
    previous_end = steps[i - 1][1]
    rows.append(((s - previous_end) * 1e3, (e - s) * 1e3, (e - previous_end) * 1e3 - compute_ms))
for label, low, high in (("arrival < 260 ms after previous end", 0, 260),
                         ("arrival 260-760 ms after previous end", 260, 760),
                         ("arrival >= 760 ms after previous end", 760, 1e9)):
    sel = [r for r in rows if low <= r[0] < high]
    if sel:
        print(f"{label}: n {len(sel)}, duration mean {statistics.mean(r[1] for r in sel):.1f} "
              f"min {min(r[1] for r in sel):.1f} max {max(r[1] for r in sel):.1f}, "
              f"effective start after previous end mean {statistics.mean(r[2] for r in sel):.1f} "
              f"sd {statistics.pstdev(r[2] for r in sel):.1f}")
early = [r for r in rows if r[0] < 260]
served = [r for r in early if r[1] < compute_ms + 100]
held = [r for r in early if r[1] >= compute_ms + 100]
if early:
    print(f"arrival < 260 ms split: served at once n {len(served)} (arrival {min(r[0] for r in served):.0f}-{max(r[0] for r in served):.0f} ms, "
          f"duration mean {statistics.mean(r[1] for r in served):.1f}); held n {len(held)} (arrival {min(r[0] for r in held):.0f}-{max(r[0] for r in held):.0f} ms, "
          f"effective start after previous end mean {statistics.mean(r[2] for r in held):.1f} sd {statistics.pstdev(r[2] for r in held):.1f})")
print(f"mean wait over the {len(rows)} classified prefills, taking held waits as duration - compute: {statistics.mean([max(0.0, r[1] - compute_ms) for r in rows]):.1f} ms")
