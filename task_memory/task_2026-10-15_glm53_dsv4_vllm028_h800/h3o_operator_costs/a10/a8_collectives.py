"""A8 census of the collectives that an A10 replay requested (lead scratch).

Usage: a8_collectives.py <A10 run root>

Reads the A11 observer's collectives.json. MeasuredCCBackend rejects a request
outside a measured curve and an unequal combine without an exact measurement,
so a finished run already holds A8; this census reports how each request was
covered and checks the recorded bounds.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads((root / "receipt.json").read_text())
rows = json.loads((root / "collectives.json").read_text())["rows"]
assert receipt["status"] == "FINISHED", receipt["status"]
assert sum(r["call_count"] for r in rows) == receipt["a11"]["collectives"]["calls"]

violations = []
groups = defaultdict(lambda: {"points": 0, "calls": 0, "exact_calls": 0, "interpolated_calls": 0,
                              "unequal_calls": 0, "max_log_distance": 0.0, "predicted_total_ms": 0.0})
for r in rows:
    key = (r["role"], *map(str, r["curve_key"]))
    g = groups[key]
    g["points"] += 1
    g["calls"] += r["call_count"]
    g["predicted_total_ms"] += r["predicted_total_ms"]
    if r["unequal_rows"] is not None:
        g["unequal_calls"] += r["call_count"]
        if not r["exact"]:
            violations.append(("unequal combine not exact", r))
    if r["exact"]:
        g["exact_calls"] += r["call_count"]
        if not r["lower_bytes"] == r["requested_bytes"] == r["upper_bytes"]:
            violations.append(("exact point with other bounds", r))
    else:
        g["interpolated_calls"] += r["call_count"]
        if not r["lower_bytes"] < r["requested_bytes"] < r["upper_bytes"]:
            violations.append(("interpolated point outside its bracket", r))
        g["max_log_distance"] = max(g["max_log_distance"], r["log_distance_ratio"])

print(f"run {root.name}: {len(rows)} distinct requests, {sum(r['call_count'] for r in rows)} calls")
print(f"{'role / operation / domain / devices':48s} {'points':>6s} {'calls':>9s} {'exact':>9s} "
      f"{'interp':>9s} {'unequal':>8s} {'max_log_dist':>12s} {'total_ms':>12s}")
for key, g in sorted(groups.items()):
    print(f"{' / '.join(key):48s} {g['points']:6d} {g['calls']:9d} {g['exact_calls']:9d} "
          f"{g['interpolated_calls']:9d} {g['unequal_calls']:8d} {g['max_log_distance']:12.3f} "
          f"{g['predicted_total_ms']:12.1f}")
print("violations", len(violations))
for reason, r in violations:
    print(" ", reason, json.dumps(r))
