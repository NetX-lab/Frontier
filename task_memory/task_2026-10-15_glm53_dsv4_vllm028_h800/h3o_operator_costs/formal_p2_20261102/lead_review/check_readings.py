"""Lead review of Q4: formal readings versus two-fix readings and the per-repeat warm and 30% checks."""
import csv
import json
from pathlib import Path

H = Path("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs")
FORMAL, TWO_FIX = H / "formal_p2_20261102/readings", H / "two_fix_20261101/readings"


def strip_paths(node):
    if isinstance(node, dict):
        return {k: strip_paths(v) for k, v in node.items() if k != "path"}
    if isinstance(node, list):
        return [strip_paths(v) for v in node]
    return node


for case in ("p2_32", "p2_64"):
    for kind in ("warm", "empty"):
        formal = json.loads((FORMAL / f"{case}_{kind}_reading.json").read_text())
        old = json.loads((TWO_FIX / f"{case}_{kind}_reading.json").read_text())
        same = strip_paths(formal) == strip_paths(old)
        # all_request_verbatim embeds the gate's own reading text, which carries no paths
        print(case, kind, "formal == two-fix (paths excluded):", same)

rows = list(csv.DictReader(open(H / "formal_p2_20261102/assessment/gate_readings.csv")))
print("gate_readings.csv columns:", list(rows[0].keys()))
for case in ("p2_32", "p2_64"):
    reading = json.loads((FORMAL / f"{case}_warm_reading.json").read_text())
    worst_warm, worst_all = 0.0, 0.0
    for label, rep in reading["warm"]["repetitions"].items():
        all_rep = reading["all_request"]["repetitions"][label]
        warm_errors = {m["metric"]: m["relative_error"] for m in rep["metrics"]}
        all_errors = {m["metric"]: m["relative_error"] for m in all_rep["metrics"]}
        excluded = [m.get("excluded_count") for m in rep["metrics"]]
        worst_warm = max(worst_warm, *warm_errors.values())
        worst_all = max(worst_all, *all_errors.values())
        print(case, label.split("-")[-1], "warm", {k: round(v * 100, 6) for k, v in warm_errors.items()},
              "verdict", rep.get("verdict"), "excluded", excluded)
        print(case, label.split("-")[-1], "all ", {k: round(v * 100, 6) for k, v in all_errors.items()},
              "verdict", all_rep.get("verdict"))
        for metric, value in warm_errors.items():
            match = [r for r in rows if r["case"] == case and label.endswith(r["repeat"]) and r["metric"] == metric]
            assert len(match) == 1, (case, label, metric, len(match))
            assert abs(float(match[0]["warm_relative_error"]) - value) < 1e-15, (match[0], value)
            assert abs(float(match[0]["all_relative_error"]) - all_errors[metric]) < 1e-15
    print(case, "max warm error %", worst_warm * 100, "max all-request error %", worst_all * 100,
          "30% predicate on every error:", worst_warm <= 0.30 and worst_all <= 0.30)
print("gate_readings.csv matches the formal readings for every repeat and metric")
