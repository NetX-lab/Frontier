"""Sweep the verified Frontier FIFO model over small prefill/dummy price changes (read-only, CPU)."""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from frontier_fifo import simulate, summary

rows = []
for gi in range(0, 301):
    g = 24.20 + gi * 0.001
    for P in (279.427, 279.7662, 279.8667):
        w, a = summary(simulate(g, P))
        rows.append(dict(g=g, P=P, cycle_excess_ms=P + 31 * g - 1000, wait_ms=w, ahead=a))
waits = [r["wait_ms"] for r in rows]
print("g 24.200-24.500 ms x 3 prefill prices: wait min %.2f max %.2f mean %.2f median %.2f sd %.2f" % (
    min(waits), max(waits), statistics.mean(waits), statistics.median(waits), statistics.stdev(waits)))
near = [r for r in rows if 24.33 <= r["g"] <= 24.39]
nw = [r["wait_ms"] for r in near]
print("g 24.330-24.390 (observed generation range): wait min %.2f max %.2f mean %.2f sd %.2f n %d" % (min(nw), max(nw), statistics.mean(nw), statistics.stdev(nw), len(nw)))
# wait as a function of cycle excess, binned
bins = {}
for r in rows:
    bins.setdefault(round(r["cycle_excess_ms"]), []).append(r["wait_ms"])
print("cycle excess (ms) -> mean wait range:", {k: (round(min(v), 1), round(max(v), 1)) for k, v in sorted(bins.items())})
for g in (23.81, 24.0, 24.38):
    print("g", g, "P 279.6 -> wait %.2f" % summary(simulate(g, 279.6))[0])
json.dump(rows, open(Path(__file__).parent / "fifo_sweep.json", "w"))
