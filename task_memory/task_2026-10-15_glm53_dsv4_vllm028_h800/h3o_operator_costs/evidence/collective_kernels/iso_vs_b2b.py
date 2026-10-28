"""Median of summary.csv median_us across passes: isolated run vs back-to-back run.

Usage: iso_vs_b2b.py <isolated summary.csv> <back-to-back summary.csv>
"""
import csv
import statistics
import sys
from collections import defaultdict


def table(path):
    values = defaultdict(list)
    meta = {}
    for row in csv.DictReader(open(path)):
        key = (row["scenario"], row["call"])
        values[key].append((float(row["median_us"]), float(row["min_us"]), float(row["p10_us"])))
        meta[key] = row
    return {key: tuple(statistics.median(v[i] for v in rows) for i in range(3)) for key, rows in values.items()}, meta


iso, meta = table(sys.argv[1])
b2b, _ = table(sys.argv[2])
print(f"{'scenario':16s} {'call':34s} {'group':5s} {'iso med/min/p10':>24s} {'b2b med/min/p10':>24s} {'d_med':>8s}")
for key in sorted(iso):
    row = meta[key]
    a = iso[key]
    b = b2b.get(key)
    text = f"{key[0]:16s} {key[1]:34s} {row['group']:5s} {a[0]:8.1f}/{a[1]:7.1f}/{a[2]:7.1f}"
    if b:
        text += f" {b[0]:8.1f}/{b[1]:7.1f}/{b[2]:7.1f} {a[0] - b[0]:8.1f}"
    else:
        text += "  (no back-to-back row)"
    print(text)
