"""Compare G1h-02 and supplement repeat medians per scope and target on CPU.

Use each package's ordinary replay reducer first, including supplement repeats.
This tool reads their CSV rows and the fixed case-to-window manifests. The
relative difference is (supplement median - G1h median) / G1h median; a zero
baseline yields null and retains the absolute difference for signed gaps.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from frontier.profiling.attention.v4_attention_windows import AttentionWindow, repeat_windows


def window_from_json(row):
    return AttentionWindow(**{key: tuple(row[key]) if key == "mix" and row[key] is not None else row[key]
                              for key in AttentionWindow.__dataclass_fields__})


def repeat_cases(grid):
    repeats = set(repeat_windows())
    result = {}
    for row in grid["windows"]:
        window = window_from_json(row)
        if window in repeats and row["status"] == "predicted":
            result.setdefault(row["case_id"], []).append(window)
    return result


def samples(table: Path, grid):
    cases = repeat_cases(grid)
    result = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for window in cases.get(row["case_id"], ()):
                for target, prefix in (("time", "time_stats"), ("gap", "gap_stats")):
                    for scope in json.loads(row["operator_scopes"]):
                        key = (window, row["attention_variant"], row["plans_tile_schedule"], scope, target)
                        result.setdefault(key, []).append(float(row[f"{prefix}.{scope}.median"]))
    return result


def compare_packages(base_table: Path, supplement_table: Path, base_grid, supplement_grid):
    base, supplement = samples(base_table, base_grid), samples(supplement_table, supplement_grid)
    rows = []
    for key in sorted(base.keys() | supplement.keys(), key=str):
        window, variant, planner, scope, target = key
        before = median(base[key]) if key in base else None
        after = median(supplement[key]) if key in supplement else None
        difference = after - before if before is not None and after is not None else None
        rows.append(dict(window=window.to_json(), variant=variant, plans_tile_schedule=planner,
                         scope=scope, target=target, base_samples=len(base.get(key, ())),
                         supplement_samples=len(supplement.get(key, ())), base_median=before,
                         supplement_median=after, absolute_difference=difference,
                         relative_difference=difference / before if difference is not None and before != 0 else None,
                         status="compared" if difference is not None else "missing"))
    return dict(rows=rows, missing=sum(row["status"] == "missing" for row in rows),
                compared=sum(row["status"] == "compared" for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--supplement-table", type=Path, required=True)
    parser.add_argument("--base-grid", type=Path, required=True)
    parser.add_argument("--supplement-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_packages(args.base_table, args.supplement_table,
                              json.loads(args.base_grid.read_text()), json.loads(args.supplement_grid.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(compared=report["compared"], missing=report["missing"])))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
