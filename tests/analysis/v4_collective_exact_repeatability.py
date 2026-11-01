"""Check collective qualification for replays that price every call at an exact measured point.

Rule D-COLLECTIVE-EXACT-REPEATABILITY: the collective input qualifies for one replay when
(1) its A8 census admits every call at an exact measured point, with no interpolation and
no violations, and (2) the points the replay uses repeat across measurement passes: the
replay-time-weighted maximum deviation of any pass median from the pooled median is at
most 10%.

Uniform points and exact unequal-SP points are rebuilt from the measured summary with
the measured backend's own row classification. Each rebuilt pooled median must equal the
price the census recorded, so the check covers exactly the points the replay used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from frontier.cc_backend.backends.measured_cc_backend import (
    _EXACT_SP_CALLS, _ROW_CURVES, MeasuredCCBackend,
)

MAX_WEIGHTED_PASS_DEVIATION = 0.10


def measured_points(summary):
    """Group summary medians by pass for every uniform and exact unequal-SP point."""
    uniform, exact = defaultdict(list), defaultdict(list)
    loader = SimpleNamespace(_source_path=summary)
    with summary.open(newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if row["group"] == "dp":
                continue
            measured = _ROW_CURVES.get((row["call"], None), _ROW_CURVES.get((row["op"], row["group"])))
            key = (*measured, int(row["group_size"]))
            size, time_us = int(row["max_input_bytes"]), float(row["median_us"])
            tokens = tuple(int(value) for value in row["tokens_per_dp_rank"].split("/") if value)
            sample = (row["pass"], time_us)
            if "single_active" in row["scenario"] or len(set(tokens)) != 1:
                if key[:2] in _EXACT_SP_CALLS:
                    rows = MeasuredCCBackend._retained_sp_rows(loader, row, key, size, line)
                    if rows is not None:
                        exact[(key, rows, size)].append(sample)
                continue
            uniform[(key, size)].append(sample)
    return uniform, exact


def check_replay(name, census_path, a8_path, uniform, exact):
    census = json.loads(census_path.read_text())["rows"]
    a8 = json.loads(a8_path.read_text())
    admission = {"exit_code": a8["exit_code"], "calls": a8["calls"], "exact_calls": a8["exact_calls"],
                 "violations": a8["violations"], "census_rows_exact": all(row["exact"] for row in census)}
    admission["pass"] = (a8["exit_code"] == 0 and a8["calls"] == a8["exact_calls"] and a8["violations"] == 0
                         and admission["census_rows_exact"])
    points = []
    for row in census:
        key = tuple(row["curve_key"])
        unequal = row["unequal_rows"]
        samples = (exact[(key, tuple(unequal), row["requested_bytes"])] if unequal is not None
                   else uniform[(key, row["requested_bytes"])])
        pooled = statistics.median(time_us for _, time_us in samples)
        if abs(pooled - row["lower_time_us"]) > 1e-9:
            raise ValueError(f"{name}: rebuilt median {pooled} differs from the census price for {row}")
        by_pass = defaultdict(list)
        for number, time_us in samples:
            by_pass[number].append(time_us)
        passes = {number: statistics.median(values) for number, values in sorted(by_pass.items())}
        points.append({"role": row["role"], "curve_key": row["curve_key"], "unequal_rows": unequal,
                       "bytes": row["requested_bytes"], "call_count": row["call_count"],
                       "predicted_total_ms": row["predicted_total_ms"], "pooled_median_us": pooled,
                       "samples": len(samples), "pass_medians_us": passes,
                       "max_pass_deviation": (max(abs(value - pooled) for value in passes.values()) / pooled
                                              if len(passes) > 1 else None)})
    total = sum(point["predicted_total_ms"] for point in points)
    single = [point for point in points if point["max_pass_deviation"] is None]
    repeated = [point for point in points if point["max_pass_deviation"] is not None]
    weighted = sum(point["predicted_total_ms"] * point["max_pass_deviation"] for point in repeated) / total
    repeatability = {"points": len(points), "predicted_total_ms": total,
                     "single_pass_points": len(single),
                     "single_pass_predicted_ms": sum(point["predicted_total_ms"] for point in single),
                     "weighted_max_pass_deviation": weighted if not single else None,
                     "weighted_max_pass_deviation_of_repeated_points": weighted,
                     "limit": MAX_WEIGHTED_PASS_DEVIATION}
    repeatability["pass"] = not single and weighted <= MAX_WEIGHTED_PASS_DEVIATION
    verdict = "PASS" if admission["pass"] and repeatability["pass"] else "FAIL"
    return {"census": str(census_path.resolve()), "a8": str(a8_path.resolve()), "admission": admission,
            "repeatability": repeatability, "verdict": verdict,
            "points": sorted(points, key=lambda point: -point["predicted_total_ms"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--replay", nargs=3, action="append", required=True, metavar=("NAME", "CENSUS", "A8"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    uniform, exact = measured_points(args.summary)
    result = {"classification": "D_COLLECTIVE_EXACT_REPEATABILITY",
              "summary": {"path": str(args.summary.resolve()),
                          "sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest()},
              "replays": {name: check_replay(name, Path(census), Path(a8), uniform, exact)
                          for name, census, a8 in args.replay}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, replay in result["replays"].items():
        print(name, replay["verdict"], json.dumps(replay["admission"]), json.dumps(replay["repeatability"]))
        for point in replay["points"]:
            deviation = point["max_pass_deviation"]
            print(f"  {point['role']:8s} {'/'.join(map(str, point['curve_key'])):24s} "
                  f"{str(point['unequal_rows'] or ''):12s} {point['bytes']:>10d} {point['predicted_total_ms']:>11.1f} ms "
                  f"passes {len(point['pass_medians_us'])} max dev "
                  + ("single pass" if deviation is None else f"{100 * deviation:6.2f}%"))


if __name__ == "__main__":
    main()
