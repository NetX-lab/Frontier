"""Analyze F2 depth versus graph kernels and predict K without assuming queue units."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from frontier.profiling.collectives.cuda_launch_depth import blocking_depth


def interpolate_depth(points, kernels):
    """Interpolate only between measured sizes; report the integer launch budget."""
    for lower, upper in zip(points, points[1:]):
        left, right = lower["kernels_per_graph"], upper["kernels_per_graph"]
        if left <= kernels <= right:
            weight = math.log(kernels / left) / math.log(right / left)
            depth = lower["depth"] * (upper["depth"] / lower["depth"]) ** weight
            return dict(kernels_per_graph=kernels, depth=depth,
                        complete_forwards=math.floor(depth), bracket=[lower, upper])
    return None


def analyze(records, native_kernels):
    if type(native_kernels) is not int or native_kernels <= 0:
        raise ValueError("Native per-forward kernel count must be a positive integer")
    by_k = defaultdict(list)
    by_sleep = defaultdict(list)
    trials = []
    plain = []
    for row in records:
        measured = blocking_depth(row)
        trials.append({k: row[k] for k in ("mode", "kernels_per_graph", "sleep_ms", "trial")} | measured)
        if measured["censored"]:
            continue
        if row["mode"] == "graph":
            by_k[int(row["kernels_per_graph"])].append(measured["depth"])
            by_sleep[(int(row["kernels_per_graph"]), row["sleep_ms"])].append(measured["depth"])
        else:
            plain.append(measured["depth"])
    points = [{"kernels_per_graph": k, "depth": statistics.median(v),
               "min_depth": min(v), "max_depth": max(v), "trials": len(v)}
              for k, v in sorted(by_k.items())]
    sleep_points = [dict(kernels_per_graph=k, sleep_ms=s, depth=statistics.median(v))
                    for (k, s), v in sorted(by_sleep.items())]
    for k in by_k:
        depths = [p["depth"] for p in sleep_points if p["kernels_per_graph"] == k]
        if min(depths) > 0 and max(depths) / min(depths) > 1.2:
            return dict(status="SLEEP_DEPENDENT", points=points, trials=trials,
                        sleep_points=sleep_points, native_kernels=native_kernels,
                        reason="Capacity changes by >20% with the sleep duration; no reliable K prediction")
    # Compare relative/log residuals, so the large-k points count as much as k=1.
    if len(points) < 3 or any(p["depth"] <= 0 for p in points):
        return dict(status="INSUFFICIENT_EVIDENCE", points=points, trials=trials,
                    reason="Need >=3 distinct uncensored positive-depth graph sizes")
    graph_capacity = math.exp(statistics.fmean(math.log(p["depth"]) for p in points))
    node_capacity = math.exp(statistics.fmean(math.log(p["depth"] * p["kernels_per_graph"])
                                            for p in points))
    def error(kind, capacity):
        return math.sqrt(statistics.fmean((math.log(p["depth"]) - math.log(
            capacity if kind == "graph_launches" else capacity / p["kernels_per_graph"]))**2
                                         for p in points))
    models = {"graph_launches": dict(capacity=graph_capacity,
                                     log_rmse=error("graph_launches", graph_capacity),
                                     predicted_K=graph_capacity),
              "kernel_nodes": dict(capacity=node_capacity, log_rmse=error("kernel_nodes", node_capacity),
                                   predicted_K=node_capacity / native_kernels)}
    best = min(models, key=lambda k: models[k]["log_rmse"])
    other = next(k for k in models if k != best)
    decisive = models[best]["log_rmse"] < 0.2 and models[best]["log_rmse"] + 0.1 < models[other]["log_rmse"]
    heldout = []
    for i, point in enumerate(points[1:-1], start=1):
        prediction = interpolate_depth(points[:i] + points[i + 1:], point["kernels_per_graph"])
        heldout.append(dict(kernels_per_graph=point["kernels_per_graph"],
                            measured_depth=point["depth"], predicted_depth=prediction["depth"],
                            relative_error=abs(prediction["depth"] / point["depth"] - 1)))
    return dict(status="PREDICTED" if decisive else "AMBIGUOUS", preferred_model=best if decisive else None,
                models=models, native_kernels=native_kernels, observed_K=19,
                predicted_K=models[best]["predicted_K"] if decisive else None,
                predicted_minus_19=models[best]["predicted_K"] - 19 if decisive else None,
                plain_depth_median=statistics.median(plain) if plain else None,
                points=points, sleep_points=sleep_points, trials=trials,
                local_interpolation=dict(
                    method="piecewise log-log interpolation between measured graph sizes; no extrapolation",
                    native=interpolate_depth(points, native_kernels),
                    heldout=heldout,
                    qualification="diagnostic synthetic graph prediction; independent of single-capacity model selection"),
                limitation="Synthetic single-stream graphs omit native concurrent streams, memory nodes and collectives")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--native-count", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    native = json.loads(args.native_count.read_text())
    counts = native["kernel_counts"]
    if len(counts) != 1:
        raise ValueError("Native dummy counts vary; select the matching native class before predicting K")
    result = analyze([json.loads(l) for l in args.input.read_text().splitlines()], counts[0])
    result["native_evidence"] = native
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("trials", "native_evidence")}, indent=2))


if __name__ == "__main__":
    main()
