"""Estimate the A6 error that measurement variation alone produces per attention scope.

For every workload/layer row measured in the reference run and in two independent
runs, compare the reference median with the mean of the two other runs, using the
reference value as the denominator as A6 does. Report the two independent runs
against each other as a check on the instrumentation difference. No model is fitted
and no table changes; the result is diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np

from v4_attention_heldout import A6_MEDIAN_LIMIT, A6_P90_LIMIT
from v4_attention_repeats import read_rows


def summary(errors):
    errors = np.asarray(errors)
    median, p90 = np.quantile(errors, [0.5, 0.9])
    return {"rows": int(errors.size), "median": float(median), "p90": float(p90),
            "above_10_percent": int((errors > A6_P90_LIMIT).sum()),
            "fraction_above_10_percent": float((errors > A6_P90_LIMIT).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, nargs=2, required=True)
    parser.add_argument("--a6-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    reference, reference_input = read_rows(args.reference)
    first, first_input = read_rows(args.comparison[0])
    second, second_input = read_rows(args.comparison[1])
    if set(reference) != set(first) or set(reference) != set(second):
        raise ValueError("The three runs do not measure the same workload/layer identities")
    a6_bytes = args.a6_result.read_bytes()
    a6_scopes = json.loads(a6_bytes)["attention"]["per_scope"]
    output = {"classification": "DIAGNOSTIC_MEASUREMENT_NOISE_FLOOR",
              "inputs": {"reference": reference_input, "comparisons": [first_input, second_input],
                         "a6_result": {"path": str(args.a6_result.resolve()),
                                       "sha256": hashlib.sha256(a6_bytes).hexdigest()}},
              "limits": {"median": A6_MEDIAN_LIMIT, "p90": A6_P90_LIMIT}, "scopes": {}}
    for scope, recorded in a6_scopes.items():
        column = f"time_stats.{scope}.median"
        reference_errors, independent_errors, by_group = [], [], {}
        for key, row in reference.items():
            if column not in row:
                continue
            a, b, c = row[column], first[key][column], second[key][column]
            if min(a, b, c) <= 0:
                raise ValueError(f"Non-positive measured duration: {row['case_id']} {scope}")
            error = abs(a - (b + c) / 2) / a
            reference_errors.append(error)
            independent_errors.append(abs(b - c) / c)
            group = f"{row['attention_variant']}/{row['graph_runtime_mode']}/{row['phase']}/" \
                    f"{'first' if row['plans_tile_schedule'] else 'reuse'}"
            by_group.setdefault(group, []).append(error)
        if not reference_errors:
            continue
        joint = recorded["joint"]
        a6 = joint.get("kernel_time_relative_error") or {}
        output["scopes"][scope] = {
            "a6_verdict": joint["verdict"],
            "a6_median": a6.get("median_absolute"), "a6_p90": a6.get("p90_absolute"),
            "reference_against_independent_mean": summary(reference_errors),
            "independent_runs_against_each_other": summary(independent_errors),
            "groups": {group: summary(errors) for group, errors in sorted(by_group.items())},
        }
        result = output["scopes"][scope]
        print(f"{scope:45s} A6 {joint['verdict']:12s} p90 {result['a6_p90'] or float('nan'):.4f} | "
              f"noise p90 {result['reference_against_independent_mean']['p90']:.4f} "
              f">10% {result['reference_against_independent_mean']['fraction_above_10_percent']:.3f} | "
              f"G2h pair p90 {result['independent_runs_against_each_other']['p90']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
