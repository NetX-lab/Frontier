"""Count every dispatch kernel of native PREFILL FULL DP dummies for F2.

Reuse the B10/runtime census launch correlation and segment boundaries. Counting
only the collective multiset in B10 would omit compute and preparation kernels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from v4_runtime_census import forward_kernel_events, lane_traces


def count_dummies(forwards):
    counts = Counter()
    for forward in forwards:
        record = forward.record
        if record["mode"] == "FULL" and not record["real"]:
            counts[len(forward.kernels_by_segment["dispatch"])] += 1
    if not counts or 0 in counts:
        raise ValueError("No complete FULL dummy dispatch kernel evidence")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lane", type=int, default=0)
    args = parser.parse_args()
    paths = [path for role, lane, path in lane_traces(args.trace_dir)
             if role == "prefill" and lane == args.lane]
    counts = Counter()
    for path in paths:
        counts.update(count_dummies(forward_kernel_events("prefill", path)))
    if not counts:
        raise ValueError("No requested native prefill lane traces")
    result = dict(role="prefill", mode="FULL", real=False, lane=args.lane,
                  kernel_counts=sorted(counts), forwards_by_kernel_count=dict(counts),
                  sources=[str(p) for p in paths],
                  method="v4_runtime_census.forward_kernel_events: every correlated dispatch kernel; all streams")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
