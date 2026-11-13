#!/usr/bin/env python3
"""Summarize Step-5 MoE profiles and their kernel traces into one JSON file.

Reads every ``moe*.csv`` under ``<root>/compute`` and, for each directory that
holds ``profiler_traces``, the CUDA kernels launched inside each ``vidur_*``
operator scope. Prints the summary as the last line of output, so it survives
the platform's log-tail retention, and writes it to ``--summary``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROW_KEYS = (
    "moe_layer_ids",
    "moe_routed_precision",
    "moe_activation",
    "num_tokens",
    "load_distribution",
    "num_experts_per_device",
    "expert_hidden_dim",
    "routing_weight_policy",
    "moe_native_backend",
    "moe_native_experts_class",
    "moe_triton_tuning_config",
    "measurement_type",
)


def summarize_csv(path: Path) -> dict:
    frame = pd.read_csv(path)
    median_columns = [
        column
        for column in frame.columns
        if column.startswith("time_stats.") and column.endswith(".median")
    ]
    rows = []
    for record in frame.to_dict("records"):
        row = {key: record[key] for key in ROW_KEYS if key in record}
        row["median_ms"] = {
            column.removeprefix("time_stats.").removesuffix(".median"): record[column]
            for column in median_columns
        }
        rows.append(row)
    return {"path": str(path), "num_rows": len(frame), "rows": rows}


def scope_kernels(trace_path: Path) -> dict[str, Counter]:
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels_by_correlation = {
        event["args"]["correlation"]: event["name"]
        for event in events
        if event.get("cat") == "kernel" and "correlation" in event.get("args", {})
    }
    launches = [
        event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in event.get("args", {})
        and "ts" in event
    ]
    scopes: dict[str, Counter] = {}
    for scope in events:
        name = scope.get("name", "")
        if scope.get("cat") != "user_annotation" or not str(name).startswith("vidur_"):
            continue
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        counter = scopes.setdefault(name.removeprefix("vidur_"), Counter())
        for launch in launches:
            if begin < launch["ts"] and launch["ts"] + launch.get("dur", 0) < end:
                kernel = kernels_by_correlation.get(launch["args"]["correlation"])
                if kernel is not None:
                    counter[kernel] += 1
    return scopes


def summarize_traces(directory: Path) -> dict:
    merged: dict[str, Counter] = {}
    trace_paths = sorted(directory.glob("*.json"))
    for trace_path in trace_paths:
        for scope, counter in scope_kernels(trace_path).items():
            merged.setdefault(scope, Counter()).update(counter)
    return {
        "path": str(directory),
        "num_traces": len(trace_paths),
        "kernels_by_scope": {scope: dict(counter) for scope, counter in sorted(merged.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = {
        "csv": [summarize_csv(path) for path in sorted(args.root.rglob("moe*.csv"))],
        "traces": [
            summarize_traces(path)
            for path in sorted(args.root.rglob("profiler_traces"))
            if path.is_dir() and any(path.glob("*.json"))
        ],
    }
    text = json.dumps(summary, sort_keys=True, default=str)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(text + "\n")
    print("MOE_PROFILE_SUMMARY " + text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
