"""CPU-only precheck of local Step attention CSVs; never controls a job."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from frontier.profiling.step5.replay import choose_graph_mode


LAYER = "step5_layer_e2e"
ROUND_OFF = 64 * sys.float_info.epsilon


def number(value, name, *, positive=False, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if integer and result != int(result):
        raise ValueError(f"{name} must be integral")
    return int(result) if integer else result


def close(left, right):
    return math.isclose(left, right, rel_tol=ROUND_OFF, abs_tol=ROUND_OFF)


def inspect_row(row, kind, max_time_cv):
    def num(key, **kwargs):
        return number(row[key], key, **kwargs)

    queries, contexts = (json.loads(row[key]) for key in ("query_lens", "context_lens"))
    if (not queries or len(queries) != len(contexts)
            or any(type(q) is not int or q <= 0 for q in queries)
            or any(type(c) is not int or c < 0 for c in contexts)):
        raise ValueError("query_lens/context_lens must contain matching valid request lengths")
    tokens, batch = num("batch_num_tokens", positive=True, integer=True), num("batch_size", positive=True, integer=True)
    max_seqs = num("native_max_num_seqs", positive=True, integer=True)
    expected_mode = choose_graph_mode(row["phase"], tokens, json.loads(row["native_cudagraph_capture_sizes"]))
    layer = num("global_layer_id", integer=True)
    prefix_status = row["prefix_continuation_validation"]
    checks = {
        "measurement_type": row["measurement_type"] == kind,
        "functional": row["output_state_validation"] == "PASS",
        "prefix_continuation": not any(contexts) or prefix_status == "PASS",
        "graph_dispatch": row["graph_mode"] == expected_mode and batch == len(queries) <= max_seqs
                          and tokens == sum(queries) and (row["phase"] != "decode" or all(q == 1 for q in queries)),
    }
    active = json.loads(row["active_operators"])
    if not active or len(set(active)) != len(active) or LAYER in active:
        raise ValueError("active_operators must name distinct non-layer timing targets")
    iterations = num("profile_iterations", positive=True, integer=True)
    replays, durations = (json.loads(row[key]) for key in ("window_replays", "window_durations_ms"))
    replays = [number(v, "window_replays", positive=True, integer=True) for v in replays]
    durations = [number(v, "window_durations_ms", positive=True) for v in durations]
    minimum = num("min_timing_window_ms", positive=True)
    checks["timing_windows"] = len(durations) == len(replays) == iterations >= 2 and all(v >= minimum for v in durations)
    singles = json.loads(row["per_iteration_time_stats"])
    cv, median_error, means = {}, {}, {}
    counts_ok = True
    for target in (*active, LAYER):
        mean = num(f"time_stats.{target}.mean", positive=True)
        std = num(f"time_stats.{target}.std")
        count = num(f"time_stats.{target}.count", positive=True, integer=True)
        median = number(singles[target]["median"], f"{target} raw median", positive=True)
        raw_count = number(singles[target]["count"], f"{target} raw count", positive=True, integer=True)
        expected_raw_count = iterations if kind == "CUDA_EVENT" and target == LAYER else sum(replays)
        counts_ok &= count == iterations >= 2 and raw_count == expected_raw_count
        means[target], cv[target] = mean, number(std / mean, f"{target} CV")
        median_error[target] = number(abs(mean - median) / median, f"{target} median error")
    checks.update(counts=counts_ok, cv=all(v <= max_time_cv for v in cv.values()),
                  median_agreement=all(v <= max_time_cv for v in median_error.values()))
    if row["stability_pass"] not in ("True", "False"):
        raise ValueError("stability_pass must be True or False")
    result = {
        "cell": f"layer{layer}_{row['phase']}_q{queries}_c{contexts}",
        "kind": kind, "candidate": kind == "KERNEL_ONLY" or expected_mode == "full",
        "checks": checks, "prefix_required": any(contexts), "expected_graph_mode": expected_mode,
        "actual_graph_mode": row["graph_mode"],
        "cv_by_target": cv, "median_relative_error_by_target": median_error,
        "max_operator_cv": max(cv[name] for name in active),
        "reported_stability_pass": row["stability_pass"] == "True",
        "minimum_window_ms": minimum, "shortest_window_ms": min(durations),
        "timing_failures": [{"target": name, "cv": cv[name], "median_relative_error": median_error[name],
                             "bound": max_time_cv} for name in cv
                            if cv[name] > max_time_cv or median_error[name] > max_time_cv],
    }
    total = math.fsum(means[name] for name in active)
    if kind == "CUDA_EVENT":
        scoped = json.loads(row["scoped_layer_time_stats"])
        outer = json.loads(row["scoped_outer_time_stats"])
        for label, stats in (("scoped_layer", scoped), ("scoped_outer", outer)):
            checks["counts"] &= number(stats["count"], f"{label}.count", integer=True) == iterations
            number(stats["std"], f"{label}.std")
        denominator = number(scoped["mean"], "scoped_layer_time_stats.mean", positive=True)
        overhead = number(outer["mean"], "scoped_outer_time_stats.mean", positive=True) / means[LAYER] - 1
        reported = float(row["instrumentation_overhead_fraction"])
        if not math.isfinite(reported) or not math.isfinite(overhead):
            raise ValueError("instrumentation_overhead_fraction must be finite")
        checks["overhead_reporting"] = close(overhead, reported)
        result["instrumentation_overhead_fraction"] = overhead
    else:
        denominator = means[LAYER]
        kernel_count = num("kernel_count", positive=True, integer=True)
        kernel_ms, copy_ms = num("kernel_time_ms", positive=True), num("memory_copy_time_ms")
        copy_count = num("memory_copy_count", integer=True)
        skipped = {key: num(key, integer=True) for key in (
            "skipped_kernel_count", "skipped_memory_copy_count", "skipped_uncorrelated_activity_count")}
        checks["cupti"] = (kernel_count > 0 and skipped["skipped_uncorrelated_activity_count"] == 0
                           and (copy_count > 0 or copy_ms == 0) and close(kernel_ms + copy_ms, math.fsum(durations)))
        result["cupti"] = {"kernel_count": kernel_count, "kernel_time_ms": kernel_ms,
                           "memory_copy_count": copy_count, "memory_copy_time_ms": copy_ms, **skipped}
    ratio = number(total / denominator, "computed operator_sum_to_layer_ratio", positive=True)
    checks["reconciliation"] = close(ratio, num("operator_sum_to_layer_ratio")) and (
        close(ratio, 1) if kind == "KERNEL_ONLY" else -ROUND_OFF <= ratio <= 1 + ROUND_OFF)
    result["operator_sum_to_layer_ratio"] = ratio
    return result


def judge(directory, expected_event_rows=12, expected_gpu_active_rows=8, max_time_cv=.05):
    directory = Path(directory)
    max_time_cv = number(max_time_cv, "max_time_cv", positive=True)
    rows, errors, summary = [], [], {}
    try:
        summary = json.loads((directory / "summary.json").read_text())
    except (OSError, ValueError) as error:
        errors.append(f"summary.json: {error}")
    for kind, filename in (("CUDA_EVENT", "step5_attention.csv"), ("KERNEL_ONLY", "step5_attention_kernel_only.csv")):
        try:
            with (directory / filename).open(newline="") as stream:
                records = list(csv.DictReader(stream))
        except (OSError, ValueError, csv.Error) as error:
            errors.append(f"{filename}: {error}")
            records = []
        for index, row in enumerate(records, 2):
            try:
                rows.append(inspect_row(row, kind, max_time_cv))
            except (KeyError, ValueError, TypeError, IndexError, OverflowError) as error:
                message = f"{filename}:{index}: {type(error).__name__}: {error}"
                errors.append(message)
                rows.append({"cell": message, "kind": kind, "candidate": kind == "KERNEL_ONLY" or row.get("graph_mode") == "full",
                             "checks": {"valid_evidence": False}})
    events = [r for r in rows if r["kind"] == "CUDA_EVENT"]
    gpu = [r for r in rows if r["kind"] == "KERNEL_ONLY"]
    candidates = [r for r in rows if r["candidate"]]
    diagnostics = [r for r in rows if not r["candidate"]]
    event_cells, candidate_cells = ({r["cell"] for r in group} for group in (events, candidates))
    checks = {"row_counts": len(events) == expected_event_rows and len(gpu) == expected_gpu_active_rows,
              "cell_coverage": len(event_cells) == len(events) and event_cells == candidate_cells
                               and len(candidate_cells) == len(candidates)}
    for key in ("measurement_type", "functional", "prefix_continuation", "graph_dispatch", "timing_windows", "counts", "reconciliation"):
        checks[key] = bool(rows) and all(r["checks"].get(key, False) for r in rows)
    checks["cupti"] = bool(gpu) and all(r["checks"].get("cupti", False) for r in gpu)
    checks["overhead_reporting"] = bool(events) and all(r["checks"].get("overhead_reporting", False) for r in events)
    for key in ("cv", "median_agreement"):
        checks[key] = bool(candidates) and all(r["checks"].get(key, False) for r in candidates)
    try:
        checks["summary_counts"] = all(number(summary[key], key, integer=True) == value for key, value in (
            ("expected_rows", expected_event_rows), ("actual_rows", len(events)),
            ("expected_gpu_active_rows", expected_gpu_active_rows), ("gpu_active_rows", len(gpu))))
    except (KeyError, ValueError, TypeError) as error:
        errors.append(f"summary.json: missing or invalid count: {error}")
        checks["summary_counts"] = False
    failed = []
    for row in rows:
        failed_checks = [key for key, passed in row["checks"].items() if not passed
                         and (row["candidate"] or key not in ("cv", "median_agreement"))]
        if failed_checks:
            failed.append({"cell": row["cell"], "kind": row["kind"], "checks": failed_checks,
                           "targets": row.get("timing_failures", []) if row["candidate"] else []})

    def bounds(values):
        return {"min": min(values), "max": max(values)} if values else None

    return {
        "verdict": "PASS" if not errors and all(checks.values()) else "FAIL", "admitted": False,
        "checks": checks, "errors": errors, "failed_cells": failed,
        "counts": {"raw_rows": len(rows), "event_rows": len(events), "gpu_active_rows": len(gpu),
                   "native_cells": len(event_cells), "latency_diagnostic_rows": len(diagnostics),
                   "full_event_candidates": sum(r["candidate"] for r in events),
                   "calibration_candidates": len(candidates),
                   "qualified_candidates": sum(all(r["checks"].values()) for r in candidates),
                   "cv_pass_candidates": sum(r["checks"].get("cv", False) for r in candidates),
                   "output_state_pass_rows": sum(r["checks"].get("functional", False) for r in rows),
                   "prefix_continuation_rows": sum(r.get("prefix_required", False) for r in rows),
                   "prefix_continuation_pass_rows": sum(r.get("prefix_required", False) and r["checks"]["prefix_continuation"] for r in rows)},
        "bounds": {"max_time_cv_and_median_relative_error": max_time_cv, "roundoff_tolerance": ROUND_OFF,
                   "event_scope_allowed": [0, 1], "gpu_active_scope_expected": 1,
                   "full_max_operator_cv": max((r["max_operator_cv"] for r in events if r["candidate"] and "max_operator_cv" in r), default=None),
                   "gpu_active_max_operator_cv": max((r["max_operator_cv"] for r in gpu if "max_operator_cv" in r), default=None),
                   "event_scope_ratio": bounds([r["operator_sum_to_layer_ratio"] for r in events if "operator_sum_to_layer_ratio" in r]),
                   "gpu_active_scope_ratio": bounds([r["operator_sum_to_layer_ratio"] for r in gpu if "operator_sum_to_layer_ratio" in r]),
                   "instrumentation_overhead_fraction": bounds([r["instrumentation_overhead_fraction"] for r in events if "instrumentation_overhead_fraction" in r])},
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-event-rows", type=int, default=12)
    parser.add_argument("--expected-gpu-active-rows", type=int, default=8)
    parser.add_argument("--max-time-cv", type=float, default=.05)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = judge(args.directory, args.expected_event_rows, args.expected_gpu_active_rows, args.max_time_cv)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
