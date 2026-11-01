"""Explain repeat-to-repeat variation of DeepSeek-V4 CSA attention scopes from kernel intervals.

In a CSA layer the query, compressor, index-query and index-compressor scopes run on
separate streams between one fork and one join. This analysis reads the retained raw
records of a fixed-geometry repeat package and reports, for every repeated workload:

1. Kernel level: each concurrent scope's duration, its correlation with the time it
   overlaps every other concurrent scope, the start offset between the two fused
   compress-store kernels, and which scope ends last before the join.
2. Row level: across repeats, the spread of each scope's row median, of the attention
   phase that ``compose_attention_phase`` builds from those medians and their gaps, and
   of the measured layer window.
3. Sample size: bootstrap spread of the median and mean of n pooled kernel-level samples.

Diagnostic only: no measured value, table or acceptance rule changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from frontier.execution_time_predictor.dsv4_composition import (
    _COMPRESSOR, _INDEX_COMPRESSOR, _INDEX_QUERY, _QUERY, compose_attention_phase,
)
from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows
from v4_attention_coverage import plan_from_spec
from v4_attention_repeatability import read_selected, row_key

CONCURRENT_SCOPES = tuple(scope for scope in (*_QUERY, *_COMPRESSOR, *_INDEX_QUERY, *_INDEX_COMPRESSOR)
                          if scope != "dsv4_query_padding")
INDEX_STORE, COMPRESSOR_STORE = _INDEX_COMPRESSOR[-1], _COMPRESSOR[-1]
# Start offsets between the two fused compress-store kernels fall in two clusters
# (|offset| < 0.4 us and offset > 3 us); 1 us separates them.
CO_START_US = 1.0
BOOTSTRAP_SAMPLE_SIZES = (1, 4, 16, 64)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1234


def spread(values):
    """Coefficient of variation and first-half versus second-half median difference."""
    mean = statistics.mean(values)
    half = len(values) // 2
    first, second = statistics.median(values[:half]), statistics.median(values[half:])
    return {"samples": len(values), "mean": mean, "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "coefficient_of_variation": statistics.stdev(values) / mean,
            "split_relative_difference": abs(second - first) / first}


def correlation(x, y):
    if statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return None
    return statistics.correlation(x, y)


def overlap_us(left, right):
    return sum(max(0.0, min(a["end_us"], b["end_us"]) - max(a["start_us"], b["start_us"]))
               for a in left for b in right)


def scope_instances(records, mapping):
    """Group kernel intervals by (original case, step, layer, scope) for layers with both stores."""
    calls = {(call["case_id"], call["step_index"], call["global_layer_id"], call["scope"]): call
             for call in records.calls}
    layers = defaultdict(lambda: defaultdict(list))
    for interval in records.intervals:
        if interval["activity"] == "kernel" and interval["scope"] in CONCURRENT_SCOPES:
            layers[(interval["case_id"], interval["step_index"], interval["global_layer_id"])][
                interval["scope"]].append(interval)
    instances = defaultdict(list)
    for (case_id, step, layer), scopes in layers.items():
        if INDEX_STORE not in scopes or COMPRESSOR_STORE not in scopes:
            continue
        for scope, kernels in scopes.items():
            duration = sum(k["end_us"] - k["start_us"] for k in kernels)
            recorded = 1000 * calls[(case_id, step, layer, scope)]["time_ms"]
            if not math.isclose(duration, recorded, rel_tol=0, abs_tol=1e-3):
                raise ValueError(f"{case_id} layer {layer} {scope}: kernels {duration} us, call {recorded} us")
        item = mapping[case_id]
        instances[item["original"]].append({"repeat": item["repeat"], "layer": layer, "scopes": scopes})
    return instances


def kernel_level(instances):
    durations = {scope: [sum(k["end_us"] - k["start_us"] for k in item["scopes"][scope]) for item in instances]
                 for scope in CONCURRENT_SCOPES if all(scope in item["scopes"] for item in instances)}
    offsets = [min(k["start_us"] for k in item["scopes"][COMPRESSOR_STORE])
               - min(k["start_us"] for k in item["scopes"][INDEX_STORE]) for item in instances]
    co_start = [abs(offset) < CO_START_US for offset in offsets]
    last = defaultdict(int)
    for item in instances:
        ends = {scope: max(k["end_us"] for k in kernels) for scope, kernels in item["scopes"].items()}
        last[max(ends, key=ends.get)] += 1
    result = {"instances": len(instances), "launch_kinds": sorted({k["launch_kind"] for item in instances
                                                                   for kernels in item["scopes"].values()
                                                                   for k in kernels}),
              "store_start_offset_us": {"minimum": min(offsets), "median": statistics.median(offsets),
                                        "maximum": max(offsets)},
              "co_start_share": sum(co_start) / len(co_start), "last_ending_scope": dict(last), "scopes": {}}
    for scope, values in durations.items():
        partners = {}
        for other in durations:
            if other != scope:
                exposure = [overlap_us(item["scopes"][scope], item["scopes"][other]) for item in instances]
                partners[other] = {"mean_overlap_us": statistics.mean(exposure),
                                   "correlation": correlation(exposure, values)}
        strongest = max((p for p in partners.items() if p[1]["correlation"] is not None),
                        key=lambda p: abs(p[1]["correlation"]), default=(None, None))
        by_class = {}
        for name, flag in (("co_start", True), ("staggered", False)):
            selected = [value for value, start in zip(values, co_start) if start == flag]
            if selected:
                by_class[name] = {"instances": len(selected), "median_us": statistics.median(selected),
                                  "minimum_us": min(selected), "maximum_us": max(selected)}
        result["scopes"][scope] = {**spread(values), "by_store_start": by_class,
                                   "strongest_overlap_partner": strongest[0],
                                   "strongest_overlap_correlation": strongest[1] and strongest[1]["correlation"],
                                   "overlap_partners": partners}
    return result


def composed_ms(row, threshold):
    scopes = json.loads(row["operator_scopes"])
    costs = {scope: row[f"time_stats.{scope}.median"] for scope in scopes}
    gaps = {scope: row[f"gap_stats.{scope}.median"] for scope in scopes}
    return compose_attention_phase(
        costs, gaps, variant=row["attention_variant"], graph_runtime_mode=row["graph_runtime_mode"],
        rows=row["padded_rows"], multi_stream_gemm_token_threshold=threshold,
        indexer_short_context=row["indexer_short_context"]).elapsed_ms


def row_level(rows, mapping, thresholds):
    groups = defaultdict(dict)
    for source, row in rows:
        item = mapping[row["case_id"]]
        key = row_key({**row, "case_id": item["original"]})
        if item["repeat"] in groups[key]:
            raise ValueError(f"Duplicate repeat: {key} {item['repeat']}")
        groups[key][item["repeat"]] = (source, row)
    result = []
    for key, repeats in sorted(groups.items()):
        ordered = [repeats[number] for number in sorted(repeats)]
        rows_only = [row for _, row in ordered]
        first = rows_only[0]
        scopes = list(json.loads(first["operator_scopes"]))
        result.append({
            "original_case_id": key[0], "variant": key[1], "plans_tile_schedule": key[2],
            "graph_runtime_mode": first["graph_runtime_mode"], "phase": first["phase"],
            "padded_rows": first["padded_rows"], "layer_ids": json.loads(first["global_layer_ids"]),
            "steps_per_row": first["num_steps_sampled"], "repeats": len(rows_only),
            "composed_attention_ms": spread([composed_ms(row, thresholds[source]) for source, row in ordered]),
            "layer_window_median_ms": spread([row["layer_window_stats.median"] for row in rows_only]),
            "scope_median_ms": {scope: spread([row[f"time_stats.{scope}.median"] for row in rows_only])
                                for scope in scopes}})
    return result


def bootstrap(values, rng):
    """p90 relative deviation of the median and mean of n draws from the pooled value."""
    centers = {"median": statistics.median(values), "mean": statistics.mean(values)}
    result = {}
    for size in BOOTSTRAP_SAMPLE_SIZES:
        deviations = {name: [] for name in centers}
        for _ in range(BOOTSTRAP_DRAWS):
            draw = rng.choices(values, k=size)
            deviations["median"].append(abs(statistics.median(draw) / centers["median"] - 1))
            deviations["mean"].append(abs(statistics.mean(draw) / centers["mean"] - 1))
        result[size] = {f"{name}_p90_relative_deviation": statistics.quantiles(values_, n=10)[8]
                        for name, values_ in deviations.items()}
    return {"pooled_samples": len(values), **{f"pooled_{k}_us": v for k, v in centers.items()},
            "by_sample_size": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", type=Path, required=True)
    parser.add_argument("--case-mapping", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = plan_from_spec(args.package_plan)
    mapping = json.loads(args.case_mapping.read_text())
    rows, inputs, thresholds, instances = [], [], {}, defaultdict(list)
    for path in args.raw:
        records, receipt = read_selected(path, None)
        inputs.append(receipt)
        thresholds[str(path)] = int(records.engine["multi_stream_gemm_token_threshold"])
        rows.extend((str(path), row) for row in build_dsv4_replay_rows(plan, [records]))
        for original, items in scope_instances(records, mapping).items():
            instances[original].extend(items)
    rng = random.Random(BOOTSTRAP_SEED)
    kernels, samples = {}, {}
    for original, items in sorted(instances.items()):
        planner_layer = min(item["layer"] for item in items)
        kernels[original] = {name: kernel_level([item for item in items if (item["layer"] == planner_layer) == first])
                             for name, first in (("first_csa_layer", True), ("reuse_csa_layers", False))}
        samples[original] = {
            name: {scope: bootstrap([sum(k["end_us"] - k["start_us"] for k in item["scopes"][scope])
                                     for item in items if (item["layer"] == planner_layer) == first], rng)
                   for scope in (INDEX_STORE, "dsv4_q_projection")}
            for name, first in (("first_csa_layer", True), ("reuse_csa_layers", False))}
    result = {"classification": "DIAGNOSTIC_CSA_STREAM_OVERLAP", "inputs": inputs,
              "package_plan": str(args.package_plan.resolve()), "case_mapping": str(args.case_mapping.resolve()),
              "co_start_us": CO_START_US, "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
              "kernel_level": kernels, "row_level": row_level(rows, mapping, thresholds),
              "sample_size": samples,
              "limits": "One retained probe; fixed geometry per workload; four CSA FULL decode and four "
                        "CSA PIECEWISE workloads. No hardware counter or clock evidence."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for original, groups in kernels.items():
        for name, group in groups.items():
            scope = group["scopes"].get(INDEX_STORE)
            print(f"{original:26s} {name:16s} n={group['instances']:3d} co_start={group['co_start_share']:.2f} "
                  f"index_store cv={scope['coefficient_of_variation']:.3f} "
                  f"by_start={ {k: round(v['median_us'], 2) for k, v in scope['by_store_start'].items()} } "
                  f"last={group['last_ending_scope']}")
    for group in result["row_level"]:
        if group["variant"] != "csa":
            continue
        composed, window = group["composed_attention_ms"], group["layer_window_median_ms"]
        index_store = group["scope_median_ms"].get(INDEX_STORE)
        query = group["scope_median_ms"].get("dsv4_q_projection")
        print(f"{group['original_case_id']:26s} {group['graph_runtime_mode']:9s} first={group['plans_tile_schedule']!s:5s} "
              f"composed cv={composed['coefficient_of_variation']:.3f} split={composed['split_relative_difference']:.3f} | "
              f"window cv={window['coefficient_of_variation']:.3f} | "
              f"index_store cv={index_store['coefficient_of_variation']:.3f} split={index_store['split_relative_difference']:.3f} | "
              f"q_projection cv={query['coefficient_of_variation']:.3f}")


if __name__ == "__main__":
    main()
