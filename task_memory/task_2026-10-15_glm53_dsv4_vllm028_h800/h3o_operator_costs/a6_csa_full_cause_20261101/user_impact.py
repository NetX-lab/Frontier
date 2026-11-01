"""Propagate attention timing errors to the forward time a same-configuration vLLM comparison sees.

Diagnostic only; no table, threshold, simulator path or verdict changes. Three inputs:

1. The recorded phase-level held-out rows, rolled up over the model's 43 layers per workload.
2. The retained P2-32 and P2-64 replays (per-forward records): how much of a real decode forward
   the varying scopes occupy.
3. The retained 16-repeat probe: layer elapsed time when the two compress-store kernels start
   together versus staggered, mean versus median, and whole-forward attention repeatability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import resource
import statistics

import numpy as np

from frontier.attention.deepseek_v4 import deepseek_v4_tile_schedule_planners
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT
from frontier.profiling.kernel_gap.dsv4_replay_gaps import (
    activity_gaps, attention_gaps_by_layer, build_layer_calls, layer_call_timing,
)
from v4_attention_coverage import plan_from_spec
from v4_attention_repeatability import read_selected

RATIO_VARIANT = {0: "uncompressed", 4: "csa", 128: "hca"}
INDEX_STORE, COMPRESSOR_STORE = "dsv4_index_compressor_fused_compress_store", "dsv4_compressor_fused_compress_store"
VARYING_SCOPES = ("dsv4_q_projection", "dsv4_index_fused_query_prepare", COMPRESSOR_STORE, INDEX_STORE)
CO_START_US = 1.0


def layer_counts(model_config_path, layers):
    ratios = json.loads(model_config_path.read_text())["compress_ratios"][:layers]
    variants = tuple(RATIO_VARIANT[ratio] for ratio in ratios)
    return Counter(zip(variants, deepseek_v4_tile_schedule_planners(variants)))


def error_summary(errors):
    values = np.asarray(errors)
    return {"count": len(values), "median_absolute": float(np.median(np.abs(values))),
            "p90_absolute": float(np.percentile(np.abs(values), 90)), "maximum_absolute": float(np.abs(values).max()),
            "signed_mean": float(values.mean()), "signed_median": float(np.median(values))}


def forward_heldout(rows, counts):
    workloads = defaultdict(dict)
    for row in rows:
        workloads[(row["case_id"], row["key"][1], row["key"][2], row["padded_rows"])][
            (row["key"][0], row["key"][3])] = row
    groups = defaultdict(list)
    complete, incomplete = 0, 0
    for (case_id, mode, phase, padded), keyed in workloads.items():
        if set(keyed) != set(counts):
            incomplete += 1
            continue
        complete += 1
        predicted = sum(counts[key] * keyed[key]["predicted_composed_us"] for key in counts)
        measured = sum(counts[key] * keyed[key]["measured_window_us"] for key in counts)
        groups[(mode, phase)].append((predicted / measured - 1, measured, padded))
    result = {"workloads_complete": complete, "workloads_incomplete": incomplete, "groups": {}}
    for (mode, phase), items in sorted(groups.items()):
        by_rows = defaultdict(list)
        for _, measured, padded in items:
            by_rows[padded].append(measured / 1000)
        result["groups"][f"{mode}/{phase}"] = {
            **error_summary([item[0] for item in items]),
            "measured_attention_ms_by_padded_rows": {
                str(padded): statistics.median(values) for padded, values in sorted(by_rows.items())}}
    return result


def replay_context(path, attention_scopes, per_scope_p90):
    totals = {"forwards": 0, "composed_ms": 0.0, "attention_scope_sum_ms": 0.0}
    scope_sums, modes, tokens = defaultdict(float), Counter(), Counter()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("phase") != "serving" or record["role"] != "decode" or not record["real"]:
                continue
            scopes = record.get("scope_ms", {})
            totals["forwards"] += 1
            totals["composed_ms"] += record["composed_ms"]
            totals["attention_scope_sum_ms"] += sum(v for k, v in scopes.items() if k in attention_scopes)
            modes[record["mode"]] += 1
            tokens[record["local_tokens"]] += 1
            for scope in VARYING_SCOPES:
                scope_sums[scope] += scopes.get(scope, 0.0)
    count = totals["forwards"]
    composed = totals["composed_ms"] / count
    scopes = {scope: value / count for scope, value in scope_sums.items()}
    bound = sum(scopes[scope] * per_scope_p90[scope] for scope in VARYING_SCOPES)
    return {"decode_real_forwards": count, "mean_composed_ms": composed, "mode_counts": dict(modes),
            "local_tokens": {str(k): v for k, v in sorted(tokens.items())},
            "mean_attention_scope_sum_ms": totals["attention_scope_sum_ms"] / count,
            "mean_varying_scope_ms": scopes, "varying_scope_share_of_forward": sum(scopes.values()) / composed,
            "bound_all_scopes_wrong_by_their_p90_ms": bound, "bound_share_of_forward": bound / composed}


def probe_instances(raw_paths, plan, mapping):
    instances = defaultdict(lambda: defaultdict(list))
    for path in raw_paths:
        records, _ = read_selected(path, None)
        by_layer = attention_gaps_by_layer(activity_gaps((records,)))
        for call in build_layer_calls((records,)):
            case_id, step = call.step["case_id"], call.step["step_index"]
            timing = layer_call_timing(by_layer.get((call.source, case_id, step, call.global_layer_id), ()))
            if timing.exclusion_reason is not None:
                raise ValueError(f"{case_id} layer {call.global_layer_id}: {timing.exclusion_reason}")
            starts = {}
            for row in call.intervals:
                if row["activity"] == "kernel" and row["scope"] in (INDEX_STORE, COMPRESSOR_STORE):
                    starts[row["scope"]] = min(starts.get(row["scope"], row["start_us"]), row["start_us"])
            offset = starts[COMPRESSOR_STORE] - starts[INDEX_STORE] if len(starts) == 2 else None
            item = mapping[case_id]
            key = (call.step["cudagraph_dispatch"]["cg_mode"], call.variant,
                   plan.layer(call.global_layer_id).plans_tile_schedule)
            instances[item["original"]][key].append(
                {"repeat": item["repeat"], "window_ms": timing.window_ms, "offset_us": offset})
    return instances


def mode_sensitivity(instances, counts):
    result = {}
    for original, keyed in sorted(instances.items()):
        mode = next(iter(keyed))[0]
        groups = {}
        delta_forward_ms = 0.0
        for planner in (True, False):
            items = keyed.get((mode, "csa", planner))
            if not items or items[0]["offset_us"] is None:
                continue
            together = [i["window_ms"] for i in items if abs(i["offset_us"]) < CO_START_US]
            staggered = [i["window_ms"] for i in items if abs(i["offset_us"]) >= CO_START_US]
            entry = {"instances": len(items), "together_share": len(together) / len(items)}
            if together and staggered:
                entry |= {"together_median_ms": statistics.median(together),
                          "staggered_median_ms": statistics.median(staggered),
                          "together_over_staggered": statistics.median(together) / statistics.median(staggered) - 1}
                delta_forward_ms += counts[("csa", planner)] * (
                    statistics.median(together) - statistics.median(staggered))
            groups["first_csa_layer" if planner else "reuse_csa_layers"] = entry
        repeats = sorted({i["repeat"] for items in keyed.values() for i in items})
        forward = []
        for repeat in repeats:
            total = 0.0
            for (_, variant, planner), items in keyed.items():
                total += counts[(variant, planner)] * statistics.median(
                    i["window_ms"] for i in items if i["repeat"] == repeat)
            forward.append(total)
        windows = [i["window_ms"] for items in keyed.values() for i in items]
        mean_over_median = {}
        for key, items in keyed.items():
            values = [i["window_ms"] for i in items]
            mean_over_median["/".join(map(str, key[1:]))] = statistics.mean(values) / statistics.median(values) - 1
        result[original] = {
            "mode": mode, "layer_samples": len(windows), "csa_groups": groups,
            "forward_attention_ms": {"repeats": len(forward), "median": statistics.median(forward),
                                     "mean": statistics.mean(forward),
                                     "coefficient_of_variation": statistics.stdev(forward) / statistics.mean(forward),
                                     "mean_over_median": statistics.mean(forward) / statistics.median(forward) - 1},
            "layer_window_mean_over_median_by_key": mean_over_median,
            "forward_difference_all_together_vs_all_staggered_ms": delta_forward_ms,
            "forward_difference_relative": delta_forward_ms / statistics.median(forward)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-heldout", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--replay", type=Path, nargs="+", required=True)
    parser.add_argument("--package-plan", type=Path, required=True)
    parser.add_argument("--case-mapping", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    counts = layer_counts(args.model_config, args.layers)
    heldout = json.loads(args.phase_heldout.read_text())
    attention_scopes = {op.name for ops in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT.values() for op in ops}
    by_key = {tuple(item["key"]): item["all"]["scopes"] for item in heldout["by_key"]}
    per_scope_p90 = {scope: max(by_key[("csa", "FULL", "decode", first)][scope]["p90_absolute_relative_error"]
                                for first in (True, False)) for scope in VARYING_SCOPES}
    plan = plan_from_spec(args.package_plan)
    mapping = json.loads(args.case_mapping.read_text())
    instances = probe_instances(args.raw, plan, mapping)
    result = {"classification": "DIAGNOSTIC_USER_VISIBLE_IMPACT",
              "layer_counts": {f"{variant}/{'first' if first else 'reuse'}": count
                               for (variant, first), count in sorted(counts.items())},
              "per_scope_p90_csa_full_decode": per_scope_p90,
              "forward_heldout": forward_heldout(heldout["rows"], counts),
              "replays": {str(path): replay_context(path, attention_scopes, per_scope_p90) for path in args.replay},
              "probe": mode_sensitivity(instances, counts),
              "limits": "Forward attention totals use the sampled layers' row values weighted by the model's layer "
                        "counts. One probe on one H200 with eight workloads. No native per-forward comparison."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("layer_counts", "per_scope_p90_csa_full_decode", "forward_heldout")}, indent=1))
    for path, item in result["replays"].items():
        print(path, json.dumps({k: v for k, v in item.items() if k != "local_tokens"}))
        print(sorted(item["local_tokens"].items(), key=lambda kv: -kv[1])[:8])
    print(json.dumps(result["probe"], indent=1))


if __name__ == "__main__":
    main()
