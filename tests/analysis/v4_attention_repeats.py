"""Compare retained attention measurements across independent profiling runs.

Match workload and layer identities, retaining runtime hashes while ignoring only
source-location strings. Report signed CSA layer-group contrasts without fitting
models, modifying tables, or treating these comparisons as qualification gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import statistics

SCOPES = (
    "dsv4_q_projection",
    "dsv4_index_fused_query_prepare",
    "dsv4_compressor_fused_compress_store",
    "dsv4_index_compressor_fused_compress_store",
)
TIMING_PREFIXES = ("time_stats.", "gap_stats.", "layer_window_stats.")


def identity(row):
    fields = {k: v for k, v in row.items() if not k.startswith(TIMING_PREFIXES)}
    routing = json.loads(fields["moe_routing"])
    routing.pop("origin")
    fields["moe_routing"] = routing
    return json.dumps(fields, sort_keys=True)


def read_rows(path):
    payload = path.read_bytes()
    rows = json.loads(payload)
    keyed = {identity(row): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate workload/layer identities: {path}")
    return keyed, {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                   "rows": len(rows)}


def csa_pairs(rows):
    pairs = {}
    for row in rows:
        if row["attention_variant"] == "csa":
            group = pairs.setdefault(row["case_id"], {})
            planner = row["plans_tile_schedule"]
            if planner in group:
                raise ValueError(f"Duplicate CSA case/group: {row['case_id']}")
            group[planner] = row
    if any(set(pair) != {True, False} for pair in pairs.values()):
        raise ValueError("A CSA case lacks a planner or reuse row")
    return pairs


def quantiles(values):
    if not values:
        return None
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return {"p05": cuts[4], "median": statistics.median(values), "p95": cuts[94]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--published-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    reference, reference_input = read_rows(args.reference)
    with args.published_table.open() as handle:
        published_rows = list(csv.DictReader(handle))
    published = {(r["case_id"], r["attention_variant"], r["plans_tile_schedule"]): r
                 for r in published_rows}
    verified = 0
    for row in reference.values():
        table_row = published[(row["case_id"], row["attention_variant"], str(row["plans_tile_schedule"]))]
        for key, value in row.items():
            if key.startswith("time_stats.") and key.endswith(".median"):
                if abs(float(table_row[key]) - value) > 1e-12:
                    raise ValueError(f"Published median differs: {row['case_id']} {key}")
                verified += 1
    base_pairs = csa_pairs(reference.values())
    output = {"classification": "DIAGNOSTIC_MEASUREMENT_REPEATABILITY", "reference": reference_input,
              "published_table": {"path": str(args.published_table.resolve()),
                                  "sha256": hashlib.sha256(args.published_table.read_bytes()).hexdigest(),
                                  "verified_reference_medians": verified}, "comparisons": []}
    for path in args.comparison:
        comparison, input_record = read_rows(path)
        missing = sorted(set(reference) - set(comparison))
        additional = sorted(set(comparison) - set(reference))
        if missing or additional:
            raise ValueError(f"Identity mismatch: {path}: missing={len(missing)}, additional={len(additional)}")
        count_mismatches = []
        for key, row in reference.items():
            for field, value in row.items():
                if field.startswith("time_stats.") and field.endswith((".count", ".kernel_count")):
                    if comparison[key].get(field) != value:
                        count_mismatches.append([row["case_id"], field])
        if count_mismatches:
            raise ValueError(f"Count mismatch: {path}: {count_mismatches}")
        pairs = csa_pairs(comparison.values())
        result = {"input": input_record, "matched_identities": len(reference),
                  "missing_identities": missing, "additional_identities": additional,
                  "count_mismatches": count_mismatches, "scopes": {}}
        for scope in SCOPES:
            column = f"time_stats.{scope}.median"
            records = []
            for case, pair in base_pairs.items():
                if not all(column in row for row in (*pair.values(), *pairs[case].values())):
                    continue
                a, b = (1000 * pair[planner][column] for planner in (True, False))
                c, d = (1000 * pairs[case][planner][column] for planner in (True, False))
                if min(a, b, c, d) <= 0:
                    raise ValueError(f"Non-positive measured duration: {case} {scope}")
                records.append({"case_id": case, "phase": pair[True]["phase"],
                    "graph_runtime_mode": pair[True]["graph_runtime_mode"],
                    "padded_rows": pair[True]["padded_rows"],
                    "reference_first_us": a, "reference_reuse_us": b,
                    "comparison_first_us": c, "comparison_reuse_us": d,
                    "reference_signed_difference_us": a - b, "comparison_signed_difference_us": c - d,
                    "reference_signed_difference_over_min": (a - b) / min(a, b),
                    "comparison_signed_difference_over_min": (c - d) / min(c, d),
                    "first_run_ratio": c / a, "reuse_run_ratio": d / b})
            large = [r for r in records if abs(r["reference_signed_difference_over_min"]) > 0.1]
            agreement = lambda r: r["reference_signed_difference_us"] * r["comparison_signed_difference_us"] > 0
            result["scopes"][scope] = {
                "pairs": len(records), "same_sign": sum(agreement(r) for r in records),
                "reference_above_10_percent": len(large),
                "large_reference_same_sign": sum(agreement(r) for r in large),
                "large_reference_recur_above_10_percent_same_sign": sum(
                    agreement(r) and abs(r["comparison_signed_difference_over_min"]) > 0.1 for r in large),
                "first_run_ratio": quantiles([r["first_run_ratio"] for r in records]),
                "reuse_run_ratio": quantiles([r["reuse_run_ratio"] for r in records]),
                "rows": records}
            print(path.parent.name, scope, {k: v for k, v in result["scopes"][scope].items()
                                          if k not in {"rows", "first_run_ratio", "reuse_run_ratio"}}, flush=True)
        output["comparisons"].append(result)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"PASS identities, counts and {verified} published medians; measurement contrasts are diagnostic")


if __name__ == "__main__":
    main()
