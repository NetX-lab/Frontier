"""Check copied replay censuses against current and projected G1h union bounds.

Projected rows supply features and scope presence, never synthetic timings to a
serving model. Scope presence is reused from measured rows of the same native
variant/mode/phase/planner/width; all such rows must agree. GPU qualification
remains necessary before these projected bounds become measured coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from frontier.attention.dsv4_features import build_dsv4_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsv4_predictor import (
    TARGET_TIME, TIME_STATS_PREFIX, MEDIAN_STAT, dsv4_model_key, dsv4_model_name,
    load_dsv4_attention_dataframe, train_dsv4_attention_models,
)
from frontier.types import MeasurementType
from v4_attention_coverage import plan_from_spec, prediction_steps

SCOPE_GROUP = ("attention_variant", "graph_runtime_mode", "phase", "plans_tile_schedule", "padded_rows")


def projected_rows(plan, measured: pd.DataFrame, model, *, excluded_case_ids=()) -> pd.DataFrame:
    """Project admitted case features with the existing measured scope inventory."""
    inventories = {}
    for coordinates, rows in measured.groupby(list(SCOPE_GROUP)):
        scopes = {tuple(sorted(json.loads(value))) for value in rows.operator_scopes}
        if len(scopes) != 1:
            raise ValueError(f"Measured scope inventory varies within {coordinates}")
        inventories[coordinates] = scopes.pop()
    layers = {(layer.attention_variant, layer.compress_ratio, layer.plans_tile_schedule)
              for layer in plan.shape.layers}
    projected = []
    for step in prediction_steps(plan):
        if step["case_id"] in excluded_case_ids:
            continue
        requests, dispatch = step["requests"], step["cudagraph_dispatch"]
        for variant, ratio, planner in sorted(layers):
            features = build_dsv4_features(
                [r["num_scheduled_tokens"] for r in requests],
                [r["num_computed_tokens"] for r in requests],
                [r["num_computed_tokens"] >= r["prompt_len"] for r in requests],
                variant=variant, compress_ratio=max(1, ratio), sliding_window=model.sliding_window,
                index_topk=model.index_topk, padded_rows=dispatch["num_tokens"])
            coordinates = (variant, dispatch["cg_mode"], features.phase, planner, dispatch["num_tokens"])
            scopes = inventories[coordinates]
            projected.append({**features.values, **dict(zip(SCOPE_GROUP, coordinates)),
                              "case_id": step["case_id"], "operator_scopes": json.dumps({s: [s] for s in scopes}),
                              **{f"{TIME_STATS_PREFIX}.{s}.{MEDIAN_STAT}": True for s in scopes}})
    return pd.DataFrame(projected)


def model_ranges(rows: pd.DataFrame) -> dict:
    """Use the trainer's own key selection and shared range computation."""
    models = train_dsv4_attention_models(rows, lambda **_: SimpleNamespace())
    return {name: {str(padded): bounds for padded, bounds in model._frontier_dsv4_workload_group_ranges.items()}
            for name, model in models.items()}


def check_requests(records, ranges: dict) -> dict:
    counts = Counter()
    misses, grouped = [], Counter()
    for index, record in enumerate(records, 1):
        counts["requests"] += 1
        counts["calls"] += record["count"]
        violations = []
        for scope in record["scopes"]:
            counts["scope_calls"] += record["count"]
            key = dsv4_model_key(scope, record["variant"], record["graph_runtime_mode"],
                                 record["phase"], record["plans_tile_schedule"])
            name = dsv4_model_name(key, TARGET_TIME)
            padded = str(float(record["values"]["padded_rows"]))
            if name not in ranges:
                reasons = [dict(column="model_key", reason="missing model key", key=key)]
            elif padded not in ranges[name]:
                reasons = [dict(column="padded_rows", value=record["values"]["padded_rows"],
                                reason="unmeasured padded_rows", measured=sorted(ranges[name]))]
            else:
                reasons = [dict(column=column, value=record["values"][column], lower=lower, upper=upper,
                                reason="outside measured group range")
                           for column, (lower, upper) in ranges[name][padded].items()
                           if not lower <= record["values"][column] <= upper]
            if reasons:
                counts["missed_scope_calls"] += record["count"]
                counts["missed_scope_requests"] += 1
                violations.append(dict(model=name, key=key, reasons=reasons))
                for reason in reasons:
                    grouped[(name, padded, reason["column"])] += record["count"]
        if violations:
            counts["missed_requests"] += 1
            counts["missed_calls"] += record["count"]
            misses.append(dict(record_index=index, request=record, violations=violations))
    for name in ("missed_requests", "missed_calls", "missed_scope_requests", "missed_scope_calls"):
        counts.setdefault(name, 0)
    return dict(counts=counts, misses=misses,
                grouped_misses=[dict(model=name, padded_rows=rows, column=column, calls=count)
                                for (name, rows, column), count in sorted(grouped.items())])


def read_requests(root: Path):
    with gzip.open(root / "attention_requests.jsonl.gz", "rt") as handle:
        yield from (json.loads(line) for line in handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--supplement-plan", type=Path, required=True)
    parser.add_argument("--census", type=Path, nargs="+", required=True,
                        help="Read-only copies of the live census roots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = plan_from_spec(args.supplement_plan)
    model = BaseModelConfig.create_from_name(plan.shape.model_name)
    replica = SimpleNamespace(attn_tensor_parallel_size=plan.shape.target_tensor_parallel_size,
                              device="h200", kv_cache_dtype="fp8_ds_mla",
                              moe_gating_routing_runtime_path="balanced_topk")
    measured = load_dsv4_attention_dataframe(str(args.table), model, replica, plan.limits.block_size,
                                             MeasurementType.KERNEL_ONLY)
    spec = json.loads(args.supplement_plan.read_text())
    projected = projected_rows(plan, measured, model, excluded_case_ids=spec["repeat_case_ids"])
    current = model_ranges(measured)
    union = model_ranges(pd.concat([measured, projected], ignore_index=True))
    reports = []
    for root in args.census:
        recorded = json.loads((root / "measured_group_ranges.json").read_text())
        records = list(read_requests(root))
        reports.append(dict(case=root.name, root=str(root),
                            recorded_current=check_requests(records, recorded),
                            recomputed_current=check_requests(records, current),
                            union=check_requests(records, union),
                            current_ranges_equal_recorded=json.loads(json.dumps(current)) == recorded))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(projected_rows=len(projected), current_models=len(current), union_models=len(union), cases=reports)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(projected_rows=len(projected), cases=[
        dict(case=r["case"], current=r["recorded_current"]["counts"], union=r["union"]["counts"],
             current_ranges_equal_recorded=r["current_ranges_equal_recorded"]) for r in reports])))
    return 1 if any(r["union"]["counts"]["missed_requests"] for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
