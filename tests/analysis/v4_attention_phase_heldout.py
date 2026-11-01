"""Evaluate the recorded A6 folds at the attention-phase level the simulator consumes.

The recorded folds are retrained with the A6 production trainer. For every held-out
row, each present scope's kernel time and signed gap are predicted (exact lookup or the
fitted estimator, as in A6), and ``compose_attention_phase`` builds the predicted
attention phase. The reference is the same composition of the row's measured medians
and gaps; the measured layer window is reported as a second reference. Rows outside
coverage are counted and skipped. Rows whose scopes are all exact lookups are also
summarized separately.

Diagnostic only: A6 evaluates each scope separately and its verdicts are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.dsv4_predictor import (
    TARGET_TIME, DeepSeekV4AttentionPredictor, dsv4_feature_columns, dsv4_model_key, dsv4_model_name,
    load_dsv4_attention_dataframe, train_dsv4_attention_models,
)
from frontier.types import ClusterType, MeasurementType
from v4_attention_heldout import (
    A6_MEDIAN_LIMIT, A6_P90_LIMIT, build_p2_config, manager_trainer, row_features,
)

FAILING_SCOPES = ("dsv4_q_projection", "dsv4_index_fused_query_prepare",
                  "dsv4_compressor_fused_compress_store", "dsv4_index_compressor_fused_compress_store")


def compose(row, costs, gaps, threshold):
    return compose_attention_phase(
        costs, gaps, variant=row.attention_variant, graph_runtime_mode=row.graph_runtime_mode,
        rows=int(row.padded_rows), multi_stream_gemm_token_threshold=threshold,
        indexer_short_context=bool(row.indexer_short_context)).elapsed_ms


def exact_lookup(models, features, row, scope):
    key = dsv4_model_key(scope, row.attention_variant, row.graph_runtime_mode, row.phase,
                         bool(row.plans_tile_schedule))
    fitted = models[dsv4_model_name(key, TARGET_TIME)]
    columns = dsv4_feature_columns(scope, row.phase)
    return tuple(float(features.values[column]) for column in columns) in fitted._frontier_exact_lookup


def summarize(items):
    return {"phase_vs_composed": statistics_of([i["relative_error"] for i in items]),
            "phase_vs_window": statistics_of([i["window_relative_error"] for i in items]),
            "scopes": {scope: statistics_of(values) for scope in FAILING_SCOPES
                       if (values := [i["scope_relative_errors"][scope] for i in items
                                      if scope in i["scope_relative_errors"]])}}


def statistics_of(errors):
    errors = np.asarray(errors)
    median, p90 = np.quantile(errors, [0.5, 0.9])
    return {"count": int(errors.size), "median_absolute_relative_error": float(median),
            "p90_absolute_relative_error": float(p90), "rows_above_10_percent": int((errors > A6_P90_LIMIT).sum()),
            "within_a6_limits": bool(median <= A6_MEDIAN_LIMIT and p90 <= A6_P90_LIMIT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recorded = json.loads(args.result.read_text())
    table = Path(recorded["input_table"]["path"])
    if hashlib.sha256(table.read_bytes()).hexdigest() != recorded["input_table"]["sha256"]:
        raise ValueError("The table differs from the recorded A6 input")
    config, binding = build_p2_config(args.scratch_dir.resolve())
    if binding != recorded["config"]:
        raise ValueError("The current case binding differs from the recorded A6 configuration")
    cluster = config.get_clusters()[ClusterType.DECODE]
    replica = cluster.replica_config
    frame = load_dsv4_attention_dataframe(str(table), replica.model_config, replica,
        cluster.replica_scheduler_config.block_size, MeasurementType.KERNEL_ONLY).reset_index(drop=True)
    row_folds = {index: unit["fold"] for unit in recorded["attention"]["fold_assignment"]
                 for index in unit["row_indices"]}
    assert set(row_folds) == set(frame.index)
    rows, skipped = [], defaultdict(int)
    scratch = args.scratch_dir.resolve()
    for fold in range(recorded["arguments"]["folds"]):
        test_indices = [i for i, assigned in row_folds.items() if assigned == fold]
        models = train_dsv4_attention_models(frame.drop(index=test_indices),
                                             manager_trainer(cluster, scratch / f"fold_{fold}"))
        predictor = DeepSeekV4AttentionPredictor(models)
        threshold = predictor.multi_stream_gemm_token_threshold
        for index in test_indices:
            row = frame.loc[index]
            scopes = list(json.loads(row.operator_scopes))
            key = (row.attention_variant, row.graph_runtime_mode, row.phase, bool(row.plans_tile_schedule))
            features = row_features(row)
            try:
                predicted = predictor.scope_times(features, variant=row.attention_variant,
                    graph_runtime_mode=row.graph_runtime_mode,
                    plans_tile_schedule=bool(row.plans_tile_schedule), scopes=scopes)
            except DeepSeekV4CoverageError:
                skipped[key] += 1
                continue
            measured_costs = {scope: float(row[f"time_stats.{scope}.median"]) for scope in scopes}
            measured_gaps = {scope: float(row[f"gap_stats.{scope}.median"]) for scope in scopes}
            measured = compose(row, measured_costs, measured_gaps, threshold)
            phase = compose(row, dict(predicted.cost_ms), dict(predicted.gap_ms), threshold)
            window = float(row["layer_window_stats.median"])
            rows.append({"row_index": int(index), "case_id": row.case_id, "fold": fold, "key": list(key),
                         "padded_rows": int(row.padded_rows),
                         "all_scopes_exact": all(exact_lookup(models, features, row, scope) for scope in scopes),
                         "measured_composed_us": 1000 * measured,
                         "predicted_composed_us": 1000 * phase, "measured_window_us": 1000 * window,
                         "relative_error": abs(phase - measured) / measured,
                         "window_relative_error": abs(phase - window) / window,
                         "scope_relative_errors": {
                             scope: abs(predicted.cost_ms[scope] - measured_costs[scope]) / measured_costs[scope]
                             for scope in FAILING_SCOPES if scope in measured_costs}})
        print(f"fold {fold + 1} complete", flush=True)
    groups = defaultdict(list)
    for item in rows:
        groups[tuple(item["key"])].append(item)
    by_key = []
    for key, items in sorted(groups.items()):
        exact = [i for i in items if i["all_scopes_exact"]]
        by_key.append({"key": list(key), "all": summarize(items),
                       "all_scopes_exact": summarize(exact) if exact else None})
    output = {"classification": "DIAGNOSTIC_PHASE_LEVEL_HELDOUT",
              "input_result": str(args.result.resolve()),
              "input_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "input_table": recorded["input_table"], "folds": recorded["arguments"],
              "limits": {"median": A6_MEDIAN_LIMIT, "p90": A6_P90_LIMIT},
              "evaluated_rows": len(rows), "total_rows": len(frame),
              "coverage_skipped_rows": [{"key": list(key), "count": count} for key, count in sorted(skipped.items())],
              "overall": summarize(rows), "by_key": by_key, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"evaluated {len(rows)} of {len(frame)} rows; skipped {sum(skipped.values())} outside coverage")
    for entry in [{"key": ["overall"], "all": output["overall"], "all_scopes_exact": None}, *by_key]:
        for subset in ("all", "all_scopes_exact"):
            summary = entry[subset]
            if summary is None:
                continue
            phase = summary["phase_vs_composed"]
            cells = " ".join(f"{scope[5:]} {100 * s['median_absolute_relative_error']:.2f}/"
                             f"{100 * s['p90_absolute_relative_error']:.2f}" for scope, s in summary["scopes"].items())
            print(f"{'/'.join(map(str, entry['key'])):38s} {subset:16s} n={phase['count']:4d} phase "
                  f"{100 * phase['median_absolute_relative_error']:5.2f}/{100 * phase['p90_absolute_relative_error']:6.2f}% "
                  f"window p90 {100 * summary['phase_vs_window']['p90_absolute_relative_error']:6.2f}% | {cells}")


if __name__ == "__main__":
    main()
