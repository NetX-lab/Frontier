"""Qualify A6 full-forward attention on the recorded folds (D-A6-FORWARD-ATTENTION).

Each complete workload weights predicted layer phases and measured layer windows
by the configured model's (variant, plans_tile_schedule) counts. Verdicts are grouped
by (graph_runtime_mode, phase), with the unchanged median 5% / p90 10% limits.
Every scope and layer phase remains diagnostic, including the old composed-median
reference and exact-lookup subset. Coverage omissions never qualify a workload.

This attention qualification does not change the predictor or the formal E2E gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError, deepseek_v4_tile_schedule_planners
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.dsv4_predictor import (
    TARGET_TIME, DeepSeekV4AttentionPredictor, dsv4_feature_columns, dsv4_model_key, dsv4_model_name,
    load_dsv4_attention_dataframe, train_dsv4_attention_models,
)
from frontier.types import ClusterType, MeasurementType
from v4_attention_heldout import (
    A6_MEDIAN_LIMIT, A6_P90_LIMIT, assign_units, build_p2_config, manager_trainer, row_features,
)


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


def signed_relative_error(predicted, measured):
    return (predicted - measured) / measured if measured != 0 else None


def attention_row_diagnostics(row, predicted, threshold):
    """Retain both phase references and errors for every present scope."""
    scopes = list(json.loads(row.operator_scopes))
    costs = {scope: float(row[f"time_stats.{scope}.median"]) for scope in scopes}
    gaps = {scope: float(row[f"gap_stats.{scope}.median"]) for scope in scopes}
    measured = compose(row, costs, gaps, threshold)
    phase = compose(row, dict(predicted.cost_ms), dict(predicted.gap_ms), threshold)
    window = float(row["layer_window_stats.median"])
    error = signed_relative_error(phase, measured)
    window_error = signed_relative_error(phase, window)
    scope_errors = {scope: signed_relative_error(predicted.cost_ms[scope], costs[scope]) for scope in scopes}
    return {
        "measured_composed_us": 1000 * measured, "predicted_composed_us": 1000 * phase,
        "measured_window_us": 1000 * window,
        "relative_error": abs(error) if error is not None else None,
        "signed_relative_error": error,
        "window_relative_error": abs(window_error) if window_error is not None else None,
        "window_signed_relative_error": window_error,
        "scope_relative_errors": {scope: abs(value) if value is not None else None
                                  for scope, value in scope_errors.items()},
        "scope_signed_relative_errors": scope_errors,
        "scope_gap_signed_errors_us": {scope: 1000 * (predicted.gap_ms[scope] - gaps[scope])
                                       for scope in scopes},
    }


def summarize(items, present_scopes=()):
    scopes = sorted(set(present_scopes) | {scope for item in items for scope in item["scope_relative_errors"]})
    return {"phase_vs_composed": statistics_of([i["relative_error"] for i in items],
                                               [i["signed_relative_error"] for i in items]),
            "phase_vs_window": statistics_of([i["window_relative_error"] for i in items],
                                             [i["window_signed_relative_error"] for i in items]),
            "scopes": {scope: statistics_of(
                [i["scope_relative_errors"][scope] for i in items if scope in i["scope_relative_errors"]],
                [i["scope_signed_relative_errors"][scope] for i in items if scope in i["scope_relative_errors"]])
                for scope in scopes},
            "scope_gap_errors_us": {scope: {
                "count": len(values), "signed_median": float(np.median(values)),
                "median_absolute": float(np.median(np.abs(values))),
                "p90_absolute": float(np.quantile(np.abs(values), 0.9)),
            } for scope in scopes
                if (values := [i["scope_gap_signed_errors_us"][scope] for i in items
                               if scope in i["scope_gap_signed_errors_us"]])}}


def statistics_of(errors, signed_errors=None):
    undefined = sum(value is None for value in errors)
    errors = np.asarray([value for value in errors if value is not None], dtype=float)
    if not np.isfinite(errors).all():
        raise ValueError("Attention errors must be finite")
    median, p90 = np.quantile(np.abs(errors), [0.5, 0.9]) if errors.size else (None, None)
    within = bool(errors.size and median <= A6_MEDIAN_LIMIT and p90 <= A6_P90_LIMIT)
    signed = [value for value in signed_errors if value is not None] if signed_errors is not None else []
    return {"count": int(errors.size), "undefined_relative_error_count": undefined,
            "median_absolute_relative_error": float(median) if median is not None else None,
            "p90_absolute_relative_error": float(p90) if p90 is not None else None,
            "rows_above_10_percent": int((np.abs(errors) > A6_P90_LIMIT).sum()),
            "signed_median_relative_error": float(np.median(signed)) if signed else None,
            "signed_mean_relative_error": float(np.mean(signed)) if signed else None,
            "within_a6_limits": within, "verdict": ("PASS" if within else "FAIL") if errors.size else "NOT_EVALUABLE"}


def model_layer_counts(model_config):
    """Use the model specs and serving tile-schedule owner, not sampled layer counts."""
    variants = tuple(spec.variant_id for spec in model_config.get_layer_attention_specs())
    if not variants or len(variants) != model_config.num_layers:
        raise ValueError("Attention specs must cover every configured model layer")
    return Counter(zip(variants, deepseek_v4_tile_schedule_planners(variants)))


def aggregate_forward_attention(rows, model_config, fold_assignment):
    """Aggregate retained phase JSON rows without training.

    ``fold_assignment`` is the original A6 ``attention.fold_assignment``: it owns
    canonical workload identity and includes workloads with no admitted row. Each
    group must have exactly one row in its recorded fold. Partial, duplicate and
    unexpected groups are NOT_EVALUABLE; invalid fold/row identity raises ValueError.
    Returned times are microseconds and errors are signed prediction-minus-window.
    PASS covers only complete evaluable workloads, never E2E acceptance.
    """
    counts = model_layer_counts(model_config)
    units, unit_by_row, coordinates = {}, {}, set()
    for unit in fold_assignment:
        unit_id = unit["unit_id"]
        coordinate = json.dumps(unit["coordinate"], sort_keys=True)
        if unit_id in units or coordinate in coordinates:
            raise ValueError("Recorded A6 workloads have duplicate canonical identity")
        units[unit_id] = unit
        coordinates.add(coordinate)
        if len(unit["row_indices"]) != unit["row_count"]:
            raise ValueError("Recorded workload row_count differs from row_indices")
        for index in unit["row_indices"]:
            if index in unit_by_row:
                raise ValueError(f"Row {index} has multiple recorded fold assignments")
            unit_by_row[index] = unit
    by_unit, seen_rows = defaultdict(list), set()
    for row in rows:
        index = row["row_index"]
        if index in seen_rows:
            raise ValueError(f"Row {index} appears more than once in the held-out predictions")
        seen_rows.add(index)
        if index not in unit_by_row:
            raise ValueError(f"Row {index} is absent from the recorded A6 folds")
        unit = unit_by_row[index]
        if row["fold"] != unit["fold"]:
            raise ValueError(f"Row {index} crosses its workload's recorded fold")
        coordinate = unit["coordinate"]
        if (row["case_id"] not in unit["case_ids"] or list(row["key"][1:3]) !=
                [coordinate["graph_runtime_mode"], coordinate["phase"]] or
                row["padded_rows"] != coordinate["padded_rows"]):
            raise ValueError(f"Row {index} differs from its recorded workload identity")
        by_unit[unit["unit_id"]].append(row)
    workloads, omitted, groups = [], [], defaultdict(list)
    for unit_id, unit in sorted(units.items()):
        items = by_unit[unit_id]
        keyed = defaultdict(list)
        for row in items:
            keyed[(row["key"][0], row["key"][3])].append(row)
        missing = sorted(set(counts) - set(keyed))
        unexpected = sorted(set(keyed) - set(counts))
        duplicates = sorted(key for key, values in keyed.items() if len(values) != 1)
        invalid_times = [row["row_index"] for row in items
                         if not np.isfinite([row["predicted_composed_us"], row["measured_window_us"]]).all()
                         or row["predicted_composed_us"] < 0 or row["measured_window_us"] <= 0]
        identity = {"unit_id": unit_id, "fold": unit["fold"], "workload": unit["coordinate"],
                    "case_ids": unit["case_ids"], "row_indices": [row["row_index"] for row in items],
                    "source_row_count": unit["row_count"],
                    "expected_layer_groups": len(counts), "present_layer_groups": len(keyed)}
        reasons = []
        if missing:
            reasons.append("missing_layer_groups")
        if unexpected:
            reasons.append("unexpected_layer_groups")
        if duplicates:
            reasons.append("duplicate_layer_groups")
        if unit["row_count"] != len(counts):
            reasons.append("source_layer_row_count_mismatch")
        if invalid_times:
            reasons.append("invalid_layer_window_or_prediction")
        group = (unit["coordinate"]["graph_runtime_mode"], unit["coordinate"]["phase"])
        if reasons:
            omission = {**identity, "verdict": "NOT_EVALUABLE", "reasons": reasons,
                        "missing_layer_groups": [list(key) for key in missing],
                        "unexpected_layer_groups": [list(key) for key in unexpected],
                        "duplicate_layer_groups": [list(key) for key in duplicates],
                        "invalid_timing_row_indices": invalid_times,
                        "missing_row_indices": sorted(set(unit["row_indices"]) - {row["row_index"] for row in items})}
            omitted.append(omission)
            groups[group]  # Preserve groups with no evaluable workload.
            continue
        predicted = sum(count * keyed[key][0]["predicted_composed_us"] for key, count in counts.items())
        measured = sum(count * keyed[key][0]["measured_window_us"] for key, count in counts.items())
        signed = signed_relative_error(predicted, measured)
        item = {**identity, "layer_count": sum(counts.values()), "predicted_attention_us": predicted,
                "measured_attention_us": measured, "signed_error_us": predicted - measured,
                "signed_relative_error": signed, "absolute_relative_error": abs(signed),
                "all_scopes_exact": all(row["all_scopes_exact"] for row in items)}
        workloads.append(item)
        groups[group].append(item)
    by_group = []
    for group, items in sorted(groups.items()):
        omissions = [item for item in omitted if
                     (item["workload"]["graph_runtime_mode"], item["workload"]["phase"]) == group]
        stats = statistics_of([item["absolute_relative_error"] for item in items],
                              [item["signed_relative_error"] for item in items])
        by_group.append({"key": list(group), **stats, "total_workloads": len(items) + len(omissions),
                         "omitted_workloads": len(omissions)})
    verdicts = {item["verdict"] for item in by_group}
    verdict = "FAIL" if "FAIL" in verdicts else ("PASS" if verdicts == {"PASS"} else "NOT_EVALUABLE")
    return {"qualification": "A6_FORWARD_ATTENTION", "attention_verdict": verdict,
            "e2e_acceptance": "NOT_ASSESSED", "verdict_population": "complete_evaluable_workloads_only",
            "reference": "configured_layer_count_weighted_layer_window_stats.median",
            "limits": {"median": A6_MEDIAN_LIMIT, "p90": A6_P90_LIMIT},
            "layer_count": sum(counts.values()),
            "layer_group_counts": [{"key": list(key), "count": count} for key, count in sorted(counts.items())],
            "total_workloads": len(units), "complete_workloads": len(workloads),
            "omitted_workloads": len(omitted),
            "partially_evaluable_workloads": sum(bool(item["row_indices"]) for item in omitted),
            "fully_unevaluable_workloads": sum(not item["row_indices"] for item in omitted),
            "by_group": by_group, "workloads": workloads, "omissions": omitted}


def recorded_row_folds(frame, recorded):
    """Validate the original KFold units before any retraining can leak a workload."""
    arguments = recorded["arguments"]
    units = assign_units(frame, arguments["folds"], arguments["seed"])
    if json.loads(json.dumps(units)) != recorded["attention"]["fold_assignment"]:
        raise ValueError("Recorded A6 assignments differ from the canonical workload KFold split")
    return {index: unit["fold"] for unit in units for index in unit["row_indices"]}


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
    row_folds = recorded_row_folds(frame, recorded)
    rows, skipped, skipped_rows, present_scopes = [], defaultdict(int), [], defaultdict(set)
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
            present_scopes[key].update(scopes)
            features = row_features(row)
            try:
                predicted = predictor.scope_times(features, variant=row.attention_variant,
                    graph_runtime_mode=row.graph_runtime_mode,
                    plans_tile_schedule=bool(row.plans_tile_schedule), scopes=scopes)
            except DeepSeekV4CoverageError as error:
                skipped[key] += 1
                skipped_rows.append({"row_index": int(index), "case_id": row.case_id, "fold": fold,
                                     "key": list(key), "reason": str(error), "scopes": scopes})
                continue
            rows.append({"row_index": int(index), "case_id": row.case_id, "fold": fold, "key": list(key),
                         "padded_rows": int(row.padded_rows),
                         "all_scopes_exact": all(exact_lookup(models, features, row, scope) for scope in scopes),
                         **attention_row_diagnostics(row, predicted, threshold)})
        print(f"fold {fold + 1} complete", flush=True)
    groups = defaultdict(list)
    for item in rows:
        groups[tuple(item["key"])].append(item)
    for key in skipped:
        groups[key]
    by_key = []
    for key, items in sorted(groups.items()):
        exact = [i for i in items if i["all_scopes_exact"]]
        by_key.append({"key": list(key), "all": summarize(items, present_scopes[key]),
                       "coverage_skipped_rows": skipped.get(key, 0),
                       "all_scopes_exact": summarize(exact) if exact else None})
    attention = aggregate_forward_attention(rows, replica.model_config, recorded["attention"]["fold_assignment"])
    output = {"classification": "A6_FORWARD_ATTENTION_QUALIFICATION",
              "attention_verdict": attention["attention_verdict"], "e2e_acceptance": "NOT_ASSESSED",
              "input_result": str(args.result.resolve()),
              "input_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "input_table": recorded["input_table"], "folds": recorded["arguments"],
              "limits": {"median": A6_MEDIAN_LIMIT, "p90": A6_P90_LIMIT},
              "evaluated_rows": len(rows), "total_rows": len(frame),
              "coverage_skipped_rows": [{"key": list(key), "count": count} for key, count in sorted(skipped.items())],
              "coverage_skips": skipped_rows, "attention": attention,
              "layer_phase_classification": "DIAGNOSTIC_ONLY",
              "overall": summarize(rows, set().union(*present_scopes.values())), "by_key": by_key, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"evaluated {len(rows)} of {len(frame)} rows; skipped {sum(skipped.values())} outside coverage")
    print(f"A6 forward attention: {attention['attention_verdict']}; complete {attention['complete_workloads']} / "
          f"{attention['total_workloads']} workloads; E2E acceptance NOT_ASSESSED")
    for entry in attention["by_group"]:
        print(f"{'/'.join(entry['key'])}: {entry['verdict']} n={entry['count']} "
              f"median/p90 {percent(entry['median_absolute_relative_error'])}/"
              f"{percent(entry['p90_absolute_relative_error'])}; omitted {entry['omitted_workloads']}")
    for entry in [{"key": ["overall"], "all": output["overall"], "all_scopes_exact": None}, *by_key]:
        for subset in ("all", "all_scopes_exact"):
            summary = entry[subset]
            if summary is None:
                continue
            phase = summary["phase_vs_composed"]
            cells = " ".join(f"{scope[5:]} {percent(s['median_absolute_relative_error'])}/"
                             f"{percent(s['p90_absolute_relative_error'])}" for scope, s in summary["scopes"].items())
            print(f"{'/'.join(map(str, entry['key'])):38s} {subset:16s} n={phase['count']:4d} phase "
                  f"{percent(phase['median_absolute_relative_error'])}/{percent(phase['p90_absolute_relative_error'])} "
                  f"window p90 {percent(summary['phase_vs_window']['p90_absolute_relative_error'])} | {cells}")


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


if __name__ == "__main__":
    main()
