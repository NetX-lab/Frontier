"""Attribute existing A6 failures to exact feature lookups without retraining.

Replay the recorded folds with production lookup aggregation and admission.
Interpolation requests are counted but never assigned synthetic predictions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, load_dsv4_attention_dataframe,
    train_dsv4_attention_models,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import _build_exact_feature_lookup
from frontier.types import ClusterType, MeasurementType
from v4_attention_heldout import A6_P90_LIMIT, build_p2_config, row_features


class InterpolationRequired(Exception):
    """The existing fold requires an estimator rather than an exact lookup."""


def measured_lookup(**kwargs):
    def predict(_):
        raise InterpolationRequired

    return SimpleNamespace(
        _frontier_feature_names=tuple(kwargs["feature_cols"]),
        _frontier_exact_lookup=_build_exact_feature_lookup(
            kwargs["df"], kwargs["feature_cols"], kwargs["target_col"]),
        predict=predict,
    )


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
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()
    if table_sha != recorded["input_table"]["sha256"]:
        raise ValueError("The table differs from the recorded A6 input")
    config, binding = build_p2_config(args.scratch_dir.resolve())
    if binding != recorded["config"]:
        raise ValueError("The current case binding differs from the recorded A6 configuration")
    cluster = config.get_clusters()[ClusterType.DECODE]
    replica = cluster.replica_config
    frame = load_dsv4_attention_dataframe(str(table), replica.model_config, replica,
        cluster.replica_scheduler_config.block_size, MeasurementType.KERNEL_ONLY)
    frame = frame.reset_index(drop=True)
    units = recorded["attention"]["fold_assignment"]
    row_folds = {index: unit["fold"] for unit in units for index in unit["row_indices"]}
    assert set(row_folds) == set(frame.index)
    scopes = [scope for scope, value in recorded["attention"]["per_scope"].items()
              if value["joint"]["verdict"] == "FAIL"]
    rows_by_scope = defaultdict(list)
    rejected, interpolated = defaultdict(int), defaultdict(int)
    for fold in range(recorded["arguments"]["folds"]):
        test_indices = [i for i, assigned in row_folds.items() if assigned == fold]
        fitted = train_dsv4_attention_models(frame.drop(index=test_indices), measured_lookup)
        predictor = DeepSeekV4AttentionPredictor(fitted)
        for index in test_indices:
            row = frame.loc[index]
            features = row_features(row)
            for scope in scopes:
                if scope not in json.loads(row.operator_scopes):
                    continue
                try:
                    predicted = predictor.scope_times(features, variant=row.attention_variant,
                        graph_runtime_mode=row.graph_runtime_mode,
                        plans_tile_schedule=bool(row.plans_tile_schedule), scopes=[scope]).cost_ms[scope]
                except DeepSeekV4CoverageError:
                    rejected[scope] += 1
                    continue
                except InterpolationRequired:
                    interpolated[scope] += 1
                    continue
                actual = float(row[f"time_stats.{scope}.median"])
                rows_by_scope[scope].append(dict(
                    row_index=index, case_id=row.case_id, fold=fold, variant=row.attention_variant,
                    graph_runtime_mode=row.graph_runtime_mode, phase=row.phase,
                    padded_rows=int(row.padded_rows), source_revision=row.source_revision,
                    measured_us=1000 * actual, predicted_us=1000 * predicted,
                    absolute_error_us=1000 * abs(predicted - actual),
                    relative_error=abs(predicted - actual) / abs(actual),
                ))
        print(f"fold {fold + 1} exact admission complete", flush=True)
    output = dict(input_result=str(args.result.resolve()),
                  input_result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),
                  input_table=recorded["input_table"], classification="DIAGNOSTIC_ATTRIBUTION",
                  scope_results={})
    for scope in scopes:
        rows = rows_by_scope[scope]
        expected = recorded["attention"]["per_scope"][scope]["joint"]
        assert len(rows) == expected["exact_lookup_count"], scope
        assert rejected[scope] == expected["not_evaluable_count"], scope
        assert interpolated[scope] == expected["interpolated_count"], scope
        errors = np.array([row["relative_error"] for row in rows])
        median, p90 = np.quantile(errors, [0.5, 0.9])
        if not interpolated[scope]:
            reference = expected["kernel_time_relative_error"]
            assert np.isclose(median, reference["median_absolute"], rtol=1e-10, atol=1e-12), scope
            assert np.isclose(p90, reference["p90_absolute"], rtol=1e-10, atol=1e-12), scope
        grouped = defaultdict(list)
        for row in rows:
            key = (row["variant"], row["graph_runtime_mode"], row["phase"], row["padded_rows"])
            grouped[key].append(row)
        output["scope_results"][scope] = dict(
            exact_count=len(rows), rejected_count=rejected[scope], interpolation_count=interpolated[scope],
            median_absolute_relative_error=float(median), p90_absolute_relative_error=float(p90),
            exact_rows_above_10_percent=int((errors > A6_P90_LIMIT).sum()),
            exact_failure_fraction_of_all_evaluable=float((errors > A6_P90_LIMIT).sum() / expected["evaluable_count"]),
            groups=[dict(key=key, count=len(values), above_10_percent=sum(r["relative_error"] > A6_P90_LIMIT for r in values),
                         median_absolute_relative_error=float(np.median([r["relative_error"] for r in values])),
                         p90_absolute_relative_error=float(np.quantile([r["relative_error"] for r in values], 0.9)))
                    for key, values in sorted(grouped.items())],
            rows=rows,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("PASS: counts match A6; exact-only scope statistics match recorded results")


if __name__ == "__main__":
    main()
