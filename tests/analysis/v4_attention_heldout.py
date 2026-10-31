"""Design A5/A6 audit of complete V4 replay workloads, using the PD runtime trainer.

Run from the repository root with --table, --folds, --seed, --scratch-dir and
--output-dir. Scratch must belong to the caller; caches never use a run directory.
The result is deterministic and deliberately has no timestamps or wall times.
"""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from frontier.attention.deepseek_v4 import (
    DeepSeekV4CoverageError, build_deepseek_v4_attention_workload,
    deepseek_v4_tile_schedule_planners,
)
from frontier.attention.dsv4_features import (
    DSV4_FEATURE_COLUMNS, DeepSeekV4BatchFeatures, dsv4_feature_columns,
)
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT
from frontier.cc_backend import CCBackendFactory
from frontier.entities import Batch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, THRESHOLD_COLUMN, TARGET_TIME,
    dsv4_model_key, dsv4_model_name, load_dsv4_attention_dataframe,
    train_dsv4_attention_models,
)
from frontier.execution_time_predictor.kernel_gap import read_dsv4_kernel_gap_ms
from frontier.execution_time_predictor.prediction_family_trainers import PredictionFamilyTrainers
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.operators.deepseek_v4_axes import DEEPSEEK_V4_OPERATOR_AXES
from frontier.types import ClusterType, MeasurementType

# design.md section 1, A6: gate the kernel time, not the signed scope gap.
A6_MEDIAN_LIMIT = 0.05
A6_P90_LIMIT = 0.10
UNIT_COLUMNS = (
    "graph_runtime_mode", "phase", "padded_rows", "dispatch_num_requests",
    "prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes",
)
REQUEST_COLUMNS = UNIT_COLUMNS[-3:]
REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800"
DRIVER = TASK / "h4a_p0_diagnostic/h4a4_driver.py"
P2_CASE = "p2_32"


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def workload_coordinate(row):
    """Canonical request lists: formatting and case IDs do not create new units."""
    return tuple(
        None if pd.isna(row[column]) else
        tuple(json.loads(row[column])) if column in REQUEST_COLUMNS else json_value(row[column])
        for column in UNIT_COLUMNS
    )


def assign_units(df, folds, seed):
    grouped = defaultdict(list)
    for index, row in df.iterrows():
        grouped[workload_coordinate(row)].append(int(index))
    units = []
    for coordinate in sorted(grouped, key=lambda value: json.dumps(value)):
        indices = grouped[coordinate]
        units.append(dict(
            unit_id=len(units), coordinate=dict(zip(UNIT_COLUMNS, coordinate)),
            row_indices=indices, row_count=len(indices),
            case_ids=sorted(set(df.loc[indices, "case_id"].astype(str))),
        ))
    for fold, (_, test) in enumerate(KFold(n_splits=folds, shuffle=True, random_state=seed).split(units)):
        for index in test:
            units[index]["fold"] = fold
    return units


def row_features(row):
    columns = ("padded_rows",) if row.phase == "dummy" else tuple(dict.fromkeys(
        column for columns in DSV4_FEATURE_COLUMNS.values() for column in columns
    )) + ("min_query_len", "max_query_len", "min_context_len", "max_context_len", "max_key_len")
    return DeepSeekV4BatchFeatures(row.phase, {column: int(row[column]) for column in columns})


def runtime_scopes(row, model, replica):
    """Use the serving workload owner, including its short-context/dummy branches."""
    if row.phase == "dummy":
        batch = DummyForwardBatch(0, 0, 0)
    else:
        prefill, cached, decode = (json.loads(row[column]) for column in REQUEST_COLUMNS)
        queries, contexts = prefill + [1] * len(decode), cached + decode
        batch = Batch(replica_id=0, requests=[Request(0.0, max(1, q + c), 1)
                      for q, c in zip(queries, contexts)], num_tokens=queries,
                      num_context_tokens=contexts,
                      request_is_decoding=[False] * len(prefill) + [True] * len(decode), is_moe=True)
    # P/D scope selection uses the same owner; the row's phase selects its role.
    role = ClusterType.PREFILL if row.phase == "prefill" else ClusterType.DECODE
    batch._cluster_type = role
    mode, padded = row.graph_runtime_mode, int(row.padded_rows)
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, batch.total_num_tokens,
            padded, max(1, len(batch.requests)), padded,
        )
    variants = tuple(spec.variant_id for spec in model.get_layer_attention_specs())
    planners = deepseek_v4_tile_schedule_planners(variants)
    layer = next(index for index, variant in enumerate(variants)
                 if variant == row.attention_variant and planners[index] == bool(row.plans_tile_schedule))
    workload = build_deepseek_v4_attention_workload(
        model, replica, batch, layer, cluster_type=role,
        execution_mode="eager" if mode == "NONE" else "cuda_graph", prefix_caching=False,
        decode_context_parallel_size=1, prefill_context_parallel_size=1,
    )
    return set(workload["operators"])


class PerColumnRanges:
    """Replay the superseded admission while retaining scope_times prediction code."""

    def __init__(self, groups):
        self.bounds = {"padded_rows": (min(groups), max(groups))}
        for ranges in groups.values():
            for column, (lower, upper) in ranges.items():
                previous = self.bounds.get(column, (lower, upper))
                self.bounds[column] = (min(lower, previous[0]), max(upper, previous[1]))

    def get(self, padded_rows):
        return self.bounds


def per_column_predictor(models):
    legacy = {}
    for name, model in models.items():
        fitted = copy.copy(model)
        fitted._frontier_dsv4_workload_group_ranges = PerColumnRanges(model._frontier_dsv4_workload_group_ranges)
        legacy[name] = fitted
    return DeepSeekV4AttentionPredictor(legacy)


def rejected_column(error):
    match = re.search(r"([a-z_]+)=[^ ]+ outside", str(error))
    if match:
        return match.group(1)
    if "missing for key" in str(error):
        return "model_key"
    raise ValueError(f"Unrecognized admission rejection: {error}") from error


def predict_scope(predictor, features, row, scope):
    try:
        result = predictor.scope_times(features, variant=row.attention_variant,
                                       graph_runtime_mode=row.graph_runtime_mode,
                                       plans_tile_schedule=bool(row.plans_tile_schedule), scopes=[scope])
        return (result.cost_ms[scope], result.gap_ms[scope]), None
    except DeepSeekV4CoverageError as error:
        return None, dict(column=rejected_column(error), message=str(error))


def error_stats(values):
    if not values:
        return dict(count=0, median_absolute=None, p90_absolute=None, signed_median=None)
    array = np.asarray(values)
    return dict(count=len(values), median_absolute=float(np.median(np.abs(array))),
                p90_absolute=float(np.percentile(np.abs(array), 90)), signed_median=float(np.median(array)))


def gate(stats):
    if stats["count"] == 0:
        return "NOT_EVALUABLE"
    return "PASS" if (stats["median_absolute"] <= A6_MEDIAN_LIMIT
                      and stats["p90_absolute"] <= A6_P90_LIMIT) else "FAIL"


def admission_metrics(records, admission):
    time_errors, gap_errors, combined_errors = [], [], []
    rejected, examples = Counter(), {}
    exact_count = combined_zero = 0
    for record in records:
        prediction = record[admission]
        if prediction is None:
            reason = record[admission + "_rejection"]
            rejected[reason["column"]] += 1
            examples.setdefault(reason["column"], dict(
                row_index=record["row_index"], unit_id=record["unit_id"],
                fold=record["fold"], workload=record["workload"], **reason))
            continue
        measured, measured_gap = record["measured"]
        predicted, predicted_gap = prediction
        time_errors.append((predicted - measured) / measured)
        gap_errors.append((predicted_gap - measured_gap) * 1000.0)
        if measured + measured_gap == 0:
            combined_zero += 1
        else:
            combined_errors.append(((predicted + predicted_gap) - (measured + measured_gap))
                                   / (measured + measured_gap))
        exact_count += record["exact_lookup"]
    time = error_stats(time_errors)
    return dict(heldout_count=len(records), evaluable_count=time["count"], exact_lookup_count=exact_count,
                interpolated_count=time["count"] - exact_count,
                not_evaluable_count=sum(rejected.values()), not_evaluable_by_column=dict(sorted(rejected.items())),
                rejection_examples=examples, kernel_time_relative_error=time,
                gap_absolute_error_us=error_stats(gap_errors), combined_time_relative_error=error_stats(combined_errors),
                combined_zero_denominator_count=combined_zero, verdict=gate(time))


def summarize_records(records):
    joint = admission_metrics(records, "joint")
    old = admission_metrics(records, "per_column")
    shared = [record for record in records if record["joint"] is not None and record["per_column"] is not None]
    mismatches = [record for record in shared if record["joint"] != record["per_column"]]
    if any(record["joint"] is not None and record["per_column"] is None for record in records):
        raise AssertionError("Joint admission must be a subset of the old per-column admission")
    return dict(joint=joint, per_column=old,
                shared_admission=dict(count=len(shared), prediction_mismatch_count=len(mismatches),
                                      joint=admission_metrics(shared, "joint"),
                                      per_column=admission_metrics(shared, "per_column")))


def evaluate_attention(df, model, replica, folds, seed, trainer_factory):
    df = df.reset_index(drop=True)
    units = assign_units(df, folds, seed)
    unit_by_row = {index: unit for unit in units for index in unit["row_indices"]}
    legal_zero = Counter()
    zero_mismatches = []
    selections = {}
    for index, row in df.iterrows():
        scopes = set(json.loads(row.operator_scopes))
        selected = runtime_scopes(row, model, replica)
        selections[index] = selected
        family = {operator.name for operator in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[row.attention_variant]}
        absent = family - scopes
        legal_zero.update(absent)
        if selected - scopes:
            zero_mismatches.append(dict(row_index=int(index), case_id=row.case_id,
                                        coordinate=units[unit_by_row[index]["unit_id"]]["coordinate"],
                                        variant=row.attention_variant, planner=bool(row.plans_tile_schedule),
                                        requested_absent_scopes=sorted(selected - scopes)))
        if scopes - selected:
            zero_mismatches.append(dict(row_index=int(index), case_id=row.case_id,
                                        measured_but_unselected_scopes=sorted(scopes - selected)))
    records, fold_reports = [], []
    for fold in range(folds):
        test_indices = sorted(index for unit in units if unit["fold"] == fold for index in unit["row_indices"])
        training = df.drop(index=test_indices)
        test = df.loc[test_indices]
        training_units = {workload_coordinate(row) for _, row in training.iterrows()}
        test_units = {workload_coordinate(row) for _, row in test.iterrows()}
        if training_units & test_units:
            raise AssertionError("A held-out workload appears in the training rows")
        print(f"fold {fold + 1}/{folds}: {len(training_units)} training units, {len(test_units)} held out", flush=True)
        models = train_dsv4_attention_models(training, trainer_factory(fold))
        joint = DeepSeekV4AttentionPredictor(models)
        legacy = per_column_predictor(models)
        fold_reports.append(dict(fold=fold, training_unit_count=len(training_units), test_unit_count=len(test_units),
                                 training_row_count=len(training), test_row_count=len(test), model_count=len(models),
                                 training_test_unit_overlap=0))
        for index, row in test.iterrows():
            features = row_features(row)
            # Only the serving owner selects the scopes to price. Absent scopes
            # contribute exactly zero because the predictor is never asked for them.
            for scope in sorted(selections[index] & set(json.loads(row.operator_scopes))):
                key = dsv4_model_key(scope, row.attention_variant, row.graph_runtime_mode,
                                     row.phase, bool(row.plans_tile_schedule))
                prediction, rejection = predict_scope(joint, features, row, scope)
                old_prediction, old_rejection = predict_scope(legacy, features, row, scope)
                fitted = models.get(dsv4_model_name(key, TARGET_TIME))
                feature_key = tuple(float(features.values[column]) for column in dsv4_feature_columns(scope, row.phase))
                records.append(dict(
                    scope=scope, key=json.dumps(key, separators=(",", ":")), fold=fold, row_index=int(index),
                    unit_id=unit_by_row[index]["unit_id"], workload=unit_by_row[index]["coordinate"],
                    measured=(float(row[f"time_stats.{scope}.median"]), float(row[f"gap_stats.{scope}.median"])),
                    joint=prediction, joint_rejection=rejection,
                    per_column=old_prediction, per_column_rejection=old_rejection,
                    exact_lookup=fitted is not None and feature_key in fitted._frontier_exact_lookup,
                ))
    by_scope, by_key = defaultdict(list), defaultdict(list)
    for record in records:
        by_scope[record["scope"]].append(record)
        by_key[record["key"]].append(record)
    scopes = {}
    for scope in DSV4_FEATURE_COLUMNS:
        summary = summarize_records(by_scope[scope])
        summary["legally_zero_count"] = legal_zero[scope]
        if not by_scope[scope] and legal_zero[scope] and not zero_mismatches:
            summary["joint"]["verdict"] = "LEGALLY_ZERO"
        scopes[scope] = summary
    return dict(
        unit_columns=UNIT_COLUMNS, unit_count=len(units), row_count=len(df),
        unit_size_counts=dict(sorted(Counter(unit["row_count"] for unit in units).items())),
        fold_assignment=units, folds=fold_reports, per_scope=scopes,
        per_key={key: summarize_records(rows) for key, rows in sorted(by_key.items())},
        legally_zero=dict(count=sum(legal_zero.values()), per_scope=dict(sorted(legal_zero.items())),
                          mismatch_count=len(zero_mismatches), mismatches=zero_mismatches,
                          verdict="PASS" if not zero_mismatches else "FAIL"),
        admission_only=dict(shared_count=sum(record["joint"] is not None and record["per_column"] is not None
                                             for record in records),
                            prediction_mismatch_count=sum(record["joint"] != record["per_column"] for record in records
                                                          if record["joint"] is not None and record["per_column"] is not None)),
    )


def config_attribute_reads(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return sorted({node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                   and isinstance(node.value, ast.Name) and node.value.id in ("config", "execution_time_predictor_config")})


def source_reference(function):
    return f"{Path(inspect.getsourcefile(function)).resolve().relative_to(REPO)}:{inspect.getsourcelines(function)[1]}"


def build_p2_config(scratch):
    spec = importlib.util.spec_from_file_location("b19_h4a4", DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    case = {**driver.CASES[P2_CASE], "overrides": {}}
    case.pop("table")
    fields = dict(
        decode_cuda_graph_mode="full_and_piecewise", replica_config_network_device="h200_dgx",
        analytical_kv_cache_transfer_config_nixl_read_completion="engine_step",
        cluster_config_decode_replica_scheduler_config_max_concurrent_batches=2,
        cudagraph_capture_sizes=json.loads((REPO / "tests/fixtures/dsv4_native_dp_sync_rows.json").read_text())["capture_sizes"],
    )
    config, _ = driver.build_config(P2_CASE, case, scratch / "config", fields)
    return config, dict(case=P2_CASE, input_path=str(case["input"]), input_sha256=case["input_sha256"],
                        driver_path=str(DRIVER.relative_to(REPO)), diagnostic_table_override_removed=True,
                        operator_path_fields=fields)


def manager_trainer(cluster, cache):
    """Minimal manager state for its real fit/cache/count path, without simulation init."""
    cache.mkdir(parents=True, exist_ok=True)
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cache_dir = str(cache)
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    manager._trained_models_kernel_only = {}
    manager._models_by_precision_kernel_only = {}
    replica = cluster.replica_config
    predictor_config = cluster.execution_time_predictor_config
    gap_path = predictor_config.kernel_gap_input_file.format(DEVICE=replica.device, MODEL=replica.model_name)
    manager._kernel_gap_input_file = gap_path
    manager._kernel_gap_ms = read_dsv4_kernel_gap_ms(gap_path, replica.attn_tensor_parallel_size)
    return functools.partial(
        manager._train_single_model, execution_time_predictor_config=predictor_config,
        training_context={"cluster_type": str(ClusterType.DECODE), "device": replica.device,
                          "model_name": replica.model_name,
                          "model_architecture_profile": replica.model_config.get_model_architecture_profile().profile_id},
    )


def trainer_metadata(cluster):
    manager = ExecutionTimePredictionModelManager
    fitting = set(config_attribute_reads(PredictionFamilyTrainers._fit_single_model))
    fitting.update(config_attribute_reads(manager._create_estimator_and_params))
    hashing = set(config_attribute_reads(manager._get_hash_relevant_config))
    # getattr in the existing cache identity reads this field too.
    hashing.add("cpu_overhead_kernel_only_input_file")
    config = cluster.execution_time_predictor_config
    # Only the selected predictor type's estimator branch is reachable.
    fitting &= config.__dataclass_fields__.keys()
    hashing &= config.__dataclass_fields__.keys()
    fields = sorted(fitting | hashing)
    return dict(
        runtime_path="Simulator -> train_dsv4_for_manager -> ExecutionTimePredictionModelManager._train_single_model",
        trainer_source=source_reference(PredictionFamilyTrainers._train_single_model),
        fit_source=source_reference(PredictionFamilyTrainers._fit_single_model),
        predictor_type=str(config.get_type()),
        config_fields={field: getattr(config, field) for field in fields},
        estimator_and_cv_fields=sorted(fitting), cache_identity_fields=sorted(hashing),
        scorer_source=source_reference(manager._get_scorer), cache_policy="one private directory per outer fold",
        note="PREFILL and DECODE have the same fitted-model and cache configuration; DECODE trainer context used.",
    )


def collective_heldout(curve):
    """D-A6-COLLECTIVE-UNIT: each interior payload point is left out and interpolated from its neighbours."""
    loo = curve["loo_absolute_relative_error"]
    stats = dict(count=loo["point_count"], median_absolute=loo["median"], p90_absolute=loo["p90"],
                 max_absolute=loo["max"])
    return dict(key=list(curve["key"]), point_count=curve["point_count"], byte_range=list(curve["byte_range"]),
                heldout=stats, verdict=gate(stats))


def family_classification(collective_path, backend):
    from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend, _Curve
    from frontier.execution_time_predictor.dsv4_operator_predictor import DeepSeekV4OperatorPredictor
    from frontier.execution_time_predictor import dsv4_runtime
    from frontier.execution_time_predictor.kernel_gap import _read_gap_rows_ms
    headers = list(pd.read_csv(collective_path, nrows=0).columns)
    return [
        dict(family="standalone V4 operators", prediction="exact_only", operators=sorted(DEEPSEEK_V4_OPERATOR_AXES),
             source=source_reference(DeepSeekV4OperatorPredictor.operator),
             admission_source=source_reference(DeepSeekV4OperatorPredictor._exact)),
        dict(family="MoE gating and routed rank classes", prediction="exact_only",
             sources=[source_reference(DeepSeekV4OperatorPredictor.gate),
                      source_reference(DeepSeekV4OperatorPredictor.grouped_gemm)],
             admission_source=source_reference(DeepSeekV4OperatorPredictor._exact)),
        dict(family="runner preparation, sampling and block rows", prediction="exact_only_or_legal_zero",
             sources=[source_reference(DeepSeekV4OperatorPredictor.runner),
                      source_reference(DeepSeekV4OperatorPredictor.block_table_writes)]),
        dict(family="attention signed scope gaps", prediction="same_admission_and_estimator_as_attention",
             heldout="included in this result; absolute microseconds, not gated"),
        dict(family="non-attention kernel gaps", prediction="exact_segment_constant_times_exact_activity_count",
             sources=[source_reference(_read_gap_rows_ms), source_reference(dsv4_runtime._operator_gaps)],
             heldout="not an interpolating estimator; one measured constant per segment and TP"),
        dict(family="unequal SP collectives", prediction="exact_only",
             source=source_reference(MeasuredCCBackend._predict_exact_sp)),
        dict(family="uniform measured collectives", prediction="interpolating",
             sources=[source_reference(MeasuredCCBackend.predict_operator_collective), source_reference(_Curve.predict_us)],
             input_path=str(collective_path), input_sha256=hashlib.sha256(collective_path.read_bytes()).hexdigest(),
             table_columns=headers,
             heldout_unit="one measured payload point of a curve with all its samples (D-A6-COLLECTIVE-UNIT)",
             heldout_source=source_reference(_Curve.describe),
             curves=[collective_heldout(curve) for curve in backend.describe_curves()]),
    ]


def percent(value):
    return "n/a" if value is None else f"{100 * value:.3f}%"


def markdown_summary(result):
    attention = result["attention"]
    lines = ["# V4 attention A5/A6 held-out result", "",
             f"{attention['row_count']} layer rows in {attention['unit_count']} complete workload units; "
             f"K={result['arguments']['folds']}, seed={result['arguments']['seed']}.", "",
             "Kernel time gates: median absolute relative error <= 5%, p90 <= 10% (design A6). "
             "PASS applies to evaluable rows; NOT_EVALUABLE rows cannot establish accuracy.", "",
             "| Scope | Held out | Evaluable | Joint rejected | Old rejected | Median | p90 | Verdict |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for scope, metrics in attention["per_scope"].items():
        joint, old = metrics["joint"], metrics["per_column"]
        stats = joint["kernel_time_relative_error"]
        lines.append(f"| {scope} | {joint['heldout_count']} | {joint['evaluable_count']} | "
                     f"{joint['not_evaluable_count']} | {old['not_evaluable_count']} | "
                     f"{percent(stats['median_absolute'])} | {percent(stats['p90_absolute'])} | {joint['verdict']} |")
    zero = attention["legally_zero"]
    lines += ["", f"Legally zero: {zero['count']} absent-scope contributions checked; "
              f"{zero['mismatch_count']} runtime selection mismatches ({zero['verdict']}).",
              f"Shared admission: {attention['admission_only']['shared_count']} predictions; "
              f"{attention['admission_only']['prediction_mismatch_count']} value changes.", "",
              "Uniform measured collectives, one payload point left out at a time (D-A6-COLLECTIVE-UNIT):", "",
              "| Curve | Points | Held out | Median | p90 | Max | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    collectives = next(family for family in result["other_families"] if "curves" in family)
    for curve in collectives["curves"]:
        stats = curve["heldout"]
        lines.append(f"| {tuple(curve['key'])} | {curve['point_count']} | {stats['count']} | "
                     f"{percent(stats['median_absolute'])} | {percent(stats['p90_absolute'])} | "
                     f"{percent(stats['max_absolute'])} | {curve['verdict']} |")
    lines += ["",
              "All model-key statistics, signed-gap errors, combined-time errors, rejection columns/examples, "
              "fold assignments, data hashes and trainer fields are in result.json. No timestamps are emitted."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    config, config_receipt = build_p2_config(scratch)
    if args.seed != config.seed:
        raise ValueError(f"P2 input seed is {config.seed}, requested seed is {args.seed}")
    clusters = config.get_clusters()
    cluster = clusters[ClusterType.DECODE]
    metadata = trainer_metadata(cluster)
    prefill_metadata = trainer_metadata(clusters[ClusterType.PREFILL])
    if metadata["config_fields"] != prefill_metadata["config_fields"]:
        raise ValueError("PREFILL and DECODE trainer configurations differ; separate evaluation is required")
    metadata["operator_path_fields_affect_trainer"] = sorted(
        set(config_receipt["operator_path_fields"]) & set(metadata["config_fields"]))
    replica = cluster.replica_config
    df = load_dsv4_attention_dataframe(str(args.table), replica.model_config, replica,
                                      cluster.replica_scheduler_config.block_size, MeasurementType.KERNEL_ONLY)
    attention = evaluate_attention(df, replica.model_config, replica, args.folds, args.seed,
                                   lambda fold: manager_trainer(cluster, scratch / f"fold_{fold}"))
    backend_config = cluster.cc_backend_config
    collective_path = Path(backend_config.summary_csv_path.format(
        profiling_data_dir=backend_config.profiling_data_dir, NETWORK_DEVICE=replica.network_device)).resolve()
    # The same construction as Cluster._initialize_cc_backend for a measured backend.
    backend = CCBackendFactory.create(
        backend_type=backend_config.get_type(), config=backend_config, cluster_type=ClusterType.DECODE,
        device_type=replica.device, network_device=replica.network_device, num_devices=replica.world_size,
    )
    result = dict(
        schema_version="v4_attention_heldout.v1", arguments=dict(folds=args.folds, seed=args.seed),
        gate_thresholds=dict(median=A6_MEDIAN_LIMIT, p90=A6_P90_LIMIT),
        input_table=dict(path=str(args.table.resolve()), sha256=hashlib.sha256(args.table.read_bytes()).hexdigest()),
        source_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     check=True, timeout=30).stdout.strip(),
        source_dirty_paths=subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                           "frontier", "data", "tests"], capture_output=True, text=True,
                                          check=True, timeout=30).stdout.splitlines(),
        source_dirty_path_scope="frontier, data, tests; generated audit outputs are excluded",
        python=sys.version.split()[0], config=config_receipt, trainer=metadata,
        attention=attention, other_families=family_classification(collective_path, backend),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.md").write_text(markdown_summary(result))
    print(f"wrote {args.output_dir / 'result.json'}", flush=True)
    return result


if __name__ == "__main__":
    main()
