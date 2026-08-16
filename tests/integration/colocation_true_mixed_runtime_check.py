"""Validate non-dummy co-location true-mixed scheduling and prediction.

The deterministic traces first admit one request and then admit one or two
prefill requests while the first request is decoding. The resulting batch must
contain one decode request plus the requested prefill count and must dispatch
both ``attn_prefill_mixed`` and ``attn_decode_in_mixed``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd

from frontier.config import SimulationConfig
from frontier.execution_time_predictor import (
    shared_prediction_model_manager as manager_module,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    canonicalize_prediction_key,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType, MeasurementType
from frontier.utils.random import set_seeds


MODEL_NAME = "llama2_7b_dense_example"
DEVICE = "h800"
TARGET_OPERATORS = ("attn_prefill_mixed", "attn_decode_in_mixed")
EXPECTED_REQUEST_LENGTHS = ((32, 16), (31, 2), (31, 2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-mode", choices=("offline", "online"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--expected-prefill-request-count",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument("--expect-fit", choices=("positive", "zero"), required=True)
    parser.add_argument(
        "--expect-cache-state",
        choices=("empty", "populated"),
        required=True,
    )
    parser.add_argument("--first-request-metrics", type=Path)
    return parser.parse_args()


def _predictor_args(args: argparse.Namespace) -> list[str]:
    return [
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_linear_op_input_file",
        str(args.profile_dir / "linear_op.csv"),
        "--random_forrest_execution_time_predictor_config_atten_input_file",
        str(args.profile_dir / "attention.csv"),
        "--random_forrest_execution_time_predictor_config_num_estimators",
        "2",
        "--random_forrest_execution_time_predictor_config_max_depth",
        "4",
        "--random_forrest_execution_time_predictor_config_min_samples_split",
        "2",
        "--random_forrest_execution_time_predictor_config_k_fold_cv_splits",
        "2",
        "--random_forrest_execution_time_predictor_config_num_training_job_threads",
        "1",
        "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",
        "--random_forrest_execution_time_predictor_config_kv_cache_prediction_granularity",
        "64",
        "--random_forrest_execution_time_predictor_config_prediction_min_kv_cache_size",
        "0",
        "--random_forrest_execution_time_predictor_config_prediction_max_batch_size",
        "8",
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        "4096",
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        "4096",
        "--random_forrest_execution_time_predictor_config_enable_prediction_domain_diagnostics",
    ]


def _cli_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
        "--simulation_mode",
        args.simulation_mode,
        "--sys_arch",
        "co-location",
        "--no-enable_parallel_clusters",
        "--cc_backend_config_type",
        "analytical",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_device",
        DEVICE,
        "--replica_config_model_name",
        MODEL_NAME,
        "--replica_config_num_pipeline_stages",
        "1",
        "--replica_config_attn_tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--replica_config_attn_data_parallel_size",
        "1",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "none",
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "64",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks",
        "512",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type",
        "trace_replay",
        "--trace_request_generator_config_trace_file",
        str(args.trace_file),
        "--trace_request_generator_config_max_tokens",
        "4096",
        "--metrics_config_output_dir",
        str(args.metrics_dir),
        "--metrics_config_cache_dir",
        str(args.cache_dir),
        "--metrics_config_run_id",
        args.run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        *_predictor_args(args),
    ]
    if args.simulation_mode == "offline":
        cli_args.append("--offline_use_generated_request_arrivals")
    return cli_args


def _cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.pkl")) if cache_dir.exists() else []


def _profile_scalar(model: object, field: str) -> object:
    binding = getattr(model, "_frontier_operator_binding", None)
    if not isinstance(binding, dict):
        raise AssertionError("persisted model lacks operator binding metadata")
    profile = binding.get("profile_structure")
    if not isinstance(profile, dict) or field not in profile:
        raise AssertionError(f"operator binding lacks profile field {field!r}")
    value = profile[field]
    if isinstance(value, list):
        raise AssertionError(f"profile field {field!r} is not scalar: {value!r}")
    return value


def _assert_model_identity(
    model: object,
    operator_name: str,
    tensor_parallel_size: int,
) -> dict[str, object]:
    binding = getattr(model, "_frontier_operator_binding", None)
    if not isinstance(binding, dict):
        raise AssertionError(f"{operator_name} lacks operator binding")
    expected = {
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "operator_family": "attention",
        "operator_name": operator_name,
    }
    for field, expected_value in expected.items():
        if binding.get(field) != expected_value:
            raise AssertionError(
                f"{operator_name} binding {field}={binding.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    actual_tp = int(_profile_scalar(model, "num_tensor_parallel_workers"))
    if actual_tp != tensor_parallel_size:
        raise AssertionError(
            f"{operator_name} uses TP={actual_tp}, expected {tensor_parallel_size}"
        )
    measurement_type = str(_profile_scalar(model, "measurement_type"))
    if measurement_type != "CUDA_EVENT":
        raise AssertionError(
            f"{operator_name} measurement family is {measurement_type!r}"
        )
    feature_domain = getattr(model, "_frontier_feature_domain", None)
    if not isinstance(feature_domain, dict):
        raise AssertionError(f"{operator_name} lacks feature-domain metadata")
    if feature_domain.get("runtime_prediction_policy") != "allow_model_prediction":
        raise AssertionError(
            f"{operator_name} does not allow legal model prediction"
        )
    return {
        "model_hash": str(getattr(model, "_frontier_model_hash")),
        "device": binding["device"],
        "model_name": binding["model_name"],
        "operator_family": binding["operator_family"],
        "operator_name": binding["operator_name"],
        "tensor_parallel_size": actual_tp,
        "measurement_type": measurement_type,
        "feature_names": list(getattr(model, "_frontier_feature_names")),
        "exact_lookup_rows": len(getattr(model, "_frontier_exact_lookup", {})),
        "domain_kind": feature_domain.get("domain_kind"),
        "runtime_prediction_policy": feature_domain.get(
            "runtime_prediction_policy"
        ),
    }


def _assert_monolithic_artifacts(
    simulator: Simulator,
    tensor_parallel_size: int,
) -> tuple[SklearnExecutionTimePredictor, dict[str, object]]:
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    if predictor._cluster_type != ClusterType.MONOLITHIC:
        raise AssertionError(
            f"co-location predictor has cluster type {predictor._cluster_type}"
        )
    if predictor._model_manager is not None:
        raise AssertionError(
            "co-location must own its predictor artifacts directly, not through "
            "a disaggregated shared manager"
        )
    eager_models = predictor._models_eager
    summaries: dict[str, object] = {}
    for operator_name in TARGET_OPERATORS:
        if operator_name not in eager_models:
            raise AssertionError(f"MONOLITHIC eager models lack {operator_name}")
        summaries[operator_name] = _assert_model_identity(
            eager_models[operator_name],
            operator_name,
            tensor_parallel_size,
        )
    return predictor, summaries


def _source_before_on_demand_call(
    predictor: SklearnExecutionTimePredictor,
    model_name: str,
    features: dict[str, float],
) -> tuple[str, str, tuple[int | float, ...]]:
    model_info = predictor._predictions[model_name]
    feature_names = tuple(model_info["_feature_names"])
    feature_key = canonicalize_prediction_key(
        tuple(features[name] for name in feature_names)
    )
    family_name = predictor._measurement_family_name(
        predictor._active_measurement_type
    )
    exact_lookup = model_info.get("_exact_lookup") or {}
    runtime_cache = predictor._runtime_cache[family_name][model_name]
    if feature_key in exact_lookup:
        source = "exact_lookup"
    elif feature_key in runtime_cache:
        source = "runtime_cache"
    else:
        source = "model_prediction"
    return source, family_name, feature_key


def _find_request_metrics(metrics_root: Path) -> Path:
    matches = sorted(metrics_root.rglob("request_metrics.csv"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one request_metrics.csv below {metrics_root}, got {matches}"
        )
    return matches[0]


def _read_and_validate_request_metrics(
    path: Path,
    expected_prefill_request_count: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    expected_request_lengths = (
        EXPECTED_REQUEST_LENGTHS[: 1 + expected_prefill_request_count]
    )
    frame = pd.read_csv(path).sort_values("Request Id").reset_index(drop=True)
    if len(frame) != len(expected_request_lengths):
        raise AssertionError(
            f"expected {len(expected_request_lengths)} request rows, got {len(frame)}"
        )
    actual_lengths = tuple(
        (int(row.request_num_prefill_tokens), int(row.request_num_decode_tokens))
        for row in frame.itertuples(index=False)
    )
    if actual_lengths != expected_request_lengths:
        raise AssertionError(
            f"request length matrix={actual_lengths}, expected={expected_request_lengths}"
        )
    key_fields = (
        "request_e2e_time",
        "request_execution_time",
        "ttft",
        "cluster_prefill_computation",
        "cluster_decode_computation",
    )
    key_metrics: dict[str, list[float]] = {}
    for field in key_fields:
        values = [float(value) for value in frame[field]]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise AssertionError(f"{field} must contain finite non-negative values")
        key_metrics[field] = values
    return frame, {
        "row_count": len(frame),
        "request_lengths": [list(values) for values in actual_lengths],
        "key_metrics_ms": key_metrics,
    }


def _compare_request_metrics(
    first_path: Path,
    reload_frame: pd.DataFrame,
) -> dict[str, object]:
    first_frame = pd.read_csv(first_path).sort_values("Request Id").reset_index(drop=True)
    if len(first_frame) != len(reload_frame):
        raise AssertionError(
            f"first/reload row count differs: {len(first_frame)} != {len(reload_frame)}"
        )
    common_columns = sorted(set(first_frame.columns) & set(reload_frame.columns))
    compared_fields: set[str] = set()
    compared_values = 0
    differences: list[dict[str, object]] = []
    max_abs_diff = 0.0
    for row_index in range(len(first_frame)):
        for field in common_columns:
            try:
                first_value = float(first_frame.at[row_index, field])
                reload_value = float(reload_frame.at[row_index, field])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(first_value) or not math.isfinite(reload_value):
                continue
            compared_fields.add(field)
            compared_values += 1
            abs_diff = abs(first_value - reload_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if abs_diff != 0.0:
                differences.append(
                    {
                        "request_id": int(reload_frame.at[row_index, "Request Id"]),
                        "field": field,
                        "first": first_value,
                        "reload": reload_value,
                        "abs_diff": abs_diff,
                    }
                )
    if differences:
        raise AssertionError(
            f"reload request metrics differ in {len(differences)} values: "
            f"{differences[:10]}"
        )
    return {
        "first_request_metrics": str(first_path),
        "common_numeric_fields": len(compared_fields),
        "compared_numeric_values": compared_values,
        "nonzero_differences": 0,
        "max_abs_diff": max_abs_diff,
    }


def _diagnostic_sources(
    diagnostics: dict[str, Any],
    operator_name: str,
) -> dict[str, int]:
    operator_records = diagnostics.get("eager", {}).get(operator_name, {})
    sources: Counter[str] = Counter()
    for coverage_record in operator_records.values():
        if not isinstance(coverage_record, dict):
            continue
        value_sources = coverage_record.get("value_sources", {})
        if isinstance(value_sources, dict):
            sources.update(
                {
                    str(source): int(count)
                    for source, count in value_sources.items()
                }
            )
    return dict(sorted(sources.items()))


def _assert_expected_mixed_features(
    operator_calls: list[dict[str, object]],
    expected_prefill_request_count: int,
) -> None:
    by_operator = {
        operator_name: next(
            (
                call
                for call in operator_calls
                if call["operator_name"] == operator_name
                and call["value_source"] == "model_prediction"
            ),
            None,
        )
        for operator_name in TARGET_OPERATORS
    }
    if any(call is None for call in by_operator.values()):
        raise AssertionError(
            "the true-mixed batch must make one legal canonical model prediction "
            f"for each mixed operator; calls={operator_calls}"
        )

    prefill_features = by_operator["attn_prefill_mixed"]["features"]
    total_prefill_tokens = float(31 * expected_prefill_request_count)
    expected_prefill = {
        "batch_size": float(expected_prefill_request_count),
        "total_tokens": total_prefill_tokens,
        "kv_cache_size": 0.0,
    }
    for field, expected in expected_prefill.items():
        if float(prefill_features[field]) != expected:
            raise AssertionError(
                f"attn_prefill_mixed {field}={prefill_features[field]!r}, "
                f"expected={expected}"
            )

    decode_features = by_operator["attn_decode_in_mixed"]["features"]
    expected_decode = {
        "decode_batch_size": 1.0,
        "num_prefill_seqs": float(expected_prefill_request_count),
        "total_prefill_tokens": total_prefill_tokens,
        "total_batch_size": float(expected_prefill_request_count + 1),
        "total_tokens": total_prefill_tokens + 1.0,
    }
    for field, expected in expected_decode.items():
        if float(decode_features[field]) != expected:
            raise AssertionError(
                f"attn_decode_in_mixed {field}={decode_features[field]!r}, "
                f"expected={expected}"
            )


def main() -> None:
    args = _parse_args()
    if args.expect_fit == "zero" and args.first_request_metrics is None:
        raise ValueError("reload checks require --first-request-metrics")
    if args.expect_fit == "positive" and args.first_request_metrics is not None:
        raise ValueError("first-run checks must not provide --first-request-metrics")
    for required_path in (
        args.profile_dir / "linear_op.csv",
        args.profile_dir / "attention.csv",
        args.trace_file,
    ):
        if not required_path.is_file():
            raise ValueError(f"required input does not exist: {required_path}")

    cache_files_before = _cache_files(args.cache_dir)
    if args.expect_cache_state == "empty" and cache_files_before:
        raise AssertionError("first-run cache must start empty")
    if args.expect_cache_state == "populated" and not cache_files_before:
        raise AssertionError("reload cache must start populated")
    if args.metrics_dir.exists():
        raise AssertionError(
            f"metrics directory must be unique and absent: {args.metrics_dir}"
        )

    counters = {"fit_calls": 0, "load_attempts": 0, "cache_hits": 0}
    batch_records: list[dict[str, object]] = []
    on_demand_calls: list[dict[str, object]] = []
    execution_context_stack: list[dict[str, object]] = []
    original_fit = manager_module.GridSearchCV.fit
    original_load = SklearnExecutionTimePredictor._load_model_from_cache
    original_create_batch = VLLMv1EngineReplicaScheduler._create_batch
    original_on_demand = SklearnExecutionTimePredictor._get_on_demand_prediction
    original_prefill = (
        SklearnExecutionTimePredictor._get_attention_prefill_execution_time
    )
    original_decode = SklearnExecutionTimePredictor._get_attention_decode_execution_time

    def counted_fit(grid_search, *fit_args, **fit_kwargs):
        counters["fit_calls"] += 1
        return original_fit(grid_search, *fit_args, **fit_kwargs)

    def counted_load(predictor, *load_args, **load_kwargs):
        counters["load_attempts"] += 1
        model = original_load(predictor, *load_args, **load_kwargs)
        if model is not None:
            counters["cache_hits"] += 1
        return model

    def captured_create_batch(scheduler, requests, num_tokens):
        phases = ["decode" if request.is_prefill_complete else "prefill" for request in requests]
        request_records = [
            {
                "request_id": int(request.id),
                "phase": phase,
                "scheduled_tokens": int(tokens),
                "processed_tokens_before": int(request.num_processed_tokens),
                "num_prefill_tokens": int(request.num_prefill_tokens),
                "num_decode_tokens": int(request.num_decode_tokens),
            }
            for request, tokens, phase in zip(requests, num_tokens, phases)
        ]
        batch = original_create_batch(scheduler, requests, num_tokens)
        if scheduler._cluster_type == ClusterType.MONOLITHIC:
            batch_records.append(
                {
                    "batch_id": int(batch.id),
                    "cluster_type": scheduler._cluster_type.name,
                    "num_prefill_tokens": int(batch.num_prefill_tokens),
                    "num_decode_tokens": int(batch.num_decode_tokens),
                    "prefill_request_count": phases.count("prefill"),
                    "decode_request_count": phases.count("decode"),
                    "requests": request_records,
                }
            )
        return batch

    def captured_on_demand(predictor, model_name, features):
        if model_name not in TARGET_OPERATORS:
            return original_on_demand(predictor, model_name, features)
        source, family_name, feature_key = _source_before_on_demand_call(
            predictor,
            model_name,
            features,
        )
        value = original_on_demand(predictor, model_name, features)
        context = execution_context_stack[-1] if execution_context_stack else {}
        model = predictor._predictions[model_name]["_model"]
        binding = model._frontier_operator_binding
        on_demand_calls.append(
            {
                "context": str(context.get("kind", "unknown")),
                "batch_id": context.get("batch_id"),
                "component": context.get("component"),
                "operator_name": model_name,
                "measurement_family": family_name,
                "value_source": source,
                "feature_key": list(feature_key),
                "features": {name: float(value) for name, value in features.items()},
                "prediction_value_ms": float(value),
                "model_hash": str(model._frontier_model_hash),
                "binding_device": binding["device"],
                "binding_model_name": binding["model_name"],
                "binding_operator_name": binding["operator_name"],
                "binding_tensor_parallel_size": int(
                    _profile_scalar(model, "num_tensor_parallel_workers")
                ),
            }
        )
        return value

    def captured_prefill(predictor, batch):
        execution_context_stack.append(
            {
                "kind": "simulation",
                "component": "prefill",
                "batch_id": int(batch.id),
            }
        )
        try:
            return original_prefill(predictor, batch)
        finally:
            execution_context_stack.pop()

    def captured_decode(predictor, batch):
        execution_context_stack.append(
            {
                "kind": "simulation",
                "component": "decode",
                "batch_id": int(batch.id),
            }
        )
        try:
            return original_decode(predictor, batch)
        finally:
            execution_context_stack.pop()

    sys.argv = [sys.argv[0], *_cli_args(args)]
    with (
        mock.patch.object(manager_module.GridSearchCV, "fit", counted_fit),
        mock.patch.object(
            SklearnExecutionTimePredictor,
            "_load_model_from_cache",
            counted_load,
        ),
        mock.patch.object(
            VLLMv1EngineReplicaScheduler,
            "_create_batch",
            captured_create_batch,
        ),
        mock.patch.object(
            SklearnExecutionTimePredictor,
            "_get_on_demand_prediction",
            captured_on_demand,
        ),
        mock.patch.object(
            SklearnExecutionTimePredictor,
            "_get_attention_prefill_execution_time",
            captured_prefill,
        ),
        mock.patch.object(
            SklearnExecutionTimePredictor,
            "_get_attention_decode_execution_time",
            captured_decode,
        ),
    ):
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        simulator = Simulator(config)
        predictor, artifact_summary = _assert_monolithic_artifacts(
            simulator,
            args.tensor_parallel_size,
        )
        simulator.run()

        true_mixed_batches = [
            batch
            for batch in batch_records
            if batch["prefill_request_count"] > 0 and batch["decode_request_count"] > 0
        ]
        if len(true_mixed_batches) != 1:
            raise AssertionError(
                f"expected exactly one true-mixed batch, got {true_mixed_batches}"
            )
        true_mixed_batch = true_mixed_batches[0]
        if (
            true_mixed_batch["prefill_request_count"],
            true_mixed_batch["decode_request_count"],
        ) != (args.expected_prefill_request_count, 1):
            raise AssertionError(
                "true-mixed batch has the wrong prefill/decode request counts "
                f"request: {true_mixed_batch}"
            )
        mixed_batch_id = true_mixed_batch["batch_id"]
        mixed_batch_calls = [
            call
            for call in on_demand_calls
            if call["context"] == "simulation" and call["batch_id"] == mixed_batch_id
        ]
        _assert_expected_mixed_features(
            mixed_batch_calls,
            args.expected_prefill_request_count,
        )

        first_model_predictions = {
            operator_name: next(
                call
                for call in mixed_batch_calls
                if call["operator_name"] == operator_name
                and call["value_source"] == "model_prediction"
            )
            for operator_name in TARGET_OPERATORS
        }
        with predictor._temporary_measurement_type(MeasurementType.CUDA_EVENT):
            for operator_name, first_call in first_model_predictions.items():
                execution_context_stack.append(
                    {
                        "kind": "explicit_cache_replay",
                        "component": first_call["component"],
                        "batch_id": mixed_batch_id,
                    }
                )
                try:
                    repeated_value = predictor._get_on_demand_prediction(
                        operator_name,
                        dict(first_call["features"]),
                    )
                finally:
                    execution_context_stack.pop()
                if float(repeated_value) != float(first_call["prediction_value_ms"]):
                    raise AssertionError(
                        f"{operator_name} runtime-cache replay changed the value"
                    )

    if args.expect_fit == "positive" and counters["fit_calls"] <= 0:
        raise AssertionError("first run did not fit any prediction model")
    if args.expect_fit == "zero" and counters["fit_calls"] != 0:
        raise AssertionError(
            f"reload unexpectedly fit {counters['fit_calls']} prediction models"
        )
    if args.expect_fit == "zero" and counters["cache_hits"] <= 0:
        raise AssertionError("reload did not load any persisted prediction model")

    cache_files_after = _cache_files(args.cache_dir)
    if args.expect_fit == "positive" and not cache_files_after:
        raise AssertionError("first run did not persist any cache artifacts")
    if args.expect_fit == "zero" and len(cache_files_after) != len(cache_files_before):
        raise AssertionError(
            "reload changed the persisted cache artifact count: "
            f"{len(cache_files_before)} -> {len(cache_files_after)}"
        )

    diagnostics = predictor.get_prediction_domain_diagnostics()
    runtime_cache_summary: dict[str, object] = {}
    for operator_name in TARGET_OPERATORS:
        sources = _diagnostic_sources(diagnostics, operator_name)
        if sources.get("model_prediction", 0) < 1:
            raise AssertionError(
                f"{operator_name} diagnostics lack canonical model prediction: {sources}"
            )
        if sources.get("runtime_cache", 0) < 1:
            raise AssertionError(
                f"{operator_name} diagnostics lack runtime-cache reuse: {sources}"
            )
        entries = len(predictor._runtime_cache["eager"][operator_name])
        if entries != 1:
            raise AssertionError(
                f"{operator_name} runtime cache has {entries} entries, expected 1"
            )
        runtime_cache_summary[operator_name] = {
            "entries": entries,
            "diagnostic_value_sources": sources,
        }

    metrics_path = _find_request_metrics(args.metrics_dir)
    metrics_frame, metrics_summary = _read_and_validate_request_metrics(
        metrics_path,
        args.expected_prefill_request_count,
    )
    parity = (
        _compare_request_metrics(args.first_request_metrics, metrics_frame)
        if args.first_request_metrics is not None
        else None
    )
    simulation_sources = {
        operator_name: dict(
            sorted(
                Counter(
                    call["value_source"]
                    for call in on_demand_calls
                    if call["context"] == "simulation"
                    and call["batch_id"] == true_mixed_batch["batch_id"]
                    and call["operator_name"] == operator_name
                ).items()
            )
        )
        for operator_name in TARGET_OPERATORS
    }
    replay_calls = [
        call for call in on_demand_calls if call["context"] == "explicit_cache_replay"
    ]
    if {
        (call["operator_name"], call["value_source"]) for call in replay_calls
    } != {(operator_name, "runtime_cache") for operator_name in TARGET_OPERATORS}:
        raise AssertionError(f"explicit replay did not use runtime cache: {replay_calls}")

    print(
        json.dumps(
            {
                "simulation_mode": args.simulation_mode,
                "tensor_parallel_size": args.tensor_parallel_size,
                "cache": {
                    "pkl_before": len(cache_files_before),
                    "pkl_after": len(cache_files_after),
                    **counters,
                },
                "artifacts": artifact_summary,
                "true_mixed_batch": true_mixed_batch,
                "simulation_mixed_value_sources": simulation_sources,
                "mixed_model_prediction_calls": {
                    operator_name: first_model_predictions[operator_name]
                    for operator_name in TARGET_OPERATORS
                },
                "runtime_cache": runtime_cache_summary,
                "explicit_runtime_cache_replays": replay_calls,
                "request_metrics_path": str(metrics_path),
                "request_metrics": metrics_summary,
                "first_reload_parity": parity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
