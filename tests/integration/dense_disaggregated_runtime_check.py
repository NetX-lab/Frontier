"""Validate non-dummy dense PDD and PD-AF runtime roles.

The check exercises real Simulator initialization and execution for every
supported compute TP. It verifies cluster-scoped artifact identity, actual
measurement-family dispatch, persisted-model reload, and exact request-metric
parity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd

from frontier.config import SimulationConfig
from frontier.execution_time_predictor import (
    shared_prediction_model_manager as manager_module,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType
from frontier.utils.random import set_seeds


MODEL_NAME = "llama2_7b_dense_example"
DEVICE = "h800"
EXPECTED_REQUEST_LENGTH = (8, 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("pdd", "pdaf"), required=True)
    parser.add_argument("--simulation-mode", choices=("offline", "online"), required=True)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2, 4, 8),
        required=True,
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
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
        "--random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file",
        str(args.profile_dir / "linear_op_kernel_only.csv"),
        "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file",
        str(args.profile_dir / "attention_kernel_only.csv"),
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
        "2",
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        "4096",
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        "4096",
    ]


def _common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--simulation_mode",
        args.simulation_mode,
        "--no-enable_parallel_clusters",
        "--cc_backend_config_type",
        "analytical",
        "--replica_config_model_name",
        MODEL_NAME,
        "--replica_config_moe_routing_mode",
        "uniform_random",
        "--replica_config_moe_routing_seed",
        "42",
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "64",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks",
        "128",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        "1",
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(EXPECTED_REQUEST_LENGTH[0]),
        "--fixed_request_length_generator_config_decode_tokens",
        str(EXPECTED_REQUEST_LENGTH[1]),
        "--interval_generator_config_type",
        "poisson",
        "--poisson_request_interval_generator_config_qps",
        "1.0",
        "--metrics_config_output_dir",
        str(args.metrics_dir),
        "--metrics_config_cache_dir",
        str(args.cache_dir),
        "--metrics_config_run_id",
        args.run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        *_predictor_args(args),
    ]
    if args.simulation_mode == "offline":
        values.append("--offline_use_generated_request_arrivals")
    return values


def _dense_replica_args(
    prefix: str,
    tp: int,
    *,
    attention: bool,
    ffn: bool = True,
) -> list[str]:
    values = [
        f"--{prefix}_replica_config_num_pipeline_stages",
        "1",
        f"--{prefix}_replica_config_device",
        DEVICE,
    ]
    if ffn:
        values.extend(
            [
                f"--{prefix}_replica_config_moe_tensor_parallel_size",
                str(tp),
                f"--{prefix}_replica_config_moe_expert_parallel_size",
                "1",
                f"--{prefix}_replica_config_total_expert_num",
                "1",
                f"--{prefix}_replica_config_router_topk",
                "1",
            ]
        )
    if attention:
        values.extend(
            [
                f"--{prefix}_replica_config_attn_tensor_parallel_size",
                str(tp),
                f"--{prefix}_replica_config_attn_data_parallel_size",
                "1",
            ]
        )
    return values


def _pdd_args(args: argparse.Namespace) -> list[str]:
    tp = args.tensor_parallel_size
    return [
        "--sys_arch",
        "pd-disaggregation",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_cluster_num_replicas",
        "1",
        *_dense_replica_args("cluster_config_prefill", tp, attention=True),
        *_dense_replica_args("cluster_config_decode", tp, attention=True),
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "none",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "200.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.5",
        *_common_args(args),
    ]


def _pdaf_args(args: argparse.Namespace) -> list[str]:
    tp = args.tensor_parallel_size
    return [
        "--sys_arch",
        "pd-af-disaggregation",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_attn_cluster_num_replicas",
        "1",
        "--cluster_config_decode_ffn_cluster_num_replicas",
        "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch",
        "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch",
        "1",
        "--cluster_config_decode_attn_micro_batch_size",
        "1",
        *_dense_replica_args("cluster_config_prefill", tp, attention=True),
        *_dense_replica_args(
            "cluster_config_decode_attn",
            tp,
            attention=True,
            ffn=False,
        ),
        *_dense_replica_args("cluster_config_decode_ffn", tp, attention=False),
        "--cluster_config_prefill_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type",
        "orca",
        "--m2n_transfer_config_type",
        "analytical",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "200.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.5",
        "--analytical_m2_n_transfer_config_memory_bandwidth_gbps",
        "200.0",
        "--analytical_m2_n_transfer_config_network_latency_ms",
        "0.05",
        *_common_args(args),
    ]


def _expected_roles(
    architecture: str,
) -> dict[ClusterType, dict[str, tuple[str, ...]]]:
    if architecture == "pdd":
        return {
            ClusterType.PREFILL: {
                "eager": ("attn_prefill", "mlp_up_proj"),
            },
            ClusterType.DECODE: {
                "eager": ("attn_decode", "mlp_up_proj"),
            },
        }
    return {
        ClusterType.PREFILL: {
            "eager": ("attn_prefill", "mlp_up_proj"),
        },
        ClusterType.DECODE_ATTN: {
            "eager": ("attn_decode", "post_attention_layernorm"),
            "kernel_only": ("attn_decode", "post_attention_layernorm"),
        },
        ClusterType.DECODE_FFN: {
            "kernel_only": ("mlp_up_proj", "mlp_down_proj", "mlp_act"),
        },
    }


def _expected_runtime_families(architecture: str) -> dict[str, str]:
    if architecture == "pdd":
        return {"PREFILL": "eager", "DECODE": "eager"}
    return {
        "PREFILL": "eager",
        "DECODE_ATTN": "kernel_only",
        "DECODE_FFN": "kernel_only",
    }


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


def _assert_role_artifacts(
    simulator: Simulator,
    architecture: str,
    tp: int,
) -> dict[str, object]:
    manager = simulator._execution_time_prediction_model_manager
    expected = _expected_roles(architecture)
    summaries: dict[str, object] = {}
    runtime_cache_ids = {
        id(simulator._predictors[cluster_type]._runtime_cache)
        for cluster_type in expected
    }
    if len(runtime_cache_ids) != len(expected):
        raise AssertionError("runtime prediction caches must be cluster-local")

    for cluster_type, expected_families in expected.items():
        context = manager.get_training_context(cluster_type)
        if context["device"] != DEVICE:
            raise AssertionError(
                f"{cluster_type.name} device={context['device']!r}, expected={DEVICE!r}"
            )
        models_by_family = manager.get_models_for_cluster(cluster_type)
        predictor = simulator._predictors[cluster_type]
        family_summary: dict[str, object] = {}
        for family_name in ("eager", "kernel_only"):
            models = models_by_family[family_name]
            should_exist = family_name in expected_families
            if bool(models) != should_exist:
                raise AssertionError(
                    f"{cluster_type.name}/{family_name} model presence mismatch"
                )
            predictor_models = getattr(predictor, f"_models_{family_name}")
            if set(predictor_models) != set(models):
                raise AssertionError(
                    f"{cluster_type.name}/{family_name} consumer keys differ"
                )
            if any(predictor_models[name] is not model for name, model in models.items()):
                raise AssertionError(
                    f"{cluster_type.name}/{family_name} consumer objects differ"
                )

            sentinels = {}
            for operator_name in expected_families.get(family_name, ()):
                if operator_name not in models:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name} lacks {operator_name}"
                    )
                model = models[operator_name]
                binding = model._frontier_operator_binding
                expected_measurement = (
                    "CUDA_EVENT" if family_name == "eager" else "KERNEL_ONLY"
                )
                if binding.get("device") != DEVICE:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{operator_name} "
                        f"uses device={binding.get('device')!r}"
                    )
                if binding.get("model_name") != MODEL_NAME:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{operator_name} "
                        f"uses model={binding.get('model_name')!r}"
                    )
                if binding.get("operator_name") != operator_name:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{operator_name} "
                        "has the wrong operator binding"
                    )
                actual_tp = int(
                    _profile_scalar(model, "num_tensor_parallel_workers")
                )
                expected_tp = (
                    1
                    if (
                        cluster_type is ClusterType.DECODE_ATTN
                        and operator_name == "post_attention_layernorm"
                    )
                    else tp
                )
                if actual_tp != expected_tp:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{operator_name} "
                        f"uses TP={actual_tp}, expected={expected_tp}"
                    )
                actual_measurement = str(_profile_scalar(model, "measurement_type"))
                if actual_measurement != expected_measurement:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{operator_name} "
                        f"uses measurement={actual_measurement!r}"
                    )
                sentinels[operator_name] = {
                    "model_hash": str(model._frontier_model_hash),
                    "tensor_parallel_size": actual_tp,
                    "measurement_type": actual_measurement,
                }
            family_summary[family_name] = {
                "model_count": len(models),
                "sentinels": sentinels,
            }
        summaries[cluster_type.name] = {
            "device": context["device"],
            "attn_tensor_parallel_size": context["attn_tensor_parallel_size"],
            "ffn_tensor_parallel_size": context["moe_tensor_parallel_size"],
            "families": family_summary,
        }
    return summaries


def _cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.pkl")) if cache_dir.exists() else []


def _find_request_metrics(metrics_root: Path) -> Path:
    matches = sorted(metrics_root.rglob("request_metrics.csv"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one request_metrics.csv below {metrics_root}, got {matches}"
        )
    return matches[0]


def _read_request_metrics(
    path: Path,
    architecture: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise AssertionError(f"expected one request row in {path}, got {len(frame)}")
    row = frame.iloc[0]
    expected_tokens = {
        "request_num_prefill_tokens": float(EXPECTED_REQUEST_LENGTH[0]),
        "request_num_decode_tokens": float(EXPECTED_REQUEST_LENGTH[1]),
        "request_num_tokens": float(sum(EXPECTED_REQUEST_LENGTH)),
    }
    for field, expected in expected_tokens.items():
        if float(row[field]) != expected:
            raise AssertionError(f"{field}={row[field]!r}, expected={expected}")

    required_positive = [
        "request_e2e_time",
        "request_execution_time",
        "ttft",
        "transfer_kv_cache",
        "cluster_prefill_computation",
    ]
    if architecture == "pdd":
        required_positive.append("cluster_decode_computation")
    else:
        required_positive.extend(
            [
                "transfer_m2n_total",
                "cluster_decode_attn_computation",
                "cluster_decode_ffn_computation",
            ]
        )
    metrics: dict[str, float] = {}
    for field in required_positive:
        value = float(row[field])
        if not math.isfinite(value) or value <= 0.0:
            raise AssertionError(f"{field} must be finite and positive, got {value}")
        metrics[field] = value
    return frame, metrics


def _compare_request_metrics(
    first_path: Path,
    reload_frame: pd.DataFrame,
) -> dict[str, object]:
    first_frame = pd.read_csv(first_path)
    if len(first_frame) != len(reload_frame):
        raise AssertionError(
            f"first/reload row count differs: {len(first_frame)} != {len(reload_frame)}"
        )
    numeric_fields: set[str] = set()
    compared_values = 0
    differences: list[dict[str, object]] = []
    max_abs_diff = 0.0
    for row_index in range(len(first_frame)):
        for field in sorted(set(first_frame.columns) & set(reload_frame.columns)):
            try:
                first_value = float(first_frame.at[row_index, field])
                reload_value = float(reload_frame.at[row_index, field])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(first_value) or not math.isfinite(reload_value):
                continue
            numeric_fields.add(field)
            compared_values += 1
            abs_diff = abs(first_value - reload_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if abs_diff != 0.0:
                differences.append(
                    {
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
        "common_numeric_fields": len(numeric_fields),
        "compared_numeric_values": compared_values,
        "nonzero_differences": 0,
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    args = _parse_args()
    if args.expect_fit == "zero" and args.first_request_metrics is None:
        raise ValueError("reload checks require --first-request-metrics")
    if args.expect_fit == "positive" and args.first_request_metrics is not None:
        raise ValueError("first-run checks must not provide --first-request-metrics")
    for required_path in (
        args.profile_dir / "linear_op.csv",
        args.profile_dir / "attention.csv",
        args.profile_dir / "linear_op_kernel_only.csv",
        args.profile_dir / "attention_kernel_only.csv",
    ):
        if not required_path.is_file():
            raise ValueError(f"required profile does not exist: {required_path}")

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
    runtime_calls: list[dict[str, object]] = []
    original_fit = manager_module.GridSearchCV.fit
    original_load = ExecutionTimePredictionModelManager._load_model_from_cache
    original_predict_stage = (
        SklearnDisaggregationExecutionTimePredictor.predict_stage_execution_time
    )

    def counted_fit(grid_search, *fit_args, **fit_kwargs):
        counters["fit_calls"] += 1
        return original_fit(grid_search, *fit_args, **fit_kwargs)

    def counted_load(model_manager, *load_args, **load_kwargs):
        counters["load_attempts"] += 1
        model = original_load(model_manager, *load_args, **load_kwargs)
        if model is not None:
            counters["cache_hits"] += 1
        return model

    def captured_predict_stage(
        predictor,
        batch,
        stage_id,
        cluster_type,
        num_layers=1,
        layer_id=0,
    ):
        result = original_predict_stage(
            predictor,
            batch,
            stage_id,
            cluster_type,
            num_layers=num_layers,
            layer_id=layer_id,
        )
        family = predictor._measurement_family_name(
            predictor._active_measurement_type
        )
        runtime_calls.append(
            {
                "cluster_type": cluster_type.name,
                "measurement_family": family,
                "batch_id": int(batch.id),
                "num_prefill_tokens": int(batch.num_prefill_tokens),
                "num_decode_tokens": int(batch.num_decode_tokens),
                "stage_id": int(stage_id),
                "layer_id": int(layer_id),
                "num_layers": int(num_layers),
            }
        )
        return result

    cli_args = _pdd_args(args) if args.architecture == "pdd" else _pdaf_args(args)
    sys.argv = [sys.argv[0], *cli_args]
    with (
        mock.patch.object(manager_module.GridSearchCV, "fit", counted_fit),
        mock.patch.object(
            ExecutionTimePredictionModelManager,
            "_load_model_from_cache",
            counted_load,
        ),
        mock.patch.object(
            SklearnDisaggregationExecutionTimePredictor,
            "predict_stage_execution_time",
            captured_predict_stage,
        ),
    ):
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        simulator = Simulator(config)
        role_summary = _assert_role_artifacts(
            simulator,
            args.architecture,
            args.tensor_parallel_size,
        )
        simulator.run()

    expected_runtime_families = _expected_runtime_families(args.architecture)
    runtime_summary = Counter(
        (call["cluster_type"], call["measurement_family"])
        for call in runtime_calls
    )
    for cluster_name, family_name in expected_runtime_families.items():
        if runtime_summary[(cluster_name, family_name)] <= 0:
            raise AssertionError(
                f"runtime did not execute {cluster_name}/{family_name}: "
                f"{dict(runtime_summary)}"
            )
    unexpected = [
        call
        for call in runtime_calls
        if expected_runtime_families.get(call["cluster_type"])
        != call["measurement_family"]
    ]
    if unexpected:
        raise AssertionError(
            f"runtime used an unexpected measurement family: {unexpected[:10]}"
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

    metrics_path = _find_request_metrics(args.metrics_dir)
    metrics_frame, key_metrics = _read_request_metrics(
        metrics_path,
        args.architecture,
    )
    parity = (
        _compare_request_metrics(args.first_request_metrics, metrics_frame)
        if args.first_request_metrics is not None
        else None
    )

    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "simulation_mode": args.simulation_mode,
                "tensor_parallel_size": args.tensor_parallel_size,
                "cache": {
                    "pkl_before": len(cache_files_before),
                    "pkl_after": len(cache_files_after),
                    **counters,
                },
                "roles": role_summary,
                "runtime_family_calls": {
                    f"{cluster}/{family}": count
                    for (cluster, family), count in sorted(runtime_summary.items())
                },
                "request_metrics_path": str(metrics_path),
                "key_metrics_ms": key_metrics,
                "first_reload_parity": parity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
