"""Run a non-dummy heterogeneous PDD or PD-AF ownership check.

The check exercises the real Simulator initialization and runtime path. It
verifies that every compute role receives only artifacts matching its device,
TP, and measurement family, and that a reload performs no estimator fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from frontier.simulator import Simulator
from frontier.types import ClusterType
from frontier.utils.random import set_seeds


MODEL_NAME = "Qwen3-30B-A3B-tiny"
DEVICE_RTX = "rtx_pro_6000"
DEVICE_H800 = "h800"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("pdd", "pdaf"),
        required=True,
    )
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--expect-fit",
        choices=("positive", "zero"),
        required=True,
    )
    parser.add_argument(
        "--expect-cache-state",
        choices=("empty", "populated"),
        required=True,
    )
    return parser.parse_args()


def _predictor_args(args: argparse.Namespace) -> list[str]:
    compute_root = args.profile_root / "compute" / "{DEVICE}" / "{MODEL}"
    return [
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_linear_op_input_file",
        str(compute_root / "linear_op.csv"),
        "--random_forrest_execution_time_predictor_config_atten_input_file",
        str(compute_root / "attention.csv"),
        "--random_forrest_execution_time_predictor_config_moe_input_file",
        str(compute_root / "moe.csv"),
        "--random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file",
        str(compute_root / "linear_op_kernel_only.csv"),
        "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file",
        str(compute_root / "attention_kernel_only.csv"),
        "--random_forrest_execution_time_predictor_config_moe_kernel_only_input_file",
        str(compute_root / "moe_kernel_only.csv"),
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
    return [
        "--simulation_mode",
        "offline",
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
        "8",
        "--fixed_request_length_generator_config_decode_tokens",
        "1",
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


def _pdd_args(args: argparse.Namespace) -> list[str]:
    return [
        "--sys_arch",
        "pd-disaggregation",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_cluster_num_replicas",
        "1",
        "--cluster_config_prefill_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_total_expert_num",
        "16",
        "--cluster_config_prefill_replica_config_router_topk",
        "8",
        "--cluster_config_prefill_replica_config_device",
        DEVICE_RTX,
        "--cluster_config_decode_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size",
        "2",
        "--cluster_config_decode_replica_config_attn_data_parallel_size",
        "1",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size",
        "2",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_decode_replica_config_total_expert_num",
        "16",
        "--cluster_config_decode_replica_config_router_topk",
        "8",
        "--cluster_config_decode_replica_config_device",
        DEVICE_H800,
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
        "--cluster_config_prefill_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_total_expert_num",
        "16",
        "--cluster_config_prefill_replica_config_router_topk",
        "8",
        "--cluster_config_prefill_replica_config_device",
        DEVICE_RTX,
        "--cluster_config_decode_attn_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size",
        "2",
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size",
        "1",
        "--cluster_config_decode_attn_replica_config_device",
        DEVICE_H800,
        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size",
        "4",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num",
        "16",
        "--cluster_config_decode_ffn_replica_config_router_topk",
        "8",
        "--cluster_config_decode_ffn_replica_config_device",
        DEVICE_H800,
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
) -> dict[ClusterType, dict[str, object]]:
    if architecture == "pdd":
        return {
            ClusterType.PREFILL: {
                "device": DEVICE_RTX,
                "tp": 1,
                "families": {"eager": True, "kernel_only": False},
                "sentinels": {"eager": ("attn_prefill", "moe_grouped_gemm")},
            },
            ClusterType.DECODE: {
                "device": DEVICE_H800,
                "tp": 2,
                "families": {"eager": True, "kernel_only": False},
                "sentinels": {"eager": ("attn_decode", "moe_grouped_gemm")},
            },
        }
    return {
        ClusterType.PREFILL: {
            "device": DEVICE_RTX,
            "tp": 1,
            "families": {"eager": True, "kernel_only": False},
            "sentinels": {"eager": ("attn_prefill", "moe_grouped_gemm")},
        },
        ClusterType.DECODE_ATTN: {
            "device": DEVICE_H800,
            "tp": 2,
            "families": {"eager": True, "kernel_only": True},
            "sentinels": {
                "eager": ("attn_decode",),
                "kernel_only": ("attn_decode",),
            },
        },
        ClusterType.DECODE_FFN: {
            "device": DEVICE_H800,
            "tp": 4,
            "families": {"eager": False, "kernel_only": True},
            "sentinels": {"kernel_only": ("moe_grouped_gemm",)},
        },
    }


def _profile_scalar(model: object, field: str) -> object:
    binding = getattr(model, "_frontier_operator_binding", None)
    if not isinstance(binding, dict):
        raise AssertionError("persisted model lacks operator binding metadata")
    structure = binding.get("profile_structure")
    if not isinstance(structure, dict) or field not in structure:
        raise AssertionError(f"operator binding lacks profile field {field!r}")
    value = structure[field]
    if isinstance(value, list):
        raise AssertionError(f"profile field {field!r} is not scalar: {value!r}")
    return value


def _assert_cluster_views(
    simulator: Simulator,
    architecture: str,
) -> dict[str, object]:
    manager = simulator._execution_time_prediction_model_manager
    expected = _expected_roles(architecture)
    summaries: dict[str, object] = {}

    runtime_caches = [
        simulator._predictors[cluster_type]._runtime_cache
        for cluster_type in expected
    ]
    if len({id(cache) for cache in runtime_caches}) != len(runtime_caches):
        raise AssertionError("runtime prediction caches must be cluster-local")

    for cluster_type, role in expected.items():
        context = manager.get_training_context(cluster_type)
        if context["device"] != role["device"]:
            raise AssertionError(
                f"{cluster_type.name} training device mismatch: {context['device']}"
            )

        models_by_family = manager.get_models_for_cluster(cluster_type)
        predictor = simulator._predictors[cluster_type]
        family_summary: dict[str, object] = {}
        for family_name, should_exist in role["families"].items():
            models = models_by_family[family_name]
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

            expected_measurement = (
                "CUDA_EVENT" if family_name == "eager" else "KERNEL_ONLY"
            )
            hashes = {}
            for model_name, model in models.items():
                binding = model._frontier_operator_binding
                if binding.get("device") != role["device"]:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{model_name} uses "
                        f"device={binding.get('device')!r}"
                    )
                if _profile_scalar(model, "measurement_type") != expected_measurement:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{model_name} has the "
                        "wrong measurement family"
                    )
                hashes[model_name] = model._frontier_model_hash

            sentinel_summary = {}
            for model_name in role.get("sentinels", {}).get(family_name, ()):
                if model_name not in models:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name} lacks {model_name}"
                    )
                model = models[model_name]
                actual_tp = int(
                    _profile_scalar(model, "num_tensor_parallel_workers")
                )
                if actual_tp != role["tp"]:
                    raise AssertionError(
                        f"{cluster_type.name}/{family_name}/{model_name} uses "
                        f"TP={actual_tp}, expected {role['tp']}"
                    )
                sentinel_summary[model_name] = {
                    "device": model._frontier_operator_binding["device"],
                    "tp": actual_tp,
                    "measurement_type": _profile_scalar(
                        model, "measurement_type"
                    ),
                    "model_hash": model._frontier_model_hash,
                }

            family_summary[family_name] = {
                "model_count": len(models),
                "sentinels": sentinel_summary,
                "model_hashes": hashes,
            }

        summaries[cluster_type.name] = {
            "device": role["device"],
            "tp": role["tp"],
            "families": family_summary,
        }

    try:
        manager.get_model("attn_pre_proj")
    except ValueError as exc:
        if "ambiguous" not in str(exc):
            raise
    else:
        raise AssertionError("unscoped heterogeneous model access must fail")

    return summaries


def _read_request_metrics(metrics_root: Path) -> tuple[Path, dict[str, float]]:
    matches = sorted(metrics_root.rglob("request_metrics.csv"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one request_metrics.csv below {metrics_root}, got {matches}"
        )
    frame = pd.read_csv(matches[0])
    if len(frame) != 1:
        raise AssertionError(f"expected one request row, got {len(frame)}")
    row = frame.iloc[0]
    expected_tokens = {
        "request_num_prefill_tokens": 8.0,
        "request_num_decode_tokens": 1.0,
        "request_num_tokens": 9.0,
    }
    for field, expected in expected_tokens.items():
        if float(row[field]) != expected:
            raise AssertionError(f"{field}={row[field]!r}, expected {expected}")

    numeric_fields = (
        "request_e2e_time",
        "request_execution_time",
        "ttft",
        "transfer_kv_cache",
        "transfer_m2n_total",
        "cluster_prefill_computation",
        "cluster_decode_computation",
        "cluster_decode_attn_computation",
        "cluster_decode_ffn_computation",
    )
    metrics = {}
    for field in numeric_fields:
        if field not in row or pd.isna(row[field]):
            continue
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise AssertionError(f"{field} must be finite and non-negative")
        metrics[field] = value
    return matches[0], metrics


def main() -> None:
    args = _parse_args()
    cli_args = _pdd_args(args) if args.architecture == "pdd" else _pdaf_args(args)
    cache_files_before = sorted(args.cache_dir.glob("*.pkl"))
    if args.expect_cache_state == "empty" and cache_files_before:
        raise AssertionError("first-run cache must start empty")
    if args.expect_cache_state == "populated" and not cache_files_before:
        raise AssertionError("reload cache must start populated")
    if args.metrics_dir.exists():
        raise AssertionError(
            f"metrics directory must be unique and absent: {args.metrics_dir}"
        )

    counters = {"fit_calls": 0, "load_attempts": 0, "cache_hits": 0}
    original_fit = manager_module.GridSearchCV.fit
    original_load = ExecutionTimePredictionModelManager._load_model_from_cache

    def counted_fit(grid_search, *fit_args, **fit_kwargs):
        counters["fit_calls"] += 1
        return original_fit(grid_search, *fit_args, **fit_kwargs)

    def counted_load(model_manager, *load_args, **load_kwargs):
        counters["load_attempts"] += 1
        model = original_load(model_manager, *load_args, **load_kwargs)
        if model is not None:
            counters["cache_hits"] += 1
        return model

    sys.argv = [sys.argv[0], *cli_args]
    with mock.patch.object(manager_module.GridSearchCV, "fit", counted_fit), mock.patch.object(
        ExecutionTimePredictionModelManager,
        "_load_model_from_cache",
        counted_load,
    ):
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        simulator = Simulator(config)
        role_summary = _assert_cluster_views(simulator, args.architecture)
        simulator.run()

    if args.expect_fit == "positive" and counters["fit_calls"] <= 0:
        raise AssertionError("first run did not fit any prediction model")
    if args.expect_fit == "zero" and counters["fit_calls"] != 0:
        raise AssertionError(
            f"reload unexpectedly fit {counters['fit_calls']} prediction models"
        )
    if args.expect_fit == "zero" and counters["cache_hits"] <= 0:
        raise AssertionError("reload did not load any persisted prediction model")

    cache_files_after = sorted(args.cache_dir.glob("*.pkl"))
    if len(cache_files_after) < len(cache_files_before):
        raise AssertionError("cache reload removed persisted artifacts")
    if args.expect_fit == "positive" and not cache_files_after:
        raise AssertionError("first run did not persist any cache artifacts")
    if args.expect_fit == "zero" and len(cache_files_after) != len(cache_files_before):
        raise AssertionError(
            "reload changed the persisted cache artifact count: "
            f"{len(cache_files_before)} -> {len(cache_files_after)}"
        )

    metrics_path, key_metrics = _read_request_metrics(args.metrics_dir)
    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "cache": {
                    "pkl_before": len(cache_files_before),
                    "pkl_after": len(cache_files_after),
                    **counters,
                },
                "roles": role_summary,
                "request_metrics_path": str(metrics_path),
                "key_metrics_ms": key_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
