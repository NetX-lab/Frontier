#!/usr/bin/env python3
"""Generate Frontier simulation artifacts for operator-parity golden cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.e2e.attention_equivalence.profile_manifest import file_manifest, write_json_report
from tests.e2e.operator_parity.run_golden_matrix import (
    DEFAULT_WORKLOAD_PROFILES,
    DEFAULT_SIMULATION_ARTIFACTS,
    GoldenParallelism,
    GOLDEN_CONFIG_FILENAMES,
    OP_TRACES_CSV,
    OP_TRACES_JSONL,
    REQUEST_METRICS_CSV,
    WORKLOAD_PROFILES,
    build_golden_cases,
    normalize_op_traces_jsonl_to_csv,
)

PROFILE_FLAG_BY_FILE = {
    "attention.csv": "--random_forrest_execution_time_predictor_config_atten_input_file",
    "attention_kernel_only.csv": "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file",
    "linear_op.csv": "--random_forrest_execution_time_predictor_config_linear_op_input_file",
    "linear_op_kernel_only.csv": "--random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file",
    "moe.csv": "--random_forrest_execution_time_predictor_config_moe_input_file",
    "moe_kernel_only.csv": "--random_forrest_execution_time_predictor_config_moe_kernel_only_input_file",
}

WORKLOAD_SHAPE_FIELDS = (
    "num_requests",
    "prefill_tokens",
    "decode_tokens",
    "qps",
)


def cli_simulation_mode(workload_type: str) -> str:
    if workload_type == "offline_batch":
        return "offline"
    if workload_type == "online_serving":
        return "online"
    raise ValueError(f"Unsupported simulation workload type: {workload_type!r}")


def decode_cuda_graph_mode(measurement_type: str) -> str:
    if measurement_type == "CUDA_EVENT":
        return "none"
    if measurement_type == "KERNEL_ONLY":
        return "full_decode_only"
    raise ValueError(f"Unsupported measurement_type: {measurement_type!r}")


def profile_file_flags(case: Mapping[str, Any]) -> dict[str, str]:
    profile_files = case.get("reference_input_manifest", {}).get("profile_files")
    if not isinstance(profile_files, dict):
        raise ValueError(f"case {case.get('name')!r} is missing reference profile_files")

    flags: dict[str, str] = {}
    for filename, flag in PROFILE_FLAG_BY_FILE.items():
        if filename not in profile_files:
            continue
        file_info = profile_files[filename]
        if not isinstance(file_info, dict) or not file_info.get("path"):
            raise ValueError(f"case {case.get('name')!r} has invalid manifest for {filename}")
        flags[flag] = str(file_info["path"])
    return flags


def _append_flags(command: list[str], flags: Mapping[str, str]) -> None:
    for flag, value in flags.items():
        command.extend([flag, value])


def _case_workload_shape(case: Mapping[str, Any]) -> dict[str, int | float]:
    case_manifest = case["case_manifest"]
    if "workload_profile" not in case_manifest:
        raise ValueError(
            f"case {case.get('name')!r} is missing required golden workload_profile metadata"
        )
    workload_profile_name = str(case_manifest["workload_profile"])
    try:
        expected_profile = WORKLOAD_PROFILES[workload_profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported golden workload profile: {workload_profile_name!r}; "
            f"supported={sorted(WORKLOAD_PROFILES)}"
        ) from exc

    if "workload_shape" not in case_manifest:
        raise ValueError(
            f"case {case.get('name')!r} is missing required golden workload_shape metadata"
        )
    raw_shape = case_manifest["workload_shape"]
    if not isinstance(raw_shape, Mapping):
        raise ValueError(
            f"case {case.get('name')!r} has invalid golden workload_shape metadata"
        )
    missing_fields = [
        field
        for field in WORKLOAD_SHAPE_FIELDS
        if field not in raw_shape
    ]
    if missing_fields:
        raise ValueError(
            f"case {case.get('name')!r} has incomplete golden workload_shape metadata: "
            f"missing={missing_fields}"
        )

    shape: dict[str, int | float] = {
        "num_requests": _require_int_workload_shape_value(
            case=case,
            raw_shape=raw_shape,
            field="num_requests",
        ),
        "prefill_tokens": _require_int_workload_shape_value(
            case=case,
            raw_shape=raw_shape,
            field="prefill_tokens",
        ),
        "decode_tokens": _require_int_workload_shape_value(
            case=case,
            raw_shape=raw_shape,
            field="decode_tokens",
        ),
        "qps": _require_numeric_workload_shape_value(
            case=case,
            raw_shape=raw_shape,
            field="qps",
        ),
    }
    expected_shape = expected_profile.to_dict()
    if shape != expected_shape:
        raise ValueError(
            "golden workload_shape mismatch: "
            f"case={case.get('name')!r}, workload_profile={workload_profile_name!r}, "
            f"expected={expected_shape}, actual={shape}"
        )
    return shape


def _require_int_workload_shape_value(
    *,
    case: Mapping[str, Any],
    raw_shape: Mapping[str, Any],
    field: str,
) -> int:
    value = raw_shape[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"case {case.get('name')!r} has non-integer golden workload_shape "
            f"value for {field!r}: {value!r}"
        )
    return value


def _require_numeric_workload_shape_value(
    *,
    case: Mapping[str, Any],
    raw_shape: Mapping[str, Any],
    field: str,
) -> int | float:
    value = raw_shape[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"case {case.get('name')!r} has non-numeric golden workload_shape "
            f"value for {field!r}: {value!r}"
        )
    return value


def _common_frontier_flags(
    *,
    case: Mapping[str, Any],
    output_root: Path,
) -> list[str]:
    case_manifest = case["case_manifest"]
    workload_type = str(case_manifest["simulation_mode"])
    measurement_type = str(case_manifest["measurement_type"])
    workload_shape = _case_workload_shape(case)
    return [
        "--simulation_mode",
        cli_simulation_mode(workload_type),
        "--replica_config_model_name",
        str(case_manifest["model_name"]),
        "--cc_backend_config_type",
        "analytical",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        decode_cuda_graph_mode(measurement_type),
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "128",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "64",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(workload_shape["num_requests"]),
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(workload_shape["prefill_tokens"]),
        "--fixed_request_length_generator_config_decode_tokens",
        str(workload_shape["decode_tokens"]),
        "--interval_generator_config_type",
        "poisson",
        "--poisson_request_interval_generator_config_qps",
        str(workload_shape["qps"]),
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        "1024",
        "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",
        "--metrics_config_output_dir",
        str(output_root),
        "--metrics_config_run_id",
        str(case["name"]),
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_store_token_completion_metrics",
        "--metrics_config_store_utilization_metrics",
        "--metrics_config_enable_op_level_tracing",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


def _case_parallelism(case: Mapping[str, Any]) -> GoldenParallelism:
    raw_parallelism = case["case_manifest"].get("parallelism", {})
    if not isinstance(raw_parallelism, dict):
        raise ValueError(f"case {case.get('name')!r} has invalid parallelism metadata")
    return GoldenParallelism(
        attn_tensor_parallel_size=int(
            raw_parallelism.get("attn_tensor_parallel_size", 1)
        ),
        attn_data_parallel_size=int(
            raw_parallelism.get("attn_data_parallel_size", 1)
        ),
        moe_tensor_parallel_size=int(
            raw_parallelism.get("moe_tensor_parallel_size", 1)
        ),
        moe_expert_parallel_size=int(
            raw_parallelism.get("moe_expert_parallel_size", 1)
        ),
        num_pipeline_stages=int(raw_parallelism.get("num_pipeline_stages", 1)),
    )


def _colocation_flags(parallelism: GoldenParallelism) -> list[str]:
    return [
        "--sys_arch",
        "co-location",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_device",
        "h800",
        "--replica_config_attn_tensor_parallel_size",
        str(parallelism.attn_tensor_parallel_size),
        "--replica_config_attn_data_parallel_size",
        str(parallelism.attn_data_parallel_size),
        "--replica_config_moe_tensor_parallel_size",
        str(parallelism.moe_tensor_parallel_size),
        "--replica_config_moe_expert_parallel_size",
        str(parallelism.moe_expert_parallel_size),
        "--replica_config_num_pipeline_stages",
        str(parallelism.num_pipeline_stages),
        "--replica_config_moe_routing_mode",
        "uniform_random",
        "--replica_config_moe_routing_seed",
        "42",
    ]


def _pdd_flags(parallelism: GoldenParallelism) -> list[str]:
    return [
        "--sys_arch",
        "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_cluster_num_replicas",
        "1",
        "--cluster_config_prefill_replica_config_device",
        "h800",
        "--cluster_config_decode_replica_config_device",
        "h800",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
        str(parallelism.attn_tensor_parallel_size),
        "--cluster_config_prefill_replica_config_attn_data_parallel_size",
        str(parallelism.attn_data_parallel_size),
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
        str(parallelism.moe_tensor_parallel_size),
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
        str(parallelism.moe_expert_parallel_size),
        "--cluster_config_prefill_replica_config_num_pipeline_stages",
        str(parallelism.num_pipeline_stages),
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size",
        str(parallelism.attn_tensor_parallel_size),
        "--cluster_config_decode_replica_config_attn_data_parallel_size",
        str(parallelism.attn_data_parallel_size),
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size",
        str(parallelism.moe_tensor_parallel_size),
        "--cluster_config_decode_replica_config_moe_expert_parallel_size",
        str(parallelism.moe_expert_parallel_size),
        "--cluster_config_decode_replica_config_num_pipeline_stages",
        str(parallelism.num_pipeline_stages),
        "--replica_config_device",
        "h800",
        "--replica_config_moe_routing_mode",
        "uniform_random",
        "--replica_config_moe_routing_seed",
        "42",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "200.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.5",
    ]


def build_frontier_command(
    case: Mapping[str, Any],
    *,
    output_root: Path,
    python_bin: str = "python",
    extra_args: Sequence[str] = (),
) -> list[str]:
    case_manifest = case["case_manifest"]
    sys_arch = str(case_manifest["sys_arch"])
    parallelism = _case_parallelism(case)
    command = [python_bin, "-m", "frontier.main"]
    command.extend(_common_frontier_flags(case=case, output_root=output_root))
    if sys_arch == "co-location":
        command.extend(_colocation_flags(parallelism))
    elif sys_arch == "pd-disaggregation":
        command.extend(_pdd_flags(parallelism))
    else:
        raise ValueError(f"Unsupported sys_arch: {sys_arch!r}")
    _append_flags(command, profile_file_flags(case))
    command.extend(extra_args)
    return command


def _expected_output_dir(case: Mapping[str, Any], side: str) -> Path:
    key = f"{side}_dir"
    if key not in case:
        raise ValueError(f"case {case.get('name')!r} is missing {key}")
    return Path(str(case[key]))


def write_simulation_artifact_manifest(case: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    trace_report = normalize_op_traces_jsonl_to_csv(
        output_dir / OP_TRACES_JSONL,
        output_dir / OP_TRACES_CSV,
    )
    artifacts = {
        REQUEST_METRICS_CSV: file_manifest(output_dir / REQUEST_METRICS_CSV),
        OP_TRACES_JSONL: file_manifest(output_dir / OP_TRACES_JSONL),
        OP_TRACES_CSV: file_manifest(output_dir / OP_TRACES_CSV),
        "system_metrics.json": file_manifest(output_dir / "system_metrics.json"),
    }
    manifest = {
        "case_name": case["name"],
        "case_manifest": dict(case["case_manifest"]),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "trace_normalization": trace_report,
    }
    write_json_report(output_dir / "simulation_artifact_manifest.json", manifest)
    return manifest


def run_simulation_case(
    case: Mapping[str, Any],
    *,
    output_root: Path,
    side: str,
    python_bin: str,
    repo_root: Path,
    allow_existing: bool = False,
    dry_run: bool = False,
    extra_args: Sequence[str] = (),
    workload_profiles: Sequence[str] = DEFAULT_WORKLOAD_PROFILES,
) -> dict[str, Any]:
    output_dir = _expected_output_dir(case, side)
    if output_dir.exists() and not allow_existing:
        raise FileExistsError(
            f"refusing to overwrite existing simulation output directory: {output_dir}"
        )
    command = build_frontier_command(
        case,
        output_root=output_root,
        python_bin=python_bin,
        extra_args=extra_args,
    )
    if dry_run:
        return {
            "case_name": case["name"],
            "side": side,
            "command": command,
            "output_dir": str(output_dir),
            "dry_run": True,
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    env["WANDB_DISABLED"] = "true"
    env["VIDUR_DISABLE_WANDB"] = "1"
    completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"simulation failed for case={case['name']} side={side} exit_code={completed.returncode}"
        )
    if not output_dir.is_dir():
        raise FileNotFoundError(f"expected simulation output directory missing: {output_dir}")
    artifact_manifest = write_simulation_artifact_manifest(case, output_dir)
    return {
        "case_name": case["name"],
        "side": side,
        "command": command,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "artifact_manifest": artifact_manifest,
    }


def _select_cases(cases: Sequence[Mapping[str, Any]], case_names: set[str], max_cases: int | None) -> list[Mapping[str, Any]]:
    selected = [case for case in cases if not case_names or str(case["name"]) in case_names]
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("no golden baseline cases selected")
    return selected


def _baseline_report_status(*, dry_run: bool, selected_case_count: int, total_case_count: int) -> str:
    if dry_run:
        return "DRY_RUN"
    if selected_case_count == total_case_count:
        return "PASS"
    return "PARTIAL_PASS"


def run_baseline_generation(
    *,
    config_root: Path,
    profile_root: Path,
    reference_root: Path,
    candidate_root: Path,
    side: str,
    python_bin: str,
    repo_root: Path,
    output_json: Path | None = None,
    case_names: set[str] | None = None,
    max_cases: int | None = None,
    allow_existing: bool = False,
    dry_run: bool = False,
    extra_args: Sequence[str] = (),
    workload_profiles: Sequence[str] = DEFAULT_WORKLOAD_PROFILES,
) -> dict[str, Any]:
    if side not in {"reference", "candidate"}:
        raise ValueError(f"side must be 'reference' or 'candidate', got {side!r}")
    cases, case_manifest = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        config_filenames=GOLDEN_CONFIG_FILENAMES,
        simulation_artifacts=DEFAULT_SIMULATION_ARTIFACTS,
        workload_profiles=workload_profiles,
    )
    selected_cases = _select_cases(cases, case_names or set(), max_cases)
    total_case_count = len(cases)
    selected_case_count = len(selected_cases)
    output_root = reference_root if side == "reference" else candidate_root
    results = [
        run_simulation_case(
            case,
            output_root=output_root,
            side=side,
            python_bin=python_bin,
            repo_root=repo_root,
            allow_existing=allow_existing,
            dry_run=dry_run,
            extra_args=extra_args,
        )
        for case in selected_cases
    ]
    status = _baseline_report_status(
        dry_run=dry_run,
        selected_case_count=selected_case_count,
        total_case_count=total_case_count,
    )
    report = {
        "status": status,
        "side": side,
        "case_manifest": case_manifest,
        "total_case_count": total_case_count,
        "selected_case_count": selected_case_count,
        "is_complete_gate": status == "PASS",
        "selection": {
            "case_names": sorted(case_names or set()),
            "max_cases": max_cases,
        },
        "results": results,
    }
    if output_json is not None:
        write_json_report(output_json, report)
    return report


def _parse_case_names(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated list must contain at least one value")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("data/config/models"))
    parser.add_argument("--profile-root", type=Path, default=Path("data/profiling/compute/h800"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--side", choices=("reference", "candidate"), default="reference")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--case-names", default="")
    parser.add_argument(
        "--workload-profiles",
        default=",".join(DEFAULT_WORKLOAD_PROFILES),
        help=(
            "Comma-separated golden workload profiles. The default preserves "
            "the archived 48-case reference layout; add long_single for "
            "profile-supported supplemental workload-shape generation."
        ),
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    extra_args = tuple(args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args)
    report = run_baseline_generation(
        config_root=args.config_root,
        profile_root=args.profile_root,
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        side=args.side,
        python_bin=args.python_bin,
        repo_root=args.repo_root,
        output_json=args.output_json,
        case_names=_parse_case_names(args.case_names),
        max_cases=args.max_cases,
        allow_existing=args.allow_existing,
        dry_run=args.dry_run,
        extra_args=extra_args,
        workload_profiles=_parse_csv_list(args.workload_profiles),
    )
    return 0 if report["status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
