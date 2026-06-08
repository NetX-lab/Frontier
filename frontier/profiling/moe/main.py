"""
MoE profiling main entry point.

This script profiles MoE compute operations following the linear_op profiling module pattern.

Design principle: EP (expert_parallel_size) is a distribution parameter,
not a compute parameter. We use num_experts_per_device instead.

Usage:
    python -m frontier.profiling.moe.main \
        --models mixtral_8x7b_moe \
        --num_gpus 4 \
        --max_tokens 1024 \
        --num_tensor_parallel_workers 1 2 4 8 \
        --expert_parallel_sizes 1 2 4 8 \
        --device a100 \
        --output_dir data/profiling

Multi-GPU modes:
    1. Ray mode (default): Uses Ray for distributed profiling across GPUs
    2. Non-Ray mode (--disable_ray): Uses torch.multiprocessing for multi-GPU support
       - When --num_gpus > 1: Spawns multiple processes, each bound to a different GPU
       - When --num_gpus = 1: Single GPU sequential execution (original behavior)
"""

from __future__ import annotations

import os
import sys
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import yaml
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from frontier.config.precision_type import PrecisionType
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
)
# Conditionally import ray - only needed when not using --disable_ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.utils import (
    EXPORTABLE_PROFILE_METHOD_CHOICES,
    ProfileMethod,
    build_profile_method_output_path,
    build_profiling_output_path,
    get_num_tokens_to_profile,
    normalize_profile_method,
    profile_method_to_measurement_type,
    require_profiling_dependencies,
)

MoEWrapper = None


def _ensure_torch_available():
    if torch is None:
        raise ImportError(
            "MoE profiling requires torch. Install the dedicated GPU profiling "
            "environment before running this entrypoint."
        )
    return torch


def _get_moe_wrapper_class():
    global MoEWrapper
    if MoEWrapper is None:
        from frontier.profiling.moe.moe_wrapper import MoEWrapper as _MoEWrapper

        MoEWrapper = _MoEWrapper
    return MoEWrapper


def _worker_init(gpu_id: int) -> None:
    """Initialize worker process with specific GPU binding."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # Device 0 within this process's visible devices


# Track CUDA initialization to prevent GPU reassignment within a worker
_CUDA_INITIALIZED = False
_CUDA_GPU_ID = None


def _worker_profile_task(
    task_args: Tuple[int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing profiling.

    Args:
        task_args: Tuple of (gpu_id, model_config_dict, wrapper_args, profile_args)

    Returns:
        Profiling result dictionary
    """
    torch_module = _ensure_torch_available()
    moe_wrapper_class = _get_moe_wrapper_class()

    global _CUDA_INITIALIZED, _CUDA_GPU_ID

    gpu_id, gpu_local_idx, model_config_dict, wrapper_args, profile_args = task_args

    # Bind to the intended GPU within the current visible set without mutating environment
    if not _CUDA_INITIALIZED:
        torch_module.cuda.set_device(gpu_local_idx)
        _CUDA_INITIALIZED = True
        _CUDA_GPU_ID = gpu_id
    elif _CUDA_GPU_ID != gpu_id:
        raise RuntimeError(
            f"Worker initialized with GPU {_CUDA_GPU_ID} but received task for GPU {gpu_id}. "
            f"This indicates a task distribution bug."
        )

    # Reconstruct ModelConfig from dict
    model_config = ModelConfig(**model_config_dict)

    # Create wrapper and run profiling
    wrapper = moe_wrapper_class(
        model_config=model_config,
        num_tensor_parallel_workers=wrapper_args["num_tensor_parallel_workers"],
        expert_parallel_size=wrapper_args["expert_parallel_size"],
        profile_method=wrapper_args["profile_method"],
        rank=0,  # Always rank 0 within this GPU's context
        output_dir=wrapper_args["output_dir"],
        use_vllm_kernel=wrapper_args["use_vllm_kernel"],
        use_fp8=wrapper_args.get("use_fp8", False),
        per_channel_quant=wrapper_args.get("per_channel_quant", False),
        block_shape=wrapper_args.get("block_shape", None),
        routing_runtime_path=wrapper_args.get(
            "routing_runtime_path", "standard_fused_topk"
        ),
        gating_runtime_context=wrapper_args.get(
            "gating_runtime_context", DEFAULT_MOE_GATING_RUNTIME_CONTEXT
        ),
    )

    result = wrapper.profile(
        num_tokens=profile_args["num_tokens"],
        load_distribution=profile_args["load_distribution"],
        seed=profile_args["seed"],
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device SKU (e.g., a100, h100, a40) - required for output path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/profiling",
        help="Root output directory for profiling results (default: data/profiling)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "mixtral_8x7b_moe",
            "qwen2_moe_example",
        ],
        help="MoE models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--expert_parallel_sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Expert parallel sizes to profile (distribution parameter, not compute parameter). "
             "For each EP size, num_experts_per_device = num_experts / expert_parallel_size. "
             "Example: --expert_parallel_sizes 1 2 4 8",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--extra_num_tokens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Additional explicit num_tokens points to include in profiling set. "
            "Useful for filling sparse runtime hotspots (e.g., 3 6 9 12 15)."
        ),
    )
    parser.add_argument(
        "--num_tokens_list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Exact num_tokens points to profile. "
            "When provided, this bypasses the default max_tokens grid."
        ),
    )
    parser.add_argument(
        "--profile_method",
        default="record_function",
        choices=EXPORTABLE_PROFILE_METHOD_CHOICES,
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=list(PrecisionType.__members__.keys()),
        help="Profiling precision type. Defaults to model config dtype when not set.",
    )
    parser.add_argument(
        "--dtype",
        dest="precision",
        type=str,
        choices=list(PrecisionType.__members__.keys()),
        help="Alias for --precision.",
    )
    parser.add_argument(
        "--enable_load_imbalance",
        action="store_true",
        default=True,
        help="Enable load imbalance profiling (default: enabled). "
             "When enabled, uses vLLM fused_moe_kernel for accurate profiling.",
    )
    parser.add_argument(
        "--disable_load_imbalance",
        dest="enable_load_imbalance",
        action="store_false",
        help="Disable load imbalance profiling.",
    )
    parser.add_argument(
        "--load_distributions",
        nargs="+",
        default=["uniform"],
        choices=["uniform", "skewed", "extremely_skewed"],
        help="Load distribution types to profile when --enable_load_imbalance is set "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--num_samples_per_distribution",
        type=int,
        default=3,
        help="Number of random samples per load distribution configuration "
             "(default: %(default)s)",
    )
    # FP8 quantization profiling arguments
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        default=None,
        help="Enable FP8 W8A8 quantization profiling. Defaults to model config when not set.",
    )
    parser.add_argument(
        "--per_channel_quant",
        action="store_true",
        default=False,
        help="Use per-channel quantization instead of per-tensor (only effective with --use_fp8).",
    )
    parser.add_argument(
        "--block_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Block dimensions for block-wise quantization (e.g., --block_shape 128 128). "
             "Defaults to model config when FP8 is enabled.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        dest="skip_confirmation",
        help="Skip interactive confirmation and proceed directly with profiling",
    )
    parser.add_argument(
        "--disable_replicated",
        action="store_true",
        help=(
            "Deprecated for MoE profiling. Canonical MoE profiling now always keeps "
            "all ops on the requested TP rows."
        ),
    )
    parser.add_argument(
        "--routing_runtime_path",
        type=str,
        default="standard_fused_topk",
        choices=["standard_fused_topk", "uniform_topk"],
        help=(
            "Routing runtime path to profile for moe_gating_routing_topk. "
            "Use uniform_topk for runtime-equivalent uniform-routing profiling."
        ),
    )
    parser.add_argument(
        "--gating_runtime_context",
        type=str,
        default=DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
        choices=["standalone_legacy", "prefill_hot"],
        help=(
            "Runtime-context variant for moe_gating profiling. "
            "prefill_hot enables a prefill-only hot-context prefix before timing."
        ),
    )
    args = parser.parse_args()
    args.profile_method = normalize_profile_method(args.profile_method)

    return args, args.enable_load_imbalance


def _precision_to_torch_dtype(precision: str) -> torch.dtype:
    torch_module = _ensure_torch_available()
    precision = precision.upper()
    if precision == "FP16":
        return torch_module.float16
    if precision == "BF16":
        return torch_module.bfloat16
    if precision == "FP32":
        return torch_module.float32
    raise ValueError(f"Unsupported precision type: {precision}")


def _resolve_precision_for_model(
    model_config: ModelConfig, requested_precision: Optional[str], model: str
) -> Tuple[torch.dtype, str]:
    config_precision = ModelConfig._dtype_to_str(model_config.dtype)
    if requested_precision is None:
        return model_config.dtype, config_precision
    requested = str(requested_precision).upper()
    if requested != config_precision:
        raise ValueError(
            f"Profiling precision mismatch for {model}: requested={requested}, model_config={config_precision}"
        )
    return _precision_to_torch_dtype(requested), config_precision


def _resolve_model_arch_for_metadata(model_config: ModelConfig) -> str:
    model_arch = getattr(model_config, "model_arch", None)
    if model_arch is None:
        return "generic"
    model_arch = str(model_arch).strip()
    if model_arch == "":
        return "generic"
    return model_arch


def _resolve_quant_signature_for_metadata(model_config: ModelConfig) -> str:
    quant_signature = model_config.get_quant_signature()
    if quant_signature is None:
        return "none"
    quant_signature = str(quant_signature).strip()
    if quant_signature == "":
        return "none"
    return quant_signature


def _resolve_fp8_settings(
    model_config: ModelConfig,
    use_fp8: Optional[bool],
    block_shape: Optional[List[int]],
    model: str,
) -> Tuple[bool, Optional[List[int]]]:
    config_method = None
    config_block_shape = None
    quant_config = model_config.quantization_config
    if quant_config is not None:
        config_method = quant_config.quant_method
        if config_method == "fp8" and quant_config.weight_block_size is not None:
            config_block_shape = list(quant_config.weight_block_size)
    config_use_fp8 = config_method == "fp8"

    if use_fp8 is not None or block_shape is not None:
        requested_use_fp8 = use_fp8 if use_fp8 is not None else config_use_fp8
        requested_block_shape = block_shape if block_shape is not None else config_block_shape
        if requested_use_fp8 != config_use_fp8 or requested_block_shape != config_block_shape:
            raise ValueError(
                f"FP8 quantization config mismatch for {model}: requested=(use_fp8={use_fp8}, block_shape={block_shape}), "
                f"model_config=(quant_method={config_method}, block_shape={config_block_shape})"
            )
    return config_use_fp8, config_block_shape


MOE_REQUIRED_TARGET_COLUMNS = (
    "time_stats.moe_gating_linear.median",
    "time_stats.moe_gating_routing_topk.median",
    "time_stats.moe_shuffling.median",
    "time_stats.moe_grouped_gemm.median",
)


def _validate_canonical_moe_result_df(df: pd.DataFrame, *, model: str) -> None:
    missing_columns = [col for col in MOE_REQUIRED_TARGET_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "MoE profiling contract validation failed: canonical MoE profiling "
            f"must emit target columns {list(MOE_REQUIRED_TARGET_COLUMNS)}, but "
            f"model={model} is missing {missing_columns}. This usually indicates "
            "a legacy split-row path or broken profiling aggregation."
        )

    broken_mask = df[list(MOE_REQUIRED_TARGET_COLUMNS)].isna().any(axis=1)
    if not broken_mask.any():
        return

    broken_rows = df.loc[
        broken_mask,
        [
            "num_tensor_parallel_workers",
            "expert_parallel_size",
            "num_tokens",
            *MOE_REQUIRED_TARGET_COLUMNS,
        ],
    ].copy()
    broken_rows = broken_rows.sort_values(
        ["num_tensor_parallel_workers", "expert_parallel_size", "num_tokens"]
    )

    preview_lines = []
    for _, row in broken_rows.head(8).iterrows():
        missing_targets = [
            col.removeprefix("time_stats.").removesuffix(".median")
            for col in MOE_REQUIRED_TARGET_COLUMNS
            if pd.isna(row[col])
        ]
        preview_lines.append(
            f"tp={int(row['num_tensor_parallel_workers'])}, "
            f"ep={int(row['expert_parallel_size'])}, "
            f"tokens={int(row['num_tokens'])}, "
            f"missing={missing_targets}"
        )

    raise ValueError(
        "MoE profiling contract validation failed: canonical MoE profiling must "
        "emit all target columns on every row. Encountered NaN target values "
        f"for model={model}. This usually indicates a legacy split-row path or "
        "broken profiling aggregation. First broken rows: "
        + "; ".join(preview_lines)
    )


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any, use_vllm_kernel: bool
):
    """
    Profile all MoE operations for a given model.

    This function performs grid-search profiling over:
    - Tensor parallel sizes (TP)
    - Expert parallel sizes (EP)
    - Token counts

    For each (TP, EP) combination, it calculates num_experts_per_device and profiles.

    Multi-GPU support:
    - Ray mode: Uses Ray actors for distributed profiling
    - Non-Ray mode with num_gpus > 1: Uses ProcessPoolExecutor for multi-GPU profiling
    - Non-Ray mode with num_gpus = 1: Sequential single-GPU profiling
    """
    from frontier.profiling.moe.moe_impl import get_routing_runtime_metadata
    moe_wrapper_class = _get_moe_wrapper_class()

    model_config = ModelConfig.from_model_name(model)
    _, _ = _resolve_precision_for_model(model_config, args.precision, model)
    resolved_use_fp8, resolved_block_shape = _resolve_fp8_settings(
        model_config, args.use_fp8, args.block_shape, model
    )

    # Validate that this is a MoE model
    if not model_config.is_moe:
        raise ValueError(f"Model {model} is not a MoE model (is_moe=False)")

    all_results = []
    measurement_type = profile_method_to_measurement_type(args.profile_method).value

    # Determine available GPUs for non-Ray mode
    available_gpus = _get_available_gpus(args.num_gpus)
    actual_num_gpus = len(available_gpus)
    if actual_num_gpus < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs but only found {actual_num_gpus} visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}). "
            "Please adjust --num_gpus or CUDA_VISIBLE_DEVICES."
        )
    routing_runtime_path = getattr(args, "routing_runtime_path", "standard_fused_topk")
    routing_runtime_metadata = get_routing_runtime_metadata(routing_runtime_path)

    if args.disable_ray and actual_num_gpus > 1:
        print(f"  Multi-GPU mode: Using {actual_num_gpus} GPUs ({available_gpus})")
    elif args.disable_ray:
        print(f"  Single-GPU mode: Using GPU {available_gpus[0]}")

    # Create Ray actor factory if using Ray
    if not args.disable_ray:
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Use --disable_ray flag.")
        moe_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            moe_wrapper_class,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
        ray_promises = []

    # Nested loop: TP x EP
    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            # Skip TP > 1 for models that don't support tensor parallelism
            pbar.update(len(args.expert_parallel_sizes) * len(num_tokens_to_profile))
            continue

        def _collect_result(result):
            """Append one canonical profiling result row."""
            result = dict(result)
            result["measurement_type"] = measurement_type
            all_results.append(result)

        for expert_parallel_size in args.expert_parallel_sizes:
            # Validate EP size
            if model_config.num_experts % expert_parallel_size != 0:
                raise ValueError(
                    f"num_experts ({model_config.num_experts}) must be divisible by "
                    f"expert_parallel_size ({expert_parallel_size})"
                )

            # Generate test cases based on load imbalance settings
            if args.enable_load_imbalance:
                test_cases = []
                for num_tokens in num_tokens_to_profile:
                    for load_dist in args.load_distributions:
                        for seed in range(args.num_samples_per_distribution):
                            test_cases.append({
                                "num_tokens": num_tokens,
                                "load_distribution": load_dist,
                                "seed": seed,
                            })
            else:
                test_cases = [
                    {"num_tokens": num_tokens, "load_distribution": "uniform", "seed": None}
                    for num_tokens in num_tokens_to_profile
                ]

            # Common wrapper arguments for this (TP, EP) combination
            wrapper_args = {
                "num_tensor_parallel_workers": num_tensor_parallel_workers,
                "expert_parallel_size": expert_parallel_size,
                "profile_method": args.profile_method,
                "output_dir": args.output_dir,
                "use_vllm_kernel": use_vllm_kernel,
                "use_fp8": resolved_use_fp8,
                "per_channel_quant": args.per_channel_quant,
                "block_shape": resolved_block_shape,
            }
            if hasattr(args, "routing_runtime_path"):
                wrapper_args["routing_runtime_path"] = args.routing_runtime_path
            if hasattr(args, "gating_runtime_context"):
                wrapper_args["gating_runtime_context"] = args.gating_runtime_context

            # Convert model_config to dict for multiprocessing serialization
            model_config_dict = model_config.to_dict()

            if not args.disable_ray:
                # Ray mode: create actors and submit tasks
                model_wrappers = []
                for rank in range(args.num_gpus):
                    ray_wrapper_init_args = [
                        model_config,
                        num_tensor_parallel_workers,
                        expert_parallel_size,
                        args.profile_method,
                        rank,
                        args.output_dir,
                        use_vllm_kernel,
                        resolved_use_fp8,
                        args.per_channel_quant,
                        resolved_block_shape,
                    ]
                    if hasattr(args, "routing_runtime_path"):
                        ray_wrapper_init_args.append(args.routing_runtime_path)
                    if hasattr(args, "gating_runtime_context"):
                        ray_wrapper_init_args.append(args.gating_runtime_context)
                    model_wrappers.append(
                        moe_wrapper_actor.remote(*ray_wrapper_init_args)
                    )

                for test_case in test_cases:
                    worker_id = len(ray_promises) % args.num_gpus
                    promise = model_wrappers[worker_id].profile.remote(
                        num_tokens=test_case["num_tokens"],
                        load_distribution=test_case["load_distribution"],
                        seed=test_case["seed"],
                    )
                    ray_promises.append(promise)

                    if len(ray_promises) >= args.num_gpus:
                        results = ray.get(ray_promises)
                        for r in results:
                            _collect_result(r)
                        ray_promises = []
                    pbar.update(1)

            elif actual_num_gpus > 1:
                # Non-Ray multi-GPU mode: use ProcessPoolExecutor
                tasks_by_gpu = {gpu_id: [] for gpu_id in available_gpus}
                for idx, test_case in enumerate(test_cases):
                    gpu_id = available_gpus[idx % actual_num_gpus]
                    gpu_local_idx = available_gpus.index(gpu_id)
                    tasks_by_gpu[gpu_id].append(
                        (gpu_id, gpu_local_idx, model_config_dict, wrapper_args, test_case)
                    )

                # One executor per GPU to keep worker-to-GPU affinity stable.
                ctx = mp.get_context("spawn")
                executors = {
                    gpu_id: ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                    for gpu_id in available_gpus
                }
                futures = []
                try:
                    for gpu_id, gpu_tasks in tasks_by_gpu.items():
                        executor = executors[gpu_id]
                        for task in gpu_tasks:
                            futures.append(executor.submit(_worker_profile_task, task))
                    for future in as_completed(futures):
                        result = future.result()
                        _collect_result(result)
                        pbar.update(1)
                finally:
                    for executor in executors.values():
                        executor.shutdown(wait=True)
            else:
                # Single-GPU sequential mode
                import torch
                os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
                torch.cuda.set_device(0)

                wrapper_kwargs = {
                    "model_config": model_config,
                    "num_tensor_parallel_workers": num_tensor_parallel_workers,
                    "expert_parallel_size": expert_parallel_size,
                    "profile_method": args.profile_method,
                    "rank": 0,
                    "output_dir": args.output_dir,
                    "use_vllm_kernel": use_vllm_kernel,
                    "use_fp8": resolved_use_fp8,
                    "per_channel_quant": args.per_channel_quant,
                    "block_shape": resolved_block_shape,
                }
                if hasattr(args, "routing_runtime_path"):
                    wrapper_kwargs["routing_runtime_path"] = args.routing_runtime_path
                if hasattr(args, "gating_runtime_context"):
                    wrapper_kwargs["gating_runtime_context"] = (
                        args.gating_runtime_context
                    )

                wrapper = moe_wrapper_class(**wrapper_kwargs)

                for test_case in test_cases:
                    result = wrapper.profile(
                        num_tokens=test_case["num_tokens"],
                        load_distribution=test_case["load_distribution"],
                        seed=test_case["seed"],
                    )
                    _collect_result(result)
                    pbar.update(1)

    # Collect remaining Ray results
    if not args.disable_ray and ray_promises:
        results = ray.get(ray_promises)
        for r in results:
            result = dict(r)
            result["measurement_type"] = measurement_type
            all_results.append(result)

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    for key, value in routing_runtime_metadata.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna(value)

    _validate_canonical_moe_result_df(df, model=model)
    return df


def _get_available_gpus(num_gpus: int) -> List[int]:
    """
    Get list of available GPU IDs based on CUDA_VISIBLE_DEVICES and num_gpus.

    Returns:
        List of GPU IDs to use for profiling
    """
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES
        available = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    else:
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                available = [
                    int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()
                ]
            else:
                raise RuntimeError(
                    "Unable to discover GPUs with nvidia-smi. Set "
                    "CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi before "
                    "running MoE profiling. "
                    f"nvidia-smi stderr: {result.stderr.strip()}"
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because nvidia-smi was "
                "not found. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running MoE profiling."
            ) from exc

        if not available:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because it returned no "
                "GPU indices. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running MoE profiling."
            )

    if len(available) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only found {len(available)} visible GPUs "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})."
        )

    return available[:num_gpus]


def main():
    args, use_vllm_kernel_default = parse_args()
    require_profiling_dependencies("moe", ("torch", "vllm", "triton"))
    ray_initialized_by_main = False

    try:
        # Display execution mode
        if args.disable_ray:
            print(f"\n🔧 Execution mode: Non-Ray (multiprocessing)")
            print(f"   Requested GPUs: {args.num_gpus}")
        else:
            print(f"\n🔧 Execution mode: Ray")
            print(f"   Requested GPUs: {args.num_gpus}")

            # Check Ray availability
            if not RAY_AVAILABLE:
                raise RuntimeError("Ray is not available. Use --disable_ray flag.")
            else:
                # Initialize Ray with dashboard disabled to avoid agent issues
                if not ray.is_initialized():
                    try:
                        ray.init(
                            include_dashboard=False,
                            configure_logging=False,
                            logging_level="warning",
                        )
                        ray_initialized_by_main = True
                        print("   ✓ Ray initialized successfully")
                    except Exception as e:
                        raise RuntimeError(
                            "Ray initialization failed. Use --disable_ray for explicit "
                            "non-Ray profiling mode after verifying that this mode is "
                            "intended for the run."
                        ) from e

            available_gpus = _get_available_gpus(args.num_gpus)
            actual_num_gpus = len(available_gpus)

            if actual_num_gpus < args.num_gpus:
                raise RuntimeError(
                    f"Requested {args.num_gpus} GPUs but only found {actual_num_gpus} visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}). "
                    "Please adjust --num_gpus or CUDA_VISIBLE_DEVICES."
                )

        # Create canonical output directory structure:
        # data/profiling/<type>/<hardware>/<model_name>/<op_name>.csv
        device_dir = Path(args.output_dir) / "compute" / args.device
        device_dir.mkdir(parents=True, exist_ok=True)

        # Save config to device directory
        with (device_dir / "moe_config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.dump(vars(args), config_file)

        num_tokens_to_profile = get_num_tokens_to_profile(
            args.max_tokens,
            extra_num_tokens=args.extra_num_tokens,
            num_tokens_list=args.num_tokens_list,
        )

        # Interactive confirmation before profiling
        from frontier.profiling.utils.confirmation import (
            confirm_profiling_execution,
            build_moe_config_sections,
        )

        # Load first model config for confirmation display
        first_model = args.models[0]
        first_model_config = ModelConfig.from_model_name(first_model)
        torch_dtype, precision_str = _resolve_precision_for_model(
            first_model_config, args.precision, first_model
        )

        config_sections = build_moe_config_sections(
            args=args,
            model_config=first_model_config,
            num_tokens_count=len(num_tokens_to_profile),
            use_vllm_kernel=use_vllm_kernel_default,
            precision_str=precision_str,
            torch_dtype=torch_dtype,
        )

        if not confirm_profiling_execution(
            module_name="MoE",
            config_sections=config_sections,
            skip_confirmation=args.skip_confirmation,
        ):
            sys.exit(0)

        # Calculate total combinations: models x num_tokens x TP x EP

        if args.enable_load_imbalance:
            total_combos = itertools.product(
                args.models,
                num_tokens_to_profile,
                args.num_tensor_parallel_workers,
                args.expert_parallel_sizes,
                args.load_distributions,
                range(args.num_samples_per_distribution),
            )

        else:
            total_combos = itertools.product(
                args.models,
                num_tokens_to_profile,
                args.num_tensor_parallel_workers,
                args.expert_parallel_sizes,
            )

        pbar = tqdm(total=len(list(total_combos)))

        for model in args.models:
            # Load model config for metadata
            model_config = ModelConfig.from_model_name(model)
            _, precision_str = _resolve_precision_for_model(model_config, args.precision, model)
            result_df = profile_model(
                args,
                model,
                num_tokens_to_profile,
                pbar,
                use_vllm_kernel_default,
            )
            output_file = build_profile_method_output_path(
                output_root=args.output_dir,
                profiling_type="compute",
                hardware=args.device,
                model_name=model,
                op_name="moe",
                profile_method=args.profile_method,
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_df["profiling_precision"] = precision_str
            result_df["measurement_type"] = profile_method_to_measurement_type(args.profile_method).value
            model_arch = _resolve_model_arch_for_metadata(model_config)
            quant_signature = _resolve_quant_signature_for_metadata(model_config)

            if "model_arch" not in result_df.columns:
                result_df["model_arch"] = model_arch
            else:
                result_df["model_arch"] = (
                    result_df["model_arch"]
                    .replace(r"^\s*$", model_arch, regex=True)
                    .fillna(model_arch)
                )

            if "quant_signature" not in result_df.columns:
                result_df["quant_signature"] = quant_signature
            else:
                result_df["quant_signature"] = (
                    result_df["quant_signature"]
                    .replace(r"^\s*$", quant_signature, regex=True)
                    .fillna(quant_signature)
                )
            result_df.to_csv(output_file, index=False)
            print(f"✓ Saved MoE profiling data to: {output_file}")
    finally:
        if ray_initialized_by_main and ray is not None and ray.is_initialized():
            print("[INFO] Shutting down Ray runtime for MoE profiling cleanup.")
            ray.shutdown()


if __name__ == "__main__":
    main()
