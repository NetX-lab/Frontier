"""
Linear Operations Profiling Main Entry Point.

This module provides the main entry point for profiling linear operations
(MLP, LayerNorm, projections, etc.) for LLM inference simulation.

Multi-GPU modes:
    1. Ray mode (default): Uses Ray for distributed profiling across GPUs
    2. Non-Ray mode (--disable_ray): Uses torch.multiprocessing for multi-GPU support
       - When --num_gpus > 1: Spawns multiple processes, each bound to a different GPU
       - When --num_gpus = 1: Single GPU sequential execution (original behavior)

Usage:
    # Single GPU mode (recommended for small tasks)
    python -m frontier.profiling.linear_op.main \
        --models meta-llama/Llama-2-7b-hf \
        --num_gpus 1 \
        --disable_ray \
        --max_tokens 1024 \
        --num_tensor_parallel_workers 1 2

    # Multi-GPU mode with Ray
    python -m frontier.profiling.linear_op.main \
        --models meta-llama/Llama-2-7b-hf \
        --num_gpus 2 \
        --max_tokens 1024 \
        --num_tensor_parallel_workers 1 2

Note: When --is_moe is set, MLP-specific profiling operations are skipped
because MoE models use expert layers instead of dense MLP layers.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from frontier.config.precision_type import PrecisionType
from frontier.profiling.utils.replicated_ops import (
    deduplicate_tp1_rows,
    split_replicated_result,
)
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

# Conditionally import ray - only needed when not using --disable_ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.ray_setup_hook import (
    disable_ray_datasets_serializers,
)
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.operators.families import (
    FFN_FAMILY,
    get_family_profiling_name_set,
    get_operator_family,
    iter_operator_families,
)
from frontier.operators.typed_contracts import (
    serialize_typed_operator_contract_column,
    validate_typed_operator_metadata,
)
from frontier.profiling.common.typed_contracts import parse_typed_operator_contracts
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


def _ensure_torch_available():
    if torch is None:
        raise ImportError(
            "Linear-op profiling requires torch. Install the dedicated GPU profiling "
            "environment before running this entrypoint."
        )
    return torch


def _get_available_gpus(num_gpus: int) -> List[int]:
    """
    Get list of available GPU IDs based on CUDA_VISIBLE_DEVICES and num_gpus.

    Returns:
        List of GPU IDs to use for profiling
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
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
                    "running linear-op profiling. "
                    f"nvidia-smi stderr: {result.stderr.strip()}"
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because nvidia-smi was "
                "not found. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running linear-op profiling."
            ) from exc

        if not available:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because it returned no "
                "GPU indices. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running linear-op profiling."
            )

    if len(available) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only found {len(available)} visible GPUs "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})."
        )

    return available[:num_gpus]


# Global variable to track CUDA initialization in worker process
_CUDA_INITIALIZED = False
_CUDA_GPU_ID = None
_CUDA_GPU_LOCAL_IDX = None


def _worker_init(gpu_id: int, gpu_local_idx: int):
    """
    Initialize worker process with specific GPU binding.
    Called once when the worker is created.
    """
    global _CUDA_INITIALIZED, _CUDA_GPU_ID, _CUDA_GPU_LOCAL_IDX
    torch_module = _ensure_torch_available()

    if _CUDA_INITIALIZED:
        if _CUDA_GPU_ID != gpu_id:
            raise RuntimeError(
                f"Worker already initialized with GPU {_CUDA_GPU_ID}, cannot reinitialize for GPU {gpu_id}."
            )
        return

    torch_module.cuda.set_device(gpu_local_idx)
    _CUDA_INITIALIZED = True
    _CUDA_GPU_ID = gpu_id
    _CUDA_GPU_LOCAL_IDX = gpu_local_idx


def _worker_profile_linear_op_task(
    task_args: Tuple[Dict[str, Any], int, int, int]
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing linear op profiling.

    Args:
        task_args: Tuple of (wrapper_args, num_tokens, gpu_id, gpu_local_idx)

    Returns:
        Profiling result dictionary
    """
    global _CUDA_INITIALIZED, _CUDA_GPU_ID, _CUDA_GPU_LOCAL_IDX
    torch_module = _ensure_torch_available()
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    wrapper_args, num_tokens, gpu_id, gpu_local_idx = task_args

    if not _CUDA_INITIALIZED:
        raise RuntimeError(
            f"Worker received linear-op task for GPU {gpu_id} before GPU initializer ran."
        )
    if _CUDA_GPU_ID != gpu_id:
        raise RuntimeError(
            f"Worker initialized with GPU {_CUDA_GPU_ID} but received task for GPU {gpu_id}. "
            f"This indicates a task distribution bug."
        )
    if _CUDA_GPU_LOCAL_IDX != gpu_local_idx:
        raise RuntimeError(
            f"Worker initialized with local index {_CUDA_GPU_LOCAL_IDX} but received task for local index {gpu_local_idx}."
        )
    if torch_module.cuda.current_device() != gpu_local_idx:
        raise RuntimeError(
            f"Worker current device {torch_module.cuda.current_device()} does not match expected local index {gpu_local_idx}."
        )

    # Reconstruct ModelConfig from dict
    model_config = ModelConfig(**wrapper_args["model_config_dict"])

    # Create wrapper and run profiling
    wrapper = LinearOpWrapper(
        model_config=model_config,
        num_tensor_parallel_workers=wrapper_args["num_tensor_parallel_workers"],
        profile_method=wrapper_args["profile_method"],
        rank=0,  # rank is 0 within this single-GPU process
        output_dir=wrapper_args["output_dir"],
        profiling_plan=wrapper_args.get("profiling_plan"),
    )

    result = wrapper.profile(num_tokens)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Linear Operations Profiling")
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
        "--output_dir",
        type=str,
        default="data/profiling",
        help="Root output directory for profiling results (default: data/profiling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="unknown",
        help="Hardware SKU for output path (e.g., a100, h100, a800, rtx_pro_6000)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--attn_tp",
        type=int,
        nargs="+",
        default=None,
        help="Attention TP sizes to profile (defaults to --num_tensor_parallel_workers)",
    )
    parser.add_argument(
        "--ffn_tp",
        type=int,
        nargs="+",
        default=None,
        help="FFN TP sizes to profile (defaults to --num_tensor_parallel_workers)",
    )
    parser.add_argument(
        "--moe_tp",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Routed MoE TP sizes to profile (defaults to --ffn_tp or "
            "--num_tensor_parallel_workers)"
        ),
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
            "Useful for filling sparse runtime hotspots (e.g., 1032 1048)."
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
        "--include_target_embedded_mtp",
        action="store_true",
        help=(
            "Also profile target-embedded MTP compute families "
            "(mtp_fusion_proj, lm_head_linear)."
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
        "--use_fp8",
        action="store_true",
        default=None,
        help="Enable FP8 W8A8 quantization profiling. Defaults to model config when not set.",
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
        "--is_moe",
        action="store_true",
        help="Skip dense MLP ops profiling (MoE models use expert layers profiled by the MoE module)",
    )
    parser.add_argument(
        "--ray_enable_datasets_serializers",
        action="store_true",
        help=(
            "Enable Ray dataset serializers (imports pyarrow). "
            "Default behavior disables datasets serializers to avoid "
            "libstdc++/pyarrow conflicts in profiling."
        ),
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
            "Disable profiling for replicated ops "
            "(input_layernorm, post_attention_layernorm, add, emb)."
        ),
    )
    args = parser.parse_args()
    args.profile_method = normalize_profile_method(args.profile_method)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def _resolve_tp_ranges(
    args: argparse.Namespace,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Resolve independent attention, FFN, MoE TP ranges and loop values."""
    attn_tp = args.attn_tp or args.num_tensor_parallel_workers
    ffn_tp = args.ffn_tp or args.num_tensor_parallel_workers
    moe_tp = args.moe_tp or ffn_tp
    all_tps = set(attn_tp + ffn_tp + moe_tp)
    return attn_tp, ffn_tp, moe_tp, sorted(all_tps)


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
    model_config: ModelConfig, requested_precision: str, model: str
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


def _attach_linear_op_output_metadata(
    df: pd.DataFrame,
    *,
    precision_str: str,
    model_arch: str,
    model_architecture_profile: str,
    quant_signature: str,
    measurement_type: str,
) -> pd.DataFrame:
    """Attach required profiling metadata columns to a linear-op dataframe."""
    if df.empty:
        return df
    output_df = df.copy()

    _fill_metadata_column(output_df, "profiling_precision", precision_str)
    _fill_metadata_column(output_df, "measurement_type", measurement_type)
    _fill_metadata_column(output_df, "model_arch", model_arch)
    _fill_metadata_column(
        output_df,
        "model_architecture_profile",
        model_architecture_profile,
    )
    _fill_metadata_column(output_df, "quant_signature", quant_signature)
    return output_df


def _serialize_linear_op_output(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize profile-owned typed contracts before writing a CSV."""

    return serialize_typed_operator_contract_column(df)


def _fill_metadata_column(
    output_df: pd.DataFrame,
    column_name: str,
    expected_value: str,
) -> None:
    """Fill blank metadata cells and fail fast on conflicting non-blank values."""

    if column_name not in output_df.columns:
        output_df[column_name] = expected_value
        return

    normalized = (
        output_df[column_name]
        .replace(r"^\s*$", pd.NA, regex=True)
        .fillna(expected_value)
    )
    conflicting_values = sorted(
        {
            str(value)
            for value in normalized.dropna().unique()
            if str(value) != expected_value
        }
    )
    if conflicting_values:
        raise ValueError(
            f"{column_name} contains conflicting metadata values "
            f"{conflicting_values}; expected {expected_value!r}."
        )
    output_df[column_name] = normalized


def _split_linear_op_result(
    result: Dict[str, Any],
    replicated_op_names: set[str],
    model_config: ModelConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split one linear-op result without collapsing typed width metadata.

    New producer rows carry ``typed_operator_contracts`` and already contain
    the plan-selected compatibility width.  Leave that value untouched while
    partitioning the operator metadata with the timing stats.  Legacy rows do
    not carry the typed field, so retain their historical unpadded scalar
    fallback.
    """

    split_kwargs: Dict[str, Any] = {
        "unpadded_n_embd": model_config.embedding_dim,
    }
    if result.get("typed_operator_contracts") is None:
        split_kwargs["unpadded_n_expanded_embd"] = model_config.mlp_hidden_dim
    return split_replicated_result(
        result,
        replicated_op_names,
        **split_kwargs,
    )


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


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    """
    Profile linear operations for a given model.

    Multi-GPU support:
    - Ray mode: Uses Ray actors for distributed profiling
    - Non-Ray mode with num_gpus > 1: Uses ProcessPoolExecutor for multi-GPU profiling
    - Non-Ray mode with num_gpus = 1: Sequential single-GPU profiling
    """
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    model_config = ModelConfig.from_model_name(model)
    dtype, _ = _resolve_precision_for_model(model_config, args.precision, model)
    _resolve_fp8_settings(model_config, args.use_fp8, args.block_shape, model)

    all_results = []
    measurement_type = profile_method_to_measurement_type(args.profile_method).value

    # Determine available GPUs for non-Ray mode
    available_gpus = _get_available_gpus(args.num_gpus)
    actual_num_gpus = len(available_gpus)

    if actual_num_gpus < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs but only found {actual_num_gpus} visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}). "
            "Please adjust --num_gpus or CUDA_VISIBLE_DEVICES."
        )

    # Convert model_config to dict for multiprocessing serialization
    model_config_dict = model_config.to_dict()
    model_output_dir = build_profile_method_output_path(
        output_root=args.output_dir,
        profiling_type="compute",
        hardware=args.device,
        model_name=model,
        op_name="linear_op",
        profile_method=args.profile_method,
    ).parent

    attn_tp, ffn_tp, moe_tp, all_tps = _resolve_tp_ranges(args)

    for num_tensor_parallel_workers in all_tps:
        profiling_plan = build_profiling_plan(
            model_config=model_config,
            tp_size=num_tensor_parallel_workers,
            attn_tp=attn_tp,
            ffn_tp=ffn_tp,
            moe_tp=moe_tp,
            disable_replicated=args.disable_replicated,
            is_moe=args.is_moe,
            include_target_embedded_mtp=args.include_target_embedded_mtp,
        )

        if not profiling_plan["enabled_ops"]:
            skip_reasons = profiling_plan.get("skip_reasons", [])
            reason_text = "; ".join(skip_reasons) if skip_reasons else "no enabled ops"
            print(
                f"[WARNING] Skipping TP={num_tensor_parallel_workers} for model {model}: {reason_text}"
            )
            pbar.update(len(num_tokens_to_profile))
            continue

        # Determine whether to split replicated ops into separate TP=1 rows.
        _replicated_op_names = set(profiling_plan.get("replicated_ops", []))
        _should_split = (
            num_tensor_parallel_workers > 1
            and _replicated_op_names
            and not args.disable_replicated
        )

        def _collect_result(result):
            """Append result to all_results, splitting replicated ops if needed."""
            result = dict(result)
            result["measurement_type"] = measurement_type
            if _should_split:
                sharded_row, replicated_row = _split_linear_op_result(
                    result,
                    _replicated_op_names,
                    model_config,
                )
                all_results.append(sharded_row)
                all_results.append(replicated_row)
            else:
                all_results.append(result)

        # Common wrapper arguments
        wrapper_args = {
            "model_config_dict": model_config_dict,
            "num_tensor_parallel_workers": num_tensor_parallel_workers,
            "profile_method": args.profile_method,
            "output_dir": str(model_output_dir),
            "profiling_plan": profiling_plan,
        }

        if not args.disable_ray:
            # Ray mode
            if not RAY_AVAILABLE:
                raise RuntimeError("Ray is not available. Use --disable_ray flag.")

            promises = []
            runtime_env = {"env_vars": {"KINETO_LOG_LEVEL": "5"}}
            if not args.ray_enable_datasets_serializers:
                disable_ray_datasets_serializers()
                runtime_env["worker_process_setup_hook"] = (
                    "frontier.profiling.linear_op.ray_setup_hook."
                    "disable_ray_datasets_serializers"
                )
            model_wrapper_actor = ray.remote(
                num_cpus=1,
                num_gpus=1,
            )(
                LinearOpWrapper,
            ).options(runtime_env=runtime_env)

            model_wrappers = [
                model_wrapper_actor.remote(
                    model_config,
                    num_tensor_parallel_workers,
                    args.profile_method,
                    rank,
                    str(model_output_dir),
                    profiling_plan,
                )
                for rank in range(args.num_gpus)
            ]
            for num_tokens in num_tokens_to_profile:
                worker_id = len(promises)
                promise = model_wrappers[worker_id].profile.remote(
                    num_tokens,
                )
                promises.append(promise)

                if len(promises) >= args.num_gpus:
                    results = ray.get(promises)
                    for r in results:
                        _collect_result(r)
                    promises = []

                pbar.update(1)

            if promises:
                results = ray.get(promises)
                for r in results:
                    _collect_result(r)

        elif actual_num_gpus > 1:
            # Non-Ray multi-GPU mode: use multiple single-worker executors
            # Each executor is bound to a specific GPU
            gpu_local_idx_map = {
                gpu_id: local_idx for local_idx, gpu_id in enumerate(available_gpus)
            }

            # Distribute tasks across GPUs (round-robin)
            tasks_by_gpu = {gpu_id: [] for gpu_id in available_gpus}
            for idx, num_tokens in enumerate(num_tokens_to_profile):
                gpu_id = available_gpus[idx % actual_num_gpus]
                tasks_by_gpu[gpu_id].append(
                    (wrapper_args, num_tokens, gpu_id, gpu_local_idx_map[gpu_id])
                )

            ctx = mp.get_context('spawn')
            
            # Create one executor per GPU, each with a single worker
            executors = []
            all_futures = []
            future_to_task = {}
            try:
                for gpu_id in available_gpus:
                    executor = ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=ctx,
                        initializer=_worker_init,
                        initargs=(gpu_id, gpu_local_idx_map[gpu_id]),
                    )
                    executors.append(executor)
                    
                    # Submit all tasks for this GPU to its dedicated executor
                    for task in tasks_by_gpu[gpu_id]:
                        future = executor.submit(_worker_profile_linear_op_task, task)
                        all_futures.append(future)
                        future_to_task[future] = task
                
                # Collect results
                for future in as_completed(all_futures):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        _collect_result(result)
                    except Exception as e:
                        raise RuntimeError(
                            f"Linear-op profiling task failed for num_tokens={task[1]}."
                        ) from e
                    pbar.update(1)
            except KeyboardInterrupt:
                print(
                    "[INFO] KeyboardInterrupt received in linear-op multi-GPU profiling. "
                    "Cleaning up worker processes..."
                )
                raise
            finally:
                for executor in executors:
                    executor.shutdown(wait=True, cancel_futures=True)
        else:
            # Single-GPU sequential mode
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            torch_module = _ensure_torch_available()
            torch_module.cuda.set_device(0)

            wrapper = LinearOpWrapper(
                model_config=model_config,
                num_tensor_parallel_workers=num_tensor_parallel_workers,
                profile_method=args.profile_method,
                rank=0,
                output_dir=str(model_output_dir),
                profiling_plan=profiling_plan,
            )

            for num_tokens in num_tokens_to_profile:
                result = wrapper.profile(num_tokens)
                _collect_result(result)
                pbar.update(1)

    # Deduplicate TP=1 replicated rows produced by multiple TP sizes.
    all_results = deduplicate_tp1_rows(
        all_results, tp1_key_fields=("num_tokens", "model_arch", "measurement_type")
    )

    if not all_results:
        raise RuntimeError(
            "No profiling results collected. Ensure vLLM ops and FP8 kernels "
            "are available and retry."
        )
    df = pd.DataFrame(all_results)
    if "time_stats" not in df.columns:
        raise RuntimeError(
            f"Missing 'time_stats' column in profiling results for model '{model}'. "
            f"Available columns: {list(df.columns)}. "
            "This may indicate a profiling wrapper issue."
        )
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    return df


def _linear_profile_operator_name(column: object, family) -> str | None:
    """Resolve a flattened ``time_stats`` column through a registered family."""

    if not isinstance(column, str):
        return None
    profiling_names = get_family_profiling_name_set(family)
    if column in profiling_names:
        return column
    prefix = "time_stats."
    if not column.startswith(prefix):
        return None
    suffix = column[len(prefix) :]
    matches = tuple(
        name
        for name in profiling_names
        if suffix == name or suffix.startswith(f"{name}.")
    )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous registered profiling operator column {column!r}: {matches}"
        )
    return matches[0] if matches else None


def _registered_operator_family_ids(operator_name: str) -> tuple[str, ...]:
    """Return the registered families that expose one profiling operator."""

    matches = tuple(
        family.family_id
        for family in iter_operator_families()
        if operator_name in get_family_profiling_name_set(family)
    )
    if len(matches) > 1:
        raise ValueError(
            f"Profiling operator {operator_name!r} is registered by multiple "
            f"operator families: {matches}"
        )
    return matches


def _typed_contract_family_id(
    operator_name: str,
    metadata: Mapping[str, object],
) -> str:
    """Validate one typed contract and return its registered family owner."""

    # Producer rows and runtime loaders share one complete metadata schema.
    # Validate intrinsic fields before checking registry ownership so malformed
    # rows fail at the producer boundary instead of being partially admitted.
    validate_typed_operator_metadata(
        metadata,
        operator_name=operator_name,
        expected_metadata={},
    )

    # The shared validator requires this field, so ownership never falls back
    # to an inferred family when a producer row is incomplete.
    family_id = metadata["operator_family_id"]

    family = get_operator_family(family_id)
    if operator_name not in get_family_profiling_name_set(family):
        raise ValueError(
            f"Typed operator contract for {operator_name!r} declares family "
            f"{family_id!r}, but that family does not register the operator"
        )

    layer_kind = metadata.get("layer_kind")
    if layer_kind is not None and not isinstance(layer_kind, str):
        raise ValueError(
            f"Typed operator contract for {operator_name!r} has invalid "
            f"layer_kind {layer_kind!r}"
        )
    dense_family_id = FFN_FAMILY.family_id
    if layer_kind == "dense" and family_id != dense_family_id:
        raise ValueError(
            f"Typed operator contract for {operator_name!r} marks a non-dense "
            f"family {family_id!r} as dense"
        )
    if layer_kind not in (None, "dense") and family_id == dense_family_id:
        raise ValueError(
            f"Typed operator contract for {operator_name!r} marks the dense "
            f"family as {layer_kind!r}"
        )
    return family_id


def _typed_dense_operator_names(df: pd.DataFrame) -> set[str] | None:
    """Return typed dense operators, or ``None`` for a legacy result frame."""

    column_name = "typed_operator_contracts"
    if column_name not in df.columns:
        return None

    dense_operator_names = get_family_profiling_name_set(FFN_FAMILY)
    ownership: dict[str, str] = {}
    for row_index, raw_contracts in df[column_name].items():
        contracts = parse_typed_operator_contracts(raw_contracts)
        for operator_name, metadata in contracts.items():
            # A linear-op row carries metadata for attention, memory, and
            # optional MTP operators as well.  This helper owns only dense FFN
            # filtering; each other family is admitted by its own consumer.
            if operator_name not in dense_operator_names:
                validate_typed_operator_metadata(
                    metadata,
                    operator_name=operator_name,
                    expected_metadata={},
                )
                registered_family_ids = _registered_operator_family_ids(operator_name)
                if (
                    registered_family_ids
                    and metadata["operator_family_id"] not in registered_family_ids
                ):
                    raise ValueError(
                        f"Typed operator contract for {operator_name!r} declares "
                        f"family {metadata['operator_family_id']!r}, expected one "
                        f"of {registered_family_ids}"
                    )
                continue
            family_id = _typed_contract_family_id(operator_name, metadata)
            previous_family_id = ownership.get(operator_name)
            if previous_family_id is not None and previous_family_id != family_id:
                raise ValueError(
                    f"Typed operator contract for {operator_name!r} changes family "
                    f"from {previous_family_id!r} to {family_id!r} at row {row_index}"
                )
            ownership[operator_name] = family_id

    return {
        operator_name
        for operator_name, family_id in ownership.items()
        if family_id == FFN_FAMILY.family_id
    }


def filter_mlp_columns(df: pd.DataFrame, *, model_config: Any = None) -> pd.DataFrame:
    """Filter dense FFN columns using registered operator and typed metadata."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    dense_operator_names = get_family_profiling_name_set(FFN_FAMILY)
    typed_dense_names = _typed_dense_operator_names(df)
    if typed_dense_names is None:
        # Legacy producers did not emit typed metadata; preserve their
        # historical MoE behavior while routing classification through the
        # registered FFN family.
        keep_dense_names: set[str] = set()
    else:
        keep_dense_names = typed_dense_names
        if model_config is not None:
            profile_getter = getattr(model_config, "get_model_architecture_profile", None)
            if not callable(profile_getter):
                raise TypeError(
                    "model_config must expose get_model_architecture_profile()"
                )
            profile = profile_getter()
            active_family_ids = {
                family_id
                for contract in profile.iter_active_layer_contracts(model_config)
                for family_id in contract.operator_family_ids
            }
            if keep_dense_names and FFN_FAMILY.family_id not in active_family_ids:
                raise ValueError(
                    "Typed metadata declares dense FFN operators for a profile "
                    "without an active dense layer contract"
                )

    cols_to_drop = [
        column
        for column in df.columns
        if (
            (operator_name := _linear_profile_operator_name(column, FFN_FAMILY))
            in dense_operator_names
            and operator_name not in keep_dense_names
        )
    ]
    if cols_to_drop:
        print(f"Filtering out MLP-specific columns: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)
    return df


def main():
    args = parse_args()
    require_profiling_dependencies("linear-op", ("torch", "vllm"))

    # Display execution mode
    if args.disable_ray:
        available_gpus = _get_available_gpus(args.num_gpus)
        actual_num_gpus = len(available_gpus)
        if actual_num_gpus > 1:
            print(f"\n=== Multi-GPU Mode (ProcessPoolExecutor) ===")
            print(f"Using {actual_num_gpus} GPUs: {available_gpus}")
        else:
            print(f"\n=== Single-GPU Mode ===")
            print(f"Using GPU: {available_gpus[0]}")
    else:
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Use --disable_ray flag.")
        print(f"\n=== Ray Mode ===")
        print(f"Using {args.num_gpus} GPUs via Ray")

    # Check if is_moe flag is set
    if args.is_moe:
        print("=" * 60)
        print("NOTICE: --is_moe flag is set")
        print("Skipping dense MLP ops (mlp_up_proj, mlp_down_proj, mlp_act).")
        print("MoE expert layers are profiled separately by the MoE module.")
        print("=" * 60)

    config_dir = Path(args.output_dir) / "compute" / args.device
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "linear_op_config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.dump(vars(args), config_file)

    num_tokens_to_profile = get_num_tokens_to_profile(
        args.max_tokens,
        extra_num_tokens=args.extra_num_tokens,
        num_tokens_list=args.num_tokens_list,
    )

    # Interactive confirmation before profiling
    from frontier.profiling.utils.confirmation import (
        confirm_profiling_execution,
        build_linear_op_config_sections,
    )

    # Load first model config for confirmation display
    first_model = args.models[0]
    first_model_config = ModelConfig.from_model_name(first_model)
    torch_dtype, precision_str = _resolve_precision_for_model(
        first_model_config, args.precision, first_model
    )

    config_sections = build_linear_op_config_sections(
        args=args,
        model_config=first_model_config,
        num_tokens_count=len(num_tokens_to_profile),
        precision_str=precision_str,
        torch_dtype=torch_dtype,
    )

    if not confirm_profiling_execution(
        module_name="Linear Op",
        config_sections=config_sections,
        skip_confirmation=args.skip_confirmation,
    ):
        sys.exit(0)

    _, _, _, all_tps = _resolve_tp_ranges(args)

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        all_tps,
    )

    pbar = tqdm(total=len(list(total_combos)))

    for model in args.models:
        result_df = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )

        model_config = ModelConfig.from_model_name(model)

        # Filter out MLP columns if is_moe is set
        if args.is_moe:
            result_df = filter_mlp_columns(result_df, model_config=model_config)

        _, precision_str = _resolve_precision_for_model(model_config, args.precision, model)
        model_arch = _resolve_model_arch_for_metadata(model_config)
        result_df = _attach_linear_op_output_metadata(
            result_df,
            precision_str=precision_str,
            model_arch=model_arch,
            model_architecture_profile=(
                model_config.get_model_architecture_profile().profile_id
            ),
            quant_signature=model_config.get_quant_signature(),
            measurement_type=profile_method_to_measurement_type(args.profile_method).value,
        )
        output_file = build_profile_method_output_path(
            output_root=args.output_dir,
            profiling_type="compute",
            hardware=args.device,
            model_name=model,
            op_name="linear_op",
            profile_method=args.profile_method,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df = _serialize_linear_op_output(result_df)
        result_df.to_csv(output_file, index=False)
        print(f"✓ Saved linear-op profiling data to: {output_file}")


if __name__ == "__main__":
    main()
