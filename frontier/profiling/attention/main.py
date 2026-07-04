"""
Attention Profiling Main Entry Point.

This module provides the main entry point for profiling attention operations
for LLM inference simulation.

Multi-GPU modes:
    1. Ray mode (default): Uses Ray for distributed profiling across GPUs
    2. Non-Ray mode (--disable_ray): Uses torch.multiprocessing for multi-GPU support
       - When --num_gpus > 1: Spawns multiple processes, each bound to a different GPU
       - When --num_gpus = 1: Single GPU sequential execution (original behavior)

Usage:
    # Single GPU mode (recommended for small tasks)
    python -m frontier.profiling.attention.main \\
        --models meta-llama/Llama-2-7b-hf \\
        --num_gpus 1 \\
        --disable_ray \\
        --output_dir data/profiling

    # Multi-GPU mode (recommended for large tasks)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m frontier.profiling.attention.main \\
        --models meta-llama/Llama-2-7b-hf \\
        --num_gpus 4 \\
        --disable_ray \\
        --output_dir data/profiling
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from frontier.config.precision_type import PrecisionType
# Conditionally import ray - only needed when not using --disable_ray
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    validate_attention_profiling_dataframe,
)
from frontier.attention.string_coercion import coerce_truthy_bool
from frontier.model_architectures import MODEL_ARCHITECTURE_REGISTRY
from frontier.profiling.attention.backends import AttentionBackend
from frontier.profiling.common.parallel_config import ParallelConfig

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.metadata_utils import add_chunked_prefill_metadata
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.utils import (
    EXPORTABLE_PROFILE_METHOD_CHOICES,
    ProfileMethod,
    build_profile_method_output_path,
    build_profiling_output_path,
    get_attention_input_combinations,
    get_max_num_blocks,
    get_mixed_prefill_input_combinations,
    get_online_grid_mixed_prefill_input_combinations,
    get_true_mixed_attention_input_combinations,
    normalize_profile_method,
    profile_method_to_measurement_type,
    require_profiling_dependencies,
)


def _ensure_torch_available():
    if torch is None:
        raise ImportError(
            "Attention profiling requires torch. Install the dedicated GPU profiling "
            "environment before running this entrypoint."
        )
    return torch


def _get_available_gpus(num_gpus: int) -> List[int]:
    """
    Get list of available GPU IDs based on CUDA_VISIBLE_DEVICES and num_gpus.

    IMPORTANT: This function avoids calling torch.cuda.device_count() in the main
    process to prevent early CUDA initialization, which can interfere with
    multi-GPU multiprocessing.

    Returns:
        List of GPU IDs to use for profiling
    """
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES
        available = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    else:
        # Avoid calling torch.cuda.device_count() in main process.
        # Require nvidia-smi evidence instead of guessing GPU indices.
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
                    "running attention profiling. "
                    f"nvidia-smi stderr: {result.stderr.strip()}"
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because nvidia-smi was "
                "not found. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running attention profiling."
            ) from exc

        if not available:
            raise RuntimeError(
                "Unable to discover GPUs with nvidia-smi because it returned no "
                "GPU indices. Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi "
                "before running attention profiling."
            )

    if len(available) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only found {len(available)} visible GPUs "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})."
        )

    return available[:num_gpus]


# Global variable to track if CUDA has been initialized in this process
_CUDA_INITIALIZED = False
_CUDA_GPU_ID = None


def _worker_init_gpu(gpu_id: int, gpu_local_idx: int) -> None:
    """
    Initialize worker process with specific GPU binding.

    This function must be called BEFORE any profiling task runs in the worker.
    Bind by local CUDA index from the parent visible set instead of mutating
    CUDA_VISIBLE_DEVICES at runtime.
    """
    global _CUDA_INITIALIZED, _CUDA_GPU_ID
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


def _worker_profile_attention_task(
    task_args: Tuple[int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing attention profiling.

    Args:
        task_args: Tuple of (gpu_id, model_config_dict, wrapper_args, input_dict)

    Returns:
        Profiling result dictionary.
    """
    global _CUDA_INITIALIZED, _CUDA_GPU_ID
    torch_module = _ensure_torch_available()
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    gpu_id, gpu_local_idx, model_config_dict, wrapper_args, input_dict = task_args

    if not _CUDA_INITIALIZED:
        raise RuntimeError(
            f"Worker received task for GPU {gpu_id} before GPU initializer ran."
        )

    if _CUDA_GPU_ID != gpu_id:
        # This worker was initialized with a different GPU
        # This should not happen with proper task distribution
        raise RuntimeError(
            f"Worker initialized with GPU {_CUDA_GPU_ID} but received task for GPU {gpu_id}. "
            f"This indicates a task distribution bug."
        )
    if torch_module.cuda.current_device() != gpu_local_idx:
        raise RuntimeError(
            f"Worker current device {torch_module.cuda.current_device()} does not match expected local index {gpu_local_idx}."
        )

    # Reconstruct ModelConfig from dict
    model_config = ModelConfig(**model_config_dict)

    # Reconstruct ParallelConfig
    parallel_config = ParallelConfig(
        tensor_parallel_size=wrapper_args["tensor_parallel_size"],
        pipeline_parallel_size=1,
    )

    # Reconstruct AttentionInput
    attention_input = AttentionInput(
        batch_size=input_dict["batch_size"],
        prefill_chunk_size=input_dict["prefill_chunk_size"],
        kv_cache_size=input_dict["kv_cache_size"],
        is_prefill=input_dict["is_prefill"],
    )

    # Create wrapper and run profiling
    wrapper = AttentionWrapper(
        model_config=model_config,
        parallel_config=parallel_config,
        max_num_blocks=wrapper_args["max_num_blocks"],
        max_model_len=wrapper_args["max_model_len"],
        block_size=wrapper_args["block_size"],
        attention_backend=wrapper_args["attention_backend"],
        dtype=wrapper_args["dtype"],
        profile_method=wrapper_args.get("profile_method", "record_function"),
        output_dir=wrapper_args.get("output_dir", "data/profiling"),
    )

    result = wrapper.profile(attention_input)
    return result


def _worker_profile_mixed_attention_task(
    task_args: Tuple[int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing mixed-batch attention profiling.

    Args:
        task_args: Tuple of (gpu_id, model_config_dict, wrapper_args, mixed_input_dict)

    Returns:
        Profiling result dictionary.
    """
    from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput
    torch_module = _ensure_torch_available()
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    gpu_id, gpu_local_idx, model_config_dict, wrapper_args, mixed_input_dict = task_args

    global _CUDA_INITIALIZED, _CUDA_GPU_ID
    if not _CUDA_INITIALIZED:
        raise RuntimeError(
            f"Worker received mixed task for GPU {gpu_id} before GPU initializer ran."
        )

    if _CUDA_GPU_ID != gpu_id:
        raise RuntimeError(
            f"Worker initialized with GPU {_CUDA_GPU_ID} but received task for GPU {gpu_id}. "
            f"This indicates a task distribution bug."
        )
    if torch_module.cuda.current_device() != gpu_local_idx:
        raise RuntimeError(
            f"Worker current device {torch_module.cuda.current_device()} does not match expected local index {gpu_local_idx}."
        )

    # Reconstruct ModelConfig from dict
    model_config = ModelConfig(**model_config_dict)

    # Reconstruct ParallelConfig
    parallel_config = ParallelConfig(
        tensor_parallel_size=wrapper_args["tensor_parallel_size"],
        pipeline_parallel_size=1,
    )

    # Reconstruct MixedAttentionInput
    mixed_input = MixedAttentionInput(
        seq_lens=list(mixed_input_dict["seq_lens"]),
        kv_cache_size=mixed_input_dict["kv_cache_size"],
        mode=mixed_input_dict["mode"],
    )

    # Create wrapper and run profiling
    wrapper = AttentionWrapper(
        model_config=model_config,
        parallel_config=parallel_config,
        max_num_blocks=wrapper_args["max_num_blocks"],
        max_model_len=wrapper_args["max_model_len"],
        block_size=wrapper_args["block_size"],
        attention_backend=wrapper_args["attention_backend"],
        dtype=wrapper_args["dtype"],
        profile_method=wrapper_args.get("profile_method", "record_function"),
        output_dir=wrapper_args.get("output_dir", "data/profiling"),
    )

    result = wrapper.profile_mixed(mixed_input)
    return result


def _worker_profile_true_mixed_attention_task(
    task_args: Tuple[int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]
) -> Dict[str, Any]:
    """Worker function for multiprocessing true mixed-batch attention profiling."""
    from frontier.profiling.attention.true_mixed_batch_input import TrueMixedBatchInput
    torch_module = _ensure_torch_available()
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    gpu_id, gpu_local_idx, model_config_dict, wrapper_args, true_mixed_input_dict = task_args

    global _CUDA_INITIALIZED, _CUDA_GPU_ID
    if not _CUDA_INITIALIZED:
        raise RuntimeError(
            f"Worker received true mixed task for GPU {gpu_id} before GPU initializer ran."
        )
    if _CUDA_GPU_ID != gpu_id:
        raise RuntimeError(
            f"Worker initialized with GPU {_CUDA_GPU_ID} but received task for GPU {gpu_id}. "
            f"This indicates a task distribution bug."
        )
    if torch_module.cuda.current_device() != gpu_local_idx:
        raise RuntimeError(
            f"Worker current device {torch_module.cuda.current_device()} does not match expected local index {gpu_local_idx}."
        )

    model_config = ModelConfig(**model_config_dict)
    parallel_config = ParallelConfig(
        tensor_parallel_size=wrapper_args["tensor_parallel_size"],
        pipeline_parallel_size=1,
    )
    true_mixed_input = TrueMixedBatchInput(
        prefill_seq_lens=list(true_mixed_input_dict["prefill_seq_lens"]),
        prefill_kv_cache_sizes=list(true_mixed_input_dict["prefill_kv_cache_sizes"]),
        decode_kv_cache_sizes=list(true_mixed_input_dict["decode_kv_cache_sizes"]),
    )

    wrapper = AttentionWrapper(
        model_config=model_config,
        parallel_config=parallel_config,
        max_num_blocks=wrapper_args["max_num_blocks"],
        max_model_len=wrapper_args["max_model_len"],
        block_size=wrapper_args["block_size"],
        attention_backend=wrapper_args["attention_backend"],
        dtype=wrapper_args["dtype"],
        profile_method=wrapper_args.get("profile_method", "record_function"),
        output_dir=wrapper_args.get("output_dir", "data/profiling"),
    )

    return wrapper.profile_true_mixed(true_mixed_input)


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Profiling")
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
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum context length model can serve",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum context length of input",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--batch_size_list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit batch sizes to profile. "
            "When provided, this overrides min/max batch size filtering."
        ),
    )
    parser.add_argument(
        "--decode_kv_cache_size_list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit decode KV-cache sizes to profile. "
            "When provided, this overrides the default decode sequence-length grid."
        ),
    )
    parser.add_argument(
        "--profile_only_decode",
        action="store_true",
        help="Only profile the decode",
    )
    parser.add_argument(
        "--profile_only_prefill",
        action="store_true",
        help="Only profile the prefill",
    )
    parser.add_argument(
        "--attention_backend",
        default=AttentionBackend.FLASHINFER.value,
        choices=[e.value for e in AttentionBackend] + ["FLASHINFER_MLA"],
        help="The attention backend to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--vllm_mla_cuda_op_log",
        type=Path,
        default=None,
        help=(
            "Import measured vLLM FlashInfer MLA CUDA op rows from cuda_ops.jsonl "
            "and write a Frontier-compatible attention profiling CSV. This is an "
            "explicit truth-source import path, not a replacement for native dense "
            "FlashInfer profiling."
        ),
    )
    parser.add_argument(
        "--model_architecture_profile",
        type=str,
        default=None,
        help=(
            "Explicit model architecture profile id for import-only profiling paths. "
            "Required when --vllm_mla_cuda_op_log is used with a model that cannot "
            "be resolved from data/config/models."
        ),
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
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",
    )
    parser.add_argument(
        "--fixed_chunked_prefill_size",
        type=int,
        default=-1,
        help="Fixed chunk size for prefill profiling. Use -1 to match max_seq_len (no chunking).",
    )
    parser.add_argument(
        "--enable_chunked_prefill_grid_search",
        action="store_true",
        help="Enable chunked prefill size grid search (disabled by default when mixed prefill is off).",
    )
    # Mixed-length batch profiling arguments
    parser.add_argument(
        "--enable_mixed_prefill",
        action="store_true",
        help="Enable mixed-length batch prefill profiling",
    )
    parser.add_argument(
        "--mixed_mode",
        type=str,
        default="both",
        choices=["even", "random", "both"],
        help="Mixed prefill mode: even (same length), random (varied lengths), or both",
    )
    parser.add_argument(
        "--max_mixed_batch_size",
        type=int,
        default=8,
        help="Maximum batch size for mixed prefill profiling",
    )
    parser.add_argument(
        "--mixed_profile_strategy",
        type=str,
        default="default",
        choices=["default", "online_grid"],
        help=(
            "Mixed prefill profiling strategy: "
            "default (legacy even/random generator) or "
            "online_grid (deterministic full-grid online serving coverage)."
        ),
    )
    parser.add_argument(
        "--mixed_batch_size_min",
        type=int,
        default=2,
        help="Minimum batch size for online_grid mixed profiling.",
    )
    parser.add_argument(
        "--mixed_batch_size_max",
        type=int,
        default=32,
        help="Maximum batch size for online_grid mixed profiling.",
    )
    parser.add_argument(
        "--mixed_total_tokens_min",
        type=int,
        default=1025,
        help="Minimum total tokens per mixed batch point for online_grid strategy.",
    )
    parser.add_argument(
        "--mixed_total_tokens_max",
        type=int,
        default=1055,
        help="Maximum total tokens per mixed batch point for online_grid strategy.",
    )
    parser.add_argument(
        "--mixed_batch_size_list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit batch sizes for online_grid mixed profiling. "
            "When provided, this overrides mixed_batch_size_min/max."
        ),
    )
    parser.add_argument(
        "--mixed_total_tokens_list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit total-token values for online_grid mixed profiling. "
            "When provided, this overrides mixed_total_tokens_min/max."
        ),
    )
    parser.add_argument(
        "--mixed_shapes_per_point",
        type=int,
        default=2,
        help="Number of deterministic shapes per (batch_size,total_tokens) point in online_grid strategy.",
    )
    parser.add_argument(
        "--mixed_kv_cache_size_list",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Explicit KV cache sizes to profile for mixed prefill inputs. "
            "Defaults to 0 to preserve existing behavior."
        ),
    )
    parser.add_argument(
        "--mixed_num_samples",
        type=int,
        default=3,
        help="Number of random samples per configuration in random mode (default: 3). "
             "Increase for more comprehensive coverage of mixed-length scenarios.",
    )
    parser.add_argument(
        "--enable_true_mixed",
        action="store_true",
        help="Enable true mixed-batch profiling (prefill + decode in one batch).",
    )
    parser.add_argument(
        "--true_mixed_prefill_batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Prefill sequence counts for true mixed-batch profiling.",
    )
    parser.add_argument(
        "--true_mixed_prefill_chunk_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Prefill chunk sizes for true mixed-batch profiling.",
    )
    parser.add_argument(
        "--true_mixed_decode_batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Decode sequence counts for true mixed-batch profiling.",
    )
    parser.add_argument(
        "--true_mixed_decode_kv_cache_sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Decode KV cache sizes for true mixed-batch profiling.",
    )
    parser.add_argument(
        "--true_mixed_prefill_kv_cache_size",
        type=int,
        default=0,
        help="Prefill-side KV cache size for true mixed-batch profiling.",
    )
    parser.add_argument(
        "--max_pipeline_parallel_size",
        type=int,
        default=1,
        help="Maximum pipeline parallel size for memory calculation (must divide model layers evenly). "
             "Use 4 for models with 28 layers (e.g., Qwen2.5-7B), 8 for most others (32, 80 layers)",
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
        help="Reserved for future replicated ops support. Currently no-op for attention.",
    )
    parser.add_argument(
        "--profile_method",
        type=str,
        default="record_function",
        choices=EXPORTABLE_PROFILE_METHOD_CHOICES,
        help="Profiling method: cuda_event (wall-clock GPU time) or record_function (pure kernel time via profiler trace)",
    )
    args = parser.parse_args()
    args.profile_method = normalize_profile_method(args.profile_method)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


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


def _validate_cli_conflicts(args: argparse.Namespace) -> None:
    """Validate unsupported argument combinations with fail-fast behavior."""
    if args.profile_only_prefill and args.profile_only_decode:
        raise ValueError(
            "profile_only_prefill and profile_only_decode cannot both be enabled."
        )
    if args.profile_only_prefill and args.decode_kv_cache_size_list is not None:
        raise ValueError(
            "--decode_kv_cache_size_list cannot be combined with --profile_only_prefill."
        )
    if args.enable_true_mixed and (args.profile_only_prefill or args.profile_only_decode):
        raise ValueError(
            "--enable_true_mixed requires profiling both prefill and decode in the same batch. "
            "It is only valid for co-location mixed-batch profiling and cannot be combined with "
            "--profile_only_prefill or --profile_only_decode."
        )
    if args.attention_backend == "FLASHINFER_MLA" and args.vllm_mla_cuda_op_log is None:
        raise ValueError(
            "--attention_backend FLASHINFER_MLA requires --vllm_mla_cuda_op_log. "
            "Frontier does not yet provide a native FlashInfer MLA profiling backend."
        )
    if args.vllm_mla_cuda_op_log is not None:
        if len(args.models) != 1:
            raise ValueError("--vllm_mla_cuda_op_log requires exactly one model.")
        if len(args.num_tensor_parallel_workers) != 1:
            raise ValueError(
                "--vllm_mla_cuda_op_log requires exactly one tensor parallel size."
            )
        if args.attention_backend != "FLASHINFER_MLA":
            raise ValueError(
                "--vllm_mla_cuda_op_log requires --attention_backend FLASHINFER_MLA."
            )
        if args.enable_mixed_prefill or args.enable_true_mixed:
            raise ValueError(
                "--vllm_mla_cuda_op_log cannot be combined with mixed profiling modes."
            )


def _attach_attention_output_metadata(
    df: pd.DataFrame,
    *,
    precision_str: str,
    model_arch: str,
    model_architecture_profile: str,
    quant_signature: str,
    measurement_type: str,
) -> pd.DataFrame:
    """Attach profiling metadata columns to an attention profiling dataframe."""
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


def _prepare_standard_attention_output_dataframe(
    df: pd.DataFrame,
    *,
    precision_str: str,
    model_arch: str,
    model_architecture_profile: str,
    quant_signature: str,
    measurement_type: str,
) -> pd.DataFrame:
    output_df = _attach_attention_output_metadata(
        df,
        precision_str=precision_str,
        model_arch=model_arch,
        model_architecture_profile=model_architecture_profile,
        quant_signature=quant_signature,
        measurement_type=measurement_type,
    )
    validate_attention_profiling_dataframe(
        output_df,
        DENSE_ATTENTION_FAMILY,
        measurement_type=measurement_type,
    )
    return output_df


def _resolve_vllm_mla_model_architecture_profile(
    model_name: str,
    explicit_profile: str | None,
) -> str:
    """Resolve the architecture profile for vLLM MLA import without silent defaults."""

    if explicit_profile is not None:
        normalized_profile = explicit_profile.strip().lower()
        if not normalized_profile:
            raise ValueError("model_architecture_profile must be non-empty.")
        return MODEL_ARCHITECTURE_REGISTRY.get(normalized_profile).profile_id

    try:
        return ModelConfig.from_model_name(
            model_name
        ).get_model_architecture_profile().profile_id
    except ValueError as exc:
        raise ValueError(
            "--vllm_mla_cuda_op_log requires a resolvable model config or an "
            "explicit --model_architecture_profile. This import path must not "
            "silently write a generic architecture profile."
        ) from exc


def _run_vllm_mla_profile_import(args: argparse.Namespace) -> Path:
    """Import vLLM MLA measured rows into the canonical attention profiling CSV."""

    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
        load_vllm_mla_rows,
    )

    _validate_cli_conflicts(args)
    model = args.models[0]
    precision_str = str(args.precision or "BF16").lower()
    model_arch = str(getattr(args, "model_arch", None) or "deepseek_v2")
    model_architecture_profile = _resolve_vllm_mla_model_architecture_profile(
        model,
        getattr(args, "model_architecture_profile", None),
    )
    measurement_type = profile_method_to_measurement_type(args.profile_method).value

    vllm_rows = load_vllm_mla_rows(args.vllm_mla_cuda_op_log)
    df = build_frontier_mla_profile_dataframe(
        vllm_rows,
        model_name=model,
        model_arch=model_arch,
        precision=precision_str,
        quant_signature="none",
        measurement_type=measurement_type,
        num_tensor_parallel_workers=args.num_tensor_parallel_workers[0],
        max_model_len=args.max_model_len,
    )
    df = _attach_attention_output_metadata(
        df,
        precision_str=precision_str,
        model_arch=model_arch,
        model_architecture_profile=model_architecture_profile,
        quant_signature="none",
        measurement_type=measurement_type,
    )

    output_file = build_profile_method_output_path(
        output_root=args.output_dir,
        profiling_type="compute",
        hardware=args.device,
        model_name=model,
        op_name="attention",
        profile_method=args.profile_method,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    comparison_file = output_file.with_name(
        f"{output_file.stem}_vllm_mla_groundtruth_comparison.csv"
    )
    # This is an import round-trip check against the same vLLM rows, not an
    # independent simulator-accuracy benchmark.
    comparison_df = build_mla_profile_groundtruth_comparison(vllm_rows, df)
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved vLLM MLA imported attention results to {output_file}")
    print(f"Saved vLLM MLA groundtruth comparison to {comparison_file}")
    return output_file


def _normalize_partition_contract(
    df: pd.DataFrame,
    *,
    is_true_mixed_batch_default: bool,
    is_mixed_batch_default: bool,
) -> pd.DataFrame:
    """Normalize required marker columns for combined attention outputs."""
    if df.empty:
        return df
    output_df = df.copy()
    def _normalize_bool_series(series: pd.Series, default: bool) -> pd.Series:
        normalized = coerce_truthy_bool(series)
        return normalized.where(~series.isna(), default)

    if "is_true_mixed_batch" not in output_df.columns:
        output_df["is_true_mixed_batch"] = is_true_mixed_batch_default
    else:
        output_df["is_true_mixed_batch"] = _normalize_bool_series(
            output_df["is_true_mixed_batch"],
            is_true_mixed_batch_default,
        )
    if "is_mixed_batch" not in output_df.columns:
        output_df["is_mixed_batch"] = is_mixed_batch_default
    else:
        output_df["is_mixed_batch"] = _normalize_bool_series(
            output_df["is_mixed_batch"],
            is_mixed_batch_default,
        )
    return output_df


def _build_attention_combined_df(
    standard_df: pd.DataFrame,
    mixed_prefill_df: pd.DataFrame,
    true_mixed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build combined attention dataframe with deterministic schema markers."""
    partitions = []
    if not standard_df.empty:
        partitions.append(
            _normalize_partition_contract(
                standard_df,
                is_true_mixed_batch_default=False,
                is_mixed_batch_default=False,
            )
        )
    if not mixed_prefill_df.empty:
        partitions.append(
            _normalize_partition_contract(
                mixed_prefill_df,
                is_true_mixed_batch_default=False,
                is_mixed_batch_default=True,
            )
        )
    if not true_mixed_df.empty:
        partitions.append(
            _normalize_partition_contract(
                true_mixed_df,
                is_true_mixed_batch_default=True,
                is_mixed_batch_default=False,
            )
        )
    if not partitions:
        return pd.DataFrame()
    return pd.concat(partitions, ignore_index=True)


def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    input_combinations: List[AttentionInput],
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
):
    """
    Profile standard attention operations for a given model.

    Multi-GPU support:
    - Ray mode: Uses Ray actors for distributed profiling
    - Non-Ray mode with num_gpus > 1: Uses ProcessPoolExecutor for multi-GPU profiling
    - Non-Ray mode with num_gpus = 1: Sequential single-GPU profiling
    """
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    model_config = ModelConfig.from_model_name(model)
    resolved_dtype, _ = _resolve_precision_for_model(model_config, args.precision, model)
    dtype = resolved_dtype
    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )

    all_results = []

    # Determine available GPUs for non-Ray mode
    available_gpus = _get_available_gpus(args.num_gpus)
    actual_num_gpus = len(available_gpus)

    if actual_num_gpus < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs but only found {actual_num_gpus} visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}). "
            "Please adjust --num_gpus or CUDA_VISIBLE_DEVICES."
        )

    # Common wrapper arguments
    wrapper_args = {
        "tensor_parallel_size": num_tensor_parallel_workers,
        "max_num_blocks": max_num_blocks,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "attention_backend": args.attention_backend,
        "dtype": dtype,
        "profile_method": args.profile_method,
        "output_dir": args.output_dir,
    }

    # Convert model_config to dict for multiprocessing serialization
    model_config_dict = model_config.to_dict()

    if not args.disable_ray:
        # Ray mode
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Use --disable_ray flag.")

        promises = []
        model_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            AttentionWrapper,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                parallel_config,
                max_num_blocks,
                args.max_model_len,
                args.block_size,
                args.attention_backend,
                dtype,
                args.profile_method,
                args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]

        for attention_input in input_combinations:
            worker_id = len(promises) % args.num_gpus
            promise = model_wrappers[worker_id].profile.remote(attention_input)
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []

            pbar.update(1)

        if promises:
            results = ray.get(promises)
            all_results.extend(results)

    elif actual_num_gpus > 1:
        # Non-Ray multi-GPU mode: use multiple single-worker executors
        # Each executor is bound to a specific GPU
        gpu_local_idx_map = {
            gpu_id: local_idx for local_idx, gpu_id in enumerate(available_gpus)
        }
        tasks_by_gpu = {gpu_id: [] for gpu_id in available_gpus}
        for idx, attention_input in enumerate(input_combinations):
            gpu_id = available_gpus[idx % actual_num_gpus]
            gpu_local_idx = gpu_local_idx_map[gpu_id]
            input_dict = {
                "batch_size": attention_input.batch_size,
                "prefill_chunk_size": attention_input.prefill_chunk_size,
                "kv_cache_size": attention_input.kv_cache_size,
                "is_prefill": attention_input.is_prefill,
            }
            tasks_by_gpu[gpu_id].append(
                (gpu_id, gpu_local_idx, model_config_dict, wrapper_args, input_dict)
            )

        ctx = mp.get_context('spawn')
        executors = []
        all_futures = []
        future_to_task = {}
        try:
            for gpu_id in available_gpus:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_worker_init_gpu,
                    initargs=(gpu_id, gpu_local_idx_map[gpu_id]),
                )
                executors.append(executor)
                for task in tasks_by_gpu[gpu_id]:
                    future = executor.submit(_worker_profile_attention_task, task)
                    all_futures.append(future)
                    future_to_task[future] = task

            for future in as_completed(all_futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    input_dict = task[4]
                    raise RuntimeError(
                        "Attention profiling task failed for "
                        f"prefill_chunk_size={input_dict['prefill_chunk_size']}, "
                        f"kv_cache_size={input_dict['kv_cache_size']}, "
                        f"batch_size={input_dict['batch_size']}, "
                        f"is_prefill={input_dict['is_prefill']}."
                    ) from e
                pbar.update(1)
        except KeyboardInterrupt:
            print(
                "[INFO] KeyboardInterrupt received in attention multi-GPU profiling. "
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

        wrapper = AttentionWrapper(
            model_config=model_config,
            parallel_config=parallel_config,
            max_num_blocks=max_num_blocks,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            attention_backend=args.attention_backend,
            dtype=dtype,
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )

        for attention_input in input_combinations:
            result = wrapper.profile(attention_input)
            all_results.append(result)
            pbar.update(1)

    # Filter out None results
    all_results = list(filter(None, all_results))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    # Expand time_stats column
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    df = add_chunked_prefill_metadata(df)
    return df


def profile_mixed_prefill(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    mixed_input_combinations: List,
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
):
    """
    Profile mixed-length batch prefill attention.

    This function profiles attention performance when batches contain
    multiple sequences with potentially different lengths.

    Multi-GPU support:
    - Ray mode: Uses Ray actors for distributed profiling
    - Non-Ray mode with num_gpus > 1: Uses ProcessPoolExecutor for multi-GPU profiling
    - Non-Ray mode with num_gpus = 1: Sequential single-GPU profiling

    Args:
        args: Command line arguments.
        model: Model name to profile.
        num_tensor_parallel_workers: Number of tensor parallel workers.
        mixed_input_combinations: List of MixedAttentionInput objects.
        max_num_blocks: Maximum number of KV cache blocks.
        dtype: Data type for tensors.
        pbar: Progress bar for tracking.

    Returns:
        DataFrame containing profiling results.
    """
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    model_config = ModelConfig.from_model_name(model)
    resolved_dtype, _ = _resolve_precision_for_model(model_config, args.precision, model)
    dtype = resolved_dtype
    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )

    all_results = []

    # Determine available GPUs for non-Ray mode
    available_gpus = _get_available_gpus(args.num_gpus)
    actual_num_gpus = len(available_gpus)

    # Common wrapper arguments
    wrapper_args = {
        "tensor_parallel_size": num_tensor_parallel_workers,
        "max_num_blocks": max_num_blocks,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "attention_backend": args.attention_backend,
        "dtype": dtype,
        "profile_method": args.profile_method,
        "output_dir": args.output_dir,
    }

    # Convert model_config to dict for multiprocessing serialization
    model_config_dict = model_config.to_dict()

    if not args.disable_ray:
        # Ray mode
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Use --disable_ray flag.")

        promises = []
        model_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            AttentionWrapper,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                parallel_config,
                max_num_blocks,
                args.max_model_len,
                args.block_size,
                args.attention_backend,
                dtype,
                args.profile_method,
                args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]

        for mixed_input in mixed_input_combinations:
            worker_id = len(promises) % args.num_gpus
            promise = model_wrappers[worker_id].profile_mixed.remote(mixed_input)
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []

            pbar.update(1)

        if promises:
            results = ray.get(promises)
            all_results.extend(results)

    elif actual_num_gpus > 1:
        # Non-Ray multi-GPU mode: use multiple single-worker executors
        # Each executor is bound to a specific GPU
        gpu_local_idx_map = {
            gpu_id: local_idx for local_idx, gpu_id in enumerate(available_gpus)
        }
        tasks_by_gpu = {gpu_id: [] for gpu_id in available_gpus}
        for idx, mixed_input in enumerate(mixed_input_combinations):
            gpu_id = available_gpus[idx % actual_num_gpus]
            gpu_local_idx = gpu_local_idx_map[gpu_id]
            mixed_input_dict = {
                "seq_lens": list(mixed_input.seq_lens),
                "kv_cache_size": mixed_input.kv_cache_size,
                "mode": mixed_input.mode,
            }
            tasks_by_gpu[gpu_id].append(
                (gpu_id, gpu_local_idx, model_config_dict, wrapper_args, mixed_input_dict)
            )

        ctx = mp.get_context('spawn')
        executors = []
        all_futures = []
        future_to_task = {}
        try:
            for gpu_id in available_gpus:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_worker_init_gpu,
                    initargs=(gpu_id, gpu_local_idx_map[gpu_id]),
                )
                executors.append(executor)
                for task in tasks_by_gpu[gpu_id]:
                    future = executor.submit(_worker_profile_mixed_attention_task, task)
                    all_futures.append(future)
                    future_to_task[future] = task

            for future in as_completed(all_futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    input_dict = task[4]
                    raise RuntimeError(
                        "Mixed attention profiling task failed for "
                        f"seq_lens={input_dict['seq_lens']}, "
                        f"kv_cache_size={input_dict['kv_cache_size']}, "
                        f"mode={input_dict['mode']}."
                    ) from e
                pbar.update(1)
        except KeyboardInterrupt:
            print(
                "[INFO] KeyboardInterrupt received in mixed attention multi-GPU profiling. "
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

        wrapper = AttentionWrapper(
            model_config=model_config,
            parallel_config=parallel_config,
            max_num_blocks=max_num_blocks,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            attention_backend=args.attention_backend,
            dtype=dtype,
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )

        for mixed_input in mixed_input_combinations:
            result = wrapper.profile_mixed(mixed_input)
            all_results.append(result)
            pbar.update(1)

    # Filter out None results
    all_results = list(filter(None, all_results))

    if not all_results:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Expand time_stats column
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    df = add_chunked_prefill_metadata(df)

    return df


def profile_true_mixed_batches(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    true_mixed_input_combinations: List,
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
):
    """Profile true mixed prefill+decode batches."""
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    model_config = ModelConfig.from_model_name(model)
    resolved_dtype, _ = _resolve_precision_for_model(model_config, args.precision, model)
    dtype = resolved_dtype
    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )

    all_results = []
    available_gpus = _get_available_gpus(args.num_gpus)
    actual_num_gpus = len(available_gpus)

    wrapper_args = {
        "tensor_parallel_size": num_tensor_parallel_workers,
        "max_num_blocks": max_num_blocks,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "attention_backend": args.attention_backend,
        "dtype": dtype,
        "profile_method": args.profile_method,
        "output_dir": args.output_dir,
    }
    model_config_dict = model_config.to_dict()

    if not args.disable_ray:
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Use --disable_ray flag.")
        promises = []
        model_wrapper_actor = ray.remote(num_cpus=1, num_gpus=1)(AttentionWrapper).options(
            runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}}
        )
        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                parallel_config,
                max_num_blocks,
                args.max_model_len,
                args.block_size,
                args.attention_backend,
                dtype,
                args.profile_method,
                args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]
        for true_mixed_input in true_mixed_input_combinations:
            worker_id = len(promises) % args.num_gpus
            promise = model_wrappers[worker_id].profile_true_mixed.remote(true_mixed_input)
            promises.append(promise)
            if len(promises) >= args.num_gpus:
                all_results.extend(ray.get(promises))
                promises = []
            pbar.update(1)
        if promises:
            all_results.extend(ray.get(promises))
    elif actual_num_gpus > 1:
        gpu_local_idx_map = {
            gpu_id: local_idx for local_idx, gpu_id in enumerate(available_gpus)
        }
        tasks_by_gpu = {gpu_id: [] for gpu_id in available_gpus}
        for idx, true_mixed_input in enumerate(true_mixed_input_combinations):
            gpu_id = available_gpus[idx % actual_num_gpus]
            gpu_local_idx = gpu_local_idx_map[gpu_id]
            true_mixed_input_dict = {
                "prefill_seq_lens": list(true_mixed_input.prefill_seq_lens),
                "prefill_kv_cache_sizes": list(true_mixed_input.prefill_kv_cache_sizes),
                "decode_kv_cache_sizes": list(true_mixed_input.decode_kv_cache_sizes),
            }
            tasks_by_gpu[gpu_id].append(
                (
                    gpu_id,
                    gpu_local_idx,
                    model_config_dict,
                    wrapper_args,
                    true_mixed_input_dict,
                )
            )

        ctx = mp.get_context("spawn")
        executors = []
        all_futures = []
        future_to_task = {}
        try:
            for gpu_id in available_gpus:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_worker_init_gpu,
                    initargs=(gpu_id, gpu_local_idx_map[gpu_id]),
                )
                executors.append(executor)
                for task in tasks_by_gpu[gpu_id]:
                    future = executor.submit(_worker_profile_true_mixed_attention_task, task)
                    all_futures.append(future)
                    future_to_task[future] = task
            for future in as_completed(all_futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    task = future_to_task[future]
                    input_dict = task[4]
                    raise RuntimeError(
                        "True mixed attention profiling task failed for "
                        f"prefill_seq_lens={input_dict['prefill_seq_lens']}, "
                        f"prefill_kv_cache_sizes={input_dict['prefill_kv_cache_sizes']}, "
                        f"decode_kv_cache_sizes={input_dict['decode_kv_cache_sizes']}."
                    ) from e
                pbar.update(1)
        except KeyboardInterrupt:
            print(
                "[INFO] KeyboardInterrupt received in true mixed attention multi-GPU profiling. "
                "Cleaning up worker processes..."
            )
            raise
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
        torch_module = _ensure_torch_available()
        torch_module.cuda.set_device(0)
        wrapper = AttentionWrapper(
            model_config=model_config,
            parallel_config=parallel_config,
            max_num_blocks=max_num_blocks,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            attention_backend=args.attention_backend,
            dtype=dtype,
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )
        for true_mixed_input in true_mixed_input_combinations:
            all_results.append(wrapper.profile_true_mixed(true_mixed_input))
            pbar.update(1)

    all_results = list(filter(None, all_results))
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    df = add_chunked_prefill_metadata(df)
    return df


def main():
    args = parse_args()
    _validate_cli_conflicts(args)
    if args.vllm_mla_cuda_op_log is not None:
        _run_vllm_mla_profile_import(args)
        return

    if args.attention_backend == AttentionBackend.FLASHINFER.value:
        require_profiling_dependencies("attention", ("torch", "vllm", "flashinfer"))
    else:
        require_profiling_dependencies("attention", ("torch",))

    if args.fixed_chunked_prefill_size == 0:
        raise ValueError(
            "fixed_chunked_prefill_size must be positive or -1 to disable chunking."
        )

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

    model_configs = {}
    model_dtypes = {}
    model_precision_strs = {}
    for model in args.models:
        model_config = ModelConfig.from_model_name(model)
        dtype, precision_str = _resolve_precision_for_model(model_config, args.precision, model)
        _resolve_fp8_settings(model_config, args.use_fp8, args.block_shape, model)
        model_configs[model] = model_config
        model_dtypes[model] = dtype
        model_precision_strs[model] = precision_str

    # Generate standard attention input combinations
    input_combinations = get_attention_input_combinations(
        args.max_seq_len,
        args.min_batch_size,
        args.max_batch_size,
        args.profile_only_prefill,
        args.profile_only_decode,
        batch_size_list=args.batch_size_list,
        decode_kv_cache_size_list=args.decode_kv_cache_size_list,
        enable_chunked_prefill_grid_search=args.enable_chunked_prefill_grid_search,
        fixed_chunked_prefill_size=args.fixed_chunked_prefill_size,
    )

    # Generate mixed-length prefill input combinations if enabled
    mixed_input_combinations = []
    if args.enable_mixed_prefill:
        print(f"\n=== Mixed-length Prefill Profiling Enabled ===")
        print(f"Strategy: {args.mixed_profile_strategy}")

        if args.mixed_profile_strategy == "online_grid":
            print(
                "Grid: "
                f"batch_size=[{args.mixed_batch_size_min},{args.mixed_batch_size_max}], "
                f"total_tokens=[{args.mixed_total_tokens_min},{args.mixed_total_tokens_max}], "
                f"batch_size_list={args.mixed_batch_size_list}, "
                f"total_tokens_list={args.mixed_total_tokens_list}, "
                f"shapes_per_point={args.mixed_shapes_per_point}, "
                f"kv_cache_sizes={args.mixed_kv_cache_size_list}"
            )
            mixed_input_combinations = get_online_grid_mixed_prefill_input_combinations(
                max_seq_len=args.max_seq_len,
                min_batch_size=args.mixed_batch_size_min,
                max_batch_size=args.mixed_batch_size_max,
                min_total_tokens=args.mixed_total_tokens_min,
                max_total_tokens=args.mixed_total_tokens_max,
                shapes_per_point=args.mixed_shapes_per_point,
                kv_cache_sizes=args.mixed_kv_cache_size_list,
                batch_size_list=args.mixed_batch_size_list,
                total_tokens_list=args.mixed_total_tokens_list,
            )
        else:
            print(f"Mode: {args.mixed_mode}")
            print(f"Max batch size: {args.max_mixed_batch_size}")
            print(f"KV cache sizes: {args.mixed_kv_cache_size_list}")
            mixed_input_combinations = get_mixed_prefill_input_combinations(
                max_seq_len=args.max_seq_len,
                min_batch_size=2,  # At least 2 sequences for mixed batch
                max_batch_size=args.max_mixed_batch_size,
                mode=args.mixed_mode,
                num_samples_per_config=args.mixed_num_samples,
                kv_cache_sizes=args.mixed_kv_cache_size_list,
            )

        print(f"Generated {len(mixed_input_combinations)} mixed-batch test cases")

    # Generate true mixed (prefill+decode) input combinations if enabled
    true_mixed_input_combinations = []
    if args.enable_true_mixed:
        print(f"\n=== True Mixed-Batch Profiling Enabled (co-location only) ===")
        print(
            "Grid: "
            f"prefill_bs={args.true_mixed_prefill_batch_sizes}, "
            f"prefill_chunk={args.true_mixed_prefill_chunk_sizes}, "
            f"decode_bs={args.true_mixed_decode_batch_sizes}, "
            f"decode_kv={args.true_mixed_decode_kv_cache_sizes}, "
            f"prefill_kv={args.true_mixed_prefill_kv_cache_size}"
        )
        true_mixed_input_combinations = get_true_mixed_attention_input_combinations(
            max_seq_len=args.max_seq_len,
            prefill_batch_sizes=args.true_mixed_prefill_batch_sizes,
            prefill_chunk_sizes=args.true_mixed_prefill_chunk_sizes,
            decode_batch_sizes=args.true_mixed_decode_batch_sizes,
            decode_kv_cache_sizes=args.true_mixed_decode_kv_cache_sizes,
            prefill_kv_cache_size=args.true_mixed_prefill_kv_cache_size,
        )
        print(
            f"Generated {len(true_mixed_input_combinations)} true mixed-batch test cases"
        )

    # Interactive confirmation before profiling
    from frontier.profiling.utils.confirmation import (
        confirm_profiling_execution,
        build_attention_config_sections,
    )

    # Use first model for confirmation display
    first_model = args.models[0]
    first_model_config = model_configs[first_model]
    first_dtype = model_dtypes[first_model]
    first_precision_str = model_precision_strs[first_model]

    config_sections = build_attention_config_sections(
        args=args,
        model_config=first_model_config,
        input_combinations_count=len(input_combinations),
        mixed_combinations_count=len(mixed_input_combinations),
        true_mixed_combinations_count=len(true_mixed_input_combinations),
        precision_str=first_precision_str,
        torch_dtype=first_dtype,
    )

    if not confirm_profiling_execution(
        module_name="Attention",
        config_sections=config_sections,
        skip_confirmation=args.skip_confirmation,
    ):
        sys.exit(0)

    # Filter mixed combinations based on memory limits
    filtered_mixed_combinations = {}
    if args.enable_mixed_prefill:
        for model in args.models:
            model_config = model_configs[model]
            dtype = model_dtypes[model]
            for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
                max_num_blocks = get_max_num_blocks(
                    model_config,
                    ParallelConfig(
                        tensor_parallel_size=num_tensor_parallel_workers,
                        pipeline_parallel_size=1,
                    ),
                    args.block_size,
                    dtype,
                    max_pipeline_parallel_size=args.max_pipeline_parallel_size,
                )
                
                filtered_mixed_combinations[(model, num_tensor_parallel_workers)] = list(
                    filter(
                        lambda mixed_input: mixed_input.is_under_memory_limit(
                            max_num_blocks * args.block_size
                        ),
                        mixed_input_combinations,
                    )
                )
        
        print(f"After memory filtering: {sum(len(v) for v in filtered_mixed_combinations.values())} test cases")

    # Filter true mixed combinations based on memory limits
    filtered_true_mixed_combinations = {}
    if args.enable_true_mixed:
        for model in args.models:
            model_config = model_configs[model]
            dtype = model_dtypes[model]
            for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
                max_num_blocks = get_max_num_blocks(
                    model_config,
                    ParallelConfig(
                        tensor_parallel_size=num_tensor_parallel_workers,
                        pipeline_parallel_size=1,
                    ),
                    args.block_size,
                    dtype,
                    max_pipeline_parallel_size=args.max_pipeline_parallel_size,
                )

                filtered_true_mixed_combinations[(model, num_tensor_parallel_workers)] = list(
                    filter(
                        lambda true_mixed_input: true_mixed_input.is_under_memory_limit(
                            max_num_blocks * args.block_size
                        ),
                        true_mixed_input_combinations,
                    )
                )

        print(
            "After true-mixed memory filtering: "
            f"{sum(len(v) for v in filtered_true_mixed_combinations.values())} test cases"
        )

    # Filter standard combinations by memory
    total_combos = {}
    max_num_blocks_dict = {}
    for model in args.models:
        model_config = model_configs[model]
        dtype = model_dtypes[model]
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            max_num_blocks = get_max_num_blocks(
                model_config,
                ParallelConfig(
                    tensor_parallel_size=num_tensor_parallel_workers,
                    pipeline_parallel_size=1,
                ),
                args.block_size,
                dtype,
                max_pipeline_parallel_size=args.max_pipeline_parallel_size,
            )
            max_num_blocks_dict[(model, num_tensor_parallel_workers)] = max_num_blocks
            total_combos[(model, num_tensor_parallel_workers)] = list(
                filter(
                    lambda input_combination: input_combination.is_under_memory_limit(
                        max_num_blocks * args.block_size
                    ),
                    input_combinations,
                )
            )

    # Calculate total work for progress bar
    total_work = sum(len(v) for v in total_combos.values())
    if args.enable_mixed_prefill:
        total_work += sum(len(v) for v in filtered_mixed_combinations.values())
    if args.enable_true_mixed:
        total_work += sum(len(v) for v in filtered_true_mixed_combinations.values())
    
    pbar = tqdm(total=total_work)

    for model in args.models:
        result_df = pd.DataFrame()
        mixed_result_df = pd.DataFrame()
        true_mixed_result_df = pd.DataFrame()
        
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            # Profile standard attention (single-sequence prefill + decode)
            if not args.enable_mixed_prefill or not args.profile_only_prefill:
                standard_df = profile_model(
                        args,
                        model,
                        num_tensor_parallel_workers,
                        total_combos[(model, num_tensor_parallel_workers)],
                        max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                        model_dtypes[model],
                        pbar,
                )
                result_df = pd.concat([result_df, standard_df])
            
            # Profile mixed-length batch prefill
            if args.enable_mixed_prefill:
                mixed_df = profile_mixed_prefill(
                    args,
                    model,
                    num_tensor_parallel_workers,
                    filtered_mixed_combinations[(model, num_tensor_parallel_workers)],
                    max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                    model_dtypes[model],
                    pbar,
                )
                mixed_result_df = pd.concat([mixed_result_df, mixed_df])

            # Profile true mixed prefill+decode batches (co-location scenario)
            if args.enable_true_mixed:
                true_mixed_df = profile_true_mixed_batches(
                    args,
                    model,
                    num_tensor_parallel_workers,
                    filtered_true_mixed_combinations[(model, num_tensor_parallel_workers)],
                    max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                    model_dtypes[model],
                    pbar,
                )
                true_mixed_result_df = pd.concat([true_mixed_result_df, true_mixed_df])
        
        model_output_dir = build_profile_method_output_path(
            output_root=args.output_dir,
            profiling_type="compute",
            hardware=args.device,
            model_name=model,
            op_name="attention",
            profile_method=args.profile_method,
        ).parent
        model_output_dir.mkdir(parents=True, exist_ok=True)
        with (Path(args.output_dir) / "compute" / args.device / "attention_config.yaml").open(
            "w",
            encoding="utf-8",
        ) as config_file:
            import yaml

            yaml.dump(vars(args), config_file)

        # Load model config for metadata columns
        model_config = model_configs[model]
        precision_str = model_precision_strs[model]
        model_arch = _resolve_model_arch_for_metadata(model_config)
        model_architecture_profile = (
            model_config.get_model_architecture_profile().profile_id
        )
        quant_signature = model_config.get_quant_signature()
        measurement_type = profile_method_to_measurement_type(args.profile_method).value

        # Save standard attention results
        if not result_df.empty:
            result_df = _prepare_standard_attention_output_dataframe(
                result_df,
                precision_str=precision_str,
                model_arch=model_arch,
                model_architecture_profile=model_architecture_profile,
                quant_signature=quant_signature,
                measurement_type=measurement_type,
            )
            output_file = build_profile_method_output_path(
                output_root=args.output_dir,
                profiling_type="compute",
                hardware=args.device,
                model_name=model,
                op_name="attention",
                profile_method=args.profile_method,
            )
            result_df.to_csv(output_file, index=False)
            print(f"\nSaved standard attention results to {output_file}")

        # Save mixed-batch results separately
        if not mixed_result_df.empty:
            mixed_result_df = _attach_attention_output_metadata(
                mixed_result_df,
                precision_str=precision_str,
                model_arch=model_arch,
                model_architecture_profile=model_architecture_profile,
                quant_signature=quant_signature,
                measurement_type=measurement_type,
            )
            mixed_output_file = build_profile_method_output_path(
                output_root=args.output_dir,
                profiling_type="compute",
                hardware=args.device,
                model_name=model,
                op_name="attention_mixed",
                profile_method=args.profile_method,
            )
            mixed_result_df.to_csv(mixed_output_file, index=False)
            print(f"Saved mixed-batch attention results to {mixed_output_file}")

        # Save true mixed-batch results separately
        if not true_mixed_result_df.empty:
            true_mixed_result_df = _attach_attention_output_metadata(
                true_mixed_result_df,
                precision_str=precision_str,
                model_arch=model_arch,
                model_architecture_profile=model_architecture_profile,
                quant_signature=quant_signature,
                measurement_type=measurement_type,
            )
            true_mixed_output_file = build_profile_method_output_path(
                output_root=args.output_dir,
                profiling_type="compute",
                hardware=args.device,
                model_name=model,
                op_name="attention_true_mixed",
                profile_method=args.profile_method,
            )
            true_mixed_result_df.to_csv(true_mixed_output_file, index=False)
            print(
                "Saved true mixed-batch attention results to "
                f"{true_mixed_output_file}"
            )

        # Always export a combined file when any partition is available.
        combined_df = _build_attention_combined_df(
            result_df,
            mixed_result_df,
            true_mixed_result_df,
        )
        if not combined_df.empty:
            combined_output_file = build_profile_method_output_path(
                output_root=args.output_dir,
                profiling_type="compute",
                hardware=args.device,
                model_name=model,
                op_name="attention_combined",
                profile_method=args.profile_method,
            )
            combined_df.to_csv(combined_output_file, index=False)
            print(f"Saved combined results to {combined_output_file}")
    
    pbar.close()
    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    main()
