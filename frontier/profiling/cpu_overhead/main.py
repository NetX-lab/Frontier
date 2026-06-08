import argparse
import datetime
import gc
import os
from itertools import product
from typing import Any, List

import pandas as pd

from frontier.config.precision_type import PrecisionType
from frontier.logger import init_logger
from frontier.profiling.cpu_overhead.analytical import (
    extrapolate_cpu_overhead_for_missing_tp,
)
from frontier.profiling.cpu_overhead.backends.factory import (
    create_cpu_overhead_backend,
    get_available_cpu_overhead_backends,
)
from frontier.profiling.cpu_overhead.backends.base_backend import (
    BaseCpuOverheadProfilerBackend,
)
from frontier.profiling.cpu_overhead.planning import resolve_single_node_tp_plan
from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe

logger = init_logger(__name__)


def _require_tqdm():
    try:
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CPU overhead profiling requires 'tqdm'. Install tqdm before profiling."
        ) from exc
    return tqdm


def _get_cpu_overhead_batch_sizes(max_batch_size: int) -> list[int]:
    try:
        from frontier.profiling.utils import (  # pylint: disable=import-outside-toplevel
            get_cpu_overhead_batch_sizes_to_profile,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CPU overhead profiling requires profiling utility dependencies "
            "(including torch). Install them before profiling."
        ) from exc
    return get_cpu_overhead_batch_sizes_to_profile(max_batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU Overhead Profiling")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
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
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum batch size to profile",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        choices=[p.name for p in PrecisionType],
        help="Profiling precision type (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit batch sizes for profiling. When provided, "
            "--max_batch_size is ignored."
        ),
    )
    parser.add_argument(
        "--dtype",
        dest="precision",
        type=str,
        choices=[p.name for p in PrecisionType],
        help="Alias for --precision.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sarathi",
        choices=list(get_available_cpu_overhead_backends()),
        help="CPU overhead profiling backend (default: %(default)s).",
    )
    parser.add_argument(
        "--single_node_gpu_capacity",
        type=int,
        default=None,
        help=(
            "Override local single-node GPU capacity used for TP planning. "
            "By default, backend runtime will auto-detect."
        ),
    )
    parser.add_argument(
        "--enable_analytical_tp_modeling",
        action="store_true",
        help=(
            "Enable analytical TP extrapolation when requested TP degree exceeds "
            "single-node measurable capacity."
        ),
    )
    parser.add_argument(
        "--vllm_cpu_overhead_input_file",
        type=str,
        default=None,
        help=(
            "Replay input file (.json/.jsonl) for --backend vllm. "
            "Required when backend=vllm."
        ),
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/cpu_overhead/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    model_name: str,
    batch_sizes_to_profile: List[int],
    tensor_parallel_degrees: List[int],
    output_dir: str,
    pbar: Any,
    precision: str,
    backend: BaseCpuOverheadProfilerBackend,
    single_node_gpu_capacity: int | None,
    enable_analytical_tp_modeling: bool,
) -> dict:
    measurable_tp_degrees, missing_tp_degrees = resolve_single_node_tp_plan(
        requested_tp_degrees=tensor_parallel_degrees,
        single_node_gpu_capacity=single_node_gpu_capacity,
    )
    if missing_tp_degrees and not enable_analytical_tp_modeling:
        raise ValueError(
            "Requested TP degrees exceed single-node measurable capacity. "
            f"missing_tp_degrees={missing_tp_degrees}, "
            f"single_node_gpu_capacity={single_node_gpu_capacity}. "
            "Either reduce --num_tensor_parallel_workers or set "
            "--enable_analytical_tp_modeling."
        )
    if not measurable_tp_degrees:
        raise ValueError(
            "No measurable TP degree remains for CPU overhead profiling. "
            "Provide at least one TP degree within single-node capacity."
        )

    results = []

    for tensor_parallel_degree in measurable_tp_degrees:
        for batch_index, batch_size in enumerate(batch_sizes_to_profile):
            try:
                runner = backend.create_runner(
                    model_name,
                    batch_size,
                    tensor_parallel_degree,
                    output_dir,
                    precision,
                )
                results.append(backend.run_runner(runner))
                del runner
                # trigger garbage collection
                gc.collect()
            except Exception as e:
                logger.error(
                    f"Failed to run {model_name}_{batch_size}_{tensor_parallel_degree} due to {e}"
                )
                # update progress bar
                pbar.update(len(batch_sizes_to_profile) - batch_index)
                break

            pbar.update(1)

    if not results:
        raise ValueError(
            "CPU overhead profiling produced no measured rows. "
            f"model_name={model_name}, measurable_tp_degrees={measurable_tp_degrees}."
        )

    df = pd.DataFrame(results)
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    df["profiling_precision"] = precision
    if "cpu_overhead_source" not in df.columns:
        df["cpu_overhead_source"] = "measured"
    if missing_tp_degrees:
        logger.warning(
            "Applying analytical TP modeling for missing TP degrees %s "
            "(single-node measurable TP degrees: %s).",
            missing_tp_degrees,
            measurable_tp_degrees,
        )
        df = extrapolate_cpu_overhead_for_missing_tp(
            measured_df=df,
            target_tp_degrees=tensor_parallel_degrees,
        )
    df = validate_cpu_overhead_dataframe(df, expected_precision=precision)
    df.to_csv(f"{output_dir}/{model_name}/cpu_overhead.csv")


def main():
    args = parse_args()
    backend = create_cpu_overhead_backend(
        args.backend,
        vllm_cpu_overhead_input_file=args.vllm_cpu_overhead_input_file,
    )
    backend.start()

    try:
        detected_single_node_capacity = backend.get_local_gpu_capacity()
        effective_single_node_capacity = (
            args.single_node_gpu_capacity
            if args.single_node_gpu_capacity is not None
            else detected_single_node_capacity
        )
        if (
            args.single_node_gpu_capacity is not None
            and detected_single_node_capacity is not None
            and args.single_node_gpu_capacity != detected_single_node_capacity
        ):
            logger.warning(
                "Overriding backend-detected single-node GPU capacity %s with CLI "
                "value %s.",
                detected_single_node_capacity,
                args.single_node_gpu_capacity,
            )

        if args.batch_sizes is not None:
            batch_sizes_to_profile = sorted(set(int(v) for v in args.batch_sizes))
            if not batch_sizes_to_profile:
                raise ValueError("--batch_sizes must contain at least one value.")
            if any(value <= 0 for value in batch_sizes_to_profile):
                raise ValueError(
                    "--batch_sizes values must be positive integers, "
                    f"got {batch_sizes_to_profile}"
                )
        else:
            batch_sizes_to_profile = _get_cpu_overhead_batch_sizes(args.max_batch_size)

        input_combos = product(
            args.models, args.num_tensor_parallel_workers, batch_sizes_to_profile
        )

        tqdm = _require_tqdm()
        pbar = tqdm(total=len(list(input_combos)))

        for model_name in args.models:
            profile_model(
                model_name,
                batch_sizes_to_profile,
                args.num_tensor_parallel_workers,
                args.output_dir,
                pbar,
                args.precision,
                backend,
                effective_single_node_capacity,
                args.enable_analytical_tp_modeling,
            )
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
