"""Collective profiling entrypoint for CUDA/NCCL and ROCm/RCCL systems."""

from __future__ import annotations

import argparse
import datetime
import os
import socket
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

from frontier.logger import init_logger
from frontier.profiling.collectives.collectives_input import CollectivesInput
from frontier.profiling.common.accelerator import get_available_gpu_ids
from frontier.profiling.utils import get_collectives_inputs

logger = init_logger(__name__)

SUPPORTED_COLLECTIVE_PRECISIONS = ("FP16", "BF16", "FP32")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU collective profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Use local multiprocessing instead of Ray (single-node only).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of visible GPUs available to the local runner.",
    )
    parser.add_argument(
        "--num_workers_per_node_combinations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--max_collective_size",
        type=int,
        default=4096 * 8192,
        help="Maximum number of elements involved in the collective",
    )
    parser.add_argument(
        "--collective",
        default="all_reduce",
        choices=["all_reduce", "send_recv"],
        help="Collective to profile",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        choices=SUPPORTED_COLLECTIVE_PRECISIONS,
        help="Profiling precision type (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        dest="precision",
        type=str,
        choices=SUPPORTED_COLLECTIVE_PRECISIONS,
        help="Alias for --precision.",
    )
    args = parser.parse_args()

    if args.num_gpus <= 0:
        parser.error("--num_gpus must be positive")
    if any(value <= 0 for value in args.num_workers_per_node_combinations):
        parser.error("--num_workers_per_node_combinations must be positive")

    args.output_dir = os.path.join(
        args.output_dir,
        "collective",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def _precision_to_dtype(precision: str):
    import torch

    return {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "FP32": torch.float32,
    }[precision.upper()]


def _configure_collective_environment() -> None:
    # Ray can set this variable, but graph/profiler combinations in the legacy
    # runner reject it. These NCCL names are also honored by RCCL on ROCm.
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
    os.environ["KINETO_LOG_LEVEL"] = "5"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_collective_worker(
    rank: int,
    collectives_input: CollectivesInput,
    max_devices_per_node: int,
    master_port: int,
    result_queue: Any,
) -> None:
    import torch

    from frontier.profiling.collectives.collectives_wrapper import CollectiveWrapper

    _configure_collective_environment()
    torch.cuda.set_device(rank)
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=collectives_input.num_workers,
            init_method=f"tcp://127.0.0.1:{master_port}",
        )
        wrapper = CollectiveWrapper(
            rank=rank,
            num_workers=collectives_input.num_workers,
            comm_id=master_port,
            size=collectives_input.collective_size,
            collective=collectives_input.collective,
            devices_per_node=collectives_input.num_workers_per_node,
            max_devices_per_node=max_devices_per_node,
            dtype=_precision_to_dtype(collectives_input.precision),
        )
        result = wrapper.profile()
        if rank == 0:
            if not result["time_stats"]:
                raise RuntimeError(
                    "The GPU profiler did not capture a collective kernel. "
                    "Check Kineto NCCL/RCCL tracing support in this PyTorch build."
                )
            result_queue.put(result)
        torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_local_collective(
    collectives_input: CollectivesInput,
    max_devices_per_node: int,
) -> dict:
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    mp.spawn(
        _local_collective_worker,
        args=(
            collectives_input,
            max_devices_per_node,
            _find_free_local_port(),
            result_queue,
        ),
        nprocs=collectives_input.num_workers,
        join=True,
    )
    return result_queue.get()


def _create_ray_runner_pool():
    from frontier.profiling.collectives.benchmark_runner import BenchmarkRunner

    total_gpus_available = int(ray.cluster_resources()["GPU"])
    logger.info("Total GPUs available: %s", total_gpus_available)
    if total_gpus_available <= 0:
        raise RuntimeError("No GPUs are available to Ray.")

    all_node_ips = [node["NodeName"] for node in ray.nodes()]
    logger.info("All node IPs: %s", all_node_ips)
    if not all_node_ips:
        raise RuntimeError("Ray did not report any nodes.")

    num_nodes = len(all_node_ips)
    gpus_per_node = total_gpus_available // num_nodes
    runner_pool = []
    for gpu_id in range(total_gpus_available):
        node_ip = all_node_ips[gpu_id // gpus_per_node]
        runner_pool.append(
            BenchmarkRunner.options(
                resources={f"node:{node_ip}": 0.01}
            ).remote(gpu_id, gpus_per_node, all_node_ips[0])
        )
    return total_gpus_available, num_nodes, runner_pool


def _profile_with_ray(collectives_inputs, runner_pool) -> list[dict]:
    all_results = []
    for collectives_input in tqdm(collectives_inputs):
        promises = [
            runner.run_collective.remote(collectives_input)
            for runner in runner_pool
        ]
        results = ray.get(promises)
        if results and results[0] is not None:
            all_results.append(results[0])
    return all_results


def _profile_locally(collectives_inputs, num_gpus: int) -> list[dict]:
    return [
        _run_local_collective(collectives_input, num_gpus)
        for collectives_input in tqdm(collectives_inputs)
    ]


def _write_results(all_results: list[dict], args) -> str:
    if not all_results:
        raise RuntimeError("No valid collective profiling configurations were generated.")
    frame = pd.DataFrame(all_results)
    frame = (
        pd.json_normalize(frame["time_stats"])
        .add_prefix("time_stats.")
        .join(frame.drop(columns=["time_stats"]))
    )
    frame["profiling_precision"] = args.precision
    output_path = os.path.join(args.output_dir, f"{args.collective}.csv")
    frame.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()

    if args.disable_ray:
        available_gpus = get_available_gpu_ids(args.num_gpus)
        logger.info(
            "Local collective profiling on %s visible GPUs: %s",
            len(available_gpus),
            available_gpus,
        )
        total_gpus_available = len(available_gpus)
        num_nodes = 1
        runner_pool = None
    else:
        if not RAY_AVAILABLE:
            raise RuntimeError(
                "Ray is not installed. Use --disable_ray for native single-node "
                "multiprocessing."
            )
        ray.init()
        total_gpus_available, num_nodes, runner_pool = _create_ray_runner_pool()

    collectives_inputs = get_collectives_inputs(
        num_nodes=num_nodes,
        num_workers_per_node_combinations=args.num_workers_per_node_combinations,
        max_collective_size=args.max_collective_size,
        collective=args.collective,
        total_gpus_available=total_gpus_available,
        precision=args.precision,
    )
    if args.disable_ray:
        all_results = _profile_locally(collectives_inputs, total_gpus_available)
    else:
        all_results = _profile_with_ray(collectives_inputs, runner_pool)

    output_path = _write_results(all_results, args)
    logger.info("Wrote collective profiling results to %s", output_path)


if __name__ == "__main__":
    main()
