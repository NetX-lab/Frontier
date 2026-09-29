"""Time vLLM's tensor-parallel all-reduce on the GPUs of one node.

The ranks join vLLM's own distributed state (`init_distributed_environment`,
then `ensure_model_parallel_initialized` with one pipeline stage), so every
all-reduce goes through `tensor_model_parallel_all_reduce` and vLLM's default
dispatch in `CudaCommunicator.all_reduce`: custom all-reduce for messages below
its maximum size on a fully connected NVLink node, pynccl above. The path of
each message is read from `CustomAllreduce.should_custom_ar` and recorded.

A message holds the hidden states of `num_tokens` tokens of `--model` (hidden
size x tokens elements of the model's precision), the payload Frontier's
analytical backend prices for a TP all-reduce. Two modes are timed:

- `graph`: the `--graph_num_tokens` sizes, captured into one CUDA graph under
  vLLM's `graph_capture`, as the engine's FULL_DECODE_ONLY decode graphs are
  (custom all-reduce on graph-registered buffers).
- `eager`: the `--eager_num_tokens` sizes, called directly (custom all-reduce
  first copies the input into its registered buffer).

Each repeat runs `--calls_per_repeat` back-to-back all-reduces between CUDA
events after a CPU barrier, and the per-call time of rank 0 is recorded. Warmup
repeats are discarded.

The fit targets Frontier's analytical all-reduce time,
latency + 2 (n - 1) / n * bytes / bandwidth (`AnalyticalCCBackend.predict_allreduce`).
The bandwidth is the least-squares slope over the eager NCCL messages, and the
latency is the median over the graph custom all-reduce messages of the time
minus the bandwidth term. Every size records its fitted time and relative error.

Outputs, under `--output_dir`: `vllm_tp_allreduce.csv` with one measured row
per mode and size, and `vllm_tp_allreduce_summary.json` with the fit, the
per-size errors and the Frontier mapping (`intra_node_bandwidth_gbps`,
`network_latency_us`).

The worker must run from the Frontier tree, which the model configs are read
relative to.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from frontier.config.model_config import BaseModelConfig
from frontier.profiling.collectives.collectives_input import precision_to_dtype
from nccl_p2p_bench import bandwidth_gbps, fit_latency_bandwidth, free_port, time_statistics

CSV_NAME = "vllm_tp_allreduce.csv"
SUMMARY_NAME = "vllm_tp_allreduce_summary.json"
CSV_COLUMNS = [
    "mode", "num_tokens", "message_bytes", "effective_bytes", "path", "calls",
    "median_ms", "mean_ms", "min_ms", "repeats",
]
# (mode, path) of the messages each fitted term comes from.
BANDWIDTH_GROUP = ("eager", "nccl")
LATENCY_GROUP = ("graph", "custom")


def effective_bytes(message_bytes: int, world_size: int) -> float:
    """Ring all-reduce volume per device, as `AnalyticalCCBackend.predict_allreduce`."""
    return 2 * (world_size - 1) / world_size * message_bytes


def fit_allreduce(rows: list[dict]) -> dict:
    bandwidth_points = [(row["effective_bytes"], row["median_ms"]) for row in rows
                        if (row["mode"], row["path"]) == BANDWIDTH_GROUP]
    latency_rows = [row for row in rows if (row["mode"], row["path"]) == LATENCY_GROUP]
    if len(bandwidth_points) < 2 or not latency_rows:
        raise RuntimeError(
            f"the fit needs two {BANDWIDTH_GROUP} sizes and one {LATENCY_GROUP} size, got "
            f"{len(bandwidth_points)} and {len(latency_rows)}; see {CSV_NAME} for the recorded paths"
        )
    nccl_intercept_ms, ms_per_byte = fit_latency_bandwidth(bandwidth_points)
    latency_ms = statistics.median(
        row["median_ms"] - row["effective_bytes"] * ms_per_byte for row in latency_rows
    )
    sizes = []
    for row in rows:
        fit_ms = latency_ms + row["effective_bytes"] * ms_per_byte
        sizes.append({
            "mode": row["mode"], "num_tokens": row["num_tokens"], "path": row["path"],
            "median_ms": row["median_ms"], "fit_ms": fit_ms,
            "relative_error": (fit_ms - row["median_ms"]) / row["median_ms"],
        })

    def max_error(group: tuple[str, str] | None) -> float:
        return max(abs(size["relative_error"]) for size in sizes
                   if group is None or (size["mode"], size["path"]) == group)

    return {
        "bandwidth_from": list(BANDWIDTH_GROUP),
        "latency_from": list(LATENCY_GROUP),
        "ms_per_effective_byte": ms_per_byte,
        "nccl_intercept_ms": nccl_intercept_ms,
        "latency_ms": latency_ms,
        "max_abs_relative_error": {
            "_".join(BANDWIDTH_GROUP): max_error(BANDWIDTH_GROUP),
            "_".join(LATENCY_GROUP): max_error(LATENCY_GROUP),
            "all": max_error(None),
        },
        "frontier_mapping": {
            "intra_node_bandwidth_gbps": bandwidth_gbps(ms_per_byte),
            "network_latency_us": latency_ms * 1000,
        },
        "sizes": sizes,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict], world: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / CSV_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "world": world,
        "model": args.model,
        "timing": "rank 0 CUDA events around calls_per_repeat back-to-back all-reduces, after a "
                  "CPU barrier; per-call time; warmup repeats discarded",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "calls_per_repeat": args.calls_per_repeat,
        "fit": fit_allreduce(rows),
    }
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=1) + "\n")
    print("VLLM_TP_ALLREDUCE_SUMMARY", json.dumps(summary), flush=True)


def timed_repeats(run, cpu_group, warmup: int, repeats: int, calls: int) -> list[float]:
    import torch
    import torch.distributed as dist

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        start.record()
        run()
        end.record()
        end.synchronize()
        if index >= warmup:
            times_ms.append(start.elapsed_time(end) / calls)
    return times_ms


def run_rank(rank: int, args: argparse.Namespace, port: int) -> None:
    import torch
    import vllm
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(
        world_size=args.tensor_parallel_size, rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=rank,
    )
    ensure_model_parallel_initialized(args.tensor_parallel_size, 1)
    try:
        group = get_tp_group()
        custom = group.device_communicator.ca_comm
        model_config = BaseModelConfig.create_from_name(args.model)
        dtype = precision_to_dtype(model_config.get_default_precision().name)

        def eager_calls(inp):
            def run():
                for _ in range(args.calls_per_repeat):
                    tensor_model_parallel_all_reduce(inp)
            return run

        rows = []
        for mode, token_counts in (("graph", args.graph_num_tokens), ("eager", args.eager_num_tokens)):
            for num_tokens in token_counts:
                numel = model_config.embedding_dim * num_tokens
                if mode == "graph":
                    with graph_capture(device=device) as context:
                        inp = torch.ones(numel, dtype=dtype, device=device)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=context.stream):
                            for _ in range(args.calls_per_repeat):
                                tensor_model_parallel_all_reduce(inp)
                    run = graph.replay
                else:
                    inp = torch.ones(numel, dtype=dtype, device=device)
                    run = eager_calls(inp)
                message_bytes = numel * inp.element_size()
                row = {
                    "mode": mode, "num_tokens": num_tokens, "message_bytes": message_bytes,
                    "effective_bytes": effective_bytes(message_bytes, args.tensor_parallel_size),
                    "path": "custom" if custom is not None and custom.should_custom_ar(inp) else "nccl",
                    "calls": args.calls_per_repeat,
                }
                row.update(time_statistics(timed_repeats(
                    run, group.cpu_group, args.warmup, args.repeats, args.calls_per_repeat
                )))
                rows.append(row)
                del run, inp
        if rank == 0:
            write_outputs(args, rows, {
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "devices": [torch.cuda.get_device_name(i) for i in range(args.tensor_parallel_size)],
                "tensor_parallel_size": args.tensor_parallel_size,
                "hidden_size": model_config.embedding_dim,
                "dtype": str(dtype),
                "custom_all_reduce": None if custom is None else {
                    "disabled": custom.disabled,
                    "max_size": getattr(custom, "max_size", None),
                    "fully_connected": getattr(custom, "fully_connected", None),
                },
                "symm_mem_all_reduce": group.device_communicator.symm_mem_comm is not None,
            })
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--graph_num_tokens", type=int, nargs="+", required=True)
    parser.add_argument("--eager_num_tokens", type=int, nargs="+", required=True)
    parser.add_argument("--calls_per_repeat", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.multiprocessing as mp

    args = parse_args()
    if torch.cuda.device_count() < args.tensor_parallel_size:
        raise RuntimeError(
            f"needs {args.tensor_parallel_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(run_rank, args=(args, free_port()), nprocs=args.tensor_parallel_size, join=True)


if __name__ == "__main__":
    main()
