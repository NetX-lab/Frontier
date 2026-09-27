"""Measure NCCL point-to-point transfer time between two GPUs of one node.

Two processes on GPUs 0 and 1 join a torch.distributed NCCL group, and rank 0
sends to rank 1. Two patterns are timed:

- `size_sweep`: one message per size, powers of two from `--min_bytes` to
  `--max_bytes`. The medians are fitted by least squares to
  time = latency + bytes / bandwidth.
- `request`: the per-request KV transfer of vLLM's P2P NCCL connector in
  PUT_ASYNC mode, one message per layer of
  kv_factor * tokens * kv_heads * head_dim * dtype_bytes, sent back to back.
  The shapes come from the model configs under `data/config/models/`, sized
  by the same layout the simulator's analytical KV-cache transfer uses.

Each repeat starts after a barrier and is timed by the receiver with CUDA
events, so it ends when the last message has arrived. Warmup repeats are
discarded.

Outputs, under `--output_dir`: `nccl_p2p.csv` with one row per pattern, and
`nccl_p2p_summary.json` with the fit and the Frontier mapping per model. The
mapping takes `network_bandwidth_gbps` from the fitted bandwidth and sets
`network_latency_ms` to the measured per-request time minus the request bytes
over that bandwidth.

The worker must run from the Frontier tree, which the model configs are read
relative to.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
from pathlib import Path

from frontier.attention.memory import get_attention_runtime_kv_layout
from frontier.attention.model_binding import bind_attention_family
from frontier.config.model_config import BaseModelConfig

SENDER, RECEIVER = 0, 1
CSV_NAME = "nccl_p2p.csv"
SUMMARY_NAME = "nccl_p2p_summary.json"
CSV_COLUMNS = [
    "kind", "model", "message_bytes", "messages", "total_bytes",
    "median_ms", "mean_ms", "min_ms", "repeats",
]


def request_messages(model_name: str, num_tokens: int) -> tuple[int, int]:
    """Return (layers, bytes per layer message) of one request's KV cache."""
    model_config = BaseModelConfig.create_from_name(model_name)
    layout = get_attention_runtime_kv_layout(
        bind_attention_family(model_config).family,
        runtime_num_kv_heads_per_worker=model_config.get_runtime_num_kv_heads(),
        runtime_head_size=model_config.get_runtime_head_size(),
    )
    dtype_bytes = model_config.get_default_precision().bytes_per_element
    message_bytes = (
        layout.kv_factor * num_tokens * layout.runtime_num_kv_heads_per_worker
        * layout.runtime_head_size * dtype_bytes
    )
    return model_config.num_layers, int(message_bytes)


def sweep_sizes(min_bytes: int, max_bytes: int) -> list[int]:
    if min_bytes <= 0 or min_bytes & (min_bytes - 1) or max_bytes < min_bytes:
        raise ValueError(
            f"min_bytes must be a power of two no larger than max_bytes, "
            f"got {min_bytes} and {max_bytes}"
        )
    sizes = []
    size = min_bytes
    while size <= max_bytes:
        sizes.append(size)
        size *= 2
    return sizes


def fit_latency_bandwidth(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares fit of ms = latency_ms + bytes * ms_per_byte."""
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    ms_per_byte = sxy / sxx
    return mean_y - ms_per_byte * mean_x, ms_per_byte


def bandwidth_gbps(ms_per_byte: float) -> float:
    """Convert to the unit of Frontier's `network_bandwidth_gbps`.

    Frontier divides bytes by gbps * 1e9 / (8 * 1000) bytes per millisecond.
    """
    return 8 * 1000 / (ms_per_byte * 1e9)


def frontier_transfer_mapping(
    request_bytes: int, request_ms: float, ms_per_byte: float
) -> dict:
    return {
        "network_bandwidth_gbps": bandwidth_gbps(ms_per_byte),
        "network_latency_ms": request_ms - request_bytes * ms_per_byte,
    }


def time_statistics(times_ms: list[float]) -> dict:
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "repeats": len(times_ms),
    }


def timed_repeats(rank: int, messages: list, warmup: int, repeats: int) -> list[float]:
    import torch
    import torch.distributed as dist

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])
        start.record()
        for message in messages:
            if rank == SENDER:
                dist.send(message, dst=RECEIVER)
            else:
                dist.recv(message, src=SENDER)
        end.record()
        end.synchronize()
        if index >= warmup:
            times_ms.append(start.elapsed_time(end))
    return times_ms


def patterns(args: argparse.Namespace) -> list[dict]:
    rows = [
        {"kind": "size_sweep", "model": "", "message_bytes": size, "messages": 1}
        for size in sweep_sizes(args.min_bytes, args.max_bytes)
    ]
    for model in args.models:
        layers, message_bytes = request_messages(model, args.num_tokens)
        rows.append({
            "kind": "request", "model": model,
            "message_bytes": message_bytes, "messages": layers,
        })
    for row in rows:
        row["total_bytes"] = row["message_bytes"] * row["messages"]
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict], world: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / CSV_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    sweep = [(row["message_bytes"], row["median_ms"]) for row in rows
             if row["kind"] == "size_sweep"]
    latency_ms, ms_per_byte = fit_latency_bandwidth(sweep)
    residuals = [
        abs(y - (latency_ms + x * ms_per_byte)) / y for x, y in sweep
    ]
    summary = {
        "world": world,
        "timing": "receiver CUDA events per repeat, after a barrier; warmup repeats discarded",
        "num_tokens": args.num_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "fit": {
            "latency_ms": latency_ms,
            "ms_per_byte": ms_per_byte,
            "bandwidth_gbps": bandwidth_gbps(ms_per_byte),
            "points": len(sweep),
            "max_relative_residual": max(residuals),
        },
        "requests": [
            {
                "model": row["model"],
                "layers": row["messages"],
                "message_bytes": row["message_bytes"],
                "request_bytes": row["total_bytes"],
                "median_ms": row["median_ms"],
                **frontier_transfer_mapping(row["total_bytes"], row["median_ms"], ms_per_byte),
            }
            for row in rows if row["kind"] == "request"
        ],
    }
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=1) + "\n")
    print("NCCL_P2P_SUMMARY", json.dumps(summary), flush=True)


def run_rank(rank: int, args: argparse.Namespace, port: int) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        device_id=torch.device("cuda", rank),
    )
    try:
        rows = patterns(args)
        for row in rows:
            messages = [
                torch.empty(row["message_bytes"], dtype=torch.uint8, device="cuda")
                for _ in range(row["messages"])
            ]
            row.update(time_statistics(
                timed_repeats(rank, messages, args.warmup, args.repeats)
            ))
            del messages
        if rank == RECEIVER:
            write_outputs(args, rows, {
                "torch": torch.__version__,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "devices": [torch.cuda.get_device_name(i) for i in range(2)],
            })
    finally:
        dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--min_bytes", type=int, required=True)
    parser.add_argument("--max_bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.multiprocessing as mp

    args = parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"needs two visible GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(run_rank, args=(args, free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    main()
