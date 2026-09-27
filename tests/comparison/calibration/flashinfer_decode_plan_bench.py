"""Compare FlashInfer's CUDA-graph decode plan with its eager decode plan.

In vLLM's FULL_DECODE_ONLY mode, decode attention replays a plan made by a
`BatchDecodeWithPagedKVCacheWrapper` built with `use_cuda_graph=True` for the
padded graph batch size, so split-KV is always on and the grid is sized for
that batch. Frontier's attention profiler plans the eager wrapper for the real
batch. This bench times the decode kernels of both plans on the same KV pages,
so that any difference is measured before a profiler change is proposed.

For each model, real batch size and KV length:

- `eager`: `use_cuda_graph=False`, planned for the real batch, as the profiler
  does;
- `graph`: `use_cuda_graph=True` with fixed buffers for the padded batch (the
  smallest capture size that holds the real batch), planned as vLLM plans a
  padded graph batch: the padded rows hold no KV pages and a last-page length
  of 1.

Both wrappers use tensor cores, the NHD layout and vLLM's workspace size, and
every sequence reads its own KV pages; the query is one token per row. Each
timed call follows a read of twice the L2 size, so it reads the KV cold, as
each engine layer does, and its time is the sum of its kernel durations from
the torch profiler (the profilers' kernel-only method).

Outputs, under `--output_dir`: `decode_plan_bench.csv` with one row per model,
plan, batch and KV length, and `decode_plan_bench_summary.json` with the
graph/eager ratio of the medians per point.

The worker must run from the Frontier tree, which the model configs are read
relative to.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from math import ceil
from pathlib import Path

CSV_NAME = "decode_plan_bench.csv"
SUMMARY_NAME = "decode_plan_bench_summary.json"


def padded_batch_size(batch_size: int, capture_sizes: list[int]) -> int:
    index = bisect.bisect_left(capture_sizes, batch_size)
    if index == len(capture_sizes):
        raise ValueError(
            f"batch size {batch_size} exceeds the largest capture size {capture_sizes[-1]}"
        )
    return capture_sizes[index]


def page_layout(kv_lens: list[int], num_rows: int, block_size: int):
    """Give every sequence its own consecutive pages; rows past the sequences hold none."""
    indptr = [0]
    last_page_lens = []
    for kv_len in kv_lens:
        indptr.append(indptr[-1] + ceil(kv_len / block_size))
        last_page_lens.append((kv_len - 1) % block_size + 1)
    for _ in range(num_rows - len(kv_lens)):
        indptr.append(indptr[-1])
        # FlashInfer reads a last-page length of 0 as a full page.
        last_page_lens.append(1)
    return indptr, last_page_lens


def bench_model(args: argparse.Namespace, model_name: str) -> list[dict]:
    import flashinfer
    import torch
    from vllm.v1.attention.backends.flashinfer import FLASHINFER_WORKSPACE_BUFFER_SIZE

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

    model = ModelConfig.from_model_name(model_name)
    num_qo_heads = model.num_q_heads
    num_kv_heads = model.get_runtime_num_kv_heads()
    head_dim = model.get_runtime_head_size()
    dtype = model.dtype
    device = torch.device("cuda")

    max_rows = padded_batch_size(max(args.batch_sizes), args.capture_sizes)
    max_pages = max_rows * ceil((max(args.kv_cache_sizes) + 1) / args.block_size)
    kv_cache = torch.randn(
        max_pages, 2, args.block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    workspace = torch.zeros(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=device)
    l2_flush_buffer = torch.empty(
        2 * torch.cuda.get_device_properties(device).L2_cache_size,
        dtype=torch.uint8,
        device=device,
    )
    indices_buffer = torch.arange(max_pages, dtype=torch.int32, device=device)
    eager_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True
    )
    graph_wrappers = {}

    def graph_wrapper(num_rows: int):
        if num_rows not in graph_wrappers:
            graph_wrappers[num_rows] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=torch.zeros(num_rows + 1, dtype=torch.int32, device=device),
                paged_kv_indices_buffer=indices_buffer,
                paged_kv_last_page_len_buffer=torch.zeros(num_rows, dtype=torch.int32, device=device),
                use_tensor_cores=True,
            )
        return graph_wrappers[num_rows]

    def plan(wrapper, kv_lens: list[int], num_rows: int) -> None:
        indptr, last_page_lens = page_layout(kv_lens, num_rows, args.block_size)
        wrapper.plan(
            torch.tensor(indptr, dtype=torch.int32),
            indices_buffer[: indptr[-1]],
            torch.tensor(last_page_lens, dtype=torch.int32),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            args.block_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    points = []
    with RecordFunctionTracer(args.output_dir) as tracer:
        for kv_cache_size in args.kv_cache_sizes:
            # The decode token is already in the cache when attention runs.
            kv_len = kv_cache_size + 1
            for batch_size in args.batch_sizes:
                num_rows = padded_batch_size(batch_size, args.capture_sizes)
                for plan_name, wrapper, rows in (
                    ("eager", eager_wrapper, batch_size),
                    ("graph", graph_wrapper(num_rows), num_rows),
                ):
                    plan(wrapper, [kv_len] * batch_size, rows)
                    query = torch.randn(rows, num_qo_heads, head_dim, dtype=dtype, device=device)
                    label = f"{model_name}|{plan_name}|{batch_size}|{kv_cache_size}"
                    wrapper.run(query, kv_cache)
                    for _ in range(args.repeats):
                        l2_flush_buffer.sum()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(f"vidur_{label}"):
                            wrapper.run(query, kv_cache)
                    points.append({
                        "label": label,
                        "model": model_name,
                        "plan": plan_name,
                        "batch_size": batch_size,
                        "planned_rows": rows,
                        "kv_cache_size": kv_cache_size,
                    })
    stats = tracer.get_operation_time_stats()
    for point in points:
        point_stats = stats[point.pop("label")]
        assert point_stats["count"] == args.repeats, point_stats
        for key in ("median", "min", "max"):
            point[f"kernel_time_ms_{key}"] = point_stats[key]
    return points


def write_outputs(args: argparse.Namespace, rows: list[dict]) -> None:
    output_dir = Path(args.output_dir)
    with (output_dir / CSV_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    medians = {
        (row["model"], row["batch_size"], row["kv_cache_size"], row["plan"]): row
        for row in rows
    }
    ratios = []
    for (model, batch_size, kv_cache_size, plan_name), eager in medians.items():
        if plan_name != "eager":
            continue
        graph = medians[(model, batch_size, kv_cache_size, "graph")]
        ratios.append({
            "model": model,
            "batch_size": batch_size,
            "graph_rows": graph["planned_rows"],
            "kv_cache_size": kv_cache_size,
            "eager_median_ms": eager["kernel_time_ms_median"],
            "graph_median_ms": graph["kernel_time_ms_median"],
            "graph_over_eager": graph["kernel_time_ms_median"] / eager["kernel_time_ms_median"],
        })
    summary = {
        "capture_sizes": args.capture_sizes,
        "block_size": args.block_size,
        "repeats": args.repeats,
        "max_abs_relative_difference": max(abs(r["graph_over_eager"] - 1) for r in ratios),
        "points": ratios,
    }
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=1) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--kv_cache_sizes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--capture_sizes", type=int, nargs="+", required=True,
        help="the engine's decode CUDA graph batch sizes",
    )
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    args.capture_sizes = sorted(args.capture_sizes)
    return args


def main() -> None:
    args = parse_args()
    Path(args.output_dir, "profiler_traces").mkdir(parents=True, exist_ok=True)
    rows = [row for model in args.models for row in bench_model(args, model)]
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
