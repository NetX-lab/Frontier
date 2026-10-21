"""Profile complete native Step SWA and sparse-GQA attention layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5.inputs import build_profile_inputs, read_profile_inputs
from frontier.profiling.step5.rows import rows_to_dataframe
from frontier.profiling.utils import build_profile_method_output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stepfun-ai/Step-5-Preview")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="h800")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--prefill-tokens", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--input-file", type=Path,
                        help="Explicit Step5ProfileInput shapes, instead of the Cartesian grid")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--prefixes", nargs="+", type=int, default=[0, 512])
    parser.add_argument("--prefill-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--decode-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--decode-contexts", nargs="+", type=int, default=[512, 4104])
    parser.add_argument("--cudagraph-capture-sizes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--max-time-cv", type=float, default=0.05)
    parser.add_argument("--min-timing-window-ms", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = ModelConfig.from_model_name(args.model)
    if args.warmup_iterations < 1 or args.profile_iterations < 2:
        raise ValueError("At least one warmup and two measured iterations are required")
    if not 0 < args.max_time_cv < 1:
        raise ValueError("max-time-cv must lie strictly between zero and one")
    if not math.isfinite(args.min_timing_window_ms) or args.min_timing_window_ms <= 0:
        raise ValueError("min-timing-window-ms must be positive and finite")
    cells = (read_profile_inputs(args.input_file,
                                cudagraph_capture_sizes=args.cudagraph_capture_sizes)
             if args.input_file else build_profile_inputs(
        config, prefill_tokens=args.prefill_tokens, prefixes=args.prefixes,
        prefill_batches=args.prefill_batches, decode_batches=args.decode_batches,
        decode_contexts=args.decode_contexts, cudagraph_capture_sizes=args.cudagraph_capture_sizes,
    ))
    from frontier.attention.step5 import resolve_step5_attention_geometry

    for cell in cells:
        cell.workload(config, block_size=args.block_size, max_model_len=args.max_model_len)
        resolve_step5_attention_geometry(config, cell.global_layer_id,
                                        tensor_parallel_size=args.tensor_parallel_size)
    if args.dry_run:
        from dataclasses import asdict

        print(json.dumps({"rows": len(cells),
                          "gpu_active_rows": sum(c.graph_mode == "none" for c in cells),
                           "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
                           "enable_chunked_prefill": not args.disable_chunked_prefill,
                           "native_max_num_seqs": max(c.batch_size for c in cells),
                           "native_max_num_batched_tokens": args.max_model_len * max(c.batch_size for c in cells),
                          "cells": [asdict(c) for c in cells]}, indent=2))
        return
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    rows = []
    gpu_active_rows = []
    output = Path(build_profile_method_output_path(
        output_root=args.output_dir, profiling_type="compute", hardware=args.device,
        model_name=args.model, op_name="step5_attention", profile_method="cuda_event"))
    gpu_active_output = Path(build_profile_method_output_path(
        output_root=args.output_dir, profiling_type="compute", hardware=args.device,
        model_name=args.model, op_name="step5_attention", profile_method="record_function"))
    output.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    written_gpu_active_rows = 0
    complete = False
    try:
        for layer_id, graph_mode in sorted({(c.global_layer_id, c.graph_mode) for c in cells}):
            with VllmStep5AttentionWrapper(
                config=config, model_path=args.model_path, device=args.device,
                tensor_parallel_size=args.tensor_parallel_size, layer_id=layer_id,
                block_size=args.block_size, max_model_len=args.max_model_len,
                max_batch_size=max(c.batch_size for c in cells),
                trace_dir=Path(args.output_dir) / "traces",
                seed=args.seed,
                graph_mode=graph_mode,
                cudagraph_capture_sizes=tuple(args.cudagraph_capture_sizes),
                enable_chunked_prefill=not args.disable_chunked_prefill,
            ) as wrapper:
                for cell in (c for c in cells if c.global_layer_id == layer_id and c.graph_mode == graph_mode):
                    result = wrapper.profile(
                        cell, warmup_iterations=args.warmup_iterations,
                        profile_iterations=args.profile_iterations, max_time_cv=args.max_time_cv,
                        min_timing_window_ms=args.min_timing_window_ms)
                    first = result
                    if not result.calibration_row["stability_pass"]:
                        # Retain the first outcome, then make the one configured
                        # stability retry explicit rather than hiding it.
                        result = wrapper.profile(
                            cell, warmup_iterations=args.warmup_iterations,
                            profile_iterations=args.profile_iterations, max_time_cv=args.max_time_cv,
                            min_timing_window_ms=args.min_timing_window_ms)
                    for current, initial in ((result.event_row, first.event_row),
                                             (result.gpu_active_row, first.gpu_active_row)):
                        if current is not None:
                            current["initial_time_cv"] = initial["time_cv"]
                            current["initial_trace_path"] = initial["trace_path"]
                            current["initial_time_stats"] = json.dumps(initial["time_stats"])
                            current["stability_attempts"] = 1 if result is first else 2
                    rows.append(result.event_row)
                    if result.gpu_active_row is not None:
                        gpu_active_rows.append(result.gpu_active_row)
                    rows_to_dataframe(rows).to_csv(output, index=False)
                    if gpu_active_rows:
                        rows_to_dataframe(gpu_active_rows, measurement_type="KERNEL_ONLY").to_csv(
                            gpu_active_output, index=False)
                    written_rows = len(rows)
                    written_gpu_active_rows = len(gpu_active_rows)
                    print("STEP5_CELL_DONE", len(rows), cell, flush=True)
        complete = True
    finally:
        summary = {"expected_rows": len(cells), "actual_rows": written_rows,
                   "complete": complete, "csv": str(output),
                    "gpu_active_csv": str(gpu_active_output),
                    "gpu_active_rows": written_gpu_active_rows,
                    "expected_gpu_active_rows": sum(c.graph_mode == "none" for c in cells),
                    "stable_rows": sum(bool(row["stability_pass"]) for row in rows[:written_rows]
                                       if row["graph_mode"] != "none")
                                   + sum(bool(row["stability_pass"]) for row in gpu_active_rows[:written_gpu_active_rows]),
                   "admitted": False,
                   "limits": "Development measurements; full-grid qualification and typed admission remain open."}
        (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("STEP5_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
