"""Direct native KV zeroing and block-table writes; CPU plan, GPU worker run.

All measurement coordinates and allowances come from the JSON grid. The worker
calls the engine's own objects; it neither reconstructs their kernels nor runs
requests. Native tensors, pointer aliasing, page strides and virtual splitting
are observed at runtime, not inferred from layer count alone.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from frontier.attention.deepseek_v4 import deepseek_v4_kv_cache_groups
from frontier.profiling.attention.dsv4_attention_plan import _native_compress_ratios, _variant_of_ratio
from frontier.profiling.attention.dsv4_step_scopes import STEP_LAYER_ID, ZERO_SCOPE, WRITE_SCOPE
from frontier.profiling.model_runner.dsv4_runner_rows import COST, timing_row, write_csv

METHODS = ("cuda_event", "record_function")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class BlockGrid:
    payload: dict

    @property
    def layout(self):
        return self.payload["expected_layout"]

    def entries(self, block_sizes):
        tokens = self.payload["first_chunk_tokens"]
        result = [(tokens + size - 1) // size for size in block_sizes]
        if result != self.layout["entries_per_group"]:
            raise ValueError(f"First-chunk per-group entries {result} differ from expected {self.layout['entries_per_group']}")
        return result

    def rows(self):
        result = []
        for method in self.payload["methods"]:
            result.extend({"operator": ZERO_SCOPE, "n": n, "profile_method": method}
                          for n in self.payload["zero_block_counts"])
            result.extend({"operator": WRITE_SCOPE, "w_c": w, "r": r, "profile_method": method}
                          for w in self.payload["continuing_writes"] for r in self.payload["new_requests"]
                          if (w, r) != (0, 0))
        return result

    def validate_capacity(self, free_blocks, request_rows, block_sizes, row_widths=None):
        if max(self.payload["zero_block_counts"]) > free_blocks:
            raise ValueError(f"Zeroing n exceeds engine free blocks: {max(self.payload['zero_block_counts'])} > {free_blocks}")
        entries = self.entries(block_sizes)
        for row in self.rows():
            if row["operator"] == WRITE_SCOPE:
                used_rows = (row["w_c"] + len(entries) - 1) // len(entries) + row["r"]
                if row["r"] > request_rows or used_rows > request_rows:
                    raise ValueError(f"Write r / continuing rows exceed engine request rows: {used_rows} > {request_rows}")
        if max(entries) > free_blocks:
            raise ValueError("New-request row requires more block IDs than engine free blocks")
        if row_widths is not None and any(size > width for size, width in zip(entries, row_widths)):
            raise ValueError("First-chunk entries exceed engine block-table row width")
        return entries


def load_grid(path, hf_config):
    payload = json.loads(Path(path).read_text())
    for key in ("num_hidden_layers", "routed_expert_count", "first_chunk_tokens", "repeats", "step_hold_us"):
        _integer(payload[key], key)
    _integer(payload["warmup"], "warmup", 0)
    if payload["num_hidden_layers"] != hf_config["num_hidden_layers"]:
        raise ValueError("Grid layer count must equal native model config num_hidden_layers")
    if not hf_config["num_experts_per_tok"] <= payload["routed_expert_count"] <= hf_config["n_routed_experts"]:
        raise ValueError("Reduced expert count must be >= routing top-k and <= native routed expert count")
    if payload["methods"] != list(METHODS):
        raise ValueError(f"Grid methods must be {METHODS}")
    for key, minimum in (("zero_block_counts", 1), ("continuing_writes", 0), ("new_requests", 0)):
        values = payload[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must contain unique coordinates")
        for value in values:
            _integer(value, key, minimum)
    for key in ("segment_count", "chunks_per_page", "required_free_blocks", "request_rows"):
        _integer(payload["expected_layout"][key], key)
    grid = BlockGrid(payload)
    groups = deepseek_v4_kv_cache_groups([_variant_of_ratio(ratio) for ratio in
                                        _native_compress_ratios(hf_config)[:payload["num_hidden_layers"]]])
    if [len(group) for group in groups] != grid.layout["group_member_counts"]:
        raise ValueError("Expected KV groups differ from B4 native layer-layout rule")
    grid.validate_capacity(grid.layout["required_free_blocks"], grid.layout["request_rows"], grid.layout["block_sizes"])
    return grid


def write_staging_plan(w_c, r, entries):
    """Distinct continuing (group,row) pairs, then distinct full new rows."""
    groups = len(entries)
    continuing = []
    for index in range(w_c):
        group, row = index % groups, index // groups
        ids = tuple([1] if i == group else [] for i in range(groups))
        continuing.append((row, ids, False))
    first_new = (w_c + groups - 1) // groups
    new = [(first_new + i, tuple(list(range(1, n + 1)) for n in entries), True) for i in range(r)]
    return tuple(continuing + new)


def stage_writes(block_tables, plan):
    for row, ids, overwrite in plan:
        block_tables.append_block_ids(row, ids, overwrite=overwrite)


def runner_layout(runner):
    """Return metadata only; no addresses, credentials or tensor contents."""
    meta = runner.kv_block_zeroer._meta
    if meta is None:
        raise ValueError("Native engine has no KV-block zeroer segments")
    _, strides, pages, chunks, kernel_block, segments = meta
    groups = _cache_groups(runner)
    return {"segment_count": segments, "chunks_per_page": chunks,
            "zero_kernel_block_size": kernel_block,
            "segment_block_stride_int32": strides.cpu().tolist(),
            "segment_page_size_int32": pages.cpu().tolist(),
            "block_sizes": list(runner.block_tables.block_sizes),
            "kernel_block_sizes": list(runner.block_tables.kernel_block_sizes),
            "kv_cache_groups": groups, "group_member_counts": [len(group) for group in groups],
            "num_blocks": runner.kv_cache_config.num_blocks,
            "request_rows": runner.block_tables.max_num_reqs,
            "row_widths": [table.gpu.shape[1] for table in runner.block_tables.block_tables]}


def _cache_groups(runner):
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    from frontier.attention.deepseek_v4 import DEEPSEEK_V4_CACHES_BY_VARIANT, DEEPSEEK_V4_CACHE_METADATA_SOURCES
    from frontier.profiling.attention.dsv4_attention_runner import metadata_prefixes
    names = {}
    for _, module in runner.model.named_modules():
        if isinstance(module, DeepseekV4Attention):
            layer = int(module.prefix.split("layers.", 1)[1].split(".", 1)[0])
            prefixes = metadata_prefixes(module)
            for cache in DEEPSEEK_V4_CACHES_BY_VARIANT[_variant_of_ratio(module.compress_ratio)]:
                names[prefixes[DEEPSEEK_V4_CACHE_METADATA_SOURCES[cache]]] = (layer, cache)
    return [[list(member) for member in sorted(names[name] for name in group.layer_names)]
            for group in runner.kv_cache_config.kv_cache_groups]


def initial_writes(staging):
    grouped = {}
    for row, ids, overwrite in staging:
        if not overwrite:
            base = grouped.setdefault(row, [[] for _ in ids])
            for group, values in enumerate(ids):
                base[group].extend(values)
    return tuple((row, tuple(ids), True) for row, ids in sorted(grouped.items()))


def check_layout(grid, layout, hf_config):
    for key in ("segment_count", "chunks_per_page", "block_sizes", "group_member_counts"):
        if layout[key] != grid.layout[key]:
            raise ValueError(f"Native layout {key}: observed {layout[key]!r}, expected {grid.layout[key]!r}")
    expected = [[list(member) for member in group] for group in
                deepseek_v4_kv_cache_groups([_variant_of_ratio(ratio) for ratio in
                    _native_compress_ratios(hf_config)[:grid.payload["num_hidden_layers"]]])]
    if sorted(layout["kv_cache_groups"]) != sorted(expected):
        raise ValueError("Native KV-cache group members differ from B4 rule")
    if layout["kernel_block_sizes"] != layout["block_sizes"]:
        raise ValueError("Direct one-entry writes require native unsplit block-table entries")
    return grid.validate_capacity(layout["num_blocks"] - 1, layout["request_rows"],
                                  layout["block_sizes"], layout["row_widths"])


def measure_worker(worker, grid_payload, hf_config, output_dir):
    import torch
    from frontier.profiling.attention.dsa_kernel_trace import annotation_name
    from frontier.profiling.attention.dsv4_graph_replay_runner import _calibrate_hold
    from frontier.profiling.attention.graph_capture import CaptureRecord
    from frontier.profiling.attention.graph_replay_trace import (
        attribute_replay_trace, load_trace_events, hold_annotation_name, step_annotation_name,
    )
    runner = worker.model_runner
    grid = BlockGrid(grid_payload)
    layout = runner_layout(runner)
    entries = check_layout(grid, layout, hf_config)
    layout["num_hidden_layers"] = grid_payload["num_hidden_layers"]
    layout["routed_expert_count"] = grid_payload["routed_expert_count"]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    hold_cycles, cycles_per_us = _calibrate_hold(torch, grid_payload["step_hold_us"])
    layout["hold_cycles_per_us"] = cycles_per_us
    (root / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    result = []
    for coordinate in grid.rows():
        operator, method = coordinate["operator"], coordinate["profile_method"]
        label = "_".join(f"{k}-{v}" for k, v in coordinate.items())
        events = []
        if operator == ZERO_SCOPE:
            ids = list(range(1, coordinate["n"] + 1))
            prepare = lambda: None
            call = lambda: runner.kv_block_zeroer.zero_block_ids(ids)
        else:
            staging = write_staging_plan(coordinate["w_c"], coordinate["r"], entries)

            def prepare():
                # Restore a real initial entry through native APIs outside the
                # measured range; each measured continuation appends one entry.
                runner.block_tables.num_blocks.np.fill(0)
                base = initial_writes(staging)
                stage_writes(runner.block_tables, base)
                runner.block_tables.apply_staged_writes()
                stage_writes(runner.block_tables, staging)

            call = runner.block_tables.apply_staged_writes
        for _ in range(grid_payload["warmup"]):
            prepare()
            call()
        torch.cuda.synchronize()
        trace_path = root / f"{label}.trace.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for index in range(grid_payload["repeats"]):
                prepare()
                with torch.profiler.record_function(step_annotation_name(index)):
                    with torch.profiler.record_function(hold_annotation_name(index)):
                        torch.cuda._sleep(hold_cycles)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.profiler.record_function(annotation_name(index, STEP_LAYER_ID, operator)):
                        call()
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        attributed = attribute_replay_trace(load_trace_events(trace_path), CaptureRecord(), frozenset())
        intervals = attributed.intervals
        kernel, copies = [], []
        for index in range(grid_payload["repeats"]):
            rows = [row for row in intervals if row.step_index == index]
            hold = [row for row in rows if row.activity == "hold"]
            owned = [row for row in rows if row.owner == (STEP_LAYER_ID, operator)]
            if len(hold) != 1 or any(row.launch_end_us > hold[0].end_us for row in owned):
                raise ValueError(f"{label} repeat {index}: measured launch is not ahead of the hold; increase grid step_hold_us")
            measured_kernels = [row for row in owned if row.activity == "kernel"]
            if len(measured_kernels) != 1:
                raise ValueError(f"{label} repeat {index}: expected one native kernel, observed {len(measured_kernels)}")
            kernel.append(sum(row.end_us - row.start_us for row in measured_kernels) / 1000)
            copies.append(sum(row.end_us - row.start_us for row in owned if row.activity == "memcpy") / 1000)
        # Keep every interval, with activity/kind/owner/grid and launch times.
        from dataclasses import asdict
        trace_path.with_suffix(".intervals.json").write_text(json.dumps([asdict(row) for row in intervals], indent=2) + "\n")
        event_ms = [start.elapsed_time(end) for start, end in events] if method == "cuda_event" else None
        row = timing_row(operator, COST, {k: v for k, v in coordinate.items() if k not in ("operator", "profile_method")},
                         method, layout, kernel, copies, event_ms)
        row.update(trace_path=str(trace_path), warmup_count=grid_payload["warmup"],
                   kernel_count_per_repeat=json.dumps([sum(item.step_index == i and item.owner == (STEP_LAYER_ID, operator)
                                                          and item.activity == "kernel" for item in intervals)
                                                      for i in range(grid_payload["repeats"])]))
        result.append(row)
        write_csv(root / "block_operators.csv", result)
    return {"row_count": len(result), "layout": layout}


def direct_engine_plan(plan_args, hf_path, grid):
    """Reuse the G1h engine shapes, replacing only unused graph startup policy."""
    from dataclasses import replace
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.attention.dsa_plan import ENGINE_POLICY_EAGER_CHUNKED
    plan = build_plan(build_parser().parse_args([
        "plan", *plan_args, "--model_config_path", str(hf_path),
        "--num_layers", str(grid.payload["num_hidden_layers"]),
    ]))
    # Direct calls need the native cache allocation, without capturing model
    # graphs. The retained G1h cases are never executed by this entry point.
    return replace(plan, limits=replace(plan.limits, engine_policy=ENGINE_POLICY_EAGER_CHUNKED,
                                       cudagraph_mode=None, cudagraph_capture_sizes=(), step_hold_us=0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--model_config_path", type=Path, required=True)
    parser.add_argument("--plan_args", type=Path, help="JSON list of existing attention plan arguments")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model_dir")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_memory_utilization", type=float)
    args = parser.parse_args(argv)
    hf = json.loads(args.model_config_path.read_text())
    grid = load_grid(args.grid, hf)
    if args.dry_run:
        print(json.dumps({"rows": grid.rows(), "layout": grid.layout, "gpu_operations": 0}, indent=2))
        return 0
    if None in (args.plan_args, args.model_dir, args.output_dir, args.seed, args.gpu_memory_utilization):
        parser.error("run requires plan_args, model_dir, output_dir, seed and gpu_memory_utilization")
    from frontier.profiling.attention.dsv4_attention_runner import build_dsv4_engine
    plan = direct_engine_plan(json.loads(args.plan_args.read_text()), args.model_config_path, grid)
    # This selects the first existing engine, without a second engine builder.
    engine_index = plan.engines[0].engine_index
    llm = build_dsv4_engine(plan, engine_index, args.model_dir, args.seed, args.gpu_memory_utilization,
                            routed_expert_count=grid.payload["routed_expert_count"])
    result = llm.collective_rpc(measure_worker, args=(grid.payload, hf, str(args.output_dir)))[0]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
