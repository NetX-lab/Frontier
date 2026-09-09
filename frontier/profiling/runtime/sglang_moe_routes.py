"""Profile exact route-conditioned Qwen3.8 MoE boundaries on one ROCm node.

The input is a Frontier runtime query plan, not observed timings. Each row is
keyed by the complete cost query (including per-expert physical token counts).
This module does not fit across route histograms or load a model checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path

from frontier.profiling.runtime.sglang_moe import (
    MOE_ROUTED_PRIMITIVES, make_moe_experts_primitive, make_moe_sorting_primitive,
)
from frontier.profiling.runtime.sglang_primitives import MEASUREMENT_TYPE, validate_plan
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload, RuntimeIdentity


def read_routed_queries(path, identity, components=MOE_ROUTED_PRIMITIVES):
    payload = json.loads(Path(path).read_text())
    if RuntimeIdentity(**payload["identity"]) != identity:
        raise ValueError("Route query plan belongs to another runtime")
    result = []
    for item in payload.get("queries", ()):
        raw = dict(item["query"])
        raw["identity"] = RuntimeIdentity(**raw["identity"])
        raw["workload"] = DecodeWorkload(**raw["workload"])
        raw["physical_expert_counts"] = tuple(raw.get("physical_expert_counts", ()))
        query = CostQuery(**raw)
        if query.key != item.get("query_key"):
            raise ValueError("Route query plan key disagrees with serialized query")
        if query.component in components:
            result.append(query)
    if (not result
            or len({(query.component, query.layer_id) for query in result}) != len(result)
            or len({query.workload for query in result}) != 1):
        raise ValueError("Route query plan requires unique routed layers for one workload")
    return tuple(sorted(result, key=lambda query: (query.component, query.layer_id)))


def read_sorting_queries(path, identity):
    return read_routed_queries(path, identity, ("moe_sorting",))


def _barrier(group):
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)


def profile_routed_graph(query, count, repetitions, model, group, *, trace=False):
    import torch

    size = query.workload.physical_size
    builder = {
        "moe_sorting": make_moe_sorting_primitive,
        "moe_experts_quant_gemm_combine": make_moe_experts_primitive,
    }.get(query.component)
    if builder is None:
        raise ValueError("Unsupported routed primitive query")
    calls = [builder(size, query.physical_expert_counts, model, group,
                     validate=index in {0, count - 1}) for index in range(count)]
    if len({call[3] for call in calls}) != 1:
        raise ValueError("One graph cannot mix routed primitive implementations")
    originals = [[value.clone() for value in call[2]] for call in calls]

    def reset(call_index):
        for value, original in zip(calls[call_index][2], originals[call_index]):
            value.copy_(original)

    def check(call_index, output):
        torch.testing.assert_close(
            output, calls[call_index][1], atol=.03, rtol=.03)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Nonfinite routed primitive output")

    for call_index in (0, count - 1):
        reset(call_index)
        check(call_index, calls[call_index][0]())
    for _ in range(3):
        for call_index, (fn, *_) in enumerate(calls):
            reset(call_index)
            fn()
    _barrier(group)
    graph, outputs = torch.cuda.CUDAGraph(), []
    with group.graph_capture() as context:
        with torch.cuda.graph(graph, stream=context.stream):
            for fn, *_ in calls:
                outputs.append(fn())
    _barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for repetition in range(repetitions + 3):
        for call_index in range(count):
            reset(call_index)
        _barrier(group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid routed graph timing")
        if repetition >= 3:
            samples.append(elapsed / count)
        if repetition in {0, repetitions + 2}:
            check(0, outputs[0])
            check(count - 1, outputs[-1])

    probe_calls, probe_outputs = (calls[0], calls[-1]), []
    probe = None
    if trace:
        probe = torch.cuda.CUDAGraph()
        with group.graph_capture() as context:
            with torch.cuda.graph(probe, stream=context.stream):
                for fn, *_ in probe_calls:
                    probe_outputs.append(fn())
        _barrier(group)
    scope = f"routed:{query.component}:l{query.layer_id}:b{size}:n{count}"

    def trace_replay():
        reset(0)
        reset(count - 1)
        _barrier(group)
        with torch.profiler.record_function(scope):
            probe.replay()
            torch.cuda.synchronize()
        check(0, probe_outputs[0])
        check(count - 1, probe_outputs[-1])
        _barrier(group)

    return {
        "primitive": query.component,
        "query_key": query.key,
        "layer_id": query.layer_id,
        "physical_size": size,
        "physical_expert_counts": list(query.physical_expert_counts),
        "invocations_per_graph": count,
        "rank": group.rank_in_group,
        "backend": calls[0][3],
        "samples_ms": samples,
        "measurement_type": MEASUREMENT_TYPE,
        "correctness_checked": True,
        "moe_routed_spec": calls[0][4],
        "moe_route_workload": calls[0][5],
        "kernel_names": [],
        "kernel_trace_kind": "representative_graph" if trace else None,
        "kernel_trace_invocations": 2 if trace else 0,
        "trace_scope": scope,
    }, trace_replay


def attach_kernel_evidence(records, events):
    for row in records:
        ranges = [event for event in events
                  if event.get("name") == row["trace_scope"]
                  and event.get("ph") == "X"
                  and event.get("cat") == "user_annotation"]
        if len(ranges) != 1:
            raise ValueError("Missing or duplicate routed replay trace range")
        scope = ranges[0]
        kernels = [event for event in events
                   if event.get("cat") == "kernel" and event.get("ph") == "X"
                   and scope["ts"] <= event["ts"]
                   and event["ts"] + event["dur"] <= scope["ts"] + scope["dur"]]
        if not kernels or len(kernels) % row["kernel_trace_invocations"]:
            raise ValueError("Incomplete or unexpected routed primitive kernel evidence")
        names = [event["name"] for event in kernels]
        if row["primitive"] == "moe_sorting":
            if any("aiter::opus_moe_sorting_entry" not in name for name in names):
                raise ValueError("Unexpected routed sorting kernel evidence")
        else:
            allowed = (
                "dynamic_per_group_scaled_quant_kernel", "mxfp4_moe_sort_kernel",
                "fused_mx_quant_moe_sort_kernel", "mfma_moe1_", "mfma_moe2_",
            )
            if (any(not any(marker in name for marker in allowed) for name in names)
                    or sum("mfma_moe1_" in name for name in names) != 2
                    or sum("mfma_moe2_" in name for name in names) != 2):
                raise ValueError("Unexpected routed expert kernel evidence")
        row["kernel_names"] = sorted({event["name"] for event in kernels})
        row["kernel_count"] = len(kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--query-plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--invocations", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--primitives", choices=MOE_ROUTED_PRIMITIVES, nargs="+",
                        default=list(MOE_ROUTED_PRIMITIVES))
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    validate_plan([1], args.invocations, args.repetitions, "calibration")

    manifest = json.loads(args.capture_manifest.read_text())
    if (manifest["status"] != "complete"
            or manifest["topology"] != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1}):
        raise ValueError("Routed profiling requires the captured eight-GPU TP-only topology")
    for key, value in manifest["environment"].items():
        if not key.startswith("SGLANG_"):
            raise ValueError("Only captured SGLang runtime environment entries may be restored")
        os.environ[key] = str(value)

    import aiter
    import sglang
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed.parallel_state import (
        get_tp_group, init_distributed_environment, initialize_model_parallel,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.runtime_cost.sglang import SGLangCostContract
    from frontier.validation.runtime_cost import identity_for_capture
    from frontier.validation.sglang_capture import _git_revision, _source_digest

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.version.hip or world != 8 or rank != local_rank:
        raise ValueError("Run with torchrun --nproc_per_node=8 on one ROCm node")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1701 + rank)
    actual = {
        "torch": torch.__version__, "rocm": torch.version.hip,
        "sglang_commit": _git_revision(Path(sglang.__file__).parent),
        "aiter_commit": _git_revision(Path(aiter.__file__).parent),
    }
    if any(value is None or value != manifest["versions"][key]
           for key, value in actual.items()):
        raise ValueError(f"Profiling stack differs from the pinned capture: {actual}")
    model = ModelConfig.from_model_name(args.model)
    identity = identity_for_capture(manifest, model, args.device)
    SGLangCostContract(model, identity)
    queries = read_routed_queries(args.query_plan, identity, tuple(args.primitives))
    if args.layers is not None:
        requested = set(args.layers)
        queries = tuple(query for query in queries if query.layer_id in requested)
        if {query.layer_id for query in queries} != requested:
            raise ValueError("Requested routed profile layer is absent from the plan")

    init_fields = {field.name for field in fields(ServerArgs) if field.init}
    server = ServerArgs(**{
        key: value for key, value in manifest["server_args"].items()
        if key in init_fields})
    set_global_server_args_for_scheduler(server)
    if os.environ.get("SGLANG_SET_CPU_AFFINITY") == "1":
        from sglang.benchmark.one_batch import set_gpu_proc_affinity
        set_gpu_proc_affinity(1, world, 1, rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank, timeout=180)
    initialize_model_parallel(tensor_model_parallel_size=world)
    group = get_tp_group()
    properties = torch.cuda.get_device_properties(local_rank)
    if args.device.lower() not in properties.name.lower().replace(" ", ""):
        raise ValueError("Requested hardware does not match the active GPU")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier(group=group.cpu_group)

    records = []
    with torch.inference_mode():
        for query in queries:
            query_rows, query_replays = [], []
            for count in args.invocations:
                row, replay = profile_routed_graph(
                    query, count, args.repetitions, model, group, trace=args.trace)
                query_rows.append(row)
                if args.trace:
                    query_replays.append(replay)
                if rank == 0:
                    print(json.dumps({key: row[key] for key in (
                        "primitive", "layer_id", "physical_size",
                        "invocations_per_graph", "backend")}), flush=True)
            if args.trace:
                # Trace and release one layer at a time. Expert graphs own a
                # streaming working set of TP-sharded weights; retaining every
                # layer's probe graph until the end would multiply HBM use.
                profiler = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if rank == 0 else nullcontext()
                with profiler:
                    for replay in query_replays:
                        replay()
                if rank == 0:
                    trace_path = args.output_dir / (
                        f"{query.component}-layer{query.layer_id}.trace.json")
                    profiler.export_chrome_trace(str(trace_path))
                    attach_kernel_evidence(
                        query_rows,
                        json.loads(trace_path.read_text())["traceEvents"])
            records.extend(query_rows)

    payload = {
        "schema_version": 1,
        "profile_kind": "exact_routed_query",
        "identity": asdict(identity),
        "rank": rank,
        "world_size": world,
        "versions": actual,
        "rows": records,
        "producer_source_sha256": _source_digest(Path(__file__).resolve().parents[2]),
        "hardware": {"name": properties.name, "arch": properties.gcnArchName},
        "extra_collective_environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("ROCM_QUICK_", "NCCL_"))},
        "method": "Exact query-conditioned HIP graph replay. Per-expert counts come from the "
                  "untimed routing plan and are deterministically reconstructed into valid unique "
                  "top-k assignments. Independent inputs, packed zero-valued weights and output "
                  "buffers per expert invocation preserve streaming extents with an exact-zero "
                  "oracle; sorting has no weights. No checkpoint, cross-route fit, full-forward "
                  "timing, or target correction.",
    }
    with (args.output_dir / f"rank{rank}.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    dist.barrier(group=group.cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
