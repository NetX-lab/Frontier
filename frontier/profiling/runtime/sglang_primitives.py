"""Profile SGLang's fused primitives and TP reduction without loading a checkpoint.

Run with torchrun on one ROCm node. Inputs/weights are synthetic and shape
matched; this is not full-model performance or numerical validation. Timings
bracket an isolated multi-call graph replay, not individual kernel durations.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path

from frontier.profiling.runtime.sglang_dense import DENSE_PRIMITIVES, dense_primitive_spec, make_dense_primitive
from frontier.profiling.runtime.sglang_gdn import GDN_PRIMITIVES, gdn_core_spec, make_gdn_core_primitive
from frontier.profiling.runtime.sglang_attention import (
    ATTENTION_PRIMITIVES, attention_decode_spec, make_attention_primitive,
)
from frontier.profiling.runtime.sglang_moe import (
    MOE_ROUTING_PRIMITIVES, make_moe_routing_primitive, moe_routing_spec,
)

PRIMITIVES = ("gemma_norm", "gemma_residual_norm", "shared_gate", "attention_post", "tp_allreduce",
              *DENSE_PRIMITIVES, *GDN_PRIMITIVES, *ATTENTION_PRIMITIVES,
              *MOE_ROUTING_PRIMITIVES)
DEFAULT_PRIMITIVES = tuple(name for name in PRIMITIVES
                           if name not in GDN_PRIMITIVES + ATTENTION_PRIMITIVES
                           + MOE_ROUTING_PRIMITIVES)
MEASUREMENT_TYPE = "HIP_GRAPH_REPLAY"


def validate_plan(sizes, invocations, repetitions, split):
    if split not in {"calibration", "validation"}:
        raise ValueError("A calibration/validation split must be explicit")
    for name, values, minimum in (("sizes", sizes, 1), ("invocations", invocations, 2)):
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v < minimum for v in values):
            raise ValueError(f"Invalid {name}")
    if type(repetitions) is not int or repetitions < 5:
        raise ValueError("At least five repetitions are required")


def make_primitive(name, size, model, group, rank, *, logical_size=None,
                   physical_context_lens=None):
    """Return one independent invocation; mutations are restored before replay."""
    import torch
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add, fused_sigmoid_mul
    from sglang.srt.layers.layernorm import GemmaRMSNorm, _has_rocm_triton_gemma_rms_norm, _use_aiter
    from sglang.srt.layers.linear import RowParallelLinear

    hidden, dtype = model.embedding_dim, torch.bfloat16
    def random(shape, scale=1.):
        return (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()
    x = random((size, hidden))
    mutable = []
    backend = "sglang_triton_gemma"
    attention_spec = attention_workload = moe_spec = None
    if name in ATTENTION_PRIMITIVES:
        fn, reference, mutable, backend, attention_spec, attention_workload = make_attention_primitive(
            name, size, logical_size, physical_context_lens, model, group, rank, random)
    elif name in GDN_PRIMITIVES:
        fn, reference, mutable, backend, _ = make_gdn_core_primitive(
            size, logical_size, model, group, random)
    elif name in MOE_ROUTING_PRIMITIVES:
        fn, reference, mutable, backend, moe_spec = make_moe_routing_primitive(
            name, size, model, random)
    elif name in DENSE_PRIMITIVES:
        fn, reference, backend = make_dense_primitive(name, size, model, group, rank, random)
    elif name in {"gemma_norm", "gemma_residual_norm"}:
        if not (_use_aiter and _has_rocm_triton_gemma_rms_norm):
            raise ValueError("Expected the AITER-enabled SGLang Triton Gemma norm path; native fallback is not a matching profile")
        norm = GemmaRMSNorm(hidden, model.rms_norm_eps).to(device="cuda", dtype=dtype)
        norm.weight.data.copy_(random((hidden,), .05))
        residual = random(x.shape) if name == "gemma_residual_norm" else None
        # ROCm's wrapper can be out-of-place; restoring inputs also guards
        # against accidental in-place behavior after a runtime change.
        mutable = [x] + ([residual] if residual is not None else [])
        total = x.float() + (residual.float() if residual is not None else 0.)
        expected = ((total * torch.rsqrt(total.square().mean(-1, keepdim=True) + model.rms_norm_eps))
                    * (1. + norm.weight.float())).to(dtype)
        reference = (expected, total.to(dtype)) if residual is not None else expected
        fn = lambda: norm(x, residual)
    elif name == "shared_gate":
        weight, shared, final = random((hidden,), hidden ** -.5), random(x.shape), random(x.shape)
        reference = (final.float() + torch.sigmoid(x.float() @ weight.float())[:, None] * shared.float()).to(dtype)
        mutable = [final]
        def fn():
            fused_gate_sigmoid_mul_add(x, weight, shared, final)
            return final
        backend = "sglang_triton"
    elif name == "attention_post":
        heads, head_dim = model.num_q_heads // group.world_size, model.get_head_dim()
        width = heads * head_dim
        x = random((size, width))
        # Qwen's Q/gate projection interleaves the Q and gate slice per head.
        backing = random((size, heads, 2 * head_dim))
        gate = backing[..., head_dim:]
        linear = RowParallelLinear(model.num_q_heads * head_dim, hidden, bias=False,
            input_is_parallel=True, reduce_results=False, params_dtype=dtype,
            tp_rank=rank, tp_size=group.world_size).to(device="cuda", dtype=dtype)
        linear.weight.data.copy_(random(linear.weight.shape, width ** -.5))
        gated = (x.float() * torch.sigmoid(gate.float().reshape(size, width))).to(dtype)
        reference = torch.nn.functional.linear(gated.float(), linear.weight.float()).to(dtype)
        mutable = [x]
        def fn():
            out = fused_sigmoid_mul(x, gate, inplace=True)
            return linear(out)[0]
        backend = f"sglang_triton+{type(linear.quant_method).__name__}"
    elif name == "tp_allreduce":
        x.fill_(rank + 1)
        reference = torch.full_like(x, group.world_size * (group.world_size + 1) / 2)
        backend = group._resolve_outplace_all_reduce_method(x)
        if backend not in {"ca", "qr"}:
            raise ValueError(f"Expected a custom out-of-place reduction, got {backend}")
        fn = lambda: group.all_reduce(x)
        mutable = [x]
    else:
        raise ValueError(f"Unknown primitive {name}")
    originals = [v.clone() for v in mutable]
    def reset():
        for value, original in zip(mutable, originals):
            value.copy_(original)
    def check(output):
        values = output if isinstance(output, tuple) else (output,)
        refs = reference if isinstance(reference, tuple) else (reference,)
        if len(values) != len(refs):
            raise ValueError("Unexpected primitive output structure")
        for actual, expected in zip(values, refs):
            torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("Nonfinite primitive output")
    return fn, reset, check, backend, attention_spec, attention_workload, moe_spec


def profile_graph(name, size, count, repetitions, model, group, rank, *, logical_size=None,
                  physical_context_lens=None, trace=False):
    import torch
    import torch.distributed as dist

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)
    calls = [make_primitive(name, size, model, group, rank, logical_size=logical_size,
                            physical_context_lens=physical_context_lens)
             for _ in range(count)]
    methods = {call[3] for call in calls}
    if len(methods) != 1:
        raise ValueError("One graph cannot mix primitive implementations")
    # Check one unfused-reference comparison before capture, including in-place
    # semantics. Every captured call owns its own mutable inputs and parameters.
    for fn, reset, check, *_ in (calls[0], calls[-1]):
        reset()
        check(fn())
    for _ in range(3):
        for fn, reset, *_ in calls:
            reset()
            fn()
    barrier()
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with group.graph_capture() as context:
        with torch.cuda.graph(graph, stream=context.stream):
            for fn, *_ in calls:
                outputs.append(fn())
    barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for rep in range(repetitions + 3):
        for _, reset, *_ in calls:
            reset()
        barrier()  # resets and rank synchronization are outside the timing interval
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid graph timing")
        if rep >= 3:
            samples.append(elapsed / count)
        if rep in {0, repetitions + 2}:
            calls[0][2](outputs[0])
            calls[-1][2](outputs[-1])
    # ROCTracer can omit large HIP graph batches entirely. Verify the same
    # callables in a representative two-invocation graph, not a claimed trace
    # of every timed invocation. This probe never supplies timing estimates.
    probe_calls, probe_outputs = (calls[0], calls[-1]), []
    probe = None
    if trace:
        probe = torch.cuda.CUDAGraph()
        with group.graph_capture() as context:
            with torch.cuda.graph(probe, stream=context.stream):
                for fn, *_ in probe_calls:
                    probe_outputs.append(fn())
        barrier()
    def trace_replay():
        for _, reset, *_ in probe_calls:
            reset()
        barrier()
        with torch.profiler.record_function(f"primitive:{name}:b{size}:n{count}"):
            probe.replay()
            torch.cuda.synchronize()
        for call, output in zip(probe_calls, probe_outputs):
            call[2](output)
        barrier()
    row = {"primitive": name, "physical_size": size, "invocations_per_graph": count,
        "rank": rank, "backend": next(iter(methods)), "samples_ms": samples,
        "measurement_type": MEASUREMENT_TYPE, "correctness_checked": True,
        "kernel_names": [], "kernel_trace_kind": "representative_graph" if trace else None,
        "kernel_trace_invocations": 2 if trace else 0}
    if name in DENSE_PRIMITIVES:
        row["dense_spec"] = dense_primitive_spec(name, model, group.world_size)
    if name in GDN_PRIMITIVES:
        row["logical_size"] = logical_size
        row["gdn_core_spec"] = gdn_core_spec(model, group.world_size)
    if name in ATTENTION_PRIMITIVES:
        row["logical_size"] = logical_size
        row["physical_context_lens"] = list(physical_context_lens)
        row["attention_decode_spec"] = calls[0][4]
        row["attention_workload"] = calls[0][5]
    if name in MOE_ROUTING_PRIMITIVES:
        row["moe_routing_spec"] = calls[0][6]
    return row, trace_replay


def attach_kernel_evidence(records, events):
    """Match synchronized replay ranges; never use profiler durations as costs."""
    for row in records:
        name, size, count = (row[k] for k in ("primitive", "physical_size", "invocations_per_graph"))
        ranges = [e for e in events if e.get("name") == f"primitive:{name}:b{size}:n{count}"
            and e.get("ph") == "X" and e.get("cat") == "user_annotation"]
        if len(ranges) != 1:
            raise ValueError("Missing or duplicate primitive replay trace range")
        scope = ranges[0]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"
            and scope["ts"] <= e["ts"] and e["ts"] + e["dur"] <= scope["ts"] + scope["dur"]]
        probe_count = row["kernel_trace_invocations"]
        if probe_count != 2 or not kernels or len(kernels) % probe_count:
            raise ValueError(f"Primitive graph trace has incomplete invocation coverage: {name}, b{size}, n{count}")
        row["kernel_names"] = sorted({e["name"] for e in kernels})
        row["kernel_count"] = len(kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--logical-sizes", type=int, nargs="+")
    parser.add_argument("--active-context-lengths", type=int, nargs="+")
    parser.add_argument("--invocations", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--primitives", choices=PRIMITIVES, nargs="+", default=list(DEFAULT_PRIMITIVES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    validate_plan(args.sizes, args.invocations, args.repetitions, args.split)
    stateful = GDN_PRIMITIVES + ATTENTION_PRIMITIVES
    if (any(name in stateful for name in args.primitives)
            and (args.logical_sizes is None or len(args.logical_sizes) != len(args.sizes)
                 or any(type(v) is not int or not 1 <= v <= p
                        for v, p in zip(args.logical_sizes, args.sizes)))):
        raise ValueError("Stateful profiling requires one valid --logical-sizes entry per physical size")
    if (any(name in ATTENTION_PRIMITIVES for name in args.primitives)
            and (args.active_context_lengths is None
                 or len(args.active_context_lengths) != len(args.sizes)
                 or any(type(v) is not int or v < 1 for v in args.active_context_lengths))):
        raise ValueError("Attention profiling requires one --active-context-lengths entry per size")
    manifest = json.loads(args.capture_manifest.read_text())
    if manifest["status"] != "complete" or manifest["topology"] != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1}:
        raise ValueError("Primitive profiling currently requires the captured eight-GPU TP-only topology")
    for key, value in manifest["environment"].items():
        if not key.startswith("SGLANG_"):
            raise ValueError("Only captured SGLang runtime environment entries may be restored")
        os.environ[key] = str(value)
    import torch
    import torch.distributed as dist
    import sglang
    import aiter
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, get_tp_group
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.runtime_cost.sglang import SGLangCostContract
    from frontier.validation.runtime_cost import identity_for_capture
    from frontier.validation.sglang_capture import _git_revision, _source_digest

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.version.hip or world != 8 or rank != local_rank:
        raise ValueError("Run with torchrun --nproc_per_node=8 on one ROCm node")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(17 + rank)
    actual = {"torch": torch.__version__, "rocm": torch.version.hip,
        "sglang_commit": _git_revision(Path(sglang.__file__).parent),
        "aiter_commit": _git_revision(Path(aiter.__file__).parent)}
    if any(value is None or value != manifest["versions"][key] for key, value in actual.items()):
        raise ValueError(f"Profiling stack differs from the pinned capture: {actual}")
    model = ModelConfig.from_model_name(args.model)
    identity = identity_for_capture(manifest, model, args.device)
    SGLangCostContract(model, identity)
    if model.dtype != torch.bfloat16:
        raise ValueError("Primitive profiling currently implements BF16 activations/dense weights only")
    init_fields = {f.name for f in fields(ServerArgs) if f.init}
    server = ServerArgs(**{k: v for k, v in manifest["server_args"].items() if k in init_fields})
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
    records, trace_replays = [], []
    with torch.inference_mode():
        for size_index, size in enumerate(args.sizes):
            for name in args.primitives:
                for count in args.invocations:
                    logical_size = args.logical_sizes[size_index] if name in stateful else None
                    physical_context_lens = None
                    if name in ATTENTION_PRIMITIVES:
                        active = args.active_context_lengths[size_index]
                        physical_context_lens = ((active,) * logical_size
                                                 + (1,) * (size - logical_size))
                    row, trace_replay = profile_graph(name, size, count, args.repetitions, model, group, rank,
                        logical_size=logical_size, physical_context_lens=physical_context_lens,
                        trace=args.trace)
                    records.append(row)
                    if args.trace:
                        trace_replays.append(trace_replay)
                    if rank == 0:
                        print(json.dumps({k: row[k] for k in ("primitive", "physical_size", "invocations_per_graph", "backend")}), flush=True)
        if args.trace:
            # One profiler session avoids repeated ROCm activity-collection windows.
            # Keep the graphs alive; all numerical/timing work above is unprofiled.
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) if rank == 0 else nullcontext()
            with profiler:
                for replay in trace_replays:
                    replay()
            if rank == 0:
                trace_path = args.output_dir / "kernels.trace.json"
                profiler.export_chrome_trace(str(trace_path))
                attach_kernel_evidence(records, json.loads(trace_path.read_text())["traceEvents"])
    payload = {"schema_version": 1, "identity": asdict(identity), "split": args.split,
        "rank": rank, "world_size": world, "versions": actual, "rows": records,
        "producer_source_sha256": _source_digest(Path(__file__).resolve().parents[2]),
        "hardware": {"name": properties.name, "arch": properties.gcnArchName},
        "extra_collective_environment": {k: v for k, v in os.environ.items() if k.startswith(("ROCM_QUICK_", "NCCL_"))},
        "method": "Outer event interval / independent invocations in one isolated graph replay; resets, "
                  "barriers, reference checks and traces excluded. Synthetic shape-matched inputs/weights; "
                  "independent weights per invocation (streaming working set), first/last outputs checked. "
                  "Kernel identity from separate representative two-call graphs, not full timed-graph coverage. "
                  "No checkpoint or full-forward calibration."}
    with (args.output_dir / f"rank{rank}.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    dist.barrier(group=group.cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
