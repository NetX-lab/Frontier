"""Independent HIP graph replay for selected SGLang primitives.

Only isolated primitive callables are replayed.  This module intentionally does
not load a model, observe a router, restore a capture manifest, or emit standard
Frontier ``DEVICE_EVENT`` data.  CPU callers can validate plans and schemas;
SGLang, AITER, and CUDA/ROCm dependencies are imported inside GPU functions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .attention import ATTENTION_PRIMITIVES, make_attention_primitive
from .dense import DENSE_PRIMITIVES, dense_primitive_spec, make_dense_primitive
from .gdn import GDN_PRIMITIVES, gdn_core_spec, make_gdn_core_primitive
from .moe import MOE_ROUTING_PRIMITIVES, make_moe_routing_primitive

PRIMITIVES = (
    "gemma_norm",
    "gemma_residual_norm",
    "shared_gate",
    "attention_post",
    "tp_allreduce",
    *DENSE_PRIMITIVES,
    *GDN_PRIMITIVES,
    *ATTENTION_PRIMITIVES,
    *MOE_ROUTING_PRIMITIVES,
)
DEFAULT_PRIMITIVES = tuple(
    name
    for name in PRIMITIVES
    if name not in GDN_PRIMITIVES + ATTENTION_PRIMITIVES + MOE_ROUTING_PRIMITIVES
)
MEASUREMENT_TYPE = "HIP_GRAPH_REPLAY"


def validate_plan(
    sizes: Iterable[int], invocations: Iterable[int], repetitions: int, split: str
) -> None:
    """Validate a replay plan without touching an accelerator runtime."""

    sizes = tuple(sizes)
    invocations = tuple(invocations)
    if split not in {"calibration", "validation"}:
        raise ValueError("A calibration/validation split must be explicit")
    for name, values, minimum in (("sizes", sizes, 1), ("invocations", invocations, 2)):
        if (
            not values
            or len(values) != len(set(values))
            or any(type(value) is not int or value < minimum for value in values)
        ):
            raise ValueError(f"Invalid {name}")
    if type(repetitions) is not int or repetitions < 5:
        raise ValueError("At least five repetitions are required")


def _normalize_visibility(environment: Mapping[str, str] | None) -> dict[str, str]:
    if environment is None:
        environment = os.environ
    return {
        name: str(environment[name])
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        if str(environment.get(name, "")).strip()
    }


def local_rank_for_visibility(
    *, rank: int, world_size: int, environment: Mapping[str, str] | None = None
) -> int:
    """Return a deterministic local rank for a one-node visibility selection."""

    if type(rank) is not int or type(world_size) is not int or rank < 0 or world_size < 1:
        raise ValueError("rank and world_size must be non-negative/positive integers")
    visibility = _normalize_visibility(environment)
    selected = next(iter(visibility.values()), "")
    if selected:
        tokens = [token.strip() for token in selected.split(",") if token.strip()]
        if not tokens or any(not token.isdigit() for token in tokens):
            raise ValueError("ROCm/CUDA visibility must contain integer device IDs")
        if rank >= len(tokens):
            raise ValueError(
                f"rank {rank} is outside the visible device mapping of length {len(tokens)}"
            )
        return rank
    if rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")
    return rank


def build_replay_plan(
    *,
    primitive: str,
    sizes: Iterable[int],
    invocations: Iterable[int],
    repetitions: int,
    split: str,
    logical_sizes: Iterable[int] | None = None,
    context_lengths: Iterable[int] | None = None,
    environment: Mapping[str, str] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Build a serializable plan; no GPU package is imported."""

    if primitive not in PRIMITIVES:
        raise ValueError(f"Unknown experimental SGLang primitive: {primitive}")
    sizes = tuple(sizes)
    invocations = tuple(invocations)
    validate_plan(sizes, invocations, repetitions, split)
    if logical_sizes is not None:
        logical_sizes = tuple(logical_sizes)
        if len(logical_sizes) != len(sizes) or any(
            type(value) is not int or not 1 <= value <= size
            for value, size in zip(logical_sizes, sizes)
        ):
            raise ValueError("logical_sizes must contain one value in [1, size]")
    if context_lengths is not None:
        context_lengths = tuple(context_lengths)
        if len(context_lengths) != len(sizes) or any(
            type(value) is not int or value < 1 for value in context_lengths
        ):
            raise ValueError("context_lengths must contain one positive value per size")
    visibility = _normalize_visibility(environment)
    local_rank = local_rank_for_visibility(
        rank=rank, world_size=world_size, environment=environment
    )
    return {
        "schema_version": 1,
        "primitive": primitive,
        "sizes": list(sizes),
        "invocations": list(invocations),
        "repetitions": repetitions,
        "split": split,
        "logical_sizes": None if logical_sizes is None else list(logical_sizes),
        "context_lengths": None if context_lengths is None else list(context_lengths),
        "visibility": visibility,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "measurement_type": MEASUREMENT_TYPE,
        "experimental": True,
    }


def validate_replay_plan(plan: Mapping[str, Any]) -> None:
    """Validate a serialized plan and reject standard-data contamination."""

    required = {
        "schema_version", "primitive", "sizes", "invocations", "repetitions", "split",
        "logical_sizes", "context_lengths", "visibility", "rank", "world_size",
        "local_rank", "measurement_type", "experimental",
    }
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise ValueError("Invalid experimental replay plan schema")
    if plan["schema_version"] != 1 or plan["measurement_type"] != MEASUREMENT_TYPE:
        raise ValueError("Experimental replay plan has an unsupported measurement type")
    if plan["experimental"] is not True:
        raise ValueError("Experimental replay plans must be marked experimental")
    validate_plan(plan["sizes"], plan["invocations"], plan["repetitions"], plan["split"])
    local_rank_for_visibility(
        rank=plan["rank"], world_size=plan["world_size"], environment=plan["visibility"]
    )
    if plan["local_rank"] != plan["rank"]:
        raise ValueError("This one-node replay requires local_rank == rank")


def write_rank_artifact(
    output_dir: str | os.PathLike[str],
    *,
    rank: int,
    plan: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Write one per-rank JSON artifact, never a standard profiling CSV."""

    validate_replay_plan(plan)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "profile_kind": "experimental_sglang_primitive_replay",
        "experimental": True,
        "measurement_type": MEASUREMENT_TYPE,
        "plan": dict(plan),
        "rank": int(rank),
        "rows": [dict(row) for row in rows],
        "runtime": dict(runtime or {}),
        "visibility_provenance": dict(plan["visibility"]),
        "method": (
            "Independent primitive construction, correctness check, HIP graph capture, "
            "warmup, and event replay. Timings are research evidence only and are not "
            "accepted as standard DEVICE_EVENT training rows."
        ),
    }
    path = output_path / f"rank{int(rank)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def make_primitive(
    name: str,
    size: int,
    model: Any,
    group: Any,
    rank: int,
    *,
    logical_size: int | None = None,
    physical_context_lens: Iterable[int] | None = None,
):
    """Construct one selected native callable; imports happen at call time."""

    import torch
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add, fused_sigmoid_mul
    from sglang.srt.layers.layernorm import GemmaRMSNorm, _has_rocm_triton_gemma_rms_norm, _use_aiter
    from sglang.srt.layers.linear import RowParallelLinear

    hidden, dtype = model.embedding_dim, torch.bfloat16

    def random(shape, scale=1.0):
        return (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()

    x = random((size, hidden))
    mutable = []
    backend = "sglang_triton_gemma"
    attention_spec = attention_workload = moe_spec = None
    if name in ATTENTION_PRIMITIVES:
        if logical_size is None or physical_context_lens is None:
            raise ValueError("Attention primitive requires logical/context workload")
        return make_attention_primitive(
            name, size, logical_size, tuple(physical_context_lens), model, group, rank, random
        )
    if name in GDN_PRIMITIVES:
        if logical_size is None:
            raise ValueError("GDN primitive requires logical_size")
        return make_gdn_core_primitive(size, logical_size, model, group, random)
    if name in MOE_ROUTING_PRIMITIVES:
        return make_moe_routing_primitive(name, size, model, random)
    if name in DENSE_PRIMITIVES:
        return make_dense_primitive(name, size, model, group, rank, random)
    if name in {"gemma_norm", "gemma_residual_norm"}:
        if not (_use_aiter and _has_rocm_triton_gemma_rms_norm):
            raise ValueError("Expected the AITER-enabled SGLang Triton Gemma norm path")
        norm = GemmaRMSNorm(hidden, model.rms_norm_eps).to(device="cuda", dtype=dtype)
        norm.weight.data.copy_(random((hidden,), 0.05))
        residual = random(x.shape) if name == "gemma_residual_norm" else None
        mutable = [x] + ([residual] if residual is not None else [])
        total = x.float() + (residual.float() if residual is not None else 0.0)
        expected = ((total * torch.rsqrt(total.square().mean(-1, keepdim=True) + model.rms_norm_eps))
                    * (1.0 + norm.weight.float())).to(dtype)
        reference = (expected, total.to(dtype)) if residual is not None else expected
        fn = lambda: norm(x, residual)
    elif name == "shared_gate":
        weight, shared, final = random((hidden,), hidden ** -0.5), random(x.shape), random(x.shape)
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
        backing = random((size, heads, 2 * head_dim))
        gate = backing[..., head_dim:]
        linear = RowParallelLinear(model.num_q_heads * head_dim, hidden, bias=False,
                                   input_is_parallel=True, reduce_results=False,
                                   params_dtype=dtype, tp_rank=rank, tp_size=group.world_size).to(
                                       device="cuda", dtype=dtype)
        linear.weight.data.copy_(random(linear.weight.shape, width ** -0.5))
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

        def fn():
            return group.all_reduce(x)

        mutable = [x]
    else:
        raise ValueError(f"Unknown primitive {name}")

    originals = [value.clone() for value in mutable]

    def reset():
        for value, original in zip(mutable, originals):
            value.copy_(original)

    def check(output):
        values = output if isinstance(output, tuple) else (output,)
        refs = reference if isinstance(reference, tuple) else (reference,)
        if len(values) != len(refs):
            raise ValueError("Unexpected primitive output structure")
        for actual, expected in zip(values, refs):
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("Nonfinite primitive output")

    return fn, reset, check, backend, attention_spec, attention_workload, moe_spec


def profile_graph(*args, trace: bool = False, **kwargs):
    """Capture and replay a graph on an initialized ROCm process.

    The implementation is intentionally callable-only; no code path reaches
    this function during CPU planning. GPU timing and rank mapping remain
    unverified until an MI355X host is available.
    """

    import torch
    import torch.distributed as dist

    name, size, count, repetitions, model, group, rank = args[:7]
    calls = [make_primitive(name, size, model, group, rank, **kwargs) for _ in range(count)]
    methods = {call[3] for call in calls}
    if len(methods) != 1:
        raise ValueError("One graph cannot mix primitive implementations")

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)

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
    for repetition in range(repetitions + 3):
        for _, reset, *_ in calls:
            reset()
        barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid graph timing")
        if repetition >= 3:
            samples.append(elapsed / count)
        if repetition in {0, repetitions + 2}:
            calls[0][2](outputs[0])
            calls[-1][2](outputs[-1])

    probe = None
    probe_outputs = []
    if trace:
        probe = torch.cuda.CUDAGraph()
        with group.graph_capture() as context:
            with torch.cuda.graph(probe, stream=context.stream):
                for fn, *_ in (calls[0], calls[-1]):
                    probe_outputs.append(fn())
        barrier()

    def trace_replay():
        if probe is None:
            raise RuntimeError("trace replay requested without a trace probe")
        with torch.profiler.record_function(f"primitive:{name}:b{size}:n{count}"):
            probe.replay()
            torch.cuda.synchronize()
        for call, output in zip((calls[0], calls[-1]), probe_outputs):
            call[2](output)

    row = {
        "primitive": name,
        "physical_size": size,
        "invocations_per_graph": count,
        "rank": rank,
        "backend": next(iter(methods)),
        "samples_ms": samples,
        "measurement_type": MEASUREMENT_TYPE,
        "experimental": True,
        "correctness_checked": True,
        "kernel_names": [],
        "kernel_trace_kind": "representative_graph" if trace else None,
        "kernel_trace_invocations": 2 if trace else 0,
    }
    if name in DENSE_PRIMITIVES:
        row["dense_spec"] = dense_primitive_spec(name, model, group.world_size)
    if name in GDN_PRIMITIVES:
        row["logical_size"] = kwargs.get("logical_size")
        row["gdn_core_spec"] = gdn_core_spec(model, group.world_size)
    if name in ATTENTION_PRIMITIVES:
        row["logical_size"] = kwargs.get("logical_size")
        row["physical_context_lens"] = list(kwargs.get("physical_context_lens", ()))
    return row, trace_replay


def attach_kernel_evidence(records: list[dict[str, Any]], events: Iterable[Mapping[str, Any]]) -> None:
    """Attach identity-only kernel evidence to representative replay rows."""

    events = list(events)
    for row in records:
        scope_name = f"primitive:{row['primitive']}:b{row['physical_size']}:n{row['invocations_per_graph']}"
        ranges = [
            event for event in events
            if event.get("name") == scope_name
            and event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
        ]
        if len(ranges) != 1:
            raise ValueError("Missing or duplicate primitive replay trace range")
        scope = ranges[0]
        kernels = [
            event for event in events
            if event.get("cat") == "kernel"
            and event.get("ph") == "X"
            and float(scope["ts"]) <= float(event["ts"])
            and float(event["ts"]) + float(event["dur"]) <= float(scope["ts"]) + float(scope["dur"])
        ]
        probe_count = row.get("kernel_trace_invocations")
        if probe_count != 2 or not kernels or len(kernels) % probe_count:
            raise ValueError("Primitive graph trace has incomplete invocation coverage")
        row["kernel_names"] = sorted({str(event["name"]) for event in kernels})
        row["kernel_count"] = len(kernels)
