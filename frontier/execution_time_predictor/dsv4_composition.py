"""Device schedule of one DeepSeek-V4 layer's attention phase.

The attention layer issues its compressor and indexer scopes on auxiliary
streams. The phase elapsed is the critical path through the layer's fork/join
stream program, with every scope weighted by its device cost plus the time
between its kernels. Host time is zero: the program is a device-only timeline.

Source: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a,
models/deepseek_v4/attention.py:361-602, 895-965 and
utils/multi_stream_utils.py:16-132 (stream graph S1).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Collection, Iterable, Mapping

from frontier.profiling.kernel_gap.schema import CUDA_GRAPH, CUDA_GRAPH_PIECEWISE, EAGER

MAIN, AUX0, AUX1, AUX2 = "main", "aux0", "aux1", "aux2"
STREAMS = (MAIN, AUX0, AUX1, AUX2)

# Main-stream scopes of the phase before and after the layer's stream program.
_PREFIX = (
    "dsv4_embedding", "dsv4_embedding_allreduce", "dsv4_mhc_first_pre",
    "dsv4_mhc_ffn_post_attn_pre", "dsv4_sp_attention_allgather",
)
_SUFFIX = (
    "dsv4_output_inv_rope_quant", "dsv4_grouped_output_projection",
    "dsv4_output_projection", "dsv4_sp_attention_reduce_scatter",
)
# `project_query_and_cache_kv`; without attention metadata the insert is a padding.
_QUERY = ("dsv4_q_projection", "dsv4_fused_q_kv_prepare", "dsv4_query_padding")
_COMPRESSOR = ("dsv4_compressor_save_partial_states", "dsv4_compressor_fused_compress_store")
_INDEX_QUERY = ("dsv4_index_query_projection", "dsv4_index_fused_query_prepare")
_INDEX_COMPRESSOR = (
    "dsv4_index_compressor_save_partial_states", "dsv4_index_compressor_fused_compress_store",
)
# `_sparse_indexer_and_attn`: an eager break of a PIECEWISE capture.
_SPARSE = (
    "dsv4_index_scores", "dsv4_index_topk", "dsv4_sparse_attention",
    "dsv4_attention_output_zero",
)
# The auxiliary input projections of `_run_parallel_input_projections`, in list order.
_OUTER_AUX = {
    "uncompressed": (),
    "hca": ((AUX0, "dsv4_compressor_projection"),),
    "csa": (
        (AUX0, "dsv4_compressor_projection"),
        (AUX1, "dsv4_index_weights_projection"),
        (AUX2, "dsv4_index_compressor_projection"),
    ),
}


@dataclass(frozen=True)
class AttentionSchedule:
    """Composed device elapsed of one attention phase and each scope's start on it."""

    elapsed_ms: float
    start_ms: Mapping[str, float]


def _run(stream, names):
    return ("run", stream, tuple(names))


def _wait(waiter, signaler):
    return ("wait", waiter, signaler)


def _fork(default, aux, *, current=MAIN):
    """`execute_in_parallel` / `maybe_execute_in_parallel` when the fork is taken."""
    program = [_wait(stream, current) for stream, _ in aux]
    program += [_run(stream, names) for stream, names in aux]
    program.append(_run(current, default))
    return program + [_wait(current, stream) for stream, _ in aux]


def _serial(default, aux, *, current=MAIN):
    """The same helpers run serially: the default function first, then the others."""
    return [_run(current, default)] + [_run(current, names) for _, names in aux]


def _layer_program(variant, mode, multi_stream_gemm, indexer_short_context):
    outer = [(stream, (name,)) for stream, name in _OUTER_AUX[variant]]
    program = [_run(MAIN, _PREFIX)]
    program += (_fork if multi_stream_gemm else _serial)(("dsv4_fused_qkv_projection",), outer)
    program.append(_run(MAIN, ("dsv4_fused_qkv_rmsnorm",)))
    if variant == "csa":
        if indexer_short_context:
            indexer = [_run(AUX0, _INDEX_COMPRESSOR + ("dsv4_index_short_context_fill",))]
        else:
            # `maybe_execute_in_parallel` runs serially under a Breakable capture.
            nested = _serial if mode == "PIECEWISE" else _fork
            indexer = nested(_INDEX_QUERY, [(AUX2, _INDEX_COMPRESSOR)], current=AUX0)
        # `execute_in_parallel` has no Breakable check: it forks in every mode.
        program += [_wait(AUX0, MAIN), _wait(AUX1, MAIN), *indexer,
                    _run(AUX1, _COMPRESSOR), _run(MAIN, _QUERY),
                    _wait(MAIN, AUX0), _wait(MAIN, AUX1)]
    elif variant == "hca":
        program += (_serial if mode == "PIECEWISE" else _fork)(_QUERY, [(AUX0, _COMPRESSOR)])
    else:
        program.append(_run(MAIN, _QUERY))
    return program + [_run(MAIN, _SPARSE), _run(MAIN, _SUFFIX)]


def segment_class(mode, name):
    if mode == "FULL":
        return CUDA_GRAPH
    if mode == "PIECEWISE":
        return EAGER if name in _SPARSE else CUDA_GRAPH_PIECEWISE
    if mode == "NONE":
        return EAGER
    raise ValueError(f"V4 graph runtime mode has no segment classes: {mode}")


def _scope_frontiers(program, present):
    frontier = {stream: frozenset() for stream in STREAMS}
    predecessors = {}
    for step, stream, operand in program:
        if step == "wait":
            frontier[stream] |= frontier[operand]
            continue
        for name in operand:
            if name not in present:
                continue
            if name in predecessors:
                raise ValueError(f"V4 attention scope is placed twice: {name}")
            predecessors[name] = frontier[stream]
            frontier[stream] = frozenset({name})
    unplaced = set(present) - set(predecessors)
    if unplaced:
        raise ValueError(f"V4 attention scopes have no stream position: {sorted(unplaced)}")
    return predecessors, frontier[MAIN]


def attention_scope_predecessors(
    present: Collection[str], *, variant: str, graph_runtime_mode: str, rows: int,
    multi_stream_gemm_token_threshold: int, indexer_short_context: bool,
) -> Mapping[str, frozenset[str]]:
    """Immediate scope frontier at each position of this call's stream program.

    Absent scopes pass their stream frontier through. A wait unions the two
    frontiers, preserving all device dependencies even if a scope has no cost.
    """
    program = _layer_program(
        variant, graph_runtime_mode, rows <= multi_stream_gemm_token_threshold,
        indexer_short_context,
    )
    predecessors, _ = _scope_frontiers(program, present)
    return MappingProxyType(predecessors)


def _schedule(program, weight_ms):
    predecessors, final_main = _scope_frontiers(program, weight_ms)
    start = {}
    for name, previous in predecessors.items():
        start[name] = max((start[item] + weight_ms[item] for item in previous), default=0.0)
    elapsed = max((start[name] + weight_ms[name] for name in final_main), default=0.0)
    return AttentionSchedule(elapsed_ms=elapsed, start_ms=MappingProxyType(start))


def compose_attention_phase(
    scope_ms: Mapping[str, float], kernel_counts: Mapping[str, float],
    gap_ms: Mapping[str, float], *, variant: str, graph_runtime_mode: str, rows: int,
    multi_stream_gemm_token_threshold: int, indexer_short_context: bool,
) -> AttentionSchedule:
    """Compose one layer's attention phase on the stream program of its call.

    ``rows`` is the layer's local row count; the outer input projections fork
    when it is at most the recorded multi-stream GEMM threshold. Each scope
    weighs its cost plus its kernel count times the gap of its segment class.
    """
    if set(kernel_counts) != set(scope_ms):
        raise ValueError("V4 attention composition requires one kernel count per scope")
    weight_ms = {}
    for name, cost_ms in scope_ms.items():
        segment = segment_class(graph_runtime_mode, name)
        if segment not in gap_ms:
            raise ValueError(f"V4 kernel gap value is missing: {segment}")
        weight_ms[name] = cost_ms + kernel_counts[name] * gap_ms[segment]
    program = _layer_program(
        variant, graph_runtime_mode, rows <= multi_stream_gemm_token_threshold,
        indexer_short_context,
    )
    return _schedule(program, weight_ms)


def serial_attention_phase(scopes: Iterable[tuple[str, float]]) -> AttentionSchedule:
    """Place the scopes back to back on one stream, without gaps.

    This is the schedule of the predictors that carry no stream program
    inputs: dummy mode and the diagnostic step-cost table.
    """
    weight_ms = dict(scopes)
    return _schedule([_run(MAIN, weight_ms)], weight_ms)
