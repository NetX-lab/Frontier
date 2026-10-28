"""Measure V4 operator kernels through capture ownership and graph replay.

GraphCaptureRecorder assigns captured nodes to the current operator scope. CUPTI
clone callbacks connect those nodes to replay activities inside a measured step
range; eager kernels use the innermost annotated scope range on the launch thread.
Declared containers must own no kernel. The scope tree groups input projections,
query and KV preparation, two compressors, index preparation and top-k, sparse
attention, and output projections under each attention layer. Top-k owns its own
kernels outside its scores child. Every hook checks its declared parent, including
calls on auxiliary streams, which execute in the same Python thread.

Padded steps execute attention over the synchronized dispatched rows, without
sequence parallelism on this TP1 engine. A dummy beside a NONE peer executes
its local one token. Returning no DP-width tensor is safe for the reduced dense
MLP. Host synchronization inside execute_model and full launch queues are not
excluded by construction: calibrated hold slack detects late launches. Sampling
kernels and output copies are inside the step range as unowned eager preparation.

The worker re-captures after installing the hooks and stops the clone subscriber
before starting the profiler. This module imports GPU packages only in worker
functions, so plans and saved traces remain usable on a CPU host.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections import Counter
from functools import partial, wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from frontier.profiling.attention import dsa_vllm_runner as case_runner
from frontier.profiling.attention import dsv4_attention_runner as attention_runner
from frontier.profiling.attention.dsa_plan import (
    GRAPH_REPLAY_CUDAGRAPH_MODES,
    PIECEWISE_CUDAGRAPH_MODES,
)
from frontier.profiling.attention.dsv4_attention_plan import (
    LAYER_SCOPE,
    Dsv4ProfilePlan,
    metadata_key,
)
from frontier.profiling.attention.graph_capture import (
    CaptureRecord,
    CudaGraphQueries,
    CuptiCloneRecorder,
    GraphCaptureRecorder,
    ScopeOwner,
)
from frontier.profiling.attention.graph_replay_trace import hold_annotation_name, step_annotation_name
from frontier.profiling.common.runtime_identity import collect_driver_version

CUDAGRAPH_DISPATCH_KEY = "cudagraph_dispatch"
NUM_TOKENS_ACROSS_DP = "num_tokens_across_dp"
STEP_HOLD_US = "step_hold_us"
HOLD_CYCLES_PER_US = "hold_cycles_per_us"
HOLD_CALIBRATION_CYCLES = 100_000_000
_LOCAL_DP_SIZE, _EMULATED_DP_SIZE, _LOCAL_DP_RANK, _PEER_DP_RANK = 1, 2, 0, 1
_DP_SIZE_PARAMETER, _DP_RANK_PARAMETER = "dp_size", "dp_rank"
_DP_PEER_ROWS = ("num_tokens", "cg_mode", "uniform_token_count", "max_query_len")
_DP_UTILS_MODULE = "vllm.v1.worker.gpu.dp_utils"
_COMPILATION_MODULE = "vllm.config.compilation"
_MS_TO_US = 1000.0
DRIVER_VERSION_KEY = "driver_version"
TARGET_TENSOR_PARALLEL_SIZE_KEY = "target_tensor_parallel_size"
BREAKABLE_CUDAGRAPH_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"
V2_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_ENV_ENABLED = "1"
FULL_AND_PIECEWISE_MODE, PIECEWISE_MODE, NONE_MODE = GRAPH_REPLAY_CUDAGRAPH_MODES
_GRAPH_CAPTURE_KIND = "graph"
_BREAKABLE_CAPTURE_KIND = "breakable_segment"
_KERNEL_NODE_TYPE = "kernel"

_PROJECTIONS = "dsv4_parallel_input_projections"
_FUSED_QKV_PROJECTION = "dsv4_fused_qkv_projection"
_COMPRESSOR_PROJECTION = "dsv4_compressor_projection"
_INDEX_WEIGHTS_PROJECTION = "dsv4_index_weights_projection"
_INDEX_COMPRESSOR_PROJECTION = "dsv4_index_compressor_projection"
_FUSED_QKV_RMSNORM = "dsv4_fused_qkv_rmsnorm"
_QUERY_PROJECTION = "dsv4_q_projection"
_FUSED_Q_KV_PREPARE = "dsv4_fused_q_kv_prepare"
_COMPRESSOR = "dsv4_compressor"
_INDEX_PREPARE = "dsv4_index_prepare"
_INDEX_COMPRESSOR = "dsv4_index_compressor"
_COMPRESSOR_SAVE = "dsv4_compressor_save_partial_states"
_COMPRESSOR_STORE = "dsv4_compressor_fused_compress_store"
_INDEX_COMPRESSOR_SAVE = "dsv4_index_compressor_save_partial_states"
_INDEX_COMPRESSOR_STORE = "dsv4_index_compressor_fused_compress_store"
_INDEX_QUERY_PROJECTION = "dsv4_index_query_projection"
_INDEX_FUSED_QUERY_PREPARE = "dsv4_index_fused_query_prepare"
_INDEX_SHORT_CONTEXT_FILL = "dsv4_index_short_context_fill"
_INDEX_TOPK = "dsv4_index_topk"
_INDEX_SCORES = "dsv4_index_scores"
_SPARSE_ATTENTION = "dsv4_sparse_attention"
_OUTPUT = "dsv4_o_proj"
_OUTPUT_INV_ROPE_QUANT = "dsv4_output_inv_rope_quant"
_GROUPED_OUTPUT_PROJECTION = "dsv4_grouped_output_projection"
_OUTPUT_PROJECTION = "dsv4_output_projection"

SCOPE_TREE = (
    (LAYER_SCOPE, None),
    (_PROJECTIONS, LAYER_SCOPE),
    (_FUSED_QKV_PROJECTION, _PROJECTIONS),
    (_COMPRESSOR_PROJECTION, _PROJECTIONS),
    (_INDEX_WEIGHTS_PROJECTION, _PROJECTIONS),
    (_INDEX_COMPRESSOR_PROJECTION, _PROJECTIONS),
    (_FUSED_QKV_RMSNORM, LAYER_SCOPE),
    (_QUERY_PROJECTION, LAYER_SCOPE),
    (_FUSED_Q_KV_PREPARE, LAYER_SCOPE),
    (_COMPRESSOR, LAYER_SCOPE),
    (_INDEX_PREPARE, LAYER_SCOPE),
    (_INDEX_COMPRESSOR, _INDEX_PREPARE),
    (_COMPRESSOR_SAVE, _COMPRESSOR),
    (_COMPRESSOR_STORE, _COMPRESSOR),
    (_INDEX_COMPRESSOR_SAVE, _INDEX_COMPRESSOR),
    (_INDEX_COMPRESSOR_STORE, _INDEX_COMPRESSOR),
    (_INDEX_QUERY_PROJECTION, _INDEX_PREPARE),
    (_INDEX_FUSED_QUERY_PREPARE, _INDEX_PREPARE),
    (_INDEX_SHORT_CONTEXT_FILL, _INDEX_PREPARE),
    (_INDEX_TOPK, LAYER_SCOPE),
    (_INDEX_SCORES, _INDEX_TOPK),
    (_SPARSE_ATTENTION, LAYER_SCOPE),
    (_OUTPUT, LAYER_SCOPE),
    (_OUTPUT_INV_ROPE_QUANT, _OUTPUT),
    (_GROUPED_OUTPUT_PROJECTION, _OUTPUT),
    (_OUTPUT_PROJECTION, _OUTPUT),
)
_SCOPE_PARENTS = dict(SCOPE_TREE)
CONTAINER_SCOPES = frozenset({
    LAYER_SCOPE, _PROJECTIONS, _INDEX_PREPARE, _COMPRESSOR, _INDEX_COMPRESSOR, _OUTPUT,
})
_AUX_PROJECTION_SCOPES = (
    _COMPRESSOR_PROJECTION, _INDEX_WEIGHTS_PROJECTION, _INDEX_COMPRESSOR_PROJECTION,
)
_COMPRESSOR_CHILDREN = {
    _COMPRESSOR: (_COMPRESSOR_SAVE, _COMPRESSOR_STORE),
    _INDEX_COMPRESSOR: (_INDEX_COMPRESSOR_SAVE, _INDEX_COMPRESSOR_STORE),
}
_ATTENTION_MODULE = "vllm.models.deepseek_v4.attention"
_COMPRESSOR_MODULE = "vllm.models.deepseek_v4.compressor"
_COMPRESS_STORE_MODULE = "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl"
_INDEXER_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
_CUSTOM_OPS_MODULE = "vllm._custom_ops"
_OUTPUT_MODULE = "vllm.models.deepseek_v4.nvidia.ops.o_proj"
_MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
_INSTANCE_SCOPES = (
    ("forward", LAYER_SCOPE),
    ("_run_parallel_input_projections", _PROJECTIONS),
    ("_fused_wqa_wkv_gemm", _FUSED_QKV_PROJECTION),
    ("_fused_qnorm_rope_kv_insert", _FUSED_Q_KV_PREPARE),
    ("forward_mqa", _SPARSE_ATTENTION),
    ("_o_proj", _OUTPUT),
)
_MODULE_SCOPES = (
    (_ATTENTION_MODULE, "fused_q_kv_rmsnorm", _FUSED_QKV_RMSNORM),
    (_ATTENTION_MODULE, "fused_indexer_q_rope_quant", _INDEX_FUSED_QUERY_PREPARE),
    (_INDEXER_MODULE, "fp8_fp4_mqa_logits", _INDEX_SCORES),
    (_INDEXER_MODULE, "fp8_fp4_paged_mqa_logits", _INDEX_SCORES),
    (_CUSTOM_OPS_MODULE, "cp_gather_indexer_k_quant_cache", _INDEX_SCORES),
    (_OUTPUT_MODULE, "fused_inv_rope_fp8_quant", _OUTPUT_INV_ROPE_QUANT),
    (_OUTPUT_MODULE, "fp8_einsum", _GROUPED_OUTPUT_PROJECTION),
)
_SAVE_PARTIAL_STATES_CHILD = 0
_FUSED_COMPRESS_STORE_CHILD = 1
_COMPRESSOR_FUNCTIONS = (
    (_COMPRESSOR_MODULE, "save_partial_states", _SAVE_PARTIAL_STATES_CHILD),
    (_COMPRESSOR_MODULE, "compress_norm_rope_store_triton", _FUSED_COMPRESS_STORE_CHILD),
    (_COMPRESSOR_MODULE, "compress_norm_rope_store_two_stage_triton", _FUSED_COMPRESS_STORE_CHILD),
    (_COMPRESS_STORE_MODULE, "compress_norm_rope_store_cutedsl", _FUSED_COMPRESS_STORE_CHILD),
)
_RECORDER: GraphCaptureRecorder | None = None


def _require_recorder() -> GraphCaptureRecorder:
    if _RECORDER is None:
        raise RuntimeError("DeepSeek-V4 graph replay hooks are not installed on this worker")
    return _RECORDER


def _scope_call(layer: int, scope: str, fn: Callable[..., Any], args, kwargs) -> Any:
    recorder = _require_recorder()
    parent_scope = _SCOPE_PARENTS[scope]
    expected = None if parent_scope is None else (layer, parent_scope)
    actual = recorder.current_owner
    if actual != expected:
        raise ValueError(f"Scope {scope!r} expected parent {expected!r}, actual owner {actual!r}")
    with recorder.scope((layer, scope)):
        return case_runner._require_state().timed_call(layer, scope, fn, args, kwargs)


def _current_layer(scope: str) -> int:
    owner = _require_recorder().current_owner
    if owner is None:
        raise RuntimeError(f"Scope {scope!r} ran outside a hooked DeepSeek-V4 attention layer")
    return owner[0]


def _wrap_scope(owner: Any, attribute: str, scope: str, layer: int | None = None) -> None:
    original = getattr(owner, attribute)

    @wraps(original)
    def call(*args, **kwargs):
        layer_id = _current_layer(scope) if layer is None else layer
        return _scope_call(layer_id, scope, original, args, kwargs)

    setattr(owner, attribute, call)


def _wrap_compressor_function(module: Any, attribute: str, child_index: int) -> None:
    original = getattr(module, attribute)

    @wraps(original)
    def call(*args, **kwargs):
        layer = _current_layer(attribute)
        parent = _require_recorder().current_owner[1]
        if parent not in _COMPRESSOR_CHILDREN:
            raise ValueError(
                f"Compressor function {attribute!r} expected parent in "
                f"{tuple(_COMPRESSOR_CHILDREN)}, actual owner {_require_recorder().current_owner!r}"
            )
        scope = _COMPRESSOR_CHILDREN[parent][child_index]
        return _scope_call(layer, scope, original, args, kwargs)

    setattr(module, attribute, call)


def _wrap_parallel_projections(module: Any) -> None:
    original = module.execute_in_parallel

    @wraps(original)
    def execute_in_parallel(default_fn, aux_fns, start_event, done_events, aux_streams=None, enable=False):
        owner = _require_recorder().current_owner
        if owner is not None and owner[1] == _PROJECTIONS:
            aux_fns = [
                None if fn is None else partial(_scope_call, owner[0], scope, fn, (), {})
                for fn, scope in zip(aux_fns, _AUX_PROJECTION_SCOPES, strict=True)
            ]
        return original(default_fn, aux_fns, start_event, done_events, aux_streams, enable=enable)

    module.execute_in_parallel = execute_in_parallel


class _ShortContextFill:
    """Preserve Triton's grid selection and annotate the resulting launch."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def __getitem__(self, grid):
        original = self.kernel[grid]

        def launch(*args, **kwargs):
            return _scope_call(
                _current_layer(_INDEX_SHORT_CONTEXT_FILL), _INDEX_SHORT_CONTEXT_FILL,
                original, args, kwargs,
            )

        return launch


def _eager_scope_runner(owner: ScopeOwner, fn: Callable[[], Any]) -> Any:
    # Registration owners are checked once the complete capture is available.
    with _require_recorder().scope(owner):
        return case_runner._require_state().timed_call(owner[0], owner[1], fn, (), {})


def _wrap_step(runner: Any, discovered: attention_runner.AttentionLayers) -> None:
    original = runner.execute_model

    @wraps(original)
    def execute_model(*args, **kwargs):
        state = case_runner._require_state()
        active = state.active
        if active is None:
            return original(*args, **kwargs)
        state.open_step_range(step_annotation_name(active.step_index))
        if state.hold_cycles:
            with state.torch.profiler.record_function(hold_annotation_name(active.step_index)):
                state.torch.cuda._sleep(state.hold_cycles)
        result = original(*args, **kwargs)
        if result is not None:
            raise RuntimeError("A recorded graph-replay execute_model must return None to close its range at sampling")
        metadata = runner.execute_model_state.attn_metadata
        if not isinstance(metadata, dict):
            raise ValueError("Micro-batched attention metadata is not supported by the DSA profiler")
        for layer in sorted(discovered.modules):
            variant = discovered.variants[layer]
            for source, prefix in discovered.prefixes[layer].items():
                state.capture_metadata(
                    metadata_key(variant, source),
                    lambda source=source, prefix=prefix: attention_runner._metadata_fields(
                        metadata[prefix], attention_runner.METADATA_FIELDS[source]
                    ),
                )
        return result

    runner.execute_model = execute_model


def _wrap_sample_tokens(runner: Any) -> None:
    original = runner.sample_tokens

    @wraps(original)
    def sample_tokens(*args, **kwargs):
        result = original(*args, **kwargs)
        case_runner._require_state().close_step_range()
        return result

    runner.sample_tokens = sample_tokens


class _DpPeerReduction:
    """Write only the peer column of the native CPU all-reduce tensor."""

    def __init__(self, peer, modes):
        self.values = (peer[_DP_PEER_ROWS[0]], modes[peer[_DP_PEER_ROWS[1]]].value,
                       peer[_DP_PEER_ROWS[2]] or 0, peer[_DP_PEER_ROWS[3]] or -1)

    def all_reduce(self, tensor, group=None):
        for row, value in enumerate(self.values):
            tensor[row][_PEER_DP_RANK] = value


def _wrap_dispatch(module: Any) -> None:
    original = module.dispatch_cg_and_sync_dp
    signature = inspect.signature(original)

    @wraps(original)
    def dispatch(*args, **kwargs):
        state = case_runner._require_state()
        peer = state.matched_dp_peer
        synced_tokens = None
        if peer is None:
            result = original(*args, **kwargs)
        else:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments[_DP_SIZE_PARAMETER] != _LOCAL_DP_SIZE:
                raise ValueError("DP-peer emulation requires bound dp_size == 1")
            dp_utils = importlib.import_module(_DP_UTILS_MODULE)
            modes = importlib.import_module(_COMPILATION_MODULE).CUDAGraphMode
            old_group, old_dist = dp_utils.get_dp_group, dp_utils.dist
            try:
                dp_utils.get_dp_group = lambda: SimpleNamespace(cpu_group=None)
                dp_utils.dist = _DpPeerReduction(peer, modes)
                bound.arguments[_DP_SIZE_PARAMETER] = _EMULATED_DP_SIZE
                bound.arguments[_DP_RANK_PARAMETER] = _LOCAL_DP_RANK
                desc, tensor = original(*bound.args, **bound.kwargs)
                synced_tokens = [int(value) for value in tensor.tolist()]
                result = desc, None
            finally:
                dp_utils.get_dp_group, dp_utils.dist = old_group, old_dist
        if state.active is not None:
            desc, _ = result
            state.capture_metadata(CUDAGRAPH_DISPATCH_KEY, lambda: {
                "cg_mode": desc.cg_mode.name,
                "num_tokens": desc.num_tokens,
                "num_reqs": desc.num_reqs,
                NUM_TOKENS_ACROSS_DP: synced_tokens,
            })
        return result

    module.dispatch_cg_and_sync_dp = dispatch


def _calibrate_hold(torch, hold_us: float) -> tuple[int, float | None]:
    if hold_us == 0:
        return 0, None
    torch.cuda._sleep(HOLD_CALIBRATION_CYCLES)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(HOLD_CALIBRATION_CYCLES)
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * _MS_TO_US
    if elapsed_us <= 0:
        raise RuntimeError(f"Hold calibration elapsed time must be positive, got {elapsed_us} us")
    cycles_per_us = HOLD_CALIBRATION_CYCLES / elapsed_us
    return round(hold_us * cycles_per_us), cycles_per_us


def install_graph_replay_hooks(
    worker: Any,
    engine_index: int,
    layers: list[tuple[int, str, int]],
    plan: Dsv4ProfilePlan,
    config=None,
) -> dict[str, Any]:
    """Install the capture scope tree and record metadata outside model replay."""

    global _RECORDER
    if not worker.vllm_config.use_v2_model_runner:
        raise ValueError("DeepSeek-V4 graph replay requires use_v2_model_runner=True")
    discovered = attention_runner.discover_attention_layers(worker, layers)
    _RECORDER = GraphCaptureRecorder(CudaGraphQueries(), _eager_scope_runner)
    _wrap_step(worker.model_runner, discovered)
    _wrap_sample_tokens(worker.model_runner)
    _wrap_dispatch(importlib.import_module(_MODEL_RUNNER_MODULE))
    case_runner.install_case_recorder(
        worker, engine_index, plan.shape.step_metadata_keys() + (CUDAGRAPH_DISPATCH_KEY,)
    )
    state = case_runner._require_state()
    state.hold_cycles, cycles_per_us = _calibrate_hold(state.torch, plan.limits.step_hold_us)
    layer_inventory = []
    for layer in sorted(discovered.modules):
        module = discovered.modules[layer]
        for attribute, scope in _INSTANCE_SCOPES:
            _wrap_scope(module, attribute, scope, layer)
        _wrap_scope(module.wq_b, "forward", _QUERY_PROJECTION, layer)
        _wrap_scope(module.wo_b, "forward", _OUTPUT_PROJECTION, layer)
        if module.compressor is not None:
            _wrap_scope(module.compressor, "forward", _COMPRESSOR, layer)
        if module.indexer is not None:
            _wrap_scope(module.indexer, "forward", _INDEX_PREPARE, layer)
            _wrap_scope(module.indexer.compressor, "forward", _INDEX_COMPRESSOR, layer)
            _wrap_scope(module.indexer.wq_b, "forward", _INDEX_QUERY_PROJECTION, layer)
            _wrap_scope(module.indexer.indexer_op, "forward", _INDEX_TOPK, layer)
        layer_inventory.append({
            "global_layer_id": layer,
            "attention_variant": discovered.variants[layer],
            "attention_prefix": module.prefix,
            "compress_ratio": module.compress_ratio,
            "metadata_prefixes": discovered.prefixes[layer],
        })
    for module_name, attribute, scope in _MODULE_SCOPES:
        _wrap_scope(importlib.import_module(module_name), attribute, scope)
    for module_name, attribute, child_index in _COMPRESSOR_FUNCTIONS:
        _wrap_compressor_function(importlib.import_module(module_name), attribute, child_index)
    attention_module = importlib.import_module(_ATTENTION_MODULE)
    _wrap_parallel_projections(attention_module)
    attention_module._fill_short_context_topk_indices = _ShortContextFill(
        attention_module._fill_short_context_topk_indices
    )
    return dict(attention_runner.attention_inventory(discovered.modules, layer_inventory, worker, config),
                **{STEP_HOLD_US: plan.limits.step_hold_us, HOLD_CYCLES_PER_US: cycles_per_us})


def _capture_result(record: CaptureRecord, mode: str, sizes: tuple[int, ...]) -> dict[str, Any]:
    kinds = Counter(capture.kind for capture in record.captures)
    return {"record": record.to_json(), "summary": {
        "cudagraph_mode": mode,
        "cudagraph_capture_sizes": list(sizes),
        "captures_by_kind": {
            _GRAPH_CAPTURE_KIND: kinds[_GRAPH_CAPTURE_KIND],
            _BREAKABLE_CAPTURE_KIND: kinds[_BREAKABLE_CAPTURE_KIND],
        },
        "node_count": len(record.nodes),
        "kernel_node_count": sum(node.node_type == _KERNEL_NODE_TYPE for node in record.nodes),
        "clone_count": len(record.clones),
        "eager_break_count": len(record.eager_breaks),
        "eager_break_owner_scopes": sorted({entry.owner[1] for entry in record.eager_breaks}),
    }}


def recapture_graphs(worker: Any, cudagraph_mode: str, capture_sizes: tuple[int, ...]) -> dict[str, Any]:
    """Require the planned graph configuration and re-capture with ownership hooks."""

    runner = worker.model_runner
    cm = runner.cudagraph_manager
    actual_mode = cm.cudagraph_mode.name
    if actual_mode != cudagraph_mode:
        raise ValueError(f"Resolved cudagraph mode {actual_mode!r} differs from plan {cudagraph_mode!r}")
    actual_sizes = sorted(runner.compilation_config.cudagraph_capture_sizes)
    if actual_sizes != list(capture_sizes):
        raise ValueError(f"Resolved cudagraph capture sizes {actual_sizes} differ from plan {list(capture_sizes)}")
    if cudagraph_mode in PIECEWISE_CUDAGRAPH_MODES:
        if not cm.use_breakable_cg or os.environ.get(BREAKABLE_CUDAGRAPH_ENV) != _ENV_ENABLED:
            raise ValueError(
                f"{cudagraph_mode} requires Breakable capture and {BREAKABLE_CUDAGRAPH_ENV}=1; "
                f"use_breakable_cg={cm.use_breakable_cg}, environment={os.environ.get(BREAKABLE_CUDAGRAPH_ENV)!r}"
            )
    if cudagraph_mode == NONE_MODE:
        return _capture_result(CaptureRecord(), cudagraph_mode, capture_sizes)

    import torch
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.platforms import current_platform

    recorder = _require_recorder()
    # PyTorch refuses to share a private memory pool once all of its graphs are
    # released (the use_count check of the device and host caching allocators),
    # so the re-capture uses a fresh pool, as vLLM does for discarded captures.
    cm.graphs.clear()
    cm.pool = current_platform.graph_pool_handle()
    if cm.breakable_cg_runner is not None:
        cm.breakable_cg_runner.clear_graphs()
        cm.breakable_cg_runner.graph_pool = cm.pool
    cm._graphs_captured = False
    with CuptiCloneRecorder() as clones, recorder.install(torch.cuda.CUDAGraph, BreakableCUDAGraphCapture):
        runner.capture_model()
    record = recorder.record(clones.clones)
    if cudagraph_mode in PIECEWISE_CUDAGRAPH_MODES:
        if not record.eager_breaks:
            raise ValueError(
                f"{cudagraph_mode} recorded zero eager breaks; {BREAKABLE_CUDAGRAPH_ENV} "
                "must be enabled before vLLM imports its capture decorators"
            )
        for entry in record.eager_breaks:
            if entry.owner is None or entry.owner[1] != LAYER_SCOPE:
                raise ValueError(f"Eager break expected owner scope {LAYER_SCOPE!r}, got {entry.owner!r}")
    if cudagraph_mode == FULL_AND_PIECEWISE_MODE and not any(
        capture.kind == _GRAPH_CAPTURE_KIND for capture in record.captures
    ):
        raise ValueError(f"{cudagraph_mode} recorded no FULL graph capture")
    return _capture_result(record, cudagraph_mode, capture_sizes)


def run_engine(
    *,
    plan: Dsv4ProfilePlan,
    engine_index: int,
    model_dir: str,
    vocab_size: int,
    seed: int,
    gpu_memory_utilization: float,
    expected_attention_backend: str,
    raw_path: Path,
    trace_dir: Path,
    keep_traces: bool,
    case_ids: set[str] | None = None,
) -> None:
    """Keep capture artifacts and measure one engine with replay trace attribution."""

    mode = plan.limits.cudagraph_mode
    sizes = plan.limits.cudagraph_capture_sizes
    if mode in PIECEWISE_CUDAGRAPH_MODES and os.environ.get(BREAKABLE_CUDAGRAPH_ENV) != _ENV_ENABLED:
        raise ValueError(
            f"{mode} requires {BREAKABLE_CUDAGRAPH_ENV}=1 before building the engine "
            "so vLLM enables its eager-break decorators before import"
        )
    llm = attention_runner.build_dsv4_engine(plan, engine_index, model_dir, seed, gpu_memory_utilization)
    layers = [(layer.global_layer_id, layer.attention_variant, layer.compress_ratio)
              for layer in plan.shape.layers]
    inventory = llm.collective_rpc(
        install_graph_replay_hooks, args=(engine_index, layers, plan, attention_runner.routing_config)
    )[0]
    capture = llm.collective_rpc(recapture_graphs, args=(mode, sizes))[0]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path = raw_path.parent / f"dsv4_capture_e{engine_index}.json"
    capture_path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    record = CaptureRecord.from_json(capture["record"])
    inventory["cudagraph_capture"] = capture["summary"]
    inventory[DRIVER_VERSION_KEY] = collect_driver_version()
    inventory[TARGET_TENSOR_PARALLEL_SIZE_KEY] = plan.shape.target_tensor_parallel_size
    case_runner.measure_engine(
        llm=llm, plan=plan, engine_index=engine_index, inventory=inventory,
        vocab_size=vocab_size, seed=seed, expected_attention_backend=expected_attention_backend,
        raw_path=raw_path, trace_dir=trace_dir, keep_traces=keep_traces, case_ids=case_ids,
        log_label="dsv4_graph",
        trace_calls=partial(case_runner.graph_replay_calls, record=record, container_scopes=CONTAINER_SCOPES),
    )


__all__ = [
    "BREAKABLE_CUDAGRAPH_ENV", "CONTAINER_SCOPES", "CUDAGRAPH_DISPATCH_KEY",
    "SCOPE_TREE", "V2_MODEL_RUNNER_ENV", "install_graph_replay_hooks", "recapture_graphs", "run_engine",
]
