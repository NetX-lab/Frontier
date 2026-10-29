"""vLLM engine driver for DSA attention profiling (GPU worker only).

One process builds one in-process vLLM engine. With
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the EngineCore and its single worker run
in this process, so every ``LLMEngine.step`` is one schedule plus one model
execution. Hooks installed through ``collective_rpc`` after engine start-up
wrap:

* ``model_runner.execute_model``: classifies each scheduled step and decides
  whether the active case records it;
* each ``Indexer.forward`` (scope ``attn_dsa_indexer``): index query and key
  projections, key norm, RoPE, FP8 quantization, index logits and top-k;
* each ``MLAAttention.forward`` (scope ``attn_dsa_mla``): KV-cache write,
  query absorption, the sparse attention kernel, and the value up-projection.

The two scopes are siblings inside ``MultiHeadLatentAttentionWrapper.forward``;
the q/kv down-projections, norms, RoPE, and ``o_proj`` stay outside both and
are linear-op work. Hooks are installed after engine start-up, so the memory
profiling and dummy runs are never recorded.

Graph pilots disable async scheduling and enable chunked prefill; native
serving uses async scheduling without chunked prefill. First-tail pilots need
the final one-token chunk.

The weights are vLLM dummy weights and routing, top-k selection, and metadata
are the engine's own; nothing in the measured path is replaced.

The step classification, case window, timing, and raw-record writing below are
model independent: ``install_case_recorder`` and ``measure_engine`` also drive
the DeepSeek-V4 attention producer, which installs its own module hooks.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from frontier.profiling.attention.dsa_case_prediction import requests_match_case
from frontier.profiling.attention.dsa_kernel_trace import (
    ScopeKey,
    ScopeKernelTime,
    aggregate_kernel_trace,
    annotation_name,
    load_trace_events,
)
from frontier.profiling.attention.dsa_plan import (
    ENGINE_POLICY_EAGER_CHUNKED,
    ENGINE_POLICY_GRAPH_REPLAY,
    ENGINE_POLICY_NATIVE_DEFAULTS,
    INDEXER_SCOPE,
    MLA_SCOPE,
    PHASE_DECODE,
    PHASE_PREFILL,
    PHASE_FIRST_TAIL,
    PHASE_DUMMY,
    SHARED_INDEX_VARIANT,
    DsaCase,
    DsaEngineSpec,
    DsaProfilePlan,
    require_scope_timing,
)
from frontier.profiling.attention.dsa_rows import CUDA_EVENT, KERNEL_ONLY, MEASUREMENT_TYPES
from frontier.profiling.attention.graph_capture import CaptureRecord
from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace
from frontier.profiling.common.runtime_identity import (
    collect_device_identity,
    collect_runtime_versions,
)
from frontier.profiling.model_runner.dsv4_runner_rows import forward_correlations

RECORD_KIND = "kind"
INTERVAL_RECORD = "interval"
ENGINE_INDEX = "engine_index"
CASE_ID = "case_id"
STEP_INDEX = "step_index"
GLOBAL_LAYER_ID = "global_layer_id"
SCOPE = "scope"
LAUNCH_KIND = "launch_kind"
ACTIVITY = "activity"
NAME = "name"
START_US = "start_us"
END_US = "end_us"
STREAM = "stream"
GRAPH_ID = "graph_id"
GRAPH_NODE_ID = "graph_node_id"
ORIGINAL_NODE_ID = "original_node_id"
CAPTURE = "capture"
CAPTURE_KIND = "capture_kind"
CORRELATION = "correlation"
LAUNCH_END_US = "launch_end_us"
GRAPH_GAP_US = "graph_gap_us"
GRID, BLOCK = "grid", "block"
DUMMY_RUN_PARAMETER = "dummy_run"
_CASE_STEP_MULTIPLIER = 4
_CASE_STEP_ALLOWANCE = 64

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MLA_METADATA_FIELDS = (
    "num_reqs",
    "max_query_len",
    "max_seq_len",
    "num_actual_tokens",
    "block_size",
    "topk_tokens",
    "num_decodes",
    "num_prefills",
    "num_decode_tokens",
    "prefill_max_seq_len",
    "use_dense_mha",
)
_INDEXER_METADATA_FIELDS = (
    "max_seq_len",
    "num_decodes",
    "num_decode_tokens",
    "num_prefills",
    "num_prefill_tokens",
)
# Step record keys of the native metadata each DSA step must capture.
DSA_STEP_METADATA = ("mla_metadata", "indexer_metadata")
# CUPTI can drop the first kernel of a fresh profiler session; these tiny
# kernels run outside every DSA scope, as in RecordFunctionTracer.
_PROFILER_PRIME_KERNELS = 4


@dataclass
class _CaseRecorder:
    case_id: str
    phase: str
    batch_size: int
    skip: int
    limit: int
    measurement: str
    dp_peer: dict[str, Any] | None = None
    prompt_len: int = 0
    prompt_lens: tuple[int, ...] = ()
    tail_count: int = 1
    matches: int = 0
    recorded: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ActiveStep:
    step_index: int
    record: dict[str, Any]
    events: list[tuple[int, str, Any, Any]] = field(default_factory=list)


class _WorkerState:
    def __init__(
        self,
        engine_index: int,
        torch_module: Any,
        step_metadata: tuple[str, ...] = DSA_STEP_METADATA,
    ) -> None:
        self.engine_index = engine_index
        self.torch = torch_module
        self.step_metadata = step_metadata
        self.prompt_lens: dict[str, int] = {}
        self.case: _CaseRecorder | None = None
        self.active: _ActiveStep | None = None
        self.next_step_index = 0
        self.profiler: Any = None
        self.matched_dp_peer: dict[str, Any] | None = None
        self.step_range: Any = None
        self.hold_cycles = 0

    def _scheduled_requests(self, scheduler_output: Any) -> list[dict[str, int]]:
        for req_id in scheduler_output.finished_req_ids:
            self.prompt_lens.pop(req_id, None)
        scheduled = scheduler_output.num_scheduled_tokens
        entries: list[tuple[str, int]] = []
        for new_request in scheduler_output.scheduled_new_reqs:
            if new_request.prompt_token_ids is None:
                raise ValueError("DSA profiling prompts must carry token ids")
            self.prompt_lens[new_request.req_id] = len(new_request.prompt_token_ids)
            entries.append((new_request.req_id, new_request.num_computed_tokens))
        cached = scheduler_output.scheduled_cached_reqs
        entries.extend(zip(cached.req_ids, cached.num_computed_tokens))
        return [
            {
                "num_scheduled_tokens": scheduled[req_id],
                "num_computed_tokens": computed,
                "prompt_len": self.prompt_lens[req_id],
            }
            for req_id, computed in entries
        ]

    def begin_step(self, scheduler_output: Any, dummy_run: bool = False) -> _ActiveStep | None:
        requests = [] if dummy_run else self._scheduled_requests(scheduler_output)
        case = self.case
        if case is None:
            return None
        if dummy_run:
            matches = case.phase == PHASE_DUMMY
        elif not requests or case.phase == PHASE_DUMMY:
            matches = False
        elif case.phase == PHASE_FIRST_TAIL or case.prompt_lens:
            matches = requests_match_case(case, requests)
        else:
            prefills = sum(r["num_computed_tokens"] < r["prompt_len"] for r in requests)
            if case.phase == PHASE_PREFILL:
                matches = len(requests) == case.batch_size and prefills == case.batch_size
            else:
                matches = prefills == 0 and len(requests) == case.batch_size
        if not matches:
            return None
        self.matched_dp_peer = case.dp_peer
        case.matches += 1
        if case.matches <= case.skip or case.recorded >= case.limit:
            return None
        step_index = self.next_step_index
        self.next_step_index += 1
        return _ActiveStep(
            step_index=step_index,
            record={
                "kind": "step",
                "measurement": case.measurement,
                "engine_index": self.engine_index,
                "case_id": case.case_id,
                "phase": case.phase,
                "step_index": step_index,
                "requests": requests,
                **dict.fromkeys(self.step_metadata),
            },
        )

    def capture_metadata(self, key: str, read: Callable[[], dict[str, Any]]) -> None:
        """Store ``read()`` under ``key`` once per recorded step."""

        active = self.active
        if active is not None and active.record[key] is None:
            active.record[key] = read()

    def end_step(self, active: _ActiveStep) -> None:
        case = self.case
        record = active.record
        missing = [key for key in self.step_metadata if record[key] is None]
        if missing:
            raise RuntimeError(
                f"Step {active.step_index} of {case.case_id} ran no hooked layer "
                f"providing {missing}; the attention metadata was not captured"
            )
        if case.measurement == CUDA_EVENT:
            self.torch.cuda.synchronize()
            for layer_id, scope, start, end in active.events:
                case.calls.append(
                    {
                        "kind": "call",
                        "measurement": CUDA_EVENT,
                        "engine_index": self.engine_index,
                        "case_id": case.case_id,
                        "step_index": active.step_index,
                        "global_layer_id": layer_id,
                        "scope": scope,
                        "time_ms": start.elapsed_time(end),
                        "kernel_count": None,
                    }
                )
        case.steps.append(record)
        case.recorded += 1

    def timed_call(
        self,
        layer_id: int,
        scope: str,
        function: Callable[..., Any],
        args,
        kwargs,
        annotate: bool = True,
    ) -> Any:
        """Time one scope call of the active step.

        ``annotate=False`` marks a scope that encloses other scopes: it gets
        CUDA-event wall time only, because kernel-only annotations must not
        nest.
        """

        active = self.active
        if active is None:
            return function(*args, **kwargs)
        if self.case.measurement == CUDA_EVENT:
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            active.events.append((layer_id, scope, start, end))
            return result
        if not annotate:
            return function(*args, **kwargs)
        with self.torch.profiler.record_function(
            annotation_name(active.step_index, layer_id, scope)
        ):
            return function(*args, **kwargs)

    def open_step_range(self, name: str) -> None:
        if self.step_range is not None:
            raise RuntimeError("A recorded step range is still open")
        self.step_range = self.torch.profiler.record_function(name)
        self.step_range.__enter__()

    def close_step_range(self) -> None:
        if self.step_range is not None:
            self.step_range.__exit__(None, None, None)
            self.step_range = None

    def start_profiler(self) -> None:
        torch = self.torch
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        self.profiler.__enter__()
        prime = torch.zeros((1,), device="cuda")
        for _ in range(_PROFILER_PRIME_KERNELS):
            prime = prime + 1
        torch.cuda.synchronize()

    def stop_profiler(self, trace_path: str) -> None:
        self.torch.cuda.synchronize()
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(trace_path)
        self.profiler = None


_STATE: _WorkerState | None = None


def _require_state() -> _WorkerState:
    if _STATE is None:
        raise RuntimeError("DSA hooks are not installed on this worker")
    return _STATE


def _layer_id(module_name: str) -> int:
    match = _LAYER_INDEX.search(module_name)
    if match is None:
        raise ValueError(f"Cannot read the decoder layer index from {module_name!r}")
    return int(match.group(1))


def _scalar_fields(metadata: Any, names: tuple[str, ...]) -> dict[str, int | bool]:
    values = {}
    for name in names:
        value = getattr(metadata, name, None)
        if type(value) in (int, bool):
            values[name] = value
    return values


def _layer_metadata(layer_name: str) -> Any:
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        raise ValueError(
            "Micro-batched attention metadata is not supported by the DSA profiler"
        )
    return attn_metadata[layer_name]


def _wrap_indexer(module: Any, layer_id: int) -> None:
    original = module.forward

    def forward(*args, **kwargs):
        return _require_state().timed_call(layer_id, INDEXER_SCOPE, original, args, kwargs)

    module.forward = forward


def _wrap_mla(module: Any, layer_id: int) -> None:
    original = module.forward
    layer_name = module.layer_name
    indexer_prefix = module.indexer.k_cache.prefix if module.indexer is not None else None

    def forward(*args, **kwargs):
        state = _require_state()
        state.capture_metadata(
            "mla_metadata",
            lambda: _scalar_fields(_layer_metadata(layer_name), _MLA_METADATA_FIELDS),
        )
        if indexer_prefix is not None:
            state.capture_metadata(
                "indexer_metadata",
                lambda: _scalar_fields(
                    _layer_metadata(indexer_prefix), _INDEXER_METADATA_FIELDS
                ),
            )
        return state.timed_call(layer_id, MLA_SCOPE, original, args, kwargs)

    module.forward = forward


def _wrap_execute_model(runner: Any) -> None:
    original = runner.execute_model

    signature = inspect.signature(original)

    @wraps(original)
    def execute_model(scheduler_output, *args, **kwargs):
        state = _require_state()
        bound = signature.bind(scheduler_output, *args, **kwargs)
        bound.apply_defaults()
        active = state.begin_step(scheduler_output, bound.arguments.get(DUMMY_RUN_PARAMETER, False))
        try:
            if active is None:
                return original(scheduler_output, *args, **kwargs)
            state.active = active
            output = original(scheduler_output, *args, **kwargs)
            state.end_step(active)
            return output
        finally:
            if active is not None:
                state.active = None
            state.matched_dp_peer = None

    runner.execute_model = execute_model


def install_case_recorder(
    worker: Any, engine_index: int, step_metadata: tuple[str, ...]
) -> None:
    """Create the worker's case state and classify every scheduled step.

    Each step record carries one entry per ``step_metadata`` key, which the
    model's module hooks fill through ``capture_metadata``.
    """

    global _STATE
    if _STATE is not None:
        raise RuntimeError("DSA hooks are already installed on this worker")
    import torch

    _wrap_execute_model(worker.model_runner)
    _STATE = _WorkerState(engine_index, torch, step_metadata)


def install_dsa_hooks(
    worker: Any, engine_index: int, layer_variants: dict[int, str]
) -> dict[str, Any]:
    """Wrap the measured modules; return the engine's attention inventory."""

    install_case_recorder(worker, engine_index, DSA_STEP_METADATA)
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import Indexer

    mla_modules: dict[int, Any] = {}
    indexer_layers: set[int] = set()
    skip_topk: dict[int, bool] = {}
    model = worker.get_model()
    for name, module in model.named_modules():
        if isinstance(module, MLAAttention):
            layer_id = _layer_id(name)
            if layer_id in mla_modules:
                raise ValueError(f"Layer {layer_id} has two MLAAttention modules")
            mla_modules[layer_id] = module
            _wrap_mla(module, layer_id)
        elif isinstance(module, Indexer):
            layer_id = _layer_id(name)
            indexer_layers.add(layer_id)
            _wrap_indexer(module, layer_id)
        elif hasattr(module, "mla_attn") and hasattr(module, "skip_topk"):
            skip_topk[_layer_id(name)] = bool(module.skip_topk)

    expected_shared = {
        layer for layer, variant in layer_variants.items() if variant == SHARED_INDEX_VARIANT
    }
    if set(mla_modules) != set(layer_variants):
        raise ValueError(
            f"Engine MLA layers {sorted(mla_modules)} differ from the plan "
            f"{sorted(layer_variants)}"
        )
    if indexer_layers != set(layer_variants) - expected_shared:
        raise ValueError(
            f"Engine indexer layers {sorted(indexer_layers)} differ from the plan's "
            f"full-index layers {sorted(set(layer_variants) - expected_shared)}"
        )
    if {layer for layer, skip in skip_topk.items() if skip} != expected_shared:
        raise ValueError(
            f"Engine skip_topk layers {skip_topk} differ from the plan's shared-index "
            f"layers {sorted(expected_shared)}"
        )
    backends = {module.attn_backend.get_name() for module in mla_modules.values()}
    impls = {type(module.impl).__name__ for module in mla_modules.values()}
    if len(backends) != 1 or len(impls) != 1:
        raise ValueError(f"Layers selected different backends {backends} / {impls}")

    return {
        "attention_backend": backends.pop(),
        "attention_impl": impls.pop(),
        "layers": [
            {
                "global_layer_id": layer_id,
                "mla_layer_name": mla_modules[layer_id].layer_name,
                "has_indexer": layer_id in indexer_layers,
                "skip_topk": skip_topk.get(layer_id),
            }
            for layer_id in sorted(mla_modules)
        ],
    }


def begin_case(
    worker: Any,
    case_id: str,
    phase: str,
    batch_size: int,
    skip: int,
    limit: int,
    measurement: str,
    dp_peer: dict[str, Any] | None = None,
    prompt_len: int = 0,
    prompt_lens: tuple[int, ...] = (),
    tail_count: int = 1,
) -> None:
    state = _require_state()
    if state.case is not None:
        raise RuntimeError(f"Case {state.case.case_id} is still active")
    state.case = _CaseRecorder(case_id, phase, batch_size, skip, limit, measurement,
                               dp_peer, prompt_len, prompt_lens, tail_count)
    if measurement == KERNEL_ONLY:
        state.start_profiler()


def case_progress(worker: Any) -> int:
    return _require_state().case.recorded


def finish_case(worker: Any, trace_path: str | None) -> dict[str, Any]:
    state = _require_state()
    case = state.case
    if state.step_range is not None:
        raise RuntimeError(f"Case {case.case_id} finished with an open step range")
    state.case = None
    if case.measurement == KERNEL_ONLY:
        state.stop_profiler(trace_path)
    return {"recorded": case.recorded, "steps": case.steps, "calls": case.calls}


def _random_prompt(rng: random.Random, length: int, vocab_size: int) -> dict[str, Any]:
    return {"prompt_token_ids": [rng.randrange(vocab_size) for _ in range(length)]}


def _case_window(case: DsaCase, engine: DsaEngineSpec) -> tuple[int, int]:
    """Return how many matching steps to skip and to record."""

    if case.phase == PHASE_PREFILL:
        query_budget = engine.prefill_query_len or engine.chunk_budget
        steps_per_prompt = math.ceil(case.request_prompt_lens[0] / query_budget)
        return case.warmup * steps_per_prompt, case.measured * steps_per_prompt
    if case.phase in (PHASE_DECODE, PHASE_FIRST_TAIL, PHASE_DUMMY):
        return case.warmup, case.measured
    raise ValueError(f"Unknown case phase {case.phase!r}")


def _step_until(llm_engine: Any, done: Callable[[], bool], max_steps: int, label: str) -> None:
    steps = 0
    while not done():
        if not llm_engine.has_unfinished_requests():
            raise RuntimeError(f"{label}: requests finished before the case completed")
        if steps >= max_steps:
            raise RuntimeError(f"{label}: no completion after {max_steps} engine steps")
        llm_engine.step()
        steps += 1


def _scope_calls(
    scope_times: dict[ScopeKey, ScopeKernelTime],
    steps: list[dict[str, Any]],
    engine_index: int,
    trace_path: Path,
) -> list[dict[str, Any]]:
    by_step = {step["step_index"]: step for step in steps}
    calls = []
    for key, timing in sorted(
        scope_times.items(),
        key=lambda item: (item[0].step_index, item[0].global_layer_id, item[0].scope),
    ):
        step = by_step.get(key.step_index)
        if step is None:
            raise ValueError(f"Trace {trace_path} has scopes of unrecorded step {key}")
        calls.append(
            {
                "kind": "call",
                "measurement": KERNEL_ONLY,
                "engine_index": engine_index,
                "case_id": step["case_id"],
                "step_index": key.step_index,
                "global_layer_id": key.global_layer_id,
                "scope": key.scope,
                "time_ms": timing.kernel_time_ms,
                "kernel_count": timing.kernel_count,
            }
        )
    return calls


def _kernel_only_calls(
    trace_path: Path, steps: list[dict[str, Any]], engine_index: int
) -> list[dict[str, Any]]:
    return _scope_calls(
        aggregate_kernel_trace(load_trace_events(trace_path)), steps, engine_index, trace_path
    )


def graph_replay_calls(
    trace_path: Path,
    steps: list[dict[str, Any]],
    engine_index: int,
    *,
    record: CaptureRecord,
    container_scopes: frozenset[str],
) -> list[dict[str, Any]]:
    """Build kernel-only call rows followed by all attributed replay intervals."""

    events = load_trace_events(trace_path)
    attributed = attribute_replay_trace(events, record, container_scopes)
    forward_launches = forward_correlations(events)
    calls = _scope_calls(
        {key: timing for key, timing in attributed.scope_times.items()
         if key.scope not in container_scopes},
        steps, engine_index, trace_path,
    )
    by_step = {step[STEP_INDEX]: step for step in steps}
    captures = {capture.ordinal: capture.kind for capture in record.captures}
    for interval in attributed.intervals:
        step = by_step.get(interval.step_index)
        if step is None:
            raise ValueError(f"Trace {trace_path} has intervals of unrecorded step {interval.step_index}")
        calls.append({
            **({"model_forward": interval.correlation in forward_launches}
               if forward_launches is not None else {}),
            RECORD_KIND: INTERVAL_RECORD, ENGINE_INDEX: engine_index, CASE_ID: step[CASE_ID],
            STEP_INDEX: interval.step_index,
            GLOBAL_LAYER_ID: None if interval.owner is None else interval.owner[0],
            SCOPE: None if interval.owner is None else interval.owner[1],
            LAUNCH_KIND: interval.kind, ACTIVITY: interval.activity, NAME: interval.name,
            START_US: interval.start_us, END_US: interval.end_us, STREAM: interval.stream,
            GRAPH_ID: interval.graph_id, GRAPH_NODE_ID: interval.graph_node_id,
            ORIGINAL_NODE_ID: interval.original_node_id, CAPTURE: interval.capture,
            CAPTURE_KIND: None if interval.capture is None else captures[interval.capture],
            CORRELATION: interval.correlation, LAUNCH_END_US: interval.launch_end_us,
            GRAPH_GAP_US: interval.graph_gap_us, GRID: interval.grid, BLOCK: interval.block,
        })
    return calls


_TIMING_MEASUREMENTS = {
    ENGINE_POLICY_EAGER_CHUNKED: MEASUREMENT_TYPES,
    ENGINE_POLICY_GRAPH_REPLAY: (KERNEL_ONLY,),
}


def timing_measurements(engine_policy: str) -> tuple[str, ...]:
    """Select timing that preserves the engine policy's captured graph."""

    if engine_policy not in _TIMING_MEASUREMENTS:
        require_scope_timing(engine_policy)
    return _TIMING_MEASUREMENTS[engine_policy]


def run_dummy_step(worker: Any) -> None:
    """Execute the native metadata-bearing dummy batch and close its step range."""
    worker.execute_dummy_batch()
    _require_state().close_step_range()


def run_case(
    llm: Any,
    plan: DsaProfilePlan,
    case: DsaCase,
    measurement: str,
    rng: random.Random,
    vocab_size: int,
    trace_dir: Path,
    keep_traces: bool,
    trace_calls: Callable[[Path, list[dict[str, Any]], int], list[dict[str, Any]]] = _kernel_only_calls,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    engine_spec = plan.engine(case.engine_index)
    skip, limit = _case_window(case, engine_spec)
    begin_args = (case.case_id, case.phase, case.batch_size, skip, limit, measurement)
    if case.dp_peer is not None or case.phase == PHASE_FIRST_TAIL or case.prompt_lens:
        begin_args += (None if case.dp_peer is None else asdict(case.dp_peer), case.prompt_len)
        if case.prompt_lens or case.tail_count != 1:
            begin_args += (case.prompt_lens, case.tail_count)
    llm_engine = llm.llm_engine
    max_steps = _CASE_STEP_MULTIPLIER * case.estimated_steps + _CASE_STEP_ALLOWANCE
    label = f"{case.case_id}/{measurement}"
    params = None if case.phase == PHASE_DUMMY else SamplingParams(
        max_tokens=case.max_tokens, temperature=0.0, ignore_eos=True, detokenize=False
    )
    if case.phase == PHASE_FIRST_TAIL:
        request_ids = [f"{label}/q{index}" for index in range(case.batch_size - case.tail_count)]
        for request_id, prompt_len in zip(request_ids, case.request_prompt_lens):
            llm_engine.add_request(request_id, _random_prompt(rng, prompt_len, vocab_size), params)
        emitted = set()
        steps = 0
        while len(emitted) < len(request_ids):
            if steps >= max_steps or not llm_engine.has_unfinished_requests():
                raise RuntimeError(f"{label}: decode requests did not emit an output within {max_steps} steps")
            for output in llm_engine.step():
                if output.request_id in request_ids and any(item.token_ids for item in output.outputs):
                    emitted.add(output.request_id)
            steps += 1
        llm.collective_rpc(begin_case, args=begin_args)
        tail_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False)
        for repeat in range(case.warmup + case.measured):
            tail_ids = [f"{label}/r{repeat}/q{len(request_ids) + index}"
                        for index in range(case.tail_count)]
            for tail_id, prompt_len in zip(tail_ids, case.request_prompt_lens[-case.tail_count:]):
                llm_engine.add_request(tail_id, _random_prompt(rng, prompt_len, vocab_size), tail_params)
            finished = set()
            steps = 0
            while len(finished) < len(tail_ids):
                if steps >= max_steps or not llm_engine.has_unfinished_requests():
                    raise RuntimeError(f"{label}: tail request did not finish within {max_steps} steps")
                finished.update(output.request_id for output in llm_engine.step()
                                if output.request_id in tail_ids and output.finished)
                steps += 1
        llm_engine.abort_request(request_ids)
        while llm_engine.has_unfinished_requests():
            llm_engine.step()
    elif case.phase == PHASE_DUMMY:
        llm.collective_rpc(begin_case, args=begin_args)
        for _ in range(case.warmup + case.measured):
            llm.collective_rpc(run_dummy_step)
    else:
        llm.collective_rpc(begin_case, args=begin_args)
        if case.phase == PHASE_PREFILL:
            for repeat in range(case.warmup + case.measured):
                for index in range(case.batch_size):
                    llm_engine.add_request(
                        f"{label}/r{repeat}/q{index}",
                        _random_prompt(rng, case.request_prompt_lens[index], vocab_size),
                        params,
                    )
                while llm_engine.has_unfinished_requests():
                    llm_engine.step()
        else:
            request_ids = [f"{label}/q{index}" for index in range(case.batch_size)]
            for request_id, prompt_len in zip(request_ids, case.request_prompt_lens):
                llm_engine.add_request(
                    request_id, _random_prompt(rng, prompt_len, vocab_size), params
                )
            _step_until(
                llm_engine,
                lambda: llm.collective_rpc(case_progress)[0] >= limit,
                max_steps,
                label,
            )
            llm_engine.abort_request(request_ids)
            while llm_engine.has_unfinished_requests():
                llm_engine.step()

    trace_path = trace_dir / f"{case.case_id}.json"
    result = llm.collective_rpc(finish_case, args=(str(trace_path),))[0]
    if result["recorded"] != limit:
        raise RuntimeError(f"{label}: recorded {result['recorded']} of {limit} steps")
    calls = result["calls"]
    if measurement == KERNEL_ONLY:
        calls = trace_calls(trace_path, result["steps"], case.engine_index)
        if not keep_traces:
            trace_path.unlink()
    return [*result["steps"], *calls]


def build_engine(
    plan: DsaProfilePlan,
    engine_index: int,
    model_dir: str,
    seed: int,
    gpu_memory_utilization: float,
    **engine_options: Any,
) -> Any:
    """Build the in-process engine; ``engine_options`` adds model-specific LLM arguments."""

    policy = plan.limits.engine_policy
    if policy == ENGINE_POLICY_NATIVE_DEFAULTS:
        if engine_options:
            raise ValueError("native_defaults does not accept engine_options overrides")
        execution_options = {"enable_chunked_prefill": False}
    else:
        execution_options = {
            "enable_chunked_prefill": True,
            "async_scheduling": False,
            "long_prefill_token_threshold": plan.engine(engine_index).prefill_query_len,
        }
        if policy == ENGINE_POLICY_GRAPH_REPLAY:
            compilation = {"cudagraph_mode": plan.limits.cudagraph_mode}
            if plan.limits.cudagraph_capture_sizes:
                compilation["cudagraph_capture_sizes"] = list(plan.limits.cudagraph_capture_sizes)
            execution_options["compilation_config"] = compilation
        else:
            execution_options["enforce_eager"] = True
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    engine = plan.engine(engine_index)
    return LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        load_format="dummy",
        hf_overrides=dict(plan.hf_overrides),
        tensor_parallel_size=1,
        dtype=plan.shape.dtype,
        enable_prefix_caching=False,
        max_num_batched_tokens=engine.chunk_budget,
        max_num_seqs=engine.max_num_seqs,
        max_model_len=engine.max_model_len,
        block_size=engine.block_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        **execution_options,
        **engine_options,
    )


def measure_engine(
    *,
    llm: Any,
    plan: DsaProfilePlan,
    engine_index: int,
    inventory: dict[str, Any],
    vocab_size: int,
    seed: int,
    expected_attention_backend: str,
    raw_path: Path,
    trace_dir: Path,
    keep_traces: bool,
    case_ids: set[str] | None,
    log_label: str,
    trace_calls: Callable[[Path, list[dict[str, Any]], int], list[dict[str, Any]]] = _kernel_only_calls,
) -> None:
    """Measure the engine's cases with installed hooks; write raw records to ``raw_path``."""

    measurements = timing_measurements(plan.limits.engine_policy)
    if inventory["attention_backend"] != expected_attention_backend:
        raise RuntimeError(
            f"Engine selected attention backend {inventory['attention_backend']!r}, "
            f"expected {expected_attention_backend!r}"
        )
    device_name, device_capability = collect_device_identity()
    engine_record = {
        "kind": "engine",
        "engine_policy": plan.limits.engine_policy,
        "engine_index": engine_index,
        "chunk_budget": plan.engine(engine_index).chunk_budget,
        "kv_cache_dtype": str(llm.llm_engine.vllm_config.cache_config.cache_dtype),
        "runtime_versions": collect_runtime_versions(),
        "device_name": device_name,
        "device_capability": device_capability,
        **inventory,
    }
    cases = plan.cases_for_engine(engine_index)
    if case_ids is not None:
        unknown = case_ids - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"Cases {sorted(unknown)} are not planned on engine {engine_index}")
        cases = tuple(case for case in cases if case.case_id in case_ids)
    rng = random.Random(seed)
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(engine_record) + "\n")
        for case in cases:
            for measurement in measurements:
                records = run_case(
                    llm, plan, case, measurement, rng, vocab_size, trace_dir, keep_traces, trace_calls
                )
                handle.writelines(json.dumps(record) + "\n" for record in records)
                handle.flush()
                print(f"[{log_label}] engine {engine_index} {case.case_id} {measurement}: "
                      f"{sum(r['kind'] == 'step' for r in records)} steps", flush=True)


def run_engine(
    *,
    plan: DsaProfilePlan,
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
    """Measure every case of one engine; append raw records to ``raw_path``."""

    require_scope_timing(plan.limits.engine_policy)
    llm = build_engine(plan, engine_index, model_dir, seed, gpu_memory_utilization)
    layer_variants = {
        layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers
    }
    inventory = llm.collective_rpc(install_dsa_hooks, args=(engine_index, layer_variants))[0]
    measure_engine(
        llm=llm,
        plan=plan,
        engine_index=engine_index,
        inventory=inventory,
        vocab_size=vocab_size,
        seed=seed,
        expected_attention_backend=expected_attention_backend,
        raw_path=raw_path,
        trace_dir=trace_dir,
        keep_traces=keep_traces,
        case_ids=case_ids,
        log_label="dsa",
    )


__all__ = [
    "DSA_STEP_METADATA",
    "begin_case",
    "build_engine",
    "case_progress",
    "finish_case",
    "graph_replay_calls",
    "install_case_recorder",
    "install_dsa_hooks",
    "measure_engine",
    "run_case",
    "run_dummy_step",
    "run_engine",
    "timing_measurements",
]
