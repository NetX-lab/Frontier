"""CPU-only measurement plan for DSA attention profiling on vLLM.

DSA attention (sparse MLA plus the top-k indexer) is measured inside a real
vLLM engine because its kernels depend on the engine's attention metadata,
KV-cache layout, and per-layer index reuse. The plan fixes everything the GPU
run needs before any GPU work starts:

* a reduced model: the first ``num_layers`` decoder layers with the per-rank
  attention head count of the target tensor-parallel size, dense MLPs so no
  expert routing runs, and the index reuse pattern of the full model;
* one engine per prefill chunk budget (``max_num_batched_tokens``);
* the prefill and decode cases each engine runs.

Prefill cases send one or several prompts together. Short-query tail cases
measure decode-path extends after native chunked context construction. A
separate engine caps each request's prefill chunk to measure several cached
prefills together without changing the original engines. Decode cases run on the
engine with the largest chunk budget, which builds the decode contexts in the
fewest steps and therefore keeps the per-request context spread smallest; only
steps in which exactly the case's requests all decode are measured.

The plan imports no GPU package, so it runs in the dry-run check on a CPU host.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from frontier.attention.families import DSA_ATTENTION_FAMILY
from frontier.attention.dsa_features import dsa_decode_query_limit
from frontier.profiling.common.model_config import ModelConfig

PLAN_SCHEMA_VERSION = "dsa_attention_plan.v1"

INDEXER_SCOPE, MLA_SCOPE = (op.name for op in DSA_ATTENTION_FAMILY.operators)
DSA_SCOPES = (INDEXER_SCOPE, MLA_SCOPE)
FULL_INDEX_VARIANT = "full_index"
SHARED_INDEX_VARIANT = "shared_index"

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_FIRST_TAIL = "first_tail"
PHASE_DUMMY = "dummy"
DP_FULL, DP_PIECEWISE, DP_NONE = "FULL", "PIECEWISE", "NONE"
PILOT_PHASES = (PHASE_PREFILL, PHASE_DECODE, PHASE_FIRST_TAIL, PHASE_DUMMY)
_PILOT_ENGINES = 'engines'
_PILOT_CASES = 'cases'
_PILOT_CASE_ID = 'case_id'
_PILOT_ENGINE_INDEX = 'engine_index'
_PILOT_QUERY_LENGTH = 'prefill_query_len'
_PILOT_PHASE = 'phase'
_PILOT_BATCH_SIZE = 'batch_size'
_PILOT_PROMPT_LENGTH = 'prompt_len'
_PILOT_DP_PEER = 'dp_peer'
_PEER_TOKEN_COUNT = 'num_tokens'
_PEER_MODE = 'cg_mode'
_PEER_UNIFORM_COUNT = 'uniform_token_count'
_PEER_MAX_QUERY_LENGTH = 'max_query_len'
_STEP_HOLD_US = 'step_hold_us'
_CASES_BY_PHASE = 'cases_by_phase'

_PILOT_KEYS = frozenset({_PILOT_ENGINES, _PILOT_CASES})
_PILOT_ENGINE_KEYS = frozenset({_PILOT_ENGINE_INDEX, _PILOT_QUERY_LENGTH})
_PILOT_CASE_KEYS = frozenset({_PILOT_CASE_ID, _PILOT_ENGINE_INDEX, _PILOT_PHASE, _PILOT_BATCH_SIZE, _PILOT_PROMPT_LENGTH, _PILOT_DP_PEER})
_DP_PEER_KEYS = frozenset({_PEER_TOKEN_COUNT, _PEER_MODE, _PEER_UNIFORM_COUNT, _PEER_MAX_QUERY_LENGTH})
ENGINE_POLICY_EAGER_CHUNKED = "eager_chunked"
ENGINE_POLICY_NATIVE_DEFAULTS = "native_defaults"
ENGINE_POLICY_GRAPH_REPLAY = "graph_replay"
GRAPH_REPLAY_CUDAGRAPH_MODES = ("FULL_AND_PIECEWISE", "PIECEWISE", "NONE")
PIECEWISE_CUDAGRAPH_MODES = ("FULL_AND_PIECEWISE", "PIECEWISE")

# Native per-layer config lists that must shrink with ``num_hidden_layers``.
_DENSE_MLP_LAYER_TYPE = "dense"
_PER_LAYER_KEYS = ("mlp_layer_types", "indexer_types")
_INDEXER_TYPE_BY_VARIANT = {FULL_INDEX_VARIANT: "full", SHARED_INDEX_VARIANT: "shared"}


@dataclass(frozen=True)
class DsaLayer:
    global_layer_id: int
    attention_variant: str
    index_producer_layer: int

    @property
    def indexer_executed(self) -> bool:
        return self.attention_variant == FULL_INDEX_VARIANT


@dataclass(frozen=True)
class DsaModelShape:
    model_name: str
    model_type: str
    model_arch: str
    model_architecture_profile: str
    dtype: str
    total_num_layers: int
    num_attention_heads: int
    target_tensor_parallel_size: int
    heads_per_rank: int
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    layers: tuple[DsaLayer, ...]


@dataclass(frozen=True)
class DsaEngineSpec:
    """One measurement engine.

    Legacy engines use the largest case batch. Native-default engines retain
    the requested scheduler limit even when the measured case has one request.
    """

    engine_index: int
    chunk_budget: int
    max_num_seqs: int
    max_model_len: int
    block_size: int
    prefill_query_len: int = 0


@dataclass(frozen=True)
class DsaDpPeer:
    """One emulated DP peer's native dispatch coordinates."""

    num_tokens: int
    cg_mode: str
    uniform_token_count: int | None
    max_query_len: int | None


@dataclass(frozen=True)
class DsaCase:
    """One engine workload.

    Decode: ``batch_size`` prompts of ``prompt_len`` tokens; ``warmup`` pure
    decode steps are skipped and the next ``measured`` steps are recorded.
    Prefill: ``batch_size`` prompts of ``prompt_len`` tokens sent together
    ``warmup + measured`` times; the first ``warmup`` sends are not recorded.
    """

    case_id: str
    engine_index: int
    phase: str
    batch_size: int
    prompt_len: int
    max_tokens: int
    warmup: int
    measured: int
    estimated_steps: int
    dp_peer: DsaDpPeer | None = None


@dataclass(frozen=True)
class DsaMeasurementLimits:
    """Target scheduler and capacity limits that bound the case grid."""

    max_context_len: int
    max_kv_tokens_per_lane: int
    max_num_seqs: int
    chunk_budgets: tuple[int, ...]
    block_size: int
    decode_warmup_steps: int
    decode_measured_steps: int
    prefill_warmup_repeats: int
    prefill_measured_repeats: int
    engine_policy: Literal["eager_chunked", "native_defaults", "graph_replay"] = (
        ENGINE_POLICY_EAGER_CHUNKED
    )

    cudagraph_mode: str | None = None
    cudagraph_capture_sizes: tuple[int, ...] = ()
    step_hold_us: float = 0.0

    def __post_init__(self) -> None:
        if (type(self.step_hold_us) not in (int, float)
                or not math.isfinite(self.step_hold_us) or self.step_hold_us < 0):
            raise ValueError(f"step_hold_us must be finite and non-negative, got {self.step_hold_us!r}")
        if self.step_hold_us and self.engine_policy != ENGINE_POLICY_GRAPH_REPLAY:
            raise ValueError("Nonzero step_hold_us requires engine_policy=graph_replay")
        if self.engine_policy not in (
            ENGINE_POLICY_EAGER_CHUNKED, ENGINE_POLICY_NATIVE_DEFAULTS,
            ENGINE_POLICY_GRAPH_REPLAY,
        ):
            raise ValueError(f"Unknown DSA engine_policy {self.engine_policy!r}")
        for name in (
            "max_context_len",
            "max_kv_tokens_per_lane",
            "max_num_seqs",
            "block_size",
            "decode_measured_steps",
            "prefill_measured_repeats",
        ):
            _require_positive_int(name, getattr(self, name))
        for name in ("decode_warmup_steps", "prefill_warmup_repeats"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if not self.chunk_budgets:
            raise ValueError("chunk_budgets must not be empty")
        for budget in self.chunk_budgets:
            _require_positive_int("chunk budget", budget)
        if len(set(self.chunk_budgets)) != len(self.chunk_budgets):
            raise ValueError(f"chunk_budgets must be unique, got {self.chunk_budgets}")
        if max(self.chunk_budgets) < self.max_num_seqs:
            raise ValueError(
                "The largest chunk budget runs the decode cases and must schedule "
                f"max_num_seqs={self.max_num_seqs} decode tokens in one step, got "
                f"{max(self.chunk_budgets)}"
            )
        if self.engine_policy == ENGINE_POLICY_GRAPH_REPLAY:
            if self.cudagraph_mode not in GRAPH_REPLAY_CUDAGRAPH_MODES:
                raise ValueError(
                    "graph_replay requires cudagraph_mode in "
                    f"{GRAPH_REPLAY_CUDAGRAPH_MODES}, got {self.cudagraph_mode!r}"
                )
            if self.cudagraph_mode in PIECEWISE_CUDAGRAPH_MODES:
                sizes = self.cudagraph_capture_sizes
                if not sizes:
                    raise ValueError("graph_replay requires non-empty cudagraph_capture_sizes")
                for size in sizes:
                    _require_positive_int("cudagraph capture size", size)
                if any(left >= right for left, right in zip(sizes, sizes[1:])):
                    raise ValueError(f"cudagraph_capture_sizes must be strictly increasing, got {sizes}")
                if sizes[-1] > min(self.chunk_budgets):
                    raise ValueError(
                        f"Maximum cudagraph capture size {sizes[-1]} exceeds the smallest "
                        f"chunk budget {min(self.chunk_budgets)}"
                    )
            elif self.cudagraph_capture_sizes:
                raise ValueError("cudagraph_mode NONE requires empty cudagraph_capture_sizes")
        elif self.cudagraph_mode is not None or self.cudagraph_capture_sizes:
            raise ValueError(
                f"{self.engine_policy} requires cudagraph_mode=None and empty "
                "cudagraph_capture_sizes"
            )
        if self.engine_policy == ENGINE_POLICY_NATIVE_DEFAULTS:
            if len(self.chunk_budgets) != 1:
                raise ValueError("native_defaults requires exactly one token budget")
            if self.chunk_budgets[0] < self.max_context_len:
                raise ValueError(
                    "native_defaults disables chunking and requires token budget "
                    f">= max_context_len={self.max_context_len}"
                )


@dataclass(frozen=True)
class DsaProfilePlan:
    schema_version: str
    shape: DsaModelShape
    limits: DsaMeasurementLimits
    hf_overrides: Mapping[str, Any]
    engines: tuple[DsaEngineSpec, ...]
    cases: tuple[DsaCase, ...]

    def engine(self, engine_index: int) -> DsaEngineSpec:
        for engine in self.engines:
            if engine.engine_index == engine_index:
                return engine
        raise ValueError(
            f"engine_index {engine_index} is not in the plan "
            f"(0..{len(self.engines) - 1})"
        )

    def cases_for_engine(self, engine_index: int) -> tuple[DsaCase, ...]:
        self.engine(engine_index)
        return tuple(case for case in self.cases if case.engine_index == engine_index)

    def layer(self, global_layer_id: int) -> DsaLayer:
        for layer in self.shape.layers:
            if layer.global_layer_id == global_layer_id:
                return layer
        raise ValueError(f"Layer {global_layer_id} is not in the reduced model")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hf_overrides"] = dict(self.hf_overrides)
        if self.limits.engine_policy == ENGINE_POLICY_EAGER_CHUNKED:
            # V4 and FULL callers inherit this serializer and frozen legacy plans.
            payload["limits"].pop("engine_policy")
        if self.limits.engine_policy != ENGINE_POLICY_GRAPH_REPLAY:
            payload["limits"].pop("cudagraph_mode")
            payload["limits"].pop("cudagraph_capture_sizes")
            payload["limits"].pop(_STEP_HOLD_US)
        if self.limits.engine_policy == ENGINE_POLICY_NATIVE_DEFAULTS:
            payload["timing_collection"] = "pending_native_scope_instrumentation"
        for case in payload[_PILOT_CASES]:
            if case[_PILOT_DP_PEER] is None:
                case.pop(_PILOT_DP_PEER)
        return payload

    def summary(self) -> dict[str, Any]:
        per_engine = []
        for engine in self.engines:
            cases = self.cases_for_engine(engine.engine_index)
            per_engine.append(
                {
                    _PILOT_ENGINE_INDEX: engine.engine_index,
                    "chunk_budget": engine.chunk_budget,
                    "prefill_cases": sum(c.phase == PHASE_PREFILL for c in cases),
                    "decode_cases": sum(c.phase == PHASE_DECODE for c in cases),
                    "estimated_steps": sum(c.estimated_steps for c in cases),
                }
            )
        layers = self.shape.layers
        summary = {
            "schema_version": self.schema_version,
            "model_name": self.shape.model_name,
            "heads_per_rank": self.shape.heads_per_rank,
            "target_tensor_parallel_size": self.shape.target_tensor_parallel_size,
            "reduced_layers": [
                (layer.global_layer_id, layer.attention_variant) for layer in layers
            ],
            "num_engines": len(self.engines),
            "num_cases": len(self.cases),
            "num_prefill_cases": sum(c.phase == PHASE_PREFILL for c in self.cases),
            "num_decode_cases": sum(c.phase == PHASE_DECODE for c in self.cases),
            "estimated_steps": sum(c.estimated_steps for c in self.cases),
            _PILOT_ENGINES: per_engine,
            "hf_overrides": dict(self.hf_overrides),
        }
        if self.limits.engine_policy == ENGINE_POLICY_NATIVE_DEFAULTS:
            summary["engine_policy"] = self.limits.engine_policy
            summary["timing_collection"] = "pending_native_scope_instrumentation"
        if self.limits.engine_policy == ENGINE_POLICY_GRAPH_REPLAY:
            summary["engine_policy"] = self.limits.engine_policy
            summary["cudagraph_mode"] = self.limits.cudagraph_mode
            summary["cudagraph_capture_sizes"] = self.limits.cudagraph_capture_sizes
            summary[_STEP_HOLD_US] = self.limits.step_hold_us
            summary[_CASES_BY_PHASE] = {
                phase: sum(case.phase == phase for case in self.cases) for phase in PILOT_PHASES
            }
        return summary


def _require_positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _required_config_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    _require_positive_int(f"model config {key}", value)
    return value


def _required_config_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model config {key} must be a non-empty string, got {value!r}")
    return value


def resolve_reduced_layers(
    model_config: ModelConfig, num_layers: int
) -> tuple[DsaLayer, ...]:
    """Take the first ``num_layers`` layers of the Frontier index topology."""

    specs = model_config.get_layer_attention_specs()
    if num_layers > len(specs):
        raise ValueError(
            f"num_layers={num_layers} exceeds the model's {len(specs)} layers"
        )
    layers = []
    for spec in specs[:num_layers]:
        if spec.family_id != DSA_ATTENTION_FAMILY.family_id:
            raise ValueError(
                f"Layer {spec.global_layer_id} uses attention family "
                f"{spec.family_id!r}, not {DSA_ATTENTION_FAMILY.family_id!r}"
            )
        layers.append(
            DsaLayer(
                global_layer_id=spec.global_layer_id,
                attention_variant=spec.variant_id,
                index_producer_layer=spec.index_producer_layer_id,
            )
        )
    variants = {layer.attention_variant for layer in layers}
    if variants != {FULL_INDEX_VARIANT, SHARED_INDEX_VARIANT}:
        raise ValueError(
            "The reduced model must contain at least one full-index and one "
            f"shared-index layer; num_layers={num_layers} gives {sorted(variants)}"
        )
    return tuple(layers)


def build_reduced_hf_overrides(
    hf_config: Mapping[str, Any],
    layers: Sequence[DsaLayer],
    heads_per_rank: int,
) -> dict[str, Any]:
    """Shrink the native config to the measured layers and per-rank heads.

    Every layer keeps a dense MLP so no expert routing runs; the attention and
    index-reuse configuration stays native.
    """

    total = _required_config_int(hf_config, "num_hidden_layers")
    num_layers = len(layers)
    overrides: dict[str, Any] = {
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads_per_rank,
        "num_key_value_heads": heads_per_rank,
        "first_k_dense_replace": num_layers,
    }
    for key, value in hf_config.items():
        if not isinstance(value, list) or len(value) != total:
            continue
        if key not in _PER_LAYER_KEYS:
            raise ValueError(
                f"Model config key {key!r} lists one value per layer; the reduced "
                "model does not know how to shrink it"
            )
    if "mlp_layer_types" in hf_config:
        overrides["mlp_layer_types"] = [_DENSE_MLP_LAYER_TYPE] * num_layers
    if "indexer_types" in hf_config:
        native = list(hf_config["indexer_types"][:num_layers])
        expected = [_INDEXER_TYPE_BY_VARIANT[layer.attention_variant] for layer in layers]
        if native != expected:
            raise ValueError(
                "Native indexer_types disagree with the Frontier index topology: "
                f"native={native}, frontier={expected}"
            )
        overrides["indexer_types"] = native
    return overrides


def build_context_grid(
    block_size: int, index_topk: int, max_context: int
) -> tuple[int, ...]:
    """Powers of two from one KV block, the top-k boundary, and the maximum."""

    points = {index_topk - 1, index_topk, index_topk + 1, max_context}
    value = block_size
    while value <= max_context:
        points.add(value)
        value *= 2
    return tuple(sorted(point for point in points if 1 <= point <= max_context))


def build_batch_grid(max_num_seqs: int) -> tuple[int, ...]:
    sizes = {max_num_seqs}
    value = 1
    while value <= max_num_seqs:
        sizes.add(value)
        value *= 2
    return tuple(sorted(sizes))


def _prefill_steps(prompt_len: int, chunk_budget: int) -> int:
    return math.ceil(prompt_len / chunk_budget)


def _build_prefill_cases(
    chunk_budgets: Sequence[int],
    limits: DsaMeasurementLimits,
    context_grid: Sequence[int],
    decode_query_limit: int | None = None,
) -> list[DsaCase]:
    """One full-length prompt per engine plus each shorter single-chunk prompt.

    The full-length prompt yields chunks of the engine's budget at every
    multiple of that budget as cached context. A prompt shorter than a budget
    is one chunk with no cached context; it runs once, on the smallest engine
    whose budget exceeds it.
    """

    if (limits.engine_policy != ENGINE_POLICY_NATIVE_DEFAULTS
            and max(chunk_budgets) >= limits.max_context_len):
        raise ValueError(
            f"chunk budgets must stay below max_context_len={limits.max_context_len}"
        )
    repeats = limits.prefill_warmup_repeats + limits.prefill_measured_repeats
    long_prompt = limits.max_context_len - 1
    cases = []
    for engine_index, budget in enumerate(chunk_budgets):
        cases.append(
            DsaCase(
                case_id=f"e{engine_index}-prefill-p{long_prompt}",
                engine_index=engine_index,
                phase=PHASE_PREFILL,
                batch_size=1,
                prompt_len=long_prompt,
                max_tokens=1,
                warmup=limits.prefill_warmup_repeats,
                measured=limits.prefill_measured_repeats,
                estimated_steps=repeats * _prefill_steps(long_prompt, budget),
            )
        )
    short_queries = (
        (*build_batch_grid(min(decode_query_limit, long_prompt)), decode_query_limit + 1)
        if decode_query_limit is not None else ()
    )
    for prompt_len in sorted(set(context_grid) | set(short_queries)):
        engine_index = next(
            (index for index, budget in enumerate(chunk_budgets) if budget > prompt_len),
            None,
        )
        if engine_index is None:
            continue
        cases.append(
            DsaCase(
                case_id=f"e{engine_index}-prefill-p{prompt_len}",
                engine_index=engine_index,
                phase=PHASE_PREFILL,
                batch_size=1,
                prompt_len=prompt_len,
                max_tokens=1,
                warmup=limits.prefill_warmup_repeats,
                measured=limits.prefill_measured_repeats,
                estimated_steps=repeats,
            )
        )
    if decode_query_limit is None:
        # DeepSeek-V4 reuses the ordinary prefill plan, not sparse-MLA paths.
        return cases
    # Multiple long queries use the sparse prefill path. Schedule all prompts
    # in one step so the recorded sequence count is the requested case size.
    for engine_index, budget in enumerate(chunk_budgets):
        max_batch = min(limits.max_num_seqs, budget // (decode_query_limit + 1))
        if max_batch < 2:
            continue
        for batch_size in build_batch_grid(max_batch):
            if batch_size == 1:
                continue
            for prompt_len in sorted({decode_query_limit + 1, min(budget // batch_size, long_prompt)}):
                if batch_size * (prompt_len + 1) > limits.max_kv_tokens_per_lane:
                    continue
                cases.append(DsaCase(
                    case_id=f"e{engine_index}-prefill-b{batch_size}-p{prompt_len}",
                    engine_index=engine_index, phase=PHASE_PREFILL, batch_size=batch_size,
                    prompt_len=prompt_len, max_tokens=1,
                    warmup=limits.prefill_warmup_repeats, measured=limits.prefill_measured_repeats,
                    estimated_steps=repeats,
                ))
    return cases


def _decode_case_steps(batch: int, context: int, budget: int,
                       limits: DsaMeasurementLimits) -> tuple[int, int]:
    """Use the same context-construction allowance for grids and exact pilots."""
    build_steps = math.ceil(batch * context / (budget - batch + 1)) + 1
    return build_steps, build_steps + limits.decode_warmup_steps + limits.decode_measured_steps + 1


def _build_decode_cases(
    engine_index: int,
    budget: int,
    limits: DsaMeasurementLimits,
    context_grid: Sequence[int],
) -> list[DsaCase]:
    """Decode cases bounded by the target lane's KV capacity and model length.

    While the case's prompts prefill, a request that already finished its
    prompt decodes one token per step. ``max_tokens`` therefore covers the
    prefill steps of all prompts (each step also schedules up to ``batch``
    decode tokens) plus the warmup and measured decode steps.
    """

    cases = []
    for batch in build_batch_grid(limits.max_num_seqs):
        for context in context_grid:
            build_steps, max_tokens = _decode_case_steps(batch, context, budget, limits)
            if context + max_tokens > limits.max_context_len:
                continue
            if batch * (context + max_tokens) > limits.max_kv_tokens_per_lane:
                continue
            cases.append(
                DsaCase(
                    case_id=f"e{engine_index}-decode-b{batch}-p{context}",
                    engine_index=engine_index,
                    phase=PHASE_DECODE,
                    batch_size=batch,
                    prompt_len=context,
                    max_tokens=max_tokens,
                    warmup=limits.decode_warmup_steps,
                    measured=limits.decode_measured_steps,
                    estimated_steps=build_steps
                    + limits.decode_warmup_steps
                    + limits.decode_measured_steps,
                )
            )
    return cases


def _build_extend_cases(
    chunk_budgets: Sequence[int], limits: DsaMeasurementLimits,
    context_grid: Sequence[int], decode_query_limit: int,
) -> list[DsaCase]:
    """Short-query tails with cached keys, plus concurrent short prompts.

    Context construction uses ordinary native chunking, not injected cache or
    request state. Use the largest budget dividing the cached context so its
    final chunk has exactly the requested query length.
    """
    repeats = limits.prefill_warmup_repeats + limits.prefill_measured_repeats
    queries = build_batch_grid(decode_query_limit)
    cases = []
    for context in context_grid:
        candidates = [b for b in chunk_budgets if context % b == 0]
        if not candidates:
            continue
        budget = max(candidates)
        engine_index = chunk_budgets.index(budget)
        for query in queries:
            if query == 1 or context + query >= limits.max_context_len:
                continue
            if context + query + 1 > limits.max_kv_tokens_per_lane:
                continue
            cases.append(DsaCase(
                case_id=f"e{engine_index}-extend-q{query}-c{context}",
                engine_index=engine_index, phase=PHASE_PREFILL, batch_size=1,
                prompt_len=context + query, max_tokens=1,
                warmup=limits.prefill_warmup_repeats,
                measured=limits.prefill_measured_repeats,
                estimated_steps=repeats * _prefill_steps(context + query, budget),
            ))
    for batch in build_batch_grid(limits.max_num_seqs):
        if batch == 1:
            continue
        for query in queries:
            if query == 1 or batch * (query + 1) > limits.max_kv_tokens_per_lane:
                continue
            engine_index = next((i for i, budget in enumerate(chunk_budgets)
                                 if budget >= batch * query), None)
            if engine_index is None:
                continue
            cases.append(DsaCase(
                case_id=f"e{engine_index}-extend-b{batch}-q{query}",
                engine_index=engine_index, phase=PHASE_PREFILL, batch_size=batch,
                prompt_len=query, max_tokens=1,
                warmup=limits.prefill_warmup_repeats,
                measured=limits.prefill_measured_repeats, estimated_steps=repeats,
            ))
    return cases


def _build_cached_prefill_cases(
    engine_index: int, budget: int, query: int, limits: DsaMeasurementLimits,
    tail_queries: Sequence[int] = (),
) -> list[DsaCase]:
    """Uniform native per-request chunks keep every sequence in each step."""
    repeats = limits.prefill_warmup_repeats + limits.prefill_measured_repeats
    cases = []
    for batch in build_batch_grid(min(limits.max_num_seqs, budget // query)):
        if batch == 1:
            continue
        max_prompt = min(limits.max_context_len - 1,
                         limits.max_kv_tokens_per_lane // batch - 1)
        # Whole chunks keep the final step on the same sparse-prefill path.
        prompts = {max_prompt // query * query}
        # Two short-tail boundaries span summed keys for cached multi-request
        # extends; initial short prompts alone cannot cover this work.
        prompts.update((max_prompt - tail) // query * query + tail for tail in tail_queries)
        for prompt in sorted(prompts):
            if prompt <= query:
                continue
            cases.append(DsaCase(
                case_id=f"e{engine_index}-prefill-b{batch}-p{prompt}",
                engine_index=engine_index, phase=PHASE_PREFILL, batch_size=batch,
                prompt_len=prompt, max_tokens=1,
                warmup=limits.prefill_warmup_repeats,
                measured=limits.prefill_measured_repeats,
                estimated_steps=repeats * _prefill_steps(prompt, query),
            ))
    return cases


def _build_serving_cases(
    limits: DsaMeasurementLimits, max_context: int | None,
    min_context: int, prefill_prompt_len: int | None,
) -> tuple[tuple[DsaEngineSpec, ...], tuple[DsaCase, ...]]:
    """Exact ordinary prefill and inclusive cached-context q1 fixtures.

    Seed below the q1 window so decode warmup precedes its lower endpoint.
    The setup prefill and warmup decodes are unmeasured; every decode in the
    window is recorded. The local seed does not reproduce the request phase
    of a remote-prefill tail. The legacy zero endpoint retains its separate
    real one-token prefill, but decoding from context 1 requires zero warmup
    because a native request needs a positive prompt.
    """
    capacity = min(limits.max_context_len, limits.max_kv_tokens_per_lane)
    if prefill_prompt_len is not None:
        _require_positive_int("prefill_prompt_len", prefill_prompt_len)
        if prefill_prompt_len + 1 > capacity:
            raise ValueError("prefill_prompt_len plus one output exceeds request capacity")
    if max_context is not None:
        if type(max_context) is not int or max_context < 0:
            raise ValueError("query_one_context_max must be a non-negative int")
        if min_context > max_context:
            raise ValueError("query_one_context_min must not exceed query_one_context_max")
        lower = max(1, min_context)
        if max_context >= lower and lower <= limits.decode_warmup_steps:
            raise ValueError(
                f"query_one_context_min={min_context} starts decode at context {lower}, "
                f"which cannot hold decode_warmup_steps={limits.decode_warmup_steps} "
                "and a positive prompt"
            )
        # Seed + warmup + measured decodes + one setup output = max_context + 2.
        request_len = max_context + 2
        if request_len > capacity:
            raise ValueError(
                f"query_one_context_max={max_context} requires capacity for "
                f"{request_len} prompt-plus-output tokens"
            )
    elif prefill_prompt_len is None:
        raise ValueError("native_defaults requires an explicit prefill or q1 window")
    engine = DsaEngineSpec(
        engine_index=0, chunk_budget=min(limits.chunk_budgets),
        max_num_seqs=(limits.max_num_seqs
                      if limits.engine_policy == ENGINE_POLICY_NATIVE_DEFAULTS else 1),
        max_model_len=limits.max_context_len, block_size=limits.block_size,
    )
    prompts = set() if prefill_prompt_len is None else {prefill_prompt_len}
    if max_context is not None and min_context == 0:
        prompts.add(1)
    cases = []
    for prompt in sorted(prompts):
        cases.append(DsaCase(
            case_id=f"e0-prefill-p{prompt}", engine_index=0, phase=PHASE_PREFILL,
            batch_size=1, prompt_len=prompt, max_tokens=1,
            warmup=limits.prefill_warmup_repeats, measured=limits.prefill_measured_repeats,
            estimated_steps=(limits.prefill_warmup_repeats + limits.prefill_measured_repeats)
            * _prefill_steps(prompt, engine.chunk_budget),
        ))
    lower = max(1, min_context)
    if max_context is not None and max_context >= lower:
        count = max_context - lower + 1
        prompt = lower - limits.decode_warmup_steps
        cases.append(DsaCase(
            case_id=f"e0-decode-q1-c{lower}-{max_context}",
            engine_index=0, phase=PHASE_DECODE,
            batch_size=1, prompt_len=prompt,
            max_tokens=limits.decode_warmup_steps + count + 1,
            warmup=limits.decode_warmup_steps, measured=count,
            estimated_steps=_prefill_steps(prompt, engine.chunk_budget)
            + limits.decode_warmup_steps + count,
        ))
    return (engine,), tuple(cases)


def _strict_keys(payload: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing, unknown = expected - payload.keys(), payload.keys() - expected
    if missing or unknown:
        raise ValueError(f"{label}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def _nonnegative_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _pilot_peer(payload: Any, limits: DsaMeasurementLimits) -> DsaDpPeer | None:
    if payload is None:
        return None
    _strict_keys(payload, _DP_PEER_KEYS, _PILOT_DP_PEER)
    peer = DsaDpPeer(**payload)
    _require_positive_int("dp_peer num_tokens", peer.num_tokens)
    if peer.num_tokens > limits.chunk_budgets[0]:
        raise ValueError("dp_peer num_tokens exceeds chunk budget")
    if peer.cg_mode not in (DP_FULL, DP_PIECEWISE, DP_NONE):
        raise ValueError(f"Unknown dp_peer cg_mode {peer.cg_mode!r}")
    for name in (_PEER_UNIFORM_COUNT, _PEER_MAX_QUERY_LENGTH):
        if getattr(peer, name) is not None:
            _require_positive_int(f"dp_peer {name}", getattr(peer, name))
    if peer.cg_mode == DP_FULL and peer.uniform_token_count is None:
        raise ValueError("FULL dp_peer requires uniform_token_count")
    reachable = (peer.cg_mode == DP_NONE
                 or peer.cg_mode == DP_FULL and limits.cudagraph_mode == GRAPH_REPLAY_CUDAGRAPH_MODES[0]
                 or peer.cg_mode == DP_PIECEWISE and limits.cudagraph_mode in PIECEWISE_CUDAGRAPH_MODES)
    if not reachable:
        raise ValueError(f"dp_peer mode {peer.cg_mode} is unreachable under {limits.cudagraph_mode}")
    return peer


def build_pilot_cases(limits: DsaMeasurementLimits, payload: Mapping[str, Any]
                      ) -> tuple[tuple[DsaEngineSpec, ...], tuple[DsaCase, ...]]:
    """Load an exact bridge case list, rejecting coordinates a grid would skip.

    First tails retain native context construction; dummy cases execute the
    worker's ordinary dummy batch, including attention metadata.
    """
    if limits.engine_policy != ENGINE_POLICY_GRAPH_REPLAY:
        raise ValueError("pilot_cases requires engine_policy=graph_replay")
    if len(limits.chunk_budgets) != 1:
        raise ValueError("pilot_cases requires exactly one chunk budget")
    _strict_keys(payload, _PILOT_KEYS, "pilot_cases")
    if not isinstance(payload[_PILOT_ENGINES], list) or not payload[_PILOT_ENGINES]:
        raise ValueError("pilot engines must be a non-empty JSON list")
    if not isinstance(payload[_PILOT_CASES], list) or not payload[_PILOT_CASES]:
        raise ValueError("pilot cases must be a non-empty JSON list")
    budget = limits.chunk_budgets[0]
    engines = []
    for row in payload[_PILOT_ENGINES]:
        _strict_keys(row, _PILOT_ENGINE_KEYS, "pilot engine")
        _nonnegative_int(_PILOT_ENGINE_INDEX, row[_PILOT_ENGINE_INDEX])
        _nonnegative_int(_PILOT_QUERY_LENGTH, row[_PILOT_QUERY_LENGTH])
        if row[_PILOT_ENGINE_INDEX] != len(engines):
            raise ValueError("pilot engine indices must be contiguous from 0")
        engines.append(DsaEngineSpec(row[_PILOT_ENGINE_INDEX], budget, limits.max_num_seqs,
                                    limits.max_context_len, limits.block_size, row[_PILOT_QUERY_LENGTH]))
    cases, ids = [], set()
    for row in payload[_PILOT_CASES]:
        _strict_keys(row, _PILOT_CASE_KEYS, "pilot case")
        case_id = row[_PILOT_CASE_ID]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("pilot case_id must be a non-empty string")
        if case_id in ids:
            raise ValueError(f"Duplicate pilot case_id {case_id!r}")
        ids.add(case_id)
        index = row[_PILOT_ENGINE_INDEX]
        _nonnegative_int("case engine_index", index)
        if index >= len(engines):
            raise ValueError(f"Pilot case {case_id} references missing engine {index}")
        phase, batch, prompt = row[_PILOT_PHASE], row[_PILOT_BATCH_SIZE], row[_PILOT_PROMPT_LENGTH]
        if phase not in PILOT_PHASES:
            raise ValueError(f"Unknown pilot phase {phase!r}")
        _require_positive_int(_PILOT_BATCH_SIZE, batch)
        if batch > limits.max_num_seqs or batch > budget:
            raise ValueError("pilot batch_size exceeds max_num_seqs or chunk budget")
        _nonnegative_int(_PILOT_PROMPT_LENGTH, prompt)
        if phase != PHASE_DUMMY:
            _require_positive_int(_PILOT_PROMPT_LENGTH, prompt)
        peer = _pilot_peer(row[_PILOT_DP_PEER], limits)
        q = engines[index].prefill_query_len
        if phase in (PHASE_PREFILL, PHASE_DECODE) and q:
            raise ValueError(f"{phase} requires engine prefill_query_len == 0")
        if phase in (PHASE_PREFILL, PHASE_FIRST_TAIL) and peer is not None:
            raise ValueError(f"{phase} does not allow a dp_peer")
        if phase in (PHASE_PREFILL, PHASE_DUMMY) and batch != 1:
            raise ValueError(f"{phase} requires batch_size == 1")
        if phase == PHASE_PREFILL:
            warmup, measured = limits.prefill_warmup_repeats, limits.prefill_measured_repeats
            max_tokens, estimated = 1, (warmup + measured) * _prefill_steps(prompt, budget)
            if prompt + 1 > limits.max_context_len:
                raise ValueError("prefill prompt_len + 1 exceeds max_context_len")
            if prompt + 1 > limits.max_kv_tokens_per_lane:
                raise ValueError("prefill prompt_len + 1 exceeds max_kv_tokens_per_lane")
        elif phase == PHASE_DECODE:
            warmup, measured = limits.decode_warmup_steps, limits.decode_measured_steps
            build_steps, max_tokens = _decode_case_steps(batch, prompt, budget, limits)
            estimated = build_steps + warmup + measured
            if prompt + max_tokens > limits.max_context_len:
                raise ValueError("decode context + max_tokens exceeds max_context_len")
            if batch * (prompt + max_tokens) > limits.max_kv_tokens_per_lane:
                raise ValueError("decode batch * (context + max_tokens) exceeds max_kv_tokens_per_lane")
        elif phase == PHASE_DUMMY:
            if prompt != 0 or peer is None:
                raise ValueError("dummy requires prompt_len == 0 and a dp_peer")
            warmup, measured = limits.decode_warmup_steps, limits.decode_measured_steps
            max_tokens, estimated = 0, warmup + measured
        else:
            if q <= 0:
                raise ValueError("first_tail requires positive engine prefill_query_len")
            if (prompt - 1) % q:
                raise ValueError("first_tail requires (prompt_len - 1) divisible by prefill_query_len")
            if q + batch - 1 > budget:
                raise ValueError("first_tail query + batch_size - 1 exceeds chunk budget")
            tail_steps = (prompt - 1) // q + 1
            warmup, measured = limits.prefill_warmup_repeats, limits.prefill_measured_repeats
            max_tokens = limits.max_context_len - prompt
            build_steps = 0 if batch == 1 else math.ceil((batch - 1) * prompt / (budget - batch + 2)) + tail_steps
            estimated = build_steps + (warmup + measured) * tail_steps
            if estimated >= max_tokens:
                raise ValueError("first_tail estimated_steps must be < max_tokens")
            if (batch - 1) * limits.max_context_len + prompt + 1 > limits.max_kv_tokens_per_lane:
                raise ValueError("first_tail KV bound exceeds max_kv_tokens_per_lane")
        cases.append(DsaCase(case_id, index, phase, batch, prompt, max_tokens,
                             warmup, measured, estimated, peer))
    return tuple(engines), tuple(cases)


def build_dsa_profile_plan(
    *,
    model_name: str,
    hf_config: Mapping[str, Any],
    target_tensor_parallel_size: int,
    num_layers: int,
    limits: DsaMeasurementLimits,
    query_one_context_max: int | None = None,
    query_one_context_min: int = 0,
    prefill_prompt_len: int | None = None,
) -> DsaProfilePlan:
    if type(query_one_context_min) is not int or query_one_context_min < 0:
        raise ValueError("query_one_context_min must be a non-negative int")
    if query_one_context_max is None and query_one_context_min:
        raise ValueError("query_one_context_min requires query_one_context_max")
    model_config = ModelConfig.from_model_name(model_name)
    if not model_config.use_mla:
        raise ValueError(f"{model_name} does not use MLA attention")
    family = model_config.get_attention_family()
    if family.family_id != DSA_ATTENTION_FAMILY.family_id:
        raise ValueError(
            f"{model_name} resolves attention family {family.family_id!r}, "
            f"not {DSA_ATTENTION_FAMILY.family_id!r}"
        )
    _require_positive_int("target_tensor_parallel_size", target_tensor_parallel_size)
    _require_positive_int("num_layers", num_layers)
    num_heads = _required_config_int(hf_config, "num_attention_heads")
    if num_heads % target_tensor_parallel_size:
        raise ValueError(
            f"num_attention_heads={num_heads} is not divisible by "
            f"tensor_parallel_size={target_tensor_parallel_size}"
        )
    heads_per_rank = num_heads // target_tensor_parallel_size
    layers = resolve_reduced_layers(model_config, num_layers)
    shape = DsaModelShape(
        model_name=model_name,
        model_type=str(hf_config["model_type"]),
        model_arch=str(model_config.model_arch),
        dtype=_required_config_str(hf_config, "dtype"),
        model_architecture_profile=model_config.get_model_architecture_profile().profile_id,
        total_num_layers=_required_config_int(hf_config, "num_hidden_layers"),
        num_attention_heads=num_heads,
        target_tensor_parallel_size=target_tensor_parallel_size,
        heads_per_rank=heads_per_rank,
        hidden_size=_required_config_int(hf_config, "hidden_size"),
        q_lora_rank=_required_config_int(hf_config, "q_lora_rank"),
        kv_lora_rank=_required_config_int(hf_config, "kv_lora_rank"),
        qk_nope_head_dim=_required_config_int(hf_config, "qk_nope_head_dim"),
        qk_rope_head_dim=_required_config_int(hf_config, "qk_rope_head_dim"),
        v_head_dim=_required_config_int(hf_config, "v_head_dim"),
        index_topk=_required_config_int(hf_config, "index_topk"),
        index_n_heads=_required_config_int(hf_config, "index_n_heads"),
        index_head_dim=_required_config_int(hf_config, "index_head_dim"),
        layers=layers,
    )
    if (query_one_context_max is not None or prefill_prompt_len is not None
            or limits.engine_policy == ENGINE_POLICY_NATIVE_DEFAULTS):
        engines, cases = _build_serving_cases(
            limits, query_one_context_max, query_one_context_min, prefill_prompt_len
        )
        return DsaProfilePlan(
            schema_version=PLAN_SCHEMA_VERSION, shape=shape, limits=limits,
            hf_overrides=build_reduced_hf_overrides(hf_config, layers, heads_per_rank),
            engines=engines, cases=cases,
        )
    chunk_budgets = tuple(sorted(limits.chunk_budgets))
    context_grid = build_context_grid(
        limits.block_size, shape.index_topk, limits.max_context_len
    )
    # The largest budget runs the decode cases.
    decode_query_limit = dsa_decode_query_limit(heads_per_rank)
    cases = _build_prefill_cases(chunk_budgets, limits, context_grid, decode_query_limit)
    cases += _build_decode_cases(
        len(chunk_budgets) - 1, chunk_budgets[-1], limits, context_grid
    )
    # A single request needs no allowance for other requests' prefill. Include
    # the final legal context, beyond the conservative build allowance of the
    # original decode cases. Prefill itself samples the first output.
    output_tokens = limits.decode_warmup_steps + limits.decode_measured_steps + 1
    prompt = limits.max_context_len - output_tokens
    if prompt > 0 and limits.max_context_len <= limits.max_kv_tokens_per_lane:
        cases.append(DsaCase(
            case_id=f"e{len(chunk_budgets) - 1}-decode-b1-p{prompt}",
            engine_index=len(chunk_budgets) - 1, phase=PHASE_DECODE, batch_size=1,
            prompt_len=prompt, max_tokens=output_tokens,
            warmup=limits.decode_warmup_steps, measured=limits.decode_measured_steps,
            estimated_steps=_prefill_steps(prompt, chunk_budgets[-1]) + output_tokens,
        ))
    cases += _build_extend_cases(chunk_budgets, limits, context_grid, decode_query_limit)
    engines = [
        DsaEngineSpec(
            engine_index=index,
            chunk_budget=budget,
            max_num_seqs=max(
                case.batch_size for case in cases if case.engine_index == index
            ),
            max_model_len=limits.max_context_len,
            block_size=limits.block_size,
        )
        for index, budget in enumerate(chunk_budgets)
    ]
    # Small chunks cover many cached sequences; two large chunks cover the
    # high causal-pair count of two long requests sharing the token budget.
    queries = {1 << decode_query_limit.bit_length(), chunk_budgets[-1] // 2}
    for query in sorted(query for query in queries if query > decode_query_limit):
        cached_prefills = _build_cached_prefill_cases(
            len(engines), chunk_budgets[-1], query, limits,
            tail_queries=(2, decode_query_limit) if query == min(queries) else (),
        )
        if not cached_prefills:
            continue
        engines.append(DsaEngineSpec(
            engine_index=len(engines), chunk_budget=chunk_budgets[-1],
            max_num_seqs=max(case.batch_size for case in cached_prefills),
            max_model_len=limits.max_context_len, block_size=limits.block_size,
            prefill_query_len=query,
        ))
        cases += cached_prefills
    return DsaProfilePlan(
        schema_version=PLAN_SCHEMA_VERSION,
        shape=shape,
        limits=limits,
        hf_overrides=build_reduced_hf_overrides(hf_config, layers, heads_per_rank),
        engines=tuple(engines),
        cases=tuple(cases),
    )


def require_scope_timing(engine_policy: str) -> None:
    """Keep unqualified native-default collection and row admission explicit."""
    if engine_policy != ENGINE_POLICY_EAGER_CHUNKED:
        raise ValueError(
            f"DSA engine_policy={engine_policy!r} cannot collect or aggregate timings: "
            "native compiled/graph per-scope instrumentation is not qualified. "
            "native_defaults supports planning and engine configuration only."
        )


def load_hf_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "DSA_SCOPES",
    "ENGINE_POLICY_EAGER_CHUNKED",
    "ENGINE_POLICY_NATIVE_DEFAULTS",
    "ENGINE_POLICY_GRAPH_REPLAY",
    "GRAPH_REPLAY_CUDAGRAPH_MODES",
    "PIECEWISE_CUDAGRAPH_MODES",
    "DsaCase",
    "DsaDpPeer",
    "DP_FULL", "DP_PIECEWISE", "DP_NONE",
    "PHASE_FIRST_TAIL", "PHASE_DUMMY",
    "build_pilot_cases",
    "DsaEngineSpec",
    "DsaLayer",
    "DsaMeasurementLimits",
    "DsaModelShape",
    "DsaProfilePlan",
    "FULL_INDEX_VARIANT",
    "INDEXER_SCOPE",
    "MLA_SCOPE",
    "PHASE_DECODE",
    "PHASE_PREFILL",
    "SHARED_INDEX_VARIANT",
    "build_batch_grid",
    "build_context_grid",
    "build_dsa_profile_plan",
    "build_reduced_hf_overrides",
    "load_hf_config",
    "require_scope_timing",
    "resolve_reduced_layers",
]
