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
from typing import Any, Mapping, Sequence

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

    ``max_num_seqs`` is the largest batch among the engine's cases, so it
    never exceeds ``chunk_budget`` as vLLM requires.
    """

    engine_index: int
    chunk_budget: int
    max_num_seqs: int
    max_model_len: int
    block_size: int
    prefill_query_len: int = 0


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

    def __post_init__(self) -> None:
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
        if max(self.chunk_budgets) >= self.max_context_len:
            raise ValueError(
                f"chunk budgets must stay below max_context_len={self.max_context_len}"
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
        return payload

    def summary(self) -> dict[str, Any]:
        per_engine = []
        for engine in self.engines:
            cases = self.cases_for_engine(engine.engine_index)
            per_engine.append(
                {
                    "engine_index": engine.engine_index,
                    "chunk_budget": engine.chunk_budget,
                    "prefill_cases": sum(c.phase == PHASE_PREFILL for c in cases),
                    "decode_cases": sum(c.phase == PHASE_DECODE for c in cases),
                    "estimated_steps": sum(c.estimated_steps for c in cases),
                }
            )
        layers = self.shape.layers
        return {
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
            "engines": per_engine,
            "hf_overrides": dict(self.hf_overrides),
        }


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
            build_steps = math.ceil(batch * context / (budget - batch + 1)) + 1
            max_tokens = (
                build_steps + limits.decode_warmup_steps + limits.decode_measured_steps + 1
            )
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
) -> list[DsaCase]:
    """Uniform native per-request chunks keep every sequence in each step."""
    repeats = limits.prefill_warmup_repeats + limits.prefill_measured_repeats
    cases = []
    for batch in build_batch_grid(min(limits.max_num_seqs, budget // query)):
        if batch == 1:
            continue
        prompt = min(limits.max_context_len - 1,
                     limits.max_kv_tokens_per_lane // batch - 1)
        # Whole chunks keep the final step on the same sparse-prefill path.
        prompt = prompt // query * query
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


def build_dsa_profile_plan(
    *,
    model_name: str,
    hf_config: Mapping[str, Any],
    target_tensor_parallel_size: int,
    num_layers: int,
    limits: DsaMeasurementLimits,
) -> DsaProfilePlan:
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


def load_hf_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "DSA_SCOPES",
    "DsaCase",
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
    "resolve_reduced_layers",
]
