"""CPU-only measurement plan for DeepSeek-V4 compressed attention on vLLM.

DeepSeek-V4 attention has three layer variants that share one sliding-window
(SWA) KV cache and differ in their compressed KV path:

* ``uncompressed``: SWA attention only;
* ``hca``: a compressor writes one KV entry per ``hca`` ratio tokens and the
  sparse attention reads every compressed entry plus the SWA window;
* ``csa``: a compressor writes one KV entry per ``csa`` ratio tokens and a
  lightning indexer selects ``index_topk`` of them for the sparse attention.

The kernels depend on the engine's attention metadata, cache layout, and
per-layer-type tile-scheduler planning, so they are measured inside a real
vLLM engine. As in the DSA plan, everything the GPU run needs is fixed here:

* a reduced model with the leading decoder layers of the native layer order,
  enough to hold two layers of every variant, so each variant has the layer
  that plans the decode tile schedule and a layer that reuses the plan; the
  attention head count and output groups are the per-rank values of the
  target tensor-parallel size;
* one engine per prefill chunk budget, with the DSA case builders.

The context grid adds the indexer's short-context boundary: while
``max_seq_len // csa_ratio <= index_topk`` every compressed entry is selected
and vLLM skips index scoring and top-k.

The plan imports no GPU package, so it runs in the dry-run check on a CPU host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from frontier.attention.families import (
    DEEPSEEK_V4_ATTENTION_FAMILY,
    DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT,
)
from frontier.profiling.attention.dsa_plan import (
    PHASE_DECODE,
    PHASE_PREFILL,
    DsaEngineSpec,
    DsaMeasurementLimits,
    DsaProfilePlan,
    _build_decode_cases,
    _build_prefill_cases,
    _require_positive_int,
    _required_config_int,
    _required_config_str,
    build_context_grid,
)
from frontier.profiling.common.model_config import ModelConfig

PLAN_SCHEMA_VERSION = "dsv4_attention_plan.v1"

UNCOMPRESSED_VARIANT, CSA_VARIANT, HCA_VARIANT = (
    DEEPSEEK_V4_ATTENTION_FAMILY.supported_variants
)
# vLLM builds the lightning indexer only on layers with this compress ratio
# ("Only C4A uses sparse attention and hence has indexer").
INDEXER_COMPRESS_RATIO = 4
# Each variant needs a tile-schedule planning layer and a reusing layer.
_LAYERS_PER_VARIANT = 2
_PER_LAYER_KEY = "compress_ratios"

LAYER_SCOPE = "dsv4_attention_layer"

# Canonical family operators reached only when no attention metadata exists
# (vLLM memory profiling and dummy runs), never in a measured step.
WARMUP_ONLY_OPERATORS = ("dsv4_query_padding", "dsv4_attention_output_zero")

# Native attention metadata each variant reads, by source.
METADATA_SOURCES_BY_VARIANT = {
    UNCOMPRESSED_VARIANT: ("swa",),
    HCA_VARIANT: ("swa", "attention", "compressor"),
    CSA_VARIANT: ("swa", "attention", "compressor", "indexer", "index_compressor"),
}


@dataclass(frozen=True)
class Dsv4Scope:
    """One timed call of a DeepSeek-V4 attention layer.

    ``operators`` lists the canonical family operators whose kernels run in
    the call. ``annotated=False`` marks the enclosing layer scope, which has
    CUDA-event wall time only. ``long_context_only`` scopes run only when the
    indexer scores and selects compressed entries.
    """

    name: str
    variants: tuple[str, ...]
    operators: tuple[str, ...]
    annotated: bool = True
    long_context_only: bool = False


_ALL_VARIANTS = DEEPSEEK_V4_ATTENTION_FAMILY.supported_variants
_COMPRESSED_VARIANTS = (CSA_VARIANT, HCA_VARIANT)

DSV4_SCOPES = (
    Dsv4Scope(LAYER_SCOPE, _ALL_VARIANTS, (), annotated=False),
    # One call launches the fused q/kv GEMM on the default stream and the
    # compressor and indexer input GEMMs on aux streams.
    Dsv4Scope(
        "dsv4_parallel_input_projections",
        _ALL_VARIANTS,
        (
            "dsv4_fused_qkv_projection",
            "dsv4_compressor_projection",
            "dsv4_index_weights_projection",
            "dsv4_index_compressor_projection",
        ),
    ),
    Dsv4Scope("dsv4_fused_qkv_rmsnorm", _ALL_VARIANTS, ("dsv4_fused_qkv_rmsnorm",)),
    Dsv4Scope("dsv4_q_projection", _ALL_VARIANTS, ("dsv4_q_projection",)),
    Dsv4Scope("dsv4_fused_q_kv_prepare", _ALL_VARIANTS, ("dsv4_fused_q_kv_prepare",)),
    Dsv4Scope(
        "dsv4_compressor",
        _COMPRESSED_VARIANTS,
        ("dsv4_compressor_save_partial_states", "dsv4_compressor_fused_compress_store"),
    ),
    # The indexer forward: on short contexts the index compressor plus the
    # short-context top-k fill; otherwise the index compressor plus the index
    # query projection and its fused RoPE/quantization.
    Dsv4Scope(
        "dsv4_index_prepare",
        (CSA_VARIANT,),
        (
            "dsv4_index_compressor_save_partial_states",
            "dsv4_index_compressor_fused_compress_store",
            "dsv4_index_query_projection",
            "dsv4_index_fused_query_prepare",
            "dsv4_index_short_context_fill",
        ),
    ),
    Dsv4Scope(
        "dsv4_index_scores_topk",
        (CSA_VARIANT,),
        ("dsv4_index_scores", "dsv4_index_topk"),
        long_context_only=True,
    ),
    Dsv4Scope("dsv4_sparse_attention", _ALL_VARIANTS, ("dsv4_sparse_attention",)),
    Dsv4Scope(
        "dsv4_o_proj",
        _ALL_VARIANTS,
        (
            "dsv4_output_inv_rope_quant",
            "dsv4_grouped_output_projection",
            "dsv4_output_projection",
        ),
    ),
)
DSV4_SCOPE_NAMES = tuple(scope.name for scope in DSV4_SCOPES)
_SCOPES_BY_NAME = {scope.name: scope for scope in DSV4_SCOPES}


def scope_operators(scope_name: str, variant: str) -> tuple[str, ...]:
    """Canonical operators of ``variant`` whose kernels run in the scope."""

    variant_operators = {
        operator.name for operator in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[variant]
    }
    return tuple(
        name for name in _SCOPES_BY_NAME[scope_name].operators if name in variant_operators
    )


def expected_scopes(
    variant: str, index_scores_executed: bool, annotated_only: bool
) -> tuple[str, ...]:
    """Scopes one layer of ``variant`` runs in one measured step."""

    return tuple(
        scope.name
        for scope in DSV4_SCOPES
        if variant in scope.variants
        and (index_scores_executed or not scope.long_context_only)
        and (scope.annotated or not annotated_only)
    )


def metadata_key(variant: str, source: str) -> str:
    return f"{variant}_{source}_metadata"


@dataclass(frozen=True)
class Dsv4Layer:
    global_layer_id: int
    attention_variant: str
    compress_ratio: int  # Native ``compress_ratios`` entry.
    plans_tile_schedule: bool  # First layer of its variant; plans decode tiles.


@dataclass(frozen=True)
class Dsv4ModelShape:
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
    o_lora_rank: int
    o_groups: int
    o_groups_per_rank: int
    head_dim: int
    qk_rope_head_dim: int
    sliding_window: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    csa_compress_ratio: int
    hca_compress_ratio: int
    # Largest indexer ``max_seq_len`` that selects every compressed entry.
    short_context_max_seq_len: int
    layers: tuple[Dsv4Layer, ...]

    def index_scores_executed(self, indexer_max_seq_len: int) -> bool:
        return indexer_max_seq_len > self.short_context_max_seq_len

    def step_metadata_keys(self) -> tuple[str, ...]:
        variants = dict.fromkeys(layer.attention_variant for layer in self.layers)
        return tuple(
            metadata_key(variant, source)
            for variant in variants
            for source in METADATA_SOURCES_BY_VARIANT[variant]
        )


@dataclass(frozen=True)
class Dsv4ProfilePlan(DsaProfilePlan):
    """DSA plan structure with the DeepSeek-V4 shape and KV-cache dtype."""

    shape: Dsv4ModelShape
    kv_cache_dtype: str

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["reduced_layers"] = [
            (layer.global_layer_id, layer.attention_variant, layer.plans_tile_schedule)
            for layer in self.shape.layers
        ]
        summary["short_context_max_seq_len"] = self.shape.short_context_max_seq_len
        summary["kv_cache_dtype"] = self.kv_cache_dtype
        return summary


def _native_compress_ratios(hf_config: Mapping[str, Any]) -> list[int]:
    total = _required_config_int(hf_config, "num_hidden_layers")
    ratios = hf_config.get(_PER_LAYER_KEY)
    allowed = {total, total + hf_config.get("num_nextn_predict_layers", 0)}
    if not isinstance(ratios, list) or len(ratios) not in allowed:
        raise ValueError(
            f"model config {_PER_LAYER_KEY} must list one ratio per decoder layer "
            f"(optionally followed by the MTP layers), got {ratios!r}"
        )
    return ratios


def _variant_of_ratio(ratio: int) -> str:
    if type(ratio) is not int or ratio < 0:
        raise ValueError(f"compress ratio must be a non-negative int, got {ratio!r}")
    if max(1, ratio) == 1:
        return UNCOMPRESSED_VARIANT
    if ratio == INDEXER_COMPRESS_RATIO:
        return CSA_VARIANT
    return HCA_VARIANT


def resolve_dsv4_layers(
    model_config: ModelConfig, hf_config: Mapping[str, Any], num_layers: int | None
) -> tuple[Dsv4Layer, ...]:
    """Take the leading layers; by default the fewest holding two of every variant."""

    specs = model_config.get_layer_attention_specs()
    ratios = _native_compress_ratios(hf_config)
    if len(specs) != _required_config_int(hf_config, "num_hidden_layers"):
        raise ValueError(
            f"Frontier lists {len(specs)} attention layers; the model config has "
            f"{hf_config['num_hidden_layers']}"
        )
    family_id = DEEPSEEK_V4_ATTENTION_FAMILY.family_id
    for spec in specs:
        if spec.family_id != family_id:
            raise ValueError(
                f"Layer {spec.global_layer_id} uses attention family "
                f"{spec.family_id!r}, not {family_id!r}"
            )
        native = _variant_of_ratio(ratios[spec.global_layer_id])
        if native != spec.variant_id:
            raise ValueError(
                f"Layer {spec.global_layer_id}: native compress ratio "
                f"{ratios[spec.global_layer_id]} is {native!r}, Frontier says "
                f"{spec.variant_id!r}"
            )
    if num_layers is None:
        counts = dict.fromkeys(_ALL_VARIANTS, 0)
        for num_layers, spec in enumerate(specs, start=1):
            counts[spec.variant_id] += 1
            if min(counts.values()) >= _LAYERS_PER_VARIANT:
                break
    _require_positive_int("num_layers", num_layers)
    if num_layers > len(specs):
        raise ValueError(f"num_layers={num_layers} exceeds the model's {len(specs)} layers")
    layers = []
    planned: set[str] = set()
    for spec in specs[:num_layers]:
        layers.append(
            Dsv4Layer(
                global_layer_id=spec.global_layer_id,
                attention_variant=spec.variant_id,
                compress_ratio=ratios[spec.global_layer_id],
                plans_tile_schedule=spec.variant_id not in planned,
            )
        )
        planned.add(spec.variant_id)
    for variant in _ALL_VARIANTS:
        found = sum(layer.attention_variant == variant for layer in layers)
        if found < _LAYERS_PER_VARIANT:
            raise ValueError(
                f"The reduced model needs {_LAYERS_PER_VARIANT} {variant!r} layers "
                f"(a tile-schedule planning and a reusing layer); num_layers="
                f"{num_layers} gives {found}"
            )
    return tuple(layers)


def _single_ratio(layers: tuple[Dsv4Layer, ...], variant: str) -> int:
    ratios = {layer.compress_ratio for layer in layers if layer.attention_variant == variant}
    if len(ratios) != 1:
        raise ValueError(f"{variant!r} layers use compress ratios {sorted(ratios)}")
    return ratios.pop()


def build_dsv4_hf_overrides(
    hf_config: Mapping[str, Any],
    layers: tuple[Dsv4Layer, ...],
    heads_per_rank: int,
    o_groups_per_rank: int,
) -> dict[str, Any]:
    """Shrink the native config to the measured layers and per-rank heads.

    The indexer heads and the MoE layers stay native: vLLM replicates the
    indexer on every rank, and the expert layers do not affect attention.
    """

    total = _required_config_int(hf_config, "num_hidden_layers")
    per_layer_lengths = {total, total + hf_config.get("num_nextn_predict_layers", 0)}
    for key, value in hf_config.items():
        if isinstance(value, list) and len(value) in per_layer_lengths and key != _PER_LAYER_KEY:
            raise ValueError(
                f"Model config key {key!r} lists one value per layer; the reduced "
                "model does not know how to shrink it"
            )
    num_layers = len(layers)
    return {
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads_per_rank,
        "o_groups": o_groups_per_rank,
        _PER_LAYER_KEY: list(_native_compress_ratios(hf_config)[:num_layers]),
    }


def _per_rank(hf_config: Mapping[str, Any], key: str, tensor_parallel_size: int) -> int:
    value = _required_config_int(hf_config, key)
    if value % tensor_parallel_size:
        raise ValueError(
            f"{key}={value} is not divisible by tensor_parallel_size={tensor_parallel_size}"
        )
    return value // tensor_parallel_size


def build_dsv4_profile_plan(
    *,
    model_name: str,
    hf_config: Mapping[str, Any],
    target_tensor_parallel_size: int,
    num_layers: int | None,
    limits: DsaMeasurementLimits,
    kv_cache_dtype: str,
) -> Dsv4ProfilePlan:
    model_config = ModelConfig.from_model_name(model_name)
    family = model_config.get_attention_family()
    if family.family_id != DEEPSEEK_V4_ATTENTION_FAMILY.family_id:
        raise ValueError(
            f"{model_name} resolves attention family {family.family_id!r}, "
            f"not {DEEPSEEK_V4_ATTENTION_FAMILY.family_id!r}"
        )
    _require_positive_int("target_tensor_parallel_size", target_tensor_parallel_size)
    if not kv_cache_dtype:
        raise ValueError("kv_cache_dtype must be the target deployment's KV-cache dtype")
    layers = resolve_dsv4_layers(model_config, hf_config, num_layers)
    csa_ratio = _single_ratio(layers, CSA_VARIANT)
    index_topk = _required_config_int(hf_config, "index_topk")
    shape = Dsv4ModelShape(
        model_name=model_name,
        model_type=_required_config_str(hf_config, "model_type"),
        model_arch=str(model_config.model_arch),
        model_architecture_profile=model_config.get_model_architecture_profile().profile_id,
        dtype=_required_config_str(hf_config, "torch_dtype"),
        total_num_layers=_required_config_int(hf_config, "num_hidden_layers"),
        num_attention_heads=_required_config_int(hf_config, "num_attention_heads"),
        target_tensor_parallel_size=target_tensor_parallel_size,
        heads_per_rank=_per_rank(hf_config, "num_attention_heads", target_tensor_parallel_size),
        hidden_size=_required_config_int(hf_config, "hidden_size"),
        q_lora_rank=_required_config_int(hf_config, "q_lora_rank"),
        o_lora_rank=_required_config_int(hf_config, "o_lora_rank"),
        o_groups=_required_config_int(hf_config, "o_groups"),
        o_groups_per_rank=_per_rank(hf_config, "o_groups", target_tensor_parallel_size),
        head_dim=_required_config_int(hf_config, "head_dim"),
        qk_rope_head_dim=_required_config_int(hf_config, "qk_rope_head_dim"),
        sliding_window=_required_config_int(hf_config, "sliding_window"),
        index_topk=index_topk,
        index_n_heads=_required_config_int(hf_config, "index_n_heads"),
        index_head_dim=_required_config_int(hf_config, "index_head_dim"),
        csa_compress_ratio=csa_ratio,
        hca_compress_ratio=_single_ratio(layers, HCA_VARIANT),
        short_context_max_seq_len=(index_topk + 1) * csa_ratio - 1,
        layers=layers,
    )
    chunk_budgets = tuple(sorted(limits.chunk_budgets))
    context_grid = build_context_grid(
        limits.block_size, shape.short_context_max_seq_len, limits.max_context_len
    )
    # The largest budget runs the decode cases.
    cases = _build_prefill_cases(chunk_budgets, limits, context_grid)
    cases += _build_decode_cases(
        len(chunk_budgets) - 1, chunk_budgets[-1], limits, context_grid
    )
    engines = tuple(
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
    )
    return Dsv4ProfilePlan(
        schema_version=PLAN_SCHEMA_VERSION,
        shape=shape,
        limits=limits,
        hf_overrides=build_dsv4_hf_overrides(
            hf_config, layers, shape.heads_per_rank, shape.o_groups_per_rank
        ),
        engines=engines,
        cases=tuple(cases),
        kv_cache_dtype=kv_cache_dtype,
    )


__all__ = [
    "CSA_VARIANT",
    "DSV4_SCOPES",
    "DSV4_SCOPE_NAMES",
    "Dsv4Layer",
    "Dsv4ModelShape",
    "Dsv4ProfilePlan",
    "Dsv4Scope",
    "HCA_VARIANT",
    "INDEXER_COMPRESS_RATIO",
    "LAYER_SCOPE",
    "METADATA_SOURCES_BY_VARIANT",
    "PHASE_DECODE",
    "PHASE_PREFILL",
    "UNCOMPRESSED_VARIANT",
    "WARMUP_ONLY_OPERATORS",
    "build_dsv4_hf_overrides",
    "build_dsv4_profile_plan",
    "expected_scopes",
    "metadata_key",
    "resolve_dsv4_layers",
    "scope_operators",
]
