from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.attention.gdn.config import (
    LayerAttentionSpec,
    is_qwen3_5_profile_config,
    GatedDeltaNetConfig,
    SequenceMixerType,
    build_sequence_mixer_schedule,
    resolve_gdn_shape,
)
from frontier.attention.families import GATED_DELTA_NET_ATTENTION_FAMILY, get_attention_family


_DSA_MODEL_TYPE_MARKERS = (
    "deepseek_v32",
    "deepseek_v3_2",
    "deepseek_v3.2",
    "glm_moe_dsa",
)
_DSA_FIELD_MARKERS = (
    "dsa_topk",
    "dsa_top_k",
    "dsa_index_topk",
    "dsa_indexer",
    "index_topk",
    "index_topk_freq",
    "index_topk_pattern",
    "index_skip_topk_offset",
)
_UNKNOWN_EXOTIC_FIELD_MARKERS = (
    "sliding_window_pattern",
    "dual_chunk_attention",
    "attention_chunk_size",
)
_REQUIRED_MLA_FIELDS = (
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)

@dataclass(frozen=True)
class AttentionFamilyBinding:
    """Concrete attention family selected for one model configuration."""

    family_id: str
    variant_id: str
    frozen: bool
    reason: str

    @property
    def family(self):
        return get_attention_family(self.family_id)

    def require_enabled_for_execution(self) -> None:
        self.family.require_enabled_for_execution()


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _has_truthy_attr(config: Any, name: str) -> bool:
    return bool(_get_attr(config, name, None))


def _model_type(config: Any) -> str:
    raw_model_type = _get_attr(config, "model_type", None)
    return str(raw_model_type or "").lower()


def _has_dsa_marker(config: Any) -> bool:
    model_type = _model_type(config)
    if any(marker in model_type for marker in _DSA_MODEL_TYPE_MARKERS):
        return True
    return any(_has_truthy_attr(config, marker) for marker in _DSA_FIELD_MARKERS)


def _unknown_exotic_fields(config: Any) -> list[str]:
    return [
        marker
        for marker in _UNKNOWN_EXOTIC_FIELD_MARKERS
        if _has_truthy_attr(config, marker)
    ]


def _validate_mla_fields(config: Any) -> None:
    missing = [
        field_name
        for field_name in _REQUIRED_MLA_FIELDS
        if _get_attr(config, field_name, None) is None
    ]
    if missing:
        raise ValueError(f"MLA attention binding requires fields: {missing}")


def _validate_mfa_fields(config: Any) -> None:
    missing = []
    if _get_attr(config, "share_q_dim", None) is None:
        missing.append("share_q_dim")
    if (
        _get_attr(config, "head_dim", None) is None
        and _get_attr(config, "_head_dim", None) is None
    ):
        missing.append("head_dim")
    if missing:
        raise ValueError(f"MFA attention binding requires fields: {missing}")


def resolve_attention_head_partition(
    total_heads: int, tensor_parallel_size: int, *, name: str = "KV heads",
) -> tuple[int, int]:
    """Return local heads and replication for the shared attention TP contract."""
    if type(total_heads) is not int or total_heads <= 0:
        raise ValueError(f"{name} must be a positive int, got {total_heads!r}")
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive int")
    if total_heads >= tensor_parallel_size and total_heads % tensor_parallel_size == 0:
        return total_heads // tensor_parallel_size, 1
    if total_heads < tensor_parallel_size and tensor_parallel_size % total_heads == 0:
        return 1, tensor_parallel_size // total_heads
    raise ValueError(f"{name} and tensor_parallel_size must divide one another")


def _dense_variant_from_heads(num_q_heads: int, num_kv_heads: int) -> str:
    reason = f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "Attention head counts must be positive: "
            f"{reason}"
        )
    if num_kv_heads == num_q_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    if 1 < num_kv_heads < num_q_heads:
        return "gqa"
    raise ValueError(
        "Unsupported attention head topology: "
        f"{reason}"
    )


def bind_attention_family(config: Any) -> AttentionFamilyBinding:
    """Bind a runtime or profiling model config to an attention family.

    Dense-FFN and MoE-FFN are intentionally ignored here. This rule engine only
    classifies the attention topology/cache family.
    """
    # A hybrid Qwen3.5 model has no single whole-model attention family.  Keep
    # this legacy API available for homogeneous consumers and fail before any
    # caller can accidentally select the first/majority layer.
    if _has_hybrid_attention_schedule(config):
        raise ValueError(
            "Hybrid attention configuration requires an explicit global layer id; "
            "call bind_layer_attention(config, global_layer_id)"
        )

    if _model_type(config) == "step5":
        raise ValueError(
            "Step attention requires an explicit global layer id; "
            "call bind_layer_attention(config, global_layer_id)"
        )

    if _get_attr(config, "compress_ratios") is not None:
        variants = {spec.variant_id for spec in config.get_layer_attention_specs()}
        if len(variants) > 1:
            raise ValueError(
                "Mixed compression variants require an explicit global layer id; "
                "call bind_layer_attention(config, global_layer_id)"
            )

    return _bind_homogeneous_attention_family(config)


def _bind_homogeneous_attention_family(config: Any) -> AttentionFamilyBinding:
    """Classify homogeneous topology without consulting layer resolution."""
    if _model_type(config) == "step5":
        from frontier.attention.step5 import validate_step5_config

        validate_step5_config(config)
        return AttentionFamilyBinding(
            family_id="step5_attention", variant_id="sparse_gqa", frozen=True,
            reason="Step per-layer SWA/sparse GQA shares raw KV layout; execution is unqualified",
        )
    if (
        _model_type(config) == "deepseek_v4"
        or _get_attr(config, "compress_ratios") is not None
    ):
        return AttentionFamilyBinding(
            family_id="deepseek_v4_attention", variant_id="uncompressed",
            frozen=True,
            reason="Compressed attention requires explicit execution coverage",
        )
    if _has_dsa_marker(config):
        return AttentionFamilyBinding(
            family_id="dsa_attention",
            variant_id="dsa",
            frozen=True,
            reason="DSA marker detected; truth backend remains frozen",
        )

    if bool(_get_attr(config, "use_mla", False)) and bool(
        _get_attr(config, "use_mfa", False)
    ):
        raise ValueError("use_mla and use_mfa are mutually exclusive")

    if bool(_get_attr(config, "use_mla", False)):
        _validate_mla_fields(config)
        return AttentionFamilyBinding(
            family_id="latent_mla_attention",
            variant_id="mla",
            frozen=False,
            reason="use_mla=True with required latent cache fields",
        )

    if bool(_get_attr(config, "use_mfa", False)):
        _validate_mfa_fields(config)
        num_q_heads = int(_get_attr(config, "num_q_heads"))
        num_kv_heads = int(_get_attr(config, "num_kv_heads"))
        if num_kv_heads != 1:
            raise ValueError(
                "use_mfa=True requires num_kv_heads=1 for Step3Text dense-KV "
                f"MFA topology, got num_kv_heads={num_kv_heads}"
            )
        variant_id = _dense_variant_from_heads(num_q_heads, num_kv_heads)
        return AttentionFamilyBinding(
            family_id="dense_attention",
            variant_id=variant_id,
            frozen=False,
            reason=(
                "use_mfa=True with Step3Text dense-KV attention topology; "
                f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
            ),
        )

    exotic_fields = _unknown_exotic_fields(config)
    if exotic_fields:
        raise ValueError(
            "Unrecognized exotic attention fields require an explicit family "
            f"binding before execution: {exotic_fields}"
        )

    num_q_heads = int(_get_attr(config, "num_q_heads"))
    num_kv_heads = int(_get_attr(config, "num_kv_heads"))
    reason = f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"

    variant_id = _dense_variant_from_heads(num_q_heads, num_kv_heads)

    return AttentionFamilyBinding(
        family_id="dense_attention",
        variant_id=variant_id,
        frozen=False,
        reason=reason,
    )


def resolve_runtime_attention_family(config: Any):
    """Resolve the family needed by model-wide runtime cache/layout helpers.

    ``bind_attention_family`` intentionally rejects a hybrid model because a
    whole-model prediction cannot choose between GDN and full attention.  A
    runtime KV-cache layout is different: only full-attention layers allocate
    KV pages, so hybrid models may use their unique full-attention family for
    model-wide head/layout metadata.  Per-layer execution must continue to use
    ``bind_layer_attention``.
    """

    if not _has_hybrid_attention_schedule(config):
        return _bind_homogeneous_attention_family(config).family

    specs = config.get_layer_attention_specs()
    full_families = tuple(
        spec.family_id for spec in specs if bool(spec.is_full_attention)
    )
    if not full_families:
        raise ValueError(
            "Hybrid runtime attention metadata requires at least one "
            "full-attention layer"
        )
    family_ids = tuple(dict.fromkeys(full_families))
    if len(family_ids) != 1:
        raise ValueError(
            "Hybrid runtime cache/layout metadata requires one full-attention "
            f"family, got {family_ids}"
        )
    return get_attention_family(family_ids[0])


def _has_hybrid_attention_schedule(config: Any) -> bool:
    """Return whether the supported profile contains any GDN layer."""

    # The profile predicate is the classification boundary.  Once a config
    # explicitly selects Qwen3.5, schedule/shape errors must propagate so a
    # malformed supported configuration cannot silently fall through to the
    # homogeneous dense binder.
    if not is_qwen3_5_profile_config(config):
        return False
    specs = config.get_layer_attention_specs()
    return any(spec.is_gdn for spec in specs)


def bind_layer_attention(config: Any, global_layer_id: int) -> LayerAttentionSpec:
    """Bind one global decoder layer to its attention family and variant."""

    if type(global_layer_id) is not int:
        raise ValueError(
            f"global_layer_id must be an int, got {global_layer_id!r}"
        )
    spec = config.get_layer_attention_spec(global_layer_id)
    get_attention_family(spec.family_id)
    return spec


def resolve_attention_topology(
    config: Any,
) -> tuple[tuple[LayerAttentionSpec, ...], GatedDeltaNetConfig | None]:
    """Normalize external topology and GDN shape once at the model boundary."""
    num_layers = config.num_layers
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("config must declare a positive num_layers")
    if _model_type(config) == "step5":
        from frontier.attention.step5 import STEP5_ATTENTION_VARIANTS, validate_step5_config

        validate_step5_config(config)
        return tuple(
            LayerAttentionSpec(layer_id, "step5_attention", STEP5_ATTENTION_VARIANTS[layer_type])
            for layer_id, layer_type in enumerate(config.layer_types)
        ), None
    binding = _bind_homogeneous_attention_family(config)
    if binding.family_id == "deepseek_v4_attention":
        ratios = _get_attr(config, "compress_ratios")
        if not isinstance(ratios, (tuple, list)) or len(ratios) < num_layers:
            raise ValueError("compress_ratios must cover every backbone layer")
        variants = {0: "uncompressed", 4: "csa", 128: "hca"}
        if any(
            type(ratio) is not int or ratio not in variants
            for ratio in ratios[:num_layers]
        ):
            raise ValueError("compress_ratios supports only 0, 4 and 128")
        return tuple(
            LayerAttentionSpec(layer_id, binding.family_id, variants[ratios[layer_id]])
            for layer_id in range(num_layers)
        ), None
    if (
        binding.family_id == "dsa_attention"
        and _get_attr(config, "index_topk") is not None
    ):
        return _resolve_index_attention_specs(config), None
    if not is_qwen3_5_profile_config(config):
        return tuple(
            LayerAttentionSpec(layer_id, binding.family_id, binding.variant_id)
            for layer_id in range(num_layers)
        ), None

    gdn_config = resolve_gdn_shape(config)
    schedule = build_sequence_mixer_schedule(
        num_layers=num_layers,
        layer_types=config.layer_types,
        full_attention_interval=config.full_attention_interval,
        has_gated_delta_net=gdn_config is not None,
    )
    gdn_variant, = GATED_DELTA_NET_ATTENTION_FAMILY.supported_variants
    return tuple(
        LayerAttentionSpec(layer_id, GATED_DELTA_NET_ATTENTION_FAMILY.family_id, gdn_variant)
        if mixer is SequenceMixerType.GATED_DELTA_NET
        else LayerAttentionSpec(layer_id, binding.family_id, binding.variant_id)
        for layer_id, mixer in enumerate(schedule)
    ), gdn_config


def _resolve_index_attention_specs(config: Any) -> tuple[LayerAttentionSpec, ...]:
    """Resolve native backbone index reuse without claiming observed execution."""

    for field_name in ("index_topk", "index_n_heads", "index_head_dim"):
        value = _get_attr(config, field_name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be a positive int")
    frequency = _get_attr(config, "index_topk_freq")
    frequency = 1 if frequency is None else frequency
    offset = _get_attr(config, "index_skip_topk_offset")
    offset = 2 if offset is None else offset
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("index_topk_freq must be a positive int")
    if type(offset) is not int or offset < 0:
        raise ValueError("index_skip_topk_offset must be a non-negative int")
    pattern = _get_attr(config, "index_topk_pattern")
    if pattern is not None and (
        not isinstance(pattern, str) or any(value not in "FS" for value in pattern)
    ):
        raise ValueError("index_topk_pattern must contain only F and S")
    specs = []
    producer = None
    for layer_id in range(config.num_layers):
        shared = (
            layer_id < len(pattern) and pattern[layer_id] == "S"
            if pattern is not None
            else max(layer_id - offset + 1, 0) % frequency != 0
        )
        if not shared:
            producer = layer_id
        if producer is None:
            raise ValueError(f"Shared index layer {layer_id} has no index producer")
        specs.append(LayerAttentionSpec(
            layer_id, "dsa_attention",
            "shared_index" if shared else "full_index", producer,
        ))
    return tuple(specs)


def resolve_layer_attention_specs(config: Any) -> tuple[LayerAttentionSpec, ...]:
    """Resolve external configuration using the authoritative topology rules."""
    return resolve_attention_topology(config)[0]
