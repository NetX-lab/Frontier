"""Profiling plan construction for decoupled attention/FFN TP profiling."""

from __future__ import annotations

from typing import Dict, List, Sequence

from frontier.attention.model_binding import bind_attention_family
from frontier.operators.families import (
    FFN_FAMILY,
    MEMORY_FAMILY,
    SHARE_EXPERT_FAMILY,
    get_operator_family,
    get_family_profiling_names,
)
from frontier.model_architectures import get_model_architecture_profile
from frontier.operators.spec import TensorParallelMode
from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_method_contract,
    get_target_embedded_mtp_methods,
    get_target_embedded_mtp_linear_ops,
    get_target_embedded_mtp_same_tp_linear_ops,
)

TARGET_EMBEDDED_MTP_OPS = list(get_target_embedded_mtp_linear_ops())


def _dedupe_preserving_order(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _memory_profiling_names(model_config) -> List[str]:
    operators = []
    for operator in MEMORY_FAMILY.profiling_ops():
        if (
            operator.name == "post_attention_layernorm"
            and not getattr(model_config, "post_attn_norm", False)
        ):
            continue
        operators.append(operator)
    return _dedupe_preserving_order(
        [operator.profiling_name() for operator in operators]
    )


def _ffn_profiling_names() -> List[str]:
    return list(get_family_profiling_names(FFN_FAMILY))


def _share_expert_profiling_names() -> List[str]:
    return list(get_family_profiling_names(SHARE_EXPERT_FAMILY))


def _typed_linear_profiling_names(
    typed_layer_contracts: Sequence[Dict[str, object]],
    *,
    selected_tp_size: int | None = None,
) -> List[str]:
    """Return linear FFN names owned by active typed layer contracts.

    Routed MoE operators have a separate profiling producer. Restrict this
    helper to the FFN and shared-expert families, while deriving ownership from
    the profile declarations instead of the model's legacy ``is_moe`` flag.
    """

    linear_family_ids = {
        FFN_FAMILY.family_id,
        SHARE_EXPERT_FAMILY.family_id,
    }
    names: List[str] = []
    for contract in typed_layer_contracts:
        if selected_tp_size is not None and contract.get(
            "selected_tensor_parallel_size"
        ) != selected_tp_size:
            continue
        for family_id in contract.get("operator_family_ids", ()):
            if family_id not in linear_family_ids:
                continue
            family = get_operator_family(str(family_id))
            names.extend(get_family_profiling_names(family))
    return _dedupe_preserving_order(names)


def memory_operator_enabled(
    enabled_ops: Sequence[str] | set[str] | None,
    operator_name: str,
) -> bool:
    if enabled_ops is None:
        return True
    enabled_op_set = set(enabled_ops)
    for operator in MEMORY_FAMILY.profiling_ops():
        if operator.name == operator_name:
            return operator.profiling_name() in enabled_op_set
    raise ValueError(f"Unknown MEMORY profiling operator: {operator_name}")


def _bool_config_value(model_config, name: str) -> bool:
    value = getattr(model_config, name, False)
    if callable(value):
        value = value()
    return bool(value)


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _validate_positive_int(value: object, *, name: str) -> int:
    """Validate one strict positive integer used by a profiling plan."""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _normalize_tp_domain(values: Sequence[int], *, name: str) -> List[int]:
    """Normalize a TP domain without coercing booleans or string values."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain positive integer TP sizes, got {values!r}")
    try:
        values_list = list(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain positive integer TP sizes, got {values!r}"
        ) from exc

    normalized: List[int] = []
    for value in values_list:
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{name} must contain positive integer TP sizes, got {value!r}"
            )
        normalized.append(value)
    return sorted(set(normalized))


def _target_embedded_mtp_family_id() -> str:
    """Resolve the target-embedded MTP family ID from its registry."""

    family_ids = {
        str(get_target_embedded_mtp_method_contract(method)["mtp_family"])
        for method in get_target_embedded_mtp_methods()
    }
    if len(family_ids) != 1:
        raise ValueError(
            "Target-embedded MTP registry must expose exactly one family ID, "
            f"got {sorted(family_ids)}"
        )
    return next(iter(family_ids))


def _supports_share_expert(model_config) -> bool:
    if not hasattr(model_config, "supports_share_expert"):
        raise TypeError("linear-op profiling requires model_config.supports_share_expert()")
    return model_config.supports_share_expert()


def _typed_layer_contracts(
    model_config,
    *,
    profile,
    tp_size: int,
    attn_tp: Sequence[int],
    ffn_tp: Sequence[int],
    moe_tp: Sequence[int],
) -> List[Dict[str, object]]:
    """Materialize active profile-owned FFN contracts for one plan.

    A linear-op plan may be evaluated for a TP size that belongs to only one
    semantic domain. Keep the complete configured TP domain in the metadata and
    mark the current size as selected only when that domain owns it. This lets
    the wrapper build the matching dense/shared module without collapsing
    routed and dense widths into the legacy scalar field.
    """

    mixed_model = bool(
        getattr(profile, "_is_mixed_model", lambda _config: False)(model_config)
    )
    domain_sizes = {
        TensorParallelMode.REPLICATED: [1],
        TensorParallelMode.ATTENTION_TP: _normalize_tp_domain(
            attn_tp, name="attn_tp"
        ),
        TensorParallelMode.FFN_TP: _normalize_tp_domain(ffn_tp, name="ffn_tp"),
        TensorParallelMode.MOE_TP: _normalize_tp_domain(moe_tp, name="moe_tp"),
    }

    contracts: List[Dict[str, object]] = []
    for spec in profile.iter_active_layer_contracts(model_config):
        width = spec.resolve_width(model_config, mixed_model=mixed_model)
        sizes = domain_sizes[spec.tensor_parallel_mode]
        selected_size = None
        if not (
            getattr(model_config, "no_tensor_parallel", False)
            and spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED
        ):
            selected_size = tp_size if tp_size in sizes else None
        if (
            selected_size is not None
            and spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED
            and width % selected_size != 0
        ):
            raise ValueError(
                f"{spec.layer_kind.value} width {width} must be divisible by "
                f"selected tensor parallel size {selected_size}"
            )
        contracts.append(
            {
                "profile_id": profile.profile_id,
                "layer_kind": spec.layer_kind.value,
                "dimension_source": spec.dimension_source.value,
                "effective_ffn_width": width,
                "tensor_parallel_mode": spec.tensor_parallel_mode.value,
                "expert_parallel_mode": spec.expert_parallel_mode.value,
                # Linear-op planning does not select an EP domain. The
                # dedicated MoE producer supplies concrete EP identities.
                "selected_expert_parallel_size": None,
                "operator_family_ids": list(spec.operator_family_ids),
                "tensor_parallel_sizes": list(sizes),
                "selected_tensor_parallel_size": selected_size,
                "selected_padded_ffn_width": (
                    _pad_to_multiple(width, selected_size)
                    if selected_size is not None
                    else None
                ),
            }
        )
    return contracts


def _non_layer_operator_contract(
    *,
    profile_id: str,
    operator_family_id: str,
    tensor_parallel_mode: TensorParallelMode,
    tensor_parallel_sizes: Sequence[int],
    selected_tensor_parallel_size: int | None,
) -> Dict[str, object]:
    """Build metadata for operators without a dense/routed/shared width."""

    return {
        "profile_id": profile_id,
        "operator_family_id": operator_family_id,
        "operator_family_ids": [operator_family_id],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": tensor_parallel_mode.value,
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": list(tensor_parallel_sizes),
        "selected_tensor_parallel_size": selected_tensor_parallel_size,
        "selected_padded_ffn_width": None,
    }


def _add_operator_contract(
    contracts: Dict[str, Dict[str, object]],
    operator_name: str,
    metadata: Dict[str, object],
) -> None:
    """Insert one profiling-name contract and reject conflicting ownership."""

    existing = contracts.get(operator_name)
    if existing is None:
        contracts[operator_name] = metadata
        return
    if existing != metadata:
        raise ValueError(
            f"Profiling operator {operator_name!r} has conflicting typed contracts: "
            f"{existing!r} versus {metadata!r}"
        )


def _typed_operator_contracts(
    model_config,
    *,
    profile,
    typed_layer_contracts: Sequence[Dict[str, object]],
    attn_tp: Sequence[int],
    memory_ops: Sequence[str],
    linear_attention,
    include_target_embedded_mtp: bool,
    tp_size: int,
) -> Dict[str, Dict[str, object]]:
    """Bind every profiling operator to its registry-owned typed metadata."""

    contracts: Dict[str, Dict[str, object]] = {}
    layer_contract_by_family: Dict[str, Dict[str, object]] = {}
    for layer_contract in typed_layer_contracts:
        family_ids = layer_contract.get("operator_family_ids", ())
        for family_id in family_ids:
            if family_id in layer_contract_by_family:
                raise ValueError(
                    f"Operator family {family_id!r} is owned by multiple active "
                    "typed layer contracts"
                )
            layer_contract_by_family[str(family_id)] = layer_contract

    for family_id, layer_contract in layer_contract_by_family.items():
        family = get_operator_family(family_id)
        metadata = dict(layer_contract)
        metadata["operator_family_id"] = family_id
        for operator in family.profiling_ops():
            _add_operator_contract(
                contracts,
                operator.profiling_name(),
                dict(metadata),
            )

    normalized_attn_tp = _normalize_tp_domain(attn_tp, name="attn_tp")
    attention_binding = bind_attention_family(model_config)
    replicated_attention_ops = set(linear_attention.replicated_ops)
    sharded_attention_ops = set(linear_attention.sharded_ops)
    overlap = replicated_attention_ops.intersection(sharded_attention_ops)
    if overlap:
        raise ValueError(
            "Architecture profile declares attention operators as both replicated "
            f"and sharded: {sorted(overlap)}"
        )
    for operator_name in _dedupe_preserving_order(
        [*linear_attention.replicated_ops, *linear_attention.sharded_ops]
    ):
        mode = (
            TensorParallelMode.REPLICATED
            if operator_name in replicated_attention_ops
            else TensorParallelMode.ATTENTION_TP
        )
        selected_size = 1 if mode is TensorParallelMode.REPLICATED else None
        if mode is TensorParallelMode.ATTENTION_TP and not getattr(
            model_config, "no_tensor_parallel", False
        ):
            selected_size = tp_size if tp_size in normalized_attn_tp else None
        _add_operator_contract(
            contracts,
            operator_name,
            _non_layer_operator_contract(
                profile_id=profile.profile_id,
                operator_family_id=attention_binding.family_id,
                tensor_parallel_mode=mode,
                tensor_parallel_sizes=(
                    [1] if mode is TensorParallelMode.REPLICATED else normalized_attn_tp
                ),
                selected_tensor_parallel_size=selected_size,
            ),
        )

    memory_operator_names = set(memory_ops)
    memory_family = get_operator_family(MEMORY_FAMILY.family_id)
    for operator in memory_family.profiling_ops():
        operator_name = operator.profiling_name()
        if operator_name not in memory_operator_names:
            continue
        _add_operator_contract(
            contracts,
            operator_name,
            _non_layer_operator_contract(
                profile_id=profile.profile_id,
                operator_family_id=memory_family.family_id,
                tensor_parallel_mode=TensorParallelMode.REPLICATED,
                tensor_parallel_sizes=[1],
                selected_tensor_parallel_size=1,
            ),
        )

    if include_target_embedded_mtp:
        mtp_family_id = _target_embedded_mtp_family_id()
        for operator_name in TARGET_EMBEDDED_MTP_OPS:
            selected_size = None
            if not getattr(model_config, "no_tensor_parallel", False):
                selected_size = tp_size if tp_size in normalized_attn_tp else None
            _add_operator_contract(
                contracts,
                operator_name,
                _non_layer_operator_contract(
                    profile_id=profile.profile_id,
                    operator_family_id=mtp_family_id,
                    tensor_parallel_mode=TensorParallelMode.ATTENTION_TP,
                    tensor_parallel_sizes=normalized_attn_tp,
                    selected_tensor_parallel_size=selected_size,
                ),
            )

    return contracts


def build_profiling_plan(
    model_config,
    tp_size: int,
    attn_tp: Sequence[int],
    ffn_tp: Sequence[int],
    disable_replicated: bool = False,
    is_moe: bool = False,
    include_target_embedded_mtp: bool = False,
    moe_tp: Sequence[int] | None = None,
    include_ffn: bool = True,
) -> Dict[str, object]:
    tp_size = _validate_positive_int(tp_size, name="tp_size")
    if moe_tp is None:
        # Preserve the historical single-domain CLI behavior when callers do
        # not provide an explicit routed-MoE TP domain.
        moe_tp = ffn_tp

    attn_tp = _normalize_tp_domain(attn_tp, name="attn_tp")
    ffn_tp = _normalize_tp_domain(ffn_tp, name="ffn_tp")
    moe_tp = _normalize_tp_domain(moe_tp, name="moe_tp")
    attn_tp_set = set(attn_tp)
    ffn_tp_set = set(ffn_tp)

    skip_reasons: List[str] = []

    if getattr(model_config, "no_tensor_parallel", False) and tp_size > 1:
        skip_reasons.append("no_tensor_parallel")
        attn_sharded_enabled = False
        ffn_sharded_enabled = False
    else:
        attn_sharded_enabled = tp_size in attn_tp_set
        if attn_sharded_enabled:
            if model_config.embedding_dim % tp_size != 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"embedding_dim={model_config.embedding_dim} not divisible by TP={tp_size}"
                )
            if model_config.num_q_heads % tp_size != 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"num_q_heads={model_config.num_q_heads} not divisible by TP={tp_size}"
                )
            if model_config.num_kv_heads <= 0:
                attn_sharded_enabled = False
                skip_reasons.append(
                    f"num_kv_heads must be positive, got {model_config.num_kv_heads}"
                )
            elif model_config.num_kv_heads >= tp_size:
                if model_config.num_kv_heads % tp_size != 0:
                    attn_sharded_enabled = False
                    skip_reasons.append(
                        f"num_kv_heads={model_config.num_kv_heads} not divisible by TP={tp_size}"
                    )
            else:
                if tp_size % model_config.num_kv_heads != 0:
                    attn_sharded_enabled = False
                    skip_reasons.append(
                        f"TP={tp_size} must be divisible by num_kv_heads={model_config.num_kv_heads} for KV-head replication"
                    )

        ffn_sharded_enabled = include_ffn and tp_size in ffn_tp_set

    padded_n_embd = model_config.embedding_dim
    architecture_profile = get_model_architecture_profile(model_config)
    typed_layer_contracts = (
        _typed_layer_contracts(
            model_config,
            profile=architecture_profile,
            tp_size=tp_size,
            attn_tp=attn_tp,
            ffn_tp=ffn_tp,
            moe_tp=moe_tp,
        )
        if include_ffn
        else []
    )
    selected_typed_contracts = [
        contract
        for contract in typed_layer_contracts
        if contract["selected_tensor_parallel_size"] is not None
    ]

    # Mixed-layer profiles can place the standard dense FFN in the attention
    # TP domain. Keep the historical FFN-list behavior for pure models while
    # deriving the mixed decision from the registered family owner.
    if include_ffn and architecture_profile._is_mixed_model(model_config):
        ffn_family_ids = {
            FFN_FAMILY.family_id,
            SHARE_EXPERT_FAMILY.family_id,
        }
        ffn_sharded_enabled = any(
            tp_size == contract["selected_tensor_parallel_size"]
            and any(
                family_id in ffn_family_ids
                for family_id in contract["operator_family_ids"]
            )
            for contract in typed_layer_contracts
        )
        if getattr(model_config, "no_tensor_parallel", False) and tp_size > 1:
            ffn_sharded_enabled = False

    # ``padded_n_expanded_embd`` remains a compatibility field for callers
    # that have not migrated to the typed list. Prefer the selected dense
    # domain, then routed/shared domains, and finally the legacy scalar.
    padded_n_expanded_embd = model_config.mlp_hidden_dim
    if include_ffn:
        for preferred_kind in ("dense", "routed", "shared"):
            matching = [
                contract
                for contract in selected_typed_contracts
                if contract["layer_kind"] == preferred_kind
            ]
            if matching:
                padded_n_expanded_embd = int(matching[0]["selected_padded_ffn_width"])
                break
    if ffn_sharded_enabled:
        padded_n_embd = _pad_to_multiple(model_config.embedding_dim, tp_size)

    replicated_enabled = not disable_replicated
    attn_enabled = attn_sharded_enabled or replicated_enabled
    ffn_enabled = include_ffn and (ffn_sharded_enabled or replicated_enabled)

    linear_attention = architecture_profile.linear_attention
    memory_ops = _memory_profiling_names(model_config)
    typed_operator_contracts = _typed_operator_contracts(
        model_config,
        profile=architecture_profile,
        typed_layer_contracts=typed_layer_contracts,
        attn_tp=attn_tp,
        memory_ops=memory_ops,
        linear_attention=linear_attention,
        include_target_embedded_mtp=include_target_embedded_mtp,
        tp_size=tp_size,
    )

    # MEMORY_FAMILY declaration order keeps the pre-attention normalization
    # ahead of architecture-specific attention projections. Preserve that
    # established output order while deriving the names from the registry.
    memory_pre_attention_ops = memory_ops[:1]
    memory_post_attention_ops = memory_ops[1:]
    replicated_ops: List[str] = []
    replicated_ops.extend(memory_pre_attention_ops)
    replicated_ops.extend(linear_attention.replicated_ops)
    replicated_ops.extend(memory_post_attention_ops)
    target_embedded_same_tp_ops: List[str] = []
    if include_target_embedded_mtp:
        target_embedded_same_tp_ops.extend(
            [
                op_name
                for op_name in get_target_embedded_mtp_same_tp_linear_ops()
                if op_name != "post_attention_layernorm"
                or getattr(model_config, "post_attn_norm", False)
            ]
        )
        same_tp_ops_set = set(target_embedded_same_tp_ops)
        replicated_ops = [
            op_name for op_name in replicated_ops if op_name not in same_tp_ops_set
        ]

    enabled_ops: List[str] = []
    if replicated_enabled:
        enabled_ops.extend(replicated_ops)
    if include_target_embedded_mtp and (tp_size == 1 or attn_sharded_enabled):
        enabled_ops.extend(target_embedded_same_tp_ops)

    if attn_sharded_enabled:
        enabled_ops.extend(linear_attention.sharded_ops)
        if include_target_embedded_mtp:
            enabled_ops.extend(TARGET_EMBEDDED_MTP_OPS)

    if ffn_sharded_enabled:
        enabled_ops.extend(
            _typed_linear_profiling_names(
                typed_layer_contracts,
                selected_tp_size=tp_size,
            )
        )

    all_ops: List[str] = []
    all_ops.extend(replicated_ops)
    all_ops.extend(linear_attention.sharded_ops)
    if include_target_embedded_mtp:
        all_ops.extend(TARGET_EMBEDDED_MTP_OPS)
    all_ops.extend(_typed_linear_profiling_names(typed_layer_contracts))
    # Remove duplicates while preserving order.
    all_ops = list(dict.fromkeys(all_ops))
    enabled_ops = list(dict.fromkeys(enabled_ops))
    disabled_ops = [op for op in all_ops if op not in set(enabled_ops)]

    return {
        "tp_size": tp_size,
        "attn_enabled": attn_enabled,
        "ffn_enabled": ffn_enabled,
        "attn_sharded_enabled": attn_sharded_enabled,
        "ffn_sharded_enabled": ffn_sharded_enabled,
        "replicated_enabled": replicated_enabled,
        "disable_replicated": disable_replicated,
        "enabled_ops": enabled_ops,
        "disabled_ops": disabled_ops,
        "replicated_ops": replicated_ops if replicated_enabled else [],
        "padded_n_embd": padded_n_embd,
        "padded_n_expanded_embd": padded_n_expanded_embd,
        "typed_layer_contracts": typed_layer_contracts,
        "typed_operator_contracts": typed_operator_contracts,
        "skip_reasons": skip_reasons,
    }
