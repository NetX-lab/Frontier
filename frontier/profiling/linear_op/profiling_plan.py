"""Profiling plan construction for decoupled attention/FFN TP profiling."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

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


def _normalize_tp_domain(values: Sequence[int], *, name: str) -> List[int]:
    """Normalize a TP domain while preserving strict integer semantics."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain positive integer TP sizes")
    try:
        normalized = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain positive integer TP sizes") from exc
    if any(type(value) is not int or value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integer TP sizes, got {values!r}")
    return sorted(set(normalized))


def _typed_layer_contracts(
    model_config,
    *,
    profile,
    tp_size: int,
    attn_tp: Sequence[int],
    ffn_tp: Sequence[int],
    moe_tp: Sequence[int],
) -> List[Dict[str, object]]:
    """Materialize profile-owned typed layer domains for one plan."""

    domains = {
        TensorParallelMode.REPLICATED: [1],
        TensorParallelMode.ATTENTION_TP: _normalize_tp_domain(attn_tp, name="attn_tp"),
        TensorParallelMode.FFN_TP: _normalize_tp_domain(ffn_tp, name="ffn_tp"),
        TensorParallelMode.MOE_TP: _normalize_tp_domain(moe_tp, name="moe_tp"),
    }
    mixed_model = profile._is_mixed_model(model_config)
    contracts: List[Dict[str, object]] = []
    for spec in profile.iter_active_layer_contracts(model_config):
        width = spec.resolve_width(model_config, mixed_model=mixed_model)
        tensor_parallel_sizes = list(domains[spec.tensor_parallel_mode])
        selected_size = None
        if not (
            getattr(model_config, "no_tensor_parallel", False)
            and spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED
        ):
            if tp_size in tensor_parallel_sizes:
                selected_size = tp_size
        if (
            selected_size is not None
            and spec.tensor_parallel_mode is not TensorParallelMode.REPLICATED
            and width % selected_size != 0
        ):
            raise ValueError(
                f"{spec.layer_kind.value} width {width} must be divisible by TP={selected_size}"
            )
        contracts.append(
            {
                "profile_id": profile.profile_id,
                "operator_family_ids": list(spec.operator_family_ids),
                "layer_kind": spec.layer_kind.value,
                "dimension_source": spec.dimension_source.value,
                "effective_ffn_width": width,
                "tensor_parallel_mode": spec.tensor_parallel_mode.value,
                "expert_parallel_mode": spec.expert_parallel_mode.value,
                "selected_expert_parallel_size": None,
                "tensor_parallel_sizes": tensor_parallel_sizes,
                "selected_tensor_parallel_size": selected_size,
                "selected_padded_ffn_width": (
                    _pad_to_multiple(width, selected_size)
                    if selected_size is not None
                    else None
                ),
            }
        )
    return contracts


def _non_layer_contract(
    *,
    profile_id: str,
    family_id: str,
    mode: TensorParallelMode,
    tensor_parallel_sizes: Sequence[int],
    selected_tensor_parallel_size: int | None,
) -> Dict[str, object]:
    return {
        "profile_id": profile_id,
        "operator_family_id": family_id,
        "operator_family_ids": [family_id],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": mode.value,
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": list(tensor_parallel_sizes),
        "selected_tensor_parallel_size": selected_tensor_parallel_size,
        "selected_padded_ffn_width": None,
    }


def _typed_family_ids(contract: Mapping[str, object]) -> tuple[str, ...]:
    """Return validated operator-family IDs from one typed contract mapping."""

    raw_family_ids = contract.get("operator_family_ids")
    if isinstance(raw_family_ids, (str, bytes)) or not isinstance(
        raw_family_ids, Sequence
    ):
        raise ValueError(
            "typed layer contract operator_family_ids must be a sequence of strings"
        )
    family_ids = tuple(raw_family_ids)
    if not family_ids or any(
        not isinstance(family_id, str) or not family_id
        for family_id in family_ids
    ):
        raise ValueError(
            "typed layer contract operator_family_ids must contain non-empty strings"
        )
    return family_ids


def _selected_padded_width(contract: Mapping[str, object]) -> int | None:
    """Return a typed contract's optional padded width after narrowing it."""

    value = contract.get("selected_padded_ffn_width")
    if value is not None and type(value) is not int:
        raise ValueError(
            "typed layer contract selected_padded_ffn_width must be an int or null"
        )
    return value


def _add_typed_operator_contract(
    contracts: Dict[str, Dict[str, object]],
    operator_name: str,
    metadata: Dict[str, object],
) -> None:
    existing = contracts.get(operator_name)
    if existing is not None and existing != metadata:
        raise ValueError(
            f"profiling operator {operator_name!r} has conflicting typed contracts"
        )
    contracts[operator_name] = dict(metadata)


def _typed_operator_contracts(
    model_config,
    *,
    profile,
    typed_layer_contracts: Sequence[Dict[str, object]],
    producer_operator_names: Sequence[str],
    attn_tp: Sequence[int],
    memory_ops: Sequence[str],
    include_target_embedded_mtp: bool,
    tp_size: int,
) -> Dict[str, Dict[str, object]]:
    """Bind typed metadata to operators owned by the linear producer."""

    contracts: Dict[str, Dict[str, object]] = {}
    producer_names = set(producer_operator_names)
    for layer_contract in typed_layer_contracts:
        for family_id in _typed_family_ids(layer_contract):
            family = get_operator_family(family_id)
            metadata = dict(layer_contract)
            metadata["operator_family_id"] = str(family_id)
            for operator in family.profiling_ops():
                if operator.profiling_name() not in producer_names:
                    continue
                _add_typed_operator_contract(
                    contracts, operator.profiling_name(), metadata
                )

    normalized_attn_tp = _normalize_tp_domain(attn_tp, name="attn_tp")
    attention_family_id = bind_attention_family(model_config).family_id
    replicated_attention_ops = set(profile.linear_attention.replicated_ops)
    sharded_attention_ops = set(profile.linear_attention.sharded_ops)
    overlap = replicated_attention_ops.intersection(sharded_attention_ops)
    if overlap:
        raise ValueError(
            f"attention operators appear in both TP domains: {sorted(overlap)}"
        )
    for operator_name in [
        *profile.linear_attention.replicated_ops,
        *profile.linear_attention.sharded_ops,
    ]:
        if operator_name not in producer_names:
            continue
        is_replicated = operator_name in replicated_attention_ops
        mode = TensorParallelMode.REPLICATED if is_replicated else TensorParallelMode.ATTENTION_TP
        selected_size = 1 if is_replicated else (tp_size if tp_size in normalized_attn_tp else None)
        _add_typed_operator_contract(
            contracts,
            operator_name,
            _non_layer_contract(
                profile_id=profile.profile_id,
                family_id=attention_family_id,
                mode=mode,
                tensor_parallel_sizes=[1] if is_replicated else normalized_attn_tp,
                selected_tensor_parallel_size=selected_size,
            ),
        )

    memory_family = get_operator_family(MEMORY_FAMILY.family_id)
    memory_names = set(memory_ops)
    for operator in memory_family.profiling_ops():
        operator_name = operator.profiling_name()
        if operator_name not in memory_names or operator_name not in producer_names:
            continue
        _add_typed_operator_contract(
            contracts,
            operator_name,
            _non_layer_contract(
                profile_id=profile.profile_id,
                family_id=memory_family.family_id,
                mode=TensorParallelMode.REPLICATED,
                tensor_parallel_sizes=[1],
                selected_tensor_parallel_size=1,
            ),
        )

    if include_target_embedded_mtp:
        for operator_name in TARGET_EMBEDDED_MTP_OPS:
            if operator_name not in producer_names:
                continue
            _add_typed_operator_contract(
                contracts,
                operator_name,
                _non_layer_contract(
                    profile_id=profile.profile_id,
                    family_id="target_embedded_mtp",
                    mode=TensorParallelMode.ATTENTION_TP,
                    tensor_parallel_sizes=normalized_attn_tp,
                    selected_tensor_parallel_size=(
                        tp_size if tp_size in normalized_attn_tp else None
                    ),
                ),
            )
    return contracts


def _supports_share_expert(model_config) -> bool:
    if not hasattr(model_config, "supports_share_expert"):
        raise TypeError("linear-op profiling requires model_config.supports_share_expert()")
    return model_config.supports_share_expert()


def build_profiling_plan(
    model_config,
    tp_size: int,
    attn_tp: Sequence[int],
    ffn_tp: Sequence[int],
    disable_replicated: bool = False,
    is_moe: bool = False,
    include_target_embedded_mtp: bool = False,
    moe_tp: Sequence[int] | None = None,
) -> Dict[str, object]:
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError(f"tp_size must be a positive int, got {tp_size!r}")
    attn_tp = _normalize_tp_domain(attn_tp, name="attn_tp")
    ffn_tp = _normalize_tp_domain(ffn_tp, name="ffn_tp")
    moe_tp = _normalize_tp_domain(moe_tp if moe_tp is not None else ffn_tp, name="moe_tp")
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

        ffn_sharded_enabled = tp_size in ffn_tp_set

    padded_n_embd = model_config.embedding_dim
    architecture_profile = get_model_architecture_profile(model_config)
    typed_layer_contracts = _typed_layer_contracts(
        model_config,
        profile=architecture_profile,
        tp_size=tp_size,
        attn_tp=attn_tp,
        ffn_tp=ffn_tp,
        moe_tp=moe_tp,
    )
    selected_typed_contracts = [
        contract
        for contract in typed_layer_contracts
        if contract["selected_tensor_parallel_size"] is not None
    ]
    if architecture_profile._is_mixed_model(model_config):
        ffn_sharded_enabled = any(
            tp_size == contract["selected_tensor_parallel_size"]
            and "ffn" in _typed_family_ids(contract)
            for contract in typed_layer_contracts
        )
        if getattr(model_config, "no_tensor_parallel", False) and tp_size > 1:
            ffn_sharded_enabled = False
    padded_n_expanded_embd = model_config.mlp_hidden_dim
    # ``is_moe`` means that the regular dense MLP targets are intentionally
    # omitted from this producer.  Keep its untimed shape placeholder on the
    # routed domain so mixed models do not allocate a dense-width MLP while
    # collecting attention/shared-expert timings.  Dense profiling keeps the
    # dense contract as its first choice.
    preferred_kinds = ("routed", "shared", "dense") if is_moe else ("dense", "routed", "shared")
    for kind in preferred_kinds:
        matches = [
            contract
            for contract in selected_typed_contracts
            if contract["layer_kind"] == kind
            and _selected_padded_width(contract) is not None
        ]
        if matches:
            padded_width = _selected_padded_width(matches[0])
            if padded_width is None:
                raise ValueError("selected typed contract is missing its padded width")
            padded_n_expanded_embd = padded_width
            break
    if ffn_sharded_enabled:
        padded_n_embd = _pad_to_multiple(model_config.embedding_dim, tp_size)

    replicated_enabled = not disable_replicated
    attn_enabled = attn_sharded_enabled or replicated_enabled
    ffn_enabled = ffn_sharded_enabled or replicated_enabled

    linear_attention = architecture_profile.linear_attention
    memory_ops = _memory_profiling_names(model_config)

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
        if not is_moe:
            enabled_ops.extend(_ffn_profiling_names())
        if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
            enabled_ops.extend(_share_expert_profiling_names())

    all_ops: List[str] = []
    all_ops.extend(replicated_ops)
    all_ops.extend(linear_attention.sharded_ops)
    if include_target_embedded_mtp:
        all_ops.extend(TARGET_EMBEDDED_MTP_OPS)
    if not is_moe:
        all_ops.extend(_ffn_profiling_names())
    if getattr(model_config, "is_moe", False) and _supports_share_expert(model_config):
        all_ops.extend(_share_expert_profiling_names())
    # Remove duplicates while preserving order.
    all_ops = list(dict.fromkeys(all_ops))
    enabled_ops = list(dict.fromkeys(enabled_ops))
    disabled_ops = [op for op in all_ops if op not in set(enabled_ops)]

    typed_operator_contracts = _typed_operator_contracts(
        model_config,
        profile=architecture_profile,
        typed_layer_contracts=typed_layer_contracts,
        producer_operator_names=enabled_ops,
        attn_tp=attn_tp,
        memory_ops=memory_ops,
        include_target_embedded_mtp=include_target_embedded_mtp,
        tp_size=tp_size,
    )

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
