"""Model, operator-family and layer-contract identity helpers.

These pure functions answer identity questions the predictor stack asks in
several places: which operator family a profiling model name belongs to,
which architecture profile a replica resolves to, and what the cache
identity of a resolved layer contract is.  They hold no state, so they sit
in a leaf module every predictor component can import.
"""

import hashlib
import json
import pandas as pd

from frontier.model_architectures import (
    LayerKind,
    ModelArchitectureProfile,
    ResolvedLayerContract,
    get_model_architecture_profile,
)
from frontier.moe_gating_runtime import get_moe_gating_base_model_name
from frontier.operators.families import (
    MOE_FAMILY,
    get_family_profiling_names,
    get_operator_family,
)
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    matches_resolved_layer_contract,
)
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

MIGRATION_HELP_COMMAND = (
    "python -m frontier.profiling.migrate_csv_metadata --help"
)
def _get_moe_family_model_names() -> List[str]:
    return list(get_family_profiling_names(MOE_FAMILY))


def _get_moe_family_operator_by_model_name(model_name: str):
    moe_ops = {
        operator.profiling_name(): operator
        for operator in MOE_FAMILY.profiling_ops()
    }
    if model_name not in moe_ops:
        raise ValueError(f"Unsupported MoE op: {model_name}")
    return moe_ops[model_name]


def _get_moe_gating_family_model_names() -> List[str]:
    return [
        operator.profiling_name()
        for operator in MOE_FAMILY.profiling_ops()
        if operator.precision_name() == "moe_gating"
    ]


def _get_prefill_hot_moe_gating_model_names() -> List[str]:
    return [
        f"{model_name}__prefill_hot"
        for model_name in _get_moe_gating_family_model_names()
    ]


def _resolve_model_architecture_profile(
    model_config: Any,
    *,
    allow_generic: bool = False,
) -> Optional[ModelArchitectureProfile]:
    if model_config is None:
        return None
    getter = getattr(model_config, "get_model_architecture_profile", None)
    if callable(getter):
        return cast(Optional[ModelArchitectureProfile], getter())

    # Lightweight test/config adapters that predate the typed contract do not
    # expose a profile accessor or typed widths.  Keep those callers on the
    # scalar compatibility path instead of treating the generic fallback as a
    # complete typed declaration.  An explicit profile or typed width opts the
    # adapter into strict profile-owned resolution.
    typed_fields = (
        "model_architecture_profile",
        "dense_mlp_hidden_dim",
        "routed_mlp_hidden_dim",
        "share_expert_dim",
    )
    if not allow_generic and not any(
        getattr(model_config, field_name, None) is not None
        for field_name in typed_fields
    ):
        return None
    return get_model_architecture_profile(model_config)


def _resolve_model_architecture_profile_id(model_config) -> str:
    architecture_profile = _resolve_model_architecture_profile(model_config)
    if architecture_profile is None:
        return "generic"
    return architecture_profile.profile_id


def _resolve_profile_typed_family_for_query(
    architecture_profile: ModelArchitectureProfile,
    op_name: str,
) -> Optional[Tuple[str, LayerKind]]:
    """Resolve a query to the profile-owned typed operator family."""

    if not isinstance(op_name, str) or not op_name:
        raise ValueError("typed operator query name must be a non-empty string")
    matches: list[Tuple[str, LayerKind]] = []
    for layer_contract in architecture_profile.layer_contracts:
        for family_id in layer_contract.operator_family_ids:
            family = get_operator_family(family_id)
            if any(
                op_name == operator.name
                or op_name == operator.profiling_name()
                for operator in family.operators
            ):
                matches.append((family_id, layer_contract.layer_kind))
    if len(matches) > 1:
        raise ValueError(
            f"Operator query {op_name!r} belongs to multiple typed layer families: "
            f"{sorted(family_id for family_id, _ in matches)}"
        )
    return matches[0] if matches else None
def _serialize_selected_layer_cache_identity(
    layer_contract: Optional[ResolvedLayerContract],
) -> Optional[str]:
    """Serialize the selected semantic domain used by a model cache.

    Physical ``layer_id`` and producer-side domain envelopes do not identify a
    trained estimator.  Keep only the selected fields that affect estimator
    admission and reuse, in deterministic JSON form.
    """

    if layer_contract is None:
        return None
    if not isinstance(layer_contract, ResolvedLayerContract):
        raise TypeError(
            "layer_contract must be a ResolvedLayerContract when provided"
        )
    metadata = layer_contract.typed_metadata_identity()
    selected_fields = (
        "profile_id",
        "operator_family_id",
        "layer_kind",
        "dimension_source",
        "effective_ffn_width",
        "tensor_parallel_mode",
        "expert_parallel_mode",
        "selected_expert_parallel_size",
        "selected_tensor_parallel_size",
        "selected_padded_ffn_width",
    )
    return json.dumps(
        {field_name: metadata[field_name] for field_name in selected_fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _get_contract_hash(
    layer_contract: Optional[ResolvedLayerContract],
) -> str:
    """Return a short deterministic hash for an optional selected contract."""

    identity = _serialize_selected_layer_cache_identity(layer_contract)
    if identity is None:
        return "none"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _validate_typed_parallel_selection(
    layer_contract: ResolvedLayerContract,
    *,
    tensor_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
) -> None:
    """Validate explicit loader selectors against a resolved contract."""

    if not isinstance(layer_contract, ResolvedLayerContract):
        raise TypeError("layer_contract must be a ResolvedLayerContract")
    contract_tp = layer_contract.tensor_parallel_size
    if (
        contract_tp is not None
        and tensor_parallel_size is not None
        and contract_tp != tensor_parallel_size
    ):
        raise ValueError(
            f"typed layer contract TP {contract_tp} conflicts with "
            f"tensor_parallel_size {tensor_parallel_size}"
        )
    contract_ep = layer_contract.expert_parallel_size
    if (
        contract_ep is not None
        and expert_parallel_size is not None
        and contract_ep != expert_parallel_size
    ):
        raise ValueError(
            f"typed layer contract EP {contract_ep} conflicts with "
            f"expert_parallel_size {expert_parallel_size}"
        )


def _typed_row_matches_contract(
    raw_contracts: Any,
    layer_contract: ResolvedLayerContract,
    *,
    operator_name: Optional[str],
) -> bool:
    """Match one parsed or serialized row to its exact typed operator contract."""

    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError(
            "typed profiling loading requires a non-empty operator_name when "
            f"the canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
        )
    if not isinstance(layer_contract.operator_family_id, str) or not layer_contract.operator_family_id:
        raise ValueError(
            "typed profiling loading requires a layer contract with an operator family id"
        )
    return matches_resolved_layer_contract(
        raw_contracts,
        layer_contract,
        operator_name=operator_name,
    )


def _normalize_layer_contract_context(
    training_context: Optional[Mapping[str, Any]],
    explicit_layer_contract: Optional[ResolvedLayerContract] = None,
) -> Tuple[Optional[ResolvedLayerContract], Dict[str, Any]]:
    """Resolve one contract and keep every context representation consistent."""

    context = dict(training_context or {})
    context_contract = context.get("layer_contract")
    if context_contract is not None and not isinstance(
        context_contract, ResolvedLayerContract
    ):
        raise TypeError(
            "training_context['layer_contract'] must be a ResolvedLayerContract"
        )
    if explicit_layer_contract is not None and not isinstance(
        explicit_layer_contract, ResolvedLayerContract
    ):
        raise TypeError("layer_contract must be a ResolvedLayerContract")

    if context_contract is not None and explicit_layer_contract is not None:
        if not context_contract.is_semantically_equivalent(explicit_layer_contract):
            raise ValueError(
                "conflicting layer_contract values were provided through the "
                "explicit argument and training_context"
            )

    resolved_contract = explicit_layer_contract or context_contract
    context_identity = context.get("layer_contract_identity")
    if context_identity is not None and not isinstance(context_identity, str):
        raise TypeError(
            "training_context['layer_contract_identity'] must be a string"
        )
    selected_identity = _serialize_selected_layer_cache_identity(resolved_contract)
    if context_identity is not None and context_identity != selected_identity:
        raise ValueError(
            "training_context['layer_contract_identity'] does not match the "
            "supplied layer_contract"
        )
    if resolved_contract is None:
        return None, context

    context["layer_contract"] = resolved_contract
    context["layer_contract_identity"] = selected_identity
    context["layer_kind"] = resolved_contract.layer_kind.value
    context["effective_ffn_width"] = resolved_contract.effective_ffn_width
    context["tensor_parallel_mode"] = resolved_contract.tensor_parallel_mode.value
    context["expert_parallel_mode"] = resolved_contract.expert_parallel_mode.value
    return resolved_contract, context


def _add_layer_contract_to_training_context(
    training_context: Mapping[str, Any],
    layer_contract: Optional[ResolvedLayerContract],
) -> Dict[str, Any]:
    """Copy a training context and attach a resolved typed contract."""

    _, context = _normalize_layer_contract_context(
        training_context,
        explicit_layer_contract=layer_contract,
    )
    return context


def _layer_contract_kwargs(
    layer_contract: Optional[ResolvedLayerContract],
    *,
    operator_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return typed keyword arguments only for an opted-in contract path."""

    if layer_contract is None:
        return {}
    kwargs: Dict[str, Any] = {"layer_contract": layer_contract}
    if operator_name is not None:
        kwargs["operator_name"] = operator_name
    return kwargs
def _is_moe_gating_family_model_name(model_name: str) -> bool:
    base_model_name = get_moe_gating_base_model_name(model_name)
    return _get_moe_family_operator_by_model_name(
        base_model_name
    ).precision_name() == "moe_gating"
def _build_exact_feature_lookup(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[Tuple[float, ...], float]:
    """Build exact profiling-row lookups before falling back to regression."""
    if df.empty:
        return {}
    grouped = df.groupby(feature_cols, dropna=False)[target_col].mean()
    lookup: Dict[Tuple[float, ...], float] = {}
    for key, value in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        lookup[tuple(float(item) for item in key_tuple)] = float(value)
    return lookup
