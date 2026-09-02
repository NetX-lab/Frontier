"""Canonical serialization for profile-owned typed operator contracts.

The contract schema is shared by profiling producers and runtime consumers.
Keeping its implementation in the operator layer prevents runtime predictors
from depending on profiling implementation modules.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Sequence

import pandas as pd


TYPED_OPERATOR_CONTRACTS_COLUMN = "typed_operator_contracts"


# One field list is shared by every typed metadata producer and consumer.
TYPED_METADATA_REQUIRED_FIELDS = (
    "profile_id",
    "operator_family_id",
    "operator_family_ids",
    "layer_kind",
    "dimension_source",
    "effective_ffn_width",
    "tensor_parallel_mode",
    "expert_parallel_mode",
    "selected_expert_parallel_size",
    "tensor_parallel_sizes",
    "selected_tensor_parallel_size",
    "selected_padded_ffn_width",
)


def _positive_int(value: Any, *, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"typed metadata field {field_name!r} must be a positive int, "
            f"got {value!r}"
        )


def _optional_positive_int(value: Any, *, field_name: str) -> None:
    if value is not None:
        _positive_int(value, field_name=field_name)


def _positive_int_sequence(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"typed metadata field {field_name!r} must be a sequence of positive ints, "
            f"got {value!r}"
        )
    normalized = tuple(value)
    for item in normalized:
        _positive_int(item, field_name=field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"typed metadata field {field_name!r} must not contain duplicates, "
            f"got {value!r}"
        )
    return normalized


def _enum_value(value: Any, enum_type: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"typed metadata field {field_name!r} must be a string, got {value!r}"
        )
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        allowed = sorted({member.value for member in enum_type})
        raise ValueError(
            f"typed metadata field {field_name!r} has unsupported value {value!r}; "
            f"expected one of {allowed}"
        ) from exc


def _validate_contract_mapping(value: Mapping[Any, Any]) -> dict[str, dict[str, Any]]:
    """Validate and copy one operator-to-contract mapping."""

    if not value:
        raise ValueError(
            "typed_operator_contracts must contain at least one operator contract"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for operator_name, metadata in value.items():
        if not isinstance(operator_name, str) or not operator_name.strip():
            raise ValueError(
                "typed_operator_contracts operator names must be non-empty strings"
            )
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "typed_operator_contracts metadata must be mappings; "
                f"got {type(metadata).__name__} for {operator_name!r}"
            )
        normalized[operator_name] = deepcopy(dict(metadata))
    return normalized


def validate_typed_operator_metadata(
    value: Mapping[Any, Any],
    *,
    operator_name: str,
    expected_metadata: Mapping[str, Any],
    expected_tensor_parallel_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate one complete typed row against registry-derived expectations.

    The serializer/parser intentionally remain permissive so they can preserve
    unknown future fields and inspect legacy payloads.  Admission points use
    this stricter validator: every current contract field is required, scalar
    types are checked without Python coercion, and the supplied semantic fields
    must agree with facts resolved by the operator/architecture registries.

    ``expected_metadata`` is deliberately supplied by the caller rather than
    reconstructed here.  This keeps this schema owner independent from any one
    model while making the registry boundary explicit at each consumer.
    ``expected_tensor_parallel_sizes`` is optional because a single dataset may
    carry several legal TP sizes; when omitted, the selected size still must be
    a member of the row's declared domain.
    """

    if not isinstance(value, Mapping):
        raise ValueError(
            f"typed metadata for {operator_name!r} must be a mapping, "
            f"got {type(value).__name__}"
        )
    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError("typed metadata validation requires a non-empty operator_name")
    if not isinstance(expected_metadata, Mapping):
        raise TypeError("expected_metadata must be a mapping")

    metadata = dict(value)
    missing_fields = [
        field_name
        for field_name in TYPED_METADATA_REQUIRED_FIELDS
        if field_name not in metadata
    ]
    if missing_fields:
        raise ValueError(
            f"typed metadata for {operator_name!r} is missing required fields: "
            f"{', '.join(missing_fields)}"
        )

    profile_id = metadata["profile_id"]
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(
            f"typed metadata field 'profile_id' for {operator_name!r} must be a "
            f"non-empty string, got {profile_id!r}"
        )

    operator_family_id = metadata["operator_family_id"]
    if not isinstance(operator_family_id, str) or not operator_family_id.strip():
        raise ValueError(
            f"typed metadata field 'operator_family_id' for {operator_name!r} "
            f"must be a non-empty string, got {operator_family_id!r}"
        )

    family_ids = metadata["operator_family_ids"]
    if isinstance(family_ids, (str, bytes)) or not isinstance(family_ids, Sequence):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} "
            f"must be a sequence of strings, got {family_ids!r}"
        )
    family_ids = tuple(family_ids)
    if not family_ids or any(
        not isinstance(family_id, str) or not family_id.strip()
        for family_id in family_ids
    ):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} "
            f"must contain non-empty strings, got {family_ids!r}"
        )
    if len(set(family_ids)) != len(family_ids):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} "
            f"must not contain duplicates, got {family_ids!r}"
        )
    if operator_family_id not in family_ids:
        raise ValueError(
            f"typed metadata for {operator_name!r} names family "
            f"{operator_family_id!r} outside operator_family_ids={family_ids!r}"
        )

    # Import the enums lazily so this low-level serialization module stays safe
    # during package initialization and does not create an architecture cycle.
    from frontier.model_architectures import (
        ExpertParallelMode,
        LayerDimensionSource,
        LayerKind,
    )
    from frontier.operators.spec import TensorParallelMode

    layer_kind = metadata["layer_kind"]
    if layer_kind is not None:
        layer_kind = _enum_value(layer_kind, LayerKind, field_name="layer_kind")
    dimension_source = metadata["dimension_source"]
    if dimension_source is not None:
        dimension_source = _enum_value(
            dimension_source,
            LayerDimensionSource,
            field_name="dimension_source",
        )
    tensor_parallel_mode = _enum_value(
        metadata["tensor_parallel_mode"],
        TensorParallelMode,
        field_name="tensor_parallel_mode",
    )
    expert_parallel_mode = _enum_value(
        metadata["expert_parallel_mode"],
        ExpertParallelMode,
        field_name="expert_parallel_mode",
    )

    effective_width = metadata["effective_ffn_width"]
    if effective_width is not None:
        _positive_int(effective_width, field_name="effective_ffn_width")
    selected_expert_parallel_size = metadata["selected_expert_parallel_size"]
    _optional_positive_int(
        selected_expert_parallel_size,
        field_name="selected_expert_parallel_size",
    )
    tensor_parallel_sizes = _positive_int_sequence(
        metadata["tensor_parallel_sizes"],
        field_name="tensor_parallel_sizes",
    )
    selected_tensor_parallel_size = metadata["selected_tensor_parallel_size"]
    _optional_positive_int(
        selected_tensor_parallel_size,
        field_name="selected_tensor_parallel_size",
    )
    if (
        selected_tensor_parallel_size is not None
        and selected_tensor_parallel_size not in tensor_parallel_sizes
    ):
        raise ValueError(
            f"typed metadata for {operator_name!r} selects TP="
            f"{selected_tensor_parallel_size} outside tensor_parallel_sizes="
            f"{list(tensor_parallel_sizes)}"
        )

    selected_padded_width = metadata["selected_padded_ffn_width"]
    _optional_positive_int(
        selected_padded_width,
        field_name="selected_padded_ffn_width",
    )

    if layer_kind is None:
        if dimension_source is not None:
            raise ValueError(
                f"typed metadata for non-layer operator {operator_name!r} must set "
                "dimension_source=None"
            )
        if effective_width is not None:
            raise ValueError(
                f"typed metadata for non-layer operator {operator_name!r} must set "
                "effective_ffn_width=None"
            )
        if selected_padded_width is not None:
            raise ValueError(
                f"typed metadata for non-layer operator {operator_name!r} must set "
                "selected_padded_ffn_width=None"
            )
    else:
        if dimension_source is None or effective_width is None:
            raise ValueError(
                f"typed metadata for layer-bound operator {operator_name!r} must "
                "declare dimension_source and effective_ffn_width"
            )
        if selected_tensor_parallel_size is not None and selected_padded_width is None:
            raise ValueError(
                f"typed metadata for layer-bound operator {operator_name!r} must "
                "declare selected_padded_ffn_width when selected_tensor_parallel_size "
                "is set"
            )
        expected_sources = {
            LayerKind.DENSE.value: LayerDimensionSource.DENSE.value,
            LayerKind.ROUTED.value: LayerDimensionSource.ROUTED.value,
            LayerKind.SHARED.value: LayerDimensionSource.SHARED.value,
        }
        if expected_sources.get(layer_kind) != dimension_source:
            raise ValueError(
                f"typed metadata for {operator_name!r} pairs layer_kind="
                f"{layer_kind!r} with incompatible dimension_source="
                f"{dimension_source!r}"
            )
        if (
            selected_padded_width is not None
            and selected_padded_width < effective_width
        ):
            raise ValueError(
                f"typed metadata field 'selected_padded_ffn_width' for "
                f"{operator_name!r} has padded width {selected_padded_width} "
                f"below effective width {effective_width}"
            )

    if (
        selected_padded_width is not None
        and selected_tensor_parallel_size is not None
        and selected_padded_width % selected_tensor_parallel_size != 0
    ):
        raise ValueError(
            f"typed metadata for {operator_name!r} has padded width "
            f"{selected_padded_width} not divisible by selected TP "
            f"{selected_tensor_parallel_size}"
        )

    if expected_tensor_parallel_sizes is not None:
        expected_domain = _positive_int_sequence(
            expected_tensor_parallel_sizes,
            field_name="expected_tensor_parallel_sizes",
        )
        if tensor_parallel_sizes != expected_domain:
            raise ValueError(
                f"typed metadata for {operator_name!r} declares TP domain "
                f"{list(tensor_parallel_sizes)}, expected {list(expected_domain)}"
            )

    enum_fields = {
        "layer_kind": LayerKind,
        "dimension_source": LayerDimensionSource,
        "tensor_parallel_mode": TensorParallelMode,
        "expert_parallel_mode": ExpertParallelMode,
    }
    for field_name, expected_value in expected_metadata.items():
        if field_name not in metadata:
            raise ValueError(
                f"typed metadata for {operator_name!r} is missing expected field "
                f"{field_name!r}"
            )
        actual_value = metadata[field_name]
        if field_name == "operator_family_ids":
            actual_value = tuple(actual_value)
            expected_value = tuple(expected_value)
        elif field_name in enum_fields and expected_value is not None:
            expected_value = _enum_value(
                expected_value,
                enum_fields[field_name],
                field_name=field_name,
            )
        # A non-layer resolver may know only that the selected TP is absent
        # (for example, a multi-TP dataset queried without a row-level TP).
        # Its empty expected domain means "leave the producer domain intact";
        # an explicit expected domain remains an exact check above.
        if (
            field_name in {"tensor_parallel_sizes", "selected_tensor_parallel_size"}
            and "tensor_parallel_sizes" in expected_metadata
            and not expected_metadata["tensor_parallel_sizes"]
        ):
            continue
        if not typed_metadata_value_matches(actual_value, expected_value):
            raise ValueError(
                f"typed metadata for {operator_name!r} field {field_name!r} "
                f"disagrees with the registry-derived contract: "
                f"actual={actual_value!r}, expected={expected_value!r}"
            )

    return metadata


def build_non_layer_typed_operator_expectation(
    *,
    operator_name: str,
    model_config: Any,
    architecture_profile: Any,
    selected_tensor_parallel_size: int | None,
) -> dict[str, Any]:
    """Resolve the expected metadata for an attention/memory non-layer op.

    Operator family and TP mode come from the existing registries.  Architecture
    profiles additionally declare linear-attention aliases that are not physical
    operator registrations; those aliases are bound through the selected
    attention family.  The returned mapping is suitable for
    :func:`validate_typed_operator_metadata`.
    """

    if selected_tensor_parallel_size is not None:
        _positive_int(
            selected_tensor_parallel_size,
            field_name="selected_tensor_parallel_size",
        )
    if architecture_profile is None:
        raise ValueError("architecture_profile is required for typed metadata resolution")

    from frontier.attention.model_binding import bind_attention_family
    from frontier.operators.families import iter_operator_families
    from frontier.operators.spec import TensorParallelMode

    registered_matches: list[tuple[str, Any]] = []
    for family in iter_operator_families():
        for operator in family.profiling_ops():
            names = {operator.name}
            profiling_name = operator.profiling_name()
            if isinstance(profiling_name, str):
                names.add(profiling_name)
            if operator_name in names:
                registered_matches.append((family.family_id, operator))

    registered_family_ids = {family_id for family_id, _ in registered_matches}
    if len(registered_family_ids) > 1:
        raise ValueError(
            f"Operator {operator_name!r} has ambiguous registered families: "
            f"{sorted(registered_family_ids)}"
        )

    linear_attention = getattr(architecture_profile, "linear_attention", None)
    if linear_attention is None:
        raise TypeError(
            "architecture_profile must expose linear_attention for non-layer "
            "typed metadata resolution"
        )
    replicated_ops = tuple(getattr(linear_attention, "replicated_ops", ()))
    sharded_ops = tuple(getattr(linear_attention, "sharded_ops", ()))
    overlap = set(replicated_ops).intersection(sharded_ops)
    if overlap:
        raise ValueError(
            "architecture profile declares attention operators in both TP domains: "
            f"{sorted(overlap)}"
        )

    attention_ops = set(replicated_ops).union(sharded_ops)
    attention_family_id = None
    attention_tp_mode = None
    if operator_name in attention_ops:
        attention_family_id = bind_attention_family(model_config).family_id
        attention_tp_mode = (
            TensorParallelMode.REPLICATED
            if operator_name in replicated_ops
            else TensorParallelMode.ATTENTION_TP
        )

    if registered_family_ids and attention_family_id is not None:
        registered_family_id = next(iter(registered_family_ids))
        if registered_family_id != attention_family_id:
            raise ValueError(
                f"Operator {operator_name!r} has conflicting registry ownership: "
                f"{registered_family_id!r} versus {attention_family_id!r}"
            )

    if attention_family_id is not None:
        family_id = attention_family_id
        tp_mode = attention_tp_mode
    elif registered_matches:
        family_id = next(iter(registered_family_ids))
        tp_modes = {operator.tp_mode for _, operator in registered_matches}
        if len(tp_modes) != 1:
            raise ValueError(
                f"Operator {operator_name!r} has conflicting registered TP modes: "
                f"{sorted(repr(mode) for mode in tp_modes)}"
            )
        tp_mode = next(iter(tp_modes))
        if tp_mode is None:
            raise ValueError(
                f"Operator {operator_name!r} has no registered TP mode"
            )
    else:
        raise ValueError(
            f"Operator {operator_name!r} is absent from the operator and attention "
            "registries"
        )

    selected_tp = (
        1
        if tp_mode is TensorParallelMode.REPLICATED
        else selected_tensor_parallel_size
    )
    # Non-layer callers provide one selected TP value rather than a complete
    # profiling envelope. Materialize that value as the declared domain so the
    # expected mapping has the same complete shape as a producer row. An
    # unknown selected value remains an empty domain; callers that own a wider
    # envelope can still enforce it through ``expected_tensor_parallel_sizes``.
    tensor_parallel_sizes = [] if selected_tp is None else [selected_tp]
    expected = {
        "profile_id": getattr(architecture_profile, "profile_id", None),
        "operator_family_id": family_id,
        "operator_family_ids": [family_id],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": tp_mode.value,
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": tensor_parallel_sizes,
        "selected_tensor_parallel_size": selected_tp,
        "selected_padded_ffn_width": None,
    }
    return expected


def serialize_typed_operator_contracts(value: Mapping[Any, Any]) -> str:
    """Encode typed operator contracts as deterministic canonical JSON."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "typed_operator_contracts must be a mapping before CSV serialization; "
            f"got {type(value).__name__}"
        )
    normalized = _validate_contract_mapping(value)
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "typed_operator_contracts contains a value that is not JSON serializable"
        ) from exc


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def parse_typed_operator_contracts(value: Any) -> dict[str, dict[str, Any]]:
    """Decode one canonical JSON contract value and validate its shape."""

    if isinstance(value, Mapping):
        return _validate_contract_mapping(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("typed_operator_contracts must contain UTF-8 JSON") from exc
    if not isinstance(value, str) or _is_missing_scalar(value):
        raise ValueError(
            "typed_operator_contracts must contain a non-empty JSON object"
        )

    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "typed_operator_contracts must use canonical JSON object syntax"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("typed_operator_contracts JSON value must be an object")
    return _validate_contract_mapping(decoded)


def typed_metadata_value_matches(actual: Any, expected: Any) -> bool:
    """Match one typed metadata value without Python's bool/int coercion."""

    return type(actual) is type(expected) and actual == expected


def matches_resolved_layer_contract(
    value: Any,
    layer_contract: Any,
    *,
    operator_name: str | None = None,
) -> bool:
    """Match canonical row metadata against one profile-owned layer contract.

    Typed row metadata is keyed by profiling operator.  Requiring an exact
    operator scope prevents a sibling operator with the same typed domain from
    satisfying the query; legacy scalar filtering belongs only to files
    without the typed metadata column.
    """

    if not hasattr(layer_contract, "typed_metadata_identity"):
        raise TypeError(
            "layer_contract must expose typed_metadata_identity() when matching "
            "typed profiling metadata"
        )
    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError(
            "typed contract matching requires a non-empty operator_name"
        )
    contracts = parse_typed_operator_contracts(value)
    candidates = (
        contracts[operator_name],
    ) if operator_name in contracts else ()
    expected = layer_contract.typed_metadata_identity()
    for metadata in candidates:
        # A typed column is an admitted producer schema, not a best-effort
        # hint. Validate the complete row first so malformed family/domain or
        # padding data remains visible to the caller. A structurally valid row
        # from a sibling domain is simply not a match for this query.
        validate_typed_operator_metadata(
            metadata,
            operator_name=operator_name,
            expected_metadata={},
        )

        # A producer may publish one row for a complete legal TP envelope,
        # while a resolver query selects one member of that envelope.  Match
        # the selected value against the declared domain instead of requiring
        # the producer's envelope to collapse to the query's singleton.
        expected_family_ids = tuple(expected["operator_family_ids"])
        actual_family_ids = tuple(metadata["operator_family_ids"])
        if not set(expected_family_ids).issubset(actual_family_ids):
            return False
        if metadata["operator_family_id"] != expected["operator_family_id"]:
            return False
        expected_tp_sizes = tuple(expected["tensor_parallel_sizes"])
        actual_tp_sizes = tuple(metadata["tensor_parallel_sizes"])
        if expected["selected_tensor_parallel_size"] is not None:
            if not set(expected_tp_sizes).issubset(actual_tp_sizes):
                return False
        elif actual_tp_sizes != expected_tp_sizes:
            return False

        if metadata["selected_padded_ffn_width"] != expected[
            "selected_padded_ffn_width"
        ]:
            return False

        scalar_fields = (
            field_name
            for field_name in expected
            if field_name
            not in {
                "operator_family_id",
                "operator_family_ids",
                "tensor_parallel_sizes",
                "selected_padded_ffn_width",
            }
        )
        if all(
            field_name in metadata
            and typed_metadata_value_matches(metadata[field_name], expected[field_name])
            for field_name in scalar_fields
        ):
            return True
    return False


def serialize_typed_operator_contract_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose typed-contract column contains canonical JSON."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    result = frame.copy()
    if TYPED_OPERATOR_CONTRACTS_COLUMN not in result.columns:
        return result
    result[TYPED_OPERATOR_CONTRACTS_COLUMN] = result[
        TYPED_OPERATOR_CONTRACTS_COLUMN
    ].map(serialize_typed_operator_contracts)
    return result


def parse_typed_operator_contract_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose typed-contract column contains validated mappings."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    result = frame.copy()
    if TYPED_OPERATOR_CONTRACTS_COLUMN not in result.columns:
        return result
    result[TYPED_OPERATOR_CONTRACTS_COLUMN] = result[
        TYPED_OPERATOR_CONTRACTS_COLUMN
    ].map(parse_typed_operator_contracts)
    return result


__all__ = [
    "TYPED_OPERATOR_CONTRACTS_COLUMN",
    "TYPED_METADATA_REQUIRED_FIELDS",
    "build_non_layer_typed_operator_expectation",
    "parse_typed_operator_contract_column",
    "parse_typed_operator_contracts",
    "matches_resolved_layer_contract",
    "serialize_typed_operator_contract_column",
    "serialize_typed_operator_contracts",
    "typed_metadata_value_matches",
    "validate_typed_operator_metadata",
]
