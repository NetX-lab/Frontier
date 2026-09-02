"""Canonical schema for typed profiling operator contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pandas as pd


TYPED_OPERATOR_CONTRACTS_COLUMN = "typed_operator_contracts"
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


def _require_positive_int(value: Any, *, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"typed metadata field {field_name!r} must be a positive int, got {value!r}"
        )


def _require_optional_positive_int(value: Any, *, field_name: str) -> None:
    if value is not None:
        _require_positive_int(value, field_name=field_name)


def _normalize_positive_ints(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"typed metadata field {field_name!r} must be a sequence of positive ints, "
            f"got {value!r}"
        )
    normalized = tuple(value)
    for item in normalized:
        _require_positive_int(item, field_name=field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"typed metadata field {field_name!r} must not contain duplicates, got {value!r}"
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
        allowed = sorted(member.value for member in enum_type)
        raise ValueError(
            f"typed metadata field {field_name!r} has unsupported value {value!r}; "
            f"expected one of {allowed}"
        ) from exc


def _copy_contract_mapping(value: Mapping[Any, Any]) -> dict[str, dict[str, Any]]:
    if not value:
        raise ValueError("typed_operator_contracts must contain at least one contract")
    normalized: dict[str, dict[str, Any]] = {}
    for operator_name, metadata in value.items():
        if not isinstance(operator_name, str) or not operator_name.strip():
            raise ValueError("typed contract operator names must be non-empty strings")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"typed contract for {operator_name!r} must be a mapping, "
                f"got {type(metadata).__name__}"
            )
        normalized[operator_name] = deepcopy(dict(metadata))
    return normalized


def typed_metadata_value_matches(actual: Any, expected: Any) -> bool:
    """Compare values without allowing bool/int coercion."""

    return type(actual) is type(expected) and actual == expected


def validate_typed_operator_metadata(
    value: Mapping[Any, Any],
    *,
    operator_name: str,
    expected_metadata: Mapping[str, Any],
    expected_tensor_parallel_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate one complete metadata object against resolver expectations."""

    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError("typed metadata validation requires a non-empty operator_name")
    if not isinstance(value, Mapping):
        raise ValueError(
            f"typed metadata for {operator_name!r} must be a mapping, got {type(value).__name__}"
        )
    if not isinstance(expected_metadata, Mapping):
        raise TypeError("expected_metadata must be a mapping")

    metadata = dict(value)
    missing = [
        field_name
        for field_name in TYPED_METADATA_REQUIRED_FIELDS
        if field_name not in metadata
    ]
    if missing:
        raise ValueError(
            f"typed metadata for {operator_name!r} is missing required fields: {', '.join(missing)}"
        )

    from frontier.model_architectures import ExpertParallelMode, LayerDimensionSource, LayerKind
    from frontier.operators.spec import TensorParallelMode

    for field_name in ("profile_id", "operator_family_id"):
        field_value = metadata[field_name]
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"typed metadata field {field_name!r} for {operator_name!r} "
                f"must be a non-empty string, got {field_value!r}"
            )

    family_ids = metadata["operator_family_ids"]
    if isinstance(family_ids, (str, bytes)) or not isinstance(family_ids, Sequence):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} "
            "must be a sequence of strings"
        )
    family_ids = tuple(family_ids)
    if not family_ids or any(not isinstance(item, str) or not item.strip() for item in family_ids):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} "
            "must contain non-empty strings"
        )
    if len(set(family_ids)) != len(family_ids):
        raise ValueError(
            f"typed metadata field 'operator_family_ids' for {operator_name!r} contains duplicates"
        )
    if metadata["operator_family_id"] not in family_ids:
        raise ValueError(
            f"typed metadata for {operator_name!r} names an operator family outside operator_family_ids"
        )

    layer_kind = metadata["layer_kind"]
    if layer_kind is not None:
        layer_kind = _enum_value(layer_kind, LayerKind, field_name="layer_kind")
    dimension_source = metadata["dimension_source"]
    if dimension_source is not None:
        dimension_source = _enum_value(
            dimension_source, LayerDimensionSource, field_name="dimension_source"
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
        _require_positive_int(effective_width, field_name="effective_ffn_width")
    _require_optional_positive_int(
        metadata["selected_expert_parallel_size"],
        field_name="selected_expert_parallel_size",
    )
    tensor_parallel_sizes = _normalize_positive_ints(
        metadata["tensor_parallel_sizes"], field_name="tensor_parallel_sizes"
    )
    selected_tp = metadata["selected_tensor_parallel_size"]
    _require_optional_positive_int(selected_tp, field_name="selected_tensor_parallel_size")
    if selected_tp is not None and selected_tp not in tensor_parallel_sizes:
        raise ValueError(
            f"typed metadata for {operator_name!r} selects TP={selected_tp} outside "
            f"tensor_parallel_sizes={list(tensor_parallel_sizes)}"
        )
    padded_width = metadata["selected_padded_ffn_width"]
    _require_optional_positive_int(padded_width, field_name="selected_padded_ffn_width")

    if layer_kind is None:
        if dimension_source is not None or effective_width is not None or padded_width is not None:
            raise ValueError(
                f"typed metadata for non-layer operator {operator_name!r} must use null width fields"
            )
    else:
        expected_sources = {
            LayerKind.DENSE.value: LayerDimensionSource.DENSE.value,
            LayerKind.ROUTED.value: LayerDimensionSource.ROUTED.value,
            LayerKind.SHARED.value: LayerDimensionSource.SHARED.value,
        }
        if dimension_source is None or effective_width is None:
            raise ValueError(
                f"typed metadata for layer-bound operator {operator_name!r} requires width fields"
            )
        if expected_sources.get(layer_kind) != dimension_source:
            raise ValueError(
                f"typed metadata for {operator_name!r} pairs layer_kind={layer_kind!r} "
                f"with dimension_source={dimension_source!r}"
            )
        if selected_tp is not None and padded_width is None:
            raise ValueError(
                f"typed metadata field 'selected_padded_ffn_width' for {operator_name!r} "
                "is required when TP is selected"
            )
        if padded_width is not None and padded_width < effective_width:
            raise ValueError(
                f"typed metadata field 'selected_padded_ffn_width' for {operator_name!r} "
                "has padded width below effective width"
            )
    if selected_tp is not None and padded_width is not None and padded_width % selected_tp != 0:
        raise ValueError(
            f"typed metadata field 'selected_padded_ffn_width' for {operator_name!r} "
            "has padded width not divisible by selected TP"
        )
    if expected_tensor_parallel_sizes is not None:
        expected_domain = _normalize_positive_ints(
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
                f"typed metadata for {operator_name!r} is missing expected field {field_name!r}"
            )
        actual_value = metadata[field_name]
        if field_name == "operator_family_ids":
            actual_value = tuple(actual_value)
            expected_value = tuple(expected_value)
        elif field_name in enum_fields and expected_value is not None:
            expected_value = _enum_value(expected_value, enum_fields[field_name], field_name=field_name)
        if not typed_metadata_value_matches(actual_value, expected_value):
            raise ValueError(
                f"typed metadata for {operator_name!r} field {field_name!r} disagrees: "
                f"actual={actual_value!r}, expected={expected_value!r}"
            )
    return metadata


def serialize_typed_operator_contracts(value: Mapping[Any, Any]) -> str:
    """Serialize a contract mapping as deterministic canonical JSON."""

    if not isinstance(value, Mapping):
        raise ValueError("typed_operator_contracts must be a mapping")
    normalized = _copy_contract_mapping(value)
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("typed_operator_contracts contains a non-JSON value") from exc


def _missing_scalar(value: Any) -> bool:
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    ) or (
        isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}
    )


def parse_typed_operator_contracts(value: Any) -> dict[str, dict[str, Any]]:
    """Parse one mapping or its canonical JSON representation."""

    if isinstance(value, Mapping):
        return _copy_contract_mapping(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("typed contract JSON must be UTF-8") from exc
    if not isinstance(value, str) or _missing_scalar(value):
        raise ValueError("typed_operator_contracts must contain a non-empty JSON object")
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("typed_operator_contracts must use JSON object syntax") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("typed_operator_contracts JSON value must be an object")
    return _copy_contract_mapping(decoded)


def validate_typed_operator_contracts(value: Any) -> dict[str, dict[str, Any]]:
    """Parse and validate every operator metadata object in one typed row.

    Parsing validates the JSON envelope and mapping shape. Loader admission
    additionally needs the complete current metadata schema for every operator,
    including operators outside the caller's selected query. Keep that stricter
    pass here so every consumer uses one canonical validation boundary.
    """

    contracts = parse_typed_operator_contracts(value)
    for operator_name, metadata in contracts.items():
        validate_typed_operator_metadata(
            metadata,
            operator_name=operator_name,
            expected_metadata={},
        )
    return contracts


def matches_resolved_layer_contract(value: Any, layer_contract: Any, *, operator_name: str) -> bool:
    """Validate and compare one operator's typed metadata with a resolved contract."""

    if not hasattr(layer_contract, "typed_metadata_identity"):
        raise TypeError("layer_contract must expose typed_metadata_identity()")
    contracts = parse_typed_operator_contracts(value)
    metadata = contracts.get(operator_name)
    if metadata is None:
        return False
    expected = layer_contract.typed_metadata_identity()
    validate_typed_operator_metadata(
        metadata,
        operator_name=operator_name,
        expected_metadata={},
    )
    expected_family_ids = tuple(expected["operator_family_ids"])
    if metadata["operator_family_id"] != expected["operator_family_id"]:
        return False
    if not set(expected_family_ids).issubset(tuple(metadata["operator_family_ids"])):
        return False
    expected_tp_sizes = tuple(expected["tensor_parallel_sizes"])
    actual_tp_sizes = tuple(metadata["tensor_parallel_sizes"])
    if expected["selected_tensor_parallel_size"] is None:
        if actual_tp_sizes != expected_tp_sizes:
            return False
    elif not set(expected_tp_sizes).issubset(actual_tp_sizes):
        return False
    for field_name in (
        "profile_id",
        "layer_kind",
        "dimension_source",
        "effective_ffn_width",
        "tensor_parallel_mode",
        "expert_parallel_mode",
        "selected_expert_parallel_size",
        "selected_tensor_parallel_size",
        "selected_padded_ffn_width",
    ):
        if not typed_metadata_value_matches(metadata[field_name], expected[field_name]):
            return False
    return True


def serialize_typed_operator_contract_column(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    result = frame.copy()
    if TYPED_OPERATOR_CONTRACTS_COLUMN in result.columns:
        result[TYPED_OPERATOR_CONTRACTS_COLUMN] = result[
            TYPED_OPERATOR_CONTRACTS_COLUMN
        ].map(serialize_typed_operator_contracts)
    return result


def parse_typed_operator_contract_column(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    result = frame.copy()
    if TYPED_OPERATOR_CONTRACTS_COLUMN in result.columns:
        result[TYPED_OPERATOR_CONTRACTS_COLUMN] = result[
            TYPED_OPERATOR_CONTRACTS_COLUMN
        ].map(parse_typed_operator_contracts)
    return result


__all__ = [
    "TYPED_METADATA_REQUIRED_FIELDS",
    "TYPED_OPERATOR_CONTRACTS_COLUMN",
    "matches_resolved_layer_contract",
    "parse_typed_operator_contract_column",
    "parse_typed_operator_contracts",
    "serialize_typed_operator_contract_column",
    "serialize_typed_operator_contracts",
    "typed_metadata_value_matches",
    "validate_typed_operator_contracts",
    "validate_typed_operator_metadata",
]
