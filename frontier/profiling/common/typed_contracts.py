"""Compatibility imports for the canonical typed profiling contract schema."""

from frontier.operators.typed_contracts import (
    TYPED_METADATA_REQUIRED_FIELDS,
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    matches_resolved_layer_contract,
    parse_typed_operator_contract_column,
    parse_typed_operator_contracts,
    serialize_typed_operator_contract_column,
    serialize_typed_operator_contracts,
    typed_metadata_value_matches,
    validate_typed_operator_contracts,
    validate_typed_operator_metadata,
)

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
