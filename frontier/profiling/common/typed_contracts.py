"""Backward-compatible profiling import for typed operator serialization.

The implementation lives in :mod:`frontier.operators.typed_contracts` so
runtime predictors do not depend on profiling modules. Existing profiling
scripts may continue importing this compatibility surface.
"""

from frontier.operators.typed_contracts import (
    TYPED_METADATA_REQUIRED_FIELDS,
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    parse_typed_operator_contract_column,
    parse_typed_operator_contracts,
    serialize_typed_operator_contract_column,
    serialize_typed_operator_contracts,
)

__all__ = [
    "TYPED_OPERATOR_CONTRACTS_COLUMN",
    "TYPED_METADATA_REQUIRED_FIELDS",
    "parse_typed_operator_contract_column",
    "parse_typed_operator_contracts",
    "serialize_typed_operator_contract_column",
    "serialize_typed_operator_contracts",
]
