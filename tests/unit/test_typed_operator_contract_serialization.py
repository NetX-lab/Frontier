"""Regression tests for the typed operator contract CSV representation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import LayerKind
from frontier.operators.typed_contracts import matches_resolved_layer_contract
from frontier.profiling.linear_op import main as linear_op_main
from tests.performance.profiling.run_h200_frozen_manifest_model import _read_csv
from frontier.profiling.common.typed_contracts import (
    parse_typed_operator_contract_column,
    parse_typed_operator_contracts,
    serialize_typed_operator_contract_column,
    serialize_typed_operator_contracts,
)


def _contract_payload() -> dict[str, dict[str, object]]:
    return {
        "mlp_up_proj": {
            "operator_family_id": "ffn",
            "layer_kind": "dense",
            "effective_ffn_width": 18432,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "expert_parallel_mode": "off",
        },
        "moe_grouped_gemm": {
            "operator_family_id": "moe",
            "layer_kind": "routed",
            "effective_ffn_width": 5120,
            "tensor_parallel_sizes": [1],
            "selected_tensor_parallel_size": None,
            "expert_parallel_mode": "on",
        },
    }


def test_typed_operator_contracts_use_canonical_json() -> None:
    payload = _contract_payload()

    encoded = serialize_typed_operator_contracts(payload)

    assert encoded == serialize_typed_operator_contracts(payload)
    assert json.loads(encoded) == payload
    assert "'mlp_up_proj'" not in encoded
    assert parse_typed_operator_contracts(encoded) == payload


@pytest.mark.parametrize("value", [{}, "{}"])
def test_typed_operator_contracts_reject_empty_mappings(value: object) -> None:
    """An empty typed column must not silently discard all operator timing columns."""

    with pytest.raises(ValueError, match="at least one operator contract"):
        parse_typed_operator_contracts(value)


def test_typed_operator_contract_serializer_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one operator contract"):
        serialize_typed_operator_contracts({})


def test_typed_operator_contract_column_round_trips_through_csv(tmp_path) -> None:
    payload = _contract_payload()
    frame = pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [payload],
        }
    )

    serialized = serialize_typed_operator_contract_column(frame)
    path = tmp_path / "linear_op.csv"
    serialized.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    parsed = parse_typed_operator_contract_column(loaded)

    assert parsed.loc[0, "typed_operator_contracts"] == payload


@pytest.mark.parametrize(
    "value",
    [
        "{'mlp_up_proj': {'layer_kind': 'dense'}}",
        "[]",
        "not-json",
        "",
    ],
)
def test_typed_operator_contract_parser_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="typed_operator_contracts"):
        parse_typed_operator_contracts(value)


def test_typed_operator_contract_column_is_unchanged_when_absent() -> None:
    frame = pd.DataFrame({"num_tokens": [2]})

    assert serialize_typed_operator_contract_column(frame).equals(frame)
    assert parse_typed_operator_contract_column(frame).equals(frame)


def test_linear_output_boundary_serializes_typed_contracts() -> None:
    frame = pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [_contract_payload()],
        }
    )

    output = linear_op_main._serialize_linear_op_output(frame)  # pylint: disable=protected-access

    assert isinstance(output.loc[0, "typed_operator_contracts"], str)
    assert parse_typed_operator_contracts(
        output.loc[0, "typed_operator_contracts"]
    ) == _contract_payload()


def test_predictor_loader_validates_typed_contracts(tmp_path) -> None:
    path = tmp_path / "linear_op.csv"
    frame = pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts(_contract_payload())
            ],
        }
    )
    frame.to_csv(path, index=False)

    loaded = SklearnExecutionTimePredictor._read_input_file(  # pylint: disable=protected-access
        object(), str(path)
    )

    assert loaded.loc[0, "typed_operator_contracts"] == frame.loc[
        0, "typed_operator_contracts"
    ]


def test_predictor_loader_rejects_python_repr_typed_contracts(tmp_path) -> None:
    path = tmp_path / "linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [
                "{'mlp_up_proj': {'layer_kind': 'dense'}}"
            ],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="typed_operator_contracts"):
        SklearnExecutionTimePredictor._read_input_file(  # pylint: disable=protected-access
            object(), str(path)
        )


def test_frozen_profile_reader_rejects_python_repr_typed_contracts(tmp_path) -> None:
    path = tmp_path / "linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [
                "{'mlp_up_proj': {'layer_kind': 'dense'}}"
            ],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="typed_operator_contracts"):
        _read_csv(path)


def test_frozen_profile_reader_accepts_canonical_typed_contracts(tmp_path) -> None:
    path = tmp_path / "linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [2],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts(_contract_payload())
            ],
        }
    ).to_csv(path, index=False)

    loaded = _read_csv(path)

    assert loaded.loc[0, "typed_operator_contracts"] == serialize_typed_operator_contracts(
        _contract_payload()
    )


def _step3_contract(*, layer_kind: LayerKind, expert_parallel_size: int | None):
    config = BaseModelConfig.create_from_name("step3-moe-noquant")
    return config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=4 if layer_kind is LayerKind.ROUTED else 0,
        layer_kind=layer_kind,
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=expert_parallel_size,
    )


def test_typed_metadata_identity_carries_selected_expert_parallel_size() -> None:
    contract = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)

    assert contract.typed_metadata_identity()["selected_expert_parallel_size"] == 8


def test_typed_metadata_identity_carries_complete_parallel_and_family_fields() -> None:
    """The profile-owned identity must expose every producer admission field."""

    contract = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)
    metadata = contract.typed_metadata_identity()

    assert metadata["operator_family_ids"] == ["moe"]
    assert metadata["tensor_parallel_sizes"] == [1]
    assert metadata["selected_padded_ffn_width"] == 5120


def test_typed_matcher_rejects_metadata_from_a_different_expert_parallel_size() -> None:
    expected = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)
    wrong_ep = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=2)

    assert not matches_resolved_layer_contract(
        {"moe_grouped_gemm": wrong_ep.typed_metadata_identity()},
        expected,
        operator_name="moe_grouped_gemm",
    )


def test_typed_matcher_accepts_exact_expert_parallel_size() -> None:
    expected = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)

    assert matches_resolved_layer_contract(
        {"moe_grouped_gemm": expected.typed_metadata_identity()},
        expected,
        operator_name="moe_grouped_gemm",
    )


def test_typed_matcher_rejects_float_for_integer_width() -> None:
    expected = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    metadata["effective_ffn_width"] = float(metadata["effective_ffn_width"])

    with pytest.raises(ValueError, match="effective_ffn_width"):
        matches_resolved_layer_contract(
            {"moe_grouped_gemm": metadata},
            expected,
            operator_name="moe_grouped_gemm",
        )


def test_typed_matcher_rejects_bool_for_integer_expert_parallel_size() -> None:
    expected = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=1)
    metadata = expected.typed_metadata_identity()
    # ``True == 1`` in Python; the contract still requires the exact integer
    # representation emitted by the profiling producer.
    metadata["selected_expert_parallel_size"] = True

    with pytest.raises(ValueError, match="selected_expert_parallel_size"):
        matches_resolved_layer_contract(
            {"moe_grouped_gemm": metadata},
            expected,
            operator_name="moe_grouped_gemm",
        )


def test_typed_matcher_rejects_missing_selected_expert_parallel_size() -> None:
    expected = _step3_contract(layer_kind=LayerKind.ROUTED, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    metadata.pop("selected_expert_parallel_size", None)

    with pytest.raises(ValueError, match="selected_expert_parallel_size"):
        matches_resolved_layer_contract(
            {"moe_grouped_gemm": metadata},
            expected,
            operator_name="moe_grouped_gemm",
        )


def test_typed_matcher_requires_explicit_none_for_expert_parallel_off() -> None:
    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    assert metadata["selected_expert_parallel_size"] is None
    metadata.pop("selected_expert_parallel_size")

    with pytest.raises(ValueError, match="selected_expert_parallel_size"):
        matches_resolved_layer_contract(
            {"mlp_up_proj": metadata},
            expected,
            operator_name="mlp_up_proj",
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("operator_family_ids", ["moe"]),
        ("tensor_parallel_sizes", [1]),
        ("selected_padded_ffn_width", 9999),
    ],
)
def test_typed_matcher_rejects_wrong_complete_contract_fields(
    field_name: str, wrong_value: object
) -> None:
    """Loader matching must cover fields beyond the compact contract identity."""

    config = BaseModelConfig.create_from_name("step3-moe-noquant")
    expected = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = expected.typed_metadata_identity()
    metadata.update(
        {
            "operator_family_ids": [expected.operator_family_id],
            "tensor_parallel_sizes": [expected.tensor_parallel_size],
            "selected_padded_ffn_width": expected.effective_ffn_width,
        }
    )
    metadata[field_name] = wrong_value

    with pytest.raises(ValueError, match=field_name):
        matches_resolved_layer_contract(
            {"mlp_up_proj": metadata},
            expected,
            operator_name="mlp_up_proj",
        )


def test_typed_metadata_identity_contains_all_row_admission_fields() -> None:
    """The profile resolver must emit the complete canonical row identity."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)

    assert set(expected.typed_metadata_identity()) == {
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
    }


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("operator_family_id", "moe"),
        ("effective_ffn_width", 5120),
        ("selected_padded_ffn_width", 20480),
    ],
)
def test_typed_matcher_returns_false_for_valid_sibling_extended_fields(
    field_name: str, wrong_value: object
) -> None:
    """A valid sibling domain is filtered, not treated as malformed metadata."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    if field_name == "operator_family_id":
        # Keep the row structurally valid while selecting a sibling family.
        metadata["operator_family_ids"] = ["moe"]
    metadata[field_name] = wrong_value

    assert not matches_resolved_layer_contract(
        {"mlp_up_proj": metadata},
        expected,
        operator_name="mlp_up_proj",
    )


def test_typed_matcher_returns_false_for_valid_sibling_tp_domain() -> None:
    """A valid sibling TP envelope is filtered, not treated as malformed."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    metadata["tensor_parallel_sizes"] = [1, 2, 4]
    metadata["selected_tensor_parallel_size"] = 4

    assert not matches_resolved_layer_contract(
        {"mlp_up_proj": metadata},
        expected,
        operator_name="mlp_up_proj",
    )


@pytest.mark.parametrize(
    ("field_name", "error_match", "mutate"),
    [
        (
            "operator_family_id",
            "operator_family_id",
            lambda metadata: metadata.update(
                {"operator_family_ids": ["moe"]}
            ),
        ),
        (
            "selected_tensor_parallel_size",
            "outside tensor_parallel_sizes",
            lambda metadata: metadata.update(
                {"tensor_parallel_sizes": [1]}
            ),
        ),
        (
            "selected_padded_ffn_width",
            "selected_padded_ffn_width",
            lambda metadata: metadata.update(
                {"selected_padded_ffn_width": 1024}
            ),
        ),
    ],
)
def test_typed_matcher_rejects_malformed_selected_metadata(
    field_name: str, error_match: str, mutate
) -> None:
    """Malformed row structure fails before sibling-domain filtering."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    if field_name == "operator_family_id":
        metadata["operator_family_id"] = "ffn"
    elif field_name == "selected_tensor_parallel_size":
        metadata["selected_tensor_parallel_size"] = 8
    mutate(metadata)

    with pytest.raises(ValueError, match=error_match):
        matches_resolved_layer_contract(
            {"mlp_up_proj": metadata},
            expected,
            operator_name="mlp_up_proj",
        )


def test_typed_matcher_rejects_tp_domain_without_selected_size() -> None:
    """A row whose declared TP envelope omits the selected TP is malformed."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    metadata["tensor_parallel_sizes"] = [1, 2, 4]

    with pytest.raises(ValueError, match="outside tensor_parallel_sizes"):
        matches_resolved_layer_contract(
            {"mlp_up_proj": metadata},
            expected,
            operator_name="mlp_up_proj",
        )


def test_typed_matcher_accepts_complete_producer_tp_domain() -> None:
    """A selected TP query can consume a row carrying the full producer envelope."""

    expected = _step3_contract(layer_kind=LayerKind.DENSE, expert_parallel_size=8)
    metadata = expected.typed_metadata_identity()
    metadata["operator_family_ids"] = ["ffn", "future_ffn_alias"]
    metadata["tensor_parallel_sizes"] = [1, 2, 4, 8]

    assert matches_resolved_layer_contract(
        {"mlp_up_proj": metadata},
        expected,
        operator_name="mlp_up_proj",
    )
