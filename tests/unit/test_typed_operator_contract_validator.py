"""Direct coverage for the strict typed operator metadata validator."""

from __future__ import annotations

from copy import deepcopy

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.operators.typed_contracts import (
    TYPED_METADATA_REQUIRED_FIELDS,
    build_non_layer_typed_operator_expectation,
    validate_typed_operator_metadata,
)
from frontier.profiling.common.typed_contracts import (
    TYPED_METADATA_REQUIRED_FIELDS as PROFILING_TYPED_METADATA_REQUIRED_FIELDS,
)


def test_profiling_and_runtime_use_one_typed_metadata_field_list() -> None:
    assert PROFILING_TYPED_METADATA_REQUIRED_FIELDS is TYPED_METADATA_REQUIRED_FIELDS


def _layer_metadata(layer_kind: str) -> dict[str, object]:
    contracts = {
        "dense": {
            "operator_name": "mlp_up_proj",
            "operator_family_id": "ffn",
            "dimension_source": "dense_mlp_hidden_dim",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 18432,
        },
        "routed": {
            "operator_name": "moe_grouped_gemm",
            "operator_family_id": "moe",
            "dimension_source": "routed_mlp_hidden_dim",
            "effective_ffn_width": 5120,
            "tensor_parallel_mode": "moe_tp",
            "expert_parallel_mode": "on",
            "selected_expert_parallel_size": 8,
            "tensor_parallel_sizes": [1],
            "selected_tensor_parallel_size": None,
            "selected_padded_ffn_width": None,
        },
        "shared": {
            "operator_name": "share_expert_up_proj",
            "operator_family_id": "share_expert",
            "dimension_source": "share_expert_dim",
            "effective_ffn_width": 5120,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 5120,
        },
    }
    selected = contracts[layer_kind]
    operator_name = selected.pop("operator_name")
    family_id = selected["operator_family_id"]
    return {
        "operator_name": operator_name,
        "profile_id": "step3_text",
        "operator_family_id": family_id,
        "operator_family_ids": [family_id],
        "layer_kind": layer_kind,
        **selected,
    }


def _non_layer_attention_metadata() -> dict[str, object]:
    return {
        "operator_name": "attn_pre_proj",
        "profile_id": "step3_text",
        "operator_family_id": "dense_attention",
        "operator_family_ids": ["dense_attention"],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": "attention_tp",
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": [8],
        "selected_tensor_parallel_size": 8,
        "selected_padded_ffn_width": None,
    }


def _validate(metadata: dict[str, object]) -> dict[str, object]:
    expected = {
        key: value for key, value in metadata.items() if key != "operator_name"
    }
    return validate_typed_operator_metadata(
        metadata,
        operator_name=metadata["operator_name"],
        expected_metadata=expected,
    )


@pytest.mark.parametrize("layer_kind", ["dense", "routed", "shared"])
def test_validator_accepts_each_profile_owned_layer_contract(
    layer_kind: str,
) -> None:
    metadata = _layer_metadata(layer_kind)

    validated = _validate(metadata)
    assert all(validated[key] == value for key, value in metadata.items())


def test_validator_accepts_non_layer_attention_contract() -> None:
    metadata = _non_layer_attention_metadata()

    assert _validate(metadata)["layer_kind"] is None


def test_validator_rejects_missing_required_field() -> None:
    metadata = _layer_metadata("dense")
    del metadata["tensor_parallel_sizes"]

    with pytest.raises(ValueError, match="tensor_parallel_sizes"):
        _validate(metadata)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("effective_ffn_width", "18432"),
        ("selected_tensor_parallel_size", True),
        ("tensor_parallel_sizes", [8, 8]),
    ],
)
def test_validator_rejects_invalid_field_types_or_domains(
    field_name: str,
    bad_value: object,
) -> None:
    metadata = _layer_metadata("dense")
    metadata[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        _validate(metadata)


def test_validator_rejects_selected_tp_outside_declared_domain() -> None:
    metadata = _layer_metadata("dense")
    metadata["selected_tensor_parallel_size"] = 4

    with pytest.raises(ValueError, match="outside tensor_parallel_sizes"):
        _validate(metadata)


def test_validator_rejects_width_source_mismatch() -> None:
    metadata = _layer_metadata("dense")
    metadata["dimension_source"] = "routed_mlp_hidden_dim"

    with pytest.raises(ValueError, match="incompatible dimension_source"):
        _validate(metadata)


def test_validator_rejects_layer_row_without_selected_padded_width() -> None:
    metadata = _layer_metadata("dense")
    metadata["selected_padded_ffn_width"] = None

    with pytest.raises(ValueError, match="selected_padded_ffn_width"):
        _validate(metadata)


def test_validator_rejects_non_layer_width_fields() -> None:
    metadata = _non_layer_attention_metadata()
    metadata["effective_ffn_width"] = 5120

    with pytest.raises(ValueError, match="non-layer"):
        _validate(metadata)


def test_validator_can_enforce_a_known_tp_domain() -> None:
    metadata = _non_layer_attention_metadata()

    with pytest.raises(ValueError, match=r"expected \[2, 4, 8\]"):
        validate_typed_operator_metadata(
            metadata,
            operator_name=metadata["operator_name"],
            expected_metadata={
                key: value
                for key, value in metadata.items()
                if key != "operator_name"
            },
            expected_tensor_parallel_sizes=[2, 4, 8],
        )


def test_validator_does_not_mutate_input_mapping() -> None:
    metadata = _layer_metadata("routed")
    original = deepcopy(metadata)

    _validate(metadata)

    assert metadata == original


@pytest.mark.parametrize(
    ("operator_name", "selected_tp", "expected_domain"),
    [
        ("attn_pre_proj", 8, [8]),
        ("post_attention_layernorm", 8, [1]),
    ],
)
def test_non_layer_expectation_declares_complete_tp_domain(
    operator_name: str,
    selected_tp: int,
    expected_domain: list[int],
) -> None:
    """Non-layer expectations must satisfy the same complete schema as rows."""

    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    expectation = build_non_layer_typed_operator_expectation(
        operator_name=operator_name,
        model_config=model_config,
        architecture_profile=model_config.get_model_architecture_profile(),
        selected_tensor_parallel_size=selected_tp,
    )

    assert expectation["tensor_parallel_sizes"] == expected_domain
    validated = validate_typed_operator_metadata(
        expectation,
        operator_name=operator_name,
        expected_metadata=expectation,
    )
    assert validated["tensor_parallel_sizes"] == expected_domain
