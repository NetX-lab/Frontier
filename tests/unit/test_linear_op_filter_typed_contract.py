"""Regression coverage for typed linear-op column filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from frontier.operators.typed_contracts import serialize_typed_operator_contracts
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan

from frontier.profiling.linear_op.main import filter_mlp_columns


def _frame_with_contracts(contracts: dict[str, dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_stats.mlp_up_proj.mean": [1.0],
            "time_stats.mlp_down_proj.mean": [2.0],
            "time_stats.mlp_act.mean": [3.0],
            "time_stats.moe_grouped_gemm.mean": [4.0],
            "time_stats.share_expert_up_proj.mean": [5.0],
            "typed_operator_contracts": [contracts],
        }
    )


def _dense_contracts() -> dict[str, dict[str, object]]:
    return {
        "mlp_up_proj": {
            "profile_id": "test_profile",
            "operator_family_id": "ffn",
            "operator_family_ids": ["ffn"],
            "layer_kind": "dense",
            "dimension_source": "dense_mlp_hidden_dim",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 18432,
        },
        "mlp_down_proj": {
            "profile_id": "test_profile",
            "operator_family_id": "ffn",
            "operator_family_ids": ["ffn"],
            "layer_kind": "dense",
            "dimension_source": "dense_mlp_hidden_dim",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 18432,
        },
        "mlp_act": {
            "profile_id": "test_profile",
            "operator_family_id": "ffn",
            "operator_family_ids": ["ffn"],
            "layer_kind": "dense",
            "dimension_source": "dense_mlp_hidden_dim",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 18432,
        },
        "moe_grouped_gemm": {
            "profile_id": "test_profile",
            "operator_family_id": "moe",
            "operator_family_ids": ["moe"],
            "layer_kind": "routed",
            "dimension_source": "routed_mlp_hidden_dim",
            "effective_ffn_width": 5120,
            "tensor_parallel_mode": "moe_tp",
            "expert_parallel_mode": "on",
            "selected_expert_parallel_size": 8,
            "tensor_parallel_sizes": [1],
            "selected_tensor_parallel_size": None,
            "selected_padded_ffn_width": None,
        },
        "share_expert_up_proj": {
            "profile_id": "test_profile",
            "operator_family_id": "share_expert",
            "operator_family_ids": ["share_expert"],
            "layer_kind": "shared",
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


def test_pure_routed_typed_contract_filters_dense_mlp_columns() -> None:
    """A pure routed profile keeps routed/shared columns and drops dense MLP."""

    contracts = _dense_contracts()
    contracts = {
        name: metadata
        for name, metadata in contracts.items()
        if metadata["operator_family_id"] != "ffn"
    }
    frame = _frame_with_contracts(contracts)

    filtered = filter_mlp_columns(frame)

    assert "time_stats.mlp_up_proj.mean" not in filtered.columns
    assert "time_stats.mlp_down_proj.mean" not in filtered.columns
    assert "time_stats.mlp_act.mean" not in filtered.columns
    assert "time_stats.moe_grouped_gemm.mean" in filtered.columns
    assert "time_stats.share_expert_up_proj.mean" in filtered.columns
    assert "typed_operator_contracts" in filtered.columns


def test_mixed_typed_contract_preserves_dense_boundary_columns() -> None:
    """A mixed profile retains dense boundary MLP columns declared by its contract."""

    frame = _frame_with_contracts(_dense_contracts())

    filtered = filter_mlp_columns(frame)

    assert "time_stats.mlp_up_proj.mean" in filtered.columns
    assert "time_stats.mlp_down_proj.mean" in filtered.columns
    assert "time_stats.mlp_act.mean" in filtered.columns
    assert "time_stats.moe_grouped_gemm.mean" in filtered.columns


def test_typed_contracts_can_be_loaded_from_canonical_json() -> None:
    """CSV-shaped typed metadata remains authoritative after serialization."""

    frame = _frame_with_contracts(_dense_contracts())
    frame["typed_operator_contracts"] = frame["typed_operator_contracts"].map(
        lambda value: serialize_typed_operator_contracts(
            {"mlp_up_proj": value["mlp_up_proj"]}
        )
    )

    filtered = filter_mlp_columns(frame)

    assert "time_stats.mlp_up_proj.mean" in filtered.columns


def test_legacy_rows_without_typed_contracts_keep_historical_filter() -> None:
    """Rows from legacy producers retain the old dense-MoE filtering behavior."""

    frame = _frame_with_contracts(_dense_contracts()).drop(
        columns=["typed_operator_contracts"]
    )

    filtered = filter_mlp_columns(frame)

    assert "time_stats.mlp_up_proj.mean" not in filtered.columns
    assert "time_stats.mlp_down_proj.mean" not in filtered.columns


def test_partial_typed_metadata_fails_at_producer_admission() -> None:
    """Typed producer rows must contain the complete shared metadata schema."""

    frame = _frame_with_contracts(
        {
            "mlp_up_proj": {
                "operator_family_id": "ffn",
                "layer_kind": "dense",
            }
        }
    )

    with pytest.raises(ValueError, match="required fields"):
        filter_mlp_columns(frame)


def test_filter_rejects_empty_typed_mapping_instead_of_dropping_mlp_columns() -> None:
    frame = pd.DataFrame(
        {
            "typed_operator_contracts": ["{}"],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    )

    with pytest.raises(ValueError, match="at least one operator contract"):
        filter_mlp_columns(frame)


def test_wrong_registered_family_fails_at_producer_admission() -> None:
    """A complete row cannot assign an FFN operator to another family."""

    metadata = _dense_contracts()["mlp_up_proj"].copy()
    metadata["operator_family_id"] = "moe"
    metadata["operator_family_ids"] = ["moe"]

    with pytest.raises(ValueError, match="does not register"):
        filter_mlp_columns(_frame_with_contracts({"mlp_up_proj": metadata}))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tensor_parallel_sizes", [8, 8]),
        ("selected_padded_ffn_width", 1024),
    ],
)
def test_invalid_typed_domain_fails_at_producer_admission(
    field_name: str,
    value: object,
) -> None:
    """Invalid TP domains and padded widths fail with field context."""

    metadata = _dense_contracts()["mlp_up_proj"].copy()
    metadata[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        filter_mlp_columns(_frame_with_contracts({"mlp_up_proj": metadata}))


def test_profile_owned_attention_aliases_do_not_block_mlp_filter() -> None:
    """Non-FFN profile aliases remain available to their own consumers."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )
    frame = pd.DataFrame(
        {
            "time_stats.mlp_up_proj.mean": [1.0],
            "typed_operator_contracts": [plan["typed_operator_contracts"]],
        }
    )

    filtered = filter_mlp_columns(frame, model_config=config)

    assert "time_stats.mlp_up_proj.mean" in filtered.columns


def test_profile_owned_alias_with_partial_metadata_fails_admission() -> None:
    """Alias rows still use the shared strict schema outside FFN filtering."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )
    contracts = {
        operator_name: dict(metadata)
        for operator_name, metadata in plan["typed_operator_contracts"].items()
    }
    del contracts["attn_pre_proj_qkv"]["profile_id"]
    frame = pd.DataFrame(
        {
            "time_stats.mlp_up_proj.mean": [1.0],
            "typed_operator_contracts": [contracts],
        }
    )

    with pytest.raises(ValueError, match="profile_id"):
        filter_mlp_columns(frame, model_config=config)
