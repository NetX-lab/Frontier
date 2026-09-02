"""Regression coverage for strict typed metadata admission in AttentionTrainer."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import numpy as np

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.training.attention_trainer import AttentionTrainer
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    serialize_typed_operator_contracts,
)


def _trainer_and_contract() -> tuple[AttentionTrainer, dict[str, object]]:
    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.model_config = config
    trainer.compute_dataset_path = "typed-attention.csv"
    return trainer, plan["typed_operator_contracts"]["attn_pre_proj"]


def _frame(metadata: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_stats.attn_pre_proj.median": [1.0],
            TYPED_OPERATOR_CONTRACTS_COLUMN: [
                serialize_typed_operator_contracts({"attn_pre_proj": metadata})
            ],
        }
    )


def test_attention_trainer_accepts_complete_non_layer_typed_contract() -> None:
    trainer, metadata = _trainer_and_contract()

    trainer._validate_typed_compute_dataset(_frame(metadata))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("operator_family_id", "memory"),
        ("layer_kind", "dense"),
        ("dimension_source", "dense_mlp_hidden_dim"),
        ("effective_ffn_width", 1024),
        ("tensor_parallel_mode", "replicated"),
        ("expert_parallel_mode", "on"),
        ("selected_tensor_parallel_size", 4),
    ],
)
def test_attention_trainer_rejects_wrong_non_layer_typed_contract_field(
    field_name: str,
    bad_value: object,
) -> None:
    trainer, metadata = _trainer_and_contract()
    malformed = deepcopy(metadata)
    malformed[field_name] = bad_value

    with pytest.raises(ValueError, match="typed metadata"):
        trainer._validate_typed_compute_dataset(_frame(malformed))


def test_attention_trainer_rejects_missing_non_layer_typed_contract_field() -> None:
    trainer, metadata = _trainer_and_contract()
    malformed = deepcopy(metadata)
    del malformed["tensor_parallel_sizes"]

    with pytest.raises(ValueError, match="typed metadata"):
        trainer._validate_typed_compute_dataset(_frame(malformed))


def test_attention_trainer_rejects_operator_family_list_mismatch() -> None:
    trainer, metadata = _trainer_and_contract()
    malformed = deepcopy(metadata)
    malformed["operator_family_ids"] = ["memory"]

    with pytest.raises(ValueError, match="typed metadata"):
        trainer._validate_typed_compute_dataset(_frame(malformed))


@pytest.mark.parametrize("missing_value", [pd.NA, float("nan"), None])
def test_attention_trainer_typed_metadata_row_accepts_scalar_missing_tp(
    missing_value: object,
) -> None:
    trainer, _ = _trainer_and_contract()
    trainer.tensor_parallel_size = 8

    row = pd.Series({"num_tensor_parallel_workers": missing_value})

    assert trainer._typed_metadata_row_tp_size(row) == 8


def test_attention_trainer_typed_metadata_row_rejects_non_scalar_tp() -> None:
    trainer, _ = _trainer_and_contract()
    row = pd.Series({"num_tensor_parallel_workers": np.array([8, 16])})

    with pytest.raises(ValueError, match="num_tensor_parallel_workers"):
        trainer._typed_metadata_row_tp_size(row)
