"""Focused regressions for the independent predictor's typed row loader."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ResolvedLayerContract
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.typed_contracts import (
    serialize_typed_operator_contracts,
)
from frontier.types import ClusterType, MeasurementType


class _LoaderProbePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def _config_and_contract() -> tuple[ModelConfig, ResolvedLayerContract]:
    config = ModelConfig.from_model_name("step3-moe-noquant")
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        ffn_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    return config, contract


def _predictor(config: ModelConfig, contract: ResolvedLayerContract):
    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._resolve_typed_linear_contract = lambda _name: contract
    predictor._get_compute_model_names = lambda: ["mlp_up_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: None
    predictor._validate_active_measurement_type = lambda _metadata, _path=None: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None
    return predictor


def _base_row(config: ModelConfig) -> dict[str, object]:
    return {
        "n_head": config.num_q_heads,
        "n_kv_head": config.num_kv_heads,
        "n_embd": config.embedding_dim,
        "n_expanded_embd": config.routed_mlp_hidden_dim,
        "use_gated_mlp": config.use_gated_mlp,
        "vocab_size": config.vocab_size,
        "num_tensor_parallel_workers": 8,
        "time_stats.mlp_up_proj.median": 1.0,
    }


def _typed_row(
    config: ModelConfig,
    contract: ResolvedLayerContract,
    **overrides: object,
) -> dict[str, object]:
    metadata = contract.typed_metadata_identity()
    metadata.update(
        {
            "operator_family_ids": ["ffn"],
            "tensor_parallel_sizes": [1, 2, 4, 8],
            "selected_padded_ffn_width": contract.effective_ffn_width,
        }
    )
    metadata.update(overrides)
    row = _base_row(config)
    row["typed_operator_contracts"] = serialize_typed_operator_contracts(
        {"mlp_up_proj": metadata}
    )
    return row


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operator_family_ids", ["moe"]),
        ("tensor_parallel_sizes", [1]),
        ("selected_padded_ffn_width", 36864),
    ),
)
def test_loader_rejects_typed_rows_with_wrong_domain_metadata(
    tmp_path,
    field: str,
    value: object,
) -> None:
    config, contract = _config_and_contract()
    path = tmp_path / f"wrong_{field}.csv"
    pd.DataFrame([_typed_row(config, contract, **{field: value})]).to_csv(
        path, index=False
    )

    # Invalid typed metadata must fail at the row-admission boundary with the
    # offending field still visible to the caller.
    with pytest.raises(ValueError, match=field):
        _predictor(config, contract)._load_compute_df(
            str(path),
            tensor_parallel_size=8,
            operator_name="mlp_up_proj",
            layer_contract=contract,
        )


def test_loader_keeps_legacy_no_column_rows_usable(tmp_path) -> None:
    config, contract = _config_and_contract()
    path = tmp_path / "legacy.csv"
    row = _base_row(config)
    row["n_expanded_embd"] = contract.effective_ffn_width
    pd.DataFrame([row]).to_csv(path, index=False)

    loaded = _predictor(config, contract)._load_compute_df(
        str(path),
        tensor_parallel_size=8,
        operator_name="mlp_up_proj",
        layer_contract=contract,
    )

    assert loaded["n_expanded_embd"].tolist() == [contract.effective_ffn_width]
