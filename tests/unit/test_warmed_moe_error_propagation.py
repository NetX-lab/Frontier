"""Regression tests for errors raised while selecting warmed MoE rows."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_gating_runtime import (
    DIRECT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_prediction_model_name,
)
from frontier.training.moe_trainer import MoETrainer
from frontier.types import ClusterType, MeasurementType


_GATING_MODEL_NAME = "moe_gating_linear"
_WARMED_GATING_MODEL_NAME = get_moe_gating_prediction_model_name(
    _GATING_MODEL_NAME,
    requested_context=PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
)
_SEMANTIC_ERROR = "typed metadata operator_family_ids mismatch"


class _MoEPredictorProbe(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("Estimator construction is outside this test seam")


def _gating_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1, 1],
            "num_tensor_parallel_workers": [1, 1],
            "expert_parallel_size": [1, 1],
            "gating_runtime_context": [
                DIRECT_MOE_GATING_RUNTIME_CONTEXT,
                PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
            ],
            "gating_runtime_context_impl": [
                "none",
                PREFILL_WARMED_MOE_GATING_RUNTIME_IMPL,
            ],
            "time_stats.moe_gating_linear.median": [1.0, 2.0],
        }
    )


def _raise_semantic_error_for_warmed_rows(
    df: pd.DataFrame,
    *,
    requested_context: str,
    source_name: str,
) -> pd.DataFrame:
    if requested_context == PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT:
        raise ValueError(_SEMANTIC_ERROR)
    return filter_moe_gating_rows_by_runtime_context(
        df,
        requested_context=requested_context,
        source_name=source_name,
    )


def test_shared_manager_preserves_warmed_row_semantic_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    rows = _gating_rows()
    moe_file = tmp_path / "moe.csv"
    linear_file = tmp_path / "linear_op.csv"
    rows.to_csv(moe_file, index=False)
    linear_file.write_text("placeholder\n", encoding="utf-8")

    model_config = BaseModelConfig.create_from_name("qwen3-a3b-30b-moe")
    replica_config = SimpleNamespace(
        device="h200",
        model_name=model_config.get_name(),
        model_config=model_config,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        speculative_decoding_config=None,
        moe_routing_distribution_type="balanced",
    )
    routed = model_config.get_model_architecture_profile().resolve_layer_contract(
        model_config,
        operator_name=_GATING_MODEL_NAME,
        moe_tp_size=1,
        expert_parallel_size=1,
    )

    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._get_ffn_contract_signature = lambda *_args, **_kwargs: "unit"
    manager._resolve_typed_layer_contract = lambda *_args, **_kwargs: routed
    manager._validate_moe_dataset_contract = lambda *_args, **_kwargs: None
    manager._load_moe_df = lambda *_args, **_kwargs: rows
    manager._load_linear_op_df = lambda *_args, **_kwargs: pd.DataFrame(
        {
            "num_tokens": [1],
            "time_stats.post_attention_layernorm.median": [1.0],
        }
    )
    manager._train_single_model = lambda **kwargs: kwargs["model_name"]
    manager._is_mixed_layer_moe_model = lambda *_args, **_kwargs: False

    monkeypatch.setattr(module, "_get_moe_family_model_names", lambda: [_GATING_MODEL_NAME])
    monkeypatch.setattr(
        module,
        "_get_moe_gating_family_model_names",
        lambda: [_GATING_MODEL_NAME],
    )
    monkeypatch.setattr(
        module,
        "filter_moe_gating_rows_by_runtime_context",
        _raise_semantic_error_for_warmed_rows,
    )

    with pytest.raises(ValueError, match=_SEMANTIC_ERROR):
        manager._train_ffn_models_for_cluster(
            ClusterType.MONOLITHIC,
            replica_config,
            execution_time_predictor_config=SimpleNamespace(),
            linear_ops_file=str(linear_file),
            moe_file=str(moe_file),
            is_moe_model=True,
            trained_model_signatures=set(),
        )


def test_independent_predictor_preserves_warmed_row_semantic_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as module

    rows = _gating_rows()
    moe_file = tmp_path / "moe.csv"
    rows.to_csv(moe_file, index=False)

    predictor = object.__new__(_MoEPredictorProbe)
    predictor._moe_input_file = str(moe_file)
    predictor._model_config = BaseModelConfig.create_from_name(
        "qwen3-a3b-30b-moe"
    )
    predictor._replica_config = SimpleNamespace(
        model_name="qwen3-a3b-30b-moe",
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._moe_routing_distribution_type = "balanced"
    predictor._get_profiling_metadata = lambda *_args, **_kwargs: {}
    predictor._validate_active_measurement_type = lambda *_args, **_kwargs: None
    predictor._validate_moe_dataset_contract = (
        lambda *_args, **_kwargs: rows
    )
    predictor._register_profiling_metadata_for_ops = (
        lambda *_args, **_kwargs: None
    )
    predictor._train_model = lambda **kwargs: kwargs["model_name"]

    monkeypatch.setattr(module, "_get_moe_family_model_names", lambda: [_GATING_MODEL_NAME])
    monkeypatch.setattr(
        module,
        "_get_moe_gating_family_model_names",
        lambda: [_GATING_MODEL_NAME],
    )
    monkeypatch.setattr(
        module,
        "filter_moe_gating_rows_by_runtime_context",
        _raise_semantic_error_for_warmed_rows,
    )

    with pytest.raises(ValueError, match=_SEMANTIC_ERROR):
        predictor._train_moe_models()


def test_standalone_trainer_preserves_warmed_row_semantic_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import frontier.training.moe_trainer as module

    rows = _gating_rows()
    trainer = MoETrainer(
        dataset_path=str(tmp_path / "moe.csv"),
        output_dir=str(tmp_path / "models"),
        num_experts=8,
        router_topk=2,
        hidden_dim=16,
        expert_hidden_dim=32,
        moe_tensor_parallel_size=1,
        expert_parallel_size=1,
        model_name="qwen3-a3b-30b-moe",
    )

    def load_rows() -> pd.DataFrame:
        trainer.df = rows
        return rows

    trainer._load_dataset = load_rows
    trainer._get_model_names = lambda: [
        _GATING_MODEL_NAME,
        _WARMED_GATING_MODEL_NAME,
    ]
    trainer._train_single_model = lambda **kwargs: kwargs["model_name"]
    monkeypatch.setattr(
        module,
        "filter_moe_gating_rows_by_runtime_context",
        _raise_semantic_error_for_warmed_rows,
    )

    with pytest.raises(ValueError, match=_SEMANTIC_ERROR):
        trainer.train()


def test_standalone_trainer_skips_missing_warmed_slice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Keep the narrow compatibility path for a genuinely absent context."""
    import frontier.training.moe_trainer as module

    rows = _gating_rows().iloc[[0]].copy()
    trainer = MoETrainer(
        dataset_path=str(tmp_path / "moe.csv"),
        output_dir=str(tmp_path / "models"),
        num_experts=8,
        router_topk=2,
        hidden_dim=16,
        expert_hidden_dim=32,
        moe_tensor_parallel_size=1,
        expert_parallel_size=1,
        model_name="qwen3-a3b-30b-moe",
    )

    def load_rows() -> pd.DataFrame:
        trainer.df = rows
        return rows

    trainer._load_dataset = load_rows
    trainer._get_model_names = lambda: [
        _GATING_MODEL_NAME,
        _WARMED_GATING_MODEL_NAME,
    ]
    trainer._train_single_model = lambda **kwargs: kwargs["model_name"]
    monkeypatch.setattr(
        module,
        "filter_moe_gating_rows_by_runtime_context",
        filter_moe_gating_rows_by_runtime_context,
    )

    assert trainer.train() == {_GATING_MODEL_NAME: _GATING_MODEL_NAME}
