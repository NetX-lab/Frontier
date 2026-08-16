"""Contract tests for standalone dense-attention training entrypoints."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
import frontier.training.attention_trainer as attention_trainer_module
import frontier.training.cli as training_cli
from frontier.training.attention_trainer import AttentionTrainer
from frontier.training.base_trainer import BaseTrainer
from frontier.training.cli import parse_args
from frontier.types import MeasurementType


class _HashTrainer(BaseTrainer):
    def _load_dataset(self):
        raise AssertionError("not used")

    def _get_model_names(self):
        raise AssertionError("not used")

    def _get_feature_cols(self, model_name: str):
        raise AssertionError("not used")

    def _get_target_col(self, model_name: str):
        raise AssertionError("not used")


class _HashPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return RandomForestRegressor(random_state=0)

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def to_dict(self) -> dict:
        return {"test": True}


def _attention_argv(*extra: str) -> list[str]:
    return [
        "frontier-training",
        "attention",
        "--layer_dataset_path",
        "attention.csv",
        "--model_name",
        "meta-llama/Llama-2-7b-hf",
        "--measurement_type",
        "CUDA_EVENT",
        *extra,
    ]


def test_attention_cli_exposes_kv_cache_prediction_granularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _attention_argv("--kv_cache_prediction_granularity", "32"),
    )

    args = parse_args()

    assert args.kv_cache_prediction_granularity == 32


@pytest.mark.parametrize("granularity", ["0", "-1"])
def test_attention_cli_rejects_non_positive_granularity(
    monkeypatch: pytest.MonkeyPatch,
    granularity: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _attention_argv("--kv_cache_prediction_granularity", granularity),
    )

    with pytest.raises(ValueError, match="kv_cache_prediction_granularity.*positive integer"):
        parse_args()


def test_attention_factory_passes_granularity_to_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ModelConfig:
        embedding_dim = 1
        num_q_heads = 1
        num_kv_heads = 1
        num_layers = 1

    captured: dict[str, object] = {}

    class _Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "frontier.config.model_config.BaseModelConfig.create_from_name",
        lambda _name: _ModelConfig(),
    )
    monkeypatch.setattr(attention_trainer_module, "AttentionTrainer", _Trainer)

    result = attention_trainer_module.create_attention_trainer_from_model_config(
        layer_dataset_path="attention.csv",
        output_dir="cache",
        model_name="test-model",
        tensor_parallel_size=1,
        kv_cache_prediction_granularity=32,
    )

    assert result is not None
    assert captured["kv_cache_prediction_granularity"] == 32


@pytest.mark.parametrize("granularity", [0, -1, 1.5, True])
def test_attention_trainer_rejects_invalid_granularity_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    granularity,
) -> None:
    def _unexpected_model_load(_name):
        raise AssertionError("model configuration must not load for invalid granularity")

    monkeypatch.setattr(
        "frontier.config.model_config.BaseModelConfig.create_from_name",
        _unexpected_model_load,
    )

    with pytest.raises(ValueError, match="kv_cache_prediction_granularity.*positive integer"):
        AttentionTrainer(
            layer_dataset_path="attention.csv",
            output_dir="cache",
            model_name="test-model",
            device="a100",
            kv_cache_prediction_granularity=granularity,
        )


def test_train_attention_forwards_cli_granularity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_dataset = tmp_path / "attention.csv"
    layer_dataset.touch()
    captured: dict[str, object] = {}

    class _Trainer:
        def train(self):
            return {}

    def _create_trainer(**kwargs):
        captured.update(kwargs)
        return _Trainer()

    monkeypatch.setattr(
        training_cli,
        "create_attention_trainer_from_model_config",
        _create_trainer,
    )
    args = SimpleNamespace(
        layer_dataset_path=str(layer_dataset),
        compute_dataset_path=None,
        output_dir=str(tmp_path / "cache"),
        model_name="test-model",
        device="a100",
        tensor_parallel_size=1,
        block_size=16,
        kv_cache_prediction_granularity=32,
        predictor_type="random_forest",
        measurement_type="CUDA_EVENT",
        k_fold_cv_splits=2,
        num_training_job_threads=1,
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
    )

    assert training_cli.train_attention(args) == 0
    assert captured["kv_cache_prediction_granularity"] == 32


def test_required_attention_model_missing_features_fails_fast() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = "attention.csv"
    trainer.output_dir = "cache"
    trainer.kv_cache_prediction_granularity = 64
    trainer._load_dataset = lambda: pd.DataFrame()
    trainer._load_layer_dataset = lambda: pd.DataFrame(
        {
            "is_decode": [True],
            "is_mixed_batch": [False],
            "is_true_mixed_batch": [False],
            "time_stats.attn_decode.median": [1.0],
        }
    )
    trainer._get_model_names = lambda: ["attn_decode"]
    trainer._get_feature_cols = lambda _name: ["batch_size", "kv_cache_size"]
    trainer._get_target_col = lambda _name: "time_stats.attn_decode.median"

    with pytest.raises(
        ValueError,
        match="attn_decode.*batch_size.*kv_cache_size.*attention.csv",
    ):
        trainer.train()


def test_layer_schema_error_names_affected_model_dataset_and_reprofiling() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = "/profiles/attention.csv"
    columns = {
        "is_decode",
        *AttentionTrainer.DENSE_LAYER_REQUIRED_FEATURE_COLUMNS,
        *AttentionTrainer.DENSE_LAYER_TARGET_COLUMNS,
    }
    columns.remove("total_tokens")
    dataframe = pd.DataFrame({column: [1] for column in columns})

    with pytest.raises(
        ValueError,
        match=(
            "attn_kv_cache_save.*total_tokens.*"
            "/profiles/attention.csv.*Re-run attention profiling"
        ),
    ):
        trainer._verify_layer_dataset_columns(dataframe)


def test_optional_empty_mixed_attention_model_remains_skippable() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = "attention.csv"
    trainer.output_dir = "cache"
    trainer.kv_cache_prediction_granularity = 64
    trainer._load_dataset = lambda: pd.DataFrame()
    trainer._load_layer_dataset = lambda: pd.DataFrame(
        {
            "is_decode": [True],
            "is_mixed_batch": [False],
            "is_true_mixed_batch": [False],
            "time_stats.attn_decode.median": [1.0],
        }
    )
    trainer._get_model_names = lambda: ["attn_prefill_mixed"]
    trainer._get_feature_cols = lambda _name: ["batch_size", "kv_cache_size"]
    trainer._get_target_col = lambda _name: "time_stats.attn_prefill.median"

    assert trainer.train() == {}


def test_attention_training_uses_configured_granularity_for_true_mixed_rows() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = "attention.csv"
    trainer.output_dir = "cache"
    trainer.kv_cache_prediction_granularity = 128
    trainer._load_dataset = lambda: pd.DataFrame()
    trainer._load_layer_dataset = lambda: pd.DataFrame(
        {
            "is_decode": [False],
            "is_mixed_batch": [False],
            "is_true_mixed_batch": [True],
            "batch_size": [3],
            "prefill_seq_lens": ["[8, 12]"],
            "prefill_kv_cache_sizes": ["[128, 256]"],
            "time_stats.attn_prefill.median": [1.0],
        }
    )
    trainer._get_model_names = lambda: ["attn_prefill_mixed"]
    captured: dict[str, object] = {}

    def _train_single_model(**kwargs):
        captured.update(kwargs)
        return object()

    trainer._train_single_model = _train_single_model

    models = trainer.train()

    assert set(models) == {"attn_prefill_mixed"}
    training_df = captured["df"]
    assert isinstance(training_df, pd.DataFrame)
    assert training_df.iloc[0]["kv_cache_size"] == 256


def test_attention_granularity_identity_component_is_shared_across_producers(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "avg_seq_len": [8.0, 12.0],
            "kv_cache_size": [128, 256],
            "time_stats.attn_prefill.median": [0.1, 0.2],
            "profiling_precision": ["FP16", "FP16"],
        }
    )
    feature_cols = ["avg_seq_len", "kv_cache_size"]
    target_col = "time_stats.attn_prefill.median"
    hashes_by_granularity: dict[int, str] = {}

    for granularity in (32, 64):
        config = RandomForrestExecutionTimePredictorConfig(
            num_estimators=[7],
            max_depth=[3],
            min_samples_split=[2],
            kv_cache_prediction_granularity=granularity,
        )
        trainer = _HashTrainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path / f"trainer-{granularity}"),
            predictor_type="random_forest",
            num_estimators=[7],
            max_depth=[3],
            min_samples_split=[2],
        )
        trainer._profiling_precision = PrecisionType.FP16
        trainer._measurement_type = MeasurementType.CUDA_EVENT
        trainer.kv_cache_prediction_granularity = granularity
        manager = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        predictor = _HashPredictor.__new__(_HashPredictor)
        predictor._config = config
        predictor._active_measurement_type = MeasurementType.CUDA_EVENT

        hashes = {
            trainer._get_model_hash(
                "attn_prefill_mixed",
                dataframe,
                feature_cols,
                target_col,
            ),
            manager._get_model_hash(
                "attn_prefill_mixed",
                dataframe,
                config,
                "FP16",
                MeasurementType.CUDA_EVENT,
                feature_cols=feature_cols,
                target_col=target_col,
            ),
            predictor._get_model_hash(
                "attn_prefill_mixed",
                dataframe,
                feature_cols,
                target_col,
            ),
        }

        assert len(hashes) == 1
        hashes_by_granularity[granularity] = hashes.pop()

    assert hashes_by_granularity[32] != hashes_by_granularity[64]


def test_base_trainer_filters_nan_rows_before_model_identity_and_cache_lookup(
    tmp_path,
) -> None:
    trainer = _HashTrainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path / "trainer"),
        predictor_type="random_forest",
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
        k_fold_cv_splits=2,
    )
    dataframe = pd.DataFrame(
        {
            "num_tokens": [1.0, 2.0, float("nan"), 4.0],
            "time_stats.attn_pre_proj.median": [0.1, 0.2, 0.3, float("nan")],
            "profiling_precision": ["FP16"] * 4,
        }
    )
    captured: dict[str, pd.DataFrame] = {}
    cached_model = object()
    trainer._build_operator_binding = lambda _name, _df: {}

    def _get_model_hash(_name, df, *_args, **_kwargs):
        captured["identity_df"] = df.copy()
        return "model-hash"

    def _load_model_from_cache(*_args, **_kwargs):
        return cached_model

    trainer._get_model_hash = _get_model_hash
    trainer._load_model_from_cache = _load_model_from_cache

    result = trainer._train_single_model(
        "attn_pre_proj",
        dataframe,
        ["num_tokens"],
        "time_stats.attn_pre_proj.median",
    )

    assert result is cached_model
    assert captured["identity_df"]["num_tokens"].tolist() == [1.0, 2.0]
    assert captured["identity_df"]["time_stats.attn_pre_proj.median"].tolist() == [
        0.1,
        0.2,
    ]


def test_base_trainer_rejects_missing_training_columns_with_dataset_context(
    tmp_path,
) -> None:
    trainer = _HashTrainer(
        dataset_path="/profiles/linear_op.csv",
        output_dir=str(tmp_path / "trainer"),
        predictor_type="random_forest",
    )
    dataframe = pd.DataFrame({"num_tokens": [1, 2]})

    with pytest.raises(
        ValueError,
        match=(
            "attn_pre_proj.*time_stats.attn_pre_proj.median.*"
            "/profiles/linear_op.csv.*Re-run profiling"
        ),
    ):
        trainer._train_single_model(
            "attn_pre_proj",
            dataframe,
            ["num_tokens"],
            "time_stats.attn_pre_proj.median",
        )


def test_base_trainer_rejects_all_nan_training_rows(tmp_path) -> None:
    trainer = _HashTrainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path / "trainer"),
        predictor_type="random_forest",
    )
    dataframe = pd.DataFrame(
        {
            "num_tokens": [float("nan"), 2.0],
            "time_stats.attn_pre_proj.median": [0.1, float("nan")],
        }
    )

    with pytest.raises(ValueError, match="empty after dropping NaN rows"):
        trainer._train_single_model(
            "attn_pre_proj",
            dataframe,
            ["num_tokens"],
            "time_stats.attn_pre_proj.median",
        )
