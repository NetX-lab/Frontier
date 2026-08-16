"""Producer/consumer parity for the explicit attention KV-grid lower bound."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import frontier.training.attention_trainer as attention_trainer_module
import frontier.training.cli as training_cli
from frontier.training.attention_trainer import AttentionTrainer
from frontier.training.base_trainer import BaseTrainer
from frontier.training.cli import parse_args
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.model_cache_contract import (
    build_canonical_operator_binding,
)
from frontier.config.precision_type import PrecisionType
from frontier.types import ClusterType, MeasurementType


class _HashTrainer(BaseTrainer):
    def _load_dataset(self):
        raise AssertionError("not used")

    def _get_model_names(self):
        raise AssertionError("not used")

    def _get_feature_cols(self, model_name: str):
        raise AssertionError("not used")

    def _get_target_col(self, model_name: str):
        raise AssertionError("not used")


class _RuntimePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return RandomForestRegressor(random_state=0)

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }


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


def test_attention_cli_exposes_prediction_min_kv_cache_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _attention_argv("--prediction_min_kv_cache_size", "32"),
    )

    args = parse_args()

    assert args.prediction_min_kv_cache_size == 32


@pytest.mark.parametrize("value", ["-1", "1.5"])
def test_attention_cli_rejects_invalid_prediction_min_kv_cache_size(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _attention_argv("--prediction_min_kv_cache_size", value),
    )

    if value == "-1":
        with pytest.raises(ValueError, match="prediction_min_kv_cache_size"):
            parse_args()
    else:
        # argparse reports malformed integer syntax through SystemExit after
        # printing the option name; the exception value itself is only `2`.
        with pytest.raises(SystemExit):
            parse_args()


def test_attention_factory_passes_prediction_min_to_trainer(
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

    attention_trainer_module.create_attention_trainer_from_model_config(
        layer_dataset_path="attention.csv",
        output_dir="cache",
        model_name="test-model",
        tensor_parallel_size=1,
        prediction_min_kv_cache_size=32,
    )

    assert captured["prediction_min_kv_cache_size"] == 32


def test_train_attention_forwards_prediction_min_from_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
        prediction_min_kv_cache_size=32,
        predictor_type="random_forest",
        measurement_type="CUDA_EVENT",
        k_fold_cv_splits=2,
        num_training_job_threads=1,
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
    )

    assert training_cli.train_attention(args) == 0
    assert captured["prediction_min_kv_cache_size"] == 32


def test_attention_factory_rejects_invalid_prediction_min_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_model_load(_name):
        raise AssertionError("model configuration must not load")

    monkeypatch.setattr(
        "frontier.config.model_config.BaseModelConfig.create_from_name",
        _unexpected_model_load,
    )

    with pytest.raises(ValueError, match="prediction_min_kv_cache_size"):
        attention_trainer_module.create_attention_trainer_from_model_config(
            layer_dataset_path="attention.csv",
            output_dir="cache",
            model_name="test-model",
            prediction_min_kv_cache_size=-1,
        )


def test_attention_trainer_propagates_prediction_min_into_base_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _ModelConfig:
        embedding_dim = 1
        num_q_heads = 1
        num_kv_heads = 1
        num_layers = 1
        mlp_hidden_dim = 1
        vocab_size = 1
        use_gated_mlp = False
        uses_fused_add_norm = True

    monkeypatch.setattr(
        "frontier.config.model_config.BaseModelConfig.create_from_name",
        lambda _name: _ModelConfig(),
    )

    trainer = AttentionTrainer(
        layer_dataset_path="attention.csv",
        output_dir=str(tmp_path),
        model_name="test-model",
        device="a100",
        prediction_min_kv_cache_size=32,
    )

    assert trainer.prediction_min_kv_cache_size == 32
    assert trainer.execution_time_predictor_config.prediction_min_kv_cache_size == 32


@pytest.mark.parametrize("value", [-1, 1.5, True, "32"])
def test_base_trainer_rejects_invalid_prediction_min_before_training(value, tmp_path) -> None:
    with pytest.raises(ValueError, match="prediction_min_kv_cache_size"):
        _HashTrainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path),
            predictor_type="random_forest",
            prediction_min_kv_cache_size=value,
        )


def test_base_trainer_serializes_prediction_min_in_execution_config(tmp_path) -> None:
    trainer = _HashTrainer(
        dataset_path="unused.csv",
        output_dir=str(tmp_path),
        predictor_type="random_forest",
        prediction_min_kv_cache_size=32,
    )

    assert trainer.execution_time_predictor_config.prediction_min_kv_cache_size == 32


def test_prediction_min_keeps_model_identity_equal_across_all_producers(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "batch_size": [1, 2],
            "kv_cache_size": [32, 64],
            "time_stats.attn_decode.median": [0.1, 0.2],
            "profiling_precision": ["FP16", "FP16"],
        }
    )
    feature_cols = ["batch_size", "kv_cache_size"]
    target_col = "time_stats.attn_decode.median"
    binding = build_canonical_operator_binding(
        "attn_decode",
        dataframe=dataframe,
    )
    hashes_by_minimum: dict[int, str] = {}

    for minimum in (0, 1, 32):
        config = RandomForrestExecutionTimePredictorConfig(
            num_estimators=[7],
            max_depth=[3],
            min_samples_split=[2],
            kv_cache_prediction_granularity=32,
            prediction_min_kv_cache_size=minimum,
        )
        trainer = _HashTrainer(
            dataset_path="unused.csv",
            output_dir=str(tmp_path / f"trainer-{minimum}"),
            predictor_type="random_forest",
            num_estimators=[7],
            max_depth=[3],
            min_samples_split=[2],
            prediction_min_kv_cache_size=minimum,
        )
        trainer._profiling_precision = PrecisionType.FP16
        trainer._measurement_type = MeasurementType.CUDA_EVENT
        trainer.kv_cache_prediction_granularity = 32
        manager = ExecutionTimePredictionModelManager.__new__(
            ExecutionTimePredictionModelManager
        )
        predictor = _RuntimePredictor.__new__(_RuntimePredictor)
        predictor._config = config
        predictor._active_measurement_type = MeasurementType.CUDA_EVENT

        hashes = {
            trainer._get_model_hash(
                "attn_decode",
                dataframe,
                feature_cols,
                target_col,
                operator_binding=binding,
            ),
            manager._get_model_hash(
                "attn_decode",
                dataframe,
                config,
                "FP16",
                MeasurementType.CUDA_EVENT,
                feature_cols=feature_cols,
                target_col=target_col,
                operator_binding=binding,
            ),
            predictor._get_model_hash(
                "attn_decode",
                dataframe,
                feature_cols,
                target_col,
                operator_binding=binding,
            ),
        }

        assert len(hashes) == 1
        hashes_by_minimum[minimum] = hashes.pop()

    # The lower bound chooses a runtime materialization subset; it does not
    # change the estimator trained from identical profile rows.
    assert len(set(hashes_by_minimum.values())) == 1


def test_prediction_min_is_serialized_in_runtime_prediction_context() -> None:
    context_hashes: dict[int, str] = {}

    for minimum in (0, 32):
        predictor = _RuntimePredictor.__new__(_RuntimePredictor)
        predictor._config = RandomForrestExecutionTimePredictorConfig(
            prediction_min_kv_cache_size=minimum,
            kv_cache_prediction_granularity=32,
        )
        predictor._cluster_type = ClusterType.DECODE_ATTN
        predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=1)
        predictor._model_config = SimpleNamespace(
            num_q_heads=32,
            num_kv_heads=4,
            embedding_dim=2048,
            mlp_hidden_dim=768,
            use_gated_mlp=True,
            vocab_size=151936,
        )
        predictor._block_size = 16
        predictor._max_tokens = 4096
        predictor._active_measurement_type = MeasurementType.CUDA_EVENT
        for attribute in (
            "_compute_input_file",
            "_attention_input_file",
            "_compute_input_file_eager",
            "_attention_input_file_eager",
            "_compute_input_file_kernel_only",
            "_attention_input_file_kernel_only",
            "_all_reduce_input_file",
            "_send_recv_input_file",
            "_cpu_overhead_input_file",
            "_pp_stage_boundary_input_file",
            "_pp_receiver_head_input_file",
            "_pp_producer_send_path_input_file",
            "_pp_prefill_consumer_active_input_file",
        ):
            setattr(predictor, attribute, f"/{attribute[1:]}.csv")
        predictor._model_manager = None

        serialized = predictor.to_dict()
        assert serialized["prediction_min_kv_cache_size"] == minimum
        context_hashes[minimum] = predictor._get_prediction_context_hash(
            "attn_decode"
        )

    assert context_hashes[0] != context_hashes[32]


def test_attention_grid_start_is_aligned_with_runtime_kv_quantization() -> None:
    predictor = _RuntimePredictor.__new__(_RuntimePredictor)
    predictor._cluster_type = ClusterType.DECODE_ATTN
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_config = SimpleNamespace(use_mla=False)
    predictor._is_mla_attention_family = lambda: False
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    predictor._serving_max_tokens_per_request = 96
    predictor._config = RandomForrestExecutionTimePredictorConfig(
        prediction_max_batch_size=1,
        prediction_max_tokens_per_request=96,
        prediction_min_kv_cache_size=1,
        kv_cache_prediction_granularity=32,
    )
    predictor._models = {"attn_decode": object()}
    observed_kv_grid: list[int] = []

    def _capture_prediction(_name: str, _model: object, features: pd.DataFrame):
        observed_kv_grid.extend(features["kv_cache_size"].tolist())
        return {}

    predictor._get_model_prediction = _capture_prediction  # type: ignore[method-assign]

    predictor._predict_for_attention_layer_models()

    assert observed_kv_grid == [32, 64, 96]


def test_attention_grid_preserves_zero_start_when_prediction_min_is_zero() -> None:
    predictor = _RuntimePredictor.__new__(_RuntimePredictor)
    predictor._config = RandomForrestExecutionTimePredictorConfig(
        prediction_max_tokens_per_request=96,
        prediction_min_kv_cache_size=0,
        kv_cache_prediction_granularity=32,
    )

    assert predictor._get_attention_prediction_kv_cache_size_range().tolist() == [
        0,
        32,
        64,
        96,
    ]


def test_runtime_decode_key_uses_the_same_zero_anchored_quantization() -> None:
    predictor = _RuntimePredictor.__new__(_RuntimePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.DECODE_ATTN
    predictor._config = RandomForrestExecutionTimePredictorConfig(
        prediction_min_kv_cache_size=1,
        kv_cache_prediction_granularity=32,
    )
    request = SimpleNamespace(
        _is_prefill_complete=True,
        num_processed_tokens=1,
        num_processed_decode_tokens=0,
        num_emitted_decode_tokens=0,
    )
    batch = SimpleNamespace(requests=[request])
    key = predictor._get_batch_decode_attention_params(batch)
    predictor._predictions = {"attn_decode": {key: 1.25}}

    assert key == (1, 32)
    assert predictor._get_lookup_or_predict(
        "attn_decode",
        key,
        ["batch_size", "kv_cache_size"],
    ) == 1.25
