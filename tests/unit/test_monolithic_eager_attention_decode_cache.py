from __future__ import annotations

from types import SimpleNamespace

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    MeasurementType,
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


def _dense_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_mla=False,
        num_q_heads=32,
        num_kv_heads=8,
        get_model_architecture_profile=ModelArchitectureProfile.generic,
    )


class _DummyPredictorImpl(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise RuntimeError("Not used in this unit test")


def test_attention_layer_prediction_builds_decode_cache_for_monolithic_eager_family() -> None:
    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_config = _dense_model_config()
    predictor._models = {"attn_decode": object()}
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=128,
        prediction_max_prefill_chunk_size=128,
        kv_cache_prediction_granularity=64,
    )

    captured = {}

    def _fake_get_model_prediction(model_name, _model, features_df):
        captured["model_name"] = model_name
        captured["feature_columns"] = list(features_df.columns)
        captured["num_rows"] = len(features_df)
        return {(1, 0): 0.1}

    predictor._get_model_prediction = _fake_get_model_prediction  # type: ignore[attr-defined]

    predictions = predictor._predict_for_attention_layer_models()

    assert "attn_decode" in predictions
    assert captured == {
        "model_name": "attn_decode",
        "feature_columns": ["batch_size", "kv_cache_size"],
        "num_rows": 6,
    }


def test_attention_layer_prediction_builds_decode_cache_for_pd_decode_eager_family() -> None:
    predictor = _DummyPredictorImpl.__new__(_DummyPredictorImpl)
    predictor._cluster_type = ClusterType.DECODE
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_config = _dense_model_config()
    predictor._models = {"attn_decode": object()}
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=128,
        prediction_max_prefill_chunk_size=128,
        kv_cache_prediction_granularity=64,
    )

    captured = {}

    def _fake_get_model_prediction(model_name, _model, features_df):
        captured["model_name"] = model_name
        captured["feature_columns"] = list(features_df.columns)
        captured["num_rows"] = len(features_df)
        return {(1, 0): 0.1}

    predictor._get_model_prediction = _fake_get_model_prediction  # type: ignore[attr-defined]

    predictions = predictor._predict_for_attention_layer_models()

    assert "attn_decode" in predictions
    assert captured == {
        "model_name": "attn_decode",
        "feature_columns": ["batch_size", "kv_cache_size"],
        "num_rows": 6,
    }
