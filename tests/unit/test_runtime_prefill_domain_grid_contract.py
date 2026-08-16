"""Runtime standard-prefill grids must start inside their declared domain."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.prediction_cache_contract import (
    attach_feature_domain,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


class _CountingPrefillModel:
    n_features_in_ = 2

    def __init__(self) -> None:
        self.calls = 0
        self.last_features: pd.DataFrame | None = None
        self._frontier_model_hash = "prefill-domain-grid-v1"

    def predict(self, dataframe: pd.DataFrame) -> list[float]:
        self.calls += 1
        self.last_features = dataframe.copy()
        return [1.0 for _ in range(len(dataframe))]


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("estimator construction is not part of this test")

    def _get_grid_search_params(self):
        raise AssertionError("grid-search construction is not part of this test")


def _build_predictor(tmp_path: Path, model: _CountingPrefillModel) -> _ConcretePredictor:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = ClusterType.PREFILL
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._is_mla_attention_family = lambda: False
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    predictor._cache_dir = str(tmp_path)
    predictor._config = SimpleNamespace(
        no_cache=False,
        prediction_max_batch_size=8,
        prediction_max_tokens_per_request=64,
        prediction_max_prefill_chunk_size=96,
        kv_cache_prediction_granularity=32,
        prediction_min_kv_cache_size=0,
    )
    predictor._models = {"attn_prefill": model}
    predictor._get_prediction_context_hash = lambda *_args, **_kwargs: "prefill-context-v1"
    return predictor


def test_prefill_grid_uses_declared_lower_bound_before_estimator(tmp_path: Path) -> None:
    model = _CountingPrefillModel()
    attach_feature_domain(
        model,
        pd.DataFrame(
            {
                "kv_cache_size": [0, 0, 0, 64],
                "prefill_chunk_size_squared": [1024, 4096, 9216, 4096],
            }
        ),
        ["kv_cache_size", "prefill_chunk_size_squared"],
        operator_name="attn_prefill",
    )
    predictor = _build_predictor(tmp_path, model)

    predictions = predictor._predict_for_attention_layer_models()

    assert predictions["attn_prefill"]
    assert model.calls == 1
    assert model.last_features is not None
    assert int(model.last_features["prefill_chunk_size_squared"].min()) == 1
    assert int(model.last_features["prefill_chunk_size_squared"].max()) == 96**2


def test_prefill_grid_predicts_configured_kv_upper_bound_outside_measured_domain(
    tmp_path: Path,
) -> None:
    model = _CountingPrefillModel()
    attach_feature_domain(
        model,
        pd.DataFrame(
            {
                "kv_cache_size": [0, 0, 0, 64],
                "prefill_chunk_size_squared": [1024, 4096, 9216, 4096],
            }
        ),
        ["kv_cache_size", "prefill_chunk_size_squared"],
        operator_name="attn_prefill",
    )
    predictor = _build_predictor(tmp_path, model)
    predictor._config.prediction_max_tokens_per_request = 96

    predictions = predictor._predict_for_attention_layer_models()
    assert predictions["attn_prefill"]
    assert model.calls == 1
