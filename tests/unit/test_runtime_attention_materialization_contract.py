"""Runtime attention materialization must obey the persisted profile domain."""

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


class _CountingAttentionModel:
    n_features_in_ = 2

    def __init__(self) -> None:
        self.calls = 0
        self._frontier_model_hash = "runtime-attention-model-v1"

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.calls += 1
        return [float(row.batch_size + row.kv_cache_size) for row in features.itertuples()]


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("estimator construction is not part of this contract test")

    def _get_grid_search_params(self):
        raise AssertionError("grid-search construction is not part of this contract test")


def _build_runtime_predictor(
    tmp_path: Path,
    model: _CountingAttentionModel,
    *,
    prediction_min_kv_cache_size: int,
) -> _ConcretePredictor:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = ClusterType.DECODE_ATTN
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._is_mla_attention_family = lambda: False
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    predictor._cache_dir = str(tmp_path)
    predictor._config = SimpleNamespace(
        no_cache=False,
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=4,
        prediction_max_prefill_chunk_size=4,
        kv_cache_prediction_granularity=1,
        prediction_min_kv_cache_size=prediction_min_kv_cache_size,
    )
    predictor._models = {"attn_decode": model}
    predictor._get_prediction_context_hash = lambda *_args, **_kwargs: "runtime-context-v1"
    return predictor


def _attach_complete_decode_domain(model: _CountingAttentionModel) -> None:
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 1, 1, 2, 2, 2, 2],
            "kv_cache_size": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["batch_size", "kv_cache_size"],
        operator_name="attn_decode",
    )


def test_runtime_materialization_positive_subset_hits_cache_without_repredict(
    tmp_path: Path,
) -> None:
    model = _CountingAttentionModel()
    _attach_complete_decode_domain(model)
    predictor = _build_runtime_predictor(
        tmp_path,
        model,
        prediction_min_kv_cache_size=1,
    )

    first_predictions = predictor._predict_for_attention_layer_models()
    first_files = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    assert len(first_predictions["attn_decode"]) == 8
    assert model.calls == 1
    assert first_files

    second_predictions = predictor._predict_for_attention_layer_models()
    second_files = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    assert second_predictions == first_predictions
    assert model.calls == 1
    assert second_files == first_files


def test_runtime_materialization_predicts_unmeasured_kv_zero_key(
    tmp_path: Path,
) -> None:
    model = _CountingAttentionModel()
    _attach_complete_decode_domain(model)
    predictor = _build_runtime_predictor(
        tmp_path,
        model,
        prediction_min_kv_cache_size=0,
    )

    predictions = predictor._predict_for_attention_layer_models()

    assert len(predictions["attn_decode"]) == 10
    assert model.calls == 1
    assert list(tmp_path.iterdir())
