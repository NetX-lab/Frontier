"""Regression tests for finite prediction lookup and on-demand model fallback."""

from __future__ import annotations

from collections import defaultdict
import math
from types import SimpleNamespace
from typing import Any

import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


class _CountingFiniteModel:
    def __init__(self, result: float) -> None:
        self.n_features_in_ = 1
        self._frontier_feature_names = ["num_tokens"]
        self.result = result
        self.calls = 0

    def predict(self, features: Any) -> list[float]:
        self.calls += 1
        assert list(features.columns) == ["num_tokens"]
        return [self.result]


class _CountingTwoFeatureModel:
    def __init__(self, feature_names: tuple[str, str], result: float) -> None:
        self.n_features_in_ = len(feature_names)
        self._frontier_feature_names = list(feature_names)
        self.result = result
        self.calls = 0

    def predict(self, features: Any) -> list[float]:
        self.calls += 1
        assert list(features.columns) == self._frontier_feature_names
        return [self.result]


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        raise AssertionError("not used")

    def _get_estimator(self):
        raise AssertionError("not used")


def _build_predictor(
    *,
    model_result: float = 4.5,
    runtime_value: float | None = None,
    exact_lookup: dict[tuple[float, ...], float] | None = None,
) -> tuple[_ConcretePredictor, _CountingFiniteModel]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))

    model = _CountingFiniteModel(model_result)
    if exact_lookup is not None:
        model._frontier_exact_lookup = exact_lookup
    predictor._models = {"test_operator": model}
    predictor._predictions = {"test_operator": {(1,): 1.0}}
    if runtime_value is not None:
        predictor._runtime_cache["eager"]["test_operator"][(2.0,)] = runtime_value
    return predictor, model


def test_finite_exact_measured_row_precedes_materialized_model_value() -> None:
    predictor, model = _build_predictor(
        model_result=6.0,
        exact_lookup={(1.0,): 9.0},
    )

    value = predictor._get_named_linear_op_execution_time(
        op_name="test_operator",
        num_tokens=1,
    )

    assert value == 9.0
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def test_finite_prediction_miss_uses_model_once_and_reuses_runtime_cache() -> None:
    predictor, model = _build_predictor()

    first = predictor._get_named_linear_op_execution_time(
        op_name="test_operator",
        num_tokens=2,
    )
    second = predictor._get_named_linear_op_execution_time(
        op_name="test_operator",
        num_tokens=2,
    )

    assert first == 4.5
    assert second == 4.5
    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["test_operator"] == {(2.0,): 4.5}


def test_finite_prediction_miss_does_not_use_nearest_materialized_row() -> None:
    predictor, model = _build_predictor(model_result=7.25)

    value = predictor._get_named_linear_op_execution_time(
        op_name="test_operator",
        num_tokens=2,
    )

    assert value == 7.25
    assert value != 1.0
    assert model.calls == 1


def test_finite_model_schema_mismatch_fails_before_exact_lookup() -> None:
    predictor, model = _build_predictor()
    model._frontier_feature_names = ["batch_size"]

    with pytest.raises(ValueError, match="Prediction feature schema mismatch"):
        predictor._get_named_linear_op_execution_time(
            op_name="test_operator",
            num_tokens=1,
        )

    assert model.calls == 0


def test_invalid_finite_runtime_cache_value_fails_before_model_invocation() -> None:
    predictor, model = _build_predictor(runtime_value=math.nan)

    with pytest.raises(ValueError, match="runtime cache value.*finite and non-negative"):
        predictor._get_named_linear_op_execution_time(
            op_name="test_operator",
            num_tokens=2,
        )

    assert model.calls == 0


@pytest.mark.parametrize("result", [math.nan, math.inf, -1.0])
def test_invalid_finite_model_output_fails_without_cache(result: float) -> None:
    predictor, model = _build_predictor(model_result=result)

    with pytest.raises(ValueError, match="Prediction output.*finite and non-negative"):
        predictor._get_named_linear_op_execution_time(
            op_name="test_operator",
            num_tokens=2,
        )

    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def _build_attention_decode_predictor() -> tuple[_ConcretePredictor, _CountingTwoFeatureModel, Any]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    model = _CountingTwoFeatureModel(("batch_size", "kv_cache_size"), 4.25)
    predictor._models = {"attn_decode": model}
    predictor._predictions = {"attn_decode": {(1, 64): 1.0}}
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._supports_operation = lambda _operation: True
    predictor._get_batch_decode_attention_params = lambda _batch: (2, 128)
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._get_late_decode_only_calibration_scale = lambda *_args: None
    predictor._get_calibration_scale = lambda *_args: 1.0
    batch = SimpleNamespace(num_prefill_tokens=0)
    return predictor, model, batch


def test_attention_decode_finite_miss_uses_model_once_and_reuses_runtime_cache() -> None:
    predictor, model, batch = _build_attention_decode_predictor()

    first = predictor._get_attention_decode_execution_time(batch)
    second = predictor._get_attention_decode_execution_time(batch)

    assert first == 4.25
    assert second == 4.25
    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["attn_decode"] == {
        (2.0, 128.0): 4.25
    }


def _build_attention_prefill_predictor() -> tuple[_ConcretePredictor, _CountingTwoFeatureModel, Any]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    model = _CountingTwoFeatureModel(
        ("kv_cache_size", "prefill_chunk_size_squared"),
        5.5,
    )
    predictor._models = {"attn_prefill": model}
    predictor._predictions = {"attn_prefill": {(64, 256): 1.0}}
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    predictor._supports_operation = lambda _operation: True
    predictor._get_batch_prefill_attention_params = lambda _batch: [(128, 32)]
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    batch = SimpleNamespace(num_prefill_tokens=32, id=1, num_tokens=[32])
    return predictor, model, batch


def test_attention_prefill_finite_miss_uses_model_once_and_reuses_runtime_cache() -> None:
    predictor, model, batch = _build_attention_prefill_predictor()

    first = predictor._get_attention_prefill_execution_time(batch)
    second = predictor._get_attention_prefill_execution_time(batch)

    assert first == 5.5
    assert second == 5.5
    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["attn_prefill"] == {
        (128.0, 1024.0): 5.5
    }
