"""Regression tests for the predictor's validated on-demand lookup path."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import math

import pandas as pd
import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import MeasurementType


class _CountingModel:
    def __init__(self, feature_names: tuple[str, ...], result: float) -> None:
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = feature_names
        self._frontier_feature_names = list(feature_names)
        self.result = result
        self.calls = 0
        self.inputs: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.calls += 1
        self.inputs.append(features.copy())
        return [self.result]


class _LookupWithoutIteration(dict[tuple[float, ...], float]):
    def items(self):
        raise AssertionError("exact lookup must use the canonical key directly")


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        raise AssertionError("not used")

    def _get_estimator(self):
        raise AssertionError("not used")


def _build_predictor(
    *,
    feature_names: tuple[str, ...] = ("num_tokens",),
    result: float = 1.5,
    **record_overrides: Any,
) -> tuple[_ConcretePredictor, _CountingModel]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))

    model = _CountingModel(feature_names, result)
    record: dict[str, Any] = {
        "_on_demand_prediction": True,
        "_feature_names": list(feature_names),
        "_model": model,
    }
    record.update(record_overrides)
    predictor._predictions = {"test_operator": record}
    return predictor, model


def test_exact_measured_value_precedes_runtime_cache_without_writing_it() -> None:
    predictor, model = _build_predictor(
        _exact_lookup={(4.0,): 2.25},
    )
    predictor._runtime_cache["eager"]["test_operator"][(4.0,)] = 9.0

    value = predictor._get_on_demand_prediction(
        "test_operator",
        {"num_tokens": 4},
    )

    assert value == 2.25
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {(4.0,): 9.0}


def test_exact_measured_value_does_not_create_runtime_cache_entry() -> None:
    predictor, model = _build_predictor(
        _exact_lookup={(4.0,): 2.25},
    )

    assert (
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})
        == 2.25
    )
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def test_exact_lookup_hit_uses_canonical_numeric_key_without_scanning_other_rows() -> None:
    predictor, model = _build_predictor(
        _exact_lookup=_LookupWithoutIteration({(4.0,): 2.25, (5.0,): 3.5}),
    )

    assert (
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})
        == 2.25
    )
    assert model.calls == 0


def test_malformed_exact_lookup_value_fails_before_model_fallback() -> None:
    predictor, model = _build_predictor(
        _exact_lookup={(4.0,): None},
    )

    with pytest.raises(ValueError, match="exact lookup value.*not numeric"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0


def test_boolean_exact_lookup_value_is_rejected_before_model_fallback() -> None:
    predictor, model = _build_predictor(
        _exact_lookup={(4.0,): True},
    )

    with pytest.raises(ValueError, match="exact lookup value.*numeric"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0


def test_legal_model_miss_is_cached_using_declared_feature_order() -> None:
    predictor, model = _build_predictor(
        feature_names=("batch_size", "num_tokens"),
        result=3.75,
    )
    features = {"num_tokens": 8, "batch_size": 2}

    assert predictor._get_on_demand_prediction("test_operator", features) == 3.75
    assert predictor._get_on_demand_prediction("test_operator", features) == 3.75

    assert model.calls == 1
    assert list(model.inputs[0].columns) == ["batch_size", "num_tokens"]
    assert predictor._runtime_cache["eager"]["test_operator"] == {(2.0, 8.0): 3.75}


def test_boolean_runtime_cache_value_is_rejected_before_return() -> None:
    predictor, model = _build_predictor()
    predictor._runtime_cache["eager"]["test_operator"][(4.0,)] = True

    with pytest.raises(ValueError, match="runtime cache value.*numeric"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0


def test_missing_feature_is_rejected_before_runtime_cache_lookup() -> None:
    predictor, model = _build_predictor(
        feature_names=("num_tokens", "batch_size"),
    )
    predictor._runtime_cache["eager"]["test_operator"][(4.0,)] = 8.0

    with pytest.raises(ValueError, match="missing required features"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {(4.0,): 8.0}


def test_extra_feature_is_rejected_before_model_invocation() -> None:
    predictor, model = _build_predictor()

    with pytest.raises(ValueError, match="unexpected features"):
        predictor._get_on_demand_prediction(
            "test_operator",
            {"num_tokens": 4, "batch_size": 2},
        )

    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def test_model_feature_schema_mismatch_is_rejected_before_prediction() -> None:
    predictor, model = _build_predictor()
    model._frontier_feature_names = ["batch_size"]
    model.feature_names_in_ = ("batch_size",)

    with pytest.raises(ValueError, match="feature schema mismatch"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_nonfinite_or_negative_feature_is_rejected_before_prediction(value: float) -> None:
    predictor, model = _build_predictor()

    with pytest.raises(ValueError, match="finite|non-negative"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": value})

    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def test_declared_feature_bound_is_rejected_before_prediction() -> None:
    predictor, model = _build_predictor(
        _feature_bounds={"num_tokens": (1.0, 8.0)},
    )

    with pytest.raises(ValueError, match="above declared maximum"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 9})

    assert model.calls == 0


def test_declared_relational_constraint_is_rejected_before_prediction() -> None:
    predictor, model = _build_predictor(
        feature_names=("batch_size", "total_tokens"),
        _feature_constraints=[
            {
                "type": "linear_lte",
                "terms": {"batch_size": 1.0, "total_tokens": -1.0},
                "max": 0.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="relational constraint"):
        predictor._get_on_demand_prediction(
            "test_operator",
            {"batch_size": 3, "total_tokens": 2},
        )

    assert model.calls == 0


def test_identity_mismatch_is_rejected_before_prediction() -> None:
    predictor, model = _build_predictor(
        _identity={"model_name": "demo", "device": "h800"},
    )
    predictor._prediction_identity = {"model_name": "demo", "device": "a800"}

    with pytest.raises(ValueError, match="identity mismatch"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 0


@pytest.mark.parametrize("result", [-1.0, math.nan, math.inf])
def test_invalid_model_output_is_rejected_without_runtime_cache(result: float) -> None:
    predictor, model = _build_predictor(result=result)

    with pytest.raises(ValueError, match="finite and non-negative"):
        predictor._get_on_demand_prediction("test_operator", {"num_tokens": 4})

    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["test_operator"] == {}


def test_exact_lookup_metadata_survives_model_cache_round_trip(tmp_path) -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cache_dir = str(tmp_path)
    predictor._config = SimpleNamespace(no_cache=False)
    predictor._get_model_hash = lambda _model_name, _df: "roundtrip"

    legacy_model = _CountingModel(("batch_size", "num_tokens"), result=99.0)
    predictor._store_model_in_cache("test_operator", "roundtrip", legacy_model)
    dataframe = pd.DataFrame(
        {
            "batch_size": [2, 3],
            "num_tokens": [8, 8],
            "target": [2.25, 3.5],
        }
    )

    loaded = predictor._train_model(
        model_name="test_operator",
        df=dataframe,
        feature_cols=["batch_size", "num_tokens"],
        target_col="target",
        persist_exact_lookup=True,
    )
    reloaded = predictor._load_model_from_cache("test_operator", "roundtrip")

    expected = {(2.0, 8.0): 2.25, (3.0, 8.0): 3.5}
    assert loaded._frontier_exact_lookup == expected
    assert reloaded._frontier_exact_lookup == expected
