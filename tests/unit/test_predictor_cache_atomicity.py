"""Regression tests for interrupted non-dummy predictor cache writes."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class _ConcretePredictor(SklearnExecutionTimePredictor):
    """Minimal concrete shell for exercising cache persistence methods."""

    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def _predictor_without_initialization(cache_dir) -> SklearnExecutionTimePredictor:
    predictor = object.__new__(_ConcretePredictor)
    predictor._cache_dir = str(cache_dir)
    predictor._config = SimpleNamespace(no_cache=False)
    return predictor


def _manager_without_initialization(cache_dir) -> ExecutionTimePredictionModelManager:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cache_dir = str(cache_dir)
    manager._all_dummy_mode = False
    return manager


@pytest.mark.parametrize("builder", [_predictor_without_initialization, _manager_without_initialization])
def test_model_cache_write_does_not_publish_partial_pickle(tmp_path, monkeypatch, builder) -> None:
    """A failed replacement must leave the previous complete model readable."""

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "model_hash.pkl"
    cache_file.write_bytes(pickle.dumps({"version": "old"}))
    writer = builder(cache_dir)

    def interrupted_dump(value, stream, protocol=None):
        stream.write(b"partial-pickle")
        stream.flush()
        raise RuntimeError("simulated interrupted cache write")

    monkeypatch.setattr(pickle, "dump", interrupted_dump)

    with pytest.raises(RuntimeError, match="simulated interrupted cache write"):
        writer._store_model_in_cache("model", "hash", {"version": "new"})

    assert pickle.loads(cache_file.read_bytes()) == {"version": "old"}
    assert not list(cache_dir.glob("*.tmp"))


def test_prediction_cache_write_does_not_publish_partial_pickle(tmp_path, monkeypatch) -> None:
    """Prediction lookup caches must have the same interruption contract."""

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "pred_hash_predictions.pkl"
    cache_file.write_bytes(pickle.dumps({("old",): 1.0}))
    predictor = _predictor_without_initialization(cache_dir)

    def interrupted_dump(value, stream, protocol=None):
        stream.write(b"partial-pickle")
        stream.flush()
        raise RuntimeError("simulated interrupted prediction cache write")

    monkeypatch.setattr(pickle, "dump", interrupted_dump)

    with pytest.raises(RuntimeError, match="simulated interrupted prediction cache write"):
        predictor._store_model_predication_cache("pred", "hash", {("new",): 2.0})

    assert pickle.loads(cache_file.read_bytes()) == {("old",): 1.0}
    assert not list(cache_dir.glob("*.tmp"))
