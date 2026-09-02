from __future__ import annotations

import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)


class _SklearnProbe(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("Estimator construction is outside this test seam")


class _MoEProbe(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("Estimator construction is outside this test seam")


def _sklearn_probe() -> SklearnExecutionTimePredictor:
    predictor = object.__new__(_SklearnProbe)
    predictor._register_missing_profiling_metadata = lambda *_args: pytest.fail(
        "missing metadata must not be registered for a failed input"
    )
    return predictor


def _moe_probe(path: str) -> SklearnMoEExecutionTimePredictor:
    predictor = object.__new__(_MoEProbe)
    predictor._moe_input_file = path
    return predictor


def test_profiling_metadata_registration_raises_for_missing_file(tmp_path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match=str(path)):
        _sklearn_probe()._register_profiling_metadata_from_file(str(path), ["op"])


def test_profiling_metadata_registration_preserves_csv_read_error(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("not used\n", encoding="utf-8")

    def fail_read_csv(_path):
        raise ValueError("malformed profiling row")

    import frontier.execution_time_predictor.sklearn_execution_time_predictor as module

    monkeypatch.setattr(module.pd, "read_csv", fail_read_csv)

    with pytest.raises(ValueError, match=f"{path}.*malformed profiling row") as raised:
        _sklearn_probe()._register_profiling_metadata_from_file(str(path), ["op"])

    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "malformed profiling row"


def test_moe_training_raises_for_missing_file(tmp_path) -> None:
    path = tmp_path / "missing_moe.csv"

    with pytest.raises(FileNotFoundError, match=str(path)):
        _moe_probe(str(path))._train_moe_models()


def test_moe_training_preserves_csv_read_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "broken_moe.csv"
    path.write_text("not used\n", encoding="utf-8")

    def fail_read_csv(_path):
        raise ValueError("malformed MoE profiling row")

    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as module

    monkeypatch.setattr(module.pd, "read_csv", fail_read_csv)

    with pytest.raises(ValueError, match=f"{path}.*malformed MoE profiling row") as raised:
        _moe_probe(str(path))._train_moe_models()

    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "malformed MoE profiling row"
