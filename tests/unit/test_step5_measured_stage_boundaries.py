"""Optional measured Stage boundaries use the existing cache, not native defaults."""

import json
import pickle

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.step5 import STEP5_STAGE_BOUNDARY_NAMES
from frontier.operators.binding import bind_operator_query
from frontier.types import ClusterType, MeasurementType
import test_step5_measured_ffn as compute_fixtures
from test_step5_measured_training import _manager


@pytest.fixture(scope="module")
def model():
    return BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")


@pytest.fixture(params=[MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT])
def measurement(request, monkeypatch):
    actual = request.param
    source_metadata = compute_fixtures._metadata
    monkeypatch.setattr(compute_fixtures, "_metadata", lambda model: {
        **source_metadata(model), "measurement_type": actual.value,
    })
    return actual


def _fit(manager, model, path):
    return manager._train_step5_input_layernorm_for_cluster(
        ClusterType.MONOLITHIC, compute_fixtures._replica(model), compute_fixtures._config(),
        str(path), set(),
    )


def test_optional_stage_targets_fit_register_and_reload_original_names(model, measurement, tmp_path, monkeypatch):
    linear, _ = compute_fixtures._tables(tmp_path, model, include_stage_boundaries=True)
    manager = _manager(tmp_path / "cache", measurement)
    models = _fit(manager, model, linear)
    assert set(models) == {"input_layernorm", *STEP5_STAGE_BOUNDARY_NAMES}
    for name in STEP5_STAGE_BOUNDARY_NAMES:
        estimator = models[name]
        assert manager.get_model(name) is estimator
        count = manager.get_model(f"{name}_kernel_count")
        assert estimator._frontier_kernel_count_model is count
        assert estimator._frontier_exact_lookup[(16.,)] == pytest.approx(.16)
        assert estimator.predict(pd.DataFrame({"num_tokens": [16]}))[0] > 0
        assert count._frontier_exact_lookup[(16.,)] == 7
        assert estimator._frontier_cache_model_name == name
        assert count._frontier_cache_model_name == f"{name}_kernel_count"
        identity = json.loads(estimator._frontier_measurement_identity)
        assert identity["measurement_type"] == measurement.value
        assert identity["evidence_kind"] == "cpu_fixture"
        assert identity["scope"] == "stage"
        assert identity["tensor_parallel_size"] == 1
        assert identity["operator_precision"] == "BF16"
        assert identity["feature_columns"] == ["num_tokens"]
        assert identity["tensor_dtype"]["input"] == "BF16"
        assert identity["tensor_dtype"]["output"] == "FP32"
        assert estimator._frontier_feature_bounds == {"num_tokens": (16., 32.)}
        operator = bind_operator_query(name).operator
        assert not operator.predictor_target
        assert operator.profiling_target and operator.is_applicable(model)
        assert not operator.is_applicable(None)
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 6
    for path in (tmp_path / "cache").glob("*.pkl"):
        with path.open("rb") as stream:
            stored = pickle.load(stream)
        assert stored._frontier_cache_model_name
        assert stored._frontier_feature_names == ["num_tokens"]
        assert stored._frontier_measurement_identity
    warm = _manager(tmp_path / "cache", measurement)
    monkeypatch.setattr(warm, "_create_estimator_and_params", lambda *_: pytest.fail("warm boundary refitted"))
    loaded = _fit(warm, model, linear)
    for name in STEP5_STAGE_BOUNDARY_NAMES:
        assert loaded[name]._frontier_model_hash == models[name]._frontier_model_hash
        assert loaded[name]._frontier_kernel_count_model._frontier_model_hash == models[name]._frontier_kernel_count_model._frontier_model_hash


def test_absent_boundary_targets_leave_norm_only(model, measurement, tmp_path):
    linear, _ = compute_fixtures._tables(tmp_path, model)
    manager = _manager(tmp_path / "cache", measurement)
    assert set(_fit(manager, model, linear)) == {"input_layernorm"}
    assert not list((tmp_path / "cache").glob("input_cast_*.pkl"))
    assert not list((tmp_path / "cache").glob("final_residual_*.pkl"))


def test_explicit_columns_need_no_new_producer_contract_or_other_boundary(model, measurement, tmp_path):
    linear, _ = compute_fixtures._tables(tmp_path, model, include_stage_boundaries=True)
    frame = pd.read_csv(linear).drop(columns=[
        "time_stats.final_residual.median", "time_stats.final_residual.kernel_count",
    ])
    frame["typed_operator_contracts"] = frame["typed_operator_contracts"].map(lambda value: json.dumps({
        name: contract for name, contract in json.loads(value).items()
        if name not in STEP5_STAGE_BOUNDARY_NAMES
    }))
    frame.to_csv(linear, index=False)
    models = _fit(_manager(tmp_path / "cache", measurement), model, linear)
    assert set(models) == {"input_layernorm", "input_cast"}
    assert models["input_cast"]._frontier_kernel_count_model._frontier_exact_lookup[(16.,)] == 7


@pytest.mark.parametrize("name", STEP5_STAGE_BOUNDARY_NAMES)
def test_explicit_duration_requires_counts_only_for_kernel_only_steps(model, measurement, tmp_path, name):
    linear, _ = compute_fixtures._tables(tmp_path, model, include_stage_boundaries=True)
    frame = pd.read_csv(linear).drop(columns=[f"time_stats.{name}.kernel_count"])
    frame.to_csv(linear, index=False)
    manager = _manager(tmp_path / "cache", measurement)
    if measurement is MeasurementType.CUDA_EVENT:
        # Event-timed steps pay no per-kernel gap, so they fit without kernel counts.
        models = _fit(manager, model, linear)
        assert not hasattr(models[name], "_frontier_kernel_count_model")
        assert not list((tmp_path / "cache").glob(f"{name}_kernel_count_*.pkl"))
        return
    manager._train_single_model = lambda *_, **__: pytest.fail("invalid boundary reached fitting")
    with pytest.raises(ValueError, match=f"actual positive integral kernel coverage: time_stats.{name}"):
        _fit(manager, model, linear)
    assert not list((tmp_path / "cache").glob("*.pkl"))


@pytest.mark.parametrize("column,value,message", [
    ("time_stats.input_cast.median", float("nan"), "positive finite duration"),
    ("time_stats.input_cast.median", float("inf"), "positive finite duration"),
    ("time_stats.input_cast.kernel_count", float("inf"), "actual positive integral kernel"),
    ("time_stats.input_cast.kernel_count", 0, "actual positive integral kernel"),
    ("time_stats.input_cast.kernel_count", 1.5, "actual positive integral kernel"),
    ("n_embd", 1, "source hidden width"),
])
def test_malformed_explicit_stage_targets_are_rejected(model, measurement, tmp_path, column, value, message):
    linear, _ = compute_fixtures._tables(tmp_path, model, include_stage_boundaries=True)
    frame = pd.read_csv(linear)
    frame.loc[frame.num_tensor_parallel_workers.eq(1), column] = value
    frame.to_csv(linear, index=False)
    manager = _manager(tmp_path / "cache", measurement)
    with pytest.raises(ValueError, match=message):
        _fit(manager, model, linear)
    assert not list((tmp_path / "cache").glob("*.pkl"))
