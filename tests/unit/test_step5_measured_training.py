"""Labelled CPU fixtures validate shared Step fitting, not native measurements."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.step5 import describe_step5_layer
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.step5 import (
    STEP5_FEATURE_COLUMNS,
    STEP5_OPERATOR_NAMES,
    STEP5_SCOPE_NAMES,
    STEP5_VARIANTS,
    load_step5_attention_rows,
    step5_measurement_type,
    step5_row_features,
)
from frontier.profiling.step5.inputs import Step5ProfileInput
from frontier.profiling.step5.rows import build_row, rows_to_dataframe
from frontier.types import ClusterType, MeasurementType


# Explicit root fixture launches, independent of sample count and trace totals.
FIXTURE_ROOT_KERNEL_COUNT = 23


@pytest.fixture(scope="module")
def model():
    return BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")


def _replica(model):
    return SimpleNamespace(
        model_config=model, device="h800", model_name="stepfun-ai/Step-5-Preview",
        attn_tensor_parallel_size=8,
    )


def _scheduler():
    return SimpleNamespace(block_size=16, max_model_len=8192)


def _row(model, layer_id=0, *, query=16, context=4104, graph_mode="none", phase="prefill",
         has_attention_metadata=True):
    cell = Step5ProfileInput(layer_id, (query,), (context,), phase, graph_mode=graph_mode,
                             has_attention_metadata=has_attention_metadata)
    metadata = describe_step5_layer(model, layer_id, attention_tp_size=8, moe_tp_size=1,
                                   expert_parallel_size=1)
    workload = cell.workload(model, block_size=16, max_model_len=8192)
    scopes = metadata.attention_scopes(workload)
    active = {name for name, _ in scopes}
    stats = {}
    for index, name in enumerate((*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES), start=1):
        value = index / 10 + query / 1000 if name in active or name == "step5_layer_e2e" else 0.
        stats[name] = dict(min=value, max=value, mean=value, median=value, std=0., count=10)
        if name == "step5_layer_e2e":
            stats[name]["kernel_count"] = FIXTURE_ROOT_KERNEL_COUNT
            if not has_attention_metadata:
                # Independent positive fixture coverage, never regular-decode substitution.
                stats[name].update(min=.321, max=.321, mean=.321, median=.321, kernel_count=17)
    measurement = step5_measurement_type(graph_mode)
    basis = ("cuda_event_native_runtime_latency" if measurement is MeasurementType.CUDA_EVENT
             else "correlated_kernel_plus_memcpy_duration_sum")
    row = build_row(
        model, cell, tp=8, block_size=16, max_model_len=8192,
        runtime_signature="cpu_fixture_unadmitted", device="h800", time_stats=stats,
        measurement_type=measurement.value,
        evidence={"attention_backend": "cpu_fixture_unadmitted", "timing_basis": basis,
                  "active_operators": json.dumps(sorted(active)), "kernel_count": 999,
                  "evidence_kind": "cpu_fixture"},
    )
    # Predictor-only root ownership is not an additional native instrumentation scope.
    contracts = json.loads(row["typed_operator_contracts"])
    row["typed_operator_contracts"] = json.dumps({name: contracts[name] for name in STEP5_SCOPE_NAMES})
    row.update(has_attention_metadata=int(has_attention_metadata),
               fusion=json.dumps(scopes), block_table_columns=workload.block_table_columns,
               score_regions=workload.score_regions)
    return row


def _write(tmp_path, rows, measurement=MeasurementType.KERNEL_ONLY):
    frame = rows_to_dataframe(rows, measurement_type=measurement)
    path = tmp_path / "attention.csv"
    frame.to_csv(path, index=False)
    return path


def _manager(cache, measurement=MeasurementType.KERNEL_ONLY):
    cache.mkdir(exist_ok=True)
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = measurement
    manager._cache_dir = str(cache)
    manager._all_dummy_mode = False
    manager._kernel_gap_ms = None
    for family in ("eager", "device_event", "kernel_only"):
        for prefix in ("_trained_models_", "_models_by_precision_", "_model_profiling_precision_"):
            setattr(manager, f"{prefix}{family}", {})
        manager.__dict__[f"_trained_models_{family}_by_contract"] = {}
        manager.__dict__[f"_models_by_precision_{family}_by_contract"] = {}
    return manager


def _train(manager, model, path):
    config = RandomForrestExecutionTimePredictorConfig(
        num_estimators=[2], max_depth=[3], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )
    return manager._train_attn_models_for_cluster(
        ClusterType.MONOLITHIC, _replica(model), config, _scheduler(),
        str(path.parent / "deliberately_absent_linear_op.csv"), str(path), set(),
    )


@pytest.mark.parametrize("graph, expected", [
    ("none", MeasurementType.KERNEL_ONLY), ("full", MeasurementType.CUDA_EVENT),
    ("piecewise", MeasurementType.CUDA_EVENT),
])
def test_step_graph_policy_preserves_real_measurement_method(graph, expected):
    assert step5_measurement_type(graph) is expected


@pytest.mark.parametrize("graph,statistic,expected", [("full", "mean", 1.25), ("none", "median", 1.5)])
def test_root_target_selects_real_candidate_not_independent_event_median(model, tmp_path, graph, statistic, expected):
    measurement = step5_measurement_type(graph)
    rows = [_row(model, layer, query=1 if graph == "full" else 16,
                 phase="decode" if graph == "full" else "prefill", graph_mode=graph)
            for layer in (0, 3)]
    for row in rows:
        row["time_stats"]["step5_layer_e2e"].update(min=1., max=2., mean=1.25, median=1.5, std=.4)
    path = _write(tmp_path, rows, measurement)
    models = _train(_manager(tmp_path / "cache", measurement), model, path)
    root = models["step5_layer_e2e"]
    assert root._frontier_target_col == f"time_stats.step5_layer_e2e.{statistic}"
    assert set(root._frontier_exact_lookup.values()) == {expected}
    assert json.loads(root._frontier_measurement_identity)["duration_statistic"] == statistic


def test_step_loader_derives_precision_and_context_features(model, tmp_path):
    source = [_row(model, 0), _row(model, 3, context=4120)]
    path = _write(tmp_path, source)
    manager = _manager(tmp_path / "cache")
    frame = manager._load_attention_df(str(path), _replica(model), _scheduler())
    assert list(frame.profiling_precision) == ["BF16", "BF16"]
    assert set(frame.attention_variant) == {"swa", "sparse_gqa"}
    assert tuple(step5_row_features(source[0])) == STEP5_FEATURE_COLUMNS
    assert frame.max_context_len.tolist() == [4104, 4120]
    assert frame.min_query_len.tolist() == [16, 16]
    assert frame.graph_mode_code.tolist() == [0, 0]
    assert frame.attrs["measurement_identity"]["runtime_stack_signature"] == "cpu_fixture_unadmitted"
    assert STEP5_ATTENTION_FAMILY.execution_enabled
    assert not model.get_model_architecture_profile().attention_linear_ops.sharded_ops
    assert STEP5_OPERATOR_NAMES == ("step5_layer_e2e",)
    assert len(STEP5_SCOPE_NAMES) == 9
    assert set(STEP5_SCOPE_NAMES) == {op.name for op in STEP5_ATTENTION_FAMILY.e2e_trace_ops()}
    assert not set(STEP5_OPERATOR_NAMES) & set(STEP5_SCOPE_NAMES)


def test_step_loader_derives_validated_features_once_per_source_row(model, tmp_path, monkeypatch):
    from frontier.execution_time_predictor import step5

    source = [_row(model, 0), _row(model, 3, context=4120)]
    expected = pd.DataFrame([step5_row_features(row) for row in source])
    path = _write(tmp_path, source)
    feature_calls = []

    def features(row):
        feature_calls.append(row["global_layer_id"])
        return step5_row_features(row)

    monkeypatch.setattr(step5, "step5_row_features", features)
    frame = load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)
    assert feature_calls == [0, 3]
    pd.testing.assert_frame_equal(frame[list(STEP5_FEATURE_COLUMNS)], expected)


@pytest.mark.parametrize("graph", ["full", "piecewise"])
def test_step_loader_accepts_native_graph_event_rows(model, tmp_path, graph):
    row = _row(model, 3, query=1, phase="decode", graph_mode=graph)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    frame = load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.CUDA_EVENT)
    assert frame.iloc[0].measurement_type == "CUDA_EVENT"
    assert frame.iloc[0].phase == "decode"


def test_step_loader_supports_every_source_layer_without_inventing_coverage(model, tmp_path):
    assert len(model.layer_types) == 92
    path = _write(tmp_path, [_row(model, layer) for layer in range(92)])
    frame = load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)
    assert set(frame.global_layer_id) == set(range(92))
    assert frame.attention_variant.value_counts().to_dict() == {"swa": 69, "sparse_gqa": 23}
    models = _train(_manager(tmp_path / "cache"), model, path)
    root = models["step5_layer_e2e"]
    assert root._frontier_step5_domains["global_layer_id"] == tuple(float(layer) for layer in range(92))
    assert len(root._frontier_step5_coverage) == 92


def test_step_native_backend_identity_is_per_variant_but_runtime_is_not(model, tmp_path):
    rows = [_row(model, 0), _row(model, 3)]
    rows[0]["attention_backend"] = "FLASH_ATTN"
    rows[1]["attention_backend"] = "STEP5_SPARSE_ATTN"
    path = _write(tmp_path, rows)
    frame = load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)
    assert frame.attrs["measurement_identity"]["attention_backend"] == {
        "swa": "FLASH_ATTN", "sparse_gqa": "STEP5_SPARSE_ATTN",
    }
    models = _train(_manager(tmp_path / "cache"), model, path)
    assert json.loads(models["step5_layer_e2e"]._frontier_measurement_identity)["attention_backend"] == {
        "swa": "FLASH_ATTN", "sparse_gqa": "STEP5_SPARSE_ATTN",
    }
    rows[1]["runtime_stack_signature"] = "different_image_and_runtime_unadmitted"
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError, match="runtime_stack_signature"):
        load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)


def test_step_rejects_backend_changes_inside_one_variant(model, tmp_path):
    rows = [_row(model, 0), _row(model, 1)]
    rows[1]["attention_backend"] = "different_backend_same_variant"
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError, match="attention_backend"):
        load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)


def test_step_eager_event_diagnostics_cannot_be_relabelled_as_root_physical_cost(model, tmp_path):
    row = _row(model)
    row["measurement_type"] = "CUDA_EVENT"
    row["timing_basis"] = "cuda_event_native_runtime_latency"
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    with pytest.raises(ValueError, match="coverage is unavailable"):
        load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.CUDA_EVENT)


@pytest.mark.parametrize("change, message", [
    ({"measurement_type": "CUDA_EVENT"}, "measurement_type mismatch"),
    ({"timing_basis": "cuda_event_native_runtime_latency"}, "do not relabel"),
    ({"profiling_precision": "FP16"}, "profiling_precision conflicts"),
    ({"model_dtype": "float16"}, "model_dtype conflicts"),
    ({"global_layer_id": 3}, "geometry disagrees|attention_variant disagrees"),
    ({"time_stats.step5_layer_e2e.kernel_count": None}, "kernel coverage missing"),
    ({"time_stats.step5_layer_e2e.kernel_count": 1.5}, "kernel coverage missing"),
    ({"time_stats.step5_sparse_attention.kernel_count": 1}, "inactive scope"),
    ({"time_stats.step5_sparse_attention.std": 1}, "inactive physical scope"),
    ({"num_tensor_parallel_workers": 4}, "coverage is unavailable"),
    ({"cache_dtype": "fp8"}, "coverage is unavailable"),
    ({"max_model_len": 16384}, "max_model_len differs"),
    ({"active_operators": "[]"}, "active_operators differ"),
    ({"time_stats.step5_qkv_projection.median": float("nan")}, "finite nonnegative"),
])
def test_step_bad_coverage_is_rejected_before_fit(model, tmp_path, change, message):
    path = _write(tmp_path, [_row(model)])
    frame = pd.read_csv(path)
    for name, value in change.items():
        frame[name] = value
    frame.to_csv(path, index=False)
    manager = _manager(tmp_path / "cache")
    manager._train_single_model = lambda **kwargs: pytest.fail("invalid coverage reached fit")
    with pytest.raises(ValueError, match=message):
        _train(manager, model, path)
    assert not list((tmp_path / "cache").glob("*.pkl"))


def test_step_real_fit_atomic_cache_exact_lookup_and_warm_reload(model, tmp_path, monkeypatch):
    rows = [_row(model, layer, query=query) for layer in (0, 3) for query in (16, 32)]
    path = _write(tmp_path, rows)
    manager = _manager(tmp_path / "cache")
    models = _train(manager, model, path)
    expected_names = {*STEP5_OPERATOR_NAMES, *(f"{name}_kernel_count" for name in STEP5_OPERATOR_NAMES)}
    assert set(models) == expected_names
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == len(expected_names)
    assert not list((tmp_path / "cache").glob("*.tmp"))
    for row in rows:
        key = tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
        for name in STEP5_OPERATOR_NAMES:
            estimator = models[name]
            assert isinstance(estimator, RandomForestRegressor)
            assert estimator._frontier_exact_lookup[key] == pytest.approx(row["time_stats"][name]["median"])
            assert estimator._frontier_kernel_count_model._frontier_exact_lookup[key] == FIXTURE_ROOT_KERNEL_COUNT
            assert tuple(estimator._frontier_feature_names) == STEP5_FEATURE_COLUMNS
            assert key in estimator._frontier_step5_coverage
            query_frame = pd.DataFrame([step5_row_features(row)], columns=STEP5_FEATURE_COLUMNS)
            assert estimator.predict(query_frame)[0] > 0
        child_sum = sum(row["time_stats"][name]["median"] for name in json.loads(row["active_operators"]))
        assert models["step5_layer_e2e"]._frontier_exact_lookup[key] != pytest.approx(child_sum)
    assert models["step5_layer_e2e"]._frontier_step5_domains["global_layer_id"] == (0., 3.)
    assert models["step5_layer_e2e"]._frontier_step5_domains["attention_variant_code"] == tuple(float(v) for v in STEP5_VARIANTS.values())
    assert not set(STEP5_SCOPE_NAMES) & set(models)
    assert set(manager._trained_models_kernel_only) == expected_names
    assert not manager._trained_models_eager
    for cache_file in (tmp_path / "cache").glob("*.pkl"):
        with cache_file.open("rb") as stream:
            stored = pickle.load(stream)
        assert stored._frontier_step5_coverage
        assert stored._frontier_feature_bounds

    warm_manager = _manager(tmp_path / "cache")
    monkeypatch.setattr(warm_manager, "_create_estimator_and_params",
                        lambda *_: pytest.fail("warm cache attempted refit"))
    reloaded = _train(warm_manager, model, path)
    for name in expected_names:
        assert reloaded[name]._frontier_exact_lookup == models[name]._frontier_exact_lookup
        assert reloaded[name]._frontier_step5_coverage == models[name]._frontier_step5_coverage
        assert reloaded[name]._frontier_model_hash == models[name]._frontier_model_hash

    # An existing path and hash are insufficient when its provenance was changed.
    cached = next((tmp_path / "cache").glob("step5_layer_e2e_*.pkl"))
    with cached.open("rb") as stream:
        tampered = pickle.load(stream)
    tampered._frontier_measurement_identity = "{}"
    with cached.open("wb") as stream:
        pickle.dump(tampered, stream)
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        _train(warm_manager, model, path)


def test_step_graph_fit_registers_event_family_and_never_generic_projections(model, tmp_path):
    path = _write(tmp_path, [_row(model, layer, query=1, phase="decode", graph_mode="full")
                             for layer in (0, 3)], MeasurementType.CUDA_EVENT)
    manager = _manager(tmp_path / "cache", MeasurementType.CUDA_EVENT)
    models = _train(manager, model, path)
    assert set(models) == set(manager._trained_models_eager)
    assert not manager._trained_models_kernel_only
    assert not {"attn_pre_proj", "attn_post_proj", "attn_rope"} & set(models)
    assert set(models) == {"step5_layer_e2e", "step5_layer_e2e_kernel_count"}
