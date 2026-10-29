"""Truthful CPU producer rows through the existing measured fit/cache/lookup."""

import json

import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS, step5_row_features
from frontier.execution_time_predictor.step5_cost import predict_step5_attention_time
from frontier.profiling.step5.inputs import MRv2DummyDescriptor, Step5ProfileInput
from frontier.profiling.step5.rows import build_row
from frontier.types import MeasurementType
from test_step5_measured_training import (
    STEP5_OPERATOR_NAMES, STEP5_SCOPE_NAMES, _manager, _train, _write,
)
from test_step5_measured_replay import _batch, _bundle_identity, _predictor


def producer_row(model, cell, *, duration, count):
    metadata = describe_step5_layer(model, cell.global_layer_id, attention_tp_size=8,
                                   moe_tp_size=1, expert_parallel_size=1)
    scopes = metadata.attention_scopes(cell.workload(model, block_size=16, max_model_len=8192))
    active = {name for name, _ in scopes}
    stats = {}
    for name in (*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES):
        value = duration if name in active or name in STEP5_OPERATOR_NAMES else 0.
        stats[name] = dict(min=value, max=value, mean=value, median=value, std=0., count=10)
        if name in STEP5_OPERATOR_NAMES:
            stats[name]["kernel_count"] = count
    measurement = MeasurementType.KERNEL_ONLY if cell.graph_mode == "none" else MeasurementType.CUDA_EVENT
    return build_row(model, cell, tp=8, block_size=16, max_model_len=8192,
        runtime_signature="cpu_fixture_unadmitted", device="h800", time_stats=stats,
        measurement_type=measurement.value, evidence={
            "attention_backend": "cpu_fixture_unadmitted", "evidence_kind": "cpu_fixture",
            "active_operators": json.dumps(sorted(active)),
            "timing_basis": ("correlated_kernel_plus_memcpy_duration_sum" if cell.graph_mode == "none"
                             else "cuda_event_native_runtime_latency"),
        })


@pytest.mark.parametrize("tokens,requests,graph,max_query", [
    (8, 3, "none", 3), (8, 8, "full", 1), (16, None, "piecewise", None),
])
def test_actual_mrv2_producer_mixed_fit_cache_and_exact_count(
        tmp_path, monkeypatch, tokens, requests, graph, max_query):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(tokens, requests, graph, max_query, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    dummy = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    request = Step5ProfileInput(3, dummy.query_lens, dummy.context_lens, dummy.phase, graph,
                               block_table_columns=512)
    rows = [producer_row(model, request, duration=.25, count=23),
            producer_row(model, dummy, duration=.5, count=17)]
    method = MeasurementType.KERNEL_ONLY if graph == "none" else MeasurementType.CUDA_EVENT
    path = _write(tmp_path, rows, method)
    models = _train(_manager(tmp_path / "cache", method), model, path)
    keys = [tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS) for row in rows]
    assert keys[0] != keys[1], "request-owned and PAD_SLOT_ID physical rows must not alias"
    numeric = [{name: value for name, value in step5_row_features(row).items()
                if name != "native_workload_code"} for row in rows]
    assert numeric[0] == numeric[1]
    for name, expected in (("step5_layer_e2e", [.25, .5]), ("step5_layer_e2e_kernel_count", [23, 17])):
        estimator = models[name]
        assert [estimator._frontier_exact_lookup[key] for key in keys] == expected
        assert estimator._frontier_step5_coverage == frozenset(keys)
        identity = json.loads(estimator._frontier_measurement_identity)
        assert identity["physical_scope"] == {
            row["native_workload"]: row["physical_scope"] for row in rows
        }
        assert estimator._frontier_step5_domains["native_workload_code"] == (0., 1.)
    manager = _manager(tmp_path / "cache", method)
    monkeypatch.setattr(manager, "_create_estimator_and_params", lambda *_: pytest.fail("warm refit"))
    warm = _train(manager, model, path)
    for name in models:
        assert warm[name]._frontier_model_hash == models[name]._frontier_model_hash
        assert warm[name]._frontier_exact_lookup == models[name]._frontier_exact_lookup

    identity = _bundle_identity(model, warm)
    bundle = manager.publish_model_bundle(identity, warm)
    loaded = manager.load_model_bundle(bundle["path"], identity)
    predictor = _predictor(model, loaded, method)
    for row, cell in zip(rows, (request, dummy)):
        # Exercise the existing cost path with truthful producer features. Native
        # scheduler binding is deliberately outside this CPU consumer regression.
        monkeypatch.setattr("frontier.execution_time_predictor.step5_cost.step5_batch_features",
            lambda *_: (step5_row_features(row), cell.workload(model, block_size=16, max_model_len=8192)))
        cost = predict_step5_attention_time(predictor, object(), 3)
        assert cost.total_time() == row["time_stats"]["step5_layer_e2e"]["mean"]
        assert cost.kernel_count == row["time_stats"]["step5_layer_e2e"]["kernel_count"]
        assert cost.operator_costs["step5_layer_e2e"].exact_lookup


@pytest.mark.parametrize("trained_kind", ["attention_workload", "mrv2_dp_dummy"])
def test_same_shape_missing_kind_coverage_fails_without_borrowing_exact_key(tmp_path, monkeypatch, trained_kind):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(2, 2, "full", None, "CPU_OBSERVED_FULL_NULL", 1)
    dummy = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    request = Step5ProfileInput(3, (1, 1), (0, 0), "decode", "full", block_table_columns=512)
    cells = {"attention_workload": request, "mrv2_dp_dummy": dummy}
    row = producer_row(model, cells[trained_kind], duration=.25, count=23)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    models = _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)
    predictor = _predictor(model, models, MeasurementType.CUDA_EVENT)
    missing_kind = next(kind for kind in cells if kind != trained_kind)
    query = producer_row(model, cells[missing_kind], duration=.5, count=17)
    assert models["step5_layer_e2e"]._frontier_step5_domains["native_workload_code"] == (
        step5_row_features(row)["native_workload_code"],)
    monkeypatch.setattr("frontier.execution_time_predictor.step5_cost.step5_batch_features",
        lambda *_: (step5_row_features(query), cells[missing_kind].workload(model, block_size=16, max_model_len=8192)))
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, object(), 3)


@pytest.mark.parametrize("change", [
    {"native_workload": None}, {"native_workload": "unknown"},
    {"native_workload": "attention_workload"},
    {"native_workload": "attention_workload", "physical_scope": "native_attention_layer_without_tp_allreduce"},
    {"physical_scope": "native_attention_layer_without_tp_allreduce"},
    {"native_runner": "MRv1"}, {"native_runner": None},
    {"dummy_owned_requests": 1}, {"dummy_local_num_tokens": 2},
    {"dummy_is_padding": False}, {"dummy_has_prefill": True},
    {"slot_mapping_kind": "request_slots"}, {"cache_policy": "live_pages_mutated"},
    {"valid_token_semantics": "owned_serving_tokens"}, {"has_attention_metadata": 0},
    {"dummy_descriptor": None}, {"dummy_descriptor": "{}"},
    {"phase": "prefill"}, {"context_lens": "[1,1]"}, {"padded_tokens": 1},
])
def test_invalid_mrv2_kind_or_provenance_rejects_before_fit(tmp_path, change):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(2, 2, "full", None, "CPU_OBSERVED_FULL_NULL", 1)
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    row = producer_row(model, cell, duration=.5, count=17)
    row.update(change)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    manager = _manager(tmp_path / "cache", MeasurementType.CUDA_EVENT)
    manager._train_single_model = lambda **_: pytest.fail("invalid provenance reached fit")
    with pytest.raises(ValueError):
        _train(manager, model, path)
    assert not list((tmp_path / "cache").glob("*.pkl"))


@pytest.mark.parametrize("change", [
    {"num_tokens": 3}, {"num_tokens": True}, {"num_reqs": 1}, {"num_reqs": 3},
    {"graph_mode": "none"}, {"max_query_len": 2}, {"uniform_token_count": 2},
    {"uniform_token_count": None}, {"evidence": ""},
])
def test_descriptor_must_match_actual_split_counts_phase_mode(tmp_path, change):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(2, 2, "full", None, "CPU_OBSERVED_FULL_NULL", 1)
    row = producer_row(model, Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512),
                       duration=.5, count=17)
    values = json.loads(row["dummy_descriptor"])
    values.update(change)
    row["dummy_descriptor"] = json.dumps(values)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    with pytest.raises(ValueError, match="descriptor"):
        _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)


def test_native_observed_full_null_dispatch_is_preserved_and_consumed(tmp_path):
    # Source: capture01 prefill_pid_2199:86 and decode_pid_2594:302.
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    for tokens in (1, 2):
        descriptor = MRv2DummyDescriptor(tokens, tokens, "full", None,
            "capture01_native_forward_start_InputBatch", 1)
        cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=448)
        assert cell.query_lens == (1,) * tokens
        row = producer_row(model, cell, duration=.5, count=17)
        assert json.loads(row["dummy_descriptor"])["max_query_len"] is None
        path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
        models = _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)
        key = tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
        assert models["step5_layer_e2e"]._frontier_exact_lookup[key] == .5


def test_kind_bound_scope_tampering_is_rejected_by_cost_lookup(tmp_path, monkeypatch):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(2, 2, "full", None, "CPU_OBSERVED_FULL_NULL", 1)
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    row = producer_row(model, cell, duration=.5, count=17)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    models = _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)
    for estimator in models.values():
        identity = json.loads(estimator._frontier_measurement_identity)
        identity["physical_scope"]["mrv2_dp_dummy"] = "native_attention_layer_without_tp_allreduce"
        estimator._frontier_measurement_identity = json.dumps(identity)
    predictor = _predictor(model, models, MeasurementType.CUDA_EVENT)
    monkeypatch.setattr("frontier.execution_time_predictor.step5_cost.step5_batch_features",
        lambda *_: (step5_row_features(row), cell.workload(model, block_size=16, max_model_len=8192)))
    with pytest.raises(ValueError, match="kind-bound physical identity"):
        predict_step5_attention_time(predictor, object(), 3)


def test_request_owned_runtime_cannot_borrow_same_shape_dummy_key(tmp_path):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(1, 1, "full", None, "CPU_OBSERVED_FULL_NULL", 1)
    cell = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    row = producer_row(model, cell, duration=.5, count=17)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    models = _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)
    predictor = _predictor(model, models, MeasurementType.CUDA_EVENT)
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, _batch(query=1, context=0, graph="full"), 3)


def test_all_three_truthful_kinds_share_one_cache_without_scope_aliasing(tmp_path):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(1, 1, "none", None, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    cells = [Step5ProfileInput(3, (1,), (0,), "decode", block_table_columns=512),
             Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512),
             Step5ProfileInput(3, (1,), (0,), "decode", has_attention_metadata=False)]
    rows = [producer_row(model, cell, duration=value, count=count)
            for cell, value, count in zip(cells, (.25, .5, .75), (23, 17, 11))]
    path = _write(tmp_path, rows)
    models = _train(_manager(tmp_path / "cache"), model, path)
    root = models["step5_layer_e2e"]
    assert root._frontier_step5_domains["native_workload_code"] == (0., 1., 2.)
    assert json.loads(root._frontier_measurement_identity)["physical_scope"] == {
        row["native_workload"]: row["physical_scope"] for row in rows
    }
    for row in rows:
        key = tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
        assert root._frontier_exact_lookup[key] == row["time_stats"]["step5_layer_e2e"]["median"]
        assert models["step5_layer_e2e_kernel_count"]._frontier_exact_lookup[key] == row["time_stats"]["step5_layer_e2e"]["kernel_count"]


def test_old_step_feature_schema_is_not_accepted_for_cost_predictions(tmp_path):
    from frontier.execution_time_predictor.step5_cost import step5_predictions

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    row = producer_row(model, Step5ProfileInput(3, (1,), (0,), "decode", "full"), duration=.25, count=23)
    path = _write(tmp_path, [row], MeasurementType.CUDA_EVENT)
    models = _train(_manager(tmp_path / "cache", MeasurementType.CUDA_EVENT), model, path)
    models["step5_layer_e2e"]._frontier_feature_names = [
        name for name in STEP5_FEATURE_COLUMNS if name != "native_workload_code"
    ]
    with pytest.raises(ValueError, match="feature schema mismatch"):
        step5_predictions(models, MeasurementType.CUDA_EVENT)
