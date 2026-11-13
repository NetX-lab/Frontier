"""Actual shared fit/cache/bundle/cost replay on explicitly unadmitted CPU rows."""

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config.model_config import BaseModelConfig
from frontier.entities import DummyForwardBatch, ExecutionTime
from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS, step5_row_features
from frontier.execution_time_predictor.step5_cost import (
    predict_step5_attention_time, step5_batch_features, step5_predictions,
)
from frontier.types import ClusterType, MeasurementType
from test_execution_time_op_times import _base_execution_kwargs
from test_on_demand_prediction_contract import _ConcretePredictor
from test_step5_measured_training import (
    FIXTURE_ROOT_KERNEL_COUNT, _manager, _replica, _row, _scheduler, _train, _write,
)


def _predictor(model, models, method):
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._model_config = model
    predictor._replica_config = _replica(model)
    predictor._replica_config.moe_tensor_parallel_size = 1
    predictor._replica_config.moe_expert_parallel_size = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._block_size = _scheduler().block_size
    predictor._step5_max_model_len = _scheduler().max_model_len
    predictor._kernel_gap_ms = {"eager": .001, "cuda_graph": .002}
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._step_measurement_type = MeasurementType.CUDA_EVENT
    for suffix, measurement in (("eager", MeasurementType.CUDA_EVENT),
                                ("kernel_only", MeasurementType.KERNEL_ONLY)):
        selected = models if measurement is method else {}
        setattr(predictor, f"_models_{suffix}", selected)
        setattr(predictor, f"_predictions_{suffix}", step5_predictions(selected, measurement))
        for kind in ("compute", "attention", "moe", "cpu_overhead"):
            setattr(predictor, f"_{kind}_input_file_{suffix}", "")
    predictor._activate_measurement_type(MeasurementType.CUDA_EVENT)
    return predictor


def _batch(query=16, context=4104, graph="none"):
    return SimpleNamespace(
        num_tokens=[query], num_context_tokens=[context],
        num_prefill_tokens=query if graph == "none" else 0,
        num_decode_tokens=0 if graph == "none" else query,
        get_effective_total_tokens_for_compute=lambda _: query,
        get_decode_cuda_graph_runtime_mode=lambda: graph.upper(),
    )


def _bundle_identity(model, models):
    identity = json.loads(models["step5_layer_e2e"]._frontier_measurement_identity)
    identity.update(
        model_name=model.get_name(), role="monolithic",
        topology={"tp": 8, "dp": 1, "ep": 1, "pp": 1},
        graph_policy="declared_cpu_fixture", coverage={
            "operators": sorted(models),
            "feature_keys": [list(key) for key in sorted(models["step5_layer_e2e"]._frontier_step5_coverage)],
        },
    )
    return identity


@pytest.mark.parametrize("graph,query,phase,method", [
    ("none", 16, "prefill", MeasurementType.KERNEL_ONLY),
    ("full", 1, "decode", MeasurementType.CUDA_EVENT),
])
def test_real_fit_bundle_load_lookup_and_physical_cost_replay(tmp_path, graph, query, phase, method):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    # These source-supported live lengths allocate the same 512-column physical
    # table as the configured serving capacity; DSA decode history stays aligned.
    context = 8192 - query if phase == "prefill" else 8184
    source = [_row(model, layer, query=query, context=context,
                   graph_mode=graph, phase=phase) for layer in (0, 3)]
    path = _write(tmp_path, source, method)
    manager = _manager(tmp_path / "cache", method)
    models = _train(manager, model, path)
    identity = _bundle_identity(model, models)
    bundle = manager.publish_model_bundle(identity, models)
    assert identity["evidence_kind"] == "cpu_fixture" and "native_admission" not in bundle
    loaded = manager.load_model_bundle(bundle["path"], identity)
    predictor = _predictor(model, loaded, method)
    batch = _batch(query=query, context=context, graph=graph)

    for row in source:
        layer_id = row["global_layer_id"]
        actual, _ = step5_batch_features(predictor, batch, layer_id)
        assert actual == step5_row_features(row)
        attention = predict_step5_attention_time(predictor, batch, layer_id)
        assert predictor._active_measurement_type is MeasurementType.CUDA_EVENT
        expected = row["time_stats"]["step5_layer_e2e"]["median"]
        assert attention.total_time() == pytest.approx(expected)
        assert attention.kernel_count == FIXTURE_ROOT_KERNEL_COUNT
        proof = attention.operator_costs["step5_layer_e2e"]
        assert proof.evidence_kind == "cpu_fixture" and proof.exact_lookup
        assert proof.measurement_type is method
        assert proof.includes_kernel_gap == (method is MeasurementType.CUDA_EVENT)
        assert set(attention.operator_times.op_times) == {"step5_layer_e2e"}

        # Replay only this measured physical owner; all unrelated fixture costs are zero.
        def replay():
            predictor._record_attention_kernels(attention.kernel_count, attention.measurement_type)
            kwargs = {name: (0. if isinstance(value, (int, float)) and not isinstance(value, bool)
                             else value) for name, value in _base_execution_kwargs().items()}
            kwargs.update(num_layers_per_pipeline_stage=1,
                          attention_operator_times=attention.operator_times)
            return ExecutionTime(**kwargs)

        execution = predictor._predict_layer_with_kernel_gap(replay)
        gap = .001 * FIXTURE_ROOT_KERNEL_COUNT if method is MeasurementType.KERNEL_ONLY else 0.
        assert execution.get_single_layer_attention_scope_time() == pytest.approx(expected + gap)
        assert execution.kernel_gap_time == pytest.approx(gap)
        assert execution.attention_time_component.kernel_count == FIXTURE_ROOT_KERNEL_COUNT
        assert execution.attention_time_component.operator_costs == attention.operator_costs
    assert len(list(Path(manager._cache_dir).glob("*.pkl"))) == 2


def test_missing_native_shape_or_gap_never_gets_a_fixture_default(tmp_path):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    path = _write(tmp_path, [_row(model)])
    models = _train(_manager(tmp_path / "cache"), model, path)
    predictor = _predictor(model, models, MeasurementType.KERNEL_ONLY)
    runtime, _ = step5_batch_features(predictor, _batch(), 0)
    assert runtime["block_table_columns"] == 512
    assert step5_row_features(_row(model))["block_table_columns"] == 258
    with pytest.raises(ValueError, match="block_table_columns=512.0 not in"):
        predict_step5_attention_time(predictor, _batch(), 0)
    with pytest.raises(ValueError, match="valid_tokens=32.0 not in"):
        predict_step5_attention_time(predictor, _batch(query=32), 0)
    with pytest.raises(ValueError, match="no measured layer with the same attention block"):
        predict_step5_attention_time(predictor, _batch(), 3)
    predictor._kernel_gap_ms = None
    with pytest.raises(ValueError, match="kernel_gap.csv coverage"):
        predict_step5_attention_time(predictor, _batch(), 0)


def test_unmeasured_layers_and_shapes_are_priced_within_the_measured_category(tmp_path):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    contexts = (4104, 8184)
    rows = [_row(model, layer, query=1, context=context, graph_mode="full", phase="decode",
                 block_table_columns=512) for layer in (0, 3) for context in contexts]
    method = MeasurementType.CUDA_EVENT
    models = _train(_manager(tmp_path / "cache", method), model, _write(tmp_path, rows, method))
    predictor = _predictor(model, models, method)
    # Layers 4 and 7 have the attention blocks of the measured layers 0 and 3.
    for measured, layer in ((0, 4), (3, 7)):
        for context in (*contexts, 6000):
            batch = _batch(query=1, context=context, graph="full")
            expected = predict_step5_attention_time(predictor, batch, measured)
            actual = predict_step5_attention_time(predictor, batch, layer)
            assert actual.total_time() == expected.total_time() > 0
            assert actual.operator_costs["step5_layer_e2e"].exact_lookup == (context in contexts)
    with pytest.raises(ValueError, match="max_context_len=8190.0 not in"):
        predict_step5_attention_time(predictor, _batch(query=1, context=8190, graph="full"), 4)
    # Measured DSA contexts are region-aligned, so a DSA context up to one region
    # above the highest measured context is priced at that context; the lower
    # bound stays exact.
    beyond = predict_step5_attention_time(predictor, _batch(query=1, context=8190, graph="full"), 7)
    boundary = predict_step5_attention_time(predictor, _batch(query=1, context=8184, graph="full"), 7)
    assert beyond.total_time() == boundary.total_time() > 0
    assert not beyond.operator_costs["step5_layer_e2e"].exact_lookup
    with pytest.raises(ValueError, match="min_context_len=4096.0 not in"):
        predict_step5_attention_time(predictor, _batch(query=1, context=4096, graph="full"), 7)
    # Eager prompt tails were not measured.
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, _batch(query=16, context=4104), 4)


@pytest.mark.parametrize("layer_id", [0, 3])
def test_request_free_batch_has_its_own_no_cache_identity(layer_id):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    predictor = _predictor(model, {}, MeasurementType.KERNEL_ONLY)
    batch = DummyForwardBatch(0, 0, 0)
    features, workload = step5_batch_features(predictor, batch, layer_id)
    assert features["has_attention_metadata"] == 0
    assert features["block_table_columns"] == features["score_regions"] == 0
    assert features["batch_num_tokens"] == 1
    assert not workload.has_attention_metadata
    assert workload.graph_mode == "none"
    assert batch.requests == []


@pytest.mark.parametrize("layer_id", [0, 3])
def test_request_free_exact_replay_retains_positive_source_work(tmp_path, layer_id):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    source = _row(model, layer_id, query=1, context=0, phase="decode",
                  has_attention_metadata=False)
    regular = _row(model, layer_id, query=1, context=0, phase="decode")
    path = _write(tmp_path, [source, regular])
    models = _train(_manager(tmp_path / "cache"), model, path)
    predictor = _predictor(model, models, MeasurementType.KERNEL_ONLY)
    batch = DummyForwardBatch(0, 0, 0)
    features, workload = step5_batch_features(predictor, batch, layer_id)
    assert features == step5_row_features(source)
    assert features != step5_row_features(regular)
    metadata = describe_step5_layer(model, layer_id, attention_tp_size=8,
                                   moe_tp_size=1, expert_parallel_size=1)
    scopes = metadata.attention_scopes(workload)
    assert scopes == metadata.request_free_attention_scopes()
    work = {item for _, items in scopes for item in items}
    assert {"q_projection", "qk_norm", "output_projection", "output_zero_fill"} <= work
    assert not {"raw_kv_write", "summary_update", "selector_scoring", "sparse_gqa"} & work
    timing = predict_step5_attention_time(predictor, batch, layer_id)
    assert timing.total_time() == pytest.approx(.321)
    assert timing.kernel_count == 17
    assert timing.measurement_type is MeasurementType.KERNEL_ONLY
    assert timing.operator_costs["step5_layer_e2e"].exact_lookup
    assert predictor._active_measurement_type is MeasurementType.CUDA_EVENT

    # Ordinary single-token coverage cannot supply the request-free root.
    key = tuple(features[name] for name in STEP5_FEATURE_COLUMNS)
    for estimator in models.values():
        estimator._frontier_step5_coverage -= {key}
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, batch, layer_id)


@pytest.mark.parametrize("change,match", [
    ({"has_attention_metadata": 2}, "must be 0 or 1"),
    ({"graph_mode": "full"}, "request-free workload"),
    ({"block_table_columns": 1}, "request-free workload"),
    ({"context_lens": "[1]"}, "request-free workload"),
])
def test_request_free_rows_reject_unproven_geometry(change, match):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    row = _row(model, 3, query=1, context=0, phase="decode", has_attention_metadata=False)
    with pytest.raises(ValueError, match=match):
        step5_row_features({**row, **change})


def test_request_free_loader_rejects_regular_fusion_and_missing_identity(tmp_path):
    from frontier.execution_time_predictor.step5 import load_step5_attention_rows

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    row = _row(model, 3, query=1, context=0, phase="decode", has_attention_metadata=False)
    row["fusion"] = _row(model, 3, query=1, context=0, phase="decode")["fusion"]
    path = _write(tmp_path, [row])
    with pytest.raises(ValueError, match="fusion scopes differ"):
        load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)
    del row["has_attention_metadata"]
    path = _write(tmp_path, [row])
    with pytest.raises(ValueError, match="missing columns.*has_attention_metadata"):
        load_step5_attention_rows(path, _replica(model), _scheduler(), MeasurementType.KERNEL_ONLY)
