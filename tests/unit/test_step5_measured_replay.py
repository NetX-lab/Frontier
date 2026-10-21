"""Actual shared fit/cache/bundle/cost replay on explicitly unadmitted CPU rows."""

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.entities import ExecutionTime
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
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, _batch(), 0)
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, _batch(query=32), 0)
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predict_step5_attention_time(predictor, _batch(), 3)
    predictor._kernel_gap_ms = None
    with pytest.raises(ValueError, match="kernel_gap.csv coverage"):
        predict_step5_attention_time(predictor, _batch(), 0)
