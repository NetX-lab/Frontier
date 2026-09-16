from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
from predictor_cache_fixtures import CacheFixtureDisaggregationPredictor, CacheFixturePredictor, cache_model

from frontier.config import (
    MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig, VllmV1SchedulerConfig,
)
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType


class _DummySklearnPredictor(SklearnExecutionTimePredictor):
    def __init__(self):
        replica = ReplicaConfig(model_name="meta-llama/Llama-2-7b-hf")
        replica.model_config = replace(replica.model_config, num_kv_heads=8)
        super().__init__(
            RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True),
            replica,
            VllmV1SchedulerConfig(),
            MetricsConfig(),
        )

    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _IdentityQuantizationManager:
    def adjust_compute_time(self, _op_name, value, _cluster_type):
        return value

    def adjust_tensor_size(self, _op_name, value, _cluster_type):
        return value


class _RecordingQuantizationManager:
    def __init__(self):
        self.adjusted_compute_ops: list[str] = []

    def adjust_compute_time(self, op_name, value, _cluster_type):
        self.adjusted_compute_ops.append(op_name)
        return value

    def adjust_tensor_size(self, _op_name, value, _cluster_type):
        return value


class _DummyBatch:
    id = 1
    size = 4
    num_tokens = 32
    total_num_tokens = 32
    num_prefill_tokens = 16
    num_decode_tokens = 16
    is_idle = False
    spec_decode_metadata = None
    requests = [SimpleNamespace(num_prefill_tokens=16)]


def _build_predictor() -> _DummySklearnPredictor:
    predictor = _DummySklearnPredictor()
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._get_attention_rope_execution_time = lambda _batch: 1.0
    predictor._get_attention_kv_cache_save_execution_time = lambda _batch: 2.0
    predictor._get_attention_decode_execution_time = lambda _batch: 3.0
    predictor._get_attention_prefill_execution_time = lambda _batch: 4.0
    predictor._get_attention_layer_pre_proj_execution_time = lambda _batch: 5.0
    predictor._get_attention_layer_post_proj_execution_time = lambda _batch: 6.0
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: 7.0
    predictor._get_mlp_layer_down_proj_execution_time = lambda _batch: 8.0
    predictor._get_mlp_layer_act_execution_time = lambda _batch: 9.0
    predictor._get_attn_norm_layer_act_execution_time = lambda _batch: 10.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 11.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 12.0
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._get_pipeline_parallel_communication_time = lambda _batch: 0.0
    predictor._get_tensor_parallel_communication_time = lambda _batch: 0.0
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = (
        lambda *_args, **_kwargs: None
    )
    predictor._activate_measurement_type = lambda *_args, **_kwargs: None
    predictor._validate_prediction_value = (
        lambda value, _op_name, _batch, _context: value
    )
    return predictor


def test_dense_predictor_exposes_include_attention_contract() -> None:
    for predictor_type in (
        BaseExecutionTimePredictor,
        SklearnExecutionTimePredictor,
    ):
        parameter = inspect.signature(
            predictor_type.predict_stage_execution_time
        ).parameters["include_attention"]
        assert parameter.default is True


def test_dense_predictor_rejects_post_attention_only_scope() -> None:
    predictor = _build_predictor()

    with pytest.raises(
        ValueError,
        match="Dense prediction does not support include_attention=False",
    ):
        predictor.predict_stage_execution_time(
            batch=_DummyBatch(),
            stage_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            include_attention=False,
        )


def test_dense_predict_stage_execution_time_scales_linearly(monkeypatch):
    predictor = _build_predictor()
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    batch = _DummyBatch()

    exec_1 = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
    )
    exec_5 = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=5,
    )

    assert exec_5.model_time_ms == pytest.approx(exec_1.model_time_ms * 5)
    assert exec_5.model_time_ms != pytest.approx(exec_1.model_time_ms * 25)

    assert exec_5.global_layer_ids == (0, 1, 2, 3, 4)
    for layer in exec_5.layer_execution_times:
        assert layer.get_single_layer_attention_time() == pytest.approx(
            exec_1.get_single_layer_attention_time()
        )
        assert layer.get_single_layer_block_time() == pytest.approx(
            exec_1.get_single_layer_block_time()
        )
    with pytest.raises(ValueError, match="exactly one layer"):
        exec_5.get_single_layer_block_time()
    assert exec_1.attention_operator_times is None
    assert exec_5.attention_operator_times is None


def test_homogeneous_stage_snapshots_numerical_components_once(monkeypatch):
    from frontier.entities import ExecutionTime

    predictor = _DummySklearnPredictor()
    snapshots = []
    original = ExecutionTime._clone_mutable_components

    def record_snapshot(timing):
        snapshots.append(timing)
        return original(timing)

    monkeypatch.setattr(ExecutionTime, "_clone_mutable_components", record_snapshot)
    stage = predictor.predict_stage_execution_time(
        _DummyBatch(), stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        num_layers=5, layer_id=3,
    )
    assert len(snapshots) == 1
    assert stage.global_layer_ids == (3, 4, 5, 6, 7)
    assert len({id(layer) for layer in stage.layer_execution_times}) == 5
    assert len({id(layer._attention_time) for layer in stage.layer_execution_times}) == 1


@pytest.mark.parametrize("disaggregated", [False, True])
def test_homogeneous_stage_computes_block_once_and_keeps_range_validation(
    monkeypatch, disaggregated,
):
    from frontier.entities import Batch, ExecutionTime, Request

    predictor = (
        CacheFixtureDisaggregationPredictor(model_config=replace(cache_model(), is_moe=False))
        if disaggregated else _DummySklearnPredictor()
    )
    role = ClusterType.PREFILL if disaggregated else ClusterType.MONOLITHIC
    batch = Batch(
        replica_id=0,
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], is_moe=False,
    )
    calls = []
    original = ExecutionTime.get_single_layer_block_time

    def record_block(timing):
        calls.append(timing.global_layer_id)
        return original(timing)

    monkeypatch.setattr(ExecutionTime, "get_single_layer_block_time", record_block)
    stage = predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=role, num_layers=5, layer_id=3,
    )
    assert stage.global_layer_ids == (3, 4, 5, 6, 7)
    assert stage.model_time_ms > 0.0
    assert stage.model_time_ms == stage.model_time_ms
    assert calls == [3]

    with pytest.raises(ValueError, match="out of range"):
        predictor.predict_stage_execution_time(
            batch, stage_id=0, cluster_type=role, num_layers=2,
            layer_id=predictor._model_config.num_layers - 1,
        )
    assert calls == [3]


def test_shared_numerical_source_keeps_different_model_owned_attention_identities():
    predictor = CacheFixturePredictor(hybrid=True)
    source = predictor._get_dummy_execution_time(_DummyBatch(), 0).finalized_copy()
    stage = predictor._assemble_stage([source, source], first_layer_id=2)
    specs = [predictor._model_config.get_layer_attention_spec(layer) for layer in (2, 3)]
    assert specs[0].family_id != specs[1].family_id
    assert stage.global_layer_ids == (2, 3)
    assert stage.attention_family_ids == tuple(spec.family_id for spec in specs)
    assert stage.attention_variant_ids == tuple(spec.variant_id for spec in specs)

@pytest.mark.parametrize("role", [
    ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_ATTN,
    ClusterType.DECODE_FFN,
])
def test_disaggregated_dense_stage_reuses_one_snapshot_with_exact_oracle(monkeypatch, role):
    from frontier.entities import Batch, ExecutionTime, Request

    predictor = CacheFixtureDisaggregationPredictor(
        cluster_type=role, model_config=replace(cache_model(), is_moe=False),
    )
    batch = Batch(
        replica_id=0,
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], is_moe=False,
    )
    first_layer, num_layers = 3, 5
    expected = predictor._assemble_stage([
        predictor._predict_disaggregated_layer_execution_time(
            batch, 0, role, 1, first_layer + offset,
            include_stage_owned=offset == 0,
        )
        for offset in range(num_layers)
    ], first_layer_id=first_layer)
    calls = []
    snapshots = []
    original_predict = predictor._predict_disaggregated_layer_execution_time
    original_snapshot = ExecutionTime._clone_mutable_components

    def record_prediction(*args, **kwargs):
        calls.append((args, kwargs))
        return original_predict(*args, **kwargs)

    def record_snapshot(timing):
        snapshots.append(timing)
        return original_snapshot(timing)

    monkeypatch.setattr(predictor, "_predict_disaggregated_layer_execution_time", record_prediction)
    monkeypatch.setattr(ExecutionTime, "_clone_mutable_components", record_snapshot)
    actual = predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=role, num_layers=num_layers,
        layer_id=first_layer,
    )
    assert len(calls) == len(snapshots) == 1
    assert actual.global_layer_ids == expected.global_layer_ids == (3, 4, 5, 6, 7)
    assert actual.model_time_ms == expected.model_time_ms
    assert actual.total_time == expected.total_time
    assert actual.op_times == expected.op_times
    assert actual.communication_operator_times == expected.communication_operator_times
    assert [layer.get_single_layer_block_time() for layer in actual.layer_execution_times] == [
        layer.get_single_layer_block_time() for layer in expected.layer_execution_times
    ]


def test_dense_predict_stage_execution_time_quantization_uses_role_names(
    monkeypatch,
) -> None:
    predictor = _build_predictor()
    quant_manager = _RecordingQuantizationManager()
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: quant_manager,
    )
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_enabled_predictor_metric_name_by_role",
        lambda _family, role: {
            "cache_write": "runtime_cache",
            "prefill_kernel": "runtime_prefill",
            "decode_kernel": "runtime_decode",
        }[role.value],
    )

    batch = _DummyBatch()

    predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
    )

    assert "runtime_cache" in quant_manager.adjusted_compute_ops
    assert "runtime_prefill" in quant_manager.adjusted_compute_ops
    assert "runtime_decode" in quant_manager.adjusted_compute_ops
    assert "attn_kv_cache_save" not in quant_manager.adjusted_compute_ops
    assert "attn_prefill" not in quant_manager.adjusted_compute_ops
    assert "attn_decode" not in quant_manager.adjusted_compute_ops
