from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


class _DummySklearnPredictor(SklearnExecutionTimePredictor):
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
    predictor = _DummySklearnPredictor.__new__(_DummySklearnPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
    )
    predictor._model_config = SimpleNamespace(
        use_mla=False,
        num_q_heads=32,
        num_kv_heads=8,
        get_model_architecture_profile=ModelArchitectureProfile.generic,
    )

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


def test_dense_predict_stage_execution_time_scales_linearly(monkeypatch):
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_predictor()
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

    assert exec_5.get_single_layer_attention_time() == pytest.approx(
        exec_1.get_single_layer_attention_time()
    )
    assert exec_5.get_single_layer_block_time() == pytest.approx(
        exec_1.get_single_layer_block_time()
    )
    assert exec_1.attention_operator_times is None
    assert exec_5.attention_operator_times is None


def test_dense_predict_stage_execution_time_quantization_uses_role_names(
    monkeypatch,
) -> None:
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

    predictor = _build_predictor()
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
