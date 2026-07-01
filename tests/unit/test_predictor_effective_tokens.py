"""
Unit tests for compute prediction token selection in execution time predictors.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from frontier.types import ClusterType, MeasurementType
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)


class DummySklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    """Minimal concrete class for unit testing abstract predictor methods."""

    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class DummySklearnMoEExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    """Minimal concrete class for unit testing abstract MoE predictor methods."""

    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def test_sklearn_execution_time_predictor_uses_effective_tokens():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.DECODE_ATTN
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_pre_proj": {(13,): 1.0}}

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=13)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_attention_layer_pre_proj_execution_time(
        predictor, batch
    )

    assert result == 1.0
    batch.get_effective_total_tokens_rounded.assert_called_once_with(
        ClusterType.DECODE_ATTN
    )


def test_sklearn_execution_time_predictor_uses_non_multiple_token_key_for_post_attn_norm():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._model_config = SimpleNamespace(post_attn_norm=True)
    predictor._predictions = {"post_attention_layernorm": {(1033,): 9.0}}

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=1033)

    result = SklearnExecutionTimePredictor._get_mlp_norm_layer_act_execution_time(
        predictor, batch
    )

    assert result == 9.0
    batch.get_effective_total_tokens_rounded.assert_called_once_with(
        ClusterType.MONOLITHIC
    )


def test_sklearn_moe_predictor_uses_effective_tokens_for_gating():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.DECODE_ATTN
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._model_config = SimpleNamespace()
    predictor._predictions = {"moe_gating_linear": {(32,): 2.0}}

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)

    result = SklearnMoEExecutionTimePredictor._get_gating_linear_time(
        predictor, batch
    )

    assert result == 2.0
    batch.get_effective_total_tokens_rounded.assert_called_once_with(
        ClusterType.DECODE_ATTN
    )


def test_sklearn_moe_predictor_uses_effective_tokens_for_local_ep_routed_tokens():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.DECODE
    predictor._router_topk = 2
    predictor._moe_ep_size = 4
    predictor._use_expert_parallel_alltoall_path = MagicMock(return_value=False)

    batch = SimpleNamespace(
        total_num_tokens=1,
        is_pure_decode_batch=True,
        get_effective_total_tokens_rounded=lambda _cluster_type: 8,
    )

    result = SklearnMoEExecutionTimePredictor._get_local_ep_routed_tokens(
        predictor, batch
    )

    assert result == 4


def test_sklearn_moe_predictor_uses_effective_tokens_for_uniform_legacy_tokens_input():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.DECODE
    predictor._router_topk = 2
    predictor._moe_routing_mode = "uniform_legacy"
    predictor._is_grouped_gemm_on_demand_mode = MagicMock(return_value=False)
    predictor._get_local_ep_routed_tokens = MagicMock(return_value=8)

    batch = SimpleNamespace(
        total_num_tokens=1,
        get_effective_total_tokens_rounded=lambda _cluster_type: 8,
    )

    result = SklearnMoEExecutionTimePredictor._get_moe_tokens_input(predictor, batch)

    assert result == 16


def test_sklearn_execution_time_predictor_applies_attn_pre_proj_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_pre_proj": {(16,): 2.0}}
    predictor._attn_pre_proj_calibration_scale = 0.5

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=16)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_attention_layer_pre_proj_execution_time(
        predictor, batch
    )

    assert result == 1.0


def test_sklearn_execution_time_predictor_applies_prefill_phase_attn_pre_proj_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_pre_proj": {(16,): 2.0}}
    predictor._attn_pre_proj_calibration_scale = 1.0
    predictor._prefill_phase_attn_pre_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=16)
    batch.num_prefill_tokens = 8

    result = SklearnExecutionTimePredictor._get_attention_layer_pre_proj_execution_time(
        predictor, batch
    )

    assert result == 0.8


def test_sklearn_execution_time_predictor_keeps_global_attn_pre_proj_scale_for_decode_only_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_pre_proj": {(16,): 2.0}}
    predictor._attn_pre_proj_calibration_scale = 0.5
    predictor._prefill_phase_attn_pre_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=16)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_attention_layer_pre_proj_execution_time(
        predictor, batch
    )

    assert result == 1.0


def test_sklearn_execution_time_predictor_applies_attn_post_proj_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_post_proj": {(8,): 3.0}}
    predictor._attn_post_proj_calibration_scale = 2.0

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=8)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_attention_layer_post_proj_execution_time(
        predictor, batch
    )

    assert result == 6.0


def test_sklearn_execution_time_predictor_applies_prefill_phase_attn_post_proj_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_post_proj": {(8,): 3.0}}
    predictor._attn_post_proj_calibration_scale = 1.0
    predictor._prefill_phase_attn_post_proj_calibration_scale = 0.6

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=8)
    batch.num_prefill_tokens = 4

    result = SklearnExecutionTimePredictor._get_attention_layer_post_proj_execution_time(
        predictor, batch
    )

    assert result == pytest.approx(1.8)


def test_sklearn_execution_time_predictor_keeps_global_attn_post_proj_scale_for_decode_only_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_post_proj": {(8,): 3.0}}
    predictor._attn_post_proj_calibration_scale = 2.0
    predictor._prefill_phase_attn_post_proj_calibration_scale = 0.6

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=8)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_attention_layer_post_proj_execution_time(
        predictor, batch
    )

    assert result == 6.0


def test_sklearn_execution_time_predictor_applies_attn_kv_cache_save_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_kv_cache_save": {(4,): 3.0}}
    predictor._attn_kv_cache_save_calibration_scale = 1.5

    batch = MagicMock()
    batch.total_num_tokens = 4
    batch.num_decode_tokens = 0
    batch.num_prefill_tokens = 0
    batch.requests = [object(), object()]

    result = SklearnExecutionTimePredictor._get_attention_kv_cache_save_execution_time(
        predictor, batch
    )

    assert result == 4.5


def test_sklearn_execution_time_predictor_applies_prefill_phase_attn_kv_cache_save_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_kv_cache_save": {(4,): 3.0}}
    predictor._attn_kv_cache_save_calibration_scale = 1.0
    predictor._prefill_phase_attn_kv_cache_save_calibration_scale = 1.8

    batch = MagicMock()
    batch.total_num_tokens = 4
    batch.num_decode_tokens = 0
    batch.num_prefill_tokens = 4
    batch.requests = [object(), object()]

    result = SklearnExecutionTimePredictor._get_attention_kv_cache_save_execution_time(
        predictor, batch
    )

    assert result == 5.4


def test_sklearn_execution_time_predictor_keeps_global_attn_kv_cache_save_scale_for_decode_only_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_kv_cache_save": {(4,): 3.0}}
    predictor._attn_kv_cache_save_calibration_scale = 1.5
    predictor._prefill_phase_attn_kv_cache_save_calibration_scale = 1.8

    batch = MagicMock()
    batch.total_num_tokens = 4
    batch.num_decode_tokens = 4
    batch.num_prefill_tokens = 0
    batch.requests = [object(), object()]

    result = SklearnExecutionTimePredictor._get_attention_kv_cache_save_execution_time(
        predictor, batch
    )

    assert result == 4.5


def test_sklearn_execution_time_predictor_applies_mlp_up_proj_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_up_proj": {(32,): 5.0}}
    predictor._mlp_up_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_mlp_layer_up_proj_execution_time(
        predictor, batch
    )

    assert result == 2.0


def test_sklearn_execution_time_predictor_applies_prefill_phase_mlp_up_proj_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_up_proj": {(32,): 5.0}}
    predictor._mlp_up_proj_calibration_scale = 1.0
    predictor._prefill_phase_mlp_up_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 16

    result = SklearnExecutionTimePredictor._get_mlp_layer_up_proj_execution_time(
        predictor, batch
    )

    assert result == 2.0


def test_sklearn_execution_time_predictor_keeps_global_mlp_up_proj_scale_for_decode_only_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_up_proj": {(32,): 5.0}}
    predictor._mlp_up_proj_calibration_scale = 0.8
    predictor._prefill_phase_mlp_up_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_mlp_layer_up_proj_execution_time(
        predictor, batch
    )

    assert result == 4.0


def test_sklearn_execution_time_predictor_applies_mixed_attn_decode_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._predictions = {"attn_decode_in_mixed": {"_on_demand_prediction": True}}
    predictor._attn_decode_in_mixed_calibration_scale = 4.5
    predictor._get_batch_decode_attention_params = MagicMock(return_value=(1, 128))
    predictor._get_batch_decode_mixed_features = MagicMock(
        return_value={"prefill_seq_len": 9, "decode_kv_cache_size": 128}
    )
    predictor._get_on_demand_prediction = MagicMock(return_value=2.0)

    batch = MagicMock()
    batch.num_prefill_tokens = 9

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 9.0
    predictor._get_on_demand_prediction.assert_called_once_with(
        "attn_decode_in_mixed",
        {"prefill_seq_len": 9, "decode_kv_cache_size": 128},
    )


def test_sklearn_execution_time_predictor_leaves_mixed_attn_decode_unscaled_by_default():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._predictions = {"attn_decode_in_mixed": {"_on_demand_prediction": True}}
    predictor._attn_decode_in_mixed_calibration_scale = None
    predictor._get_batch_decode_attention_params = MagicMock(return_value=(1, 128))
    predictor._get_batch_decode_mixed_features = MagicMock(return_value={})
    predictor._get_on_demand_prediction = MagicMock(return_value=2.0)

    batch = MagicMock()
    batch.num_prefill_tokens = 9

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 2.0


def test_sklearn_execution_time_predictor_applies_mlp_down_proj_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_down_proj": {(32,): 5.0}}
    predictor._mlp_down_proj_calibration_scale = 0.4

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)

    result = SklearnExecutionTimePredictor._get_mlp_layer_down_proj_execution_time(
        predictor, batch
    )

    assert result == 2.0


def test_sklearn_execution_time_predictor_applies_decode_phase_mlp_down_proj_scale_for_decode_only_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_down_proj": {(32,): 5.0}}
    predictor._mlp_down_proj_calibration_scale = 0.4
    predictor._decode_phase_mlp_down_proj_calibration_scale = 1.13

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 0

    result = SklearnExecutionTimePredictor._get_mlp_layer_down_proj_execution_time(
        predictor, batch
    )

    assert result == pytest.approx(5.65)


def test_sklearn_execution_time_predictor_keeps_global_mlp_down_proj_scale_for_mixed_batch():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"mlp_down_proj": {(32,): 5.0}}
    predictor._mlp_down_proj_calibration_scale = 0.4
    predictor._decode_phase_mlp_down_proj_calibration_scale = 1.13

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 8

    result = SklearnExecutionTimePredictor._get_mlp_layer_down_proj_execution_time(
        predictor, batch
    )

    assert result == 2.0

def test_sklearn_moe_predictor_uses_on_demand_moe_shuffling_prediction() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_shuffling": {"_on_demand_prediction": True}}
    predictor._router_topk = 8
    predictor._moe_ep_size = 1
    predictor._replica_config = SimpleNamespace(total_expert_num=16)
    predictor._model_config = SimpleNamespace(embedding_dim=2048, mlp_hidden_dim=768)
    predictor._moe_shuffling_calibration_scale = 1.0
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT

    captured = {}

    def _fake_get_on_demand_prediction(model_name, features):
        captured["model_name"] = model_name
        captured["features"] = features
        return 0.75

    predictor._get_on_demand_prediction = _fake_get_on_demand_prediction

    batch = MagicMock()
    batch.total_num_tokens = 12

    result = SklearnMoEExecutionTimePredictor._get_moe_shuffling_time(
        predictor,
        batch,
        moe_tokens_input={0: 64, 1: 32},
    )

    assert result == 0.75
    assert captured["model_name"] == "moe_shuffling"
    assert captured["features"]["total_routed_tokens"] == 96
    assert "load_imbalance_cv" in captured["features"]


def test_sklearn_moe_predictor_builds_uniform_allocation_for_on_demand_shuffling() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_shuffling": {"_on_demand_prediction": True}}
    predictor._router_topk = 8
    predictor._moe_ep_size = 1
    predictor._replica_config = SimpleNamespace(total_expert_num=4)
    predictor._model_config = SimpleNamespace(embedding_dim=2048, mlp_hidden_dim=768)
    predictor._moe_shuffling_calibration_scale = 2.0
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT

    captured = {}

    def _fake_get_on_demand_prediction(model_name, features):
        captured["model_name"] = model_name
        captured["features"] = features
        return 0.5

    predictor._get_on_demand_prediction = _fake_get_on_demand_prediction

    batch = MagicMock()
    batch.total_num_tokens = 12

    result = SklearnMoEExecutionTimePredictor._get_moe_shuffling_time(
        predictor,
        batch,
        moe_tokens_input=80,
    )

    assert result == 1.0
    assert captured["model_name"] == "moe_shuffling"
    assert captured["features"]["total_routed_tokens"] == 80
    assert captured["features"]["num_experts_per_device"] == 4


def test_sklearn_moe_predictor_applies_moe_shuffling_calibration_scale():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_shuffling": {(32,): 4.0}}
    predictor._moe_shuffling_calibration_scale = 0.25

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)

    result = SklearnMoEExecutionTimePredictor._get_moe_shuffling_time(predictor, batch)

    assert result == 1.0


def test_sklearn_moe_predictor_applies_decode_phase_moe_shuffling_scale_for_decode_only_batch():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_shuffling": {(32,): 4.0}}
    predictor._moe_shuffling_calibration_scale = 0.25
    predictor._decode_phase_moe_shuffling_calibration_scale = 1.5

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 0

    result = SklearnMoEExecutionTimePredictor._get_moe_shuffling_time(predictor, batch)

    assert result == 6.0


def test_sklearn_moe_predictor_keeps_global_moe_shuffling_scale_for_mixed_batch():
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_shuffling": {(32,): 4.0}}
    predictor._moe_shuffling_calibration_scale = 0.25
    predictor._decode_phase_moe_shuffling_calibration_scale = 1.5

    batch = MagicMock()
    batch.get_effective_total_tokens_rounded = MagicMock(return_value=32)
    batch.num_prefill_tokens = 8

    result = SklearnMoEExecutionTimePredictor._get_moe_shuffling_time(predictor, batch)

    assert result == 1.0


def test_sklearn_execution_time_predictor_uses_padded_decode_batch_size_for_attn_decode():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_decode": {(8, 1024): 5.0}}
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)

    requests = [
        SimpleNamespace(_is_prefill_complete=True, num_processed_tokens=1000)
        for _ in range(5)
    ]
    batch = SimpleNamespace(
        requests=requests,
        num_prefill_tokens=0,
        num_decode_tokens=5,
        get_effective_decode_batch_size_for_attention=lambda: 8,
        decode_cuda_graph_metadata=SimpleNamespace(
            padded_decode_batch_size=8,
        ),
    )

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 5.0


def test_sklearn_execution_time_predictor_applies_attn_decode_calibration_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_decode": {(8, 1024): 5.0}}
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attn_decode_calibration_scale = 1.6
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)

    requests = [
        SimpleNamespace(_is_prefill_complete=True, num_processed_tokens=1000)
        for _ in range(5)
    ]
    batch = SimpleNamespace(
        requests=requests,
        num_prefill_tokens=0,
        num_decode_tokens=5,
        get_effective_decode_batch_size_for_attention=lambda: 8,
        decode_cuda_graph_metadata=SimpleNamespace(
            padded_decode_batch_size=8,
        ),
    )

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 8.0


def test_sklearn_execution_time_predictor_applies_late_decode_only_attn_decode_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_decode": {(8, 1024): 5.0}}
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attn_decode_calibration_scale = 1.0
    predictor._late_decode_attn_decode_calibration_scale = 1.2
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)

    requests = [
        SimpleNamespace(
            _is_prefill_complete=True,
            num_processed_decode_tokens=2,
            num_processed_tokens=1000,
        )
        for _ in range(5)
    ]
    batch = SimpleNamespace(
        requests=requests,
        num_prefill_tokens=0,
        num_decode_tokens=5,
        get_effective_decode_batch_size_for_attention=lambda: 8,
        decode_cuda_graph_metadata=SimpleNamespace(
            padded_decode_batch_size=8,
        ),
    )

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 6.0


def test_sklearn_execution_time_predictor_keeps_first_pure_decode_on_global_scale():
    predictor = DummySklearnExecutionTimePredictor.__new__(
        DummySklearnExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"attn_decode": {(8, 1024): 5.0}}
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attn_decode_calibration_scale = 1.0
    predictor._late_decode_attn_decode_calibration_scale = 1.2
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)

    requests = [
        SimpleNamespace(
            _is_prefill_complete=True,
            num_processed_decode_tokens=1,
            num_processed_tokens=1000,
        )
        for _ in range(5)
    ]
    batch = SimpleNamespace(
        requests=requests,
        num_prefill_tokens=0,
        num_decode_tokens=5,
        get_effective_decode_batch_size_for_attention=lambda: 8,
        decode_cuda_graph_metadata=SimpleNamespace(
            padded_decode_batch_size=8,
        ),
    )

    result = SklearnExecutionTimePredictor._get_attention_decode_execution_time(
        predictor, batch
    )

    assert result == 5.0


def test_sklearn_moe_predictor_applies_moe_grouped_gemm_calibration_scale() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_grouped_gemm": {(16,): 4.0}}
    predictor._moe_grouped_gemm_calibration_scale = 1.75
    predictor._max_tokens = 16

    result = SklearnMoEExecutionTimePredictor._get_grouped_gemm_time(
        predictor, 16
    )

    assert result == 7.0


def test_sklearn_moe_predictor_applies_decode_phase_moe_grouped_gemm_scale_for_decode_only_batch() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_grouped_gemm": {(16,): 4.0}}
    predictor._moe_grouped_gemm_calibration_scale = 1.75
    predictor._decode_phase_moe_grouped_gemm_calibration_scale = 1.5
    predictor._max_tokens = 16

    batch = MagicMock()
    batch.num_prefill_tokens = 0

    result = SklearnMoEExecutionTimePredictor._get_grouped_gemm_time(
        predictor,
        16,
        batch=batch,
    )

    assert result == 6.0


def test_sklearn_moe_predictor_keeps_global_moe_grouped_gemm_scale_for_mixed_batch() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._predictions = {"moe_grouped_gemm": {(16,): 4.0}}
    predictor._moe_grouped_gemm_calibration_scale = 1.75
    predictor._decode_phase_moe_grouped_gemm_calibration_scale = 1.5
    predictor._max_tokens = 16

    batch = MagicMock()
    batch.num_prefill_tokens = 4

    result = SklearnMoEExecutionTimePredictor._get_grouped_gemm_time(
        predictor,
        16,
        batch=batch,
    )

    assert result == 7.0


def test_sklearn_moe_predictor_uses_effective_tokens_for_shuffling_allocation() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._cluster_type = ClusterType.DECODE
    predictor._router_topk = 2
    predictor._build_uniform_per_expert_tokens = MagicMock(
        return_value={0: 8, 1: 8}
    )

    batch = SimpleNamespace(
        total_num_tokens=1,
        get_effective_total_tokens_rounded=lambda _cluster_type: 8,
    )

    result = SklearnMoEExecutionTimePredictor._resolve_shuffling_per_expert_tokens(
        predictor,
        batch,
    )

    assert result == {0: 8, 1: 8}
    predictor._build_uniform_per_expert_tokens.assert_called_once_with(16)


def test_sklearn_moe_predictor_accepts_effective_token_conservation_for_padded_decode_batch() -> None:
    predictor = DummySklearnMoEExecutionTimePredictor.__new__(
        DummySklearnMoEExecutionTimePredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.DECODE
    predictor._router_topk = 2
    predictor._moe_ep_size = 2
    predictor._moe_tp_size = 1
    predictor._supports_operation = MagicMock(return_value=True)
    predictor._model_config = SimpleNamespace(
        post_attn_norm=True,
        supports_share_expert=lambda: False,
    )
    predictor._get_gating_linear_time = MagicMock(return_value=1.0)
    predictor._get_gating_routing_topk_time = MagicMock(return_value=2.0)
    predictor._get_moe_shuffling_time = MagicMock(return_value=3.0)
    predictor._get_mlp_norm_layer_act_execution_time = MagicMock(return_value=4.0)
    predictor._get_grouped_gemm_time = MagicMock(return_value=5.0)

    batch = SimpleNamespace(
        id="batch-0",
        total_num_tokens=1,
        requests=[],
        get_effective_total_tokens_rounded=lambda _cluster_type: 8,
    )

    result = SklearnMoEExecutionTimePredictor.predict_moe_layer_time(
        predictor,
        batch,
        layer_id=0,
        cluster_type=ClusterType.DECODE,
        per_expert_tokens={0: 8, 1: 8},
    )

    assert result.moe_grouped_gemm_time == 5.0
