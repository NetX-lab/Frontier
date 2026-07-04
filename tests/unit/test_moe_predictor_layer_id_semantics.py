from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from frontier.attention.ops import AttentionOperatorRole
from frontier.execution_time_predictor import sklearn_moe_execution_time_predictor
from frontier.entities.execution_time import ExecutionTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


class _DummySklearnMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DummyBatch:
    def __init__(self) -> None:
        self.id = 1
        self.size = 1
        self.num_tokens = 16
        self.total_num_tokens = 16
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 16
        self.requests = []
        self.is_idle = False

    def get_effective_total_tokens_rounded(self, _cluster_type) -> int:
        return int(self.total_num_tokens)


class _DummyModelConfig:
    def __init__(
        self,
        architecture_profile: ModelArchitectureProfile,
        moe_layer_ids: set[int] | None = None,
    ) -> None:
        self._architecture_profile = architecture_profile
        self._moe_layer_ids = moe_layer_ids
        self.embedding_dim = 7168
        self.num_q_heads = 32
        self.num_kv_heads = 8
        self.share_expert_dim = (
            1 if architecture_profile.always_supports_share_expert else None
        )
        self.share_q_dim = 512
        self.is_moe = True

    def is_step3_text(self) -> bool:
        return self._architecture_profile.step3_text_compatible

    def is_step2_mini(self) -> bool:
        return self._architecture_profile.step2_mini_compatible

    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return self._architecture_profile

    def get_head_dim(self) -> int:
        return self.embedding_dim // self.num_q_heads

    def is_moe_layer(self, layer_id: int) -> bool:
        if self._moe_layer_ids is None:
            return True
        return layer_id in self._moe_layer_ids

    def supports_share_expert(self) -> bool:
        return self._architecture_profile.supports_share_expert(self)


class _DummyReplicaConfig:
    def __init__(self, model_config: _DummyModelConfig | None) -> None:
        self.model_config = model_config


def _build_base_execution_time() -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=61,
        attention_rope_execution_time=1.0,
        attention_kv_cache_save_execution_time=1.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=2.0,
        attention_layer_pre_proj_execution_time=3.0,
        attention_layer_post_proj_execution_time=4.0,
        attn_norm_time=1.0,
        mlp_norm_time=1.0,
        add_time=0.0,
        add_attn_residual_time=0.5,
        add_ffn_residual_time=0.5,
        tensor_parallel_communication_time=0.0,
        attn_tensor_parallel_allreduce_time=0.3,
        moe_tensor_parallel_allreduce_time=0.7,
        tensor_parallel_allgather_time=0.2,
        share_expert_tensor_parallel_allreduce_time=0.1,
        dp_input_allreduce_time=0.4,
        dp_output_allreduce_time=0.6,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=2.0,
        moe_gating_time=0.9,
        moe_gating_linear_time=0.4,
        moe_gating_routing_topk_time=0.5,
        moe_shuffling_time=1.5,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=True,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        moe_grouped_gemm_time=5.0,
        share_expert_up_proj_time=1.0,
        share_expert_down_proj_time=1.0,
        share_expert_act_time=1.0,
    )


def _build_dense_execution_time() -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=61,
        attention_rope_execution_time=1.0,
        attention_kv_cache_save_execution_time=1.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=2.0,
        attention_layer_pre_proj_execution_time=3.0,
        attention_layer_post_proj_execution_time=4.0,
        attn_norm_time=1.0,
        mlp_norm_time=0.8,
        add_time=0.0,
        add_attn_residual_time=0.5,
        add_ffn_residual_time=0.5,
        tensor_parallel_communication_time=0.0,
        attn_tensor_parallel_allreduce_time=0.3,
        moe_tensor_parallel_allreduce_time=0.3,
        tensor_parallel_allgather_time=0.0,
        share_expert_tensor_parallel_allreduce_time=0.0,
        dp_input_allreduce_time=0.0,
        dp_output_allreduce_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_gating_linear_time=0.0,
        moe_gating_routing_topk_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        mlp_layer_up_proj_execution_time=1.1,
        mlp_layer_down_proj_execution_time=1.2,
        mlp_layer_act_execution_time=0.7,
        moe_grouped_gemm_time=0.0,
        share_expert_up_proj_time=0.0,
        share_expert_down_proj_time=0.0,
        share_expert_act_time=0.0,
    )


def _build_predictor() -> _DummySklearnMoEPredictor:
    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 61
    predictor._moe_routing_mode = "simulation"
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.generic())
    predictor._predictions_eager = {"loaded": True}
    predictor._predictions_kernel_only = {"loaded": True}
    predictor._supports_operation = lambda _operation: True
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    predictor._log_architecture_attention_shape = lambda _batch: None
    predictor._require_predictions_for_measurement_type = (
        lambda _measurement_type, _batch: None
    )
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._get_moe_tokens_input = MagicMock(return_value={0: 16})
    predictor._get_execution_time_internal = MagicMock(
        return_value=_build_base_execution_time()
    )
    return predictor


def test_predict_stage_execution_time_forwards_layer_id_to_moe_tokens_input() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(
        ModelArchitectureProfile.generic(),
        moe_layer_ids={17},
    )
    batch = _DummyBatch()

    predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=17,
    )

    predictor._get_moe_tokens_input.assert_called_once_with(batch, layer_id=17)
    assert predictor._get_execution_time_internal.call_args.kwargs["include_moe"] is True


def test_predict_stage_execution_time_skips_moe_tokens_for_dense_layer() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(
        ModelArchitectureProfile.generic(),
        moe_layer_ids={4, 5, 6},
    )

    dense_base = _build_dense_execution_time()
    moe_base = _build_base_execution_time()

    def _fake_get_execution_time_internal(*_args, **kwargs):
        return moe_base if kwargs["include_moe"] else dense_base

    predictor._get_execution_time_internal = MagicMock(
        side_effect=_fake_get_execution_time_internal
    )

    batch = _DummyBatch()
    result = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    predictor._get_moe_tokens_input.assert_not_called()
    call_kwargs = predictor._get_execution_time_internal.call_args.kwargs
    assert call_kwargs["include_moe"] is False
    assert call_kwargs["moe_tokens_input"] is None
    assert result._is_moe is False
    assert result._moe_grouped_gemm_time == pytest.approx(0.0)
    assert result._mlp_layer_up_proj_execution_time == pytest.approx(1.1)


def test_predict_stage_execution_time_keeps_per_layer_components_and_scales_linearly() -> None:
    predictor = _build_predictor()
    batch = _DummyBatch()

    exec_1 = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=3,
    )
    exec_5 = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=5,
        layer_id=3,
    )

    assert exec_1.get_single_layer_attention_time() == pytest.approx(12.0)
    assert exec_5.get_single_layer_attention_time() == pytest.approx(12.0)
    assert exec_5.model_time_ms == pytest.approx(exec_1.model_time_ms * 5)


def test_moe_predictor_attention_op_trace_labels_use_dense_role_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _build_predictor()
    batch = _DummyBatch()
    messages: list[str] = []

    monkeypatch.setattr(
        sklearn_moe_execution_time_predictor,
        "get_enabled_predictor_metric_name_by_role",
        lambda _family, role: {
            AttentionOperatorRole.CACHE_WRITE: "role_cache",
            AttentionOperatorRole.PREFILL_KERNEL: "role_prefill",
        }[role],
        raising=False,
    )
    monkeypatch.setattr(
        sklearn_moe_execution_time_predictor.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message
        ),
    )

    predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=3,
    )

    log_text = "\n".join(messages)
    assert "[ATTENTION][role_prefill]" in log_text
    assert "[ATTENTION][role_cache]" in log_text
    assert "[ATTENTION][attn_prefill]" not in log_text
    assert "[ATTENTION][attn_kv_cache_save]" not in log_text


def test_monolithic_decode_shared_domain_lane_moe_times_respects_dummy_mode() -> None:
    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.25
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.generic())

    # These methods are prediction-cache dependent in non-dummy mode.
    # The dummy-mode fast path must bypass them completely.
    predictor._get_mlp_norm_layer_act_execution_time = MagicMock(
        side_effect=AssertionError("dummy mode path should not read post_attention_layernorm cache")
    )
    predictor._get_gating_linear_time = MagicMock(
        side_effect=AssertionError("dummy mode path should not read gating_linear cache")
    )
    predictor._get_gating_routing_topk_time = MagicMock(
        side_effect=AssertionError("dummy mode path should not read gating_topk cache")
    )
    predictor._get_moe_shuffling_time = MagicMock(
        side_effect=AssertionError("dummy mode path should not read shuffling cache")
    )
    predictor._get_grouped_gemm_time = MagicMock(
        side_effect=AssertionError("dummy mode path should not read grouped_gemm cache")
    )

    batch = _DummyBatch()
    lane_times = predictor.predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        batch=batch,
        layer_id=7,
    )

    expected_lane_time = 1.25 * 5.0
    assert lane_times == {
        0: pytest.approx(expected_lane_time),
        1: pytest.approx(expected_lane_time),
    }


def test_share_expert_overlap_scaling_applies_for_step3_model() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.step3_text())

    assert predictor._apply_share_expert_tp_allreduce_overlap(3.0) == pytest.approx(2.0)


def test_share_expert_overlap_scaling_skips_non_step3_model() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.generic())

    assert predictor._apply_share_expert_tp_allreduce_overlap(3.0) == pytest.approx(3.0)


def test_share_expert_overlap_scaling_uses_replica_model_config_when_needed() -> None:
    predictor = _build_predictor()
    predictor._model_config = None
    predictor._replica_config = _DummyReplicaConfig(
        model_config=_DummyModelConfig(ModelArchitectureProfile.step3_text())
    )

    assert predictor._apply_share_expert_tp_allreduce_overlap(3.0) == pytest.approx(2.0)


def test_share_expert_overlap_scaling_handles_non_positive_time() -> None:
    predictor = _build_predictor()

    assert predictor._apply_share_expert_tp_allreduce_overlap(0.0) == pytest.approx(0.0)
    assert predictor._apply_share_expert_tp_allreduce_overlap(-1.0) == pytest.approx(0.0)


def test_share_expert_overlap_scaling_uses_configured_visibility_scale() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.step3_text())
    predictor._share_expert_tp_allreduce_visibility_scale = 0.5

    assert predictor._apply_share_expert_tp_allreduce_overlap(3.0) == pytest.approx(1.5)


def test_step3_prefill_allgather_uses_per_device_bytes_in_moe_predictor() -> None:
    class _DummyReplicaConfigForAllgather:
        num_pipeline_stages = 1
        attn_tensor_parallel_size = 1
        moe_tensor_parallel_size = 8

    class _IdentityQuantManager:
        @staticmethod
        def adjust_tensor_size(_op_name: str, size_bytes: int, _cluster_type) -> int:
            return size_bytes

    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = _DummyReplicaConfigForAllgather()
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.step3_text(), moe_layer_ids={0})
    predictor._num_layers_per_pipeline_stage = 1
    predictor._moe_ep_size = 1
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._supports_operation = lambda _operation: True
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    predictor._log_architecture_attention_shape = lambda _batch: None

    predictor._get_pipeline_parallel_communication_time = lambda _batch: 0.0
    predictor._get_tensor_parallel_communication_time = lambda _batch: 0.0
    predictor._get_moe_tensor_parallel_allreduce_time = lambda _batch: 0.0
    predictor._get_expert_parallel_communication_time = lambda _batch: 0.0
    predictor._get_gating_linear_time = lambda _batch: 0.0
    predictor._get_gating_routing_topk_time = lambda _batch: 0.0
    predictor._get_moe_shuffling_time = lambda _batch, moe_tokens_input: 0.0
    predictor._get_grouped_gemm_time = lambda _moe_tokens_input, **_kwargs: 0.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_attention_rope_execution_time = lambda _batch: 0.0
    predictor._get_attention_kv_cache_save_execution_time = lambda _batch: 0.0
    predictor._get_attention_decode_execution_time = lambda _batch: 0.0
    predictor._get_attention_prefill_execution_time = lambda _batch: 0.0
    predictor._get_attention_layer_pre_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_layer_post_proj_execution_time = lambda _batch: 0.0
    predictor._get_attn_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_share_expert_up_proj_execution_time = lambda _batch: 0.0
    predictor._get_share_expert_down_proj_execution_time = lambda _batch: 0.0
    predictor._get_share_expert_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor.predict_dp_moe_allreduce_times = lambda _batch, _cluster_type: (0.0, 0.0)
    predictor.predict_allgather_time = MagicMock(return_value=0.0)
    predictor.predict_allreduce_time = MagicMock(return_value=0.0)

    batch = _DummyBatch()
    with patch(
        "frontier.execution_time_predictor.sklearn_moe_execution_time_predictor.get_quantization_manager",
        return_value=_IdentityQuantManager(),
    ):
        predictor._get_execution_time_internal(
            batch=batch,
            pipeline_stage=0,
            moe_tokens_input=16,
            include_moe=True,
        )

    expected_total_bytes = predictor._model_config.embedding_dim * 2 * batch.total_num_tokens
    expected_per_device_bytes = expected_total_bytes // predictor._replica_config.moe_tensor_parallel_size
    predictor.predict_allgather_time.assert_called_once_with(
        data_size_bytes=expected_per_device_bytes,
        num_devices=predictor._replica_config.moe_tensor_parallel_size,
        cluster_type=ClusterType.MONOLITHIC,
        comm_domain="MOE_TP",
    )
