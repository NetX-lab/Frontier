from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from frontier.attention.ops import AttentionOperatorRole
from frontier.execution_time_predictor import sklearn_moe_execution_time_predictor
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
    WorkloadDistributionType,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.moe_ep_workload import EPLaneWorkload, materialize_layer_ep_workload
from frontier.types import ClusterType


class _DummySklearnMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DummyDisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def test_attention_dp_moe_communication_is_rejected_when_not_one() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._replica_config = SimpleNamespace(attn_dp=2)

    with pytest.raises(ValueError, match="attention-DP communication is retired"):
        predictor.predict_dp_moe_allreduce_times(
            batch=SimpleNamespace(),
            cluster_type=ClusterType.PREFILL,
        )


class _DummyBatch:
    def __init__(self, lane_workload: EPLaneWorkload | None = None) -> None:
        self.id = 1
        self.size = 1
        self.num_tokens = 16
        self.total_num_tokens = 16
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 16
        self.requests = []
        self.is_idle = False
        if lane_workload is not None:
            self.lane_workload = lane_workload

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
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._num_layers_per_pipeline_stage = 61
    predictor._moe_routing_distribution_type = "random"
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
    predictor._get_moe_tokens_input = MagicMock(
        side_effect=lambda batch, layer_id: batch.lane_workload
    )
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
    lane_workload = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(0, 1),
        local_token_counts=(8, 8),
        routed_token_count=16,
        router_topk=2,
    )
    batch = _DummyBatch(lane_workload=lane_workload)

    predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=17,
    )

    predictor._get_moe_tokens_input.assert_called_once_with(batch, layer_id=17)
    internal_call = predictor._get_execution_time_internal.call_args.kwargs
    assert internal_call["include_moe"] is True
    assert internal_call["layer_id"] == 17


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


def test_mixed_dense_layer_rejects_post_attention_scope() -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(
        ModelArchitectureProfile.generic(),
        moe_layer_ids={4, 5, 6},
    )

    with pytest.raises(ValueError, match="MoE layer"):
        predictor.predict_stage_execution_time(
            _DummyBatch(),
            stage_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            num_layers=1,
            layer_id=1,
            include_attention=False,
        )


def _build_dummy_mixed_layer_predictor(
    model_config: _DummyModelConfig,
) -> _DummySklearnMoEPredictor:
    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.25
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 1
    predictor._moe_ep_size = 1
    predictor._router_topk = 2
    predictor._model_config = model_config
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
    )
    predictor._log_architecture_attention_shape = lambda _batch: None
    return predictor


def test_dummy_monolithic_mixed_dense_layer_uses_dense_components() -> None:
    predictor = _build_dummy_mixed_layer_predictor(
        _DummyModelConfig(
            ModelArchitectureProfile.generic(),
            moe_layer_ids={4, 5, 6},
        )
    )

    execution_time = predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    assert execution_time._is_moe is False
    assert execution_time._moe_gating_time == pytest.approx(0.0)
    assert execution_time._moe_shuffling_time == pytest.approx(0.0)
    assert execution_time._moe_grouped_gemm_time == pytest.approx(0.0)
    assert execution_time._expert_parallel_communication_time == pytest.approx(0.0)
    assert execution_time._mlp_layer_up_proj_execution_time > 0.0
    assert execution_time._mlp_layer_down_proj_execution_time > 0.0
    assert execution_time._mlp_layer_act_execution_time > 0.0
    assert execution_time.moe_operator_times is None


def test_dummy_monolithic_mixed_dense_layer_preserves_ffn_tp_allreduce() -> None:
    """Dense mixed layers retain the shared FFN TP communication contract."""

    predictor = _build_dummy_mixed_layer_predictor(
        _DummyModelConfig(
            ModelArchitectureProfile.generic(),
            moe_layer_ids={4, 5, 6},
        )
    )
    # MONOLITHIC dense FFN shares the attention-TP domain.  The MoE-TP field
    # belongs to routed expert layers and must not trigger this dense contract.
    predictor._replica_config.attn_tensor_parallel_size = 2

    execution_time = predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    assert execution_time._is_moe is False
    assert execution_time._moe_tensor_parallel_allreduce_time == pytest.approx(1.25)
    assert execution_time.mlp_operator_times is not None
    assert execution_time.mlp_operator_times.get_required_time("mlp_up_proj") > 0.0


def test_dummy_monolithic_attention_only_excludes_ffn_components() -> None:
    """The dummy attention probe must not fabricate a post-attention block."""

    predictor = _build_dummy_mixed_layer_predictor(
        _DummyModelConfig(
            ModelArchitectureProfile.generic(),
            moe_layer_ids={4, 5, 6},
        )
    )
    predictor._replica_config.attn_tensor_parallel_size = 2

    execution_time = predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=4,
        include_ffn=False,
    )

    assert execution_time._is_moe is False
    assert execution_time.get_single_layer_attention_time() > 0.0
    assert execution_time.get_single_layer_post_attention_time() == pytest.approx(0.0)
    assert execution_time._mlp_norm_time == pytest.approx(0.0)
    assert execution_time._add_time == pytest.approx(0.0)
    assert execution_time._add_attn_residual_time == pytest.approx(0.0)
    assert execution_time._add_ffn_residual_time == pytest.approx(0.0)
    assert execution_time._mlp_layer_up_proj_execution_time == pytest.approx(0.0)
    assert execution_time._mlp_layer_down_proj_execution_time == pytest.approx(0.0)
    assert execution_time._mlp_layer_act_execution_time == pytest.approx(0.0)
    assert execution_time._moe_tensor_parallel_allreduce_time == pytest.approx(0.0)
    assert execution_time.mlp_operator_times is None
    assert execution_time.moe_operator_times is None


def test_dummy_monolithic_mixed_moe_layer_keeps_moe_components() -> None:
    predictor = _build_dummy_mixed_layer_predictor(
        _DummyModelConfig(
            ModelArchitectureProfile.generic(),
            moe_layer_ids={4, 5, 6},
        )
    )

    execution_time = predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=4,
    )

    assert execution_time._is_moe is True
    assert execution_time._moe_gating_time > 0.0
    assert execution_time._moe_grouped_gemm_time > 0.0
    assert execution_time.moe_operator_times is not None


def test_mixed_share_expert_dense_layer_uses_shared_expert_profile_rows() -> None:
    """Step3 mixed dense layers use shared-expert rows, not absent MLP rows."""

    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 1
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.step3_text())
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        attn_dp=1,
    )
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime()
    predictor._get_model_architecture_profile = lambda: ModelArchitectureProfile.step3_text()
    predictor._get_share_expert_up_proj_execution_time = lambda _batch: 1.0
    predictor._get_share_expert_down_proj_execution_time = lambda _batch: 2.0
    predictor._get_share_expert_act_execution_time = lambda _batch: 3.0
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: (_ for _ in ()).throw(
        AssertionError("dense Step3 layer looked up mlp_up_proj")
    )
    predictor._get_mlp_layer_down_proj_execution_time = lambda _batch: (_ for _ in ()).throw(
        AssertionError("dense Step3 layer looked up mlp_down_proj")
    )
    predictor._get_mlp_layer_act_execution_time = lambda _batch: (_ for _ in ()).throw(
        AssertionError("dense Step3 layer looked up mlp_act")
    )
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._get_pp_producer_send_path_runtime_time = lambda *_args: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda *_args: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = lambda *_args: 0.0
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._get_mtp_terminal_overshoot_time = MagicMock(return_value=0.0)
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False

    execution_time = predictor._get_execution_time_internal(
        _DummyBatch(),
        pipeline_stage=0,
        include_moe=False,
        include_ffn=True,
        layer_id=17,
    )

    assert execution_time._is_moe is False
    assert execution_time._mlp_layer_up_proj_execution_time == pytest.approx(1.0)
    assert execution_time._mlp_layer_down_proj_execution_time == pytest.approx(2.0)
    assert execution_time._mlp_layer_act_execution_time == pytest.approx(3.0)
    assert predictor._get_mtp_terminal_overshoot_time.call_args.kwargs["layer_id"] == 17


def test_attention_only_probe_does_not_lookup_dense_ffn_profile() -> None:
    """Attention-only probes must not require dense FFN profiling rows."""

    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 1
    predictor._moe_ep_size = 2
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.generic())
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        attn_dp=1,
    )
    predictor._predictions_eager = {"loaded": True}
    predictor._predictions_kernel_only = {"loaded": True}
    predictor._moe_routing_distribution_type = "random"
    predictor._supports_operation = lambda _operation: True
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime(
        attention_prefill_execution_time=2.0,
        attn_norm_time=1.0,
    )
    predictor._get_pp_producer_send_path_runtime_time = lambda *_args: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda *_args: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = lambda *_args: 0.0
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._model_config.supports_share_expert = lambda: False

    def _unexpected_ffn_lookup(_batch):
        raise AssertionError("attention-only prediction looked up a dense FFN row")

    predictor._get_mlp_layer_up_proj_execution_time = _unexpected_ffn_lookup
    predictor._get_mlp_layer_down_proj_execution_time = _unexpected_ffn_lookup
    predictor._get_mlp_layer_act_execution_time = _unexpected_ffn_lookup
    predictor._get_mlp_norm_layer_act_execution_time = _unexpected_ffn_lookup
    predictor._get_add_layer_act_execution_time = _unexpected_ffn_lookup

    result = predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=0,
        include_ffn=False,
    )

    assert result.get_single_layer_attention_time() == pytest.approx(3.0)
    assert result.get_single_layer_post_attention_time() == pytest.approx(0.0)


def test_common_moe_post_attention_probe_skips_attention_lookup() -> None:
    """Shared co-location DECODE EP lanes must expose a post-attention contract."""

    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 1
    predictor._moe_ep_size = 2
    predictor._model_config = _DummyModelConfig(ModelArchitectureProfile.generic())
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        attn_dp=1,
    )
    predictor._get_model_architecture_profile = lambda: ModelArchitectureProfile.generic()
    predictor.predict_attention_layer_time = lambda **_kwargs: (
        (_ for _ in ()).throw(
            AssertionError("post-attention EP lane looked up attention")
        )
    )
    predictor._predict_expert_parallel_phase_operator_times = lambda _batch, **_kwargs: {
        "expert_parallel_alltoall_dispatch": 0.25,
        "expert_parallel_alltoall_combine": 0.75,
    }
    predictor._get_gating_linear_time = lambda _batch: 0.5
    predictor._get_gating_routing_topk_time = lambda _batch: 0.5
    predictor._get_moe_shuffling_time = lambda _batch, **_kwargs: 0.5
    predictor._get_grouped_gemm_time = lambda _tokens, **_kwargs: 5.0
    predictor._predict_comm_operator = lambda *_args, **_kwargs: 0.75
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 1.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 2.0
    predictor.predict_dp_moe_allreduce_times = lambda *_args, **_kwargs: (0.0, 0.0)
    predictor._get_pp_producer_send_path_runtime_time = lambda *_args: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda *_args: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = lambda *_args: 0.0
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._model_config.supports_share_expert = lambda: False

    lane_workload = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=2,
        total_expert_num=2,
        owned_expert_ids=(0,),
        local_token_counts=(4,),
        routed_token_count=4,
        router_topk=1,
    )
    result = predictor._get_execution_time_internal(
        _DummyBatch(),
        pipeline_stage=0,
        moe_tokens_input=lane_workload,
        include_moe=True,
        include_ffn=True,
        include_attention=False,
    )

    assert result.get_single_layer_attention_time() == pytest.approx(0.0)
    assert result._attn_tensor_parallel_allreduce_time == pytest.approx(0.0)
    assert result._tensor_parallel_communication_time == pytest.approx(0.0)
    phases = (
        result.get_single_layer_moe_pre_dispatch_time(),
        result.get_single_layer_moe_dispatch_time(),
        result.get_single_layer_moe_post_dispatch_compute_time(),
        result.get_single_layer_moe_combine_time(),
        result.get_single_layer_moe_post_combine_time(),
    )
    assert sum(phases) == pytest.approx(
        result.get_single_layer_post_attention_time()
    )


def test_predict_stage_execution_time_keeps_per_layer_components_and_scales_linearly() -> None:
    predictor = _build_predictor()
    lane_workload = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(0, 1),
        local_token_counts=(8, 8),
        routed_token_count=16,
        router_topk=2,
    )
    batch = _DummyBatch(lane_workload=lane_workload)

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
    lane_workload = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(0, 1),
        local_token_counts=(8, 8),
        routed_token_count=16,
        router_topk=2,
    )
    batch = _DummyBatch(lane_workload=lane_workload)
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


def test_mixed_dense_layer_trace_is_not_labeled_as_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _build_predictor()
    predictor._model_config = _DummyModelConfig(
        ModelArchitectureProfile.generic(),
        moe_layer_ids={4, 5, 6},
    )
    predictor._get_execution_time_internal = MagicMock(
        return_value=_build_dense_execution_time()
    )
    messages: list[str] = []
    monkeypatch.setattr(
        sklearn_moe_execution_time_predictor.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message
        ),
    )

    predictor.predict_stage_execution_time(
        _DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    log_text = "\n".join(messages)
    assert "[DENSE_FFN][mlp_up_proj]" in log_text
    assert "[DENSE_FFN][mlp_down_proj]" in log_text
    assert "[DENSE_FFN][mlp_act]" in log_text
    assert "[MOE][moe_grouped_gemm]" not in log_text


def test_monolithic_decode_shared_domain_lane_moe_times_respects_dummy_mode() -> None:
    predictor = _DummySklearnMoEPredictor.__new__(_DummySklearnMoEPredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.25
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._num_layers_per_pipeline_stage = 1
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        attn_dp=1,
        num_pipeline_stages=1,
    )
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

    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0},
        target_replica_id=0,
        global_layer_id=7,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
    )
    predictor._materialize_layer_ep_workload = MagicMock(return_value=workload)

    batch = _DummyBatch()
    lane_times = predictor.predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        batch=batch,
        layer_id=7,
    )

    assert lane_times == {
        0: pytest.approx(1.25 * 5.0),
        1: pytest.approx(1.25 * 3.0),
    }


def test_share_expert_visibility_scaling_hook_is_removed() -> None:
    assert not hasattr(
        SklearnMoEExecutionTimePredictor,
        "_apply_share_expert_tp_allreduce_overlap",
    )


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


def test_monolithic_predictor_exposes_global_per_replica_layer_routing_details() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._global_routing_allocations = {
        0: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        1: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
    }
    predictor._replica_config = SimpleNamespace(cluster_num_replicas=2)

    details = predictor._build_shared_routing_details()

    assert details == {
        0: {
            0: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            1: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        },
        1: {
            0: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            1: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        },
    }


def test_monolithic_routing_allocations_use_global_expert_ids() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._model_config = SimpleNamespace(num_layers=2)
    predictor._replica_config = SimpleNamespace(total_expert_num=4)
    predictor._moe_ep_size = 2
    predictor._moe_routing_seed = 7
    predictor._moe_routing_distribution_type = "random"

    allocations = predictor._init_routing_allocations()

    assert set(allocations) == {0, 1}
    assert set(allocations[0]) == {0, 1, 2, 3}
    assert set(allocations[1]) == {0, 1, 2, 3}
    assert sum(allocations[0].values()) == pytest.approx(1.0)
    assert sum(allocations[1].values()) == pytest.approx(1.0)


def test_shared_materializer_rejects_missing_layer_instead_of_using_layer_zero() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        total_expert_num=2,
        moe_expert_parallel_size=1,
        router_topk=1,
    )
    predictor._monolithic_routing_details = {0: {0: {0: 0.5, 1: 0.5}}}
    batch = SimpleNamespace(replica_id=0, total_num_tokens=4)

    with pytest.raises(ValueError, match="missing global_layer_id 3"):
        predictor._materialize_layer_ep_workload(
            batch=batch,
            cluster_type=ClusterType.MONOLITHIC,
            layer_id=3,
        )


def test_routing_initialization_rejects_non_divisible_ep_topology() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._model_config = SimpleNamespace(num_layers=2)
    predictor._replica_config = SimpleNamespace(total_expert_num=5)
    predictor._moe_ep_size = 2
    predictor._moe_routing_seed = 7
    predictor._moe_routing_distribution_type = "balanced"

    with pytest.raises(ValueError, match="divisible"):
        predictor._init_global_routing_allocations()


def test_dense_model_skips_global_routing_initialization() -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._model_config = SimpleNamespace(num_layers=2, is_moe=False)
    predictor._replica_config = SimpleNamespace(
        total_expert_num=0,
        moe_expert_parallel_size=1,
    )
    predictor._moe_ep_size = 1
    predictor._moe_routing_seed = 7
    predictor._moe_routing_distribution_type = "balanced"

    assert predictor._init_global_routing_allocations() == {}


def test_disaggregation_dense_model_has_no_routing_details() -> None:
    predictor = object.__new__(_DummyDisaggregationPredictor)
    predictor._enable_dummy_mode = True
    predictor._actual_replica_ids = [0]
    predictor._cluster_config = SimpleNamespace(
        prefill_replica_config=SimpleNamespace(
            model_config=SimpleNamespace(num_layers=2, is_moe=False),
            total_expert_num=0,
            moe_expert_parallel_size=1,
        ),
        prefill_cluster_num_replicas=1,
    )

    assert predictor._simulate_and_store_routing(ClusterType.PREFILL) == {}


@pytest.mark.parametrize("distribution_type", ["balanced", "random", "skewed", "zipf"])
def test_global_routing_allocations_use_canonical_distribution_types(
    distribution_type: str,
) -> None:
    predictor = object.__new__(_DummySklearnMoEPredictor)
    predictor._model_config = SimpleNamespace(num_layers=2)
    predictor._replica_config = SimpleNamespace(total_expert_num=4)
    predictor._moe_ep_size = 2
    predictor._moe_routing_seed = 11
    predictor._moe_routing_distribution_type = distribution_type

    first = predictor._init_global_routing_allocations()
    second = predictor._init_global_routing_allocations()

    assert first == second
    assert set(first) == {0, 1}
    for ratios in first.values():
        assert set(ratios) == {0, 1, 2, 3}
        assert sum(ratios.values()) == pytest.approx(1.0)

    if distribution_type == "balanced":
        assert all(value == pytest.approx(0.25) for value in first[0].values())
    else:
        assert len(set(first[0].values())) > 1


@pytest.mark.parametrize("distribution_type", ["balanced", "random", "skewed", "zipf"])
def test_disaggregation_routing_matches_shared_predictor_rng_contract(
    distribution_type: str,
) -> None:
    """All architectures must materialize the same per-layer routing ratios."""
    shared_predictor = object.__new__(_DummySklearnMoEPredictor)
    shared_predictor._model_config = SimpleNamespace(num_layers=2, is_moe=True)
    shared_predictor._replica_config = SimpleNamespace(total_expert_num=8)
    shared_predictor._moe_ep_size = 2
    shared_predictor._moe_routing_seed = 17
    shared_predictor._moe_routing_distribution_type = distribution_type
    expected = shared_predictor._init_global_routing_allocations()

    disaggregation_predictor = object.__new__(_DummyDisaggregationPredictor)
    disaggregation_predictor._distribution_seed = 17
    disaggregation_predictor._workload_distribution_type = (
        WorkloadDistributionType(distribution_type)
    )

    for layer_id in range(2):
        actual = disaggregation_predictor._generate_expert_allocations(
            total_expert_num=8,
            expert_parallel_size=2,
            replica_id=0,
            layer_id=layer_id,
        )
        assert actual == pytest.approx(
            [expected[layer_id][expert_id] for expert_id in range(8)]
        )
