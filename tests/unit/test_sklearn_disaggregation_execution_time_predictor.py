from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.entities import EPBatchGroup, Request
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


class _DummyDisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _ProfileOnlyStep3ModelConfig:
    is_moe = True
    embedding_dim = 128

    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return ModelArchitectureProfile.step3_text()

    def supports_share_expert(self) -> bool:
        return True


class _Step3NamedGenericProfileModelConfig:
    is_moe = True
    model_type = "step3_text"
    embedding_dim = 128

    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return ModelArchitectureProfile.generic()

    def supports_share_expert(self) -> bool:
        return True


def _dummy_predictor(model_config: object) -> SklearnDisaggregationExecutionTimePredictor:
    predictor = _DummyDisaggregationPredictor.__new__(_DummyDisaggregationPredictor)
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=model_config,
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
    )
    return predictor


def test_dummy_decode_attn_residual_skip_uses_profile_capability_not_legacy_identity() -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    assert execution_time.add_attn_residual_time == 0.0


def test_dummy_decode_ffn_tp_collectives_use_profile_capability_not_legacy_identity() -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE_FFN,
    )

    # DECODE_FFN dummy mode uses the configured per-operation time directly;
    # no empirical stage correction is applied.
    assert execution_time.moe_tensor_parallel_allgather_time == 10.0
    assert execution_time.share_expert_tensor_parallel_allreduce_time == 10.0


def test_dummy_unified_decode_post_attention_zeroes_attention_tp_communication() -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())

    lane = EPBatchGroup(
        requests=[Request(0.0, 0, 4)],
        num_tokens=[4],
        replica_id=0,
        ep_id=0,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 4, 1: 0},
        cluster_type=ClusterType.DECODE,
        is_moe=True,
    )
    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=lane,
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE,
        include_attention=False,
    )

    assert execution_time._attn_tensor_parallel_allreduce_time == pytest.approx(0.0)
    assert execution_time._communication_time.tensor_parallel_allreduce_time == pytest.approx(
        0.0
    )


def test_dummy_decode_attn_allows_zero_moe_ep_for_attention_only_cluster() -> None:
    predictor = _DummyDisaggregationPredictor.__new__(_DummyDisaggregationPredictor)
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=_ProfileOnlyStep3ModelConfig(),
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
        num_pipeline_stages=1,
    )

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE_ATTN,
    )

    assert execution_time._is_moe is False
    assert execution_time.expert_parallel_communication_time == 0.0


def test_dummy_decode_ffn_rejects_zero_moe_ep_for_moe_cluster() -> None:
    predictor = _DummyDisaggregationPredictor.__new__(_DummyDisaggregationPredictor)
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=_ProfileOnlyStep3ModelConfig(),
        attn_tensor_parallel_size=0,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=0,
        num_pipeline_stages=1,
    )

    with pytest.raises(
        ValueError,
        match="positive integer moe_expert_parallel_size, got 0",
    ):
        predictor._get_dummy_execution_time_for_cluster(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            cluster_type=ClusterType.DECODE_FFN,
        )


@pytest.mark.parametrize(
    "cluster_type",
    (ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_FFN),
)
def test_dummy_moe_clusters_publish_named_ep_phase_times(
    cluster_type: ClusterType,
) -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=cluster_type,
    )

    assert execution_time.get_single_layer_moe_dispatch_time() == pytest.approx(10.0)
    assert execution_time.get_single_layer_moe_combine_time() == pytest.approx(10.0)
    assert execution_time.expert_parallel_communication_time == pytest.approx(20.0)


def test_common_dummy_moe_predictor_zero_lane_has_no_routed_compute() -> None:
    """Zero-routed lanes keep shared work but cannot fabricate expert compute."""
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 2.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._model_config = _ProfileOnlyStep3ModelConfig()
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        attn_dp=1,
        num_pipeline_stages=1,
    )

    lane = EPBatchGroup(
        requests=[Request(0.0, 0, 0)],
        num_tokens=[0],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 0, 1: 0},
        cluster_type=ClusterType.MONOLITHIC,
        is_moe=True,
    )

    execution_time = predictor._get_dummy_execution_time(lane, pipeline_stage=0)

    assert execution_time.get_single_layer_moe_post_dispatch_compute_time() == 0.0
    assert execution_time.get_single_layer_moe_pre_dispatch_time() > 0.0
    assert execution_time.get_single_layer_moe_dispatch_time() > 0.0
    assert execution_time.get_single_layer_moe_combine_time() > 0.0


def test_common_dummy_moe_predictor_zero_explicit_allocation_has_no_routed_compute() -> None:
    """The explicit allocation API must honor zero routed tokens too."""
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 2.0
    predictor._model_config = _ProfileOnlyStep3ModelConfig()

    moe_time = predictor.predict_moe_layer_time(
        batch_or_group=SimpleNamespace(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        per_expert_tokens={0: 0, 1: 0},
    )

    assert moe_time.moe_grouped_gemm_time == 0.0
    assert moe_time.moe_shuffling_time > 0.0
    assert moe_time.operator_times is not None
    assert moe_time.operator_times.get_required_time("moe_grouped_gemm") == 0.0


@pytest.mark.parametrize(
    "cluster_type",
    (ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_FFN),
)
def test_disaggregation_dummy_zero_lane_has_no_routed_compute(
    cluster_type: ClusterType,
) -> None:
    """All disaggregation MoE roles must preserve the zero-routed contract."""
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())
    lane = EPBatchGroup(
        requests=[Request(0.0, 0, 0)],
        num_tokens=[0],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 0, 1: 0},
        cluster_type=cluster_type,
        is_moe=True,
    )

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=lane,
        pipeline_stage=0,
        cluster_type=cluster_type,
        include_attention=cluster_type is not ClusterType.DECODE_FFN,
    )

    assert execution_time.get_single_layer_moe_post_dispatch_compute_time() == 0.0
    assert execution_time.get_single_layer_moe_pre_dispatch_time() > 0.0
    assert execution_time.get_single_layer_moe_dispatch_time() > 0.0
    assert execution_time.get_single_layer_moe_combine_time() > 0.0


def test_pdd_predictor_has_no_direct_step3_identity_branches() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root
        / "frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py"
    ).read_text(encoding="utf-8")

    assert ".is_step3_text()" not in source
    assert "_log_step3_attention_shape" not in source


def test_pdd_predictor_fails_fast_when_cluster_model_config_is_missing() -> None:
    predictor = _dummy_predictor(None)

    try:
        predictor._get_dummy_execution_time_for_cluster(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            cluster_type=ClusterType.DECODE_ATTN,
        )
    except ValueError as exc:
        assert "model_config" in str(exc)
    else:
        raise AssertionError("PDD predictor must not fall back to a generic profile")


def test_pdd_predictor_uses_profile_not_step3_named_legacy_identity() -> None:
    predictor = _dummy_predictor(_Step3NamedGenericProfileModelConfig())

    decode_attn_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    decode_ffn_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.DECODE_FFN,
    )

    assert decode_attn_time.add_attn_residual_time == 10.0
    assert decode_ffn_time.moe_tensor_parallel_allgather_time == 0.0
    assert decode_ffn_time.share_expert_tensor_parallel_allreduce_time == 0.0


def test_disaggregation_grouped_gemm_delegates_with_lane_batch(monkeypatch) -> None:
    predictor = _DummyDisaggregationPredictor.__new__(_DummyDisaggregationPredictor)
    expected = 3.5

    def _base_grouped_gemm(_self, allocation, *, batch=None):
        assert allocation == {0: 2, 1: 4}
        assert batch is not None
        return expected

    monkeypatch.setattr(
        SklearnMoEExecutionTimePredictor,
        "_get_grouped_gemm_time",
        _base_grouped_gemm,
    )

    assert predictor._get_grouped_gemm_time(
        {0: 2, 1: 4}, batch=SimpleNamespace(id=7)
    ) == expected


def test_disaggregation_moe_tp_allreduce_uses_lane_routed_tokens() -> None:
    lane = EPBatchGroup(
        requests=[Request(0.0, 0, 3)],
        num_tokens=[3],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={2: 0, 3: 3},
        cluster_type=ClusterType.DECODE,
        is_moe=True,
    )

    assert (
        SklearnDisaggregationExecutionTimePredictor._get_moe_tp_routed_tokens(
            lane,
            ClusterType.DECODE,
        )
        == 3
    )


def test_disaggregation_moe_tp_allreduce_rejects_empty_lane_routing_map() -> None:
    lane = EPBatchGroup(
        requests=[Request(0.0, 0, 0)],
        num_tokens=[0],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={},
        cluster_type=ClusterType.DECODE,
        is_moe=True,
    )

    with pytest.raises(ValueError, match="requires a non-empty"):
        SklearnDisaggregationExecutionTimePredictor._get_moe_tp_routed_tokens(
            lane,
            ClusterType.DECODE,
        )


def test_disaggregation_moe_tp_allreduce_keeps_source_tokens_for_shared_batch() -> None:
    batch = SimpleNamespace(
        get_effective_total_tokens_rounded=lambda cluster_type: (
            8 if cluster_type is ClusterType.PREFILL else 5
        )
    )

    assert (
        SklearnDisaggregationExecutionTimePredictor._get_moe_tp_routed_tokens(
            batch,
            ClusterType.PREFILL,
        )
        == 8
    )


def test_disaggregation_dense_layer_uses_shared_expert_profile_rows() -> None:
    predictor = _DummyDisaggregationPredictor.__new__(_DummyDisaggregationPredictor)
    predictor._enable_dummy_mode = False
    predictor._model_config = _ProfileOnlyStep3ModelConfig()
    predictor._supports_operation = lambda operation: operation in {
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
        "post_attention_layernorm",
    }
    predictor._get_share_expert_up_proj_execution_time = lambda _batch: 1.0
    predictor._get_share_expert_down_proj_execution_time = lambda _batch: 2.0
    predictor._get_share_expert_act_execution_time = lambda _batch: 3.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 4.0
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: (_ for _ in ()).throw(
        AssertionError("standard MLP profile must not be requested")
    )
    batch = SimpleNamespace(
        id=9,
        total_num_tokens=8,
        requests=[SimpleNamespace(id=1, num_prefill_tokens=8)],
    )

    result = predictor.predict_mlp_layer_time(
        batch, layer_id=2, cluster_type=ClusterType.PREFILL
    )

    assert result.mlp_layer_up_proj_execution_time == 1.0
    assert result.mlp_layer_down_proj_execution_time == 2.0
    assert result.mlp_layer_act_execution_time == 3.0
    assert result.mlp_norm_time == 4.0


def test_pdd_decode_post_attention_prediction_skips_attention_lookup() -> None:
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.DECODE
    predictor._replica_config = SimpleNamespace(
        total_expert_num=4,
        moe_expert_parallel_size=2,
    )
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda *_args: False
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    predictor._get_communication_time = lambda *_args: SimpleNamespace(
        tensor_parallel_time=4.0,
        pipeline_parallel_time=0.0,
    )
    predictor._get_overhead_time = lambda *_args: SimpleNamespace(
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        pp_producer_send_path_runtime_time=0.0,
        pp_receiver_head_runtime_time=0.0,
        pp_prefill_consumer_active_runtime_time=0.0,
        pp_stage_boundary_residual_runtime_time=0.0,
        pp_stage_boundary_handoff_time=0.0,
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    model_config = SimpleNamespace(
        is_moe=True,
        is_moe_layer=lambda layer_id: layer_id == 2,
        embedding_dim=128,
    )
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=model_config,
        moe_tensor_parallel_size=1,
    )
    predictor.predict_attention_layer_time = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(
            AssertionError("post-attention EP lane looked up attention")
        )
    )
    predictor.predict_moe_layer_time = lambda *_args, **_kwargs: SimpleNamespace(
        moe_grouped_gemm_time=5.0,
        moe_gating_time=1.0,
        moe_shuffling_time=0.5,
        share_expert_up_proj_time=0.0,
        share_expert_down_proj_time=0.0,
        share_expert_act_time=0.0,
    )
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 1.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 2.0
    predictor._predict_named_ep_phase_operator_times = lambda **_kwargs: {
        "expert_parallel_alltoall_dispatch": 0.25,
        "expert_parallel_alltoall_combine": 0.75,
    }
    predictor._predict_one_op_time = (
        lambda _name, value, *_args, **_kwargs: value
    )
    batch = SimpleNamespace(
        id=7,
        per_expert_tokens={0: 4, 1: 0},
    )

    result = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.DECODE,
        num_layers=1,
        layer_id=2,
        include_attention=False,
    )

    assert result.get_single_layer_attention_time() == pytest.approx(0.0)
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
