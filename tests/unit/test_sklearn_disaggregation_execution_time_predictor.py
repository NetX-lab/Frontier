from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from frontier.entities import EPBatchGroup, Request
from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators.families import get_comm_operator
from frontier.operators.spec import CommPayloadContext
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


@pytest.mark.parametrize(
    ("attn_tp", "moe_tp", "expected_attn_tp_time", "expected_moe_tp_time"),
    (
        (4, 1, 10.0, 0.0),
        (1, 2, 0.0, 10.0),
    ),
)
def test_dummy_shared_domain_uses_separate_attention_and_moe_tp_domains(
    attn_tp: int,
    moe_tp: int,
    expected_attn_tp_time: float,
    expected_moe_tp_time: float,
) -> None:
    model_config = _ProfileOnlyStep3ModelConfig()
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=model_config,
        attn_tensor_parallel_size=attn_tp,
        moe_tensor_parallel_size=moe_tp,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
    )

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.PREFILL,
    )

    assert execution_time._attn_tensor_parallel_allreduce_time == pytest.approx(
        expected_attn_tp_time
    )
    assert execution_time._moe_tensor_parallel_allreduce_time == pytest.approx(
        expected_moe_tp_time
    )


@pytest.mark.parametrize(
    ("cluster_type", "batch_request"),
    (
        (ClusterType.PREFILL, Request(0.0, 4, 0)),
        (ClusterType.DECODE, Request(0.0, 0, 4)),
    ),
)
def test_dummy_shared_domain_post_attention_zeroes_attention_components(
    cluster_type: ClusterType,
    batch_request: Request,
) -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())

    lane = EPBatchGroup(
        requests=[batch_request],
        num_tokens=[4],
        replica_id=0,
        ep_id=0,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 4, 1: 0},
        cluster_type=cluster_type,
        is_moe=True,
    )
    execution_time = predictor._get_dummy_execution_time_for_cluster(
        batch=lane,
        pipeline_stage=0,
        cluster_type=cluster_type,
        include_attention=False,
    )

    assert execution_time.get_single_layer_attention_time() == pytest.approx(0.0)
    assert execution_time._attn_tensor_parallel_allreduce_time == pytest.approx(0.0)
    assert execution_time._communication_time.tensor_parallel_allreduce_time == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    "cluster_type",
    (ClusterType.PREFILL, ClusterType.DECODE),
)
def test_dummy_layer_scaling_preserves_named_tp_components(
    cluster_type: ClusterType,
) -> None:
    predictor = _dummy_predictor(_ProfileOnlyStep3ModelConfig())
    predictor._enable_dummy_mode = True
    predictor._num_layers_per_pipeline_stage = 32
    predictor._log_architecture_attention_shape = lambda _batch: None

    full_execution_time = predictor.predict_stage_execution_time(
        batch=SimpleNamespace(),
        stage_id=0,
        cluster_type=cluster_type,
        num_layers=32,
    )
    execution_time = predictor.predict_stage_execution_time(
        batch=SimpleNamespace(),
        stage_id=0,
        cluster_type=cluster_type,
        num_layers=1,
    )
    full_communication_time = full_execution_time.communication_time_component
    communication_time = execution_time.communication_time_component

    assert full_execution_time.get_single_layer_attention_time() == pytest.approx(
        60.0
    )
    assert full_communication_time.attn_tensor_parallel_allreduce_time == pytest.approx(
        10.0
    )
    assert full_communication_time.moe_tensor_parallel_allreduce_time == pytest.approx(
        10.0
    )
    assert full_execution_time.get_single_layer_attention_scope_time() == pytest.approx(
        70.0
    )
    assert execution_time.get_single_layer_attention_time() == pytest.approx(1.875)
    assert communication_time.attn_tensor_parallel_allreduce_time == pytest.approx(
        0.3125
    )
    assert communication_time.moe_tensor_parallel_allreduce_time == pytest.approx(
        0.3125
    )
    assert execution_time.get_single_layer_attention_scope_time() == pytest.approx(
        2.1875
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


@pytest.mark.parametrize(
    "cluster_type",
    (ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_FFN),
)
@pytest.mark.parametrize(
    ("moe_tp_size", "expected_time", "expected_central_calls"),
    ((2, 7.0, 1), (1, 0.0, 0)),
)
def test_disaggregation_moe_live_paths_use_registered_role_context(
    monkeypatch,
    cluster_type: ClusterType,
    moe_tp_size: int,
    expected_time: float,
    expected_central_calls: int,
) -> None:
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = cluster_type
    model_config = SimpleNamespace(
        is_moe=True,
        embedding_dim=128,
        is_moe_layer=lambda _layer_id: True,
        get_model_architecture_profile=lambda: ModelArchitectureProfile.generic(),
    )
    replica_config = SimpleNamespace(
        model_config=model_config,
        total_expert_num=4,
        attn_tensor_parallel_size=4,
        moe_tensor_parallel_size=moe_tp_size,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
    )
    predictor._replica_config = replica_config
    predictor._model_config = model_config
    predictor._get_cluster_replica_config = lambda _cluster_type: replica_config
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    predictor._get_communication_time = lambda *_args, **_kwargs: SimpleNamespace(
        tensor_parallel_time=3.0,
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
    predictor.predict_attention_layer_time = lambda *_args, **_kwargs: AttentionTime()
    predictor.predict_moe_layer_time = lambda *_args, **_kwargs: SimpleNamespace(
        moe_grouped_gemm_time=5.0,
        moe_gating_time=1.0,
        moe_shuffling_time=0.5,
        share_expert_up_proj_time=0.0,
        share_expert_down_proj_time=0.0,
        share_expert_act_time=0.0,
        total_time=lambda: 6.5,
    )
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 1.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 2.0
    predictor.predict_dp_moe_allreduce_times = lambda *_args: (0.0, 0.0)
    predictor._predict_named_ep_phase_operator_times = lambda **_kwargs: {
        "expert_parallel_alltoall_dispatch": 0.25,
        "expert_parallel_alltoall_combine": 0.75,
    }
    predictor._predict_one_op_time = (
        lambda _name, value, *_args, **_kwargs: value
    )
    predictor.predict_allreduce_time = MagicMock(return_value=3.0)
    predictor._predict_moe_tp_allreduce_time = MagicMock(
        side_effect=AssertionError(
            "disaggregation live paths must use the registered communication operator"
        )
    )
    predictor._predict_comm_operator_with_context = MagicMock(return_value=7.0)
    quantization_manager = object()
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor.get_quantization_manager",
        lambda: quantization_manager,
    )
    lane = EPBatchGroup(
        requests=[Request(0.0, 100, 0)],
        num_tokens=[100],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[7],
        per_expert_tokens={0: 0, 1: 3},
        cluster_type=cluster_type,
        is_moe=True,
    )

    result = predictor.predict_stage_execution_time(
        lane,
        stage_id=0,
        cluster_type=cluster_type,
        num_layers=1,
        layer_id=2,
    )

    predictor._predict_moe_tp_allreduce_time.assert_not_called()
    assert (
        predictor._predict_comm_operator_with_context.call_count
        == expected_central_calls
    )
    if expected_central_calls:
        operator, context = predictor._predict_comm_operator_with_context.call_args.args
        assert operator is get_comm_operator("moe_tensor_parallel_allreduce")
        assert isinstance(context, CommPayloadContext)
        assert context.batch is lane
        assert context.model_config is model_config
        assert context.replica_config is replica_config
        assert context.cluster_type is cluster_type
        assert context.quantization_manager is quantization_manager
    assert result._moe_tensor_parallel_allreduce_time == pytest.approx(expected_time)


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


def test_pdd_attention_only_prediction_preserves_global_layer_id() -> None:
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.PREFILL
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    predictor._get_communication_time = lambda *_args: SimpleNamespace(
        tensor_parallel_time=0.0,
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
    predictor._predict_one_op_time = (
        lambda _name, value, *_args, **_kwargs: value
    )
    predictor.predict_attention_layer_time = MagicMock(
        return_value=AttentionTime(attention_prefill_execution_time=3.0)
    )

    predictor.predict_stage_execution_time(
        batch=SimpleNamespace(id=17),
        stage_id=3,
        cluster_type=ClusterType.PREFILL,
        num_layers=1,
        layer_id=17,
        include_ffn=False,
    )

    predictor.predict_attention_layer_time.assert_called_once()
    assert predictor.predict_attention_layer_time.call_args.kwargs["layer_id"] == 17


@pytest.mark.parametrize(
    "cluster_type",
    (ClusterType.PREFILL, ClusterType.DECODE),
)
def test_pdd_shared_domain_post_attention_prediction_skips_attention_lookup(
    cluster_type: ClusterType,
) -> None:
    predictor = _DummyDisaggregationPredictor.__new__(
        _DummyDisaggregationPredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = cluster_type
    predictor._replica_config = SimpleNamespace(
        total_expert_num=4,
        moe_expert_parallel_size=2,
    )
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda *_args: False
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    communication_attention_flags: list[bool] = []

    def _get_communication_time(
        *_args,
        include_attention: bool = True,
    ) -> SimpleNamespace:
        communication_attention_flags.append(include_attention)
        return SimpleNamespace(
            tensor_parallel_time=4.0 if include_attention else 0.0,
            pipeline_parallel_time=0.0,
        )

    predictor._get_communication_time = _get_communication_time
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
        total_time=lambda: 6.5,
    )
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 1.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 2.0
    predictor.predict_dp_moe_allreduce_times = lambda *_args: (0.0, 0.0)
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
        total_num_tokens=4,
        requests=[SimpleNamespace(id=7)],
    )

    result = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=cluster_type,
        num_layers=1,
        layer_id=2,
        include_attention=False,
    )

    assert communication_attention_flags == [False]
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
