from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
