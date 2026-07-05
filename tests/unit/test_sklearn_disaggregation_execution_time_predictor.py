from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
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

    # DECODE_FFN dummy mode applies its existing 0.02 calibration before capability logic.
    assert execution_time.moe_tensor_parallel_allgather_time == 0.2
    assert execution_time.share_expert_tensor_parallel_allreduce_time == 0.2


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
