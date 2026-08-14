from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.config import BaseExecutionTimePredictorConfig, ClusterConfig
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


class _Predictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _Batch:
    num_prefill_tokens = 0
    requests = []
    total_num_tokens = 32

    def get_effective_total_tokens_rounded(self, _cluster_type: ClusterType) -> int:
        return 32


class _CCBackend:
    def predict_all_to_all(self, **_kwargs) -> float:
        return 7.0

    def predict_allreduce(self, **_kwargs) -> float:
        return 11.0


def test_moe_shuffling_uses_raw_prediction_without_calibration_scale() -> None:
    predictor = object.__new__(_Predictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = lambda _operation: True
    predictor._predictions = {"moe_shuffling": {(32,): 4.0}}
    # This legacy field is deliberately non-unit. It must not affect a canonical
    # MoE prediction after the no-scaling migration.
    predictor._moe_shuffling_calibration_scale = 0.25

    result = predictor._get_moe_shuffling_time(_Batch())

    assert result == pytest.approx(4.0)


def test_moe_grouped_gemm_uses_raw_prediction_without_calibration_scale() -> None:
    predictor = object.__new__(_Predictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = lambda _operation: True
    predictor._predictions = {"moe_grouped_gemm": {(16,): 4.0}}
    predictor._max_tokens = 16
    predictor._moe_grouped_gemm_calibration_scale = 1.75

    result = predictor._get_grouped_gemm_time(16)

    assert result == pytest.approx(4.0)


def test_ep_communication_uses_raw_collective_prediction_without_calibration_scale() -> None:
    predictor = object.__new__(_Predictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = SimpleNamespace(embedding_dim=8)
    predictor._replica_config = SimpleNamespace(
        moe_expert_parallel_size=2,
        moe_tensor_parallel_size=1,
    )
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._cc_backend = _CCBackend()
    predictor._enable_dummy_mode = False
    predictor._should_strip_collective_sim_allreduce_launch_overhead = (
        lambda _batch: False
    )
    predictor._expert_parallel_communication_calibration_scale = 0.25

    result = predictor._get_expert_parallel_communication_time(_Batch())

    assert result == pytest.approx(7.0)


def test_share_expert_visibility_hook_is_removed() -> None:
    assert not hasattr(_Predictor, "_apply_share_expert_tp_allreduce_overlap")


def test_moe_predictor_source_has_no_empirical_visibility_or_calibration_hooks() -> None:
    source = (
        __import__(
            "pathlib"
        ).Path(
            "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py"
        ).read_text(encoding="utf-8")
    )

    assert "share_expert_tp_allreduce_visibility_scale" not in source
    assert "_get_moe_compute_calibration_scale" not in source
    assert "_get_expert_parallel_communication_calibration_scale" not in source


def test_moe_scaling_fields_are_not_public_configuration() -> None:
    forbidden = {
        "moe_shuffling_calibration_scale",
        "decode_phase_moe_shuffling_calibration_scale",
        "moe_grouped_gemm_calibration_scale",
        "decode_phase_moe_grouped_gemm_calibration_scale",
        "expert_parallel_communication_calibration_scale",
        "decode_phase_expert_parallel_communication_calibration_scale",
        "late_decode_expert_parallel_communication_calibration_scale",
        "share_expert_tp_allreduce_visibility_scale",
    }

    assert forbidden.isdisjoint(BaseExecutionTimePredictorConfig.__dataclass_fields__)
    assert forbidden.isdisjoint(ClusterConfig.__dataclass_fields__)
    assert not hasattr(ModelArchitectureProfile.generic(), "share_expert_tp_allreduce_visibility_scale")
