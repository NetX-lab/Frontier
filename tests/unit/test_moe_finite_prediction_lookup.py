"""Regression tests for finite MoE prediction lookup and model fallback."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.types import ClusterType, MeasurementType


class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("not used")


class _CountingFiniteModel:
    def __init__(self, result: float) -> None:
        self.n_features_in_ = 1
        self._frontier_feature_names = ["num_tokens"]
        self.result = result
        self.calls = 0
        self.seen_values: list[float] = []

    def predict(self, features: Any) -> list[float]:
        self.calls += 1
        assert list(features.columns) == ["num_tokens"]
        self.seen_values.append(float(features.iloc[0]["num_tokens"]))
        return [self.result]


class _CountingLoadAwareModel:
    def __init__(self, feature_names: tuple[str, ...], result: float) -> None:
        self.n_features_in_ = len(feature_names)
        self._frontier_feature_names = list(feature_names)
        self.feature_names_in_ = feature_names
        self.result = result
        self.calls = 0

    def predict(self, features: Any) -> list[float]:
        self.calls += 1
        assert tuple(features.columns) == self.feature_names_in_
        return [self.result]


def _build_predictor(
    model_name: str,
    *,
    model_result: float = 4.5,
    materialized_tokens: int = 1,
    max_tokens: int = 4,
) -> tuple[_ConcreteMoEPredictor, _CountingFiniteModel]:
    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._max_tokens = max_tokens
    predictor._router_topk = 2
    predictor._supports_operation = lambda _operation: True
    predictor._model_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
    )
    predictor._replica_config = SimpleNamespace(total_expert_num=8)
    predictor._get_moe_compute_calibration_scale = lambda *args, **kwargs: 1.0
    predictor._select_moe_gating_prediction_model_name = (
        lambda base_model_name, _batch: base_model_name
    )

    model = _CountingFiniteModel(model_result)
    predictor._models = {model_name: model}
    predictor._predictions = {
        model_name: {(materialized_tokens,): 1.0},
    }
    return predictor, model


def _batch(effective_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_prefill_tokens=0,
        get_effective_total_tokens_rounded=lambda _cluster_type: effective_tokens,
    )


def _lane_workload(expert_tokens: dict[int, int]) -> EPLaneWorkload:
    """Build the canonical EP=1 lane used by scalar/load-aware lookup tests."""

    total_expert_num = 8
    owned_expert_ids = tuple(range(total_expert_num))
    local_token_counts = tuple(int(expert_tokens.get(expert_id, 0)) for expert_id in owned_expert_ids)
    if set(expert_tokens) - set(owned_expert_ids):
        raise ValueError("test expert map contains an out-of-domain expert")
    return EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=1,
        total_expert_num=total_expert_num,
        owned_expert_ids=owned_expert_ids,
        local_token_counts=local_token_counts,
        routed_token_count=sum(local_token_counts),
        router_topk=2,
    )


@pytest.mark.parametrize(
    ("method_name", "model_name"),
    [
        ("_get_gating_linear_time", "moe_gating_linear"),
        ("_get_gating_routing_topk_time", "moe_gating_routing_topk"),
    ],
)
def test_moe_gating_finite_miss_uses_model_once_and_reuses_runtime_cache(
    method_name: str,
    model_name: str,
) -> None:
    predictor, model = _build_predictor(model_name)
    batch = _batch(2)

    first = getattr(predictor, method_name)(batch)
    second = getattr(predictor, method_name)(batch)

    assert first == 4.5
    assert second == 4.5
    assert model.calls == 1
    assert model.seen_values == [2.0]
    assert predictor._runtime_cache["eager"][model_name] == {(2.0,): 4.5}


def test_moe_shuffling_legacy_finite_miss_uses_model_fallback() -> None:
    predictor, model = _build_predictor("moe_shuffling")
    batch = _batch(2)

    value = predictor._get_moe_shuffling_time(batch)

    assert value == 4.5
    assert model.calls == 1
    assert model.seen_values == [2.0]
    assert predictor._runtime_cache["eager"]["moe_shuffling"] == {(2.0,): 4.5}


def test_moe_grouped_gemm_finite_miss_preserves_query_above_profiled_upper_bound() -> None:
    predictor, model = _build_predictor(
        "moe_grouped_gemm",
        model_result=6.25,
        materialized_tokens=4,
        max_tokens=4,
    )

    first = predictor._get_grouped_gemm_time(6)
    second = predictor._get_grouped_gemm_time(6)

    assert first == 6.25
    assert second == 6.25
    assert model.calls == 1
    assert model.seen_values == [6.0]
    assert predictor._runtime_cache["eager"]["moe_grouped_gemm"] == {
        (6.0,): 6.25
    }


def test_moe_grouped_gemm_allocation_miss_preserves_original_pre_routing_tokens() -> None:
    predictor, model = _build_predictor(
        "moe_grouped_gemm",
        model_result=7.0,
        materialized_tokens=4,
        max_tokens=4,
    )

    value = predictor._get_grouped_gemm_time(
        _lane_workload({0: 6, 1: 6}),
        batch=_batch(6),
    )

    assert value == 7.0
    assert model.calls == 1
    assert model.seen_values == [6.0]


def _build_load_aware_predictor(
    *,
    model_result: float = 8.5,
    exact_lookup: dict[tuple[float, ...], float] | None = None,
) -> tuple[_ConcreteMoEPredictor, _CountingLoadAwareModel]:
    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._router_topk = 2
    predictor._supports_operation = lambda _operation: True
    predictor._model_config = SimpleNamespace(
        embedding_dim=4096,
        mlp_hidden_dim=11008,
    )
    predictor._replica_config = SimpleNamespace(total_expert_num=8)
    predictor._get_moe_compute_calibration_scale = lambda *args, **kwargs: 1.0

    feature_names = tuple(predictor.MOE_LOAD_IMBALANCE_FEATURES)
    model = _CountingLoadAwareModel(feature_names, model_result)
    predictor._predictions = {
        "moe_grouped_gemm": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": list(feature_names),
            "_exact_lookup": exact_lookup or {},
        }
    }
    return predictor, model


def test_moe_load_imbalance_runtime_features_match_training_schema() -> None:
    predictor, _model = _build_load_aware_predictor()

    features = predictor._build_moe_load_imbalance_features(
        _lane_workload({0: 4, 1: 4}),
        batch=_batch(4),
    )

    assert tuple(features) == tuple(predictor.MOE_LOAD_IMBALANCE_FEATURES)
    assert "seed" not in features
    assert "load_distribution" not in features


def test_equivalent_moe_allocations_share_canonical_runtime_cache() -> None:
    predictor, model = _build_load_aware_predictor()

    first = predictor._get_grouped_gemm_time(
        _lane_workload({0: 4, 1: 4}),
        batch=_batch(4),
    )
    second = predictor._get_grouped_gemm_time(
        _lane_workload({2: 4, 3: 4}),
        batch=_batch(4),
    )

    assert first == 8.5
    assert second == 8.5
    assert model.calls == 1
    assert len(predictor._runtime_cache["eager"]["moe_grouped_gemm"]) == 1
    assert not hasattr(
        predictor, "_runtime_grouped_gemm_on_demand_prediction_cache"
    )


def test_moe_exact_lookup_precedes_runtime_cache_for_load_imbalance_query() -> None:
    predictor, model = _build_load_aware_predictor()
    features = predictor._build_moe_load_imbalance_features(
        _lane_workload({0: 4, 1: 4}),
        batch=_batch(4),
    )
    key = tuple(float(features[name]) for name in predictor.MOE_LOAD_IMBALANCE_FEATURES)
    predictor._predictions["moe_grouped_gemm"]["_exact_lookup"] = {key: 2.25}
    predictor._runtime_cache["eager"]["moe_grouped_gemm"][key] = 9.0

    value = predictor._get_grouped_gemm_time(
        _lane_workload({0: 4, 1: 4}),
        batch=_batch(4),
    )

    assert value == 2.25
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["moe_grouped_gemm"][key] == 9.0
