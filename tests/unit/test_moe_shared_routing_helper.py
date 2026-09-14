"""Contract tests for the shared deterministic MoE routing generator."""

from types import SimpleNamespace

import numpy as np
import pytest

from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
    WorkloadDistributionType,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import generate_moe_routing_ratios


class _DummySharedPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DummyDisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


@pytest.mark.parametrize("distribution_type", ("balanced", "random", "skewed", "zipf"))
def test_shared_routing_helper_is_deterministic_and_normalized(
    distribution_type: str,
) -> None:
    first = generate_moe_routing_ratios(
        total_expert_num=8,
        distribution_type=distribution_type,
        seed=17,
        layer_id=3,
    )
    second = generate_moe_routing_ratios(
        total_expert_num=8,
        distribution_type=distribution_type,
        seed=17,
        layer_id=3,
    )

    assert first == second
    assert tuple(first) == tuple(range(8))
    assert sum(first.values()) == pytest.approx(1.0)
    assert all(np.isfinite(value) and value >= 0.0 for value in first.values())


def test_random_routing_helper_uses_layer_id_in_seed_arithmetic() -> None:
    layer_zero = generate_moe_routing_ratios(
        total_expert_num=8,
        distribution_type="random",
        seed=17,
        layer_id=0,
    )
    layer_one = generate_moe_routing_ratios(
        total_expert_num=8,
        distribution_type="random",
        seed=17,
        layer_id=1,
    )

    assert layer_zero != layer_one


@pytest.mark.parametrize(
    ("total_expert_num", "distribution_type", "seed", "layer_id"),
    ((0, "balanced", 1, 0), (4, "balanced", -1, 0), (4, "balanced", 1, -1)),
)
def test_shared_routing_helper_rejects_invalid_generation_inputs(
    total_expert_num: int,
    distribution_type: str,
    seed: int,
    layer_id: int,
) -> None:
    with pytest.raises(ValueError):
        generate_moe_routing_ratios(
            total_expert_num=total_expert_num,
            distribution_type=distribution_type,
            seed=seed,
            layer_id=layer_id,
        )


def test_standard_and_disaggregation_predictors_share_ratio_source() -> None:
    shared = object.__new__(_DummySharedPredictor)
    shared._model_config = SimpleNamespace(num_layers=2)
    shared._replica_config = SimpleNamespace(total_expert_num=8)
    shared._cluster_type = None
    shared._moe_ep_size = 2
    shared._moe_routing_seed = 17
    shared._moe_routing_distribution_type = "random"
    shared_allocations = shared._init_global_routing_allocations()

    disaggregation = object.__new__(_DummyDisaggregationPredictor)
    disaggregation._distribution_seed = 17
    disaggregation._workload_distribution_type = WorkloadDistributionType.RANDOM

    for layer_id in range(2):
        disaggregation_allocations = disaggregation._generate_expert_allocations(
            total_expert_num=8,
            expert_parallel_size=2,
            replica_id=0,
            layer_id=layer_id,
        )
        assert disaggregation_allocations == pytest.approx(
            [shared_allocations[layer_id][expert_id] for expert_id in range(8)]
        )
