from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType


class _MoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _Batch:
    replica_id = 3
    total_num_tokens = 4


def _replica_config() -> SimpleNamespace:
    return SimpleNamespace(
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=2,
    )


def test_monolithic_predictor_materializes_global_routing_with_shared_tie_break() -> None:
    predictor = object.__new__(_MoEPredictor)
    predictor._monolithic_routing_details = {
        3: {
            7: {
                0: 0.1875,
                1: 0.3125,
                2: 0.375,
                3: 0.125,
            }
        }
    }
    predictor._replica_config = _replica_config()

    result = predictor._calculate_expert_token_allocation(
        batch=_Batch(),
        cluster_type=ClusterType.MONOLITHIC,
        layer_id=7,
    )

    assert result == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.values()) == 8


def test_disaggregation_predictor_uses_shared_materializer_tie_break() -> None:
    predictor = object.__new__(_DisaggregationPredictor)
    predictor._prefill_routing_details = {
        3: {
            7: {
                0: 0.1875,
                1: 0.3125,
                2: 0.375,
                3: 0.125,
            }
        }
    }
    predictor._cluster_config = SimpleNamespace(
        prefill_replica_config=_replica_config(),
    )
    predictor._replica_config = _replica_config()

    result = predictor._calculate_expert_token_allocation(
        batch=_Batch(),
        cluster_type=ClusterType.PREFILL,
        layer_id=7,
    )

    assert result == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.values()) == 8


def test_moe_layer_prediction_has_no_one_token_conservation_tolerance() -> None:
    source = Path(
        "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py"
    ).read_text(encoding="utf-8")

    assert "abs(total_allocated_tokens - expected_tokens) > 1" not in source


def test_decode_sync_collective_has_no_uniform_routing_fallback() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "_build_uniform_per_expert_tokens" not in source


def test_moe_predictor_has_one_routing_integerizer() -> None:
    source = Path(
        "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py"
    ).read_text(encoding="utf-8")

    assert "def _build_proportional_per_expert_tokens" not in source
    assert "def _build_balanced_per_expert_tokens" not in source


def test_round_robin_scheduler_has_no_legacy_token_distributor() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "def _distribute_tokens_within_replica" not in source
    assert "def _distribute_batches_to_replicas_round_robin" not in source


def test_base_scheduler_has_no_second_ep_integerizer() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "def _conserve_tokens_allocation" not in source
    assert "def _get_ep_subset_routed_token_total" not in source
    assert "def _get_cached_ep_subset_routed_token_allocation" not in source
    assert "def _get_ep_subset_routed_token_allocation" not in source
