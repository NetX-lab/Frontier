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

    result = predictor._materialize_layer_ep_workload(
        batch=_Batch(),
        cluster_type=ClusterType.MONOLITHIC,
        layer_id=7,
    )

    assert dict(result.global_per_expert_tokens) == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.global_per_expert_tokens.values()) == 8


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

    result = predictor._materialize_layer_ep_workload(
        batch=_Batch(),
        cluster_type=ClusterType.PREFILL,
        layer_id=7,
    )

    assert dict(result.global_per_expert_tokens) == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.global_per_expert_tokens.values()) == 8


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


def test_disaggregation_predictor_preserves_present_all_zero_expert_map() -> None:
    source = Path(
        "frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py"
    ).read_text(encoding="utf-8")

    assert 'hasattr(batch, "per_expert_tokens") and batch.per_expert_tokens' not in source


def test_decode_ffn_scheduler_uses_replica_local_ep_capacity_name() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")
    round_robin_source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "self._replica_ep_size = int(" in source
    assert "self._replica_dp_size" in source
    assert "_replica_dp_size" in round_robin_source
    assert "Use ep_id as dp_id for compatibility" not in source
    assert "replica_local_id=ep_id" in source


def test_cluster_scheduler_child_map_uses_replica_local_identity() -> None:
    production_paths = [
        Path("frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/lor_cluster_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/random_cluster_scheduler.py"),
        Path("frontier/scheduler/cluster_scheduler/sticky_lor_cluster_scheduler.py"),
        Path(
            "frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.py"
        ),
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    )

    assert "_dp_replica_schedulers" not in combined
    assert "get_dp_replica_scheduler" not in combined
    assert "get_dp_replica_stage_scheduler" not in combined
    assert "self._replica_schedulers" in combined
    assert "def get_replica_scheduler(" in combined
    assert "def get_replica_stage_scheduler(" in combined


def test_round_robin_decode_attn_load_tracker_is_replica_scoped() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "_replica_load_tracker" in source
    assert "_replica_dp_load_tracker" not in source
    assert "intra-Replica attention-DP" in source


def test_decode_collective_has_no_legacy_aggregate_helpers() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    # The predictor-only aggregate helper is part of the canonical shared-DP
    # EP wave path; legacy scalar DP synchronization remains removed below.
    assert "def _create_virtual_global_batch" in source
    assert "def _get_decode_sync_participant_count" not in source
    assert "predict_dp_gather_time" not in source
    assert "predict_dp_scatter_time" not in source
    assert "Legacy DECODE aggregate synchronization is removed" in source
