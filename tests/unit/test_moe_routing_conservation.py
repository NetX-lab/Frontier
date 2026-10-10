from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from predictor_cache_fixtures import (
    CacheFixturePredictor,
    CacheFixtureDisaggregationPredictor,
)

from frontier.entities import Batch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch
from frontier.scheduler.utils.batch_builders import build_virtual_global_batch
from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.scheduler.utils.layer_workload import materialize_layer_workload
from frontier.types import ClusterType


def _predictor_package_source() -> str:
    return "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("frontier/execution_time_predictor").glob("*.py"))
    )


class _MoEPredictor(CacheFixturePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DisaggregationPredictor(CacheFixtureDisaggregationPredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _Batch:
    replica_id = 3
    total_num_tokens = 4

    def get_effective_total_tokens_for_compute(self, cluster_type):
        return self.total_num_tokens


def test_monolithic_predictor_materializes_global_routing_with_shared_tie_break() -> None:
    predictor = _MoEPredictor(replica_id=3, total_experts=4, ep_size=2)
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

    result = predictor._materialize_layer_ep_workload(
        batch=_Batch(),
        cluster_type=ClusterType.MONOLITHIC,
        layer_id=7,
    )

    assert dict(result.global_per_expert_tokens) == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.global_per_expert_tokens.values()) == 8


def test_disaggregation_predictor_uses_shared_materializer_tie_break() -> None:
    predictor = _DisaggregationPredictor(replica_id=3, total_experts=4, ep_size=2)
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

    result = predictor._materialize_layer_ep_workload(
        batch=_Batch(),
        cluster_type=ClusterType.PREFILL,
        layer_id=7,
    )

    assert dict(result.global_per_expert_tokens) == {0: 2, 1: 2, 2: 3, 3: 1}
    assert sum(result.global_per_expert_tokens.values()) == 8


def _decode_step(*, padded_total_tokens: int | None, num_rows: int = 9) -> Batch:
    """One-token decode rows; with a FULL decode graph the step runs at the capture size."""
    requests = [
        Request(arrived_at=0.0, num_prefill_tokens=1, num_decode_tokens=1)
        for _ in range(num_rows)
    ]
    batch = Batch(3, requests, [1] * num_rows, is_moe=True)
    if padded_total_tokens is not None:
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            config_mode="full_decode_only",
            runtime_mode="FULL",
            capture_hit=False,
            is_mixed_batch=False,
            original_total_tokens=num_rows,
            padded_total_tokens=padded_total_tokens,
            original_decode_batch_size=num_rows,
            padded_decode_batch_size=padded_total_tokens,
        )
    return batch


@pytest.mark.parametrize(
    ("padded_total_tokens", "routed_tokens"), ((None, 9), (16, 16))
)
def test_routed_width_follows_decode_graph_padding_at_both_sites(
    padded_total_tokens, routed_tokens
) -> None:
    predictor = _MoEPredictor(replica_id=3, total_experts=4, ep_size=2)
    predictor._monolithic_routing_details = {
        3: {7: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}}
    }
    scheduler = SimpleNamespace(
        _config=SimpleNamespace(replica_config=predictor._replica_config),
        _cluster_type=ClusterType.MONOLITHIC,
        _predictor=predictor,
    )
    batch = _decode_step(padded_total_tokens=padded_total_tokens)

    workloads = (
        predictor._materialize_layer_ep_workload(
            batch=batch, cluster_type=ClusterType.MONOLITHIC, layer_id=7
        ),
        materialize_layer_workload(
            scheduler=scheduler, batch=batch, target_replica_id=3, global_layer_id=7
        ),
    )

    for workload in workloads:
        assert workload.routing_token_count == routed_tokens
        assert sum(workload.global_per_expert_tokens.values()) == 2 * routed_tokens


def _dummy_lane() -> DummyForwardBatch:
    return DummyForwardBatch(replica_id=3, pipeline_stage_id=0, forward_index=0)


# Two attention-DP lanes of one forward. With decode graphs, vLLM pads each
# lane to its capture size and then every DP rank, a dummy pass included, to
# the largest one; eager lanes run their own tokens.
DP_COHORTS = {
    "graph_9_and_9": (
        lambda: _decode_step(padded_total_tokens=16),
        lambda: _decode_step(padded_total_tokens=16),
        32,
    ),
    "graph_9_and_3": (
        lambda: _decode_step(padded_total_tokens=16),
        lambda: _decode_step(padded_total_tokens=4, num_rows=3),
        32,
    ),
    "graph_9_and_dummy": (lambda: _decode_step(padded_total_tokens=16), _dummy_lane, 32),
    "eager_9_and_9": (
        lambda: _decode_step(padded_total_tokens=None),
        lambda: _decode_step(padded_total_tokens=None),
        18,
    ),
    "eager_9_and_dummy": (lambda: _decode_step(padded_total_tokens=None), _dummy_lane, 10),
}


@pytest.mark.parametrize("name", DP_COHORTS)
def test_a_dp_cohort_routes_its_padded_compute_width(name) -> None:
    first_lane, second_lane, routed_tokens = DP_COHORTS[name]
    lanes = {0: first_lane(), 1: second_lane()}
    predictor = _MoEPredictor(replica_id=3, total_experts=4, ep_size=2)
    predictor._monolithic_routing_details = {
        3: {7: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}}
    }
    scheduler = SimpleNamespace(
        _config=SimpleNamespace(replica_config=predictor._replica_config),
        _cluster_type=ClusterType.MONOLITHIC,
        _predictor=predictor,
    )

    wave_inputs = prepare_ep_wave_inputs(
        source_batches=lanes,
        batch=lanes[0],
        step_id_getter=lambda _batch: 0,
        aggregate_batch_builder=build_virtual_global_batch,
        cluster_type=ClusterType.MONOLITHIC,
    )
    workloads = (
        predictor._materialize_layer_ep_workload(
            batch=wave_inputs.aggregate_batch,
            cluster_type=ClusterType.MONOLITHIC,
            layer_id=7,
        ),
        materialize_layer_workload(
            scheduler=scheduler,
            batch=wave_inputs.aggregate_batch,
            target_replica_id=3,
            global_layer_id=7,
        ),
    )

    for workload in workloads:
        assert workload.routing_token_count == routed_tokens
        assert workload.total_routed_assignments == 2 * routed_tokens
    # The lanes keep their request tokens.
    assert [lane.total_num_tokens for lane in lanes.values()] == [
        lane.total_num_tokens for lane in (first_lane(), second_lane())
    ]


def test_moe_layer_prediction_has_no_one_token_conservation_tolerance() -> None:
    source = _predictor_package_source()

    assert "abs(total_allocated_tokens - expected_tokens) > 1" not in source


def test_decode_sync_collective_has_no_uniform_routing_fallback() -> None:
    source = Path(
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "_build_uniform_per_expert_tokens" not in source


def test_moe_predictor_has_one_routing_integerizer() -> None:
    source = _predictor_package_source()

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
    replica_state_source = Path(
        "frontier/scheduler/utils/replica_schedulers.py"
    ).read_text(encoding="utf-8")
    round_robin_source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "self._replica_ep_size = int(" in source
    assert "self._replica_dp_size" in source
    assert "_replica_dp_size" in round_robin_source
    assert "Use ep_id as dp_id for compatibility" not in source
    assert "replica_local_id=ep_id" in replica_state_source


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
    collective_source = Path(
        "frontier/scheduler/utils/decode_collective.py"
    ).read_text(encoding="utf-8")

    # The predictor-only aggregate helper is part of the canonical shared-DP
    # EP wave path; legacy scalar DP synchronization remains removed below.
    assert "def _create_virtual_global_batch" in source
    assert "def _get_decode_sync_participant_count" not in source
    assert "predict_dp_gather_time" not in source
    assert "predict_dp_scatter_time" not in source
    assert "Legacy DECODE aggregate synchronization is removed" in collective_source
