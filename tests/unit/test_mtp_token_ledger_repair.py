from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request, SpecDecodeBatchMetadata
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.moe_ep_workload import materialize_layer_ep_workload
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class _DummyPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=8,
        num_decode_tokens=16,
        num_processed_tokens=8,
    )
    request._is_prefill_complete = True
    return request


def _target_embedded_batch() -> Batch:
    batch = Batch(
        replica_id=0,
        requests=[_request(), _request()],
        num_tokens=[3, 2],
        is_moe=True,
    )
    batch.spec_decode_metadata = SpecDecodeBatchMetadata(
        method="qwen3_moe_mtp",
        planned_draft_tokens_per_request=[2, 1],
        verify_tokens_per_request=[3, 2],
        accepted_draft_tokens_per_request=[0, 0],
        rejected_draft_tokens_per_request=[2, 1],
        committed_tokens_per_request=[1, 1],
        uses_lookahead_slots=True,
    )
    return batch


def test_target_embedded_mtp_compute_width_uses_physical_batch_rows() -> None:
    batch = _target_embedded_batch()

    assert batch.total_num_tokens == 5
    assert batch.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC) == 5
    assert batch.get_effective_total_tokens_for_transfer(ClusterType.MONOLITHIC) == 5


def test_structural_mtp_first_block_keeps_complete_verification_shape() -> None:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._replica_config = SimpleNamespace(
        model_name="target-model",
        attn_tensor_parallel_size=1,
        num_pipeline_stages=1,
        speculative_decoding_config=SimpleNamespace(
            spec_model_name="",
            mtp_n_predict=2,
            mtp_num_layers=1,
        ),
    )
    predictor._model_config = SimpleNamespace(is_moe=False)
    predictor._cluster_type = ClusterType.MONOLITHIC
    captured_batches: list[Batch] = []

    def _capture_structural_step(
        *, predictor, contract, synthetic_batch, active_request_count
    ):
        del predictor, contract, active_request_count
        captured_batches.append(synthetic_batch)
        return 7.0

    predictor._predict_mtp_structural_step_time_ms = _capture_structural_step

    result = predictor._get_structural_mtp_proposer_time(
        _target_embedded_batch(),
        method_name="qwen3_moe_mtp",
    )

    assert result == pytest.approx(7.0)
    assert len(captured_batches) == 1
    assert captured_batches[0].num_tokens == [3, 2]
    assert captured_batches[0].spec_decode_metadata is not None
    assert captured_batches[0].spec_decode_metadata.verify_tokens_per_request == [3, 2]


def test_moe_materializer_conserves_pre_routing_tokens_and_topk_assignments() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=5,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
    )

    assert workload.routing_token_count == 5
    assert workload.total_routed_assignments == 10
    assert sum(workload.per_ep_routed_tokens.values()) == 10
    assert sum(workload.lane(ep_id).routed_token_count for ep_id in (0, 1)) == 10


@pytest.mark.parametrize(
    ("planned_drafts", "expected_next_width"),
    [(0, 1), (1, 1), (2, 2), (4, 4)],
)
def test_monolithic_mtp_prefill_boundary_schedules_remaining_draft_slots(
    planned_drafts: int,
    expected_next_width: int,
) -> None:
    scheduler = VLLMv1EngineReplicaScheduler.__new__(
        VLLMv1EngineReplicaScheduler
    )
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._scheduled_num_computed_tokens_by_request = {}

    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=8,
        num_decode_tokens=32,
        num_processed_tokens=9,
    )
    request._is_prefill_complete = True
    request.initialize_spec_decode_state(
        enabled=True,
        method="qwen3_moe_mtp",
        num_speculative_tokens=planned_drafts,
        method_uses_lookahead_slots=True,
    )
    request.set_spec_next_planned_draft_tokens(planned_drafts)

    assert request.num_processed_decode_tokens == 1
    assert scheduler._get_scheduler_num_computed_tokens(request) == 8
    assert scheduler._get_request_next_num_tokens(request) == expected_next_width


def test_monolithic_mtp_boundary_returns_full_width_after_frontier_advances() -> None:
    scheduler = VLLMv1EngineReplicaScheduler.__new__(
        VLLMv1EngineReplicaScheduler
    )
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._scheduled_num_computed_tokens_by_request = {}

    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=8,
        num_decode_tokens=32,
        num_processed_tokens=10,
    )
    request._is_prefill_complete = True
    request.initialize_spec_decode_state(
        enabled=True,
        method="qwen3_moe_mtp",
        num_speculative_tokens=2,
        method_uses_lookahead_slots=True,
    )
    request.set_spec_next_planned_draft_tokens(2)

    assert scheduler._get_scheduler_num_computed_tokens(request) == 9
    assert scheduler._get_request_next_num_tokens(request) == 3
