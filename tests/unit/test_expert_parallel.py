from types import SimpleNamespace

import pytest

from frontier.moe_ep_workload import materialize_layer_ep_workload
from frontier.entities import EPBatchGroup, Request
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.utils.expert_parallel import (
    materialize_wave_workload,
    prepare_combine_timing,
    summarize_alltoall_payload,
    validate_barrier_arrival,
    validate_token_conservation,
)
from frontier.types import ClusterType
from frontier.model_architectures import ExpertParallelCollective


def _routing_details():
    return {0: {3: {0: 0.25, 1: 0.75, 2: 0.0, 3: 0.0}}}


def test_materialize_wave_workload_aggregates_source_tokens():
    batches = [
        (SimpleNamespace(total_num_tokens=2), None),
        (SimpleNamespace(total_num_tokens=3), None),
    ]
    workload = materialize_wave_workload(
        batches,
        0,
        3,
        _routing_details(),
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=1,
    )
    assert workload.routing_token_count == 5
    assert workload.participant_ep_ids == (0, 1)
    assert sum(workload.per_ep_routed_tokens.values()) == 5


def test_validate_token_conservation_rejects_mismatched_lane():
    batches = [(SimpleNamespace(total_num_tokens=2), None)]
    workload = materialize_wave_workload(
        batches,
        0,
        3,
        _routing_details(),
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=1,
    )
    lane = workload.lane(0)
    with pytest.raises(ValueError, match="Token conservation violated"):
        validate_token_conservation(lane.routed_token_count + 1, lane, "unit")


def _dispatch_batch(ep_id: int, global_id: int = 4) -> EPBatchGroup:
    lane = EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(ep_id * 2, ep_id * 2 + 1),
        local_token_counts=(ep_id + 1, 0),
        routed_token_count=ep_id + 1,
        router_topk=1,
    )
    batch = EPBatchGroup(
        requests=[Request(0.0, 0, ep_id + 1)],
        num_tokens=[ep_id + 1],
        replica_id=0,
        ep_id=ep_id,
        time=0.0,
        source_batch_ids=[global_id],
        lane_workload=lane,
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    batch.set_global_id(global_id)
    return batch


def test_validate_barrier_arrival_reports_completion_without_mutation():
    rooms = {0: {0: {}}}
    batch = _dispatch_batch(0)
    result = validate_barrier_arrival(
        phase="dispatch",
        waiting_rooms=rooms,
        get_replica=lambda _: SimpleNamespace(ep_size=2),
        default_ep_size=2,
        replica_id=0,
        stage_id=0,
        batch=batch,
        ep_id=0,
    )
    assert result[0] == 4
    assert result[1] is None
    assert result[2] == frozenset({0, 1})
    assert result[3] is False
    assert rooms == {0: {0: {}}}


def test_validate_barrier_arrival_rejects_duplicate_lane():
    batch = _dispatch_batch(0)
    rooms = {0: {0: {4: {"batches": {0: batch}, "arrival_times": {0: 1.0}}}}}
    with pytest.raises(ValueError, match="duplicate ep_id arrival"):
        validate_barrier_arrival(
            phase="dispatch",
            waiting_rooms=rooms,
            get_replica=lambda _: SimpleNamespace(ep_size=2),
            default_ep_size=2,
            replica_id=0,
            stage_id=0,
            batch=batch,
            ep_id=0,
        )


def test_summarize_alltoall_payload_uses_largest_lane():
    batches = {0: _dispatch_batch(0), 1: _dispatch_batch(1)}
    payload = summarize_alltoall_payload(batches, hidden_size=16)
    assert payload == (64, {0: 1, 1: 2}, 2, 16)


def test_prepare_combine_timing_returns_collective_and_post_compute_times():
    batches = {0: _dispatch_batch(0), 1: _dispatch_batch(1)}
    batches[0].post_combine_time = 0.2
    batches[1].post_combine_time = 0.5
    timing = prepare_combine_timing(
        prospective_batches=batches,
        prospective_arrival_times={0: 1.0, 1: 1.5},
        expected_ep_size=2,
        collective_kind=ExpertParallelCollective.ALLTOALL,
        cluster_type=ClusterType.DECODE_FFN,
        hidden_size=16,
        predict_alltoall=lambda **_: 4.0,
        predict_allgather=lambda **_: 9.0,
    )
    assert timing.sync_time == pytest.approx(1.5)
    assert timing.exec_time_ms == pytest.approx(4.0)
    assert timing.combine_end_time == pytest.approx(1.504)
    assert timing.final_event_time == pytest.approx(2.004)
    assert timing.data_size_bytes == 64
