from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.entities import EPBatchGroup, Request
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def _lane_batch(*, ep_id: int, local_token_counts: tuple[int, int]) -> EPBatchGroup:
    lane_workload = EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(ep_id * 2, ep_id * 2 + 1),
        local_token_counts=local_token_counts,
        routed_token_count=sum(local_token_counts),
        router_topk=2,
    )
    return EPBatchGroup(
        requests=[Request(0.0, 0, sum(local_token_counts))],
        num_tokens=[sum(local_token_counts)],
        replica_id=0,
        ep_id=ep_id,
        time=0.0,
        source_batch_ids=[7],
        lane_workload=lane_workload,
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )


def test_scheduler_resolves_execution_time_from_typed_lane_descriptor() -> None:
    zero_lane = _lane_batch(ep_id=0, local_token_counts=(0, 0))
    zero_lane.execution_time = 0.0
    active_lane = _lane_batch(ep_id=1, local_token_counts=(1, 0))
    active_lane.execution_time = 0.75

    assert BaseClusterScheduler._resolve_ep_execution_time(
        {0: zero_lane, 1: active_lane}
    ) == pytest.approx(0.75)


def test_scheduler_rejects_ep_entity_without_typed_lane_descriptor() -> None:
    legacy_lane = SimpleNamespace(execution_time=0.0, per_expert_tokens={0: 0})

    with pytest.raises(ValueError, match="EPLaneWorkload descriptor"):
        BaseClusterScheduler._resolve_ep_execution_time({0: legacy_lane})


def _dispatch_scheduler() -> _ConcreteClusterScheduler:
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            moe_expert_parallel_size=2,
            model_config=SimpleNamespace(embedding_dim=16),
        )
    )
    scheduler.get_replica = Mock(return_value=SimpleNamespace(ep_size=2))
    scheduler._predictor = Mock()
    scheduler._predictor.predict_alltoall_time.return_value = 7.0
    scheduler._ep_alltoall_dispatch_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"batches": {}, "arrival_times": {}}
            )
        )
    )
    scheduler._ep_allgather_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"batches": {}, "arrival_times": {}}
            )
        )
    )
    return scheduler


def _dispatch_lane(*, ep_id: int, typed: bool = True) -> object:
    lane = _lane_batch(ep_id=ep_id, local_token_counts=(1, 0))
    lane.set_global_id(77)
    lane.routing_token_count = 1
    lane.router_topk = 2
    lane.total_routed_assignments = 2
    lane.decode_ffn_layer_id = 0
    if typed:
        return lane
    return SimpleNamespace(
        id=lane.id,
        global_id=lane.global_id,
        ep_id=lane.ep_id,
        replica_id=lane.replica_id,
        total_num_tokens=lane.total_num_tokens,
        routing_token_count=lane.routing_token_count,
        router_topk=lane.router_topk,
        total_routed_assignments=lane.total_routed_assignments,
        source_batch_ids=list(lane.source_batch_ids),
        requests=list(lane.requests),
        request_runtime_epochs=list(lane.request_runtime_epochs),
        schedule_epoch=lane.schedule_epoch,
        afd_stage_idx=0,
        decode_ffn_layer_id=lane.decode_ffn_layer_id,
    )


@pytest.mark.parametrize(
    ("method_name", "waiting_room_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        ("on_ep_alltoall_combine_ready", "_ep_allgather_waiting_room"),
    ],
)
def test_ep_collective_rejects_missing_lane_descriptor_before_lookup(
    method_name: str,
    waiting_room_attr: str,
) -> None:
    scheduler = _dispatch_scheduler()
    first_lane = _dispatch_lane(ep_id=0)
    missing_descriptor_lane = _dispatch_lane(ep_id=1, typed=False)

    collective_ready = getattr(scheduler, method_name)
    assert collective_ready(1.0, 0, 0, first_lane, 0) == []

    with pytest.raises(ValueError, match="EPLaneWorkload descriptor"):
        collective_ready(2.0, 0, 0, missing_descriptor_lane, 1)

    scheduler._predictor.predict_alltoall_time.assert_not_called()
    waiting_rooms = getattr(scheduler, waiting_room_attr)
    assert set(waiting_rooms[0][0][77]["batches"]) == {0}


@pytest.mark.parametrize(
    ("method_name", "waiting_room_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        ("on_ep_alltoall_combine_ready", "_ep_allgather_waiting_room"),
    ],
)
def test_ep_collective_rejects_lane_entity_width_mismatch_before_lookup(
    method_name: str,
    waiting_room_attr: str,
) -> None:
    scheduler = _dispatch_scheduler()
    first_lane = _dispatch_lane(ep_id=0)
    mismatched_lane = _dispatch_lane(ep_id=1)
    mismatched_lane._total_num_tokens = 2

    collective_ready = getattr(scheduler, method_name)
    assert collective_ready(1.0, 0, 0, first_lane, 0) == []

    with pytest.raises(ValueError, match="total_num_tokens.*routed_token_count"):
        collective_ready(2.0, 0, 0, mismatched_lane, 1)

    scheduler._predictor.predict_alltoall_time.assert_not_called()
    waiting_rooms = getattr(scheduler, waiting_room_attr)
    assert set(waiting_rooms[0][0][77]["batches"]) == {0}
