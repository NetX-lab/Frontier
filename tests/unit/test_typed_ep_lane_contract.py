from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from frontier.moe_ep_workload import (
    EPLaneWorkload,
    materialize_layer_ep_workload,
    split_global_expert_tokens_into_lanes,
)


def _ownership(total_experts: int, ep_size: int) -> dict[int, int]:
    experts_per_lane = total_experts // ep_size
    return {
        expert_id: expert_id // experts_per_lane
        for expert_id in range(total_experts)
    }


def _materialize(*, total_experts: int, ep_size: int, tokens: int = 4):
    return materialize_layer_ep_workload(
        routing_ratios={expert_id: 1.0 for expert_id in range(total_experts)},
        target_replica_id=0,
        global_layer_id=3,
        routing_token_count=tokens,
        router_topk=2,
        total_expert_num=total_experts,
        moe_expert_parallel_size=ep_size,
        expert_to_ep=_ownership(total_experts, ep_size),
    )


def test_lane_descriptor_contains_only_fixed_width_physical_workload() -> None:
    workload = _materialize(total_experts=4, ep_size=2, tokens=0)

    lane = workload.lane(1)

    assert isinstance(lane, EPLaneWorkload)
    assert lane.ep_id == 1
    assert lane.moe_expert_parallel_size == 2
    assert lane.total_expert_num == 4
    assert lane.local_expert_width == 2
    assert lane.owned_expert_ids == (2, 3)
    assert lane.local_token_counts == (0, 0)
    assert lane.routed_token_count == 0
    assert lane.router_topk == 2
    assert dict(lane.per_expert_tokens) == {2: 0, 3: 0}

    for forbidden_name in (
        "source_batch_id",
        "source_batch_ids",
        "target_replica_id",
        "global_layer_id",
        "validate_expert_width",
    ):
        assert not hasattr(lane, forbidden_name)


def test_ep_one_and_ep_many_use_the_same_descriptor_contract() -> None:
    ep_one_workload = _materialize(total_experts=4, ep_size=1)
    ep_many_workload = _materialize(total_experts=4, ep_size=2)
    ep_one = ep_one_workload.lane(0)
    ep_many = ep_many_workload.lane(0)

    assert ep_one_workload.participant_ep_ids == (0,)
    assert ep_one.local_expert_width == ep_one.total_expert_num
    assert ep_many_workload.participant_ep_ids == (0, 1)
    assert ep_many.local_expert_width == ep_many.total_expert_num // 2
    assert isinstance(ep_one.local_token_counts, tuple)
    assert isinstance(ep_many.local_token_counts, tuple)


def test_lane_descriptor_is_immutable() -> None:
    lane = _materialize(total_experts=4, ep_size=2).lane(0)

    with pytest.raises(FrozenInstanceError):
        lane.ep_id = 1  # type: ignore[misc]


def test_lane_descriptor_rejects_out_of_owner_ids() -> None:
    with pytest.raises(ValueError, match="owned_expert_ids"):
        EPLaneWorkload(
            ep_id=0,
            moe_expert_parallel_size=2,
            total_expert_num=4,
            owned_expert_ids=(0, 2),
            local_token_counts=(1, 1),
            routed_token_count=2,
            router_topk=1,
        )


def test_global_splitter_requires_complete_global_domain() -> None:
    with pytest.raises(ValueError, match="every global expert"):
        split_global_expert_tokens_into_lanes(
            {0: 1, 1: 1, 2: 0},
            total_expert_num=4,
            moe_expert_parallel_size=2,
            router_topk=1,
        )


def test_global_splitter_preserves_fixed_width_and_token_conservation() -> None:
    lanes = split_global_expert_tokens_into_lanes(
        {0: 3, 1: 0, 2: 4, 3: 1},
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=2,
    )

    assert [lane.local_token_counts for lane in lanes] == [(3, 0), (4, 1)]
    assert [lane.routed_token_count for lane in lanes] == [3, 5]
    assert sum(lane.routed_token_count for lane in lanes) == 8
