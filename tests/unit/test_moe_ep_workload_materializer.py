from __future__ import annotations

from types import MappingProxyType

import pytest

from frontier.moe_ep_workload import (
    EPLaneWorkload,
    LayerEPWorkload,
    materialize_layer_ep_workload,
    resolve_routing_details,
    split_global_expert_tokens_into_lanes,
)


def _ownership(total_experts: int = 4, ep_size: int = 2) -> dict[int, int]:
    experts_per_ep = total_experts // ep_size
    return {
        expert_id: expert_id // experts_per_ep
        for expert_id in range(total_experts)
    }


def test_materializer_uses_one_global_hamilton_pass_with_lower_id_tie_break() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={
            0: 0.1875,
            1: 0.3125,
            2: 0.375,
            3: 0.125,
        },
        target_replica_id=3,
        global_layer_id=7,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep=_ownership(),
    )

    assert isinstance(workload, LayerEPWorkload)
    assert dict(workload.global_per_expert_tokens) == {0: 2, 1: 2, 2: 3, 3: 1}
    assert dict(workload.per_ep_routed_tokens) == {0: 4, 1: 4}
    assert workload.total_routed_assignments == 8
    assert sum(workload.per_ep_routed_tokens.values()) == 8


def test_materializer_normalizes_positive_sum_once_without_scaling_latency() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 2.0, 1: 1.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=3,
        router_topk=1,
        total_expert_num=2,
        moe_expert_parallel_size=1,
        expert_to_ep={0: 0, 1: 0},
    )

    assert dict(workload.global_per_expert_tokens) == {0: 2, 1: 1}
    assert workload.routing_token_count == 3
    assert workload.total_routed_assignments == 3


def test_zero_tokens_preserve_all_experts_and_all_ep_participants() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        target_replica_id=1,
        global_layer_id=2,
        routing_token_count=0,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep=_ownership(),
    )

    assert dict(workload.global_per_expert_tokens) == {0: 0, 1: 0, 2: 0, 3: 0}
    assert dict(workload.per_ep_per_expert_tokens) == {
        0: {0: 0, 1: 0},
        1: {2: 0, 3: 0},
    }
    assert workload.participant_ep_ids == (0, 1)
    assert dict(workload.per_ep_routed_tokens) == {0: 0, 1: 0}


def test_layer_workload_exposes_typed_local_lane_identity() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        target_replica_id=2,
        global_layer_id=5,
        routing_token_count=4,
        router_topk=1,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep=_ownership(),
    )

    lane = workload.lane(1)

    assert isinstance(lane, EPLaneWorkload)
    assert lane.ep_id == 1
    assert lane.local_expert_width == 2
    assert lane.owned_expert_ids == (2, 3)
    assert lane.local_token_counts == (1, 1)
    assert dict(lane.per_expert_tokens) == {2: 1, 3: 1}
    assert not hasattr(lane, "source_batch_ids")
    assert not hasattr(lane, "target_replica_id")
    assert not hasattr(lane, "global_layer_id")


def test_lane_rejects_non_conserving_local_counts() -> None:
    with pytest.raises(ValueError, match="routed_token_count"):
        EPLaneWorkload(
            ep_id=0,
            moe_expert_parallel_size=1,
            total_expert_num=2,
            owned_expert_ids=(0, 1),
            local_token_counts=(1, 0),
            routed_token_count=2,
            router_topk=1,
        )


def test_global_map_splitter_preserves_global_ids_and_token_conservation() -> None:
    lanes = split_global_expert_tokens_into_lanes(
        {0: 3, 1: 0, 2: 4, 3: 1},
        total_expert_num=4,
        moe_expert_parallel_size=2,
    )

    assert tuple(lane.ep_id for lane in lanes) == (0, 1)
    assert [dict(lane.per_expert_tokens) for lane in lanes] == [
        {0: 3, 1: 0},
        {2: 4, 3: 1},
    ]
    assert sum(sum(lane.per_expert_tokens.values()) for lane in lanes) == 8


def test_materializer_result_and_nested_maps_are_immutable() -> None:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 1.0, 1: 1.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=2,
        router_topk=1,
        total_expert_num=2,
        moe_expert_parallel_size=1,
        expert_to_ep={0: 0, 1: 0},
    )

    assert isinstance(workload.global_per_expert_tokens, MappingProxyType)
    with pytest.raises(TypeError):
        workload.global_per_expert_tokens[0] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        workload.per_ep_per_expert_tokens[0][0] = 99  # type: ignore[index]


def test_resolver_requires_exact_replica_and_global_layer_keys() -> None:
    routing_details = {0: {4: {0: 1.0, 1: 1.0}}}

    assert resolve_routing_details(routing_details, 0, 4) == {0: 1.0, 1: 1.0}
    with pytest.raises(ValueError, match="target_replica_id"):
        resolve_routing_details(routing_details, 1, 4)
    with pytest.raises(ValueError, match="global_layer_id"):
        resolve_routing_details(routing_details, 0, 5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"routing_token_count": -1}, "routing_token_count"),
        ({"router_topk": 0}, "router_topk"),
        ({"total_expert_num": 0}, "total_expert_num"),
        ({"moe_expert_parallel_size": 0}, "moe_expert_parallel_size"),
    ],
)
def test_materializer_rejects_invalid_integer_inputs(kwargs, message: str) -> None:
    base = {
        "routing_ratios": {0: 1.0, 1: 1.0},
        "target_replica_id": 0,
        "global_layer_id": 0,
        "routing_token_count": 2,
        "router_topk": 1,
        "total_expert_num": 2,
        "moe_expert_parallel_size": 1,
        "expert_to_ep": {0: 0, 1: 0},
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        materialize_layer_ep_workload(**base)


@pytest.mark.parametrize(
    "ratios, message",
    [
        ({0: -1.0, 1: 2.0}, "non-negative"),
        ({0: float("nan"), 1: 1.0}, "finite"),
        ({0: 0.0, 1: 0.0}, "positive sum"),
        ({0: 1.0}, "expert key set"),
        ({0: 1.0, 1: 1.0, 2: 1.0}, "expert key set"),
    ],
)
def test_materializer_rejects_invalid_routing_ratios(ratios, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_layer_ep_workload(
            routing_ratios=ratios,
            target_replica_id=0,
            global_layer_id=0,
            routing_token_count=2,
            router_topk=1,
            total_expert_num=2,
            moe_expert_parallel_size=1,
            expert_to_ep={0: 0, 1: 0},
        )


def test_materializer_rejects_non_contiguous_or_incomplete_ownership() -> None:
    with pytest.raises(ValueError, match="ownership"):
        materialize_layer_ep_workload(
            routing_ratios={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            target_replica_id=0,
            global_layer_id=0,
            routing_token_count=2,
            router_topk=1,
            total_expert_num=4,
            moe_expert_parallel_size=2,
            expert_to_ep={0: 0, 1: 0, 2: 1},
        )

    with pytest.raises(ValueError, match="contiguous equal-size"):
        materialize_layer_ep_workload(
            routing_ratios={0: 1.0, 1: 1.0, 2: 1.0},
            target_replica_id=0,
            global_layer_id=0,
            routing_token_count=2,
            router_topk=1,
            total_expert_num=3,
            moe_expert_parallel_size=2,
            expert_to_ep={0: 0, 1: 0, 2: 1},
        )
