from types import SimpleNamespace

import pytest

from frontier.moe_ep_workload import materialize_layer_ep_workload
from frontier.scheduler.utils.expert_parallel import (
    materialize_wave_workload,
    validate_token_conservation,
)


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

