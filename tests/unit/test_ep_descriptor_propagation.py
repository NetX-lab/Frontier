from __future__ import annotations

from types import MappingProxyType

import pytest

from frontier.entities import Request
from frontier.entities.batch import EPBatchGroup
from frontier.entities.batch_stage import BatchStage
from frontier.moe_ep_workload import EPLaneWorkload, materialize_layer_ep_workload
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    EPBatchGroupPlan,
)
from frontier.types import ClusterType


def _lane() -> EPLaneWorkload:
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        target_replica_id=0,
        global_layer_id=2,
        routing_token_count=2,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
    )
    return workload.lane(1)


def test_ep_batch_group_plan_carries_typed_lane_descriptor() -> None:
    lane = _lane()

    plan = EPBatchGroupPlan(
        replica_id=0,
        ep_id=1,
        layer_global_id=2,
        afd_stage_idx=0,
        group_time=1.0,
        pre_routing_effective_total_tokens=2,
        source_batches=(),
        source_batch_ids=(7,),
        lane_workload=lane,
    )

    assert plan.lane_workload is lane
    assert plan.per_expert_tokens == tuple(lane.per_expert_tokens.items())
    assert not hasattr(plan, "_per_expert_tokens")


def test_ep_batch_group_stores_descriptor_and_exposes_read_only_projection() -> None:
    lane = _lane()
    batch = EPBatchGroup(
        requests=[Request(0.0, 0, 2)],
        num_tokens=[2],
        replica_id=0,
        ep_id=lane.ep_id,
        time=0.0,
        source_batch_ids=[7],
        lane_workload=lane,
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )

    assert batch.lane_workload is lane
    assert isinstance(batch.per_expert_tokens, MappingProxyType)
    assert dict(batch.per_expert_tokens) == dict(lane.per_expert_tokens)
    with pytest.raises(TypeError):
        batch.per_expert_tokens[2] = 99  # type: ignore[index]


def test_batch_stage_keeps_descriptor_without_copying_a_mutable_map() -> None:
    lane = _lane()
    stage = BatchStage(
        batch_id=7,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.1,
        model_execution_time=0.1,
        requests=[Request(0.0, 0, 2)],
        num_tokens=[2],
        cluster_type=ClusterType.DECODE_FFN,
    )

    stage.attach_lane_workload(lane)

    assert stage.lane_workload is lane
    assert not hasattr(stage, "per_expert_tokens")
