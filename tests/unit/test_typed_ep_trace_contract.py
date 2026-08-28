import logging

from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


def _trace_identity() -> dict[str, object]:
    return {
        "replica_id": 0,
        "stage_id": 1,
        "request_ids": (11,),
        "request_runtime_epochs": (2,),
        "iteration_ids": (3,),
        "schedule_epoch": 4,
        "afd_stage_idx": -1,
        "operation_id": 7,
        "operation_kind": "ep_ffn",
    }


def test_ep_trace_helper_consumes_typed_lane_descriptor(caplog) -> None:
    lane_workload = EPLaneWorkload(
        ep_id=1,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(2, 3),
        local_token_counts=(0, 2),
        routed_token_count=2,
        router_topk=1,
    )

    with caplog.at_level(
        logging.INFO,
        logger="frontier.scheduler.cluster_scheduler.base_cluster_scheduler",
    ):
        BaseClusterScheduler._log_ep_workload_trace(
            cluster_type=ClusterType.DECODE_FFN,
            batch_id=7,
            layer_id=3,
            lane_workload=lane_workload,
            lane_compute_ms=3.0,
            routed_compute_ms=2.0,
            lane_comm_ms=1.0,
            pre_dispatch_ms=1.0,
            dispatch_ms=0.5,
            combine_ms=0.5,
            post_combine_ms=0.0,
            trace_identity=_trace_identity(),
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "ep_id=1" in message
    assert "moe_ep_size=2" in message
    assert "per_expert_tokens={2: 0, 3: 2}" in message
