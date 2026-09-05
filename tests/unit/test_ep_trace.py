from types import SimpleNamespace

import pytest

from frontier.scheduler.utils import ep_trace


def _request(request_id: int, *, epoch: int = 0, token_index: int = 1):
    return SimpleNamespace(
        id=request_id,
        runtime_epoch=epoch,
        current_decode_token_index=token_index,
    )


def _batch(*requests, layer_id: int = 3, source_ids=(17,)):
    return SimpleNamespace(
        requests=list(requests),
        source_batch_ids=list(source_ids),
        decode_ffn_layer_id=layer_id,
    )


def test_resolve_trace_identity_uses_single_source_batch_id() -> None:
    lanes = {
        0: _batch(_request(1), source_ids=(42,)),
        1: _batch(_request(2), source_ids=(42,)),
    }

    assert ep_trace.resolve_trace_identity(lanes, 99) == (42, 3)


def test_resolve_trace_identity_keeps_wave_id_for_merged_source_batches() -> None:
    lanes = {
        0: _batch(_request(1), source_ids=(42, 43)),
        1: _batch(_request(2), source_ids=(42, 43)),
    }

    assert ep_trace.resolve_trace_identity(lanes, 99) == (99, 3)


def test_build_trace_identity_expands_source_requests_and_epochs() -> None:
    source = SimpleNamespace(
        requests=[_request(7, epoch=2, token_index=4)],
        request_runtime_epochs=[2],
    )
    batch = SimpleNamespace(
        source_batches=[source],
        schedule_epoch=5,
        afd_stage_idx=1,
    )

    identity = ep_trace.build_trace_identity(
        batch=batch,
        replica_id=2,
        stage_id=1,
        operation_id=9,
        operation_kind=" ep_ffn ",
    )

    assert identity == {
        "replica_id": 2,
        "stage_id": 1,
        "request_ids": (7,),
        "request_runtime_epochs": (2,),
        "iteration_ids": (3,),
        "schedule_epoch": 5,
        "afd_stage_idx": 1,
        "operation_id": 9,
        "operation_kind": "ep_ffn",
    }


def test_build_trace_identity_rejects_duplicate_request_ids() -> None:
    batch = SimpleNamespace(requests=[_request(1), _request(1)])

    with pytest.raises(ValueError, match="request_ids must be unique"):
        ep_trace.build_trace_identity(
            batch=batch,
            replica_id=0,
            stage_id=0,
            operation_id=0,
            operation_kind="ep_ffn",
        )


def test_workload_trace_accepts_formatter_callback(caplog) -> None:
    from frontier.moe_ep_workload import EPLaneWorkload
    from frontier.types import ClusterType

    lane = EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=1,
        total_expert_num=1,
        owned_expert_ids=(0,),
        local_token_counts=(1,),
        routed_token_count=1,
        router_topk=1,
    )

    ep_trace.log_workload_trace(
        cluster_type=ClusterType.DECODE_FFN,
        batch_id=1,
        layer_id=2,
        lane_workload=lane,
        lane_compute_ms=1.0,
        routed_compute_ms=1.0,
        lane_comm_ms=0.0,
        pre_dispatch_ms=0.0,
        dispatch_ms=0.0,
        combine_ms=0.0,
        post_combine_ms=0.0,
        trace_identity={"identity": "test"},
        format_identity=lambda value: f"formatted={value['identity']}",
    )

    assert "formatted=test" in caplog.records[-1].message
