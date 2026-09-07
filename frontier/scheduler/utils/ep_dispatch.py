"""Planning and event-entry helpers for EP dispatch collectives."""

from __future__ import annotations

from typing import Any, NamedTuple
import math
from numbers import Real

from frontier.moe_ep_workload import resolve_ep_lane_workload


class EPDispatchAdvance(NamedTuple):
    """One validated EP lane ready to enter expert execution."""

    ep_id: int
    batch: Any
    ready_time: float


def prepare_dispatch_advance(
    *,
    ep_batches: dict[int, Any],
    time: float,
) -> tuple[EPDispatchAdvance, ...]:
    """Validate dispatch output and calculate per-lane expert ready times."""

    if not ep_batches:
        raise ValueError("EP dispatch collective reached with empty ep_batches")
    prepared: list[EPDispatchAdvance] = []
    for ep_id, batch in ep_batches.items():
        expert_compute_time = getattr(batch, "expert_compute_time", None)
        if expert_compute_time is None:
            raise ValueError(
                f"Missing expert_compute_time for EP batch {batch.id} "
                f"(ep_id={ep_id})"
            )
        prepared.append(
            EPDispatchAdvance(
                ep_id=int(ep_id),
                batch=batch,
                ready_time=float(time + expert_compute_time),
            )
        )
    return tuple(prepared)


def handle_dispatch_ready(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    ep_id: int,
):
    """Handle one EP dispatch lane arrival and schedule the collective."""
    from frontier.events.ep_alltoall_dispatch_collective_event import (
        EPAllToAllDispatchCollectiveEvent,
    )
    from frontier.logger import get_cluster_logger

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)
    if (
        not isinstance(time, Real)
        or isinstance(time, bool)
        or not math.isfinite(float(time))
    ):
        raise ValueError(
            f"EP dispatch arrival time must be a finite int or float, got {time!r}"
        )
    time = float(time)
    (
        batch_global_id,
        dispatch_wait_room,
        expected_ep_ids,
        is_complete,
    ) = scheduler._validate_ep_barrier_arrival(
        phase="dispatch",
        waiting_rooms=scheduler._ep_alltoall_dispatch_waiting_room,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        ep_id=ep_id,
    )
    existing_batches = {} if dispatch_wait_room is None else dispatch_wait_room["batches"]
    existing_arrival_times = {} if dispatch_wait_room is None else dispatch_wait_room["arrival_times"]
    prospective_batches = dict(existing_batches)
    prospective_arrival_times = dict(existing_arrival_times)
    prospective_batches[ep_id] = batch
    prospective_arrival_times[ep_id] = time
    expected_ep_size = len(expected_ep_ids)
    if not is_complete:
        if dispatch_wait_room is None:
            dispatch_wait_room = scheduler._ep_alltoall_dispatch_waiting_room[replica_id][stage_id][batch_global_id]
        dispatch_wait_room["batches"][ep_id] = batch
        dispatch_wait_room["arrival_times"][ep_id] = time
        return []
    prospective_room = {"batches": prospective_batches, "arrival_times": prospective_arrival_times}
    data_size_bytes, local_tokens_by_ep_id, max_local_tokens, hidden_size = scheduler._get_step3_ep_alltoall_payload_bytes(prospective_batches)
    ep_collective_exec_time_ms = scheduler._predictor.predict_alltoall_time(
        data_size_bytes=data_size_bytes,
        num_devices=expected_ep_size,
        cluster_type=scheduler._cluster_type,
        comm_domain="EP",
    )
    ep_collective_sync_time = max(prospective_arrival_times.values())
    ep_collective_exec_time_ms, collective_event_time = scheduler._validate_ep_collective_exec_time(
        phase="dispatch", exec_time_ms=ep_collective_exec_time_ms, sync_time=ep_collective_sync_time
    )
    trace_batch_id, trace_layer_id = scheduler._resolve_ep_trace_identity(prospective_batches, batch_global_id)
    first_batch = next(iter(prospective_batches.values()))
    trace_routing_token_count = getattr(first_batch, "routing_token_count", None)
    trace_router_topk = getattr(first_batch, "router_topk", None)
    trace_total_routed_assignments = getattr(first_batch, "total_routed_assignments", None)
    if any(
        getattr(ep_batch, "routing_token_count", None) != trace_routing_token_count
        or getattr(ep_batch, "router_topk", None) != trace_router_topk
        or getattr(ep_batch, "total_routed_assignments", None) != trace_total_routed_assignments
        for ep_batch in prospective_batches.values()
    ):
        raise ValueError("EP dispatch lanes disagree on routing metadata")
    trace_identity = scheduler._build_ep_trace_identity(
        batch=first_batch, replica_id=replica_id, stage_id=stage_id,
        operation_id=trace_batch_id, operation_kind="ep_ffn",
    )
    scheduler._log_ep_conservation_trace(
        cluster_type=scheduler._cluster_type, batch_id=trace_batch_id,
        layer_id=trace_layer_id, routing_token_count=trace_routing_token_count,
        router_topk=trace_router_topk, total_routed_assignments=trace_total_routed_assignments,
        per_ep_routed_tokens={
            int(lane_id): resolve_ep_lane_workload(lane_batch, required=True).routed_token_count
            for lane_id, lane_batch in prospective_batches.items()
        }, trace_identity=trace_identity,
    )
    trace_origin_s = min(prospective_arrival_times.values())
    scheduler._log_ep_barrier_trace(
        cluster_type=scheduler._cluster_type, batch_id=trace_batch_id,
        layer_id=trace_layer_id, phase="dispatch",
        expected_ep_ids=tuple(sorted(expected_ep_ids)), arrived_ep_ids=tuple(sorted(expected_ep_ids)),
        max_lane_time_ms=(ep_collective_sync_time - trace_origin_s) * 1000.0,
        collective_time_ms=ep_collective_exec_time_ms,
        barrier_time_ms=(collective_event_time - trace_origin_s) * 1000.0,
        barrier_start_time_s=trace_origin_s, barrier_end_time_s=collective_event_time,
        trace_identity=trace_identity,
    )
    if dispatch_wait_room is None:
        dispatch_wait_room = scheduler._ep_alltoall_dispatch_waiting_room[replica_id][stage_id][batch_global_id]
    dispatch_wait_room["batches"][ep_id] = batch
    dispatch_wait_room["arrival_times"][ep_id] = time
    dispatch_start_time_s = min(prospective_arrival_times.values())
    for ep_batch in prospective_batches.values():
        ep_batch._ep_dispatch_collective_start_time_s = float(dispatch_start_time_s)
    logger.info(
        f"[EP-DISPATCH][COLLECTIVE] global_id={batch_global_id}, sync_time={ep_collective_sync_time:.6f}s, "
        f"exec_time={ep_collective_exec_time_ms:.6f}ms, collective_end={collective_event_time:.6f}s, "
        f"max_local_tokens={max_local_tokens}, hidden_size={hidden_size}, local_tokens_by_ep_id={local_tokens_by_ep_id}"
    )
    return [EPAllToAllDispatchCollectiveEvent(collective_event_time, replica_id, stage_id, batch_global_id)]
