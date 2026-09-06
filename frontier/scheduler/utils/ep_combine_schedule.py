"""Scheduling callback for a completed EP AllToAll combine collective.

The scheduler remains the owner of runtime state and event factories.  This
module keeps the callback's validation, cleanup, accounting, and event list
assembly out of the cluster-scheduler coordinator.
"""

from typing import Any

from frontier.scheduler.utils.expert_parallel import validate_completion_time
from frontier.scheduler.utils import ep_trace
from frontier.scheduler.utils.ep_combine import prepare_ep_combine_completion


def schedule_combine_completion(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch_global_id: int,
    metrics_store: Any,
    combine_end_time: float,
):
    """Finish one EP combine cohort and return transfer/wakeup events."""
    from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
    from frontier.logger import get_cluster_logger

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)
    logger.info(
        f"[DEBUG] on_ep_alltoall_combine_collective_schedule called: time={time:.3f}s, "
        f"replica_id={replica_id}, stage_id={stage_id}, batch_global_id={batch_global_id}"
    )

    time, combine_end_time = validate_completion_time(time, combine_end_time)
    ep_wait_room = scheduler._ep_allgather_waiting_room[replica_id][stage_id][
        batch_global_id
    ]
    ep_batches = ep_wait_room["batches"]
    ep_trace.log_combine_completion(
        time=time,
        combine_end_time=combine_end_time,
        ep_batches=ep_batches,
        arrival_times=ep_wait_room.get("arrival_times"),
        replica_id=replica_id,
        stage_id=stage_id,
        batch_global_id=batch_global_id,
        cluster_type=scheduler._cluster_type,
        cluster_logger=logger,
        formatter=scheduler._format_ep_trace_identity,
    )
    logger.info(
        f"[DEBUG] Retrieved {len(ep_batches)} EP batches from waiting room: "
        f"ep_ids={list(ep_batches.keys())}"
    )

    completion_plan = prepare_ep_combine_completion(
        ep_batches=ep_batches,
        raw_batch_lookup=scheduler._raw_batch_waiting_for_m2n_back.get,
        cluster_name=scheduler._cluster_type.name,
        replica_id=replica_id,
        stage_id=stage_id,
        batch_global_id=batch_global_id,
        token_validator=lambda input_tokens, lane_workload, context: scheduler._validate_token_conservation(
            input_tokens=input_tokens, lane_workload=lane_workload, context=context
        ),
    )
    canonical_ep_id = completion_plan.canonical_ep_id
    ffn_execution_time = completion_plan.ffn_execution_time
    raw_batches = list(completion_plan.raw_batches)
    active_requests_by_batch = dict(completion_plan.active_requests_by_batch)
    activation_bytes_by_ep_id = dict(completion_plan.activation_bytes_by_ep_id)
    logger.info(f"[FFN-EXEC-TIME] Using EP execution time: {ffn_execution_time:.6f}s")

    stage_schedulers = {
        ep_id: scheduler.get_replica_stage_scheduler(replica_id, ep_id, stage_id)
        for ep_id in ep_batches
    }
    replica_schedulers = {
        ep_id: scheduler.get_replica_scheduler(replica_id, ep_id)
        for ep_id in ep_batches
    }
    prepared_raw_commits = []
    m2n_events = []
    for batch_id, raw_batch in raw_batches:
        active_requests = list(active_requests_by_batch[batch_id])
        m2n_events.extend(
            scheduler._create_m2n_transfer_events_for_aggregated_batch(
                raw_batch,
                time,
                source_replica_id=replica_id,
                source_replica_local_id=canonical_ep_id,
            )
        )
        prepared_raw_commits.append((batch_id, raw_batch, active_requests))

    schedule_events = [
        ReplicaStageScheduleEvent(
            time, replica_id, stage_id, scheduler._cluster_type, ep_id
        )
        for ep_id, stage_scheduler in stage_schedulers.items()
        if not callable(getattr(stage_scheduler, "is_empty", None))
        or not bool(stage_scheduler.is_empty())
    ]
    full_stage_scheduler = scheduler.get_full_stage_replica_scheduler(replica_id)
    full_stage_is_empty = getattr(full_stage_scheduler, "is_empty", None)
    if not callable(full_stage_is_empty):
        raise ValueError("DECODE_FFN full-stage Replica scheduler must expose is_empty()")
    if not bool(full_stage_is_empty()):
        schedule_events.append(
            ReplicaStageScheduleEvent(
                time, replica_id, stage_id, scheduler._cluster_type, None
            )
        )

    scheduler._ep_allgather_waiting_room[replica_id][stage_id].pop(batch_global_id)
    for ep_id, stage_scheduler in stage_schedulers.items():
        stage_scheduler.on_stage_end()
        logger.info(
            f"[CRITICAL_FIX] Released busy state for replica {replica_id}, "
            f"ep_id {ep_id}, stage {stage_id}"
        )
    for ep_id, replica_scheduler in replica_schedulers.items():
        replica_scheduler.decrement_num_running_batches()
    for ep_id, replica_scheduler in replica_schedulers.items():
        activation_bytes = activation_bytes_by_ep_id[ep_id]
        if activation_bytes:
            replica_scheduler.release_activation_memory_bytes(activation_bytes)
            metrics_store.on_replica_schedule(
                time,
                replica_id,
                replica_scheduler.memory_usage_percent,
                scheduler._cluster_type,
                replica_local_id=ep_id,
            )
    for ep_id, ep_batch in ep_batches.items():
        metrics_store.flush_frontier_stage_batch_ledger_row(
            time=time,
            batch_id=ep_batch.id,
            replica_id=replica_id,
            stage_id=stage_id,
            cluster_type=scheduler._cluster_type,
            replica_local_id=ep_id,
            completion_source="ep_alltoall_combine_collective",
        )

    memory_usage_percent = max(
        replica_scheduler.memory_usage_percent
        for replica_scheduler in replica_schedulers.values()
    )
    for batch_id, raw_batch, active_requests in prepared_raw_commits:
        scheduler._raw_batch_waiting_for_m2n_back.pop(batch_id)
        logger.info(
            f"[ISSUE-007][F2A][CREATE] batch_id={raw_batch.id}, "
            f"decode_attn_original_replica_id={getattr(raw_batch, 'decode_attn_original_replica_id', 'MISSING')}, "
            f"decode_attn_original_replica_local_id={getattr(raw_batch, 'decode_attn_original_replica_local_id', 'MISSING')}"
        )
        for request in active_requests:
            request.on_batch_stage_end(
                time, ffn_execution_time, ffn_execution_time, scheduler._cluster_type
            )
        logger.info(
            f"[FFN-EXEC-TIME] Recorded execution time for batch {batch_id}: "
            f"execution_time={ffn_execution_time:.6f}s, num_requests={len(raw_batch.requests)}"
        )
        metrics_store.on_batch_end(
            time,
            raw_batch,
            replica_id,
            memory_usage_percent,
            scheduler._cluster_type,
            canonical_ep_id,
        )
        raw_batch.time = time

    scheduler.release_stage_admission_for_batch(
        ep_batches[canonical_ep_id], stage_id=stage_id
    )
    logger.info(
        f"[DEBUG] Created {len(m2n_events)} M2N transfer events: "
        f"{[event.event_type.name if event and hasattr(event, 'event_type') and event.event_type else 'Unknown' for event in m2n_events]}"
    )
    return m2n_events + schedule_events
