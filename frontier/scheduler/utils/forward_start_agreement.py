"""Bind admitted DP lanes before their first predictor call; timing stays in events."""

from dataclasses import dataclass

from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.config import global_vars
from frontier.config.graph_policy import (
    GRAPH_MODE_ORDER, graph_policy, resolve_capture_size,
)
from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch


@dataclass(frozen=True)
class ForwardStartDescriptor:
    group_id: int
    graph_mode: str
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None
    max_query_len: int | None


def bind_forward_start(batches, *, group_id):
    """Apply native min-mode/local-eager-or-shared-capture semantics to a group."""
    policy_name = global_vars.get_decode_cuda_graph_mode()
    policy = graph_policy(policy_name)
    sizes = global_vars.get_cudagraph_capture_sizes()
    if policy_name != "none" and not sizes:
        raise ValueError("Measured Step forward agreement requires explicit cudagraph_capture_sizes")
    local = []
    for batch in batches:
        dummy = isinstance(batch, DummyForwardBatch)
        queries = (1,) if dummy else tuple(batch.num_tokens)
        uniform = 1 if dummy or all(
            query == 1 and (decoding or context > 0)
            for query, decoding, context in zip(
                queries, batch.request_is_decoding, batch.num_context_tokens,
            )
        ) else None
        tokens = sum(queries)
        mode = policy.uniform_decode if uniform == 1 else policy.other
        if mode != "NONE" and not resolve_capture_size(tokens, sizes)[0]:
            mode = "NONE"
        local.append((batch, tokens, len(queries), mode))
    mode = min((row[-1] for row in local), key=GRAPH_MODE_ORDER.__getitem__)
    width = max(row[1] for row in local)
    if mode != "NONE":
        hit, width = resolve_capture_size(width, sizes)
        if not hit:
            raise ValueError("DP-agreed graph width has no capture bucket")
    bindings = []
    for batch, tokens, requests, _ in local:
        physical_tokens = tokens if mode == "NONE" else width
        physical_requests = (
            physical_tokens if mode == "FULL" else requests if mode == "NONE" else None
        )
        # No-MTP uniform FULL captures carry uniform_token_count=1 and raw
        # max_query_len=None; PIECEWISE captures constrain neither field.
        # gpu/cudagraph_utils.py:328-333 and dp_utils.py:155-210.
        descriptor = ForwardStartDescriptor(
            group_id, mode, physical_tokens, physical_requests,
            1 if mode == "FULL" else None, None,
        )
        dummy = None
        if isinstance(batch, DummyForwardBatch):
            dummy = MRv2DummyDescriptor(
                physical_tokens, physical_requests, mode.lower(), descriptor.max_query_len,
                "MRv2 gpu/dp_utils.py:155-210; gpu/model_runner.py execute_dummy_batch InputBatch",
                descriptor.uniform_token_count,
            )
        bindings.append((batch, descriptor, dummy, tokens, requests))
    # Validate every split before changing any lane's runtime input.
    for batch, descriptor, dummy, tokens, requests in bindings:
        batch.forward_start_descriptor = descriptor
        if dummy is not None:
            batch.mrv2_dummy_descriptor = dummy
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            policy_name, mode, mode != "NONE", mode != "FULL", tokens,
            descriptor.num_tokens, requests, descriptor.num_reqs or requests,
        )


def hold_forward_start(scheduler, stage, batch, *, time, replica_id, stage_id):
    """Hold one admitted batch and resume all lanes through existing stage events."""
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent
    from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

    context = scheduler.get_stage_execution_context(replica_id, stage_id)
    if context.full_stage_capacity != scheduler._replica_dp_size:
        raise ValueError("Step forward agreement requires full-stage capacity equal to attention DP")
    lane_id = stage._replica_local_id
    lane_id = 0 if lane_id is None else lane_id
    stage.hold_forward_start(batch)
    ready = context.arrive_forward_start(batch._stage_admission_ticket, lane_id)
    stages = [
        scheduler.get_replica_stage_scheduler(replica_id, lane, stage_id)
        for lane in range(context.full_stage_capacity)
    ]
    if ready:
        batches = [lane.held_forward_start_batch for lane in stages]
        group = batch._forward_cohort_provisional_id
        if any(
            lane_batch is None or lane_batch._forward_cohort_provisional_id != group
            for lane_batch in batches
        ):
            raise ValueError("forward start lanes must hold the same admitted group")
        bind_forward_start(batches, group_id=group)
        context.resolve_forward_start()
        return [
            ReplicaStageScheduleEvent(
                time, replica_id, stage_id, scheduler._cluster_type, lane._replica_local_id,
            ) for lane in stages
        ]
    events = []
    for lane_id, lane in enumerate(stages):
        if lane.is_busy:
            continue
        if not lane.is_empty():
            if context.claim_forward_start_wakeup(lane_id):
                events.append(ReplicaStageScheduleEvent(
                    time, replica_id, stage_id, scheduler._cluster_type, lane._replica_local_id,
                ))
        elif scheduler.get_replica_scheduler(replica_id, lane_id).engine_loop_idle:
            if context.claim_forward_start_wakeup(lane_id):
                events.append(ReplicaScheduleEvent(
                    time, replica_id, scheduler._cluster_type, lane_id,
                ))
    return events
