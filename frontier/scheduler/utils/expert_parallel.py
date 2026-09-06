"""Pure helpers for materializing Replica-local expert-parallel work.

The utility validates routing and source batches, then returns immutable plans.
Scheduler-owned entity construction remains callback based so this module does
not depend on waiting rooms or scheduler state.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable, NamedTuple, Optional

from frontier.entities import Batch, Request
from frontier.moe_ep_workload import (
    EPLaneWorkload,
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    materialize_layer_ep_workload,
    resolve_ep_lane_workload,
    resolve_routing_details,
)


class EPBatchGroupPlan(NamedTuple):
    """Immutable inputs for one DECODE_FFN EP batch."""

    replica_id: int
    ep_id: int
    layer_global_id: int
    afd_stage_idx: int
    group_time: float
    pre_routing_effective_total_tokens: int
    source_batches: tuple[Batch, ...]
    source_batch_ids: tuple[int, ...]
    lane_workload: EPLaneWorkload

    @property
    def per_expert_tokens(self) -> tuple[tuple[int, int], ...]:
        """Return the legacy tuple view without a second mutable owner."""

        return tuple(self.lane_workload.per_expert_tokens.items())


class EPCombineTimingPlan(NamedTuple):
    """Pure timing and payload inputs for an EP combine collective."""

    sync_time: float
    exec_time_ms: float
    combine_end_time: float
    post_combine_time_s: float
    final_event_time: float
    data_size_bytes: int
    payload_description: str


def validate_completion_time(time: float, combine_end_time: float) -> tuple[float, float]:
    """Validate final and combine timestamps before a waiting-room lookup."""

    if not isinstance(time, Real) or isinstance(time, bool) or not math.isfinite(float(time)):
        raise ValueError("EP combine completion time must be finite")
    time = float(time)
    if (
        not isinstance(combine_end_time, Real)
        or isinstance(combine_end_time, bool)
        or not math.isfinite(float(combine_end_time))
    ):
        raise ValueError("EP combine end time must be finite")
    combine_end_time = float(combine_end_time)
    if combine_end_time > time:
        raise ValueError(
            "EP combine end time cannot be later than final completion time: "
            f"combine_end_time={combine_end_time!r}, time={time!r}"
        )
    return time, combine_end_time


def resolve_source_batch_ids(ep_batches: dict[int, Any]) -> list[int]:
    """Require every EP lane to reference the canonical lane's source batches."""

    canonical_ep_id = min(ep_batches)
    source_ids = list(ep_batches[canonical_ep_id].source_batch_ids)
    for ep_id, ep_batch in ep_batches.items():
        lane_source_ids = list(ep_batch.source_batch_ids)
        if not lane_source_ids:
            raise ValueError(f"EP combine has empty source_batch_ids for ep_id={ep_id}")
        if len(set(lane_source_ids)) != len(lane_source_ids):
            raise ValueError(
                f"EP combine has duplicate source_batch_ids for ep_id={ep_id}: "
                f"{lane_source_ids}"
            )
        if lane_source_ids != source_ids:
            raise ValueError(
                f"source_batch_ids mismatch: ep_id={ep_id} has "
                f"{lane_source_ids}, expected {source_ids}"
            )
    return source_ids


def resolve_ep_execution_time(ep_batches: dict[int, Any]) -> float:
    """Resolve synchronized FFN time while preserving zero-work lane semantics."""

    positive_execution_times: list[float] = []
    for ep_id, ep_batch in ep_batches.items():
        execution_time = getattr(ep_batch, "execution_time", None)
        if (
            not isinstance(execution_time, Real)
            or isinstance(execution_time, bool)
            or not math.isfinite(float(execution_time))
            or float(execution_time) < 0.0
        ):
            raise ValueError(
                f"Invalid execution_time for EP batch: ep_id={ep_id}, "
                f"execution_time={execution_time!r}"
            )
        lane_workload = resolve_ep_lane_workload(ep_batch, required=True)
        assert lane_workload is not None
        execution_time_value = float(execution_time)
        if execution_time_value == 0.0:
            if lane_workload.routed_token_count != 0:
                raise ValueError(
                    "EP batch has zero execution_time with routed tokens: "
                    f"ep_id={ep_id}, routed_tokens={lane_workload.routed_token_count}"
                )
            continue
        positive_execution_times.append(execution_time_value)
    if not positive_execution_times:
        raise ValueError("EP combine has no positive execution_time lane with routed tokens")
    return max(positive_execution_times)


def validate_token_conservation(
    input_tokens: int, lane_workload: EPLaneWorkload, context: str
) -> None:
    """Reject an EP lane whose routed tokens differ from its input tokens."""

    lane_workload = resolve_ep_lane_workload(lane_workload, required=True)
    assert lane_workload is not None
    total_expert_tokens = lane_workload.routed_token_count
    if total_expert_tokens != input_tokens:
        raise ValueError(
            f"Token conservation violated in {context}: "
            f"Input tokens={input_tokens}, Expert tokens={total_expert_tokens}, "
            f"Difference={input_tokens - total_expert_tokens}, "
            f"Per-expert allocation={dict(lane_workload.per_expert_tokens)}"
        )


def materialize_wave_workload(
    group: list[tuple[Batch, Any]],
    replica_id: int,
    layer_global_id: int,
    routing_details: Any,
    *,
    total_expert_num: int,
    moe_expert_parallel_size: int,
    router_topk: int,
    routing_resolver: Callable[..., Any] = resolve_routing_details,
    workload_materializer: Callable[..., LayerEPWorkload] = materialize_layer_ep_workload,
    ownership_builder: Callable[..., Any] = build_contiguous_expert_ownership,
) -> LayerEPWorkload:
    """Materialize one aggregate workload shared by all EP lanes in a wave."""

    if type(group) is not list or not group:
        raise ValueError("DECODE_FFN EP wave group must be a non-empty list")
    routing_token_count = 0
    for entry in group:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError(
                "DECODE_FFN EP wave group entries must be (batch, transfer_info) tuples"
            )
        batch_tokens = getattr(entry[0], "total_num_tokens", None)
        if type(batch_tokens) is not int or batch_tokens < 0:
            raise ValueError(
                "DECODE_FFN EP wave source batch total_num_tokens must be a "
                f"non-negative int, got {batch_tokens!r}"
            )
        routing_token_count += batch_tokens
    for value, name, minimum in (
        (total_expert_num, "total_expert_num", 1),
        (moe_expert_parallel_size, "moe_expert_parallel_size", 1),
        (router_topk, "router_topk", 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"DECODE_FFN {name} must be an exact positive int for EP materialization"
            )
    expert_to_ep = ownership_builder(total_expert_num, moe_expert_parallel_size)
    return workload_materializer(
        routing_ratios=routing_resolver(
            routing_details,
            target_replica_id=replica_id,
            global_layer_id=layer_global_id,
        ),
        target_replica_id=replica_id,
        global_layer_id=layer_global_id,
        routing_token_count=routing_token_count,
        router_topk=router_topk,
        total_expert_num=total_expert_num,
        moe_expert_parallel_size=moe_expert_parallel_size,
        expert_to_ep=expert_to_ep,
    )


def prepare_batch_group_plan(
    group: list[tuple[Batch, Any]],
    replica_id: int,
    ep_id: int,
    expert_global_ids: list[int],
    layer_global_id: int,
    routing_details: Any,
    *,
    cluster_type: Any,
    router_topk: int,
    total_expert_num: int,
    moe_expert_parallel_size: int,
    layer_workload: Optional[LayerEPWorkload],
    wave_materializer: Callable[..., LayerEPWorkload],
) -> EPBatchGroupPlan:
    """Validate one source group and prepare its lane descriptor."""

    if type(group) is not list or not group:
        raise ValueError("group must be a non-empty list")
    if type(replica_id) is not int or replica_id < 0:
        raise ValueError("DECODE_FFN EP replica_id must be an exact non-negative int")
    if type(ep_id) is not int or ep_id < 0:
        raise ValueError("DECODE_FFN ep_id must be an exact non-negative int")
    if type(layer_global_id) is not int or layer_global_id < 0:
        raise ValueError("DECODE_FFN layer_global_id must be an exact non-negative int")
    if type(expert_global_ids) is not list or any(
        type(expert_id) is not int or expert_id < 0 for expert_id in expert_global_ids
    ):
        raise ValueError("DECODE_FFN expert_global_ids must be an exact list of non-negative ints")
    if type(routing_details) is not dict:
        raise ValueError("DECODE_FFN routing_details must be an exact dict")

    stage_ids = {getattr(batch, "afd_stage_idx", None) for batch, _ in group}
    if None in stage_ids:
        raise ValueError("afd_stage_idx missing in DECODE_FFN group batches")
    if len(stage_ids) != 1:
        raise ValueError(f"afd_stage_idx mismatch in group: {sorted(stage_ids)}")
    afd_stage_idx = stage_ids.pop()
    if type(afd_stage_idx) is not int or afd_stage_idx < 0:
        raise ValueError("DECODE_FFN afd_stage_idx must be an exact non-negative int")

    source_batches: list[Batch] = []
    source_batch_ids: list[int] = []
    total_tokens = 0
    pre_routing_tokens = 0
    for batch, _ in group:
        if cluster_type.name == "DECODE_FFN":
            original_replica_id = getattr(batch, "decode_attn_original_replica_id", None)
            if original_replica_id is None:
                raise ValueError(
                    f"[ISSUE-007] Batch {batch.id} entering DECODE_FFN without "
                    "decode_attn_original_replica_id"
                )
            if type(original_replica_id) is not int or original_replica_id < 0:
                raise ValueError("DECODE_FFN source decode_attn_original_replica_id must be an exact non-negative int")
            original_local_id = getattr(batch, "decode_attn_original_replica_local_id", None)
            if original_local_id is not None and (type(original_local_id) is not int or original_local_id < 0):
                raise ValueError("DECODE_FFN source decode_attn_original_replica_local_id must be None or an exact non-negative int")
        batch_id = getattr(batch, "id", None)
        tokens = getattr(batch, "num_tokens", None)
        batch_total = getattr(batch, "total_num_tokens", None)
        if type(batch_id) is not int or batch_id < 0:
            raise ValueError("DECODE_FFN source batch id must be an exact non-negative int")
        if type(tokens) is not list or any(type(value) is not int or value < 0 for value in tokens):
            raise ValueError("DECODE_FFN source batch num_tokens must be an exact list of non-negative ints")
        if type(batch_total) is not int or batch_total < 0 or sum(tokens) != batch_total:
            raise ValueError("DECODE_FFN source batch total_num_tokens must exactly equal the sum of num_tokens")
        source_batches.append(batch)
        source_batch_ids.append(batch_id)
        total_tokens += batch_total
        pre_routing_tokens += int(batch.get_effective_total_tokens_for_compute(cluster_type))
    if pre_routing_tokens <= 0:
        raise ValueError("DECODE_FFN EP group requires positive pre-routing effective tokens")
    if type(router_topk) is not int or router_topk <= 0:
        raise ValueError("DECODE_FFN router_topk must be an exact positive int")
    if type(total_expert_num) is not int or total_expert_num <= 0:
        raise ValueError("DECODE_FFN total_expert_num must be an exact positive int for EP materialization")
    if type(moe_expert_parallel_size) is not int or moe_expert_parallel_size <= 0:
        raise ValueError("DECODE_FFN moe_expert_parallel_size must be an exact positive int for EP materialization")
    ownership = build_contiguous_expert_ownership(total_expert_num, moe_expert_parallel_size)
    expected_ids = [expert_id for expert_id in range(total_expert_num) if ownership[expert_id] == ep_id]
    if sorted(expert_global_ids) != expected_ids:
        raise ValueError(
            "DECODE_FFN expert_global_ids do not match contiguous ownership "
            f"for ep_id={ep_id}: expected={expected_ids}, got={expert_global_ids}"
        )
    if layer_workload is None:
        layer_workload = wave_materializer(group, replica_id, layer_global_id, routing_details)
    if not isinstance(layer_workload, LayerEPWorkload):
        raise ValueError("DECODE_FFN shared EP workload must be a LayerEPWorkload instance")
    if layer_workload.target_replica_id != replica_id or layer_workload.global_layer_id != layer_global_id:
        raise ValueError("DECODE_FFN shared EP workload identity does not match the source group")
    if layer_workload.routing_token_count != total_tokens:
        raise ValueError(
            "DECODE_FFN shared EP workload routing-token mismatch: "
            f"expected={total_tokens}, got={layer_workload.routing_token_count}"
        )
    lane_workload = layer_workload.lane(ep_id)
    if tuple(expert_global_ids) != lane_workload.owned_expert_ids:
        raise ValueError("DECODE_FFN expert_global_ids do not match the canonical lane descriptor")
    # Validate the lane descriptor against its own routed token count. Empty
    # lanes are valid participants in an EP wave and must remain constructible.
    validate_token_conservation(
        lane_workload.routed_token_count,
        lane_workload,
        "prepare_batch_group_plan",
    )
    return EPBatchGroupPlan(
        replica_id=replica_id,
        ep_id=ep_id,
        layer_global_id=layer_global_id,
        afd_stage_idx=afd_stage_idx,
        group_time=max((batch.time or 0.0) for batch, _ in group),
        pre_routing_effective_total_tokens=pre_routing_tokens,
        source_batches=tuple(source_batches),
        source_batch_ids=tuple(source_batch_ids),
        lane_workload=lane_workload,
    )


def materialize_batch_group(
    plan: EPBatchGroupPlan,
    *,
    create_batch_group: Callable[..., Any],
    aggregate_metadata: Callable[[tuple[Batch, ...]], tuple[Any | None, bool]],
) -> Any:
    """Construct an EPBatchGroup through scheduler-owned callbacks."""

    lane = plan.lane_workload
    token_counts = list(lane.local_token_counts)
    requests = [Request(0.0, 0, count) for count in token_counts]
    result = create_batch_group(
        requests,
        token_counts,
        plan.replica_id,
        plan.ep_id,
        plan.group_time,
        list(plan.source_batch_ids),
        lane,
    )
    result.afd_stage_idx = plan.afd_stage_idx
    result.decode_ffn_layer_id = plan.layer_global_id
    result.afd_stage_metadata, result.afd_stage_represents_all_stages = aggregate_metadata(plan.source_batches)
    routing_token_count = sum(int(batch.total_num_tokens) for batch in plan.source_batches)
    result.routing_token_count = routing_token_count
    result.router_topk = lane.router_topk
    result.total_routed_assignments = routing_token_count * lane.router_topk
    result.moe_pre_routing_effective_total_tokens = plan.pre_routing_effective_total_tokens
    result.source_batches = list(plan.source_batches)
    return result


def validate_collective_exec_time(
    *, phase: str, exec_time_ms: Real, sync_time: Real
) -> tuple[float, float]:
    """Validate EP collective latency and return its event timestamp."""

    if not isinstance(exec_time_ms, Real) or isinstance(exec_time_ms, bool):
        raise ValueError(
            f"EP {phase} collective latency must be an exact int or float, "
            f"got {exec_time_ms!r}"
        )
    if not math.isfinite(float(exec_time_ms)) or exec_time_ms < 0:
        raise ValueError(
            f"EP {phase} collective latency must be finite and non-negative, "
            f"got {exec_time_ms!r}"
        )
    if (
        not isinstance(sync_time, Real)
        or isinstance(sync_time, bool)
        or not math.isfinite(float(sync_time))
    ):
        raise ValueError(
            f"EP {phase} collective sync time must be finite, got {sync_time!r}"
        )
    exec_time_value = float(exec_time_ms)
    event_time = float(sync_time) + exec_time_value / 1000.0
    if not math.isfinite(event_time):
        raise ValueError(
            f"EP {phase} collective event time must be finite, got {event_time!r}"
        )
    if event_time < float(sync_time):
        raise ValueError(
            f"EP {phase} collective event time cannot precede its sync time: "
            f"sync={sync_time!r}, event={event_time!r}"
        )
    return exec_time_value, event_time


def validate_barrier_arrival(
    *,
    phase: str,
    waiting_rooms: Any,
    get_replica: Callable[[int], Any],
    default_ep_size: int,
    replica_id: int,
    stage_id: int,
    batch: Any,
    ep_id: int,
) -> tuple[int, Optional[dict], frozenset[int], bool]:
    """Validate one EP lane arrival without mutating its waiting room."""

    if type(replica_id) is not int or replica_id < 0:
        raise ValueError(
            f"EP {phase} replica_id must be an exact non-negative int, got {replica_id!r}"
        )
    if type(stage_id) is not int or stage_id < 0:
        raise ValueError(
            f"EP {phase} stage_id must be an exact non-negative int, got {stage_id!r}"
        )

    batch_global_id = getattr(batch, "global_id", None)
    if type(batch_global_id) is not int or batch_global_id < 0:
        raise ValueError(
            f"EP {phase} batch global_id must be an exact non-negative int, got {batch_global_id!r}"
        )
    batch_replica_id = getattr(batch, "replica_id", None)
    if type(batch_replica_id) is not int or batch_replica_id < 0:
        raise ValueError(
            f"EP {phase} batch replica_id must be an exact non-negative int, got {batch_replica_id!r}"
        )
    if batch_replica_id != replica_id:
        raise ValueError(
            f"EP {phase} event/batch replica_id mismatch: event={replica_id!r}, batch={batch_replica_id!r}"
        )

    replica = get_replica(replica_id)
    expected_ep_size = getattr(replica, "ep_size", default_ep_size)
    if type(expected_ep_size) is not int or expected_ep_size <= 0:
        raise ValueError(
            f"EP {phase} expected_ep_size must be an exact positive int, got {expected_ep_size!r}"
        )
    expected_ep_ids = frozenset(range(expected_ep_size))
    if type(ep_id) is not int or ep_id not in expected_ep_ids:
        raise ValueError(
            f"EP {phase} ep_id must be an exact int in {sorted(expected_ep_ids)}, got {ep_id!r}"
        )
    batch_ep_id = getattr(batch, "ep_id", None)
    if type(batch_ep_id) is not int or batch_ep_id != ep_id:
        raise ValueError(
            f"EP {phase} event/batch ep_id mismatch: event={ep_id!r}, batch={batch_ep_id!r}"
        )

    replica_rooms = waiting_rooms.get(replica_id)
    stage_rooms = replica_rooms.get(stage_id) if replica_rooms is not None else None
    room = stage_rooms.get(batch_global_id) if stage_rooms is not None else None
    if room is None:
        existing_ep_ids = set()
    else:
        batch_ep_ids = set(room["batches"])
        arrival_ep_ids = set(room["arrival_times"])
        if batch_ep_ids != arrival_ep_ids:
            raise ValueError(
                f"EP {phase} waiting-room batch/arrival key mismatch: "
                f"batches={sorted(batch_ep_ids, key=repr)}, arrival_times={sorted(arrival_ep_ids, key=repr)}"
            )
        if any(type(existing_ep_id) is not int for existing_ep_id in batch_ep_ids):
            raise ValueError(
                f"EP {phase} waiting room contains a non-exact ep_id: {sorted(batch_ep_ids, key=repr)}"
            )
        if not batch_ep_ids.issubset(expected_ep_ids):
            raise ValueError(
                f"EP {phase} waiting-room lane set is outside the expected ep_id domain: "
                f"lanes={sorted(batch_ep_ids)}, expected={sorted(expected_ep_ids)}"
            )
        for lane_ep_id, stored_batch in room["batches"].items():
            stored_global_id = getattr(stored_batch, "global_id", None)
            if type(stored_global_id) is not int or stored_global_id != batch_global_id:
                raise ValueError(
                    f"EP {phase} waiting-room batch global_id mismatch: room={batch_global_id!r}, "
                    f"stored={stored_global_id!r}, lane={lane_ep_id!r}"
                )
            stored_ep_id = getattr(stored_batch, "ep_id", None)
            if type(stored_ep_id) is not int or stored_ep_id != lane_ep_id:
                raise ValueError(
                    f"EP {phase} waiting-room batch ep_id mismatch: lane={lane_ep_id!r}, stored={stored_ep_id!r}"
                )
            stored_replica_id = getattr(stored_batch, "replica_id", None)
            if type(stored_replica_id) is not int or stored_replica_id != replica_id:
                raise ValueError(
                    f"EP {phase} waiting-room batch replica_id mismatch: event={replica_id!r}, "
                    f"stored={stored_replica_id!r}, lane={lane_ep_id!r}"
                )
        existing_ep_ids = batch_ep_ids

    if ep_id in existing_ep_ids:
        raise ValueError(
            f"EP {phase} duplicate ep_id arrival: ep_id={ep_id}, global_id={batch_global_id}"
        )
    arrived_ep_ids = existing_ep_ids | {ep_id}
    return batch_global_id, room, expected_ep_ids, arrived_ep_ids == expected_ep_ids


def summarize_alltoall_payload(
    ep_batches: dict[int, Any], hidden_size: int
) -> tuple[int, dict[int, int], int, int]:
    """Return max-lane payload bytes and token counts for an EP all-to-all."""

    if not ep_batches:
        raise ValueError("Step3 EP all-to-all payload requested with no EP batches")
    hidden_size = int(hidden_size)
    local_tokens_by_ep_id: dict[int, int] = {}
    for lane_ep_id, ep_batch in ep_batches.items():
        lane_workload = resolve_ep_lane_workload(ep_batch, required=True)
        assert lane_workload is not None
        if lane_workload.ep_id != lane_ep_id:
            raise ValueError(
                "EP batch lane key must match its EPLaneWorkload ep_id: "
                f"key={lane_ep_id!r}, descriptor={lane_workload.ep_id!r}"
            )
        entity_tokens = getattr(ep_batch, "total_num_tokens", None)
        if type(entity_tokens) is not int or entity_tokens < 0:
            raise ValueError(
                "EP batch total_num_tokens must be an exact non-negative int "
                f"for Step3 all-to-all: ep_id={lane_ep_id}, total_num_tokens={entity_tokens!r}"
            )
        local_tokens = lane_workload.routed_token_count
        if entity_tokens != local_tokens:
            raise ValueError(
                "EP batch total_num_tokens must equal its lane routed-token count "
                f"for Step3 all-to-all: ep_id={lane_ep_id}, total_num_tokens={entity_tokens}, "
                f"routed_token_count={local_tokens}"
            )
        local_tokens_by_ep_id[int(lane_ep_id)] = local_tokens
    max_local_tokens = max(local_tokens_by_ep_id.values(), default=0)
    return max_local_tokens * hidden_size * 2, local_tokens_by_ep_id, max_local_tokens, hidden_size


def prepare_combine_timing(
    *,
    prospective_batches: dict[int, Any],
    prospective_arrival_times: dict[int, float],
    expected_ep_size: int,
    collective_kind: Any,
    cluster_type: Any,
    hidden_size: int,
    predict_alltoall: Callable[..., float],
    predict_allgather: Callable[..., float],
    collective_time_validator: Callable[..., tuple[float, float]] = validate_collective_exec_time,
) -> EPCombineTimingPlan:
    """Prepare pure EP combine timing inputs without mutating scheduler state."""

    if not prospective_batches:
        raise ValueError("EP combine timing requires at least one lane")
    if not prospective_arrival_times or set(prospective_arrival_times) != set(prospective_batches):
        raise ValueError("EP combine timing requires one arrival time for every lane")
    hidden_size = int(hidden_size)
    from frontier.model_architectures import ExpertParallelCollective

    if collective_kind is ExpertParallelCollective.ALLTOALL:
        data_size_bytes, local_tokens_by_ep_id, max_local_tokens, _ = summarize_alltoall_payload(
            prospective_batches, hidden_size
        )
        payload_description = (
            f"max_local_tokens={max_local_tokens}, hidden_size={hidden_size}, "
            f"local_tokens_by_ep_id={local_tokens_by_ep_id}"
        )
        exec_time = predict_alltoall(
            data_size_bytes=data_size_bytes,
            num_devices=expected_ep_size,
            cluster_type=cluster_type,
            comm_domain="EP",
        )
    else:
        representative_batch = next(iter(prospective_batches.values()))
        total_tokens = getattr(representative_batch, "total_num_tokens", None)
        if type(total_tokens) is not int or total_tokens < 0:
            raise ValueError(
                "EP combine representative batch total_num_tokens must be an exact non-negative int"
            )
        data_size_bytes = total_tokens * hidden_size * 2
        payload_description = f"{total_tokens} tokens × {hidden_size} hidden_size"
        exec_time = predict_allgather(
            data_size_bytes=data_size_bytes,
            num_devices=expected_ep_size,
            cluster_type=cluster_type,
            comm_domain="EP",
        )

    sync_time = max(prospective_arrival_times.values())
    exec_time_ms, combine_end_time = collective_time_validator(
        phase="combine", exec_time_ms=exec_time, sync_time=sync_time
    )
    post_combine_times: list[float] = []
    for lane_ep_id, ep_batch in prospective_batches.items():
        post_combine_time = getattr(ep_batch, "post_combine_time", None)
        if (
            not isinstance(post_combine_time, Real)
            or isinstance(post_combine_time, bool)
            or not math.isfinite(float(post_combine_time))
            or float(post_combine_time) < 0.0
        ):
            raise ValueError(
                "EP combine lane is missing a finite non-negative post_combine_time in seconds: "
                f"ep_id={lane_ep_id}, value={post_combine_time!r}"
            )
        post_combine_times.append(float(post_combine_time))
    post_combine_time_s = max(post_combine_times)
    return EPCombineTimingPlan(
        sync_time=float(sync_time),
        exec_time_ms=exec_time_ms,
        combine_end_time=float(combine_end_time),
        post_combine_time_s=post_combine_time_s,
        final_event_time=float(combine_end_time) + post_combine_time_s,
        data_size_bytes=data_size_bytes,
        payload_description=payload_description,
    )
