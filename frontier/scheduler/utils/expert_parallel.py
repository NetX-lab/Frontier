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
