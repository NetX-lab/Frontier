"""Validation and logging helpers for expert-parallel trace records.

These helpers are intentionally stateless.  Scheduling state and event
orchestration remain owned by ``BaseClusterScheduler``; this module only
validates trace inputs, builds stable identities, and emits observability
records.
"""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any, Callable, Dict, Tuple

from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.utils.scheduler_diagnostics import format_ep_trace_identity
from frontier.types import ClusterType


logger = logging.getLogger("frontier.scheduler.cluster_scheduler.base_cluster_scheduler")


def log_combine_completion(
    *, ep_batches, arrival_times, time, combine_end_time, replica_id, stage_id,
    batch_global_id, cluster_type, cluster_logger, formatter,
) -> None:
    """Validate lane timestamps and log one completed dispatch/combine wave."""

    if not ep_batches:
        raise ValueError("EP all-to-all collective reached with empty ep_batches")
    if not isinstance(arrival_times, dict) or set(arrival_times) != set(ep_batches):
        raise ValueError("EP combine collective requires one arrival time for every lane")
    trace_batch_id, trace_layer_id = resolve_trace_identity(ep_batches, batch_global_id)
    trace_sync_s = max(arrival_times.values())
    if combine_end_time < trace_sync_s:
        raise ValueError("EP combine end time cannot precede the slowest lane")
    trace_identity = build_trace_identity(
        batch=next(iter(ep_batches.values())), replica_id=replica_id,
        stage_id=stage_id, operation_id=trace_batch_id, operation_kind="ep_ffn",
    )
    dispatch_end_times = {
        getattr(ep_batch, "_ep_dispatch_collective_end_time_s", None)
        for ep_batch in ep_batches.values()
    }
    if len(dispatch_end_times) != 1 or None in dispatch_end_times:
        raise ValueError(
            "EP combine collective requires one dispatch collective end time "
            f"for every lane: values={sorted(dispatch_end_times, key=repr)}"
        )
    dispatch_end_time_s = float(next(iter(dispatch_end_times)))
    dispatch_start_times = {
        getattr(ep_batch, "_ep_dispatch_collective_start_time_s", None)
        for ep_batch in ep_batches.values()
    }
    if len(dispatch_start_times) != 1 or None in dispatch_start_times:
        raise ValueError(
            "EP combine collective requires one dispatch collective arrival "
            "time for every lane: "
            f"values={sorted(dispatch_start_times, key=repr)}"
        )
    dispatch_start_time_s = float(next(iter(dispatch_start_times)))
    if dispatch_end_time_s < dispatch_start_time_s:
        raise ValueError("EP dispatch collective end cannot precede dispatch arrival")
    if trace_sync_s < dispatch_end_time_s:
        raise ValueError("EP combine lane arrival cannot precede dispatch collective end")
    log_barrier_trace(
        cluster_type=cluster_type, batch_id=trace_batch_id, layer_id=trace_layer_id,
        phase="combine", expected_ep_ids=tuple(sorted(ep_batches)),
        arrived_ep_ids=tuple(sorted(ep_batches)),
        max_lane_time_ms=(trace_sync_s - dispatch_end_time_s) * 1000.0,
        collective_time_ms=(combine_end_time - trace_sync_s) * 1000.0,
        barrier_time_ms=(combine_end_time - dispatch_end_time_s) * 1000.0,
        barrier_start_time_s=dispatch_end_time_s, barrier_end_time_s=combine_end_time,
        trace_identity=trace_identity, format_identity=formatter,
    )
    log_wave_end_trace(
        cluster_type=cluster_type, batch_id=trace_batch_id, layer_id=trace_layer_id,
        wave_start_time_s=dispatch_start_time_s,
        combine_barrier_end_time_s=combine_end_time,
        post_combine_time_ms=(time - combine_end_time) * 1000.0,
        wave_end_time_s=time, trace_identity=trace_identity, format_identity=formatter,
    )
    cluster_logger.info(
        "[EP-POST-COMBINE] batch_global_id=%s combine_end_time=%.12f "
        "final_completion_time=%.12f post_combine_duration_ms=%.6f %s",
        batch_global_id, combine_end_time, time, (time - combine_end_time) * 1000.0,
        formatter(trace_identity),
    )


def resolve_trace_identity(
    ep_batches: Dict[int, Any],
    batch_global_id: int,
) -> tuple[int, int]:
    """Resolve one logical batch/layer identity for a completed EP wave."""

    if not isinstance(ep_batches, dict) or not ep_batches:
        raise ValueError("EP trace identity requires a non-empty batch map")
    if type(batch_global_id) is not int or batch_global_id < 0:
        raise ValueError(
            "EP trace identity batch_global_id must be a non-negative int"
        )

    layer_ids: set[int] = set()
    source_id_lists: list[tuple[int, ...]] = []
    for ep_id, ep_batch in ep_batches.items():
        if type(ep_id) is not int or ep_id < 0:
            raise ValueError(f"EP trace identity has invalid ep_id={ep_id!r}")
        layer_id = getattr(ep_batch, "decode_ffn_layer_id", None)
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError(
                "EP trace identity requires decode_ffn_layer_id on every lane"
            )
        layer_ids.add(layer_id)
        raw_source_ids = getattr(ep_batch, "source_batch_ids", None)
        if not isinstance(raw_source_ids, (list, tuple)):
            raise ValueError(
                "EP trace identity requires source_batch_ids on every lane"
            )
        source_ids = tuple(raw_source_ids)
        if any(
            type(source_id) is not int or source_id < 0 for source_id in source_ids
        ):
            raise ValueError(
                "EP trace identity source_batch_ids must be non-negative ints"
            )
        source_id_lists.append(source_ids)

    if len(layer_ids) != 1:
        raise ValueError(
            f"EP trace identity has inconsistent layer IDs: {sorted(layer_ids)}"
        )
    logical_batch_id = batch_global_id
    if source_id_lists and all(len(ids) == 1 for ids in source_id_lists):
        source_ids = {ids[0] for ids in source_id_lists}
        if len(source_ids) == 1:
            logical_batch_id = next(iter(source_ids))
    return logical_batch_id, next(iter(layer_ids))


def build_trace_identity(
    *,
    batch: Any,
    replica_id: int,
    stage_id: int,
    operation_id: int,
    operation_kind: str,
    afd_stage_idx: int | None = None,
) -> dict[str, Any]:
    """Build the structured identity attached to every EP trace record."""

    if type(replica_id) is not int or replica_id < 0:
        raise ValueError("EP trace replica_id must be a non-negative int")
    if type(stage_id) is not int or stage_id < 0:
        raise ValueError("EP trace stage_id must be a non-negative int")
    if type(operation_id) is not int or operation_id < 0:
        raise ValueError("EP trace operation_id must be a non-negative int")
    if not isinstance(operation_kind, str) or not operation_kind.strip():
        raise ValueError("EP trace operation_kind must be a non-empty string")

    source_batches = getattr(batch, "source_batches", None)
    if source_batches is not None:
        if not isinstance(source_batches, (list, tuple)) or not source_batches:
            raise ValueError(
                "EP trace source_batches must be a non-empty list or tuple"
            )
        source_requests = []
        for source_batch in source_batches:
            requests_for_source = getattr(source_batch, "requests", None)
            if not isinstance(requests_for_source, (list, tuple)) or not requests_for_source:
                raise ValueError(
                    "EP trace source batch must carry a non-empty request list"
                )
            source_requests.extend(requests_for_source)
        requests = source_requests
    else:
        requests = getattr(batch, "requests", None)
    if not isinstance(requests, (list, tuple)) or not requests:
        raise ValueError("EP trace identity requires a non-empty request list")
    request_ids = [getattr(request, "id", None) for request in requests]
    if any(type(request_id) is not int or request_id < 0 for request_id in request_ids):
        raise ValueError("EP trace request_ids must be non-negative ints")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("EP trace request_ids must be unique")

    raw_epochs = None
    if source_batches is not None:
        raw_epochs = [
            int(runtime_epoch)
            for source_batch in source_batches
            for runtime_epoch in getattr(source_batch, "request_runtime_epochs", [])
        ]
    if not isinstance(raw_epochs, (list, tuple)) or len(raw_epochs) != len(request_ids):
        raw_epochs = getattr(batch, "request_runtime_epochs", None)
    if not isinstance(raw_epochs, (list, tuple)):
        raw_epochs = [getattr(request, "runtime_epoch", None) for request in requests]
    request_runtime_epochs = list(raw_epochs)
    if len(request_runtime_epochs) != len(request_ids) or any(
        type(epoch) is not int or epoch < 0 for epoch in request_runtime_epochs
    ):
        raise ValueError("EP trace request_runtime_epochs must align with request_ids")

    iteration_ids = []
    for request in requests:
        token_index = getattr(request, "current_decode_token_index", None)
        if type(token_index) is not int or token_index < 1:
            raise ValueError(
                "EP trace request current_decode_token_index must be >= 1"
            )
        iteration_ids.append(token_index - 1)

    schedule_epoch = getattr(batch, "schedule_epoch", 0)
    if type(schedule_epoch) is not int or schedule_epoch < 0:
        raise ValueError("EP trace schedule_epoch must be a non-negative int")
    if afd_stage_idx is None:
        afd_stage_idx = getattr(batch, "afd_stage_idx", None)
    if afd_stage_idx is None:
        afd_stage_idx = -1
    if type(afd_stage_idx) is not int or afd_stage_idx < -1:
        raise ValueError("EP trace afd_stage_idx must be >= -1")

    return {
        "replica_id": replica_id,
        "stage_id": stage_id,
        "request_ids": tuple(request_ids),
        "request_runtime_epochs": tuple(request_runtime_epochs),
        "iteration_ids": tuple(iteration_ids),
        "schedule_epoch": schedule_epoch,
        "afd_stage_idx": afd_stage_idx,
        "operation_id": operation_id,
        "operation_kind": operation_kind.strip(),
    }


def log_workload_trace(
    *,
    cluster_type: ClusterType,
    batch_id: int,
    layer_id: int,
    lane_workload: EPLaneWorkload,
    lane_compute_ms: float,
    routed_compute_ms: float,
    lane_comm_ms: float,
    pre_dispatch_ms: float,
    dispatch_ms: float,
    combine_ms: float,
    post_combine_ms: float,
    trace_identity: Dict[str, Any],
    format_identity: Callable[[Dict[str, Any]], str] = format_ep_trace_identity,
) -> None:
    """Emit one source-level record for a materialized EP participant."""

    if not isinstance(lane_workload, EPLaneWorkload):
        raise ValueError(
            "EP workload trace requires an EPLaneWorkload descriptor, got "
            f"{type(lane_workload).__name__}"
        )
    ep_id = lane_workload.ep_id
    moe_ep_size = lane_workload.moe_expert_parallel_size
    normalized_tokens = dict(lane_workload.per_expert_tokens)
    if type(batch_id) is not int or batch_id < 0:
        raise ValueError(
            f"EP workload batch_id must be a non-negative int, got {batch_id!r}"
        )
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError(
            f"EP workload layer_id must be a non-negative int, got {layer_id!r}"
        )
    phase_values = (
        ("pre_dispatch_ms", pre_dispatch_ms),
        ("dispatch_ms", dispatch_ms),
        ("routed_compute_ms", routed_compute_ms),
        ("combine_ms", combine_ms),
        ("post_combine_ms", post_combine_ms),
    )
    for name, value in (
        ("lane_compute_ms", lane_compute_ms),
        ("lane_comm_ms", lane_comm_ms),
        *phase_values,
    ):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"EP workload {name} must be a real number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"EP workload {name} must be finite and non-negative, got {value!r}"
            )
    expected_lane_compute_ms = math.fsum(
        float(value)
        for name, value in phase_values
        if name in {"pre_dispatch_ms", "routed_compute_ms", "post_combine_ms"}
    )
    expected_lane_comm_ms = float(dispatch_ms) + float(combine_ms)
    if not math.isclose(
        float(lane_compute_ms),
        expected_lane_compute_ms,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "EP workload lane_compute_ms does not equal "
            "pre_dispatch_ms + routed_compute_ms + post_combine_ms: "
            f"lane_compute_ms={lane_compute_ms!r}, "
            f"expected={expected_lane_compute_ms!r}"
        )
    if not math.isclose(
        float(lane_comm_ms),
        expected_lane_comm_ms,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "EP workload lane_comm_ms does not equal "
            "dispatch_ms + combine_ms: "
            f"lane_comm_ms={lane_comm_ms!r}, expected={expected_lane_comm_ms!r}"
        )

    cluster_name = getattr(cluster_type, "name", str(cluster_type))
    logger.info(
        "[EP-WORKLOAD][%s] batch_id=%d, layer_id=%d, ep_id=%d, "
        "moe_ep_size=%d, per_expert_tokens=%s, lane_compute_ms=%.6f, "
        "routed_compute_ms=%.6f, lane_comm_ms=%.6f, "
        "pre_dispatch_ms=%.6f, dispatch_ms=%.6f, "
        "combine_ms=%.6f, post_combine_ms=%.6f, %s",
        cluster_name,
        batch_id,
        layer_id,
        ep_id,
        moe_ep_size,
        dict(sorted(normalized_tokens.items())),
        float(lane_compute_ms),
        float(routed_compute_ms),
        float(lane_comm_ms),
        float(pre_dispatch_ms),
        float(dispatch_ms),
        float(combine_ms),
        float(post_combine_ms),
        format_identity(trace_identity),
    )


def log_wave_end_trace(
    *,
    cluster_type: ClusterType,
    batch_id: int,
    layer_id: int,
    wave_start_time_s: float,
    combine_barrier_end_time_s: float,
    post_combine_time_ms: float,
    wave_end_time_s: float,
    trace_identity: Dict[str, Any],
    format_identity: Callable[[Dict[str, Any]], str] | None = None,
) -> None:
    """Emit the final post-combine end of one EP wave."""

    if format_identity is None:
        from frontier.scheduler.utils.scheduler_diagnostics import (
            format_ep_trace_identity,
        )

        format_identity = format_ep_trace_identity

    for name, value in (
        ("wave_start_time_s", wave_start_time_s),
        ("combine_barrier_end_time_s", combine_barrier_end_time_s),
        ("post_combine_time_ms", post_combine_time_ms),
        ("wave_end_time_s", wave_end_time_s),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"EP wave end {name} must be finite and non-negative, got {value!r}"
            )
    expected_end_time_s = float(combine_barrier_end_time_s) + float(post_combine_time_ms) * 1e-3
    if not math.isclose(float(wave_end_time_s), expected_end_time_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "EP wave end time does not match combine end plus post-combine "
            f"time: expected={expected_end_time_s!r}, actual={wave_end_time_s!r}"
        )
    if float(wave_end_time_s) < float(wave_start_time_s):
        raise ValueError("EP wave end time cannot precede wave start")
    cluster_name = getattr(cluster_type, "name", str(cluster_type))
    wave_time_ms = (float(wave_end_time_s) - float(wave_start_time_s)) * 1000.0
    logger.info(
        "[EP-WAVE-END][%s] batch_id=%d, layer_id=%d, "
        "wave_start_time_s=%.12f, combine_barrier_end_time_s=%.12f, "
        "post_combine_time_ms=%.12f, wave_end_time_s=%.12f, "
        "wave_time_ms=%.12f, %s",
        cluster_name,
        batch_id,
        layer_id,
        float(wave_start_time_s),
        float(combine_barrier_end_time_s),
        float(post_combine_time_ms),
        float(wave_end_time_s),
        wave_time_ms,
        format_identity(trace_identity),
    )


def log_barrier_trace(
    *,
    cluster_type: ClusterType,
    batch_id: int,
    layer_id: int,
    phase: str,
    expected_ep_ids: Tuple[int, ...],
    arrived_ep_ids: Tuple[int, ...],
    max_lane_time_ms: float,
    collective_time_ms: float,
    barrier_time_ms: float,
    barrier_start_time_s: float,
    barrier_end_time_s: float,
    trace_identity: Dict[str, Any],
    format_identity: Callable[[Dict[str, Any]], str],
) -> None:
    """Validate and emit one complete EP barrier trace."""

    if phase not in {"dispatch", "combine"}:
        raise ValueError(f"unsupported EP barrier phase: {phase!r}")
    if type(batch_id) is not int or batch_id < 0:
        raise ValueError("EP barrier batch_id must be a non-negative int")
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError("EP barrier layer_id must be a non-negative int")
    expected = tuple(sorted(expected_ep_ids))
    arrived = tuple(sorted(arrived_ep_ids))
    if not expected or arrived != expected:
        raise ValueError(
            "EP barrier must log the complete participant set: "
            f"expected={expected!r}, arrived={arrived!r}"
        )
    for name, value in (
        ("max_lane_time_ms", max_lane_time_ms),
        ("collective_time_ms", collective_time_ms),
        ("barrier_time_ms", barrier_time_ms),
        ("barrier_start_time_s", barrier_start_time_s),
        ("barrier_end_time_s", barrier_end_time_s),
    ):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"EP barrier {name} must be a real number")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(
                f"EP barrier {name} must be finite and non-negative, got {value!r}"
            )
    if float(barrier_time_ms) < float(max_lane_time_ms):
        raise ValueError(
            "EP barrier time cannot be shorter than the slowest lane: "
            f"barrier={barrier_time_ms!r}, max_lane={max_lane_time_ms!r}"
        )
    expected_barrier_time_ms = float(max_lane_time_ms) + float(collective_time_ms)
    if not math.isclose(float(barrier_time_ms), expected_barrier_time_ms, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "EP barrier time does not equal lane arrival plus collective time: "
            f"barrier={barrier_time_ms!r}, max_lane={max_lane_time_ms!r}, "
            f"collective={collective_time_ms!r}, expected={expected_barrier_time_ms!r}"
        )
    expected_end_time_s = float(barrier_start_time_s) + float(barrier_time_ms) * 1e-3
    if not math.isclose(float(barrier_end_time_s), expected_end_time_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "EP barrier end time does not match start plus duration: "
            f"start={barrier_start_time_s!r}, duration_ms={barrier_time_ms!r}, "
            f"end={barrier_end_time_s!r}, expected={expected_end_time_s!r}"
        )
    cluster_name = getattr(cluster_type, "name", str(cluster_type))
    logger.info(
        "[EP-BARRIER][%s] batch_id=%d, layer_id=%d, phase=%s, "
        "expected_ep_ids=%s, arrived_ep_ids=%s, max_lane_time_ms=%.12f, "
        "collective_time_ms=%.12f, barrier_time_ms=%.12f, "
        "barrier_start_time_s=%.12f, barrier_end_time_s=%.12f, %s",
        cluster_name,
        batch_id,
        layer_id,
        phase,
        list(expected),
        list(arrived),
        float(max_lane_time_ms),
        float(collective_time_ms),
        float(barrier_time_ms),
        float(barrier_start_time_s),
        float(barrier_end_time_s),
        format_identity(trace_identity),
    )


def log_conservation_trace(
    *,
    cluster_type: ClusterType,
    batch_id: int,
    layer_id: int,
    routing_token_count: int,
    router_topk: int,
    total_routed_assignments: int,
    per_ep_routed_tokens: Dict[int, int],
    trace_identity: Dict[str, Any],
    format_identity: Callable[[Dict[str, Any]], str],
) -> None:
    """Validate and emit EP routing token conservation."""

    if type(batch_id) is not int or batch_id < 0:
        raise ValueError("EP conservation batch_id must be non-negative")
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError("EP conservation layer_id must be non-negative")
    for name, value, minimum in (
        ("routing_token_count", routing_token_count, 0),
        ("router_topk", router_topk, 1),
        ("total_routed_assignments", total_routed_assignments, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"EP conservation {name} is invalid: {value!r}")
    if routing_token_count * router_topk != total_routed_assignments:
        raise ValueError(
            "EP conservation total must equal routing_token_count * router_topk"
        )
    if not isinstance(per_ep_routed_tokens, dict) or not per_ep_routed_tokens:
        raise ValueError("EP conservation requires a non-empty per-EP map")
    normalized: dict[int, int] = {}
    for ep_id, token_count in per_ep_routed_tokens.items():
        if type(ep_id) is not int or ep_id < 0 or type(token_count) is not int or token_count < 0:
            raise ValueError("EP conservation per-EP values must be non-negative ints")
        normalized[ep_id] = token_count
    if sum(normalized.values()) != total_routed_assignments:
        raise ValueError("EP conservation per-EP total does not match routed assignments")
    cluster_name = getattr(cluster_type, "name", str(cluster_type))
    logger.info(
        "[EP-CONSERVATION][%s] batch_id=%d, layer_id=%d, "
        "routing_token_count=%d, router_topk=%d, total_routed_assignments=%d, "
        "per_ep_routed_tokens=%s, %s",
        cluster_name, batch_id, layer_id, routing_token_count, router_topk,
        total_routed_assignments, dict(sorted(normalized.items())),
        format_identity(trace_identity),
    )
