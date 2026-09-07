"""Shared planning for PREFILL and DECODE expert-parallel waves."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.scheduler.utils.expert_parallel import (
    calculate_ep_wave_timing,
    predict_ep_wave_phase_times,
)


class EPWavePlan(NamedTuple):
    """Pure inputs and timing for one shared MoE EP wave."""

    wave_inputs: Any
    layer_workload: Any
    phase_times: Any
    timing: Any
    trace_identity: Any


def prepare_moe_wave(
    *,
    source_batches: dict[int, Any],
    batch: Any,
    time: float,
    step_id_getter: Callable[[Any], int],
    aggregate_batch_builder: Callable[[Any, int, int], Any],
    materialize_workload: Callable[..., Any],
    trace_identity_builder: Callable[..., Any],
    conservation_logger: Callable[..., None],
    predictor: Any,
    lane_builder: Callable[..., Any],
    phase_getter: Callable[..., Any],
    workload_logger: Callable[..., None],
    barrier_logger: Callable[..., None],
    wave_logger: Callable[..., None],
    cluster_type: Any,
    replica_id: int,
    stage_id: int,
    layer_id: int,
) -> EPWavePlan:
    """Prepare workload, phase timing, and trace data without state mutation."""

    if not isinstance(time, (int, float)) or isinstance(time, bool):
        raise ValueError("EP wave time must be finite")
    normalized_time = float(time)
    wave_inputs = prepare_ep_wave_inputs(
        source_batches=source_batches,
        batch=batch,
        step_id_getter=step_id_getter,
        aggregate_batch_builder=aggregate_batch_builder,
    )
    return prepare_moe_wave_from_inputs(
        wave_inputs=wave_inputs,
        time=normalized_time,
        materialize_workload=materialize_workload,
        trace_identity_builder=trace_identity_builder,
        conservation_logger=conservation_logger,
        predictor=predictor,
        lane_builder=lane_builder,
        phase_getter=phase_getter,
        workload_logger=workload_logger,
        barrier_logger=barrier_logger,
        wave_logger=wave_logger,
        cluster_type=cluster_type,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
    )


def prepare_moe_wave_from_inputs(
    *,
    wave_inputs: Any,
    time: float,
    materialize_workload: Callable[..., Any],
    trace_identity_builder: Callable[..., Any],
    conservation_logger: Callable[..., None],
    predictor: Any,
    lane_builder: Callable[..., Any],
    phase_getter: Callable[..., Any],
    workload_logger: Callable[..., None],
    barrier_logger: Callable[..., None],
    wave_logger: Callable[..., None],
    cluster_type: Any,
    replica_id: int,
    stage_id: int,
    layer_id: int,
) -> EPWavePlan:
    """Build the MoE portion when lane inputs have already been normalized."""

    aggregate_batch = wave_inputs.aggregate_batch
    sample_batch = wave_inputs.sample_batch
    cohort_id = wave_inputs.step_id
    layer_workload = materialize_workload(
        batch=aggregate_batch,
        target_replica_id=replica_id,
        global_layer_id=layer_id,
    )
    trace_identity = trace_identity_builder(
        batch=sample_batch,
        replica_id=replica_id,
        stage_id=stage_id,
        operation_id=int(cohort_id),
        operation_kind="ep_ffn",
    )
    conservation_logger(
        cluster_type=cluster_type,
        batch_id=int(cohort_id),
        layer_id=layer_id,
        routing_token_count=int(layer_workload.routing_token_count),
        router_topk=int(layer_workload.router_topk),
        total_routed_assignments=int(layer_workload.total_routed_assignments),
        per_ep_routed_tokens=dict(layer_workload.per_ep_routed_tokens),
        trace_identity=trace_identity,
    )
    phase_times = predict_ep_wave_phase_times(
        layer_workload=layer_workload,
        source_batch=aggregate_batch,
        stage_id=stage_id,
        layer_id=layer_id,
        cluster_type=cluster_type,
        predictor=predictor,
        lane_builder=lane_builder,
        phase_getter=phase_getter,
        workload_logger=workload_logger,
        trace_identity=trace_identity,
        batch_id=int(cohort_id),
    )
    timing = calculate_ep_wave_timing(
        start_time_s=float(time),
        phases=phase_times,
        barrier_logger=barrier_logger,
        wave_logger=wave_logger,
        cluster_type=cluster_type,
        batch_id=int(cohort_id),
        layer_id=layer_id,
        participant_ep_ids=tuple(layer_workload.participant_ep_ids),
        trace_identity=trace_identity,
    )
    return EPWavePlan(
        wave_inputs=wave_inputs,
        layer_workload=layer_workload,
        phase_times=phase_times,
        timing=timing,
        trace_identity=trace_identity,
    )
