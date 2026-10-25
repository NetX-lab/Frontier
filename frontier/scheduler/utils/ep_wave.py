"""Shared planning for PREFILL and DECODE expert-parallel waves."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.scheduler.utils.expert_parallel import (
    EPWavePhaseTimes,
    calculate_ep_wave_timing,
    predict_ep_wave_phase_times,
)


class EPWavePlan(NamedTuple):
    """Pure inputs and timing for one shared MoE EP wave."""

    wave_inputs: Any
    layer_workload: Any
    participant_ep_ids: tuple[int, ...]
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
    capture_lane_timings: bool = False,
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
        cluster_type=cluster_type,
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
        capture_lane_timings=capture_lane_timings,
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
    capture_lane_timings: bool = False,
) -> EPWavePlan:
    """Build the MoE portion when lane inputs have already been normalized."""

    aggregate_batch = wave_inputs.aggregate_batch
    sample_batch = wave_inputs.sample_batch
    cohort_id = wave_inputs.step_id
    model = getattr(predictor, "_model_config", None)
    v4 = model is not None and model.get_model_architecture_profile().profile_id == "deepseek_v4"
    physical_rows = {}
    sp_token_widths = None
    if v4:
        from frontier.execution_time_predictor.deepseek_v4 import source_physical_rows

        physical_rows["routing_token_count"] = source_physical_rows(predictor, wave_inputs.source_batches)
    elif model is not None:
        replica = predictor._get_cluster_replica_config(cluster_type)
        if model.get_model_architecture_profile().uses_sequence_parallel_moe(
            replica.attn_tensor_parallel_size, replica.attn_dp,
            replica.num_pipeline_stages, replica.moe_expert_parallel_size,
        ):
            from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths

            padded = wave_inputs.padded_lane_token_counts(cluster_type)
            widths = sequence_parallel_moe_token_widths(
                [padded[d] for d in range(replica.attn_dp)],
                replica.attn_tensor_parallel_size,
            )
            sp_token_widths = widths
            physical_rows["routing_token_count"] = widths.gathered_tokens
    trace_identity = trace_identity_builder(
        batch=sample_batch,
        replica_id=replica_id,
        stage_id=stage_id,
        operation_id=int(cohort_id),
        operation_kind="ep_ffn",
    )
    if v4 and predictor._dsv4_step_costs is not None and not capture_lane_timings:
        # The measured lane step already contains this layer's FFN and EP work
        # (predict_step_cost_execution), so every participant's phases are zero.
        layer_workload = None
        participant_ep_ids = tuple(range(predictor._replica_config.moe_expert_parallel_size))
        zeros = (0.0,) * len(participant_ep_ids)
        phase_times = EPWavePhaseTimes(zeros, zeros, zeros, zeros, zeros, zeros)
    else:
        layer_workload = materialize_workload(
            batch=aggregate_batch,
            target_replica_id=replica_id,
            global_layer_id=layer_id,
            **physical_rows,
        )
        participant_ep_ids = tuple(layer_workload.participant_ep_ids)
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
            capture_lane_timings=capture_lane_timings,
            source_batches=wave_inputs.source_batches if v4 else None,
            sp_token_widths=sp_token_widths,
        )
    timing = calculate_ep_wave_timing(
        start_time_s=float(time),
        phases=phase_times,
        barrier_logger=barrier_logger,
        wave_logger=wave_logger,
        cluster_type=cluster_type,
        batch_id=int(cohort_id),
        layer_id=layer_id,
        participant_ep_ids=participant_ep_ids,
        trace_identity=trace_identity,
    )
    return EPWavePlan(
        wave_inputs=wave_inputs,
        layer_workload=layer_workload,
        participant_ep_ids=participant_ep_ids,
        phase_times=phase_times,
        timing=timing,
        trace_identity=trace_identity,
    )
