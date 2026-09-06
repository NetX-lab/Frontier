"""Shared scheduling for PREFILL and DECODE expert-parallel layer waves."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from frontier.scheduler.replica_stage_scheduler.stage_execution_context import FULL_STAGE_WORLD
from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs


def schedule_layer_wave(
    scheduler: Any,
    *,
    mode: str,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    layer_id: int,
    replica_local_id: int | None = None,
    cohort_batches: dict[int, Any] | None = None,
) -> list:
    """Run one shared layer wave and return its simulator events."""

    if mode not in ("prefill", "decode"):
        raise ValueError(f"unsupported shared layer-wave mode: {mode!r}")
    if not isinstance(time, Real) or not math.isfinite(float(time)):
        raise ValueError(f"{mode} EP wave time must be finite")
    wave_inputs = prepare_ep_wave_inputs(
        source_batches=scheduler._forward_step_source_batches(cohort_batches, batch),
        batch=batch,
        step_id_getter=scheduler._get_forward_step_id,
        aggregate_batch_builder=scheduler._create_virtual_global_batch,
    )
    source_batches = wave_inputs.source_batches
    step_id = wave_inputs.step_id
    for lane_id, source_batch in source_batches.items():
        if not hasattr(source_batch, "_stage_owner_replica_local_id"):
            source_batch._stage_owner_replica_local_id = (
                replica_local_id if replica_local_id is not None else lane_id
            )
    non_idle_batches = list(wave_inputs.non_idle_batches)
    model_config = scheduler._config.replica_config.model_config
    if not model_config.is_moe_layer(layer_id):
        return _schedule_dense_layer(
            scheduler,
            mode=mode,
            time=float(time),
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
            source_batches=non_idle_batches,
        )

    plan = scheduler._prepare_moe_ep_wave_plan(
        wave_inputs=wave_inputs,
        time=float(time),
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
    )
    lane_times_ms = list(plan.phase_times.lane_compute_times_ms)
    if not lane_times_ms:
        raise ValueError(f"{mode.capitalize()} layer wave produced no participant timing")
    scheduler._promote_forward_step_to_ep_wave(
        source_batches=source_batches,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id,
        cohort_id=step_id,
        participant_ep_ids=tuple(plan.layer_workload.participant_ep_ids),
    )
    timing = plan.timing
    barrier_end_time_s = timing.wave_end_time_s
    wave_time_ms = (
        timing.dispatch_barrier_time_ms
        + timing.combine_barrier_time_ms
        + timing.post_combine_barrier_time_ms
    )
    if mode == "prefill":
        for source_batch in non_idle_batches:
            ledger = getattr(source_batch, "_prefill_model_execution_components_ms_by_stage", None)
            if not isinstance(ledger, dict) or stage_id not in ledger or not isinstance(ledger[stage_id], list):
                raise ValueError(
                    "missing PREFILL model-execution component ledger for EP wave: "
                    f"replica={replica_id}, stage={stage_id}, layer={layer_id}, batch={source_batch.id}"
                )
            ledger[stage_id].append(wave_time_ms)
            source_batch._prefill_ep_wave_lane_times_ms = tuple(lane_times_ms)
            source_batch._prefill_ep_wave_workload = plan.layer_workload
        room = scheduler._prefill_sync_waiting_room[replica_id][stage_id][step_id][layer_id]["post_moe"]
    else:
        for source_batch in non_idle_batches:
            source_batch._decode_ep_wave_lane_times_ms = tuple(lane_times_ms)
        room = scheduler._decode_sync_waiting_room[replica_id][stage_id][step_id][layer_id]["post_moe"]
    if room["batches"]:
        raise ValueError(
            f"{mode.upper()} EP wave post_moe room already contains a batch: "
            f"replica={replica_id}, stage={stage_id}, layer={layer_id}, forward_cohort_id={step_id}"
        )
    room["batches"].update(source_batches)
    room["arrival_times"].update({lane_id: barrier_end_time_s for lane_id in source_batches})
    if mode == "prefill":
        from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent

        return [PrefillSyncCollectiveEvent(
            barrier_end_time_s, replica_id, stage_id, step_id, "post_moe", layer_id,
            cluster_type=scheduler._cluster_type,
        )]
    from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent

    return [DecodeSyncCollectiveEvent(
        barrier_end_time_s, replica_id, stage_id, step_id, "post_moe", layer_id,
        cluster_type=scheduler._cluster_type,
    )]


def _schedule_dense_layer(
    scheduler: Any,
    *,
    mode: str,
    time: float,
    replica_id: int,
    stage_id: int,
    layer_id: int,
    source_batches: list[Any],
) -> list:
    from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent

    events = []
    for source_batch in source_batches:
        execution_time = scheduler._predictor.predict_stage_execution_time(
            source_batch,
            stage_id,
            cluster_type=scheduler._cluster_type,
            num_layers=1,
            layer_id=layer_id,
        )
        getter = getattr(execution_time, "get_single_layer_post_attention_time", None)
        if not callable(getter):
            raise ValueError(f"{mode.capitalize()} dense predictor result is missing post-attention timing")
        dense_time_ms = float(getter())
        if not math.isfinite(dense_time_ms) or dense_time_ms < 0:
            raise ValueError(f"{mode.capitalize()} dense post-attention time must be finite and non-negative")
        scheduler.transition_stage_admission_for_layer(
            source_batch,
            stage_id=stage_id,
            layer_id=layer_id,
            operation_kind="ffn",
            scope=FULL_STAGE_WORLD,
        )
        if mode == "prefill":
            ledger = getattr(source_batch, "_prefill_model_execution_components_ms_by_stage", None)
            if not isinstance(ledger, dict) or stage_id not in ledger or not isinstance(ledger[stage_id], list):
                raise ValueError(
                    "missing PREFILL model-execution component ledger for dense layer: "
                    f"replica={replica_id}, stage={stage_id}, layer={layer_id}, batch={source_batch.id}"
                )
            ledger[stage_id].append(dense_time_ms)
        events.append(DenseLayerCompleteEvent(
            time + dense_time_ms * 1e-3,
            replica_id,
            stage_id,
            source_batch,
            layer_id,
            mode,
            scheduler._cluster_type,
        ))
    return events
