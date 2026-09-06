"""Shared scheduling implementation for PREFILL and DECODE EP waves."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
)


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
    """Schedule one PREFILL or DECODE layer through the existing scheduler APIs.

    The scheduler remains the owner of model prediction, stage admission, and
    waiting-room callbacks. This function only combines the identical wave
    control flow used by the two cluster modes.
    """

    if mode not in ("prefill", "decode"):
        raise ValueError(f"unsupported EP wave mode: {mode!r}")
    mode_name = mode.capitalize()
    if not isinstance(time, Real) or not math.isfinite(float(time)):
        raise ValueError(f"{mode} EP wave time must be finite")
    time = float(time)

    from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs

    wave_inputs = prepare_ep_wave_inputs(
        source_batches=scheduler._forward_step_source_batches(cohort_batches, batch),
        batch=batch,
        step_id_getter=scheduler._get_forward_step_id,
        aggregate_batch_builder=scheduler._create_virtual_global_batch,
    )
    source_batches = wave_inputs.source_batches
    cohort_id = wave_inputs.step_id
    for lane_id, source_batch in source_batches.items():
        if not hasattr(source_batch, "_stage_owner_replica_local_id"):
            source_batch._stage_owner_replica_local_id = (
                replica_local_id if replica_local_id is not None else lane_id
            )

    model_config = scheduler._config.replica_config.model_config
    predictor = scheduler._predictor
    non_idle_source_batches = list(wave_inputs.non_idle_batches)
    layer_workload = None
    lane_compute_times_ms: list[float] = []
    if model_config.is_moe_layer(layer_id):
        plan = scheduler._prepare_moe_ep_wave_plan(
            wave_inputs=wave_inputs,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
        )
        layer_workload = plan.layer_workload
        phase_times = plan.phase_times
        lane_compute_times_ms = list(phase_times.lane_compute_times_ms)
        scheduler._promote_forward_step_to_ep_wave(
            source_batches=source_batches,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
            cohort_id=cohort_id,
            participant_ep_ids=tuple(layer_workload.participant_ep_ids),
        )
    else:
        event_cls = (
            _load_prefill_dense_event()
            if mode == "prefill"
            else _load_decode_dense_event()
        )
        dense_events = []
        for source_batch in non_idle_source_batches:
            execution_time = predictor.predict_stage_execution_time(
                source_batch,
                stage_id,
                cluster_type=scheduler._cluster_type,
                num_layers=1,
                layer_id=layer_id,
            )
            post_attention_getter = getattr(
                execution_time, "get_single_layer_post_attention_time", None
            )
            if not callable(post_attention_getter):
                raise ValueError(
                    f"{mode_name} dense predictor result is missing post-attention timing"
                )
            dense_time_ms = float(post_attention_getter())
            if not math.isfinite(dense_time_ms) or dense_time_ms < 0:
                raise ValueError(
                    f"{mode_name} dense post-attention time must be finite and non-negative"
                )
            scheduler.transition_stage_admission_for_layer(
                source_batch,
                stage_id=stage_id,
                layer_id=layer_id,
                operation_kind="ffn",
                scope=FULL_STAGE_WORLD,
            )
            component_ledger = getattr(
                source_batch,
                "_prefill_model_execution_components_ms_by_stage",
                None,
            )
            if mode == "prefill":
                if (
                    not isinstance(component_ledger, dict)
                    or stage_id not in component_ledger
                    or not isinstance(component_ledger[stage_id], list)
                ):
                    raise ValueError(
                        "missing PREFILL model-execution component ledger for dense layer: "
                        f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                        f"batch={source_batch.id}"
                    )
                component_ledger[stage_id].append(dense_time_ms)
            dense_events.append(
                event_cls(
                    time + dense_time_ms * 1e-3,
                    replica_id,
                    stage_id,
                    source_batch,
                    layer_id,
                    mode,
                    scheduler._cluster_type,
                )
            )
        return dense_events

    if not lane_compute_times_ms:
        raise ValueError(f"{mode_name} layer wave produced no participant timing")
    timing = plan.timing
    barrier_end_time_s = timing.wave_end_time_s
    if mode == "prefill":
        wave_time_ms = (
            timing.dispatch_barrier_time_ms
            + timing.combine_barrier_time_ms
            + timing.post_combine_barrier_time_ms
        )
        for source_batch in non_idle_source_batches:
            component_ledger = getattr(
                source_batch,
                "_prefill_model_execution_components_ms_by_stage",
                None,
            )
            if (
                not isinstance(component_ledger, dict)
                or stage_id not in component_ledger
                or not isinstance(component_ledger[stage_id], list)
            ):
                raise ValueError(
                    "missing PREFILL model-execution component ledger for EP wave: "
                    f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                    f"batch={source_batch.id}"
                )
            component_ledger[stage_id].append(wave_time_ms)
            source_batch._prefill_ep_wave_lane_times_ms = tuple(lane_compute_times_ms)
            source_batch._prefill_ep_wave_workload = layer_workload
    else:
        for source_batch in non_idle_source_batches:
            source_batch._decode_ep_wave_lane_times_ms = tuple(lane_compute_times_ms)

    waiting_room = (
        scheduler._prefill_sync_waiting_room
        if mode == "prefill"
        else scheduler._decode_sync_waiting_room
    )
    sync_room = waiting_room[replica_id][stage_id][cohort_id][layer_id]["post_moe"]
    if sync_room["batches"]:
        raise ValueError(
            f"{mode_name} EP wave post_moe room already contains a batch: "
            f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
            f"forward_cohort_id={cohort_id}"
        )
    sync_room["batches"].update(source_batches)
    sync_room["arrival_times"].update(
        {lane_id: barrier_end_time_s for lane_id in source_batches}
    )
    event_cls = (
        _load_prefill_sync_event()
        if mode == "prefill"
        else _load_decode_sync_event()
    )
    return [
        event_cls(
            barrier_end_time_s,
            replica_id,
            stage_id,
            cohort_id,
            "post_moe",
            layer_id,
            cluster_type=scheduler._cluster_type,
        )
    ]


def _load_prefill_dense_event():
    from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent

    return DenseLayerCompleteEvent


def _load_decode_dense_event():
    from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent

    return DenseLayerCompleteEvent


def _load_prefill_sync_event():
    from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent

    return PrefillSyncCollectiveEvent


def _load_decode_sync_event():
    from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent

    return DecodeSyncCollectiveEvent
