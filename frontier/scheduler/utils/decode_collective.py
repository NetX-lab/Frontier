"""DECODE collective completion orchestration."""

from __future__ import annotations

from typing import Any, Optional

from frontier.entities import Batch
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import FULL_STAGE_WORLD
from frontier.scheduler.utils.collective_timing import (
    attention_delay_seconds,
    prepare_decode_final_timing,
    select_active_batch,
    validate_decode_layer_advance,
)
from frontier.scheduler.utils.request_selection import collect_active_requests


def handle_decode_sync_collective(
    scheduler: Any,
    time: float,
    replica_id: int,
    stage_id: int,
    batch_global_id: int,
    sync_stage: str,
    layer_id: int,
    metrics_store: Any,
    *,
    direct_batch: Optional[Batch] = None,
):
    """Complete one DECODE layer and schedule the next stage transition."""

    from frontier.logger import get_cluster_logger
    from frontier.events.batch_stage_end_event import BatchStageEndEvent
    from frontier.events.decode_sync_event import DecodeSyncEvent

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)
    if direct_batch is not None:
        if sync_stage != "post_moe":
            raise ValueError(
                "Direct dense DECODE completion is valid only for post_moe transition"
            )
        dp_batches = {None: direct_batch}
    else:
        rooms = scheduler._decode_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id]
        if sync_stage not in rooms:
            logger.debug(
                "[DECODE_SYNC][COLLECTIVE_SKIP] sync_stage=%s already processed for "
                "replica=%s, stage=%s, batch_global_id=%s, layer=%s",
                sync_stage, replica_id, stage_id, batch_global_id, layer_id,
            )
            return []
        dp_batches = rooms.pop(sync_stage)["batches"]

    logger.info(
        "[DECODE_SYNC][COLLECTIVE] ENTER: t=%.6fs, replica=%s, stage=%s, "
        "layer=%s, sync_stage=%s, batch_global_id=%s, dp_keys=%s",
        time, replica_id, stage_id, layer_id, sync_stage, batch_global_id,
        list(dp_batches),
    )
    sample_batch = select_active_batch(dp_batches) or next(iter(dp_batches.values()))
    canonical_ep_wave = hasattr(sample_batch, "_decode_ep_wave_lane_times_ms")
    if direct_batch is None and not canonical_ep_wave:
        raise RuntimeError(
            "Legacy DECODE aggregate synchronization is removed; collective "
            "completion requires a canonical EP_WAVE or dense full-stage handoff"
        )
    if sync_stage == "pre_moe":
        raise ValueError(
            "DECODE collective completion cannot start at pre_moe; the canonical "
            "EP_WAVE enters this method at post_moe"
        )

    stage_identity = getattr(sample_batch, "_stage_owner_replica_local_id", None)
    stage_scheduler = scheduler.get_replica_stage_scheduler(replica_id, stage_identity, stage_id)
    predictor = stage_scheduler._execution_time_predictor
    active_requests = collect_active_requests(dp_batches.values())
    validate_decode_layer_advance(
        active_requests,
        scheduler._config.replica_config.model_config.num_layers,
    )
    for request in active_requests:
        request.mb_on_step_layer_count_increment(num_layers_completed=1)

    num_layers = predictor._num_layers_per_pipeline_stage
    bounds_getter = getattr(scheduler, "get_pipeline_stage_layer_bounds", None)
    if callable(bounds_getter):
        _, stage_layer_end = bounds_getter(stage_id, num_layers)
    else:
        # Lightweight scheduler fixtures call the Base static helper directly;
        # preserve the same half-open stage bound when that method is absent.
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError("pipeline stage_id must be an exact non-negative int")
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError(
                "num_layers_per_pipeline_stage must be an exact positive int"
            )
        stage_layer_end = (stage_id + 1) * num_layers
    next_layer_id = layer_id + 1
    restored_full_stage_owners = scheduler._restore_cohort_full_stage_owners(
        source_batches=dp_batches,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=next_layer_id,
        cohort_id=batch_global_id,
        operation_kind="attention" if next_layer_id < stage_layer_end else "final",
    )

    if next_layer_id < stage_layer_end:
        next_execution = predictor.predict_stage_execution_time(
            sample_batch, stage_id, scheduler._cluster_type,
            num_layers=1, layer_id=next_layer_id, include_ffn=False,
        )
        attention_time = attention_delay_seconds(next_execution)
        events = []
        for participant_id, batch in dp_batches.items():
            if batch.is_idle:
                logger.info(
                    "[DECODE_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling "
                    "for idle batch %s (replica=%s, lane=%s, layer=%s)",
                    batch.id, replica_id, participant_id, layer_id,
                )
                continue
            transition_identity = getattr(batch, "_stage_owner_replica_local_id", None)
            if not restored_full_stage_owners:
                scheduler.transition_stage_admission_for_layer(
                    batch, stage_id=stage_id, layer_id=next_layer_id,
                    operation_kind="attention", scope=FULL_STAGE_WORLD,
                )
            events.append(
                DecodeSyncEvent(
                    time + attention_time, replica_id, stage_id, batch,
                    transition_identity, "pre_moe", next_layer_id,
                    attention_time, cluster_type=scheduler._cluster_type,
                )
            )
        logger.info(
            "[DECODE_SYNC][COLLECTIVE] post_moe completed, incremented layer count "
            "for %s unique requests, scheduled next layer pre_moe sync at t=%.6fs",
            len(active_requests), time + attention_time,
        )
        return events

    full_execution = predictor.predict_stage_execution_time(
        sample_batch, stage_id, scheduler._cluster_type,
        num_layers=num_layers, include_ffn=False,
    )
    final_timing = prepare_decode_final_timing(full_execution)
    events = []
    for participant_id, batch in dp_batches.items():
        if batch.is_idle:
            logger.info(
                "[DECODE_SYNC][IDLE_SKIP] Skip final stage-end for idle batch %s "
                "(replica=%s, lane=%s, layer=%s)",
                batch.id, replica_id, participant_id, layer_id,
            )
            continue
        scheduler._record_mtp_terminal_completion_delay(
            batch, final_timing.mtp_terminal_overshoot_time
        )
        transition_identity = getattr(batch, "_stage_owner_replica_local_id", None)
        stage_scheduler = scheduler.get_replica_stage_scheduler(
            replica_id, transition_identity, stage_id
        )
        batch_stage, _ = stage_scheduler.predict_and_create_stage(
            batch, skip_get_execution_time=True
        )
        original_start = getattr(
            batch, "_decode_stage_start_time",
            time - full_execution.total_time,
        )
        batch_stage.on_schedule(original_start)
        actual_execution = time + final_timing.total_time - original_start
        batch_stage.override_execution_time(actual_execution)
        batch_stage.override_model_execution_time(full_execution.model_time)
        corrected = scheduler._create_corrected_execution_time_for_metrics(
            full_execution, actual_execution, original_start
        )
        corrected._trace_execution_time_override = full_execution
        metrics_store.on_replica_stage_schedule(
            original_start, replica_id, stage_id, batch_stage,
            corrected, scheduler._cluster_type, transition_identity,
        )
        events.append(
            BatchStageEndEvent(
                time + final_timing.total_time,
                replica_id, stage_id, stage_scheduler.is_last_stage,
                batch, batch_stage, scheduler._cluster_type, transition_identity,
            )
        )
    logger.info(
        "[DECODE_SYNC][COLLECTIVE] Last layer completed, scheduled batch stage end at t=%.6fs",
        time + final_timing.total_time,
    )
    return events
