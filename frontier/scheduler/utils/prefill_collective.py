"""PREFILL collective completion orchestration helpers."""

from typing import Any, Optional

from frontier.entities import Batch
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
)
from frontier.scheduler.utils.collective_timing import (
    attention_delay_seconds,
    prepare_prefill_final_timing,
    select_active_batch,
)


def handle_prefill_sync_collective(
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
    """Handle completion of a canonical layer-local PREFILL EP wave."""

    # Event modules import scheduler registries, so load them after the
    # scheduler package has finished initialization.
    from frontier.events.batch_stage_end_event import BatchStageEndEvent
    from frontier.events.prefill_sync_event import PrefillSyncEvent

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)

    if direct_batch is not None:
        if sync_stage != "post_moe":
            raise ValueError(
                "Direct dense PREFILL completion is valid only for post_moe transition"
            )
        sync_wait_room = {
            "batches": {None: direct_batch},
            "arrival_times": {None: time},
        }
        participant_batches = sync_wait_room["batches"]
    else:
        if sync_stage not in scheduler._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id]:
            logger.debug(
                f"[PREFILL_SYNC][COLLECTIVE_SKIP] sync_stage={sync_stage} already processed for "
                f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
            )
            return []

        sync_wait_room = scheduler._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id].pop(sync_stage)
        participant_batches = sync_wait_room["batches"]

    participant_keys = list(participant_batches.keys())
    logger.info(
        f"[PREFILL_SYNC][COLLECTIVE] ENTER: t={time:.6f}s, replica={replica_id}, stage={stage_id}, "
        f"layer={layer_id}, sync_stage={sync_stage}, batch_global_id={batch_global_id}, "
        f"participant_keys={participant_keys}, "
        f"participant_batches_type={type(participant_batches).__name__}"
    )

    if sync_stage != "post_moe":
        raise ValueError(
            "PREFILL collective completion accepts only post_moe for the "
            "canonical per-layer EP protocol"
        )

    events = []
    sample_batch = select_active_batch(participant_batches)
    if sample_batch is None:
        logger.warning(
            f"[PREFILL_SYNC][COLLECTIVE] post_moe has no non-idle batch for "
            f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
        )
        return events

    execution_time = scheduler._predictor.predict_stage_execution_time(
        sample_batch,
        stage_id,
        cluster_type=scheduler._cluster_type,
        num_layers=1,
        layer_id=layer_id,
        include_ffn=False,
    )

    num_layers = scheduler._predictor._num_layers_per_pipeline_stage
    _, stage_layer_end = scheduler.get_pipeline_stage_layer_bounds(
        stage_id,
        num_layers,
    )
    next_layer_id = layer_id + 1
    restored_full_stage_owners = scheduler._restore_cohort_full_stage_owners(
        source_batches=participant_batches,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=next_layer_id,
        cohort_id=batch_global_id,
        operation_kind=(
            "attention" if next_layer_id < stage_layer_end else "final"
        ),
    )

    if layer_id < stage_layer_end - 1:
        next_layer_execution_time = scheduler._predictor.predict_stage_execution_time(
            sample_batch,
            stage_id,
            cluster_type=scheduler._cluster_type,
            num_layers=1,
            layer_id=next_layer_id,
            include_ffn=False,
        )
        attention_time_ms = next_layer_execution_time.get_single_layer_attention_scope_time()
        attention_time = attention_delay_seconds(next_layer_execution_time)
        total_time_to_next_sync = attention_time

        for replica_local_id, batch in participant_batches.items():
            if batch.is_idle:
                logger.info(
                    f"[PREFILL_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling for idle batch {batch.id} "
                    f"(replica={replica_id}, replica_local_id={replica_local_id}, "
                    f"layer={layer_id})"
                )
                continue
            component_ledger = getattr(
                batch,
                "_prefill_model_execution_components_ms_by_stage",
                None,
            )
            if (
                not isinstance(component_ledger, dict)
                or stage_id not in component_ledger
                or not isinstance(component_ledger[stage_id], list)
            ):
                raise ValueError(
                    "missing PREFILL model-execution component ledger: "
                    f"replica={replica_id}, replica_local_id={replica_local_id}, "
                    f"stage={stage_id}, layer={layer_id}, "
                    f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                )
            component_ledger[stage_id].append(attention_time_ms)
            if not restored_full_stage_owners:
                scheduler.transition_stage_admission_for_layer(
                    batch,
                    stage_id=stage_id,
                    layer_id=next_layer_id,
                    operation_kind="attention",
                    scope=FULL_STAGE_WORLD,
                )
            events.append(
                PrefillSyncEvent(
                    time + total_time_to_next_sync,
                    replica_id,
                    stage_id,
                    batch,
                    getattr(
                        batch,
                        "_stage_owner_replica_local_id",
                        None,
                    ),
                    "pre_moe",
                    next_layer_id,
                    total_time_to_next_sync,
                    cluster_type=scheduler._cluster_type,
                )
            )
    else:
        for replica_local_id, batch in participant_batches.items():
            if batch.is_idle:
                logger.info(
                    f"[PREFILL_SYNC][IDLE_SKIP] Skip final stage-end for idle batch {batch.id} "
                    f"(replica={replica_id}, replica_local_id={replica_local_id}, "
                    f"layer={layer_id})"
                )
                continue

            stage_identity = getattr(
                batch,
                "_stage_owner_replica_local_id",
                None,
            )
            stage_scheduler = scheduler.get_replica_stage_scheduler(
                replica_id, stage_identity, stage_id
            )
            is_last_stage = stage_scheduler.is_last_stage
            pipeline_time = execution_time.pipeline_time * 1e-3
            if not hasattr(batch, "_prefill_stage_start_time"):
                raise ValueError(
                    "missing PREFILL stage start time: "
                    f"replica={replica_id}, replica_local_id={replica_local_id}, "
                    f"stage={stage_id}, layer={layer_id}, "
                    f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                )
            original_start_time = batch._prefill_stage_start_time
            component_ledger = getattr(
                batch,
                "_prefill_model_execution_components_ms_by_stage",
                None,
            )
            if (
                not isinstance(component_ledger, dict)
                or stage_id not in component_ledger
                or not isinstance(component_ledger[stage_id], list)
                or not component_ledger[stage_id]
            ):
                raise ValueError(
                    "missing PREFILL model-execution component ledger: "
                    f"replica={replica_id}, replica_local_id={replica_local_id}, "
                    f"stage={stage_id}, layer={layer_id}, "
                    f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                )
            final_timing = prepare_prefill_final_timing(
                execution_time,
                component_ledger[stage_id],
                time,
                original_start_time,
            )
            actual_model_execution_time = (
                final_timing.explicit_model_time + final_timing.pipeline_time
            )
            batch_stage, _ = stage_scheduler.predict_and_create_stage(
                batch, skip_get_execution_time=True
            )
            batch_stage.on_schedule(original_start_time)
            batch_stage.override_execution_time(final_timing.actual_execution_time)
            batch_stage.override_model_execution_time(actual_model_execution_time)

            corrected_execution_time = scheduler._create_prefill_corrected_execution_time_for_metrics(
                sample_batch,
                stage_id,
                execution_time,
                final_timing.actual_execution_time,
                original_start_time,
            )
            metrics_store.on_replica_stage_schedule(
                original_start_time,
                replica_id,
                stage_id,
                batch_stage,
                corrected_execution_time,
                scheduler._cluster_type,
                stage_identity,
            )
            events.append(
                BatchStageEndEvent(
                    final_timing.completion_time,
                    replica_id,
                    stage_id,
                    is_last_stage,
                    batch,
                    batch_stage,
                    scheduler._cluster_type,
                    stage_identity,
                )
            )
            if scheduler._should_trigger_kv_transfer(batch):
                events.extend(
                    scheduler._create_kv_transfer_events(
                        final_timing.completion_time,
                        batch,
                        replica_id,
                        stage_identity,
                    )
                )

    return events
