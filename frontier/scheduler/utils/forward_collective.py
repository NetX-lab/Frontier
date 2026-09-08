"""Complete a MONOLITHIC EP wave using its original source batches."""

from frontier.scheduler.utils.decode_collective import handle_decode_sync_collective
from frontier.scheduler.utils.prefill_collective import handle_prefill_sync_collective


def complete_forward_collective(
    scheduler, time, replica_id, stage_id, step_id, layer_id, metrics_store
):
    """Restore shared ownership once, then advance each local batch's shape."""
    stage_rooms = scheduler._prefill_sync_waiting_room[replica_id][stage_id]
    step_rooms = stage_rooms.pop(step_id)
    source_batches = step_rooms[layer_id]["post_moe"]["batches"]
    _, layer_end = scheduler.get_pipeline_stage_layer_bounds(
        stage_id, scheduler._predictor._num_layers_per_pipeline_stage
    )
    owners_restored = scheduler._restore_forward_step_full_stage_owners(
        source_batches=source_batches,
        replica_id=replica_id,
        stage_id=stage_id,
        layer_id=layer_id + 1,
        cohort_id=step_id,
        operation_kind="attention" if layer_id + 1 < layer_end else "final",
    )
    events = []
    for batch in source_batches.values():
        if batch.is_idle:
            continue
        handler = (
            handle_prefill_sync_collective
            if batch.num_prefill_tokens
            else handle_decode_sync_collective
        )
        events.extend(handler(
            scheduler, time, replica_id, stage_id, step_id, "post_moe",
            layer_id, metrics_store, direct_batch=batch,
            owners_restored=owners_restored,
        ))
    return events
