"""Build M2N transfer events for decode-attention return batches."""

from typing import Any


def current_layer_id(batch: Any) -> int:
    """Return the first incomplete request layer represented by a batch."""
    if not batch.requests:
        raise ValueError("_get_current_layer_id_from_batch: batch.requests is empty")
    for request in batch.requests:
        if not request.completed:
            return request.completed_layer_count
    return batch.requests[0].completed_layer_count


def build_aggregated_batch_transfer_events(
    scheduler: Any,
    batch: Any,
    current_time: float,
    *,
    source_replica_id: int,
    source_replica_local_id: int | None,
) -> list:
    """Create the F2A transfer event for one aggregated DECODE_FFN batch."""

    from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
    from frontier.types import ClusterType
    from frontier.logger import get_cluster_logger

    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)
    logger.info(
        f"[DEBUG] _create_m2n_transfer_events_for_aggregated_batch called: "
        f"batch_id={batch.id}, time={current_time:.3f}s, "
        f"num_requests={len(batch.requests)}"
    )
    activation_size, transfer_time = scheduler._m2n_transfer_predictor.get_transfer_info(
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        batch=batch,
        replica_config=scheduler._config.replica_config,
    )
    layer_id = scheduler._get_current_layer_id_from_batch(batch)
    event = M2NTransferStartEvent(
        time=current_time,
        source_replica_id=batch.decode_attn_original_replica_id,
        source_replica_local_id=batch.decode_attn_original_replica_local_id,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        batch=batch,
        activation_size_bytes=activation_size,
        transfer_time_ms=transfer_time,
        layer_id=layer_id,
        afd_stage_idx=batch.afd_stage_idx,
        source_execution_replica_id=source_replica_id,
        source_execution_replica_local_id=source_replica_local_id,
        target_execution_replica_id=batch.decode_attn_original_replica_id,
        target_execution_replica_local_id=batch.decode_attn_original_replica_local_id,
    )
    try:
        req_ids = [request.id for request in batch.requests]
        logger.info(
            f"[M2N][F2A][CREATE] batch_id={batch.id} reqs={req_ids} "
            f"batch_global_id={getattr(batch, 'global_id', '?')} "
            f"decode_attn_orig=(replica={getattr(batch, 'decode_attn_original_replica_id', '?')},dp={getattr(batch, 'decode_attn_original_replica_local_id', '?')}) "
            f"target={ClusterType.DECODE_ATTN.name} size={activation_size}B t_ms={transfer_time:.3f}"
        )
    except Exception:
        logger.info(f"[M2N][F2A][CREATE] batch_id={batch.id} (details unavailable)")
    return [event]
