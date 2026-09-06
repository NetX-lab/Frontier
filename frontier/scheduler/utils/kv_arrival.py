"""Architecture-specific KV-cache arrival handlers."""

from typing import Any


def handle_decode_attn_arrival(scheduler: Any, time: float, batch: Any, transfer_info: Any, logger: Any) -> list:
    """Handoff requests to DECODE_ATTN and trigger its next schedule."""
    request_ids = [request.id for request in batch.requests]
    logger.info(
        "Decode-attn cluster received KV cache at %.3fs: requests %s, "
        "batch_id=%s, transfer_size=%s bytes, source_cluster=%s",
        time,
        request_ids,
        batch.id,
        transfer_info.kv_cache_size_bytes,
        transfer_info.source_cluster_type.name,
    )

    queue_was_empty = len(scheduler._request_queue) == 0
    for request in batch.requests:
        request.on_disaggregated_decode_handoff(time, scheduler._cluster_type)
        request.on_arrival(time, scheduler._cluster_type)
        scheduler.add_request(request)
        logger.info(
            "Request %s added to decode-attn cluster queue, prefill_tokens=%s, "
            "decode_tokens=%s, num_processed_tokens=%s, total_tokens=%s, "
            "is_prefill_complete=%s, current_decode_token_index=%s, "
            "completed_layer_count=%s.",
            request.id,
            request.num_prefill_tokens,
            request.num_decode_tokens,
            request.num_processed_tokens,
            request.total_tokens,
            request.is_prefill_complete,
            request.current_decode_token_index,
            request.completed_layer_count,
        )

    if scheduler._is_periodic_scheduling_enabled:
        logger.info(
            "Requests cached for periodic scheduling (interval=%sms), current queue size: %s",
            scheduler._periodic_scheduling_interval_ms,
            len(scheduler._request_queue),
        )
        return []

    from frontier.config.global_vars import get_simulation_mode
    from frontier.events.cluster_schedule_event import ClusterScheduleEvent

    simulation_mode = get_simulation_mode()
    if not queue_was_empty:
        logger.info(
            "Decode-attn queue already has pending requests; skip redundant schedule trigger in %s mode",
            simulation_mode,
        )
        return []

    logger.info(
        "KV-cache arrival triggers immediate decode-attn scheduling in %s mode; queue size=%d",
        simulation_mode,
        len(scheduler._request_queue),
    )
    return [ClusterScheduleEvent(time, scheduler._cluster_type)]


def handle_decode_arrival(scheduler: Any, time: float, batch: Any, transfer_info: Any, logger: Any) -> list:
    """Handoff requests to unified DECODE and trigger scheduling."""
    request_ids = [request.id for request in batch.requests]
    logger.info(
        "Decode cluster received KV cache at %.3fs: requests %s, "
        "batch_id=%s, transfer_size=%s bytes, source_cluster=%s",
        time,
        request_ids,
        batch.id,
        transfer_info.kv_cache_size_bytes,
        transfer_info.source_cluster_type.name,
    )

    for request in batch.requests:
        request.on_arrival(time, scheduler._cluster_type)
        scheduler.add_request(request)
        logger.info(
            "Request %s added to decode cluster queue, prefill_tokens=%s, "
            "decode_tokens=%s, num_processed_tokens=%s, total_tokens=%s, "
            "is_prefill_complete=%s, current_decode_token_index=%s, "
            "completed_layer_count=%s.",
            request.id,
            request.num_prefill_tokens,
            request.num_decode_tokens,
            request.num_processed_tokens,
            request.total_tokens,
            request.is_prefill_complete,
            request.current_decode_token_index,
            request.completed_layer_count,
        )

    if scheduler._is_periodic_scheduling_enabled:
        logger.info(
            "Requests cached for periodic scheduling (interval=%sms), current queue size: %s",
            scheduler._periodic_scheduling_interval_ms,
            len(scheduler._request_queue),
        )
        return []

    from frontier.config.global_vars import get_simulation_mode
    from frontier.events.cluster_schedule_event import ClusterScheduleEvent

    simulation_mode = get_simulation_mode()
    if simulation_mode == "offline":
        expected_num_requests = getattr(
            scheduler._request_generator_config, "num_decode_bound_requests", None
        )
        if expected_num_requests is None:
            raise ValueError(
                "Offline DECODE scheduling requires "
                "request_generator_config.num_decode_bound_requests to be set "
                "by request generation."
            )

        current_num_requests = len(scheduler._request_queue)
        if current_num_requests > expected_num_requests:
            raise ValueError(
                "Offline DECODE received more decode-bound requests than "
                f"expected: current={current_num_requests}, expected={expected_num_requests}"
            )
        if current_num_requests < expected_num_requests:
            logger.info(
                "Offline mode: buffering decode-bound requests (%s/%s), "
                "deferring scheduling until all decode-bound requests arrive",
                current_num_requests,
                expected_num_requests,
            )
            return []
        logger.info(
            "Offline mode: all %s decode-bound requests arrived, triggering batch scheduling",
            expected_num_requests,
        )
        return [ClusterScheduleEvent(time, scheduler._cluster_type)]

    logger.info(
        "Online mode: triggering immediate cluster scheduling for %s requests",
        len(batch.requests),
    )
    return [ClusterScheduleEvent(time, scheduler._cluster_type)]
