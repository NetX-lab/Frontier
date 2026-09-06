"""M2N arrival routing for disaggregated clusters."""

from typing import Any

from frontier.types import ClusterType


def route_m2n_arrival(
    scheduler: Any,
    time: float,
    batch: Any,
    transfer_info: Any,
    *,
    expected_roundtrip_inflight: bool,
    request_end_deferred: bool,
) -> list:
    """Validate and route one M2N arrival to its cluster handler."""
    from frontier.logger import get_cluster_logger

    if type(expected_roundtrip_inflight) is not bool:
        raise ValueError(
            "M2N arrival expected_roundtrip_inflight must be an exact bool, "
            f"got {expected_roundtrip_inflight!r}"
        )
    if type(request_end_deferred) is not bool:
        raise ValueError(
            "M2N arrival request_end_deferred must be an exact bool, "
            f"got {request_end_deferred!r}"
        )
    if request_end_deferred and expected_roundtrip_inflight is not False:
        raise ValueError(
            "M2N arrival with deferred request end must validate the "
            "projected roundtrip_inflight=False state"
        )

    if scheduler._cluster_type is ClusterType.DECODE_ATTN:
        scheduler._validate_decode_attn_m2n_receipt(
            batch,
            transfer_info,
            expected_roundtrip_inflight=expected_roundtrip_inflight,
            request_end_deferred=request_end_deferred,
        )
    else:
        scheduler.preflight_m2n_arrival(batch, transfer_info)
    logger = get_cluster_logger(__name__, scheduler._cluster_type.name)

    request_ids = [request.id for request in batch.requests]
    pipeline_stage = "attn→ffn" if transfer_info.is_attn_to_ffn else "ffn→attn"
    logger.info(
        f"{scheduler._cluster_type.name} cluster received M2N data at {time:.3f}s: "
        f"requests {request_ids} from {pipeline_stage} transfer, "
        f"batch_id={batch.id}, transfer_size={transfer_info.activation_size_bytes} bytes, "
        f"source_cluster={transfer_info.source_cluster_type.name}"
    )

    if scheduler._cluster_type == ClusterType.DECODE_FFN:
        return scheduler._handle_m2n_arrival_decode_ffn(time, batch, transfer_info, logger)
    if scheduler._cluster_type == ClusterType.DECODE_ATTN:
        return scheduler._handle_m2n_arrival_decode_attn(
            time,
            batch,
            transfer_info,
            logger,
            expected_roundtrip_inflight=expected_roundtrip_inflight,
            request_end_deferred=request_end_deferred,
        )
    raise RuntimeError(
        f"Validated M2N arrival has no handler for cluster {scheduler._cluster_type.name}"
    )
