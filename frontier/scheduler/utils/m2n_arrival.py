"""M2N arrival routing for disaggregated clusters."""

from typing import Any

from frontier.entities import Batch
from frontier.types import ClusterType


def handle_decode_ffn_arrival(
    scheduler: Any, time: float, batch: Batch, transfer_info: Any, logger: Any
) -> list:
    """Queue one DECODE_ATTN-to-FFN transfer and trigger promotion."""

    from collections import defaultdict, deque
    from frontier.events.cluster_schedule_event import ClusterScheduleEvent

    (
        layer_id,
        afd_stage_idx,
        barrier_round_id,
        lane,
        barrier_expected_lanes,
        expected_lanes,
        group_key,
        expected_lane_contract,
        target_replica_id,
    ) = scheduler._validate_decode_ffn_m2n_receipt(batch, transfer_info)
    transfer_info.target_ffn_replica_id = target_replica_id
    transfer_info.target_execution_replica_id = target_replica_id
    transfer_info.target_execution_replica_local_id = None
    for request in batch.requests:
        request.on_arrival(time, scheduler._cluster_type)
    batch.decode_ffn_m2n_arrival_time = time
    room = scheduler._m2n_waiting_by_layer.get(group_key)
    if room is None:
        room = {
            "per_lane_queues": defaultdict(deque),
            "lanes_rr_order": deque(),
            "rr_cursor": 0,
            "expected_lane_contract": expected_lane_contract,
        }
        scheduler._m2n_waiting_by_layer[group_key] = room
    if lane not in room["per_lane_queues"]:
        room["per_lane_queues"][lane] = deque()
    was_empty = len(room["per_lane_queues"][lane]) == 0
    room["per_lane_queues"][lane].append((batch, transfer_info))
    if was_empty:
        room["lanes_rr_order"].append(lane)
    logger.info(
        f"[FFN-M2N-ARRIVAL] wire_layer={layer_id} afd_stage_idx={afd_stage_idx} "
        f"barrier_round_id={barrier_round_id} lane={lane} "
        f"enqueued; ready_lanes={len(room['lanes_rr_order'])}/{expected_lanes}"
    )
    promoted = scheduler._try_promote_decode_ffn_group(
        time,
        group_key,
        room,
        logger,
        allow_idle_injection=(not batch.is_idle) and not bool(barrier_expected_lanes),
        expected_lanes=expected_lanes,
        expected_lane_ids=barrier_expected_lanes or None,
    )
    return [ClusterScheduleEvent(time, scheduler._cluster_type)] if promoted else []


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


def handle_decode_attn_arrival(
    scheduler: Any,
    time: float,
    micro_batch: Any,
    transfer_info: Any,
    logger: Any,
    *,
    expected_roundtrip_inflight: bool = False,
    request_end_deferred: bool = False,
) -> list:
    """Advance one DECODE_ATTN batch after an FFN return transfer."""

    from frontier.events.cluster_schedule_event import ClusterScheduleEvent
    from frontier.events.global_batch_end_event import GlobalBatchEndEvent

    receipt = scheduler._validate_decode_attn_m2n_receipt(
        micro_batch,
        transfer_info,
        expected_roundtrip_inflight=expected_roundtrip_inflight,
        request_end_deferred=request_end_deferred,
    )
    logger.info(
        f"[AF-ARRIVAL] M2N returned micro batch {micro_batch.id} at "
        "decode-attn; advancing request states"
    )
    next_events = []
    total_layers = scheduler._config.replica_config.model_config.num_layers
    micro_batch.mb_on_step_layer_count_increment()
    is_last_layer = receipt["is_last_layer"]
    logger.info(
        f"[AF-ARRIVAL][AFTER] mb={micro_batch.id} "
        f"inflight_layers={getattr(micro_batch, 'af_inflight_layer_count', None)} "
        f"/ total_layers={total_layers}; is_mb_last_layer={is_last_layer}"
    )
    replica_id = receipt["replica_id"]
    replica_local_id = receipt["replica_local_id"]
    ready_for_reschedule = False
    if not is_last_layer:
        ready_for_reschedule = scheduler._enqueue_decode_attn_return_round(
            micro_batch,
            receipt=receipt,
            logger=logger,
        )
    else:
        global_end_time = scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            time,
            micro_batch,
        )
        current_exec_sigs = [
            Batch._get_request_execution_signature(request)
            for request in micro_batch.requests
        ]
        current_mut_sigs = [
            Batch._get_request_mutation_signature(request)
            for request in micro_batch.requests
        ]
        next_events.append(
            GlobalBatchEndEvent(
                global_end_time,
                replica_id,
                replica_local_id,
                micro_batch,
                scheduler._cluster_type,
                request_execution_signatures=current_exec_sigs,
                request_mutation_signatures=current_mut_sigs,
            )
        )
    if next_events or ready_for_reschedule:
        return next_events + [ClusterScheduleEvent(time, scheduler._cluster_type)]
    return next_events
