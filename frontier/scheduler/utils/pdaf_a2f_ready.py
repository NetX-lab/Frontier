"""PD-AF decode-attention to FFN scheduling callback.

This module owns A-to-F admission orchestration while the cluster scheduler
remains the state owner and callback entry point.
"""

from collections import defaultdict, deque
import math
from numbers import Real
from typing import Any, List

from frontier.entities import Batch
from frontier.types import ClusterType
from frontier.scheduler.utils.pdaf_transfer import LaneIdentityScope
from frontier.scheduler.utils.pdaf_a2f import prepare_a2f_admission

M2NLaneIdentityScope = LaneIdentityScope


def schedule_decode_attn_a2f_ready(
    scheduler: Any,
    time: float,
    batch: Batch,
    *,
    replica_id: int,
    replica_local_id: int | None,
    layer_id: int,
    logger,
) -> List:
    """Admit a completed DECODE_ATTN batch into the A-to-F transfer path."""
    from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent

    if scheduler._cluster_type != ClusterType.DECODE_ATTN:
        raise ValueError(
            "on_decode_attn_a2f_ready is only valid for DECODE_ATTN cluster"
        )
    if type(batch) is not Batch:
        raise ValueError(
            "DECODE_ATTN A-to-F admission requires an exact Batch, "
            f"got {type(batch).__name__}"
        )
    layer_id = scheduler._validate_decode_attn_a2f_topology_value(
        layer_id,
        field_name="layer_id",
    )
    afd_stage_idx = scheduler._validate_decode_attn_a2f_topology_value(
        getattr(batch, "afd_stage_idx", None),
        field_name="afd_stage_idx",
    )
    replica_id = scheduler._validate_decode_attn_a2f_topology_value(
        replica_id,
        field_name="replica_id",
    )
    # DECODE_ATTN has one full-stage scheduler per serving Replica.  The
    # second tuple coordinate is intentionally absent; it is not an
    # attention-DP lane and must remain ``None`` on A→F transport.
    if replica_local_id is not None:
        raise ValueError(
            "DECODE_ATTN A-to-F requires full-stage identity with "
            f"replica_local_id=None, got {replica_local_id!r}"
        )
    if (
        not isinstance(time, Real)
        or isinstance(time, bool)
        or not math.isfinite(time)
        or time < 0
    ):
        raise ValueError(
            "DECODE_ATTN A-to-F event time must be a finite non-negative "
            f"int or float, got {time!r}"
        )
    # Predictors commonly return numpy scalar real values. Normalize the
    # validated timestamp before constructing events so downstream event
    # contracts receive a built-in numeric type.
    time = float(time)
    if scheduler._m2n_transfer_predictor is None:
        raise ValueError("M2N transfer predictor not found in decode-attn cluster scheduler")

    # Bind the source Attention Replica at the A→F boundary.  The
    # DECODE_ATTN scheduler is full-stage, so there is no local DP value to
    # carry; ``None`` is the only valid second coordinate.
    original_replica_id = getattr(
        batch, "decode_attn_original_replica_id", None
    )
    if original_replica_id is not None:
        original_replica_id = scheduler._validate_decode_attn_a2f_topology_value(
            original_replica_id,
            field_name="batch original replica_id",
        )
        if original_replica_id != replica_id:
            raise ValueError(
                "DECODE_ATTN A-to-F batch source Replica mismatch: "
                f"batch={original_replica_id!r}, event={replica_id!r}"
            )
    batch.decode_attn_original_replica_id = replica_id
    original_replica_local_id = getattr(
        batch, "decode_attn_original_replica_local_id", None
    )
    if original_replica_local_id is not None:
        raise ValueError(
            "DECODE_ATTN A-to-F batch must use full-stage identity with "
            "decode_attn_original_replica_local_id=None, got "
            f"{original_replica_local_id!r}"
        )
    batch.decode_attn_original_replica_local_id = None

    replica_config = getattr(scheduler._config, "replica_config", None)
    if replica_config is None:
        raise RuntimeError(
            "DECODE_ATTN A-to-F admission requires replica_config"
        )
    model_config = getattr(replica_config, "model_config", None)
    if model_config is None:
        raise RuntimeError(
            "DECODE_ATTN A-to-F admission requires model_config"
        )
    model_is_moe = getattr(model_config, "is_moe", None)
    if type(model_is_moe) is not bool:
        raise RuntimeError(
            "DECODE_ATTN A-to-F model_config.is_moe must be an exact bool, "
            f"got {model_is_moe!r}"
        )

    scheduler._validate_decode_attn_a2f_batch_entry(
        batch=batch,
        lane=(replica_id, replica_local_id),
        layer_id=layer_id,
        afd_stage_idx=afd_stage_idx,
        model_is_moe=model_is_moe,
        context="incoming batch",
        allow_idle=False,
    )

    cohort_id = getattr(batch, "decode_attn_cohort_id", None)
    cohort_request_ids = getattr(batch, "decode_attn_cohort_request_ids", None)
    if cohort_id is not None and cohort_request_ids is not None:
        active_local_attn_lanes = scheduler._get_decode_attn_a2f_active_local_attn_lanes(
            cohort_id=cohort_id,
            request_ids=cohort_request_ids,
            afd_stage_idx=afd_stage_idx,
            layer_id=layer_id,
        )
        expected_lane_contract = tuple(
            sorted(
                scheduler._normalize_m2n_lanes(
                    active_local_attn_lanes,
                    identity_scope=M2NLaneIdentityScope.FULL_STAGE,
                    field_name=(
                        "DECODE_ATTN A-to-F active cohort local_attn topology"
                    ),
                    require_nonempty=True,
                )
            )
        )
    else:
        expected_lane_contract = tuple(
            sorted(
                scheduler._normalize_m2n_lanes(
                    scheduler._get_decode_attn_a2f_expected_lanes(
                        afd_stage_idx,
                        layer_id=layer_id,
                    ),
                    identity_scope=M2NLaneIdentityScope.FULL_STAGE,
                    field_name="DECODE_ATTN A-to-F expected lane topology",
                    require_nonempty=True,
                )
            )
        )

    group_key = (layer_id, afd_stage_idx)
    lane = (replica_id, replica_local_id)
    if lane not in expected_lane_contract:
        raise ValueError(
            "Unexpected lane observed in DECODE_ATTN A→F waiting room: "
            f"group_key={group_key}, lane={lane}, "
            f"expected_lanes={expected_lane_contract}"
        )

    idle_expected_lanes = getattr(scheduler, "_decode_attn_idle_expected_lanes", None)
    if idle_expected_lanes is not None:
        if type(idle_expected_lanes) is not set:
            raise RuntimeError(
                "DECODE_ATTN A-to-F idle lane inventory must be an exact set"
            )
        normalized_idle_expected_lanes = set(
            scheduler._normalize_m2n_lanes(
                tuple(idle_expected_lanes),
                identity_scope=M2NLaneIdentityScope.FULL_STAGE,
                field_name="DECODE_ATTN A-to-F idle lane topology",
                require_nonempty=False,
            )
        )
    else:
        normalized_idle_expected_lanes = set()

    scheduler._peek_decode_attn_barrier_round_id()

    if not model_is_moe:
        events = scheduler._release_dense_decode_ffn_a2f_without_lane_barrier(
            time,
            batch,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
            logger=logger,
        )
        if idle_expected_lanes is not None:
            idle_expected_lanes.discard(lane)
        return events

    waiting_rooms = getattr(scheduler, "_a2f_waiting_by_layer", None)
    if type(waiting_rooms) is not dict:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room inventory must be an exact dict"
        )
    room_exists = group_key in waiting_rooms
    room = waiting_rooms[group_key] if room_exists else None
    if room_exists:
        scheduler._validate_decode_attn_a2f_waiting_room(
            group_key=group_key,
            room=room,
            expected_lane_contract=expected_lane_contract,
            incoming_batch=batch,
        )
    existing_queues = {} if room is None else room["per_lane_queues"]
    plan = prepare_a2f_admission(
        existing_queues=existing_queues,
        expected_lanes=expected_lane_contract,
        idle_lanes=normalized_idle_expected_lanes,
        incoming_lane=lane,
        incoming_layer_id=layer_id,
        incoming_batch=batch,
        is_moe=model_is_moe,
        time=time,
        group_key=group_key,
        validate_room=lambda **kwargs: scheduler._validate_decode_attn_a2f_waiting_room(
            **kwargs
        ),
        build_idle_entries=lambda **kwargs: scheduler._prepare_decode_attn_idle_lanes_for_barrier(
            **kwargs
        ),
        get_transfer_info=lambda ready_batch: scheduler._m2n_transfer_predictor.get_transfer_info(
            source_cluster_type=ClusterType.DECODE_ATTN,
            target_cluster_type=ClusterType.DECODE_FFN,
            batch=ready_batch,
            replica_config=replica_config,
        ),
        validate_transfer_result=scheduler._validate_decode_attn_a2f_predictor_result,
    )
    prospective_per_lane_queues = plan.queues
    prepared_idle_lanes = list(plan.prepared_idle_lanes)
    prepared_idle_entries = list(plan.prepared_idle_entries)
    barrier_is_ready = plan.barrier_ready
    ready_lanes = plan.ready_lane_count
    picked = list(plan.picked)
    prospective_after_release = plan.queues_after_release
    non_idle_expected_lanes = plan.non_idle_lanes
    barrier_round_id = scheduler._peek_decode_attn_barrier_round_id() if barrier_is_ready else None
    transfer_plan_by_batch = {
        id(ready_batch): (activation_size, transfer_time)
        for _, _, ready_batch, activation_size, transfer_time in plan.transfer_descriptors
    }

    events = []
    prepared_phase_updates = []
    if barrier_is_ready:
        for (source_replica_id, source_replica_local_id), ready_layer_id, ready_batch in picked:
            if ready_batch.is_idle:
                continue
            activation_size, transfer_time = transfer_plan_by_batch[
                id(ready_batch)
            ]
            events.append(
                M2NTransferStartEvent(
                    time=time,
                    source_replica_id=source_replica_id,
                    source_replica_local_id=source_replica_local_id,
                    source_cluster_type=ClusterType.DECODE_ATTN,
                    target_cluster_type=ClusterType.DECODE_FFN,
                    batch=ready_batch,
                    activation_size_bytes=activation_size,
                    transfer_time_ms=transfer_time,
                    layer_id=ready_layer_id,
                    afd_stage_idx=ready_batch.afd_stage_idx,
                    source_execution_replica_id=source_replica_id,
                    source_execution_replica_local_id=source_replica_local_id,
                )
            )
            events.append(
                ReplicaScheduleEvent(
                    time,
                    source_replica_id,
                    scheduler._cluster_type,
                    source_replica_local_id,
                )
            )
            prepared_phase_updates.append(
                scheduler._set_decode_attn_batch_cohort_phase(
                    ready_batch,
                    phase="ffn_inflight",
                    replica_id=source_replica_id,
                    replica_local_id=source_replica_local_id,
                    layer_id=ready_layer_id,
                    prepare_only=True,
                )
            )

    scheduler._commit_decode_attn_batch_phases(prepared_phase_updates)

    if room_exists:
        committed_room = room
        committed_per_lane_queues = committed_room["per_lane_queues"]
        for queue_lane in tuple(committed_per_lane_queues):
            if queue_lane not in prospective_after_release:
                committed_per_lane_queues[queue_lane].clear()
    else:
        committed_per_lane_queues = defaultdict(deque)
        committed_room = {
            "per_lane_queues": committed_per_lane_queues,
            "expected_lane_contract": expected_lane_contract,
        }

    for queue_lane, prepared_queue in prospective_after_release.items():
        committed_queue = committed_per_lane_queues[queue_lane]
        committed_queue.clear()
        committed_queue.extend(prepared_queue)

    if any(committed_per_lane_queues.values()):
        waiting_rooms[group_key] = committed_room
    else:
        waiting_rooms.pop(group_key, None)
    if idle_expected_lanes is not None:
        idle_expected_lanes.discard(lane)

    if barrier_is_ready:
        for ready_lane, ready_layer_id, ready_batch in picked:
            if ready_batch.is_idle:
                logger.info(
                    f"[A2F-GROUP-RELEASE-IDLE] layer={ready_layer_id} "
                    f"afd_stage_idx={ready_batch.afd_stage_idx} slot={ready_batch.afd_stage_idx} "
                    f"lane={ready_lane}"
                )
                continue
            ready_batch.decode_attn_barrier_round_id = barrier_round_id
            ready_batch.decode_attn_barrier_expected_lanes = (
                non_idle_expected_lanes
            )
        scheduler._decode_attn_barrier_round_counter = barrier_round_id + 1

    logger.info(
        f"[A2F-GROUP-READY] layer={layer_id} afd_stage_idx={afd_stage_idx} "
        f"slot={afd_stage_idx} lane={lane} "
        f"depth={len(prospective_per_lane_queues[lane])} "
        f"ready_lanes={ready_lanes}/{len(expected_lane_contract)}"
    )
    if prepared_idle_entries:
        logger.info(
            f"[A2F-GROUP-IDLE] layer_id={layer_id} "
            f"afd_stage_idx={afd_stage_idx} "
            f"missing={sorted(prepared_idle_lanes)} "
            f"layer_hint={layer_id}"
        )
    for ready_lane, ready_layer_id, ready_batch in picked:
        if not ready_batch.is_idle:
            logger.info(
                f"[A2F-GROUP-RELEASE] layer={ready_layer_id} afd_stage_idx={ready_batch.afd_stage_idx} "
                f"slot={ready_batch.afd_stage_idx} lane={ready_lane} batch_id={ready_batch.id}"
            )
    return events
