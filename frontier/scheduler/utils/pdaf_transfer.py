"""Small, state-free helpers shared by PD-AF transfer handlers."""

import math
from enum import Enum
from numbers import Real
from typing import Any, Dict, List, Optional

from collections import defaultdict, deque

from frontier.types import ClusterType, ReplicaSchedulerType
from frontier.entities import Batch, Request


class LaneIdentityScope(Enum):
    """Identity shape accepted by one transfer lane contract."""

    FULL_STAGE = "full_stage"
    REPLICA_LOCAL = "replica_local"


def validate_a2f_predictor_result(predictor_result: Any) -> tuple[int, int | float]:
    """Validate one A-to-F transfer predictor result without coercing size."""

    if type(predictor_result) is not tuple or len(predictor_result) != 2:
        raise RuntimeError(
            "DECODE_ATTN A-to-F predictor transfer result must be an exact "
            f"(activation_size, transfer_time) tuple, got {predictor_result!r}"
        )
    activation_size, transfer_time = predictor_result
    if type(activation_size) is not int or activation_size < 0:
        raise ValueError(
            "DECODE_ATTN A-to-F predictor activation_size must be an exact "
            f"non-negative int, got {activation_size!r}"
        )
    if not isinstance(transfer_time, Real) or isinstance(transfer_time, bool):
        raise ValueError(
            "DECODE_ATTN A-to-F predictor transfer_time must be an exact int "
            f"or float, got {transfer_time!r}"
        )
    transfer_time = float(transfer_time)
    if not math.isfinite(transfer_time) or transfer_time < 0:
        raise ValueError(
            "DECODE_ATTN A-to-F predictor transfer_time must be finite and "
            f"non-negative, got {transfer_time!r}"
        )
    return activation_size, transfer_time


def normalize_lanes(
    raw_lanes,
    *,
    identity_scope: LaneIdentityScope,
    field_name: str,
    require_nonempty: bool,
) -> List[tuple[int, int | None]]:
    """Validate and normalize one exact M2N lane list."""

    if type(identity_scope) is not LaneIdentityScope:
        raise ValueError(
            "identity_scope must be an exact LaneIdentityScope, "
            f"got {identity_scope!r}"
        )
    if type(raw_lanes) not in {list, tuple}:
        raise ValueError(
            f"{field_name} must be an exact list or tuple, got {raw_lanes!r}"
        )

    allow_full_stage_identity = identity_scope is LaneIdentityScope.FULL_STAGE
    normalized_lanes: List[tuple[int, int | None]] = []
    seen_lanes = set()
    for raw_lane in raw_lanes:
        if type(raw_lane) is not tuple or len(raw_lane) != 2:
            raise ValueError(
                f"{field_name} must contain exact 2-tuples, got {raw_lane!r}"
            )
        lane_replica_id, lane_replica_local_id = raw_lane
        if type(lane_replica_id) is not int or lane_replica_id < 0:
            raise ValueError(
                f"{field_name} replica_id must be an exact non-negative int, "
                f"got {lane_replica_id!r}"
            )
        if lane_replica_local_id is not None and (
            type(lane_replica_local_id) is not int or lane_replica_local_id < 0
        ):
            raise ValueError(
                f"{field_name} replica_local_id must be an exact "
                f"non-negative int, got {lane_replica_local_id!r}"
            )
        if lane_replica_local_id is None and not allow_full_stage_identity:
            raise ValueError(
                f"{field_name} replica_local_id cannot be None in "
                f"{identity_scope.value} identity scope"
            )
        lane = (lane_replica_id, lane_replica_local_id)
        if lane in seen_lanes:
            raise ValueError(f"{field_name} contains duplicate lane {lane!r}")
        seen_lanes.add(lane)
        normalized_lanes.append(lane)

    if require_nonempty and not normalized_lanes:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_lanes
def validate_decode_ffn_waiting_room(
    *,
    group_key: tuple[int, int] | tuple[int, int, int],
    room: dict,
    expected_lane_contract: Optional[tuple[tuple[int, int], ...]] = None,
    incoming_batch: Optional[Batch] = None,
) -> tuple[tuple[int, int], ...]:
    """Validate one DECODE_FFN waiting room without mutating runtime state."""

    from frontier.entities.m2n_transfer_info import M2NTransferInfo

    if type(group_key) is not tuple or len(group_key) not in (2, 3):
        raise RuntimeError(
            "DECODE_FFN waiting-room key must be an exact "
            f"(layer, stage[, round]) tuple, got {group_key!r}"
        )
    for field_name, value in zip(
        ("layer_id", "afd_stage_idx", "barrier_round_id"),
        group_key,
    ):
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"DECODE_FFN waiting-room {field_name} must be an exact "
                f"non-negative int, got {value!r}"
            )
    layer_id, afd_stage_idx = group_key[:2]
    barrier_round_id = group_key[2] if len(group_key) == 3 else None

    if type(room) is not dict:
        raise RuntimeError(
            "DECODE_FFN waiting room must be an exact dict, "
            f"got {type(room).__name__}"
        )
    expected_room_fields = {
        "per_lane_queues",
        "lanes_rr_order",
        "rr_cursor",
        "expected_lane_contract",
    }
    if set(room) != expected_room_fields:
        missing_room_fields = expected_room_fields - set(room)
        if missing_room_fields:
            missing_field_labels = ", ".join(
                field_name.replace("_", " ")
                for field_name in sorted(missing_room_fields)
            )
            raise RuntimeError(
                "DECODE_FFN waiting room missing required fields: "
                f"{missing_field_labels}"
            )
        raise RuntimeError(
            "DECODE_FFN waiting-room schema mismatch: "
            f"expected={sorted(expected_room_fields)}, actual={sorted(room)}"
        )

    per_lane_queues = room["per_lane_queues"]
    if (
        type(per_lane_queues) is not defaultdict
        or per_lane_queues.default_factory is not deque
    ):
        raise RuntimeError(
            "DECODE_FFN waiting-room per_lane_queues must be an exact "
            "defaultdict(deque)"
        )
    lanes_rr_order = room["lanes_rr_order"]
    if type(lanes_rr_order) is not deque:
        raise RuntimeError(
            "DECODE_FFN waiting-room lanes_rr_order must be an exact deque"
        )
    rr_cursor = room["rr_cursor"]
    if type(rr_cursor) is not int or rr_cursor < 0:
        raise RuntimeError(
            "DECODE_FFN waiting-room rr_cursor must be an exact "
            f"non-negative int, got {rr_cursor!r}"
        )

    raw_room_lanes = room["expected_lane_contract"]
    if type(raw_room_lanes) is not tuple:
        raise RuntimeError(
            "DECODE_FFN waiting-room expected lane contract must be an "
            f"exact tuple, got {raw_room_lanes!r}"
        )
    room_lanes = tuple(
        normalize_lanes(
            raw_room_lanes,
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN waiting-room expected lane contract",
            require_nonempty=True,
        )
    )
    canonical_room_lanes = tuple(sorted(room_lanes))
    if room_lanes != canonical_room_lanes:
        raise RuntimeError(
            "DECODE_FFN waiting-room expected lane contract must be "
            f"canonical, got {room_lanes!r}"
        )
    if expected_lane_contract is not None:
        normalized_expected_lanes = tuple(
            sorted(
                normalize_lanes(
                    expected_lane_contract,
                    identity_scope=LaneIdentityScope.FULL_STAGE,
                    field_name="DECODE_FFN receipt expected lane contract",
                    require_nonempty=True,
                )
            )
        )
        if canonical_room_lanes != normalized_expected_lanes:
            raise ValueError(
                "Inconsistent DECODE_FFN expected lane contract for waiting "
                f"room: group_key={group_key}, "
                f"existing={canonical_room_lanes}, "
                f"received={normalized_expected_lanes}"
            )

    queue_lanes = tuple(
        normalize_lanes(
            tuple(per_lane_queues),
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN waiting-room queue lanes",
            require_nonempty=False,
        )
    )
    rr_lanes = tuple(
        normalize_lanes(
            tuple(lanes_rr_order),
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN waiting-room round-robin lanes",
            require_nonempty=False,
        )
    )
    room_lane_set = set(canonical_room_lanes)
    if not set(queue_lanes).issubset(room_lane_set):
        raise RuntimeError(
            "DECODE_FFN waiting-room queue lane is outside the expected "
            f"contract: queues={queue_lanes}, contract={canonical_room_lanes}"
        )
    if not set(rr_lanes).issubset(room_lane_set):
        raise RuntimeError(
            "DECODE_FFN waiting-room round-robin lane is outside the expected "
            f"contract: order={rr_lanes}, contract={canonical_room_lanes}"
        )

    nonempty_queue_lanes = set()
    seen_batch_identities = set()
    for queue_lane, lane_queue in per_lane_queues.items():
        if type(lane_queue) is not deque:
            raise RuntimeError(
                "DECODE_FFN waiting-room lane queue must be an exact deque: "
                f"lane={queue_lane}, got={type(lane_queue).__name__}"
            )
        if lane_queue:
            nonempty_queue_lanes.add(queue_lane)
        for queue_index, queued_entry in enumerate(lane_queue):
            if type(queued_entry) is not tuple or len(queued_entry) != 2:
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued entry must be an exact "
                    f"(batch, transfer_info) tuple: lane={queue_lane}, "
                    f"index={queue_index}, value={queued_entry!r}"
                )
            queued_batch, queued_transfer_info = queued_entry
            if type(queued_transfer_info) is not M2NTransferInfo:
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued transfer must be an exact "
                    f"M2NTransferInfo: lane={queue_lane}, index={queue_index}"
                )
            if queued_transfer_info.batch is not queued_batch:
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch identity does not "
                    "match transfer_info.batch"
                )
            queued_is_idle = getattr(queued_batch, "is_idle", None)
            if type(queued_is_idle) is not bool:
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch is_idle must be an "
                    f"exact bool, got {queued_is_idle!r}"
                )
            if incoming_batch is not None and queued_batch is incoming_batch:
                raise ValueError(
                    "DECODE_FFN waiting room already contains the incoming "
                    "batch object"
                )
            queued_batch_identity = id(queued_batch)
            if queued_batch_identity in seen_batch_identities:
                raise RuntimeError(
                    "DECODE_FFN waiting room contains a duplicate queued "
                    "batch object"
                )
            seen_batch_identities.add(queued_batch_identity)

            if (
                queued_transfer_info.source_cluster_type
                is not ClusterType.DECODE_ATTN
                or queued_transfer_info.target_cluster_type
                is not ClusterType.DECODE_FFN
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued transfer must be an exact "
                    "DECODE_ATTN -> DECODE_FFN transfer"
                )
            queued_source_replica_id = queued_transfer_info.source_replica_id
            queued_source_replica_local_id = queued_transfer_info.source_replica_local_id
            if (
                type(queued_source_replica_id) is not int
                or queued_source_replica_id < 0
                or (
                    queued_source_replica_local_id is not None
                    and (
                        type(queued_source_replica_local_id) is not int
                        or queued_source_replica_local_id < 0
                    )
                )
                or (queued_source_replica_id, queued_source_replica_local_id) != queue_lane
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued transfer lane mismatch: "
                    f"queue={queue_lane}, transfer="
                    f"{(queued_source_replica_id, queued_source_replica_local_id)}"
                )
            queued_layer_id = queued_transfer_info.layer_id
            if type(queued_layer_id) is not int or queued_layer_id != layer_id:
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued transfer layer mismatch: "
                    f"room={layer_id!r}, transfer={queued_layer_id!r}"
                )
            queued_stage_idx = queued_transfer_info.afd_stage_idx
            if (
                type(queued_stage_idx) is not int
                or queued_stage_idx != afd_stage_idx
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued transfer stage mismatch: "
                    f"room={afd_stage_idx!r}, transfer={queued_stage_idx!r}"
                )

            queued_round_id = getattr(
                queued_batch,
                "decode_attn_barrier_round_id",
                None,
            )
            if barrier_round_id is None:
                if queued_round_id is not None:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch round mismatch: "
                        f"room=None, batch={queued_round_id!r}"
                    )
            elif (
                type(queued_round_id) is not int
                or queued_round_id != barrier_round_id
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch round mismatch: "
                    f"room={barrier_round_id!r}, batch={queued_round_id!r}"
                )

            queued_expected_lanes = getattr(
                queued_batch,
                "decode_attn_barrier_expected_lanes",
                (),
            )
            if queued_expected_lanes is None:
                queued_expected_lanes = ()
            normalized_queued_expected_lanes = tuple(
                sorted(
                    normalize_lanes(
                        queued_expected_lanes,
                        identity_scope=LaneIdentityScope.FULL_STAGE,
                        field_name=(
                            "DECODE_FFN waiting-room queued batch expected "
                            "lane metadata"
                        ),
                        require_nonempty=False,
                    )
                )
            )
            if (
                normalized_queued_expected_lanes
                and normalized_queued_expected_lanes
                != canonical_room_lanes
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch lane contract "
                    f"mismatch: room={canonical_room_lanes}, "
                    f"batch={normalized_queued_expected_lanes}"
                )

            queued_batch_stage_idx = getattr(
                queued_batch,
                "afd_stage_idx",
                None,
            )
            if queued_batch_stage_idx is not None and (
                type(queued_batch_stage_idx) is not int
                or queued_batch_stage_idx != afd_stage_idx
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch stage mismatch: "
                    f"room={afd_stage_idx!r}, batch={queued_batch_stage_idx!r}"
                )
            queued_batch_layer_id = getattr(
                queued_batch,
                "decode_ffn_layer_id",
                None,
            )
            if queued_batch_layer_id is not None and (
                type(queued_batch_layer_id) is not int
                or queued_batch_layer_id != layer_id
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch layer mismatch: "
                    f"room={layer_id!r}, batch={queued_batch_layer_id!r}"
                )
            queued_original_replica_id = getattr(
                queued_batch,
                "decode_attn_original_replica_id",
                None,
            )
            queued_original_replica_local_id = getattr(
                queued_batch,
                "decode_attn_original_replica_local_id",
                None,
            )
            if queued_original_replica_id is not None and (
                type(queued_original_replica_id) is not int
                or queued_original_replica_id != queue_lane[0]
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch replica lane mismatch: "
                    f"queue={queue_lane[0]!r}, "
                    f"batch={queued_original_replica_id!r}"
                )
            if queued_original_replica_local_id is not None and (
                type(queued_original_replica_local_id) is not int
                or queued_original_replica_local_id != queue_lane[1]
            ):
                raise RuntimeError(
                    "DECODE_FFN waiting-room queued batch local identity mismatch: "
                    f"queue={queue_lane[1]!r}, batch={queued_original_replica_local_id!r}"
                )

    if set(rr_lanes) != nonempty_queue_lanes:
        raise RuntimeError(
            "DECODE_FFN waiting-room round-robin lanes must exactly match "
            "non-empty queue lanes: "
            f"order={rr_lanes}, nonempty={sorted(nonempty_queue_lanes)}"
        )
    return canonical_room_lanes
def validate_decode_ffn_receipt(
    scheduler,
    batch: Batch,
    transfer_info: "M2NTransferInfo",
) -> tuple[
    int,
    int,
    Optional[int],
    tuple[int, int],
    List[tuple[int, int]],
    int,
    tuple[int, int] | tuple[int, int, int],
    tuple[tuple[int, int], ...],
    int,
]:
    """Validate one A-to-F receipt without mutating scheduler or batch state."""

    scheduler.validate_m2n_arrival_target(transfer_info)
    if scheduler._cluster_type is not ClusterType.DECODE_FFN:
        raise ValueError(
            "DECODE_FFN receipt validation requires a DECODE_FFN scheduler, "
            f"got {scheduler._cluster_type.name}"
        )
    if batch is not transfer_info.batch:
        raise ValueError(
            "DECODE_FFN M2N batch identity mismatch: batch is not "
            "transfer_info.batch"
        )

    layer_id = getattr(transfer_info, "layer_id", None)
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError(
            "DECODE_FFN receipt layer_id must be an exact non-negative int, "
            f"got {layer_id!r}"
        )

    afd_stage_idx = getattr(transfer_info, "afd_stage_idx", None)
    if type(afd_stage_idx) is not int or afd_stage_idx < 0:
        raise ValueError(
            "DECODE_FFN receipt afd_stage_idx must be an exact non-negative int, "
            f"got {afd_stage_idx!r}"
        )

    barrier_round_id = getattr(batch, "decode_attn_barrier_round_id", None)
    if barrier_round_id is not None and (
        type(barrier_round_id) is not int or barrier_round_id < 0
    ):
        raise ValueError(
            "DECODE_FFN receipt barrier_round_id must be None or an exact "
            f"non-negative int, got {barrier_round_id!r}"
        )

    source_replica_id = getattr(transfer_info, "source_replica_id", None)
    if type(source_replica_id) is not int or source_replica_id < 0:
        raise ValueError(
            "DECODE_FFN receipt source_replica_id must be an exact "
            f"non-negative int, got {source_replica_id!r}"
        )
    source_replica_local_id = getattr(transfer_info, "source_replica_local_id", None)
    if source_replica_local_id is not None and (
        type(source_replica_local_id) is not int or source_replica_local_id < 0
    ):
        raise ValueError(
            "DECODE_FFN receipt source_replica_local_id must be None or an exact "
            "non-negative int, "
            f"got {source_replica_local_id!r}"
        )
    lane = (source_replica_id, source_replica_local_id)

    raw_expected_lanes = getattr(
        batch,
        "decode_attn_barrier_expected_lanes",
        (),
    )
    if raw_expected_lanes is None:
        raw_expected_lanes = ()
    barrier_expected_lanes = normalize_lanes(
        raw_expected_lanes,
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_FFN receipt expected lane metadata",
        require_nonempty=False,
    )

    if barrier_expected_lanes:
        if lane not in barrier_expected_lanes:
            raise ValueError(
                "Unexpected lane observed in DECODE_FFN round-scoped waiting "
                f"room: lane={lane}, expected_lanes={barrier_expected_lanes}"
            )
        expected_lanes = len(barrier_expected_lanes)
        expected_lane_contract = tuple(sorted(barrier_expected_lanes))
    else:
        scheduler_expected_lanes = normalize_lanes(
            getattr(scheduler, "_ffn_expected_lanes", None),
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN scheduler lane topology",
            require_nonempty=True,
        )
        if lane not in scheduler_expected_lanes:
            raise ValueError(
                "Unexpected lane observed in DECODE_FFN scheduler lane topology: "
                f"lane={lane}, expected_lanes={scheduler_expected_lanes}"
            )
        expected_lanes = getattr(scheduler, "_ffn_group_micro_batches", None)
        if type(expected_lanes) is not int or expected_lanes <= 0:
            raise ValueError(
                "DECODE_FFN _ffn_group_micro_batches must be an exact positive "
                f"int when expected lane metadata is empty, got {expected_lanes!r}"
            )
        expected_lane_contract = tuple(sorted(scheduler_expected_lanes))

    if barrier_round_id is None:
        group_key = (layer_id, afd_stage_idx)
    else:
        group_key = (layer_id, afd_stage_idx, barrier_round_id)

    if not hasattr(scheduler, "_m2n_waiting_by_layer"):
        raise RuntimeError(
            "DECODE_FFN scheduler missing _m2n_waiting_by_layer during receipt "
            "preflight"
        )
    if type(scheduler._m2n_waiting_by_layer) is not dict:
        raise RuntimeError(
            "DECODE_FFN _m2n_waiting_by_layer must be an exact dict"
        )
    room = scheduler._m2n_waiting_by_layer.get(group_key)
    if room is not None:
        scheduler._validate_decode_ffn_waiting_room(
            group_key=group_key,
            room=room,
            expected_lane_contract=expected_lane_contract,
            incoming_batch=batch,
        )

    lane_to_target_replica = getattr(
        scheduler,
        "_ffn_lane_to_target_replica",
        None,
    )
    if type(lane_to_target_replica) is not dict:
        raise RuntimeError(
            "DECODE_FFN receipt requires an exact lane-to-target Replica map"
        )
    if lane not in lane_to_target_replica:
        raise ValueError(
            "DECODE_FFN receipt lane has no target Replica mapping: "
            f"lane={lane}"
        )
    target_replica_id = lane_to_target_replica[lane]
    if type(target_replica_id) is not int or target_replica_id < 0:
        raise ValueError(
            "DECODE_FFN receipt target Replica mapping must be an exact "
            f"non-negative int, got {target_replica_id!r}"
        )
    for field_name in (
        "target_ffn_replica_id",
        "target_execution_replica_id",
    ):
        existing_target = getattr(transfer_info, field_name, None)
        if existing_target is None:
            continue
        if type(existing_target) is not int or existing_target < 0:
            raise ValueError(
                f"DECODE_FFN receipt {field_name} must be None or an exact "
                f"non-negative int, got {existing_target!r}"
            )
        if existing_target != target_replica_id:
            raise ValueError(
                f"DECODE_FFN receipt {field_name} does not match the "
                "lane-to-target Replica mapping: "
                f"field={existing_target}, mapping={target_replica_id}"
            )
    if transfer_info.target_execution_replica_local_id is not None:
        raise ValueError(
            "DECODE_FFN A-to-F target execution identity must not carry "
            "a Replica-local lane"
        )

    return (
        layer_id,
        afd_stage_idx,
        barrier_round_id,
        lane,
        barrier_expected_lanes,
        expected_lanes,
        group_key,
        expected_lane_contract,
        target_replica_id,
    )
def validate_decode_attn_wave_binding(
    scheduler,
    batch: Batch,
    *,
    lane: tuple[int, int],
    afd_stage_idx: int,
    requests: List[Request],
    active_requests: List[Request],
    context: str,
) -> None:
    """Validate one batch against its lane-local active cohort."""

    def require_non_negative_int(value, field_name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{field_name} must be an exact non-negative int, got {value!r}"
            )
        return value

    wave_id = require_non_negative_int(
        getattr(batch, "decode_attn_cohort_id", None),
        f"DECODE_ATTN {context} decode_attn_cohort_id",
    )
    replica_schedulers = getattr(scheduler, "_replica_schedulers", None)
    if type(replica_schedulers) is not dict:
        raise RuntimeError(
            "DECODE_ATTN replica scheduler topology must be an exact dict"
        )
    if lane not in replica_schedulers:
        raise ValueError(
            f"DECODE_ATTN {context} lane is absent from the replica scheduler "
            f"topology: lane={lane}"
        )
    replica_scheduler = replica_schedulers[lane]
    wave_states = getattr(
        replica_scheduler,
        "_decode_attn_active_cohort_states",
        None,
    )
    if type(wave_states) is not dict:
        raise RuntimeError(
            "DECODE_ATTN active cohort states must be an exact dict"
        )
    if wave_id not in wave_states:
        raise ValueError(
            f"DECODE_ATTN {context} references an inactive or unknown cohort: "
            f"cohort_id={wave_id}, lane={lane}"
        )
    wave_state = wave_states[wave_id]
    if type(wave_state) is not dict:
        raise RuntimeError(
            "DECODE_ATTN active cohort state must be an exact dict: "
            f"cohort_id={wave_id}, lane={lane}"
        )

    def require_wave_id_set(
        field_name: str,
        *,
        require_nonempty: bool,
    ) -> set[int]:
        request_ids = wave_state.get(field_name)
        if type(request_ids) is not set:
            raise RuntimeError(
                f"DECODE_ATTN cohort {field_name} must be an exact set, "
                f"got {request_ids!r}"
            )
        if require_nonempty and not request_ids:
            raise RuntimeError(
                f"DECODE_ATTN cohort {field_name} must not be empty"
            )
        for request_id in request_ids:
            if type(request_id) is not int or request_id < 0:
                raise RuntimeError(
                    f"DECODE_ATTN cohort {field_name} must contain exact "
                    f"non-negative ints, got {request_id!r}"
                )
        return request_ids

    all_request_ids = require_wave_id_set(
        "all_request_ids",
        require_nonempty=True,
    )
    pending_request_ids = require_wave_id_set(
        "pending_request_ids",
        require_nonempty=False,
    )
    if not pending_request_ids.issubset(all_request_ids):
        raise RuntimeError(
            "DECODE_ATTN cohort pending_request_ids must be a subset of "
            "all_request_ids"
        )

    batch_wave_request_ids = getattr(
        batch,
        "decode_attn_cohort_request_ids",
        None,
    )
    if type(batch_wave_request_ids) is not tuple:
        raise ValueError(
            f"DECODE_ATTN {context} decode_attn_cohort_request_ids must be an "
            f"exact tuple, got {batch_wave_request_ids!r}"
        )
    normalized_batch_wave_request_ids = [
        require_non_negative_int(
            request_id,
            f"DECODE_ATTN {context} cohort request ID",
        )
        for request_id in batch_wave_request_ids
    ]
    if len(set(normalized_batch_wave_request_ids)) != len(
        normalized_batch_wave_request_ids
    ):
        raise ValueError(
            f"DECODE_ATTN {context} cohort request IDs must not contain "
            "duplicates"
        )
    if set(normalized_batch_wave_request_ids) != all_request_ids:
        raise ValueError(
            f"DECODE_ATTN {context} cohort request IDs do not match active "
            "cohort all_request_ids: "
            f"batch={normalized_batch_wave_request_ids}, "
            f"active={sorted(all_request_ids)}"
        )

    batch_request_ids = [
        require_non_negative_int(
            getattr(request, "id", None),
            f"DECODE_ATTN {context} request ID",
        )
        for request in requests
    ]
    if len(set(batch_request_ids)) != len(batch_request_ids):
        raise ValueError(
            f"DECODE_ATTN {context} request IDs must not contain duplicates"
        )
    requests_outside_wave = sorted(
        set(batch_request_ids) - all_request_ids
    )
    if requests_outside_wave:
        raise ValueError(
            f"DECODE_ATTN {context} contains requests outside the active "
            f"cohort: request_ids={requests_outside_wave}, "
            f"cohort_id={wave_id}"
        )

    active_request_ids = {
        require_non_negative_int(
            getattr(request, "id", None),
            f"DECODE_ATTN {context} active request ID",
        )
        for request in active_requests
    }
    requests_outside_pending = sorted(
        active_request_ids - pending_request_ids
    )
    if requests_outside_pending:
        raise ValueError(
            f"DECODE_ATTN {context} requests are not pending in the active "
            f"cohort: request_ids={requests_outside_pending}, "
            f"cohort_id={wave_id}"
        )

    active_stage_indices = wave_state.get("active_stage_indices")
    if type(active_stage_indices) is not set:
        raise RuntimeError(
            "DECODE_ATTN cohort active_stage_indices must be an exact set, "
            f"got {active_stage_indices!r}"
        )
    for active_stage_idx in active_stage_indices:
        if type(active_stage_idx) is not int or active_stage_idx < 0:
            raise RuntimeError(
                "DECODE_ATTN cohort active_stage_indices must contain exact "
                f"non-negative ints, got {active_stage_idx!r}"
            )
    if afd_stage_idx not in active_stage_indices:
        raise ValueError(
            f"DECODE_ATTN {context} afd_stage_idx is not active in the "
            f"cohort: afd_stage_idx={afd_stage_idx}, "
            f"active_stage_indices={sorted(active_stage_indices)}"
        )


def validate_decode_attn_queued_batch(
    scheduler,
    queued_batch: Batch,
    *,
    queue_lane: tuple[int, int],
    round_key: tuple,
    expected_lanes: List[tuple[int, int]],
    current_batch: Batch,
) -> tuple[int, int]:
    """Validate an existing F-to-A queue entry without mutating it."""

    def require_non_negative_int(value, field_name: str) -> int:
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"{field_name} must be an exact non-negative int, got {value!r}"
            )
        return value

    if type(queued_batch) is not Batch:
        raise RuntimeError(
            "DECODE_ATTN F-to-A waiting room contains a queued object that is "
            f"not an exact Batch: lane={queue_lane}, value={queued_batch!r}"
        )
    if queued_batch is current_batch:
        raise ValueError(
            "Duplicate DECODE_ATTN F-to-A receipt for the same batch object: "
            f"round_key={round_key}, lane={queue_lane}"
        )

    replica_id, next_layer_id, afd_stage_idx = round_key[:3]
    barrier_round_id = round_key[3] if len(round_key) == 4 else None
    queued_lane = (
        require_non_negative_int(
            getattr(queued_batch, "decode_attn_original_replica_id", None),
            "DECODE_ATTN F-to-A queued batch original replica_id",
        ),
        getattr(queued_batch, "decode_attn_original_replica_local_id", None),
    )
    if queued_lane[1] is not None:
        queued_lane = (
            queued_lane[0],
            require_non_negative_int(
                queued_lane[1],
                "DECODE_ATTN F-to-A queued batch original replica_local_id",
            ),
        )
    if queued_lane != queue_lane or queued_lane not in expected_lanes:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued batch lane does not match its waiting "
            f"room: queued={queued_lane}, room={queue_lane}, "
            f"expected_lanes={expected_lanes}"
        )
    if queued_lane[0] != replica_id:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued batch belongs to a different replica: "
            f"round_key={round_key}, queued_lane={queued_lane}"
        )

    queued_global_id = require_non_negative_int(
        getattr(queued_batch, "global_id", None),
        "DECODE_ATTN F-to-A queued batch global_id",
    )
    queued_stage_idx = require_non_negative_int(
        getattr(queued_batch, "afd_stage_idx", None),
        "DECODE_ATTN F-to-A queued batch afd_stage_idx",
    )
    if queued_stage_idx != afd_stage_idx:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued batch stage does not match its waiting "
            f"room: queued={queued_stage_idx}, expected={afd_stage_idx}"
        )

    queued_round_id = getattr(
        queued_batch,
        "decode_attn_barrier_round_id",
        None,
    )
    if queued_round_id is not None:
        queued_round_id = require_non_negative_int(
            queued_round_id,
            "DECODE_ATTN F-to-A queued batch barrier_round_id",
        )
    if queued_round_id != barrier_round_id:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued batch round does not match its waiting "
            f"room: queued={queued_round_id}, expected={barrier_round_id}"
        )

    raw_queued_expected_lanes = getattr(
        queued_batch,
        "decode_attn_barrier_expected_lanes",
        (),
    )
    if raw_queued_expected_lanes is None:
        raw_queued_expected_lanes = ()
    queued_expected_lanes = normalize_lanes(
        raw_queued_expected_lanes,
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN F-to-A queued batch expected lanes",
        require_nonempty=False,
    )
    if queued_expected_lanes:
        queued_replica_lanes = tuple(
            lane for lane in queued_expected_lanes if lane[0] == replica_id
        )
        if queued_replica_lanes != tuple(expected_lanes):
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch expected lanes do not match the "
                f"waiting room: queued={queued_replica_lanes}, "
                f"room={expected_lanes}"
            )

    queued_requests = getattr(queued_batch, "requests", None)
    if type(queued_requests) is not list or not queued_requests:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued Batch requires a non-empty request list"
        )
    active_requests = []
    for queued_request in queued_requests:
        if type(queued_request) is not Request:
            raise ValueError(
                "DECODE_ATTN F-to-A queued Batch contains a queued request "
                "that is not an exact Request: "
                f"value={queued_request!r}"
            )
        completed = getattr(queued_request, "completed", None)
        if type(completed) is not bool:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued request.completed must be an exact "
                f"bool, got {completed!r}"
            )
        roundtrip_inflight = getattr(
            queued_request,
            "af_roundtrip_inflight",
            None,
        )
        if type(roundtrip_inflight) is not bool:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued request.af_roundtrip_inflight must "
                f"be an exact bool, got {roundtrip_inflight!r}"
            )
        if roundtrip_inflight is not False:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued request roundtrip must already be "
                f"complete, got {roundtrip_inflight!r}"
            )
        if not completed:
            active_requests.append(queued_request)
    if not active_requests:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued Batch requires an active request"
        )

    validate_decode_attn_wave_binding(
        scheduler,
        queued_batch,
        lane=queue_lane,
        afd_stage_idx=queued_stage_idx,
        requests=queued_requests,
        active_requests=active_requests,
        context="queued batch",
    )

    queued_layers = [
        require_non_negative_int(
            getattr(queued_request, "completed_layer_count", None),
            "DECODE_ATTN F-to-A queued request completed_layer_count",
        )
        for queued_request in active_requests
    ]
    if set(queued_layers) != {next_layer_id}:
        raise RuntimeError(
            "DECODE_ATTN F-to-A queued requests do not match the waiting-room "
            f"layer: queued={queued_layers}, expected={next_layer_id}"
        )

    queued_request_token_indices = [
        require_non_negative_int(
            getattr(queued_request, "current_decode_token_index", None),
            "DECODE_ATTN F-to-A queued request decode_token_index",
        )
        for queued_request in active_requests
    ]
    queued_replay_token_index = getattr(
        queued_batch,
        "replay_decode_token_index",
        None,
    )
    if queued_replay_token_index is None:
        if len(set(queued_request_token_indices)) != 1:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued Batch has mixed decode token indices "
                "without replay identity"
            )
        queued_token_index = queued_request_token_indices[0]
    else:
        queued_token_index = require_non_negative_int(
            queued_replay_token_index,
            "DECODE_ATTN F-to-A queued batch replay_decode_token_index",
        )
        if queued_token_index != queued_request_token_indices[0]:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch replay decode token does not "
                "match its active request head"
            )
    return queued_global_id, queued_token_index

def validate_decode_attn_receipt(
    scheduler,
    batch: Batch,
    transfer_info: "M2NTransferInfo",
    *,
    expected_roundtrip_inflight: bool,
    request_end_deferred: bool = False,
) -> Dict[str, Any]:
    """Validate one F-to-A receipt without mutating runtime state."""

    def require_non_negative_int(value, field_name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{field_name} must be an exact non-negative int, got {value!r}"
            )
        return value

    if type(expected_roundtrip_inflight) is not bool:
        raise ValueError(
            "DECODE_ATTN expected roundtrip state must be an exact bool, "
            f"got {expected_roundtrip_inflight!r}"
        )
    if type(request_end_deferred) is not bool:
        raise ValueError(
            "DECODE_ATTN request_end_deferred must be an exact bool, "
            f"got {request_end_deferred!r}"
        )
    if request_end_deferred and expected_roundtrip_inflight is not False:
        raise ValueError(
            "DECODE_ATTN deferred request end requires the projected "
            "roundtrip_inflight=False state"
        )

    scheduler.validate_m2n_arrival_target(transfer_info)
    if scheduler._cluster_type is not ClusterType.DECODE_ATTN:
        raise ValueError(
            "DECODE_ATTN receipt validation requires a DECODE_ATTN scheduler, "
            f"got {scheduler._cluster_type.name}"
        )
    if (
        transfer_info.source_cluster_type is not ClusterType.DECODE_FFN
        or transfer_info.target_cluster_type is not ClusterType.DECODE_ATTN
    ):
        raise ValueError(
            "DECODE_ATTN receipt validation requires an exact "
            "DECODE_FFN -> DECODE_ATTN transfer"
        )
    if batch is not transfer_info.batch:
        raise ValueError(
            "DECODE_ATTN M2N batch identity mismatch: batch is not "
            "transfer_info.batch"
        )

    source_replica_id = require_non_negative_int(
        getattr(transfer_info, "source_replica_id", None),
        "DECODE_ATTN receipt source_replica_id",
    )
    source_replica_local_id = getattr(transfer_info, "source_replica_local_id", None)
    if source_replica_local_id is not None:
        source_replica_local_id = require_non_negative_int(
            source_replica_local_id,
            "DECODE_ATTN receipt source_replica_local_id",
        )
    transfer_layer_id = require_non_negative_int(
        getattr(transfer_info, "layer_id", None),
        "DECODE_ATTN receipt layer_id",
    )
    transfer_stage_idx = require_non_negative_int(
        getattr(transfer_info, "afd_stage_idx", None),
        "DECODE_ATTN receipt afd_stage_idx",
    )
    replica_id = require_non_negative_int(
        getattr(batch, "decode_attn_original_replica_id", None),
        "DECODE_ATTN receipt decode_attn_original_replica_id",
    )
    replica_local_id = getattr(batch, "decode_attn_original_replica_local_id", None)
    if replica_local_id is not None:
        replica_local_id = require_non_negative_int(
            replica_local_id,
            "DECODE_ATTN receipt decode_attn_original_replica_local_id",
        )
    batch_global_id = require_non_negative_int(
        getattr(batch, "global_id", None),
        "DECODE_ATTN receipt batch.global_id",
    )
    afd_stage_idx = require_non_negative_int(
        getattr(batch, "afd_stage_idx", None),
        "DECODE_ATTN receipt batch.afd_stage_idx",
    )
    lane = (replica_id, replica_local_id)
    source_lane = (source_replica_id, source_replica_local_id)
    if source_lane != lane:
        raise ValueError(
            "DECODE_ATTN receipt source lane does not match the original ATTN "
            f"lane: source={source_lane}, original={lane}"
        )
    if transfer_stage_idx != afd_stage_idx:
        raise ValueError(
            "DECODE_ATTN receipt afd_stage_idx does not match the batch stage: "
            f"transfer={transfer_stage_idx}, batch={afd_stage_idx}"
        )

    requests = getattr(batch, "requests", None)
    if type(requests) is not list or not requests:
        raise ValueError("DECODE_ATTN F-to-A receipt requires a non-empty request list")
    active_requests = []
    for request in requests:
        if type(request) is not Request:
            raise ValueError(
                "DECODE_ATTN F-to-A incoming receipt contains a request "
                "that is not an exact Request: "
                f"value={request!r}"
            )
        completed = getattr(request, "completed", None)
        if type(completed) is not bool:
            raise ValueError(
                "DECODE_ATTN receipt request.completed must be an exact bool, "
                f"got {completed!r} for request {getattr(request, 'id', '?')}"
            )
        roundtrip_inflight = getattr(request, "af_roundtrip_inflight", None)
        if type(roundtrip_inflight) is not bool:
            raise ValueError(
                "DECODE_ATTN receipt request.af_roundtrip_inflight must be an "
                f"exact bool, got {roundtrip_inflight!r} for request "
                f"{getattr(request, 'id', '?')}"
            )
        roundtrip_matches = (
            roundtrip_inflight is True
            if request_end_deferred
            else roundtrip_inflight is expected_roundtrip_inflight
        )
        if not roundtrip_matches:
            raise ValueError(
                "DECODE_ATTN receipt request roundtrip state does not match the "
                f"admission phase: expected={expected_roundtrip_inflight}, "
                f"actual={roundtrip_inflight}, request="
                f"{getattr(request, 'id', '?')}"
            )
        if not completed:
            active_requests.append(request)
    if not active_requests:
        raise ValueError(
            "DECODE_ATTN F-to-A receipt requires at least one active request"
        )

    active_layer_ids = [
        require_non_negative_int(
            getattr(request, "completed_layer_count", None),
            "DECODE_ATTN receipt active request completed_layer_count",
        )
        for request in active_requests
    ]
    if len(set(active_layer_ids)) != 1:
        raise ValueError(
            "DECODE_ATTN receipt active requests must have a consistent layer: "
            f"layers={active_layer_ids}"
        )
    current_layer_id = active_layer_ids[0]
    if transfer_layer_id != current_layer_id:
        raise ValueError(
            "DECODE_ATTN receipt layer_id does not match the active request layer: "
            f"transfer={transfer_layer_id}, active={current_layer_id}"
        )

    total_layers = require_non_negative_int(
        getattr(scheduler._config.replica_config.model_config, "num_layers", None),
        "DECODE_ATTN model num_layers",
    )
    if total_layers == 0:
        raise ValueError("DECODE_ATTN model num_layers must be positive")
    if current_layer_id >= total_layers:
        raise ValueError(
            "DECODE_ATTN receipt active request layer must be below num_layers: "
            f"layer={current_layer_id}, num_layers={total_layers}"
        )
    next_layer_id = current_layer_id + 1
    is_last_layer = next_layer_id == total_layers

    active_decode_token_indices = [
        require_non_negative_int(
            getattr(request, "current_decode_token_index", None),
            "DECODE_ATTN receipt active request decode_token_index",
        )
        for request in active_requests
    ]
    replay_decode_token_index = getattr(
        batch,
        "replay_decode_token_index",
        None,
    )
    if replay_decode_token_index is None:
        if len(set(active_decode_token_indices)) != 1:
            raise ValueError(
                "DECODE_ATTN F-to-A receipt requires a batch-level replay decode "
                "token index for mixed active request positions; got "
                f"{active_decode_token_indices}"
            )
        decode_token_index = active_decode_token_indices[0]
    else:
        decode_token_index = require_non_negative_int(
            replay_decode_token_index,
            "DECODE_ATTN receipt replay_decode_token_index",
        )
        if decode_token_index != active_decode_token_indices[0]:
            raise ValueError(
                "DECODE_ATTN receipt replay_decode_token_index does not match the "
                "active batch head: "
                f"replay={decode_token_index}, head={active_decode_token_indices[0]}"
            )

    scheduler_expected_lanes = normalize_lanes(
        scheduler._get_decode_attn_f2a_expected_lanes(
            replica_id,
            afd_stage_idx=afd_stage_idx,
        ),
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN F-to-A scheduler lane topology",
        require_nonempty=True,
    )
    if lane not in scheduler_expected_lanes:
        raise ValueError(
            "Unexpected lane observed in DECODE_ATTN F-to-A scheduler topology: "
            f"lane={lane}, expected_lanes={scheduler_expected_lanes}"
        )
    scheduler_expected_lane_set = set(scheduler_expected_lanes)

    replica_scheduler_type = getattr(scheduler, "_replica_scheduler_type", None)
    if type(replica_scheduler_type) is not ReplicaSchedulerType:
        raise RuntimeError(
            "DECODE_ATTN receipt requires an exact replica scheduler type, "
            f"got {replica_scheduler_type!r}"
        )
    cohort_id = getattr(batch, "decode_attn_cohort_id", None)
    cohort_request_ids = getattr(
        batch,
        "decode_attn_cohort_request_ids",
        None,
    )
    wave_scheduler_types = {
        ReplicaSchedulerType.VLLM_V1,
        ReplicaSchedulerType.SGLANG,
        ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
        ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
        ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
    }
    if replica_scheduler_type in wave_scheduler_types:
        scheduler._validate_decode_attn_wave_binding(
            batch,
            lane=lane,
            afd_stage_idx=afd_stage_idx,
            requests=requests,
            active_requests=active_requests,
            context="receipt",
        )
    elif cohort_id is not None or cohort_request_ids is not None:
        raise ValueError(
            "DECODE_ATTN receipt from a non-cohort scheduler must not carry "
            "decode_attn_cohort_id or decode_attn_cohort_request_ids: "
            f"scheduler_type={replica_scheduler_type}, "
            f"cohort_id={cohort_id!r}, "
            f"cohort_request_ids={cohort_request_ids!r}"
        )

    barrier_round_id = getattr(batch, "decode_attn_barrier_round_id", None)
    if barrier_round_id is not None:
        barrier_round_id = require_non_negative_int(
            barrier_round_id,
            "DECODE_ATTN receipt barrier_round_id",
        )

    raw_expected_lanes = getattr(
        batch,
        "decode_attn_barrier_expected_lanes",
        (),
    )
    if raw_expected_lanes is None:
        raw_expected_lanes = ()
    barrier_expected_lanes = normalize_lanes(
        raw_expected_lanes,
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN receipt expected lane metadata",
        require_nonempty=False,
    )
    if barrier_expected_lanes and lane not in barrier_expected_lanes:
        raise ValueError(
            "Unexpected lane observed in DECODE_ATTN receipt expected lane "
            f"metadata: lane={lane}, expected_lanes={barrier_expected_lanes}"
        )
    scheduler_lane_sets_by_replica = {
        replica_id: scheduler_expected_lane_set,
    }
    metadata_lanes_outside_topology = []
    for expected_lane in barrier_expected_lanes:
        expected_replica_id = expected_lane[0]
        expected_replica_lane_set = scheduler_lane_sets_by_replica.get(
            expected_replica_id
        )
        if expected_replica_lane_set is None:
            expected_replica_lanes = normalize_lanes(
                scheduler._get_decode_attn_f2a_expected_lanes(
                    expected_replica_id,
                    afd_stage_idx=afd_stage_idx,
                ),
                identity_scope=LaneIdentityScope.FULL_STAGE,
                field_name=(
                    "DECODE_ATTN F-to-A scheduler lane topology for "
                    f"replica {expected_replica_id}"
                ),
                require_nonempty=True,
            )
            expected_replica_lane_set = set(expected_replica_lanes)
            scheduler_lane_sets_by_replica[expected_replica_id] = (
                expected_replica_lane_set
            )
        if expected_lane not in expected_replica_lane_set:
            metadata_lanes_outside_topology.append(expected_lane)
    if metadata_lanes_outside_topology:
        raise ValueError(
            "DECODE_ATTN receipt expected lanes are outside the scheduler "
            f"topology: outside={metadata_lanes_outside_topology}, "
            f"topology_by_replica={scheduler_lane_sets_by_replica}"
        )
    filtered_expected_lanes = tuple(
        expected_lane
        for expected_lane in barrier_expected_lanes
        if expected_lane[0] == replica_id
    )

    if barrier_round_id is None:
        round_key = (replica_id, next_layer_id, afd_stage_idx)
    else:
        round_key = (
            replica_id,
            next_layer_id,
            afd_stage_idx,
            barrier_round_id,
        )

    waiting_rooms = getattr(scheduler, "_f2a_waiting_by_round", None)
    if type(waiting_rooms) is not dict:
        raise RuntimeError(
            "DECODE_ATTN scheduler missing exact _f2a_waiting_by_round mapping"
        )

    room = waiting_rooms.get(round_key)
    if is_last_layer and room is not None:
        raise RuntimeError(
            "DECODE_ATTN final F-to-A receipt must not have an existing "
            f"waiting room: round_key={round_key}"
        )
    existing_expected_lanes: tuple[tuple[int, int], ...] = ()
    if room is not None:
        if type(room) is not dict:
            raise RuntimeError(
                "DECODE_ATTN F-to-A waiting room must be an exact dict: "
                f"round_key={round_key}"
            )
        if "expected_lanes" not in room:
            raise RuntimeError(
                "DECODE_ATTN F-to-A waiting room missing expected lanes: "
                f"round_key={round_key}"
            )
        raw_room_expected_lanes = room["expected_lanes"]
        if raw_room_expected_lanes is not None:
            existing_expected_lanes = tuple(
                normalize_lanes(
                    raw_room_expected_lanes,
                    identity_scope=LaneIdentityScope.FULL_STAGE,
                    field_name="DECODE_ATTN F-to-A waiting room expected lanes",
                    require_nonempty=True,
                )
            )
            if any(
                room_replica_id != replica_id
                for room_replica_id, _ in existing_expected_lanes
            ):
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room expected lanes contain a "
                    f"different replica: round_key={round_key}, "
                    f"expected_lanes={existing_expected_lanes}"
                )
            room_lanes_outside_topology = [
                expected_lane
                for expected_lane in existing_expected_lanes
                if expected_lane not in scheduler_expected_lane_set
            ]
            if room_lanes_outside_topology:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room expected lanes are outside "
                    f"the scheduler topology: outside={room_lanes_outside_topology}, "
                    f"expected_lanes={scheduler_expected_lanes}"
                )
        if (
            filtered_expected_lanes
            and existing_expected_lanes
            and existing_expected_lanes != filtered_expected_lanes
        ):
            raise ValueError(
                "Mismatched DECODE_ATTN F-to-A expected lanes contract for the "
                f"same round: round_key={round_key}, "
                f"existing={existing_expected_lanes}, "
                f"new={filtered_expected_lanes}"
            )

    stored_expected_lanes = (
        filtered_expected_lanes
        or existing_expected_lanes
        or None
    )
    expected_lanes = list(
        stored_expected_lanes
        if stored_expected_lanes is not None
        else tuple(scheduler_expected_lanes)
    )
    if lane not in expected_lanes:
        raise ValueError(
            "Unexpected lane observed in DECODE_ATTN F-to-A waiting room: "
            f"round_key={round_key}, lane={lane}, "
            f"expected_lanes={expected_lanes}"
        )

    if room is not None:
        per_lane_queues = room.get("per_lane_queues")
        if (
            type(per_lane_queues) is not defaultdict
            or per_lane_queues.default_factory is not deque
        ):
            raise RuntimeError(
                "DECODE_ATTN F-to-A waiting room per_lane_queues must be a "
                f"defaultdict(deque): round_key={round_key}"
            )
        room_lanes = normalize_lanes(
            list(per_lane_queues.keys()),
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_ATTN F-to-A waiting room queue lanes",
            require_nonempty=False,
        )
        queued_identities_by_lane = {}
        for room_lane in room_lanes:
            if room_lane not in expected_lanes:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room contains a queue outside "
                    f"its expected lanes: lane={room_lane}, "
                    f"expected_lanes={expected_lanes}"
                )
            lane_queue = per_lane_queues.get(room_lane)
            if type(lane_queue) is not deque:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room lane queue must be a deque: "
                    f"lane={room_lane}, queue={lane_queue!r}"
                )
            queued_identities_by_lane[room_lane] = [
                scheduler._validate_decode_attn_f2a_queued_batch(
                    queued_batch,
                    queue_lane=room_lane,
                    round_key=round_key,
                    expected_lanes=expected_lanes,
                    current_batch=batch,
                )
                for queued_batch in lane_queue
            ]

        current_identity = (batch_global_id, decode_token_index)
        if barrier_round_id is not None:
            for queued_identities in queued_identities_by_lane.values():
                for queued_identity in queued_identities:
                    if queued_identity != current_identity:
                        raise RuntimeError(
                            "DECODE_ATTN F-to-A explicit round mixes batch/token "
                            f"identities: queued={queued_identity}, "
                            f"current={current_identity}, round_key={round_key}"
                        )
        else:
            max_queue_depth = max(
                (
                    len(queued_identities)
                    for queued_identities in queued_identities_by_lane.values()
                ),
                default=0,
            )
            for queue_position in range(max_queue_depth):
                position_identities = {
                    queued_identities[queue_position]
                    for queued_identities in queued_identities_by_lane.values()
                    if queue_position < len(queued_identities)
                }
                if len(position_identities) > 1:
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A legacy FIFO position mixes "
                        f"batch/token identities: position={queue_position}, "
                        f"identities={sorted(position_identities)}, "
                        f"round_key={round_key}"
                    )

            current_lane_depth = len(
                queued_identities_by_lane.get(lane, ())
            )
            matching_position_identities = {
                queued_identities[current_lane_depth]
                for room_lane, queued_identities in queued_identities_by_lane.items()
                if room_lane != lane
                and current_lane_depth < len(queued_identities)
            }
            if (
                matching_position_identities
                and matching_position_identities != {current_identity}
            ):
                raise RuntimeError(
                    "DECODE_ATTN F-to-A legacy FIFO arrival identity does not "
                    f"match position={current_lane_depth}: "
                    f"queued={sorted(matching_position_identities)}, "
                    f"current={current_identity}, round_key={round_key}"
                )

    return {
        "current_layer_id": current_layer_id,
        "next_layer_id": next_layer_id,
        "replica_id": replica_id,
        "replica_local_id": replica_local_id,
        "lane": lane,
        "batch_global_id": batch_global_id,
        "decode_token_index": decode_token_index,
        "afd_stage_idx": afd_stage_idx,
        "barrier_round_id": barrier_round_id,
        "round_key": round_key,
        "stored_expected_lanes": stored_expected_lanes,
        "expected_lanes": expected_lanes,
        "room": room,
        "is_last_layer": is_last_layer,
    }
def prepare_dp_padding(

    picked: List[tuple],
) -> tuple[List[tuple[Any, Any]], Optional[tuple[int, List[int]]]]:
    """Build DP-padding replacements without mutating queued batches."""

    from frontier.entities.batch import AFDStageMetadata
    from frontier.entities.m2n_transfer_info import M2NTransferInfo

    batches_with_meta = []
    for picked_index, picked_entry in enumerate(picked):
        if type(picked_entry) is not tuple or len(picked_entry) != 2:
            raise RuntimeError(
                "DECODE_FFN promotion entry must be an exact "
                f"(batch, transfer_info) tuple, got {picked_entry!r} at "
                f"index {picked_index}"
            )
        batch, transfer_info = picked_entry
        if type(transfer_info) is not M2NTransferInfo:
            raise RuntimeError(
                "DECODE_FFN promotion transfer must be an exact "
                f"M2NTransferInfo at index {picked_index}"
            )
        if transfer_info.batch is not batch:
            raise RuntimeError(
                "DECODE_FFN promotion batch identity does not match "
                f"transfer_info.batch at index {picked_index}"
            )
        is_idle = getattr(batch, "is_idle", None)
        if type(is_idle) is not bool:
            raise RuntimeError(
                "DECODE_FFN promotion batch is_idle must be an exact bool, "
                f"got {is_idle!r}"
            )
        metadata = getattr(batch, "afd_stage_metadata", None)
        if is_idle or metadata is None:
            continue
        if type(metadata) is not AFDStageMetadata:
            raise RuntimeError(
                "DECODE_FFN promotion afd_stage_metadata must be an exact "
                f"AFDStageMetadata, got {type(metadata).__name__}"
            )
        requests = getattr(batch, "requests", None)
        num_tokens = getattr(batch, "num_tokens", None)
        if type(requests) is not list or type(num_tokens) is not list:
            raise RuntimeError(
                "DECODE_FFN promotion batch requests and num_tokens must be "
                "exact lists before DP padding"
            )
        if len(requests) != len(num_tokens):
            raise RuntimeError(
                "DECODE_FFN promotion batch request/token lengths mismatch: "
                f"requests={len(requests)}, num_tokens={len(num_tokens)}"
            )
        for token_count in num_tokens:
            if type(token_count) is not int or token_count < 0:
                raise RuntimeError(
                    "DECODE_FFN promotion num_tokens must contain exact "
                    f"non-negative ints, got {token_count!r}"
                )
        batches_with_meta.append((batch, metadata, num_tokens))

    if len(batches_with_meta) <= 1:
        return [], None

    num_stages = batches_with_meta[0][1].num_stages
    if type(num_stages) is not int or num_stages <= 0:
        raise RuntimeError(
            "DECODE_FFN promotion metadata num_stages must be an exact "
            f"positive int, got {num_stages!r}"
        )

    all_stage_lens = []
    for batch, metadata, num_tokens in batches_with_meta:
        if type(metadata.num_stages) is not int or metadata.num_stages <= 0:
            raise RuntimeError(
                "DECODE_FFN promotion metadata num_stages must be an exact "
                f"positive int, got {metadata.num_stages!r}"
            )
        if metadata.num_stages != num_stages:
            raise ValueError(
                "Inconsistent num_stages across DP lanes: "
                f"expected {num_stages}, got {metadata.num_stages}"
            )
        stage_lens = AFDStageMetadata.compute_stage_token_lens(
            num_reqs=len(batch.requests),
            num_tokens_per_req=list(num_tokens),
            num_stages=num_stages,
        )
        while len(stage_lens) < num_stages:
            stage_lens.append(1)
        if len(stage_lens) != num_stages:
            raise RuntimeError(
                "DECODE_FFN promotion stage-token plan does not match "
                f"num_stages: planned={len(stage_lens)}, "
                f"num_stages={num_stages}"
            )
        all_stage_lens.append(stage_lens)

    dp_stage_max_tokens = [
        max(lane_lens[stage_index] for lane_lens in all_stage_lens)
        for stage_index in range(num_stages)
    ]
    padding_plan = [
        (
            batch,
            metadata.with_dp_padding(
                dp_stage_max_tokens=dp_stage_max_tokens,
            ),
        )
        for batch, metadata, _ in batches_with_meta
    ]
    return padding_plan, (len(batches_with_meta), dp_stage_max_tokens)
def prepare_decode_attn_idle_lanes(

    *,
    time: float,
    group_key: tuple[int, int],
    idle_lanes: List[tuple[int, int]],
    is_moe: bool,
) -> List[tuple[tuple[int, int], tuple[int, Batch]]]:
    """Build A-to-F idle entries without mutating the waiting room."""

    normalized_idle_lanes = normalize_lanes(
        idle_lanes,
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN A-to-F prepared idle lanes",
        require_nonempty=False,
    )
    if type(is_moe) is not bool:
        raise RuntimeError(
            "DECODE_ATTN A-to-F idle batch is_moe must be an exact bool, "
            f"got {is_moe!r}"
        )

    layer_id, afd_stage_idx = group_key
    prepared_entries = []
    for missing_lane in normalized_idle_lanes:
        idle_batch = Batch(
            replica_id=missing_lane[0],
            requests=[],
            num_tokens=[],
            is_idle=True,
            is_moe=is_moe,
        )
        idle_batch.afd_stage_idx = afd_stage_idx
        idle_batch.decode_attn_original_replica_id = missing_lane[0]
        idle_batch.decode_attn_original_replica_local_id = missing_lane[1]
        idle_batch.time = time
        prepared_entries.append(
            (missing_lane, (layer_id, idle_batch))
        )
    return prepared_entries
def validate_decode_attn_a2f_batch_entry(

scheduler,
    *,
    batch: Batch,
    lane: tuple[int, int],
    layer_id: int,
    afd_stage_idx: int,
    model_is_moe: bool,
    context: str,
    allow_idle: bool,
) -> None:
    """Validate one A-to-F Batch before any scheduler state is touched."""

    if type(batch) is not Batch:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} must be an exact Batch, "
            f"got {type(batch).__name__}"
        )
    if type(model_is_moe) is not bool:
        raise RuntimeError(
            "DECODE_ATTN A-to-F model_config.is_moe must be an exact bool, "
            f"got {model_is_moe!r}"
        )
    normalized_lane = normalize_lanes(
        [lane],
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name=f"DECODE_ATTN A-to-F {context} lane",
        require_nonempty=True,
    )[0]
    layer_id = scheduler._validate_decode_attn_a2f_topology_value(
        layer_id,
        field_name=f"{context} layer_id",
    )
    afd_stage_idx = scheduler._validate_decode_attn_a2f_topology_value(
        afd_stage_idx,
        field_name=f"{context} afd_stage_idx",
    )

    raw_is_idle = getattr(batch, "is_idle", None)
    raw_is_moe = getattr(batch, "is_moe", None)
    if type(raw_is_idle) is not bool:
        raise RuntimeError(
            f"DECODE_ATTN A-to-F {context} is_idle must be an exact bool, "
            f"got {raw_is_idle!r}"
        )
    if type(raw_is_moe) is not bool:
        raise RuntimeError(
            f"DECODE_ATTN A-to-F {context} is_moe must be an exact bool, "
            f"got {raw_is_moe!r}"
        )
    if raw_is_moe is not model_is_moe:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} is_moe does not match model "
            f"configuration: batch={raw_is_moe}, model={model_is_moe}"
        )
    if raw_is_idle and not allow_idle:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} idle Batch is not valid for an "
            "incoming lane"
        )

    batch_stage_idx = getattr(batch, "afd_stage_idx", None)
    if type(batch_stage_idx) is not int or batch_stage_idx != afd_stage_idx:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} afd_stage_idx mismatch: "
            f"expected={afd_stage_idx}, got={batch_stage_idx!r}"
        )
    batch_lane = (
        getattr(batch, "decode_attn_original_replica_id", None),
        getattr(batch, "decode_attn_original_replica_local_id", None),
    )
    if (
        type(batch_lane[0]) is not int
        or batch_lane[0] < 0
        or (
            batch_lane[1] is not None
            and (type(batch_lane[1]) is not int or batch_lane[1] < 0)
        )
    ):
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} original lane must contain a "
            f"Replica ID and optional full-stage identity, got {batch_lane!r}"
        )
    if batch_lane != normalized_lane:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} original lane mismatch: "
            f"expected={normalized_lane}, got={batch_lane}"
        )

    requests = getattr(batch, "requests", None)
    num_tokens = getattr(batch, "num_tokens", None)
    if type(requests) is not list:
        raise RuntimeError(
            f"DECODE_ATTN A-to-F {context} requests must be an exact list, "
            f"got {type(requests).__name__}"
        )
    if type(num_tokens) is not list or len(num_tokens) != len(requests):
        raise RuntimeError(
            f"DECODE_ATTN A-to-F {context} num_tokens must be an exact list "
            "matching requests"
        )
    for token_count in num_tokens:
        if type(token_count) is not int or token_count < 0:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} num_tokens must contain exact "
                f"non-negative ints, got {token_count!r}"
            )
    if raw_is_idle:
        if requests or num_tokens:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} idle Batch must not contain "
                "requests or token counts"
            )
        active_requests: List[Request] = []
    else:
        if not requests:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} non-idle Batch must contain "
                "requests"
            )
        active_requests = []
        for request in requests:
            if type(request) is not Request:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} contains a request that "
                    f"is not an exact Request: {request!r}"
                )
            request_id = getattr(request, "id", None)
            if type(request_id) is not int or request_id < 0:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} request ID must be an exact "
                    f"non-negative int, got {request_id!r}"
                )
            completed = getattr(request, "completed", None)
            if type(completed) is not bool:
                raise RuntimeError(
                    f"DECODE_ATTN A-to-F {context} request.completed must be "
                    f"an exact bool, got {completed!r}"
                )
            request_layer_id = getattr(request, "completed_layer_count", None)
            if type(request_layer_id) is not int or request_layer_id < 0:
                raise RuntimeError(
                    f"DECODE_ATTN A-to-F {context} request layer must be an "
                    f"exact non-negative int, got {request_layer_id!r}"
                )
            if not completed:
                active_requests.append(request)
                if request_layer_id != layer_id:
                    raise ValueError(
                        f"DECODE_ATTN A-to-F {context} request layer mismatch: "
                        f"expected={layer_id}, got={request_layer_id}, "
                        f"request_id={request_id}"
                    )
        if not active_requests:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} non-idle Batch has no active "
                "requests"
            )

    decode_ffn_layer_id = getattr(batch, "decode_ffn_layer_id", None)
    if decode_ffn_layer_id is not None:
        if type(decode_ffn_layer_id) is not int or decode_ffn_layer_id < 0:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} decode_ffn_layer_id must be "
                f"None or an exact non-negative int, got {decode_ffn_layer_id!r}"
            )
        if decode_ffn_layer_id != layer_id:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} decode_ffn_layer_id mismatch: "
                f"expected={layer_id}, got={decode_ffn_layer_id}"
            )

    cohort_id = getattr(batch, "decode_attn_cohort_id", None)
    cohort_request_ids = getattr(batch, "decode_attn_cohort_request_ids", None)
    if cohort_id is None:
        if cohort_request_ids is not None:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} has cohort request IDs without "
                "a cohort ID"
            )
        return
    if type(cohort_id) is not int or cohort_id < 0:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} cohort ID must be an exact "
            f"non-negative int, got {cohort_id!r}"
        )
    if type(cohort_request_ids) is not tuple:
        raise ValueError(
            f"DECODE_ATTN A-to-F {context} cohort request IDs must be an "
            f"exact tuple, got {cohort_request_ids!r}"
        )
    validate_decode_attn_wave_binding(
        scheduler,
        batch,
        lane=normalized_lane,
        afd_stage_idx=afd_stage_idx,
        requests=requests,
        active_requests=active_requests,
        context=f"A-to-F {context}",
    )
    scheduler._validate_decode_attn_a2f_wave_phase(
        batch,
        layer_id=layer_id,
        afd_stage_idx=afd_stage_idx,
        context=context,
    )
