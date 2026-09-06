"""Pure validation helpers for PD-AF scheduler state."""

from collections import defaultdict, deque
from typing import Any, Callable, Optional

from frontier.entities import Batch
from frontier.scheduler.utils.pdaf_transfer import LaneIdentityScope


def validate_decode_attn_a2f_waiting_room(
    *,
    group_key: tuple[int, int],
    room: dict,
    expected_lane_contract: tuple[tuple[int, int], ...],
    incoming_batch: Optional[Batch] = None,
    topology_validator: Callable[..., int],
    lane_normalizer: Callable[..., list[tuple[int, int | None]]],
    batch_validator: Callable[..., None],
    model_is_moe: Any,
) -> tuple[tuple[int, int], ...]:
    """Validate one A-to-F waiting room without mutating runtime state.

    The scheduler supplies callbacks for topology and batch semantics so this
    helper remains independent of scheduler state while preserving the legacy
    validation and exception behavior.
    """

    if type(group_key) is not tuple or len(group_key) != 2:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room key must be an exact "
            f"(layer_id, afd_stage_idx) tuple, got {group_key!r}"
        )
    layer_id = topology_validator(
        group_key[0],
        field_name="waiting-room layer_id",
    )
    afd_stage_idx = topology_validator(
        group_key[1],
        field_name="waiting-room afd_stage_idx",
    )
    if type(model_is_moe) is not bool:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room model_config.is_moe must be "
            f"an exact bool, got {model_is_moe!r}"
        )

    normalized_expected_lanes = tuple(
        sorted(
            lane_normalizer(
                expected_lane_contract,
                identity_scope=LaneIdentityScope.FULL_STAGE,
                field_name="DECODE_ATTN A-to-F expected lane topology",
                require_nonempty=True,
            )
        )
    )
    if expected_lane_contract != normalized_expected_lanes:
        raise RuntimeError(
            "DECODE_ATTN A-to-F expected lane topology must be an exact "
            f"canonical tuple, got {expected_lane_contract!r}"
        )

    if type(room) is not dict:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting room must be an exact dict, "
            f"got {type(room).__name__}"
        )
    expected_room_fields = {
        "per_lane_queues",
        "expected_lane_contract",
    }
    if set(room) != expected_room_fields:
        missing_fields = expected_room_fields - set(room)
        if "expected_lane_contract" in missing_fields:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting room is missing the expected "
                "lane contract"
            )
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room schema mismatch: "
            f"expected={sorted(expected_room_fields)}, actual={sorted(room)}"
        )

    raw_room_lanes = room["expected_lane_contract"]
    if type(raw_room_lanes) is not tuple:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room expected lane contract must "
            f"be an exact tuple, got {raw_room_lanes!r}"
        )
    room_lanes = tuple(
        sorted(
            lane_normalizer(
                raw_room_lanes,
                identity_scope=LaneIdentityScope.FULL_STAGE,
                field_name=(
                    "DECODE_ATTN A-to-F waiting-room expected lane contract"
                ),
                require_nonempty=True,
            )
        )
    )
    if raw_room_lanes != room_lanes:
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room expected lane contract must "
            f"be canonical, got {raw_room_lanes!r}"
        )
    if room_lanes != normalized_expected_lanes:
        raise ValueError(
            "DECODE_ATTN A-to-F waiting-room lane contract mismatch: "
            f"room={room_lanes}, expected={normalized_expected_lanes}"
        )

    per_lane_queues = room["per_lane_queues"]
    if (
        type(per_lane_queues) is not defaultdict
        or per_lane_queues.default_factory is not deque
    ):
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room per_lane_queues must be an "
            "exact defaultdict(deque)"
        )
    queue_lanes = lane_normalizer(
        tuple(per_lane_queues),
        identity_scope=LaneIdentityScope.FULL_STAGE,
        field_name="DECODE_ATTN A-to-F waiting-room queue lanes",
        require_nonempty=False,
    )
    if not set(queue_lanes).issubset(set(room_lanes)):
        raise RuntimeError(
            "DECODE_ATTN A-to-F waiting-room queue lane is outside the "
            f"expected contract: queues={queue_lanes}, contract={room_lanes}"
        )

    seen_batch_identities = set()
    for queue_lane, lane_queue in per_lane_queues.items():
        if type(lane_queue) is not deque:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room lane queue must be an "
                f"exact deque: lane={queue_lane}, "
                f"got={type(lane_queue).__name__}"
            )
        for queue_index, queued_entry in enumerate(lane_queue):
            if type(queued_entry) is not tuple or len(queued_entry) != 2:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued entry must be "
                    f"an exact (layer_id, Batch) tuple: lane={queue_lane}, "
                    f"index={queue_index}, value={queued_entry!r}"
                )
            queued_layer_id, queued_batch = queued_entry
            if type(queued_layer_id) is not int or queued_layer_id < 0:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued layer_id must "
                    f"be an exact non-negative int, got {queued_layer_id!r}"
                )
            if queued_layer_id != layer_id:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued layer mismatch: "
                    f"room={layer_id}, queued={queued_layer_id}"
                )
            if type(queued_batch) is not Batch:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued batch must be "
                    f"an exact Batch, got {type(queued_batch).__name__}"
                )
            if incoming_batch is not None and queued_batch is incoming_batch:
                raise ValueError(
                    "DECODE_ATTN A-to-F waiting room already contains the "
                    "incoming batch object"
                )
            queued_batch_identity = id(queued_batch)
            if queued_batch_identity in seen_batch_identities:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting room contains a duplicate "
                    "queued batch object"
                )
            seen_batch_identities.add(queued_batch_identity)

            queued_is_idle = getattr(queued_batch, "is_idle", None)
            if type(queued_is_idle) is not bool:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued batch is_idle "
                    f"must be an exact bool, got {queued_is_idle!r}"
                )
            queued_stage_idx = getattr(queued_batch, "afd_stage_idx", None)
            if (
                type(queued_stage_idx) is not int
                or queued_stage_idx != afd_stage_idx
            ):
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued batch stage "
                    f"mismatch: room={afd_stage_idx}, batch={queued_stage_idx!r}"
                )
            queued_replica_id = getattr(
                queued_batch,
                "decode_attn_original_replica_id",
                None,
            )
            queued_replica_local_id = getattr(
                queued_batch,
                "decode_attn_original_replica_local_id",
                None,
            )
            if (
                type(queued_replica_id) is not int
                or (
                    queued_replica_local_id is not None
                    and type(queued_replica_local_id) is not int
                )
                or (queued_replica_id, queued_replica_local_id) != queue_lane
            ):
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room queued batch lane "
                    f"mismatch: queue={queue_lane}, batch="
                    f"{(queued_replica_id, queued_replica_local_id)}"
                )

            batch_validator(
                batch=queued_batch,
                lane=queue_lane,
                layer_id=layer_id,
                afd_stage_idx=afd_stage_idx,
                model_is_moe=model_is_moe,
                context="waiting-room queued batch",
                allow_idle=True,
            )

            cohort_id = getattr(queued_batch, "decode_attn_cohort_id", None)
            if cohort_id is not None and (
                type(cohort_id) is not int or cohort_id < 0
            ):
                raise RuntimeError(
                    "DECODE_ATTN A-to-F queued batch cohort ID must be None "
                    f"or an exact non-negative int, got {cohort_id!r}"
                )

            requests = getattr(queued_batch, "requests", None)
            if type(requests) is not list:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F queued batch requests must be an "
                    f"exact list, got {type(requests).__name__}"
                )
            if queued_is_idle:
                if requests:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued idle batch must not "
                        "contain requests"
                    )
                continue

            active_requests = []
            for request in requests:
                completed = getattr(request, "completed", None)
                if type(completed) is not bool:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued request completed state "
                        f"must be an exact bool, got {completed!r}"
                    )
                request_layer_id = getattr(
                    request,
                    "completed_layer_count",
                    None,
                )
                if type(request_layer_id) is not int or request_layer_id < 0:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued request layer must be an "
                        f"exact non-negative int, got {request_layer_id!r}"
                    )
                if not completed:
                    active_requests.append((request, request_layer_id))
            if not active_requests:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F queued non-idle batch has no active "
                    "requests"
                )
            for request, request_layer_id in active_requests:
                if request_layer_id != layer_id:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued request "
                        f"layer mismatch: room={layer_id}, request="
                        f"{request_layer_id}, request_id={request.id}"
                    )

    return room_lanes
