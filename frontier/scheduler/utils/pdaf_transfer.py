"""Small, state-free helpers shared by PD-AF transfer handlers."""

from enum import Enum
from typing import Any, List, Optional

from collections import defaultdict, deque

from frontier.types import ClusterType
from frontier.entities import Batch


class LaneIdentityScope(Enum):
    """Identity shape accepted by one transfer lane contract."""

    FULL_STAGE = "full_stage"
    REPLICA_LOCAL = "replica_local"


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
