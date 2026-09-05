"""Small, state-free helpers shared by PD-AF transfer handlers."""

from enum import Enum
from typing import List


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
