"""Pure planning for DECODE_FFN M2N grouping promotion."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class FFNPromotionPlan:
    """Validated lane and queue-head decisions before room mutation."""

    room_lanes: tuple[tuple[int, int | None], ...]
    lanes: tuple[tuple[int, int | None], ...]
    idle_lanes_to_inject: tuple[tuple[int, int | None], ...]
    picked_before_idle_injection: tuple[tuple[Any, Any], ...]


def prepare_ffn_group_promotion(
    *,
    group_key: Any,
    room: dict,
    expected_lanes: int,
    expected_lane_ids: Optional[Sequence[tuple[int, int | None]]],
    allow_idle_injection: bool,
    idle_lanes: set[tuple[int, int | None]],
    validate_room: Callable[..., Sequence[tuple[int, int | None]]],
    normalize_lanes: Callable[..., Sequence[tuple[int, int | None]]],
) -> FFNPromotionPlan | None:
    """Prepare a promotion or return ``None`` while the barrier is incomplete."""

    room_lanes = tuple(validate_room(group_key=group_key, room=room))
    normalized_expected_ids = None
    if expected_lane_ids is not None:
        normalized_expected_ids = tuple(
            sorted(
                normalize_lanes(
                    expected_lane_ids,
                    identity_scope=None,
                    field_name="DECODE_FFN promotion expected lane IDs",
                    require_nonempty=True,
                )
            )
        )
        if normalized_expected_ids != room_lanes:
            raise ValueError(
                "DECODE_FFN promotion expected lane IDs do not match the "
                f"waiting-room contract: expected={normalized_expected_ids}, room={room_lanes}"
            )
    if expected_lanes > len(room_lanes):
        raise ValueError(
            "DECODE_FFN expected lane count exceeds the waiting-room lane "
            f"contract: expected={expected_lanes}, contract={room_lanes}"
        )

    lanes = tuple(room["lanes_rr_order"])
    if len(lanes) > expected_lanes:
        raise ValueError(
            "DECODE_FFN grouping lanes exceed expected count: "
            f"lanes={len(lanes)} expected={expected_lanes}"
        )

    idle_to_inject = []
    if len(lanes) < expected_lanes and allow_idle_injection:
        normalized_idle = set(
            normalize_lanes(
                tuple(idle_lanes),
                identity_scope=None,
                field_name="DECODE_FFN idle lane inventory",
                require_nonempty=False,
            )
        )
        if not normalized_idle.issubset(set(room_lanes)):
            raise RuntimeError(
                "DECODE_FFN idle lane inventory is outside the waiting-room "
                f"contract: idle={sorted(normalized_idle)}, contract={room_lanes}"
            )
        candidate_order = normalized_expected_ids or room_lanes
        required = expected_lanes - len(lanes)
        idle_to_inject = [
            lane for lane in candidate_order
            if lane in normalized_idle and not room["per_lane_queues"].get(lane)
        ][:required]

    prospective_lanes = tuple(lanes) + tuple(idle_to_inject)
    if len(prospective_lanes) < expected_lanes:
        return None
    if len(prospective_lanes) > expected_lanes:
        raise RuntimeError(
            "DECODE_FFN prospective promotion lanes exceed the expected "
            f"count: lanes={prospective_lanes}, expected={expected_lanes}"
        )
    picked = tuple(room["per_lane_queues"][lane][0] for lane in lanes)
    return FFNPromotionPlan(
        room_lanes=room_lanes,
        lanes=lanes,
        idle_lanes_to_inject=tuple(idle_to_inject),
        picked_before_idle_injection=picked,
    )
