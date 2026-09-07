"""DECODE_FFN idle-lane injection for incomplete M2N barriers."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from numbers import Real
from typing import Any, List, Optional

from frontier.scheduler.utils.pdaf_entries import build_decode_ffn_idle_entries
from frontier.scheduler.utils.pdaf_transfer import LaneIdentityScope
from frontier.types import ClusterType


def inject_ffn_idle_lanes(
    scheduler: Any,
    time: float,
    group_key,
    room: dict,
    logger: Any,
    *,
    expected_lane_ids: Optional[List[tuple[int, int]]] = None,
) -> List[tuple[int, int]]:
    """Inject validated idle sentinel batches into one M2N waiting room."""

    if scheduler._cluster_type is not ClusterType.DECODE_FFN:
        raise ValueError("FFN idle lane injection requires a DECODE_FFN scheduler")
    if (
        not isinstance(time, Real)
        or isinstance(time, bool)
        or not math.isfinite(float(time))
    ):
        raise ValueError(
            "DECODE_FFN idle injection time must be a finite int or float, "
            f"got {time!r}"
        )
    time = float(time)
    room_lanes = scheduler._validate_decode_ffn_waiting_room(
        group_key=group_key,
        room=room,
    )
    idle_lanes = getattr(scheduler, "_ffn_idle_lanes", None)
    if type(idle_lanes) is not set:
        raise RuntimeError("DECODE_FFN _ffn_idle_lanes must be an exact set")
    if not idle_lanes:
        return []
    normalized_idle_lanes = set(
        scheduler._normalize_m2n_lanes(
            tuple(idle_lanes),
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN idle lane inventory",
            require_nonempty=False,
        )
    )
    if not normalized_idle_lanes.issubset(set(room_lanes)):
        raise RuntimeError(
            "DECODE_FFN idle lane inventory is outside the waiting-room "
            f"contract: idle={sorted(normalized_idle_lanes)}, contract={room_lanes}"
        )

    if expected_lane_ids is not None:
        normalized_expected = scheduler._normalize_m2n_lanes(
            expected_lane_ids,
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name="DECODE_FFN idle injection candidate lanes",
            require_nonempty=False,
        )
        if not set(normalized_expected).issubset(set(room_lanes)):
            raise ValueError(
                "DECODE_FFN idle injection candidate lane is outside the "
                f"waiting-room contract: candidates={normalized_expected}, "
                f"contract={room_lanes}"
            )
        candidate_lanes = [lane for lane in normalized_expected if lane in normalized_idle_lanes]
    else:
        candidate_lanes = [lane for lane in room_lanes if lane in normalized_idle_lanes]

    candidate_lanes = [
        lane for lane in candidate_lanes if not room["per_lane_queues"].get(lane)
    ]
    afd_stage_idx = group_key[1]
    barrier_round_id = group_key[2] if len(group_key) >= 3 else None
    wire_layer_id = group_key[0]
    replica_config = getattr(scheduler._config, "replica_config", None)
    if replica_config is None:
        raise RuntimeError("DECODE_FFN idle injection requires replica_config")
    model_config = getattr(replica_config, "model_config", None)
    if model_config is None:
        raise RuntimeError("DECODE_FFN idle injection requires model_config")
    is_moe = getattr(model_config, "is_moe", None)
    if type(is_moe) is not bool:
        raise RuntimeError(
            "DECODE_FFN idle injection model_config.is_moe must be an exact bool, "
            f"got {is_moe!r}"
        )
    prepared_entries = build_decode_ffn_idle_entries(
        time=time,
        lanes=candidate_lanes,
        layer_id=wire_layer_id,
        afd_stage_idx=afd_stage_idx,
        barrier_round_id=barrier_round_id,
        expected_lanes=room_lanes,
        is_moe=is_moe,
    )
    if not prepared_entries:
        return []

    prospective_room = {
        "per_lane_queues": defaultdict(
            deque,
            {
            lane: deque(queue)
            for lane, queue in room["per_lane_queues"].items()
            },
        ),
        "lanes_rr_order": deque(room["lanes_rr_order"]),
        "rr_cursor": room["rr_cursor"],
        "expected_lane_contract": room_lanes,
    }
    for missing_lane, idle_entry in prepared_entries:
        prospective_room["per_lane_queues"][missing_lane].append(idle_entry)
        prospective_room["lanes_rr_order"].append(missing_lane)
    scheduler._validate_decode_ffn_waiting_room(
        group_key=group_key,
        room=prospective_room,
    )

    idle_created: List[tuple[int, int]] = []
    for missing_lane, idle_entry in prepared_entries:
        room["per_lane_queues"][missing_lane].append(idle_entry)
        room["lanes_rr_order"].append(missing_lane)
        idle_created.append(missing_lane)
    if idle_created:
        logger.info(
            f"[FFN-M2N-IDLE] Injected idle lanes for barrier: "
            f"afd_stage_idx={afd_stage_idx} wire_layer={wire_layer_id} "
            f"barrier_round_id={barrier_round_id} missing={sorted(idle_created)}"
        )
    return idle_created
