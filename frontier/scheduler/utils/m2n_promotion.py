"""Decode-FFN M2N waiting-room promotion."""

from collections import deque
from typing import Any, List, Optional

from frontier.scheduler.utils.m2n_grouping import prepare_ffn_group_promotion
from frontier.scheduler.utils.pdaf_transfer import LaneIdentityScope


def promote_decode_ffn_group(
    scheduler: Any,
    time: float,
    group_key,
    room: dict,
    logger,
    *,
    allow_idle_injection: bool,
    expected_lanes: int | None = None,
    expected_lane_ids: Optional[List[tuple[int, int]]] = None,
) -> bool:
    """Promote one complete M2N group and commit its waiting-room mutation."""
    if type(allow_idle_injection) is not bool:
        raise ValueError(
            "DECODE_FFN allow_idle_injection must be an exact bool, "
            f"got {allow_idle_injection!r}"
        )
    if expected_lanes is None:
        expected_lanes = getattr(scheduler, "_ffn_group_micro_batches", None)
    if type(expected_lanes) is not int or expected_lanes <= 0:
        raise ValueError(
            "DECODE_FFN expected_lanes must be an exact positive int, "
            f"got {expected_lanes!r}"
        )
    waiting_rooms = getattr(scheduler, "_m2n_waiting_by_layer", None)
    if type(waiting_rooms) is not dict:
        raise RuntimeError("DECODE_FFN _m2n_waiting_by_layer must be an exact dict")
    if waiting_rooms.get(group_key) is not room:
        raise RuntimeError(
            "DECODE_FFN promotion room is not the registered waiting-room "
            f"owner for group_key={group_key!r}"
        )
    ready_groups = getattr(scheduler, "_m2n_ready_groups", None)
    if type(ready_groups) is not deque:
        raise RuntimeError("DECODE_FFN _m2n_ready_groups must be an exact deque")

    idle_lanes = getattr(scheduler, "_ffn_idle_lanes", set())
    if type(idle_lanes) is not set:
        raise RuntimeError("DECODE_FFN _ffn_idle_lanes must be an exact set")
    plan = prepare_ffn_group_promotion(
        group_key=group_key,
        room=room,
        expected_lanes=expected_lanes,
        expected_lane_ids=expected_lane_ids,
        allow_idle_injection=allow_idle_injection,
        idle_lanes=idle_lanes,
        validate_room=scheduler._validate_decode_ffn_waiting_room,
        normalize_lanes=lambda raw_lanes, **kwargs: scheduler._normalize_m2n_lanes(
            raw_lanes,
            identity_scope=LaneIdentityScope.FULL_STAGE,
            field_name=kwargs["field_name"],
            require_nonempty=kwargs["require_nonempty"],
        ),
    )
    if plan is None:
        return False

    padding_plan, padding_summary = scheduler._prepare_dp_padding_on_promotion(
        list(plan.picked_before_idle_injection)
    )
    idle_lanes_to_inject = list(plan.idle_lanes_to_inject)
    if idle_lanes_to_inject:
        injected_lanes = scheduler._inject_ffn_idle_lanes_for_barrier(
            time,
            group_key,
            room,
            logger,
            expected_lane_ids=idle_lanes_to_inject,
        )
        if injected_lanes != idle_lanes_to_inject:
            raise RuntimeError(
                "DECODE_FFN idle lane injection did not match its prepared "
                f"plan: prepared={idle_lanes_to_inject}, actual={injected_lanes}"
            )

    lanes = list(room["lanes_rr_order"])
    picked = [room["per_lane_queues"][lane][0] for lane in lanes]
    if len(picked) != expected_lanes:
        raise RuntimeError(
            "DECODE_FFN promotion head count changed after preparation: "
            f"picked={len(picked)}, expected={expected_lanes}"
        )
    for batch, padded_metadata in padding_plan:
        batch.afd_stage_metadata = padded_metadata
    for lane in lanes:
        room["per_lane_queues"][lane].popleft()

    ready = [(batch, info) for batch, info in picked if not batch.is_idle]
    if ready:
        ready_groups.append(ready)
    room["lanes_rr_order"] = deque(
        [lane for lane in room["lanes_rr_order"] if room["per_lane_queues"][lane]]
    )
    if not room["lanes_rr_order"]:
        waiting_rooms.pop(group_key, None)

    if padding_summary is not None:
        padded_lane_count, dp_stage_max_tokens = padding_summary
        logger.info(
            f"[FFN-DP-PADDING] Applied DP padding across {padded_lane_count} "
            f"lanes: dp_stage_max_tokens={dp_stage_max_tokens} "
            f"padded_total={sum(dp_stage_max_tokens)}"
        )
    return True
