"""Pure planning helpers for PD-AF attention-to-FFN admission."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from frontier.scheduler.utils.pdaf_release import prepare_a2f_release_plan


@dataclass(frozen=True)
class A2FAdmissionPlan:
    """All A-to-F decisions that can be prepared before scheduler mutation."""

    queues: Mapping[Any, deque]
    expected_lanes: tuple[tuple[int, int | None], ...]
    prepared_idle_lanes: tuple[tuple[int, int | None], ...]
    prepared_idle_entries: tuple[tuple[tuple[int, int | None], tuple[int, Any]], ...]
    barrier_ready: bool
    ready_lane_count: int
    picked: tuple[tuple[tuple[int, int | None], int, Any], ...]
    queues_after_release: Mapping[Any, deque]
    non_idle_lanes: tuple[tuple[int, int | None], ...]
    transfer_descriptors: tuple[
        tuple[tuple[int, int | None], int, Any, int, float], ...
    ]


def prepare_a2f_admission(
    *,
    existing_queues: Mapping[Any, Sequence[tuple[int, Any]]],
    expected_lanes: Sequence[tuple[int, int | None]],
    idle_lanes: set[tuple[int, int | None]],
    incoming_lane: tuple[int, int | None],
    incoming_layer_id: int,
    incoming_batch: Any,
    is_moe: bool,
    time: float,
    group_key: Any,
    validate_room: Callable[..., Any],
    build_idle_entries: Callable[..., Sequence[tuple[tuple[int, int | None], tuple[int, Any]]]],
    get_transfer_info: Callable[[Any], Any] | None = None,
    validate_transfer_result: Callable[[Any], tuple[int, float]] | None = None,
) -> A2FAdmissionPlan:
    """Prepare one A-to-F admission without mutating scheduler-owned state."""

    normalized_expected_lanes = tuple(sorted(expected_lanes))
    prospective_queues = defaultdict(deque)
    for lane, queue in existing_queues.items():
        prospective_queues[lane].extend(queue)
    prospective_queues[incoming_lane].append((incoming_layer_id, incoming_batch))

    room = {
        "per_lane_queues": prospective_queues,
        "expected_lane_contract": normalized_expected_lanes,
    }
    validate_room(
        group_key=group_key,
        room=room,
        expected_lane_contract=normalized_expected_lanes,
    )

    first_release = prepare_a2f_release_plan(
        prospective_queues,
        normalized_expected_lanes,
        set(idle_lanes),
    )
    prepared_idle_lanes = tuple(first_release.idle_lanes)
    prepared_idle_entries = tuple(
        build_idle_entries(
            time=time,
            group_key=group_key,
            idle_lanes=prepared_idle_lanes,
            is_moe=is_moe,
        )
    )
    for lane, entry in prepared_idle_entries:
        prospective_queues[lane].append(entry)
    validate_room(
        group_key=group_key,
        room=room,
        expected_lane_contract=normalized_expected_lanes,
    )

    final_release = prepare_a2f_release_plan(
        prospective_queues,
        normalized_expected_lanes,
        set(),
    )
    picked = tuple(final_release.picked)
    non_idle_lanes = tuple(
        lane for lane, _, batch in picked if not bool(getattr(batch, "is_idle", False))
    )

    transfer_descriptors = []
    if first_release.barrier_ready:
        if get_transfer_info is None or validate_transfer_result is None:
            raise ValueError("A-to-F transfer callbacks are required when the barrier is ready")
        for lane in normalized_expected_lanes:
            lane_queue = prospective_queues.get(lane)
            if not lane_queue:
                continue
            ready_layer_id, ready_batch = lane_queue[0]
            if bool(getattr(ready_batch, "is_idle", False)):
                continue
            activation_size, transfer_time = validate_transfer_result(
                get_transfer_info(ready_batch)
            )
            transfer_descriptors.append(
                (lane, ready_layer_id, ready_batch, activation_size, transfer_time)
            )

    return A2FAdmissionPlan(
        queues=prospective_queues,
        expected_lanes=normalized_expected_lanes,
        prepared_idle_lanes=prepared_idle_lanes,
        prepared_idle_entries=prepared_idle_entries,
        barrier_ready=first_release.barrier_ready,
        ready_lane_count=first_release.ready_lane_count,
        picked=picked,
        queues_after_release=final_release.queues_after_release,
        non_idle_lanes=non_idle_lanes,
        transfer_descriptors=tuple(transfer_descriptors),
    )
