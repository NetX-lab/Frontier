"""Pure A-to-F barrier release planning helpers."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class A2FReleasePlan:
    idle_lanes: tuple[tuple[int, int | None], ...]
    barrier_ready: bool
    ready_lane_count: int
    picked: tuple[tuple[tuple[int, int | None], int, Any], ...]
    queues_after_release: Mapping[Any, deque]


def prepare_a2f_release_plan(
    per_lane_queues: Mapping[Any, Sequence[tuple[int, Any]]],
    expected_lanes: Sequence[tuple[int, int | None]],
    idle_lanes: set[tuple[int, int | None]],
) -> A2FReleasePlan:
    """Plan idle injection, barrier readiness, and one-entry release per lane."""

    prospective = defaultdict(deque)
    for lane, queue in per_lane_queues.items():
        prospective[lane].extend(queue)
    prepared_idle = tuple(
        lane
        for lane in expected_lanes
        if not prospective.get(lane) and lane in idle_lanes
    )
    barrier_ready = all(
        prospective.get(lane) or lane in prepared_idle for lane in expected_lanes
    )
    ready_count = sum(
        1
        for lane in expected_lanes
        if prospective.get(lane) or lane in prepared_idle
    )
    picked = []
    after_release = defaultdict(
        deque,
        {lane: deque(queue) for lane, queue in prospective.items()},
    )
    if barrier_ready:
        for lane in expected_lanes:
            if not after_release.get(lane):
                continue
            layer_id, batch = after_release[lane].popleft()
            picked.append((lane, layer_id, batch))
    return A2FReleasePlan(
        idle_lanes=prepared_idle,
        barrier_ready=barrier_ready,
        ready_lane_count=ready_count,
        picked=tuple(picked),
        queues_after_release=after_release,
    )
