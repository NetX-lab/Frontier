from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from frontier.scheduler.utils.pdaf_a2f import prepare_a2f_admission


def _validate_room(*, group_key, room, expected_lane_contract):
    assert group_key == (3, 1)
    assert tuple(expected_lane_contract) == ((0, None), (1, None))
    assert set(room["per_lane_queues"]) <= set(expected_lane_contract)


def _build_idle_entries(**kwargs):
    return []


def _transfer_info(batch):
    return (128, 0.25)


def test_prepare_a2f_admission_returns_release_plan_without_mutating_input():
    lane_zero = (0, None)
    lane_one = (1, None)
    first_batch = SimpleNamespace(is_idle=False, id=10)
    second_batch = SimpleNamespace(is_idle=False, id=11)
    existing = defaultdict(deque, {lane_zero: deque([(3, first_batch)])})

    plan = prepare_a2f_admission(
        existing_queues=existing,
        expected_lanes=(lane_zero, lane_one),
        idle_lanes=set(),
        incoming_lane=lane_one,
        incoming_layer_id=3,
        incoming_batch=second_batch,
        is_moe=True,
        time=1.0,
        group_key=(3, 1),
        validate_room=_validate_room,
        build_idle_entries=_build_idle_entries,
        get_transfer_info=_transfer_info,
        validate_transfer_result=lambda value: value,
    )

    assert plan.barrier_ready is True
    assert plan.ready_lane_count == 2
    assert tuple(item[2].id for item in plan.picked) == (10, 11)
    assert tuple(item[2].id for item in plan.transfer_descriptors) == (10, 11)
    assert list(existing[lane_zero]) == [(3, first_batch)]
    assert lane_one not in existing


def test_prepare_a2f_admission_fails_before_mutating_input_on_transfer_error():
    lane_zero = (0, None)
    lane_one = (1, None)
    first_batch = SimpleNamespace(is_idle=False, id=10)
    second_batch = SimpleNamespace(is_idle=False, id=11)
    existing = defaultdict(deque, {
        lane_zero: deque([(3, first_batch)]),
        lane_one: deque([(3, second_batch)]),
    })

    def fail_transfer(_batch):
        raise ValueError("transfer predictor failure")

    with pytest.raises(ValueError, match="transfer predictor failure"):
        prepare_a2f_admission(
            existing_queues=existing,
            expected_lanes=(lane_zero, lane_one),
            idle_lanes=set(),
            incoming_lane=lane_one,
            incoming_layer_id=3,
            incoming_batch=second_batch,
            is_moe=True,
            time=1.0,
            group_key=(3, 1),
            validate_room=_validate_room,
            build_idle_entries=_build_idle_entries,
            get_transfer_info=fail_transfer,
            validate_transfer_result=lambda value: value,
        )

    assert list(existing[lane_zero]) == [(3, first_batch)]
    assert list(existing[lane_one]) == [(3, second_batch)]
