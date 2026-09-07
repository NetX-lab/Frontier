from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock

from frontier.scheduler.utils.m2n_idle import inject_ffn_idle_lanes
from frontier.scheduler.utils.m2n_grouping import prepare_ffn_group_promotion
from frontier.types import ClusterType


def _normalize(raw_lanes, **_kwargs):
    return list(raw_lanes)


def _validate(*, group_key, room):
    assert group_key == (4, 2)
    return tuple(room["expected_lane_contract"])


def test_prepare_ffn_group_promotion_returns_none_until_all_lanes_arrive():
    lane_zero = (0, 0)
    lane_one = (1, 0)
    room = {
        "per_lane_queues": defaultdict(deque, {lane_zero: deque([("b", "i")])}),
        "lanes_rr_order": deque([lane_zero]),
        "rr_cursor": 0,
        "expected_lane_contract": (lane_zero, lane_one),
    }
    plan = prepare_ffn_group_promotion(
        group_key=(4, 2),
        room=room,
        expected_lanes=2,
        expected_lane_ids=None,
        allow_idle_injection=False,
        idle_lanes=set(),
        validate_room=_validate,
        normalize_lanes=_normalize,
    )
    assert plan is None
    assert list(room["lanes_rr_order"]) == [lane_zero]


def test_prepare_ffn_group_promotion_selects_only_allowed_idle_lane():
    lane_zero = (0, 0)
    lane_one = (1, 0)
    batch = SimpleNamespace(id=5)
    info = SimpleNamespace(id=6)
    room = {
        "per_lane_queues": defaultdict(deque, {lane_zero: deque([(batch, info)])}),
        "lanes_rr_order": deque([lane_zero]),
        "rr_cursor": 0,
        "expected_lane_contract": (lane_zero, lane_one),
    }
    plan = prepare_ffn_group_promotion(
        group_key=(4, 2),
        room=room,
        expected_lanes=2,
        expected_lane_ids=[lane_zero, lane_one],
        allow_idle_injection=True,
        idle_lanes={lane_one},
        validate_room=_validate,
        normalize_lanes=_normalize,
    )
    assert plan is not None
    assert plan.idle_lanes_to_inject == (lane_one,)
    assert plan.picked_before_idle_injection == ((batch, info),)
    assert list(room["lanes_rr_order"]) == [lane_zero]


def test_ffn_idle_injection_keeps_new_lane_queue_as_deque():
    lane_zero = (0, None)
    lane_one = (1, None)
    room = {
        "per_lane_queues": defaultdict(deque, {lane_zero: deque()}),
        "lanes_rr_order": deque([lane_zero]),
        "rr_cursor": 0,
        "expected_lane_contract": (lane_zero, lane_one),
    }

    class FakeScheduler:
        _cluster_type = ClusterType.DECODE_FFN
        _ffn_idle_lanes = {lane_one}
        _config = SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=SimpleNamespace(is_moe=True),
            )
        )

        @staticmethod
        def _normalize_m2n_lanes(lanes, **_kwargs):
            return list(lanes)

        @staticmethod
        def _validate_decode_ffn_waiting_room(*, group_key, room):
            assert group_key == (4, 2, 7)
            assert all(type(queue) is deque for queue in room["per_lane_queues"].values())
            return room["expected_lane_contract"]

    created = inject_ffn_idle_lanes(
        FakeScheduler(),
        1.25,
        (4, 2, 7),
        room,
        Mock(),
    )

    assert created == [lane_one]
    assert type(room["per_lane_queues"][lane_one]) is deque
