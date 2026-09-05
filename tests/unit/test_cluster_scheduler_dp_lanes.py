from types import SimpleNamespace

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import (
    LORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.random_cluster_scheduler import (
    RandomClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.sticky_round_robin_cluster_scheduler import (
    StickyRoundRobinClusterScheduler,
)
from frontier.types import ClusterType


def _request(*, session_id: int | None = None) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=4,
        num_decode_tokens=2,
        session_id=session_id,
    )


def _lane_scheduler(scheduler_type, requests):
    scheduler = scheduler_type.__new__(scheduler_type)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._num_replicas = 1
    scheduler._replica_dp_size = 2
    scheduler._cluster = SimpleNamespace(replicas={7: object()})
    scheduler._request_queue = list(requests)
    scheduler._replica_schedulers = {
        (7, 0): SimpleNamespace(num_pending_requests=0),
        (7, 1): SimpleNamespace(num_pending_requests=0),
    }
    return scheduler


def test_lor_assigns_requests_to_replica_local_dp_lanes() -> None:
    scheduler = _lane_scheduler(LORClusterScheduler, [_request(), _request()])

    mapping = scheduler._schedule_lor()

    assert [(replica_id, dp_id) for replica_id, dp_id, _ in mapping] == [
        (7, 0),
        (7, 1),
    ]


def test_random_assigns_requests_to_replica_local_dp_lanes(monkeypatch) -> None:
    scheduler = _lane_scheduler(RandomClusterScheduler, [_request(), _request()])
    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.random_cluster_scheduler.randint",
        lambda _low, _high: 0,
    )

    mapping = scheduler._schedule_random()

    assert [(replica_id, dp_id) for replica_id, dp_id, _ in mapping] == [
        (7, 0),
        (7, 1),
    ]


def test_sticky_round_robin_orders_all_attention_dp_lanes() -> None:
    scheduler = _lane_scheduler(
        StickyRoundRobinClusterScheduler,
        [],
    )

    assert scheduler._get_ordered_targets() == [(7, 0), (7, 1)]
