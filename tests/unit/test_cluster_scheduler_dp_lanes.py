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
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.metrics.metrics_store import MetricsStore
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.types import ClusterType


class _BareClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        return []


class _BareReplicaScheduler(BaseReplicaScheduler):
    def _get_next_batch(self, *args, **kwargs):
        return None

    def on_batch_end(self, *args, **kwargs):
        return None


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


def test_shared_batch_global_ids_are_unique_per_replica_dp_lane() -> None:
    scheduler = _BareClusterScheduler.__new__(_BareClusterScheduler)
    scheduler._replica_dp_size = 2

    assert scheduler.make_attention_dp_batch_global_id(7, 0, 0) == 0
    assert scheduler.make_attention_dp_batch_global_id(7, 1, 0) == 1
    assert scheduler.make_attention_dp_batch_global_id(7, 0, 1) == 2


def test_decode_sync_global_ids_use_attention_dp_cardinality() -> None:
    scheduler = _BareClusterScheduler.__new__(_BareClusterScheduler)
    scheduler._replica_dp_size = 8
    scheduler._replica_ep_size = 4

    assert scheduler.make_decode_sync_global_id(7, 7, 0) == 7


def test_replica_child_batch_creation_uses_lane_scoped_global_ids() -> None:
    cluster_scheduler = _BareClusterScheduler.__new__(_BareClusterScheduler)
    cluster_scheduler._replica_dp_size = 2
    children = []
    for dp_id in (0, 1):
        child = _BareReplicaScheduler.__new__(_BareReplicaScheduler)
        child._cluster_scheduler = cluster_scheduler
        child._cluster_type = ClusterType.MONOLITHIC
        child._replica_id = 7
        child._replica_local_id = dp_id
        child._replica_is_moe = True
        child._batch_creation_counter = 0
        child._decode_sync_batch_creation_counter = 0
        children.append(child)

    batches = [child._create_batch([_request()], [6]) for child in children]

    assert [batch.global_id for batch in batches] == [0, 1]
    assert [batch._forward_cohort_id for batch in batches] == [0, 0]


def test_metrics_distinguish_attention_dp_and_ep_lane_scopes() -> None:
    assert MetricsStore._get_frontier_stage_execution_scope(
        ClusterType.MONOLITHIC,
        1,
    ) == "ATTN_DP_LANE"
    assert MetricsStore._get_frontier_stage_execution_scope(
        ClusterType.DECODE_FFN,
        1,
    ) == "EP_WAVE_LANE"
    assert MetricsStore._get_frontier_stage_execution_scope(
        ClusterType.DECODE_ATTN,
        None,
    ) == "FULL_STAGE_WORLD"


def test_stage_release_wakes_only_queued_sibling_lanes() -> None:
    scheduler = _BareClusterScheduler.__new__(_BareClusterScheduler)
    scheduler._cluster_type = ClusterType.MONOLITHIC

    def lane(*, busy: bool, empty: bool):
        stage = SimpleNamespace(
            is_busy=busy,
            is_empty=lambda: empty,
        )
        return SimpleNamespace(
            get_replica_stage_scheduler=lambda _stage_id: stage,
        )

    scheduler._replica_schedulers = {
        (7, 0): lane(busy=False, empty=False),  # current owner: excluded
        (7, 1): lane(busy=False, empty=False),  # queued sibling: wakes
        (7, 2): lane(busy=True, empty=False),   # busy sibling: skip
        (7, 3): lane(busy=False, empty=True),   # empty sibling: skip
        (8, 0): lane(busy=False, empty=False),  # different Replica: skip
    }

    events = scheduler.get_waiting_replica_stage_schedule_events(
        time=2.5,
        replica_id=7,
        stage_id=4,
        exclude_replica_local_id=0,
    )

    assert len(events) == 1
    assert isinstance(events[0], ReplicaStageScheduleEvent)
    assert events[0].time == 2.5
    assert events[0]._replica_id == 7
    assert events[0]._stage_id == 4
    assert events[0]._replica_local_id == 1
