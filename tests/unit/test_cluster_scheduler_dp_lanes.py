from types import SimpleNamespace

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import (
    LORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
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


def _round_robin_scheduler(
    *, replica_ids: list[int], dp_size: int
) -> RoundRobinClusterScheduler:
    """A round-robin scheduler over the given replicas, with an empty queue."""

    scheduler = RoundRobinClusterScheduler.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._num_replicas = len(replica_ids)
    scheduler._replica_dp_size = dp_size
    scheduler._cluster = SimpleNamespace(
        replicas={replica_id: object() for replica_id in replica_ids}
    )
    scheduler._request_queue = []
    scheduler._request_counter = 0
    return scheduler


def _placements_for_call_sizes(
    *, replica_ids: list[int], dp_size: int, call_sizes: list[int]
) -> list[tuple[int, int]]:
    """Schedule one request stream in the given batches.

    Returns the (replica id, DP lane) of each request by its position in the
    stream. The stream is the same for every call partitioning, so the result
    may not depend on `call_sizes`. Comparing by stream position rather than by
    request id is what makes two separately constructed runs comparable.
    """

    scheduler = _round_robin_scheduler(replica_ids=replica_ids, dp_size=dp_size)
    total = sum(call_sizes)
    requests = [_request() for _ in range(total)]
    position_of = {request.id: index for index, request in enumerate(requests)}
    placements: dict[int, tuple[int, int]] = {}

    offset = 0
    for size in call_sizes:
        scheduler._request_queue = requests[offset:offset + size]
        offset += size
        for replica_id, dp_id, request in scheduler._schedule_batch_mode():
            position = position_of[request.id]
            assert position not in placements, "a request was scheduled twice"
            placements[position] = (replica_id, dp_id)

    assert len(placements) == total, "every request is scheduled exactly once"
    return [placements[position] for position in range(total)]


def test_round_robin_dp_lane_does_not_depend_on_call_partitioning() -> None:
    """The defect this covers: the DP lane restarted at zero on every call.

    With one replica and two lanes, scheduling eight requests one at a time put
    every request on lane 0, while scheduling them in one call alternated. The
    lane must follow the request's position in the stream, not its position
    within the call that happened to carry it.
    """

    one_at_a_time = _placements_for_call_sizes(
        replica_ids=[7], dp_size=2, call_sizes=[1] * 8
    )
    single_burst = _placements_for_call_sizes(
        replica_ids=[7], dp_size=2, call_sizes=[8]
    )
    uneven = _placements_for_call_sizes(
        replica_ids=[7], dp_size=2, call_sizes=[3, 1, 4]
    )

    assert one_at_a_time == single_burst == uneven
    assert one_at_a_time == [(7, 0), (7, 1)] * 4


def test_round_robin_placement_is_stable_across_topologies() -> None:
    """Replica ids need not be contiguous and lanes may outnumber two."""

    cases = [
        ([7], 1),
        ([7], 4),
        ([3, 11], 1),
        ([3, 11], 2),
        ([3, 11, 42], 3),
    ]
    for replica_ids, dp_size in cases:
        burst = _placements_for_call_sizes(
            replica_ids=replica_ids, dp_size=dp_size, call_sizes=[12]
        )
        incremental = _placements_for_call_sizes(
            replica_ids=replica_ids, dp_size=dp_size, call_sizes=[1] * 12
        )
        assert burst == incremental, (replica_ids, dp_size)
        assert {replica_id for replica_id, _ in burst} <= set(replica_ids)
        assert all(0 <= dp_id < dp_size for _, dp_id in burst)


def test_round_robin_survives_an_empty_scheduling_call() -> None:
    """An empty call must neither advance the rotation nor reset it."""

    with_gap = _placements_for_call_sizes(
        replica_ids=[3, 11], dp_size=2, call_sizes=[2, 0, 2, 0, 4]
    )
    without_gap = _placements_for_call_sizes(
        replica_ids=[3, 11], dp_size=2, call_sizes=[8]
    )
    assert with_gap == without_gap


def test_round_robin_returns_results_grouped_by_replica() -> None:
    """The return order groups each call's results per replica, as before."""

    scheduler = _round_robin_scheduler(replica_ids=[3, 11], dp_size=2)
    scheduler._request_queue = [_request() for _ in range(6)]

    mapping = scheduler._schedule_batch_mode()

    replica_order = [replica_id for replica_id, _, _ in mapping]
    assert replica_order == [3, 3, 3, 11, 11, 11]
