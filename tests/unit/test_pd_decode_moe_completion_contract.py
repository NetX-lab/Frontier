from types import SimpleNamespace

import pytest

from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.types import ClusterType


class _DecodeMoeRequest:
    def __init__(
        self,
        completed_layer_count: int = 0,
        *,
        request_id: int = 0,
        completed: bool = False,
    ) -> None:
        self.id = request_id
        self.completed = completed
        self.completed_layer_count = completed_layer_count


class _DecodeMoeBatch:
    id = 31
    schedule_epoch = 0
    is_idle = False
    request_execution_signatures = [(0, 8, 1)]
    request_mutation_signatures = [(0, 8, 1, 0)]
    thinking_round_start_times = [None]

    def __init__(
        self,
        completed_layer_count: int = 0,
        *,
        requests=None,
    ) -> None:
        self.requests = (
            [_DecodeMoeRequest(completed_layer_count)]
            if requests is None
            else list(requests)
        )
        self.cluster_stage_end_calls = []

    def on_cluster_stage_end(self, time: float, cluster_type: ClusterType) -> None:
        self.cluster_stage_end_calls.append((time, cluster_type))


class _DecodeMoeReplicaScheduler:
    memory_usage_percent = 25.0

    def __init__(self) -> None:
        self.cluster_stage_end_batches = []

    def on_cluster_stage_end(self, batch) -> None:
        self.cluster_stage_end_batches.append(batch.id)


class _DecodeMoeClusterScheduler:
    def __init__(self, replica_scheduler: _DecodeMoeReplicaScheduler) -> None:
        self._replica_scheduler = replica_scheduler
        self._cluster = SimpleNamespace(
            replicas={
                1: SimpleNamespace(
                    is_moe=True,
                    dp_size=1,
                    num_moe_expert_parallel_size=1,
                )
            }
        )
        self._config = SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=SimpleNamespace(num_layers=8)
            )
        )

    def get_dp_replica_scheduler(self, replica_id: int, dp_id: int):
        assert replica_id == 1
        assert dp_id == 0
        return self._replica_scheduler


class _DecodeMoeGlobalScheduler:
    def __init__(self, cluster_scheduler: _DecodeMoeClusterScheduler) -> None:
        self._cluster_scheduler = cluster_scheduler

    def get_cluster_scheduler(self, cluster_type: ClusterType):
        assert cluster_type == ClusterType.DECODE
        return self._cluster_scheduler


class _DecodeMoeMetricsStore:
    def __init__(self) -> None:
        self.batch_end_calls = []

    def on_batch_end(self, *args, **kwargs) -> None:
        self.batch_end_calls.append((args, kwargs))


def test_local_moe_decode_stage_emits_global_batch_end_after_all_layers() -> None:
    batch = _DecodeMoeBatch()
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()

    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    next_events = event.handle_event(scheduler, metrics_store)

    assert len(next_events) == 1
    assert isinstance(next_events[0], GlobalBatchEndEvent)
    assert batch.cluster_stage_end_calls == [(2.0, ClusterType.DECODE)]
    assert replica_scheduler.cluster_stage_end_batches == [batch.id]
    assert metrics_store.batch_end_calls == []


@pytest.mark.parametrize("dp_size,ep_size", [(2, 1), (1, 2), (2, 2)])
def test_distributed_moe_decode_completes_only_at_exact_total_layers(
    dp_size: int,
    ep_size: int,
) -> None:
    batch = _DecodeMoeBatch(completed_layer_count=8)
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].dp_size = dp_size
    cluster_scheduler._cluster.replicas[1].num_moe_expert_parallel_size = ep_size
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()

    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    next_events = event.handle_event(scheduler, metrics_store)

    assert len(next_events) == 1
    assert isinstance(next_events[0], GlobalBatchEndEvent)
    assert metrics_store.batch_end_calls == []
    assert batch.cluster_stage_end_calls == [(2.0, ClusterType.DECODE)]
    assert replica_scheduler.cluster_stage_end_batches == [batch.id]


@pytest.mark.parametrize(
    "completed_layer_count,error_match",
    [(7, "undercount"), (9, "overflow")],
)
def test_distributed_moe_decode_rejects_non_exact_terminal_layer_count(
    completed_layer_count: int,
    error_match: str,
) -> None:
    batch = _DecodeMoeBatch(completed_layer_count=completed_layer_count)
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].dp_size = 2
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()
    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    with pytest.raises(ValueError, match=error_match):
        event.handle_event(scheduler, metrics_store)

    assert metrics_store.batch_end_calls == []


def test_distributed_moe_decode_rejects_terminal_batch_without_active_requests() -> None:
    batch = _DecodeMoeBatch(
        requests=[
            _DecodeMoeRequest(
                completed_layer_count=8,
                completed=True,
            )
        ]
    )
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].dp_size = 2
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()
    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    with pytest.raises(ValueError, match="no active request"):
        event.handle_event(scheduler, metrics_store)


def test_distributed_moe_decode_rejects_inconsistent_active_layer_counts() -> None:
    batch = _DecodeMoeBatch(
        requests=[
            _DecodeMoeRequest(completed_layer_count=8, request_id=1),
            _DecodeMoeRequest(completed_layer_count=7, request_id=2),
        ]
    )
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].num_moe_expert_parallel_size = 2
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()
    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        event.handle_event(scheduler, metrics_store)


def test_distributed_moe_decode_ignores_completed_request_layer_count() -> None:
    batch = _DecodeMoeBatch(
        requests=[
            _DecodeMoeRequest(completed_layer_count=8, request_id=1),
            _DecodeMoeRequest(
                completed_layer_count=3,
                request_id=2,
                completed=True,
            ),
        ]
    )
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].dp_size = 2
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()
    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )

    next_events = event.handle_event(scheduler, metrics_store)

    assert len(next_events) == 1
    assert isinstance(next_events[0], GlobalBatchEndEvent)


def test_stale_distributed_moe_cluster_end_does_not_mutate_or_emit() -> None:
    batch = _DecodeMoeBatch(completed_layer_count=8)
    replica_scheduler = _DecodeMoeReplicaScheduler()
    cluster_scheduler = _DecodeMoeClusterScheduler(replica_scheduler)
    cluster_scheduler._cluster.replicas[1].dp_size = 2
    scheduler = _DecodeMoeGlobalScheduler(cluster_scheduler)
    metrics_store = _DecodeMoeMetricsStore()
    event = ClusterBatchEndEvent(
        time=2.0,
        replica_id=1,
        batch=batch,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )
    batch.schedule_epoch = 1

    assert event.handle_event(scheduler, metrics_store) == []
    assert batch.cluster_stage_end_calls == []
    assert replica_scheduler.cluster_stage_end_batches == []
    assert metrics_store.batch_end_calls == []
