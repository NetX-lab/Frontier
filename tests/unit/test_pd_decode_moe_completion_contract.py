from types import SimpleNamespace

import pytest

from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.types import ClusterType


class _DecodeMoeRequest:
    def __init__(self, completed_layer_count: int = 0) -> None:
        self.id = 0
        self.completed = False
        self.completed_layer_count = completed_layer_count


class _DecodeMoeBatch:
    id = 31
    schedule_epoch = 0
    is_idle = False
    request_execution_signatures = [(0, 8, 1)]
    request_mutation_signatures = [(0, 8, 1, 0)]
    thinking_round_start_times = [None]

    def __init__(self, completed_layer_count: int = 0) -> None:
        self.requests = [_DecodeMoeRequest(completed_layer_count)]
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


@pytest.mark.parametrize(
    "dp_size,ep_size,completed_layer_count,expected_event_cls,expected_batch_end_calls",
    [
        (1, 1, 0, GlobalBatchEndEvent, 0),
        (2, 1, 0, ReplicaScheduleEvent, 1),
        (1, 2, 0, ReplicaScheduleEvent, 1),
        (2, 2, 0, ReplicaScheduleEvent, 1),
        (2, 1, 7, GlobalBatchEndEvent, 0),
        (1, 2, 7, GlobalBatchEndEvent, 0),
        (2, 2, 7, GlobalBatchEndEvent, 0),
    ],
)
def test_moe_decode_completion_path_is_exclusive_across_dp_ep_configs(
    dp_size,
    ep_size,
    completed_layer_count,
    expected_event_cls,
    expected_batch_end_calls,
) -> None:
    batch = _DecodeMoeBatch(completed_layer_count=completed_layer_count)
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
    assert isinstance(next_events[0], expected_event_cls)
    assert len(metrics_store.batch_end_calls) == expected_batch_end_calls
    assert batch.cluster_stage_end_calls == [(2.0, ClusterType.DECODE)]
    assert replica_scheduler.cluster_stage_end_batches == [batch.id]
