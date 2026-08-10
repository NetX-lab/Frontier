"""Contract tests for closed-loop (concurrency-capped) request generation."""
from __future__ import annotations

from collections import deque

from frontier.entities import Request
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.events.request_arrival_event import RequestArrivalEvent
from frontier.scheduler.global_scheduler.base_global_scheduler import (
    BaseGlobalScheduler,
)
from frontier.types import ClusterType


def test_backlog_state_is_empty_by_default() -> None:
    scheduler = object.__new__(BaseGlobalScheduler)
    scheduler._closed_loop_backlog = deque()
    scheduler._closed_loop_cluster_type = None

    assert scheduler.pop_next_closed_loop_request() is None
    assert scheduler.closed_loop_cluster_type is None


def test_configure_and_pop_backlog_in_order() -> None:
    scheduler = object.__new__(BaseGlobalScheduler)
    scheduler._closed_loop_backlog = deque()
    scheduler._closed_loop_cluster_type = None

    req_a = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)
    req_b = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)
    scheduler.configure_closed_loop_backlog([req_a, req_b], ClusterType.MONOLITHIC)

    assert scheduler.closed_loop_cluster_type == ClusterType.MONOLITHIC
    assert scheduler.pop_next_closed_loop_request() is req_a
    assert scheduler.pop_next_closed_loop_request() is req_b
    assert scheduler.pop_next_closed_loop_request() is None


class _StubRequest:
    """Minimal stand-in satisfying GlobalBatchEndEvent.handle_event's request contract."""

    def __init__(self, request_id: int, *, completed: bool) -> None:
        self.id = request_id
        self.completed = completed
        self.current_thinking_round_index = 0
        self.num_restarts = 0
        self.execution_epoch = 0
        self.current_decode_token_index = 0
        self.is_thinking_mode_enabled = False
        self.first_decode_token_completed_at = 0
        self.num_decode_tokens = 0


class _StubBatch:
    def __init__(self, requests) -> None:
        self.id = 1
        self.schedule_epoch = 0
        self.requests = requests
        self.on_batch_end_calls = 0

    def on_batch_end(self, *args, **kwargs) -> None:
        self.on_batch_end_calls += 1


class _StubReplicaScheduler:
    memory_usage_percent = 10.0

    def __init__(self) -> None:
        self.on_batch_end_calls = []

    def on_batch_end(self, batch) -> None:
        self.on_batch_end_calls.append(batch.id)


class _StubClusterScheduler:
    def __init__(self, replica_scheduler: _StubReplicaScheduler) -> None:
        self._replica_scheduler = replica_scheduler

    def get_dp_replica_scheduler(self, replica_id: int, dp_id: int):
        return self._replica_scheduler


class _StubGlobalScheduler:
    """Real backlog behavior (from BaseGlobalScheduler) glued to a stub cluster tree,
    so the release-on-completion wiring is exercised end to end."""

    def __init__(self, cluster_scheduler: _StubClusterScheduler) -> None:
        self._cluster_scheduler = cluster_scheduler
        self._closed_loop_backlog = deque()
        self._closed_loop_cluster_type = None

    def get_cluster_scheduler(self, cluster_type: ClusterType):
        return self._cluster_scheduler

    configure_closed_loop_backlog = BaseGlobalScheduler.configure_closed_loop_backlog
    pop_next_closed_loop_request = BaseGlobalScheduler.pop_next_closed_loop_request
    closed_loop_cluster_type = BaseGlobalScheduler.closed_loop_cluster_type


class _StubMetricsStore:
    def __init__(self) -> None:
        self.request_end_calls = []

    def _on_request_end(self, time, request) -> None:
        self.request_end_calls.append((time, request.id))


def _make_event(requests) -> GlobalBatchEndEvent:
    batch = _StubBatch(requests)
    return GlobalBatchEndEvent(
        time=5.0,
        replica_id=0,
        dp_id=0,
        batch=batch,
        cluster_type=ClusterType.MONOLITHIC,
        request_execution_signatures=[(0, 0, 0) for _ in requests],
        request_mutation_signatures=[(0, 0, 0, 0) for _ in requests],
        thinking_round_start_times=[None for _ in requests],
    )


def test_completion_releases_next_backlog_request_as_arrival_event() -> None:
    completed_request = _StubRequest(1, completed=True)
    backlog_request = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)

    replica_scheduler = _StubReplicaScheduler()
    cluster_scheduler = _StubClusterScheduler(replica_scheduler)
    scheduler = _StubGlobalScheduler(cluster_scheduler)
    scheduler.configure_closed_loop_backlog([backlog_request], ClusterType.MONOLITHIC)
    metrics_store = _StubMetricsStore()

    event = _make_event([completed_request])
    next_events = event.handle_event(scheduler, metrics_store)

    assert metrics_store.request_end_calls == [(5.0, 1)]
    arrival_events = [e for e in next_events if isinstance(e, RequestArrivalEvent)]
    assert len(arrival_events) == 1
    assert arrival_events[0]._request is backlog_request
    assert arrival_events[0]._cluster_type == ClusterType.MONOLITHIC
    assert arrival_events[0].time == 5.0
    assert backlog_request.arrived_at == 5.0
    assert scheduler.pop_next_closed_loop_request() is None  # backlog now empty
    assert any(isinstance(e, ReplicaScheduleEvent) for e in next_events)


def test_completion_with_empty_backlog_injects_no_arrival_event() -> None:
    completed_request = _StubRequest(1, completed=True)

    replica_scheduler = _StubReplicaScheduler()
    cluster_scheduler = _StubClusterScheduler(replica_scheduler)
    scheduler = _StubGlobalScheduler(cluster_scheduler)  # backlog left empty
    metrics_store = _StubMetricsStore()

    event = _make_event([completed_request])
    next_events = event.handle_event(scheduler, metrics_store)

    assert not any(isinstance(e, RequestArrivalEvent) for e in next_events)
    assert metrics_store.request_end_calls == [(5.0, 1)]


def test_incomplete_request_does_not_trigger_release() -> None:
    incomplete_request = _StubRequest(1, completed=False)
    backlog_request = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)

    replica_scheduler = _StubReplicaScheduler()
    cluster_scheduler = _StubClusterScheduler(replica_scheduler)
    scheduler = _StubGlobalScheduler(cluster_scheduler)
    scheduler.configure_closed_loop_backlog([backlog_request], ClusterType.MONOLITHIC)
    metrics_store = _StubMetricsStore()

    event = _make_event([incomplete_request])
    next_events = event.handle_event(scheduler, metrics_store)

    assert not any(isinstance(e, RequestArrivalEvent) for e in next_events)
    assert metrics_store.request_end_calls == []
    assert scheduler.pop_next_closed_loop_request() is backlog_request  # untouched
