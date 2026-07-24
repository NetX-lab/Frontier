#!/usr/bin/env python3
"""Regression tests for PD-AF DECODE_ATTN initial wave release."""

from __future__ import annotations

from types import SimpleNamespace

from frontier.events.cluster_schedule_event import ClusterScheduleEvent
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType


class _FakeReplicaScheduler:
    num_pending_requests = 0

    @property
    def memory_usage_percent(self) -> float:
        return 0.0

    def on_batch_end(self, _batch) -> None:
        return None


class _FakeGlobalScheduler:
    def __init__(self, cluster_scheduler: RoundRobinClusterScheduler) -> None:
        self._cluster_scheduler = cluster_scheduler

    def get_cluster_scheduler(
        self,
        _cluster_type: ClusterType,
    ) -> RoundRobinClusterScheduler:
        return self._cluster_scheduler


class _FakeMetricsStore:
    def on_batch_end(self, *_args, **_kwargs) -> None:
        return None

    def _on_request_end(self, *_args, **_kwargs) -> None:
        return None


def _build_request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=request_id, _arrived_at=float(request_id))


def _build_scheduler(*, threshold: int, num_requests: int) -> RoundRobinClusterScheduler:
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._request_counter = 0
    scheduler._num_replicas = 1
    scheduler._replica_dp_size = 1
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._request_queue = []
    scheduler._af_batch_queue = []
    scheduler._replica_dp_load_tracker = {(0, 0): 0}
    scheduler._dp_replica_schedulers = {(0, 0): _FakeReplicaScheduler()}
    scheduler._request_generator_config = SimpleNamespace(num_requests=num_requests)
    scheduler._decode_attn_expected_total_requests = num_requests
    scheduler._decode_attn_initial_allocation_done = False
    scheduler._decode_attn_initial_allocation_allocated_requests = 0
    scheduler._decode_attn_request_allocation_threshold = threshold
    scheduler._initial_allocation_enabled = True
    scheduler._initial_allocation_buffer = []
    scheduler._decode_attn_wave_release_pending_request_ids = set()
    scheduler._decode_attn_wave_release_completed_request_ids = set()
    return scheduler


def test_decode_attn_threshold_one_releases_only_first_initial_wave() -> None:
    scheduler = _build_scheduler(threshold=1, num_requests=8)
    scheduler._request_queue = [_build_request(index) for index in range(1, 9)]

    first_wave = scheduler.schedule()

    assert [request.id for _, _, request in first_wave] == [1]
    assert [request.id for request in scheduler._initial_allocation_buffer] == list(
        range(2, 9)
    )
    assert scheduler._request_queue == []
    assert scheduler._decode_attn_initial_allocation_done is False
    assert scheduler._decode_attn_initial_allocation_allocated_requests == 1
    assert scheduler._decode_attn_wave_release_pending_request_ids == {1}

    held_mapping = scheduler.schedule()

    assert held_mapping == []
    assert [request.id for request in scheduler._initial_allocation_buffer] == list(
        range(2, 9)
    )


def test_decode_attn_buffered_wave_releases_only_after_all_active_ids_finish() -> None:
    scheduler = _build_scheduler(threshold=2, num_requests=4)
    first = _build_request(1)
    second = _build_request(2)
    scheduler._initial_allocation_buffer = [_build_request(3), _build_request(4)]
    scheduler._decode_attn_initial_allocation_allocated_requests = 2
    scheduler._decode_attn_wave_release_pending_request_ids = {1, 2}

    first_events = scheduler.on_decode_attn_global_batch_end(
        1.0,
        SimpleNamespace(requests=[first]),
    )

    assert first_events == []
    assert scheduler._decode_attn_wave_release_completed_request_ids == {1}

    second_events = scheduler.on_decode_attn_global_batch_end(
        1.0,
        SimpleNamespace(requests=[second]),
    )

    assert scheduler._decode_attn_wave_release_completed_request_ids == set()
    assert len(second_events) == 1
    assert isinstance(second_events[0], ClusterScheduleEvent)


def test_decode_attn_releases_final_partial_wave() -> None:
    scheduler = _build_scheduler(threshold=2, num_requests=3)
    scheduler._request_queue = [_build_request(1), _build_request(2)]

    first_wave = scheduler.schedule()

    assert [request.id for _, _, request in first_wave] == [1, 2]
    assert scheduler._decode_attn_initial_allocation_done is False

    scheduler._request_queue = [_build_request(3)]
    final_wave = scheduler.schedule()

    assert [request.id for _, _, request in final_wave] == [3]
    assert scheduler._initial_allocation_buffer == []
    assert scheduler._decode_attn_initial_allocation_done is True
    assert scheduler._decode_attn_wave_release_pending_request_ids == set()


def test_decode_attn_global_end_ignores_request_outside_active_wave() -> None:
    scheduler = _build_scheduler(threshold=1, num_requests=2)
    scheduler._decode_attn_initial_allocation_allocated_requests = 1
    scheduler._initial_allocation_buffer = [_build_request(2)]
    scheduler._decode_attn_wave_release_pending_request_ids = {1}

    events = scheduler.on_decode_attn_global_batch_end(
        1.0,
        SimpleNamespace(requests=[_build_request(99)]),
    )

    assert events == []
    assert scheduler._decode_attn_wave_release_pending_request_ids == {1}
    assert scheduler._decode_attn_wave_release_completed_request_ids == set()


def test_decode_attn_without_threshold_keeps_dynamic_scheduling() -> None:
    scheduler = _build_scheduler(threshold=1, num_requests=2)
    scheduler._decode_attn_request_allocation_threshold = None
    scheduler._initial_allocation_enabled = False
    scheduler._decode_attn_initial_allocation_done = True
    scheduler._request_queue = [_build_request(1), _build_request(2)]

    mapping = scheduler.schedule()

    assert [request.id for _, _, request in mapping] == [1, 2]


def test_global_batch_end_emits_cluster_schedule_for_next_buffered_wave() -> None:
    scheduler = _build_scheduler(threshold=1, num_requests=2)
    scheduler._decode_attn_initial_allocation_allocated_requests = 1
    scheduler._initial_allocation_buffer = [_build_request(2)]
    scheduler._decode_attn_wave_release_pending_request_ids = {1}

    request = SimpleNamespace(
        id=1,
        current_thinking_round_index=0,
        num_restarts=0,
        execution_epoch=0,
        current_decode_token_index=1,
        first_decode_token_completed_at=1.0,
        num_processed_decode_tokens=1,
        num_decode_tokens=2,
        completed=False,
        pending_thinking_requeue=False,
    )
    batch = SimpleNamespace(
        id=101,
        schedule_epoch=0,
        requests=[request],
        request_execution_signatures=[(0, 0, 0)],
        request_mutation_signatures=[(0, 0, 0, 1)],
        thinking_round_start_times=[None],
        scheduled=False,
        on_batch_end=lambda *_args, **_kwargs: None,
    )

    events = GlobalBatchEndEvent(
        time=1.0,
        replica_id=0,
        dp_id=0,
        batch=batch,
        cluster_type=ClusterType.DECODE_ATTN,
    ).handle_event(_FakeGlobalScheduler(scheduler), _FakeMetricsStore())

    assert sum(isinstance(event, ReplicaScheduleEvent) for event in events) == 1
    assert sum(isinstance(event, ClusterScheduleEvent) for event in events) == 1
    assert scheduler._decode_attn_wave_release_pending_request_ids == set()
