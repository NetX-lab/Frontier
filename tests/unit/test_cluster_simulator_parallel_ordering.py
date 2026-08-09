"""Parallel ClusterSimulator ordering and dependency contracts."""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable

import pytest

from frontier.cluster_simulator import ClusterSimulator
from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from frontier.simulator import Simulator
from frontier.types import ClusterType


class _ClaimEvent:
    def __init__(self, time: float) -> None:
        self.time = time
        self._priority_number = (time, 1, 1)


class _FrontierEvent:
    def __init__(self, time: float, event_id: int) -> None:
        self.time = time
        self._id = event_id
        self._priority_number = (time, event_id, 1)


def _build_real_parallel_pair(*, max_queue_size: int = 100):
    """Build real GlobalScheduler/ClusterSimulator objects for frontier tests."""
    global_scheduler = object.__new__(BaseGlobalScheduler)
    global_scheduler._enable_parallel_mode = True
    global_scheduler._parallel_coordination_lock = threading.Lock()
    global_scheduler._inter_cluster_queue = queue.Queue(maxsize=max_queue_size)
    global_scheduler._cluster_event_buffers = defaultdict(list)
    global_scheduler._buffer_lock = threading.Lock()
    global_scheduler._events_sent = 0
    global_scheduler._events_delivered = 0
    global_scheduler._queue_full_count = 0

    owner = object.__new__(Simulator)
    owner._global_scheduler = global_scheduler
    owner._cluster_simulators = {}
    for cluster_type in (ClusterType.PREFILL, ClusterType.DECODE):
        owner._cluster_simulators[cluster_type] = ClusterSimulator(
            cluster_type=cluster_type,
            cluster_scheduler=object(),
            global_scheduler=global_scheduler,
            metrics_store=object(),
            can_process_event_priority=owner._can_parallel_cluster_process_event,
        )
    return global_scheduler, owner, owner._cluster_simulators


class _HandoffObservingLock:
    """Observe peer-visible state at the exact coordination-lock handoff."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.observe: Callable[[], tuple] | None = None
        self.handoff_state: tuple | None = None

    def __enter__(self) -> "_HandoffObservingLock":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()
        if self.observe is not None:
            self.handoff_state = self.observe()


class _CountingLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.enter_count = 0

    def __enter__(self) -> "_CountingLock":
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


class _QuiescenceCluster:
    def get_runtime_state(self) -> dict:
        return {
            "cluster_type": ClusterType.PREFILL.name,
            "queue_size": 0,
            "local_time": 0.0,
            "is_running": True,
            "is_processing_event": False,
        }


class _LegacyQuiescenceCluster:
    def get_queue_size(self) -> int:
        return 0

    def get_local_time(self) -> float:
        return 0.0

    def is_processing_event(self) -> bool:
        return False


def _build_cluster_simulator(
    *,
    coordination_lock=...,
    can_process_event_priority=...,
) -> ClusterSimulator:
    global_scheduler = SimpleNamespace(
        get_events_for_cluster=lambda _cluster_type: [],
    )
    if coordination_lock is ...:
        coordination_lock = threading.Lock()
    if coordination_lock is not None:
        global_scheduler._parallel_coordination_lock = coordination_lock
    if can_process_event_priority is ...:
        can_process_event_priority = (
            lambda _cluster_type, _event_priority: True
        )

    return ClusterSimulator(
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler=object(),
        global_scheduler=global_scheduler,
        metrics_store=object(),
        enable_event_logging=False,
        can_process_event_priority=can_process_event_priority,
    )


def test_claim_publishes_in_flight_state_before_coordination_lock_handoff() -> None:
    coordination_lock = _HandoffObservingLock()
    simulator = _build_cluster_simulator(coordination_lock=coordination_lock)
    event = _ClaimEvent(time=4.5)
    simulator.add_event(event)
    coordination_lock.observe = lambda: (
        simulator._is_processing_event,
        simulator._current_event_name,
        simulator._current_event_time,
        simulator._current_event_priority,
    )

    claimed_event = simulator._claim_next_event()

    assert claimed_event is event
    assert coordination_lock.handoff_state == (
        True,
        "_ClaimEvent",
        4.5,
        (4.5, 1, 1),
    )


def test_constructor_rejects_missing_parallel_coordination_lock() -> None:
    with pytest.raises(RuntimeError, match="parallel coordination lock"):
        _build_cluster_simulator(coordination_lock=None)


def test_constructor_rejects_missing_event_priority_gate() -> None:
    with pytest.raises(RuntimeError, match="event-priority gate callback"):
        _build_cluster_simulator(can_process_event_priority=None)


def test_event_priority_gate_denial_keeps_event_queued_and_unpublished() -> None:
    gate_calls: list[tuple[ClusterType, tuple]] = []

    def deny_event(cluster_type: ClusterType, event_priority: tuple) -> bool:
        gate_calls.append((cluster_type, event_priority))
        return False

    simulator = _build_cluster_simulator(
        can_process_event_priority=deny_event,
    )
    simulator.add_event(_ClaimEvent(time=7.0))

    assert simulator._claim_next_event() is None
    assert gate_calls == [(ClusterType.PREFILL, (7.0, 1, 1))]
    assert simulator.get_queue_size() == 1
    assert simulator._is_processing_event is False
    assert simulator._current_event_name is None
    assert simulator._current_event_time is None
    assert simulator._current_event_priority is None


def test_claim_drains_incoming_events_once_before_priority_gate() -> None:
    gate_calls: list[tuple] = []

    def gate_event(_cluster_type: ClusterType, event_priority: tuple) -> bool:
        gate_calls.append(event_priority)
        return True

    simulator = _build_cluster_simulator(
        can_process_event_priority=gate_event,
    )
    simulator.add_event(_ClaimEvent(time=7.0))
    process_calls = 0

    def record_incoming_drain() -> None:
        nonlocal process_calls
        process_calls += 1
    simulator._process_incoming_events = record_incoming_drain

    assert simulator._claim_next_event() is not None
    assert process_calls == 1
    assert gate_calls == [(7.0, 1, 1)]
    assert simulator.get_queue_size() == 0
    assert simulator._is_processing_event is True


def test_local_time_failure_clears_the_published_claim() -> None:
    simulator = _build_cluster_simulator()
    simulator.add_event(_ClaimEvent(time=9.0))

    def fail_local_time_update(
        _cluster_type: ClusterType,
        _logical_time: float,
    ) -> None:
        raise RuntimeError("logical time update failed")

    simulator._global_scheduler.update_cluster_logical_time = (
        fail_local_time_update
    )
    simulator._running = True

    with pytest.raises(RuntimeError, match="logical time update failed"):
        simulator._run_event_loop()

    assert isinstance(simulator._fatal_error, RuntimeError)
    assert simulator._running is False
    assert simulator._is_processing_event is False
    assert simulator._current_event_name is None
    assert simulator._current_event_time is None
    assert simulator._current_event_priority is None


def test_claim_rejects_candidate_when_peer_buffer_contains_earlier_event() -> None:
    global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    prefill = cluster_simulators[ClusterType.PREFILL]
    prefill.add_event(_FrontierEvent(time=8.0, event_id=1))
    with global_scheduler._buffer_lock:
        global_scheduler._cluster_event_buffers[ClusterType.DECODE].append(
            _FrontierEvent(time=5.0, event_id=2)
        )

    assert prefill._claim_next_event() is None
    assert prefill.peek_next_event_time() == 8.0
    assert prefill.peek_next_event_priority() == (8.0, 1, 1)


def test_claim_rejects_candidate_when_shared_queue_contains_earlier_event() -> None:
    global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    prefill = cluster_simulators[ClusterType.PREFILL]
    prefill.add_event(_FrontierEvent(time=8.0, event_id=1))
    global_scheduler.route_event_to_cluster(
        _FrontierEvent(time=5.0, event_id=2), ClusterType.DECODE
    )

    assert prefill._claim_next_event() is None
    assert prefill.peek_next_event_time() == 8.0
    assert prefill.peek_next_event_priority() == (8.0, 1, 1)


def test_lower_event_priority_wins_equal_timestamp_regardless_of_cluster_order() -> None:
    global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    decode = cluster_simulators[ClusterType.DECODE]
    earlier_event = _FrontierEvent(time=8.0, event_id=1)
    decode.add_event(earlier_event)
    with global_scheduler._buffer_lock:
        global_scheduler._cluster_event_buffers[ClusterType.PREFILL].append(
            _FrontierEvent(time=8.0, event_id=2)
        )

    assert decode._claim_next_event() is earlier_event


def test_decode_waits_for_lower_prefill_priority_at_equal_timestamp() -> None:
    _global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    prefill = cluster_simulators[ClusterType.PREFILL]
    decode = cluster_simulators[ClusterType.DECODE]
    earlier_prefill_event = _FrontierEvent(time=8.0, event_id=1)
    later_decode_event = _FrontierEvent(time=8.0, event_id=2)
    prefill.add_event(earlier_prefill_event)
    decode.add_event(later_decode_event)

    assert decode._claim_next_event() is None
    assert prefill._claim_next_event() is earlier_prefill_event
    assert decode._claim_next_event() is None


def test_in_flight_parent_priority_blocks_same_time_child_claim() -> None:
    global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    decode = cluster_simulators[ClusterType.DECODE]
    prefill = cluster_simulators[ClusterType.PREFILL]
    parent = _FrontierEvent(time=8.0, event_id=1)
    child = _FrontierEvent(time=8.0, event_id=2)
    decode.add_event(parent)

    assert decode._claim_next_event() is parent
    global_scheduler.route_event_to_cluster(child, ClusterType.PREFILL)

    assert prefill._claim_next_event() is None
    assert prefill.peek_next_event_time() == 8.0
    assert prefill.peek_next_event_priority() == (8.0, 2, 1)


def test_priority_gate_fails_fast_for_processing_peer_without_priority() -> None:
    owner = object.__new__(Simulator)
    owner._cluster_simulators = {
        ClusterType.PREFILL: object(),
        ClusterType.DECODE: SimpleNamespace(
            get_runtime_state=lambda: {
                "is_processing_event": True,
                "current_event_priority": None,
                "next_event_priority": None,
            }
        ),
    }
    owner._global_scheduler = SimpleNamespace(
        get_pending_inter_cluster_event_frontier=lambda: None,
    )

    with pytest.raises(RuntimeError, match="missing its current event priority"):
        owner._can_parallel_cluster_process_event(
            ClusterType.PREFILL,
            (8.0, 2, 1),
        )


def test_claim_allows_candidate_when_peer_buffer_is_later() -> None:
    global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    prefill = cluster_simulators[ClusterType.PREFILL]
    event = _FrontierEvent(time=8.0, event_id=1)
    prefill.add_event(event)
    with global_scheduler._buffer_lock:
        global_scheduler._cluster_event_buffers[ClusterType.DECODE].append(
            _FrontierEvent(time=9.0, event_id=2)
        )

    assert prefill._claim_next_event() is event


def test_claim_allows_candidate_when_no_inter_cluster_event_is_pending() -> None:
    _global_scheduler, _owner, cluster_simulators = _build_real_parallel_pair()
    prefill = cluster_simulators[ClusterType.PREFILL]
    event = _FrontierEvent(time=8.0, event_id=1)
    prefill.add_event(event)

    assert prefill._claim_next_event() is event


def test_route_drains_staged_messages_before_bounded_queue_enqueue() -> None:
    global_scheduler, _owner, _cluster_simulators = _build_real_parallel_pair(
        max_queue_size=1
    )
    staged_event = _FrontierEvent(time=1.0, event_id=1)
    routed_event = _FrontierEvent(time=2.0, event_id=2)
    global_scheduler._inter_cluster_queue.put(
        (staged_event, ClusterType.DECODE)
    )

    global_scheduler.route_event_to_cluster(routed_event, ClusterType.DECODE)

    assert global_scheduler._inter_cluster_queue.qsize() == 1
    queued_event, queued_target = global_scheduler._inter_cluster_queue.get_nowait()
    assert queued_event is routed_event
    assert queued_target is ClusterType.DECODE
    assert global_scheduler._cluster_event_buffers[ClusterType.DECODE] == [
        staged_event
    ]
    assert global_scheduler._queue_full_count == 0


def test_pending_frontier_returns_minimum_sequential_event_priority() -> None:
    global_scheduler, _owner, _cluster_simulators = _build_real_parallel_pair()
    with global_scheduler._buffer_lock:
        global_scheduler._cluster_event_buffers[ClusterType.DECODE].append(
            _FrontierEvent(time=5.0, event_id=1)
        )
        global_scheduler._cluster_event_buffers[ClusterType.PREFILL].append(
            _FrontierEvent(time=5.0, event_id=2)
        )

    with global_scheduler._parallel_coordination_lock:
        assert global_scheduler.get_pending_inter_cluster_event_frontier() == (
            5.0,
            1,
            1,
        )


def test_pending_frontier_fails_fast_when_shared_queue_was_not_drained() -> None:
    global_scheduler, _owner, _cluster_simulators = _build_real_parallel_pair()
    global_scheduler._inter_cluster_queue.put(
        (_FrontierEvent(time=5.0, event_id=1), ClusterType.DECODE)
    )

    with global_scheduler._parallel_coordination_lock:
        with pytest.raises(RuntimeError, match="not drained"):
            global_scheduler.get_pending_inter_cluster_event_frontier()


def test_parallel_quiescence_snapshot_holds_coordination_lock() -> None:
    coordination_lock = _CountingLock()
    simulator = object.__new__(Simulator)
    simulator._cluster_simulators = {
        ClusterType.PREFILL: _QuiescenceCluster(),
    }
    simulator._global_scheduler = SimpleNamespace(
        _enable_parallel_mode=True,
        _parallel_coordination_lock=coordination_lock,
        _get_inter_cluster_communication_stats_locked=lambda: {
            "queue_size": 0,
            "total_buffered_events": 0,
        },
    )

    simulator._collect_parallel_quiescence_state()

    assert coordination_lock.enter_count == 1


def test_parallel_quiescence_requires_runtime_state_api() -> None:
    simulator = object.__new__(Simulator)
    simulator._cluster_simulators = {
        ClusterType.PREFILL: _LegacyQuiescenceCluster(),
    }
    simulator._global_scheduler = SimpleNamespace(
        _enable_parallel_mode=True,
        _parallel_coordination_lock=threading.Lock(),
        _get_inter_cluster_communication_stats_locked=lambda: {
            "queue_size": 0,
            "total_buffered_events": 0,
        },
    )

    with pytest.raises(AttributeError, match="get_runtime_state"):
        simulator._collect_parallel_quiescence_state()


def test_route_events_rejects_child_before_parent_time() -> None:
    simulator = _build_cluster_simulator()
    simulator._current_event_priority = (5.0, 1, 1)
    child = _FrontierEvent(time=4.0, event_id=2)

    with pytest.raises(RuntimeError, match="not later than parent event"):
        simulator._route_events([child])

    assert simulator.get_queue_size() == 0


def test_route_events_rejects_equal_time_child_with_lower_priority() -> None:
    simulator = _build_cluster_simulator()
    simulator._current_event_priority = (5.0, 2, 1)
    child = _FrontierEvent(time=5.0, event_id=1)

    with pytest.raises(RuntimeError, match="not later than parent event"):
        simulator._route_events([child])

    assert simulator.get_queue_size() == 0


def test_route_events_rejects_child_with_same_priority_as_parent() -> None:
    simulator = _build_cluster_simulator()
    simulator._current_event_priority = (5.0, 1, 1)
    child = _FrontierEvent(time=5.0, event_id=1)

    with pytest.raises(RuntimeError, match="not later than parent event"):
        simulator._route_events([child])

    assert simulator.get_queue_size() == 0


def test_route_events_validates_all_children_before_publishing_any() -> None:
    simulator = _build_cluster_simulator()
    simulator._current_event_priority = (5.0, 1, 1)
    valid_child = _FrontierEvent(time=6.0, event_id=2)
    invalid_child = _FrontierEvent(time=4.0, event_id=3)

    with pytest.raises(RuntimeError):
        simulator._route_events([valid_child, invalid_child])

    assert simulator.get_queue_size() == 0


def test_route_events_requires_published_parent_priority() -> None:
    simulator = _build_cluster_simulator()
    child = _FrontierEvent(time=5.0, event_id=2)

    with pytest.raises(RuntimeError, match="current parent event priority"):
        simulator._route_events([child])

    assert simulator.get_queue_size() == 0


def test_route_events_accepts_empty_child_list_without_parent() -> None:
    simulator = _build_cluster_simulator()

    simulator._route_events([])

    assert simulator.get_queue_size() == 0


def test_route_events_allows_child_at_parent_time() -> None:
    simulator = _build_cluster_simulator()
    simulator._current_event_priority = (5.0, 1, 1)
    child = _FrontierEvent(time=5.0, event_id=2)

    simulator._route_events([child])

    assert simulator.peek_next_event_priority() == (5.0, 2, 1)
