"""PDD parallel simulator initialization contracts."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from frontier.cluster_simulator import ClusterSimulator
from frontier.events import RequestArrivalEvent
from frontier.simulator import Simulator
from frontier.types import ClusterType


def test_init_parallel_mode_constructs_cluster_simulators_and_seeds_events(
    tmp_path,
) -> None:
    schedulers = {
        ClusterType.PREFILL: object(),
        ClusterType.DECODE: object(),
    }
    global_scheduler = SimpleNamespace(
        _parallel_coordination_lock=threading.Lock(),
        get_cluster_scheduler=lambda cluster_type: schedulers[cluster_type],
    )

    simulator = object.__new__(Simulator)
    simulator._clusters = {
        ClusterType.PREFILL: object(),
        ClusterType.DECODE: object(),
    }
    simulator._global_scheduler = global_scheduler
    simulator._metric_store = object()
    simulator._profiler = object()
    simulator._config = SimpleNamespace(
        enable_cluster_event_logging=False,
        cluster_event_log_dir=str(tmp_path / "cluster-events"),
        cluster_event_log_level="INFO",
    )
    simulator._init_parallel_events = Mock()

    simulator._init_parallel_mode()

    assert simulator._parallel_mode is True
    assert set(simulator._cluster_simulators) == {
        ClusterType.PREFILL,
        ClusterType.DECODE,
    }
    simulator._init_parallel_events.assert_called_once_with()

    for cluster_type, cluster_simulator in simulator._cluster_simulators.items():
        assert isinstance(cluster_simulator, ClusterSimulator)
        assert cluster_simulator._cluster_type is cluster_type
        assert cluster_simulator._cluster_scheduler is schedulers[cluster_type]
        assert cluster_simulator._global_scheduler is global_scheduler
        assert cluster_simulator._metrics_store is simulator._metric_store
        assert cluster_simulator._profiler is simulator._profiler
        assert cluster_simulator._enable_event_logging is False
        callback = cluster_simulator._can_process_event_priority
        assert callback.__self__ is simulator
        assert callback.__func__ is Simulator._can_parallel_cluster_process_event


class _RecordingClusterSimulator:
    def __init__(self) -> None:
        self.events = []

    def add_event(self, event) -> None:
        self.events.append(event)


def test_init_parallel_events_routes_online_arrivals_to_prefill() -> None:
    requests = [
        SimpleNamespace(arrived_at=0.25),
        SimpleNamespace(arrived_at=0.75),
    ]
    prefill_simulator = _RecordingClusterSimulator()
    decode_simulator = _RecordingClusterSimulator()
    metrics_store = SimpleNamespace(register_total_requests=Mock())
    global_scheduler = SimpleNamespace()

    simulator = object.__new__(Simulator)
    simulator._request_generator = SimpleNamespace(
        generate=Mock(return_value=requests)
    )
    simulator._metric_store = metrics_store
    simulator._global_scheduler = global_scheduler
    simulator._cluster_simulators = {
        ClusterType.PREFILL: prefill_simulator,
        ClusterType.DECODE: decode_simulator,
    }
    simulator._config = SimpleNamespace(
        simulation_mode="online",
        is_disaggregated_mode=lambda: True,
    )

    simulator._init_parallel_events()

    metrics_store.register_total_requests.assert_called_once_with(2)
    assert decode_simulator.events == []
    assert len(prefill_simulator.events) == 2
    assert all(
        isinstance(event, RequestArrivalEvent)
        for event in prefill_simulator.events
    )
    assert [event.time for event in prefill_simulator.events] == [0.25, 0.75]
    assert [
        event.get_target_cluster() for event in prefill_simulator.events
    ] == [ClusterType.PREFILL, ClusterType.PREFILL]
