"""Verify DP routing against the pinned vLLM V1 load semantics."""

from types import SimpleNamespace

import pytest

from frontier.config import BaseClusterSchedulerConfig
from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
    ClusterSchedulerRegistry,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import ClusterSchedulerType, ClusterType


def test_local_reservations_balance_burst_but_idle_snapshot_resets_tie():
    router = VllmDPLoadBalancer(2)
    assert [router.select(0.0) for _ in range(6)] == [0, 1, 0, 1, 0, 1]
    assert router.select(0.05) == 0
    assert router.frontend_counts == [RequestLoad(1, 0), RequestLoad(0, 0)]
    assert router.select(5.05) == 0


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([RequestLoad(0, 10), RequestLoad(0, 0)], 1),
        ([RequestLoad(1, 0), RequestLoad(0, 3)], 1),
        ([RequestLoad(0, 3), RequestLoad(0, 4)], 0),
        ([RequestLoad(0, 3), RequestLoad(0, 3)], 0),
    ],
)
def test_weighted_score_uses_published_waiting_and_running(counts, expected):
    router = VllmDPLoadBalancer(2)
    for lane, load in enumerate(counts):
        router.report(0.01, lane, 0, load)
    assert router.frontend_counts == [RequestLoad(0, 0)] * 2
    assert router.select(0.06) == expected


def test_previous_step_is_published_before_latest_counts():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 10))
    router.select(0.06)
    router.report(0.10, 0, 1, RequestLoad(0, 5))
    router.report(0.12, 0, 2, RequestLoad(0, 1))
    router.select(0.16)
    assert router.frontend_counts[0] == RequestLoad(0, 5)
    router.select(0.26)
    assert router.frontend_counts[0] == RequestLoad(0, 1)


def test_unchanged_report_does_not_extend_collection_wait():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 10))
    router.report(0.04, 0, 1, RequestLoad(0, 10))
    assert router.select(0.06) == 1
    assert router.last_publish_ms == 60


def test_same_step_reports_are_collected_without_previous_step_snapshot():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 2))
    router.report(0.02, 1, 0, RequestLoad(0, 3))
    assert router.last_step_counts is None
    assert router.next_publish_ms == 70
    assert router.select(0.07) == 0
    assert router.frontend_counts == [RequestLoad(1, 2), RequestLoad(0, 3)]


def test_report_at_deadline_is_handled_before_timeout():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 10))
    router.report(0.06, 1, 0, RequestLoad(0, 2))
    assert router.last_publish_ms == -5000
    assert router.next_publish_ms == 110
    router.select(0.11)
    assert router.frontend_counts[0].running == 10


def test_suppressed_report_does_not_publish_before_same_time_peer_message():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 10))
    router.report(0.06, 0, 1, RequestLoad(0, 10))
    router.report(0.06, 1, 1, RequestLoad(0, 2))
    assert router.last_publish_ms == -5000
    assert router.last_step_counts == [RequestLoad(0, 10), RequestLoad(0, 0)]


def test_heartbeat_replaces_speculative_frontend_waiting():
    router = VllmDPLoadBalancer(2)
    router.report(0.01, 0, 0, RequestLoad(0, 10))
    assert router.select(0.06) == 1
    assert router.select(0.07) == 1
    assert router.frontend_counts[1].waiting == 2
    assert router.select(5.06) == 1
    assert router.frontend_counts[1].waiting == 1


def test_load_accessor_keeps_running_separate_from_schedulable_pending():
    lane = VLLMv1EngineReplicaScheduler.__new__(VLLMv1EngineReplicaScheduler)
    lane._cluster_type = ClusterType.MONOLITHIC
    lane._request_queue = [object(), object()]
    lane._preempted_requests = [object()]
    lane._running_requests = [object()] * 10
    assert lane.get_request_load() == RequestLoad(3, 10)
    assert lane.num_pending_requests == 3
    # Exercise the existing queue reconstruction used after admission/preemption.
    waiting = SimpleNamespace(_preempted=False)
    preempted = SimpleNamespace(_preempted=True)
    lane._set_waiting_queues_from_ordered_requests([waiting, preempted])
    assert lane.get_request_load() == RequestLoad(2, 10)


@pytest.mark.parametrize("kind", list(ClusterSchedulerType))
def test_all_cluster_policy_configs_and_registry_entries_remain_discoverable(kind):
    config = BaseClusterSchedulerConfig.create_from_type(kind)
    assert config.get_type() == kind
    assert ClusterSchedulerRegistry.get_class(kind) is not None


def test_invalid_time_fails_before_routing():
    router = VllmDPLoadBalancer(2)
    router.select(1.0)
    with pytest.raises(ValueError, match="backwards"):
        router.select(0.0)
    with pytest.raises(ValueError, match="finite"):
        router.select(float("nan"))
