"""NIXL READ completion at DECODE engine steps (nixl_read_completion='engine_step')."""

import logging
from collections import deque
from types import SimpleNamespace

import pytest

from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.entities.batch import DummyForwardBatch
from frontier.events import replica_stage_schedule_event
from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _lane(*, depth: int = 1) -> VLLMv1EngineReplicaScheduler:
    """A DECODE lane without DP whose engine has nothing in flight."""
    lane = object.__new__(VLLMv1EngineReplicaScheduler)
    lane._replica_id = 0
    lane._replica_local_id = None
    lane._cluster_type = ClusterType.DECODE
    lane._max_concurrent_batches = depth
    lane._running_requests = []
    lane._request_queue = []
    lane._preempted_requests = []
    lane._waiting_requests = []
    lane._nixl_reads_requested = []
    lane._nixl_reads_claimed = {}
    lane._nixl_reads_unposted = []
    lane._nixl_reads_posted = []
    lane._nixl_reads_found_by_batch = {}
    lane._nixl_reads_found_by_dummy_forward = []
    lane._nixl_reads_reported = []
    lane._uses_output_placeholders = False
    lane._has_engine_batch_queue = True
    lane._runs_dp_dummy_passes = False
    lane._blocking_batch_id = None
    lane._dummy_forwards_in_flight = 0
    lane._waits_at_dp_sync = False
    lane._in_flight_batch_ids = deque()
    lane._last_iteration_end = None
    lane._dp_sync_released_lane_ids = []
    lane._popped_output_events = []
    lane._preemption_followup_poll_pending = False
    return lane


def _read(request_id: int, transfer_time_ms: float) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, transfer_time_ms=transfer_time_ms)


def _batch(batch_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=batch_id)


class _Cluster:
    def __init__(self, lane) -> None:
        self.lane = lane

    def get_replica_scheduler(self, replica_id, replica_local_id):
        assert (replica_id, replica_local_id) == (0, None)
        return self.lane


class _Global:
    def __init__(self, lane) -> None:
        self.cluster = _Cluster(lane)

    def get_cluster_scheduler(self, cluster_type):
        assert cluster_type == ClusterType.DECODE
        return self.cluster


@pytest.fixture
def reported(monkeypatch):
    """Record the transfer ends that the READ reports run, instead of running them."""
    ends = []

    def handle_event(self, scheduler, metrics_store):
        ends.append((self.time, self._transfer_info))
        return [f"arrival-{self._transfer_info.request_id}"]

    monkeypatch.setattr(KVCacheTransferEndEvent, "handle_event", handle_event)
    # The engine forms no batch, so each pass is a connector-only step.
    monkeypatch.setattr(BaseReplicaScheduler, "on_schedule", lambda self, time=0.0: [])
    return ends


def _schedule(lane, time):
    return ReplicaScheduleEvent(time, 0, ClusterType.DECODE, None).handle_event(_Global(lane), None)


def test_config_defaults_to_transfer_end_and_admits_engine_step_only_for_nixl():
    assert AnalyticalKVCacheTransferConfig().nixl_read_completion == "transfer_end"
    config = AnalyticalKVCacheTransferConfig(kv_connector="nixl", nixl_read_completion="engine_step")
    assert config.nixl_read_completion == "engine_step"
    with pytest.raises(ValueError, match="kv_connector is 'p2p_nccl'"):
        AnalyticalKVCacheTransferConfig(kv_connector="p2p_nccl", nixl_read_completion="engine_step")
    with pytest.raises(ValueError, match="nixl_read_completion must be one of"):
        AnalyticalKVCacheTransferConfig(kv_connector="nixl", nixl_read_completion="engine_output")


def test_step_that_posts_a_read_never_finds_it_and_a_later_step_does():
    lane = _lane()
    first, second = _read(1, 2.0), _read(2, 5.0)
    lane.register_nixl_read(first)
    assert lane.has_unfinished_requests
    # A requested READ waits for a schedule pass to claim it.
    lane.on_engine_step_start(0.5)
    assert lane.next_nixl_read_end is None

    lane._claim_nixl_reads([])
    lane.on_engine_step_start(1.0)
    assert lane.consume_found_nixl_reads() == []
    assert lane.next_nixl_read_end == 1.0 + 2.0e-3

    lane.register_nixl_read(second)
    lane._claim_nixl_reads([])
    lane.on_engine_step_start(1.001)
    assert lane.consume_found_nixl_reads() == []
    assert lane._nixl_reads_posted == [(1.0 + 2.0e-3, first), (1.001 + 5.0e-3, second)]

    # A connector-only step reports what it finds at once.
    lane.on_engine_step_start(1.0 + 2.0e-3)
    assert lane.has_unfinished_requests
    assert lane.consume_found_nixl_reads() == [first]
    assert lane.next_nixl_read_end == 1.001 + 5.0e-3

    lane.on_engine_step_start(1.02)
    assert lane.consume_found_nixl_reads() == [second]
    assert lane.next_nixl_read_end is None
    assert not lane.has_unfinished_requests


def test_depth_two_read_rides_its_pass_batch_and_reports_with_the_finding_batch_output():
    lane = _lane(depth=2)
    read = _read(1, 2.0)
    lane.register_nixl_read(read)
    # The pass at 1.0 forms batch 11 while batch 10 is in flight.
    lane._claim_nixl_reads([_batch(11), _batch(12)])
    assert lane._nixl_reads_requested == []

    # Batch 10's own step started earlier; batch 11's start posts the READ.
    lane.on_engine_step_start(1.001, _batch(11))
    assert lane.next_nixl_read_end == 1.001 + 2.0e-3
    lane._report_nixl_reads_of_batch(_batch(10))
    assert lane.consume_found_nixl_reads() == []

    # Batch 12 starts before the wire ends, batch 13 after it.
    lane.on_engine_step_start(1.002, _batch(12))
    lane.on_engine_step_start(1.004, _batch(13))
    assert lane.next_nixl_read_end is None
    lane._report_nixl_reads_of_batch(_batch(11))
    lane._report_nixl_reads_of_batch(_batch(12))
    assert lane.consume_found_nixl_reads() == []
    assert lane.has_unfinished_requests

    lane._report_nixl_reads_of_batch(_batch(13))
    assert lane.consume_found_nixl_reads() == [read]
    assert not lane.has_unfinished_requests


def test_pass_that_forms_no_batch_leaves_its_reads_to_the_next_step_of_any_kind():
    lane = _lane(depth=2)
    early, late = _read(1, 2.0), _read(2, 3.0)
    lane.register_nixl_read(early)
    lane._claim_nixl_reads([])
    lane.register_nixl_read(late)
    lane._claim_nixl_reads([_batch(21)])

    # Batch 20 did not claim either READ, yet its start posts the unclaimed one.
    lane.on_engine_step_start(1.0, _batch(20))
    assert lane._nixl_reads_posted == [(1.0 + 2.0e-3, early)]
    lane.on_engine_step_start(1.001, _batch(21))
    assert lane._nixl_reads_posted == [(1.0 + 2.0e-3, early), (1.001 + 3.0e-3, late)]


def test_idle_lane_posts_at_once_and_reports_when_its_wake_up_finds_the_read(reported):
    lane = _lane()
    read = _read(3, 2.5)
    lane.register_nixl_read(read)

    events = _schedule(lane, 4.0)
    assert [(type(event), event.time) for event in events] == [(ReplicaScheduleEvent, 4.0 + 2.5e-3)]
    assert reported == []

    events = _schedule(lane, events[0].time)
    assert events == ["arrival-3"]
    assert reported == [(4.0 + 2.5e-3, read)]
    assert not lane.has_unfinished_requests


def test_lane_waiting_on_a_batch_leaves_the_read_to_its_next_step(reported):
    lane = _lane()
    lane._in_flight_batch_ids.append(9)
    lane._blocking_batch_id = 9
    lane.register_nixl_read(_read(4, 2.0))

    assert _schedule(lane, 4.0) == []
    assert reported == []
    assert lane._nixl_reads_posted == []


def test_decode_step_start_posts_and_checks_the_lane_reads(monkeypatch):
    lane = _lane()
    read = _read(5, 2.0)
    lane.register_nixl_read(read)
    monkeypatch.setattr(
        replica_stage_schedule_event, "start_dummy_forward", lambda *args: ["dummy-start"]
    )

    def start(time):
        batch = object.__new__(DummyForwardBatch)
        batch._forward_start_host_delay = 0.0
        return ReplicaStageScheduleEvent(time, 0, 0, ClusterType.DECODE, None)._schedule_popped_batch(
            None, batch, None, _Cluster(lane), 0, None,
            logging.getLogger(__name__),
        )

    lane._claim_nixl_reads([])
    assert start(2.0) == ["dummy-start"]
    assert lane._nixl_reads_posted == [(2.0 + 2.0e-3, read)]
    assert start(2.02) == ["dummy-start"]
    # The dummy forward's step reports the READ when it ends.
    assert lane.consume_found_nixl_reads() == []
    lane._dummy_forwards_in_flight = 1
    assert lane.on_dummy_forward_end(2.03)
    assert lane.consume_found_nixl_reads() == [read]


class _Group:
    def __init__(self, lanes) -> None:
        self.lanes = lanes

    def get_replica_scheduler(self, replica_id, replica_local_id):
        return self.lanes[replica_local_id]


def _dp_group(size: int = 2) -> list:
    """DP lanes of one Replica, each just past a dummy pass issued at 1.0."""
    lanes = [_lane() for _ in range(size)]
    group = _Group(lanes)
    for lane_id, lane in enumerate(lanes):
        lane._replica_local_id = lane_id
        lane._runs_dp_dummy_passes = True
        lane._replica_config = SimpleNamespace(attn_dp=size)
        lane._cluster_scheduler = group
        lane._dummy_pass_start = 1.0
    return lanes


def test_dp_group_with_only_posted_reads_starts_its_next_pass_at_the_earliest_wire_end():
    first, second = _dp_group()
    first._nixl_reads_posted = [(1.006, _read(1, 6.0))]
    second._nixl_reads_posted = [(1.004, _read(2, 4.0))]
    # A dummy forward that took no time ran with no forward work in the group.
    assert first.next_pass_start(1.0) == 1.004
    assert second.next_pass_start(1.0) == 1.004
    # One that took time ran with a sibling's batch, and the loop goes on.
    assert first.next_pass_start(1.002) == 1.002


def test_dp_group_with_a_request_to_schedule_starts_its_next_pass_at_once():
    first, second = _dp_group()
    first._nixl_reads_posted = [(1.006, _read(1, 6.0))]
    second._nixl_reads_found_by_dummy_forward = [_read(2, 4.0)]
    assert first.next_pass_start(1.0) == 1.0
    second._nixl_reads_found_by_dummy_forward = []
    second._nixl_reads_requested = [_read(3, 4.0)]
    assert first.next_pass_start(1.0) == 1.0
    second._nixl_reads_requested = []
    second._waiting_requests = [object()]
    assert first.next_pass_start(1.0) == 1.0
    # With no READ posted, the wave ends at its next DP sync.
    first._nixl_reads_posted = []
    second._waiting_requests = []
    assert first.next_pass_start(1.0) == 1.0
