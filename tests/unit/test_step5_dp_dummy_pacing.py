"""PREFILL queue and host/device lifecycle of native DP dummy dispatches.

Costs and host cadence here are controlled CPU fixtures, not native timing
qualification. Event tests use the existing Step PD lifecycle harness.
"""

import heapq
import logging
import sys
from types import SimpleNamespace

import pytest

from frontier.config import VllmV1SchedulerConfig, global_vars
from frontier.entities import DummyForwardBatch, Request
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.scheduler.utils.replica_config import resolve_replica_scheduler_config
from frontier.types import ClusterType
from tests.unit import test_step5_pd_placement as pd
from tests.unit.test_step5_structural_pd import _GlobalScheduler, simulation_globals
from tests.unit.test_vllm_v1_max_model_len import _build_scheduler

@pytest.fixture(autouse=True)
def quiet_lifecycle_logs():
    previous = logging.root.manager.disable
    logging.disable(logging.INFO)
    yield
    logging.disable(previous)


CADENCE_MS = 6.307
PERIOD = CADENCE_MS * 1e-3


def cluster(monkeypatch, *, cadence=CADENCE_MS, capacity=2, agreement=False):
    config_class = VllmV1SchedulerConfig
    with monkeypatch.context() as patch:
        patch.setattr(pd, "VllmV1SchedulerConfig", lambda **kwargs: config_class(**{
            **kwargs, "enable_chunked_prefill": False, "max_concurrent_batches": capacity,
        }))
        patch.setattr(pd._StageCosts, "get_dp_dummy_step_cadence_ms", lambda _: cadence, raising=False)
        source = pd._scheduler(ClusterType.PREFILL)
    if agreement:
        predictor = next(iter(source._replica_schedulers.values()))._predictor
        predictor.requires_forward_start_agreement = True
        predictor.get_kv_block_zeroing_host_time_ms = lambda _: 0.0
        monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
        monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [1, 2, 4, 8, 16, 32])
    return source


def lanes(source):
    return [source.get_replica_scheduler(source._serving_replica_id, lane) for lane in range(4)]


class Trace(pd._PDExecution):
    """Retain engine formations and device drain ownership from real DES events."""

    def __init__(self, source):
        super().__init__(_GlobalScheduler(source, pd._scheduler()))
        self.formations, self.device_ends, self.wave_ends = [], [], []
        self.steps = 0
        self.source = source

    def step(self):
        assert self.queue, "event queue drained with unfinished engine/stage work"
        self.steps += 1
        assert self.steps < 100000, "engine/stage lifecycle failed to drain"
        _, event = heapq.heappop(self.queue)
        self.time = event.time
        lane = None
        if getattr(event, "_cluster_type", None) == ClusterType.PREFILL and isinstance(event, (ReplicaScheduleEvent, BatchStageEndEvent)):
            lane = self.source.get_replica_scheduler(event._replica_id, event._replica_local_id)
        wave_before = lane._dp_wave_running if lane else False
        if isinstance(event, BatchStageEndEvent) and isinstance(event._batch, DummyForwardBatch) and lane:
            ordinal = event._batch.forward_index
            paced = ordinal in getattr(lane, "_paced_dummy_passes", set())
            host_before = (lane._last_iteration_end, lane._dp_wave_steps, lane._iteration_open)
        children = event.handle_event(self.scheduler, self.metrics)
        self.add(children)
        if isinstance(event, ReplicaScheduleEvent) and lane:
            self.formations.extend((event.time, event._replica_local_id, batch._forward_cohort_id,
                                    isinstance(batch, DummyForwardBatch), batch)
                                   for batch in event._batches)
            if wave_before and not lane._dp_wave_running:
                self.wave_ends.append(event.time)
        if isinstance(event, BatchStageEndEvent) and isinstance(event._batch, DummyForwardBatch) and lane:
            self.device_ends.append((event.time, lane._replica_local_id, ordinal, paced))
            if paced:
                assert host_before == (lane._last_iteration_end, lane._dp_wave_steps, lane._iteration_open)
                assert not any(isinstance(child, ReplicaScheduleEvent) for child in children)
                assert ordinal not in lane._paced_dummy_passes
        assert all(engine._dummy_forwards_in_flight >= 0 for engine in lanes(self.source))
        return event, children

    def until(self, condition):
        while not condition():
            self.step()

    def arrival(self, lane_id=0, prompt=32):
        request = Request(self.time, prompt, 2)
        lane = lanes(self.source)[lane_id]
        lane.add_request(request)
        self.add([ReplicaScheduleEvent(self.time, self.source._serving_replica_id, ClusterType.PREFILL, lane_id)])
        return request

    def finish(self):
        self.until(lambda: not self.queue)
        assert all(engine.engine_loop_idle and not getattr(engine, "_paced_dummy_passes", set())
                   and not engine._dp_wave_running for engine in lanes(self.source))


def test_prefill_cli_override_reaches_role_scheduler_config(monkeypatch, tmp_path):
    from frontier.config import SimulationConfig

    monkeypatch.setattr(sys, "argv", ["frontier", "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters", "--replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--cluster_config_prefill_replica_scheduler_config_max_concurrent_batches", "2",
        "--metrics_config_output_dir", str(tmp_path / "metrics"),
        "--metrics_config_cache_dir", str(tmp_path / "cache")])
    config = SimulationConfig.create_from_cli_args()
    assert config.cluster_config.prefill_replica_scheduler_config_max_concurrent_batches == 2
    roles = config.get_clusters()
    assert resolve_replica_scheduler_config(roles[ClusterType.PREFILL], ClusterType.PREFILL).max_concurrent_batches == 2
    assert resolve_replica_scheduler_config(roles[ClusterType.DECODE], ClusterType.DECODE).max_concurrent_batches is None


@pytest.mark.parametrize("fields", [
    {"enable_chunked_prefill": True}, {"scheduling_policy": "priority"},
    {"enable_prefix_caching": True},
])
def test_prefill_queue_conditions(fields):
    with pytest.raises(NotImplementedError, match="Explicit PREFILL queue capacity requires"):
        _build_scheduler(max_concurrent_batches=2, **{"enable_chunked_prefill": False, **fields})


def test_grouped_prefill_queue_accepts_two_whole_prompts(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    engine = engines[0]
    assert engine._is_grouped_kv_allocation() and engine._max_concurrent_batches == 2
    assert not engine._uses_output_placeholders
    engine._max_num_running_reqs = engine._config.batch_size_cap = 2
    engine._max_num_scheduled_tokens = engine._config.max_tokens_in_batch = 32
    engine.add_request(Request(0, 32, 2))
    engine.add_request(Request(0, 32, 2))
    [first] = engine.on_schedule(0)
    for peer in engines[1:]:
        peer.on_schedule(0)
        peer.on_dummy_forward_end(.01)
        peer.on_schedule(.01)
    [second] = engine.on_schedule(.01)
    assert first.requests != second.requests
    assert first.num_tokens == second.num_tokens == [32]
    assert list(engine._in_flight_batch_ids) == [first.id, second.id]


def test_all_dummy_barrier_uses_latest_issue_and_stale_duplicate_ticks(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    for engine in engines:
        engine._dp_wave_running = True
    for i, engine in enumerate(engines):
        [dummy] = engine.on_schedule(i * .001)
        assert dummy.forward_index == 0
    tick = .003 + PERIOD
    assert all(engine._host_returns_at == tick and engine._dummy_forwards_in_flight == 0 for engine in engines)
    assert engines[-1].consume_dp_dispatch_host_ticks() == [(i, tick) for i in range(4)]
    assert engines[-1].consume_dp_dispatch_host_ticks() == []
    for engine in engines:
        assert engine.on_schedule(tick - .0001) == []
        assert engine._iteration_open and engine._dp_wave_steps == 0
    for engine in engines:
        assert engine.on_schedule(tick) == []
        assert engine.on_schedule(tick) == []  # duplicate cannot count a second iteration
    assert all(not engine._dp_wave_running and engine._dp_wave_steps == 0 for engine in engines)
    assert all(engine.engine_loop_idle for engine in engines)


def test_three_queued_passes_drain_after_step16_without_host_effects(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    for engine in engines:
        engine._dp_wave_running = True
        engine._dp_wave_steps = 13
    issued = []
    time = 0.
    for ordinal in range(3):
        for engine in engines:
            [dummy] = engine.on_schedule(time)
            assert dummy.forward_index == ordinal
            issued.append(dummy)
        time += PERIOD
        if ordinal < 2:
            # These iterations are not synchronization boundaries.
            for engine in engines:
                assert engine._dp_dispatch_host_ready(time)
                engine._iteration_open = False
                assert not engine._end_iteration()
    for engine in engines:
        assert engine.on_schedule(time) == []
    assert all(engine.engine_loop_idle and engine._dp_wave_steps == 0 for engine in engines)
    assert all(engine._paced_dummy_passes == {0, 1, 2} for engine in engines)
    for ordinal in range(3):
        for engine in engines:
            before = (engine._last_iteration_end, engine._dp_wave_steps)
            assert engine.consume_paced_dummy_end(ordinal)
            assert engine._dummy_forwards_in_flight == 0
            assert before == (engine._last_iteration_end, engine._dp_wave_steps)
    # A second end of a drained pass is not paced; the blocking-end path rejects it.
    assert not engines[0].consume_paced_dummy_end(2)
    with pytest.raises(ValueError, match="Unknown DP dummy pass"):
        engines[0].on_dummy_forward_end(time)
    assert engines[0]._dummy_forwards_in_flight == 0


def test_mixed_cohort_dummy_hosts_block_until_device_end(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    engines[0].add_request(Request(0, 32, 2))
    [real] = engines[0].on_schedule(0)
    assert not isinstance(real, DummyForwardBatch)
    for peer in engines[1:]:
        [dummy] = peer.on_schedule(0)
        assert dummy.forward_index == real._forward_cohort_id
    assert all(engine._host_returns_at is None and not engine._paced_dummy_passes for engine in engines)
    for peer in engines[1:]:
        assert peer._dummy_forwards_in_flight == 1
        assert peer.on_schedule(.01) == []
        assert not peer.consume_paced_dummy_end(0)
        assert peer.on_dummy_forward_end(.03)
        assert peer._last_iteration_end == .03 and peer._dummy_forwards_in_flight == 0


def test_dispatch_records_allow_a_real_lane_to_reach_next_ordinal_first(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    engines[0]._record_dp_dispatch(0, 0, is_dummy=False)
    engines[0]._batch_creation_counter = 1  # Real forward 0 was assigned before dispatch.
    engines[0]._issue_dummy_pass(.01)
    for peer in engines[1:]:
        peer._issue_dummy_pass(.005)
    assert all(0 not in engine._dp_dispatch_records for engine in engines)
    assert engines[0]._dp_dispatch_records == {1: (.01, True)}
    for peer in engines[1:]:
        assert peer.on_dummy_forward_end(.02)
        peer._issue_dummy_pass(.02)
    assert all(engine._host_returns_at == .02 + PERIOD for engine in engines)
    assert all(engine._paced_dummy_passes == {1} for engine in engines)


def test_arrival_after_dummy_snapshot_ends_old_wave_then_starts_new_one(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    for engine in engines:
        engine._dp_wave_running = True
        engine.on_schedule(0)
    request = Request(.001, 32, 2)
    engines[0].add_request(request)
    for engine in engines:
        assert engine._dummy_unfinished_snapshot is False
        assert engine.on_schedule(PERIOD) == []
    assert all(not engine._dp_wave_running and engine._dummy_unfinished_snapshot is None for engine in engines)
    [real] = engines[0].on_schedule(PERIOD)
    assert real.requests == [request] and real._forward_cohort_id == 1
    assert all(engine._dp_wave_running for engine in engines)


def test_new_arrival_during_device_drain_joins_idle_lanes_and_completes(monkeypatch):
    source = cluster(monkeypatch, agreement=True)
    execution = Trace(source)
    first = execution.arrival()
    execution.until(lambda: all(not engine._dp_wave_running for engine in lanes(source))
                    and any(engine._paced_dummy_passes for engine in lanes(source)))
    assert first.is_prefill_complete
    assert all(engine.engine_loop_idle for engine in lanes(source))
    assert min(len(engine._paced_dummy_passes) for engine in lanes(source)) >= 3
    arrival_time = execution.time
    next_request = execution.arrival(lane_id=0)
    execution.until(lambda: any(row[4].requests == [next_request] for row in execution.formations))
    [formed] = [row for row in execution.formations if row[4].requests == [next_request]]
    assert formed[0] == arrival_time and formed[2] == 16
    # Peers have no request schedule event supplied here; the existing stage
    # forward agreement wakes each idle peer after the old device work drains.
    execution.until(lambda: next_request.is_prefill_complete)
    assert any(row[1] == 1 and row[2] == 16 and row[3] for row in execution.formations)
    assert next_request.id in lanes(source)[0]._pending_kv_transfer_requests
    execution.finish()
    assert len(execution.metrics.starts) >= 2  # output pop handed both requests to NIXL
    ordinals = [[row[2] for row in execution.formations if row[1] == lane] for lane in range(4)]
    assert all(sequence == list(range(32)) for sequence in ordinals)
    assert execution.wave_ends and max(time for time, _, _, _ in execution.device_ends) > execution.wave_ends[-1]
    for lane in range(4):
        ends = [ordinal for _, owner, ordinal, _ in execution.device_ends if owner == lane]
        assert ends == sorted(ends)


def test_no_profile_matches_original_busy_loop_times(monkeypatch):
    # Recorded by the same PD event harness on the saved, unmodified d47f2ea5
    # scheduler and events. Exact equality protects host formations, device
    # completions, output pops and final request completion on the default path.
    source = cluster(monkeypatch, cadence=None, capacity=None)
    execution = Trace(source)
    request = execution.arrival()
    execution.finish()
    assert [(time, lane, ordinal, dummy) for time, lane, ordinal, dummy, _ in execution.formations] == [
        (0.0, 0, 0, False), (.001, 1, 0, True), (.001, 2, 0, True), (.001, 3, 0, True),
    ]
    assert execution.device_ends == [
        (.36100000000000027, 1, 0, False), (.36100000000000027, 2, 0, False),
        (.36100000000000027, 3, 0, False),
    ]
    assert request.prefill_completed_at == .36100000000000027
    assert [(time, role, lane) for time, _, role, lane in execution.metrics.outputs] == [
        (.36100000000000027, ClusterType.PREFILL, 0),
        (.7222827635200005, ClusterType.DECODE, 0),
        (1.082282763519996, ClusterType.DECODE, 0),
    ]
    assert request.completed and request.completed_at == 1.082282763519996


@pytest.mark.parametrize("role,runs_dummy", [(ClusterType.PREFILL, False), (ClusterType.DECODE, True)])
def test_cadence_requires_prefill_dummy_engine(role, runs_dummy):
    from frontier.scheduler.replica_scheduler.vllm_v1_dp_dispatch import DataParallelDispatchBarrier

    engine = SimpleNamespace(_predictor=SimpleNamespace(get_dp_dummy_step_cadence_ms=lambda: CADENCE_MS),
                             _cluster_type=role, _runs_dp_dummy_passes=runs_dummy)
    with pytest.raises(ValueError, match="requires PREFILL DP dummy passes"):
        DataParallelDispatchBarrier._initialize_dp_dispatch(engine)


def test_duplicate_tick_during_running_wave_counts_only_one_iteration(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    for engine in engines:
        engine._dp_wave_running = True
        engine._dp_wave_steps = 12
        engine.on_schedule(0)
    for engine in engines:
        [next_dummy] = engine.on_schedule(PERIOD)
        assert next_dummy.forward_index == 1 and engine._dp_wave_steps == 13
        assert engine.on_schedule(PERIOD) == []
        assert engine._dp_wave_steps == 13
    assert all(engine._host_returns_at == 2 * PERIOD for engine in engines)


def test_mixed_dummy_pass_also_uses_its_issue_snapshot(monkeypatch):
    engines = lanes(cluster(monkeypatch))
    for engine in engines:
        engine._dp_wave_running = True
    engines[0]._record_dp_dispatch(0, 0, is_dummy=False)
    for peer in engines[1:]:
        peer._issue_dummy_pass(0)
    assert all(peer._dummy_unfinished_snapshot is False for peer in engines[1:])
    engines[1].add_request(Request(.001, 32, 2))
    assert engines[0]._end_iteration()
    for peer in engines[1:]:
        assert peer.on_dummy_forward_end(.03)
        peer._iteration_open = False
        peer._end_iteration()
    assert all(not engine._dp_wave_running and engine._dp_wave_steps == 0 for engine in engines)
    assert all(peer._dummy_unfinished_snapshot is None for peer in engines[1:])
    assert engines[1].has_unfinished_requests
