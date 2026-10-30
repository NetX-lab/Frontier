"""Host iteration, device FIFO and DP rendezvous evidence for opt-in run-ahead."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.config import RoundRobinClusterSchedulerConfig, VllmLoadBalancingClusterSchedulerConfig, VllmV1SchedulerConfig, global_vars
from frontier.entities import DummyForwardBatch, Request
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler
from frontier.scheduler.utils import ep_wave_inputs
from frontier.simulator import Simulator
from frontier.types import ClusterType
from tests.integration.test_vllm_dp_placement_runtime import _config
from tests.unit.test_vllm_dp_load_balancer import _policy_scheduler, _request


@pytest.fixture(autouse=True)
def reset_globals():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _simulator(root, patch, depth, *, arrivals=(0.0,), pp=1, moe=True, dp=2,
               policy=RoundRobinClusterSchedulerConfig):
    root.mkdir(parents=True, exist_ok=True)
    trace = root / "trace.csv"
    with trace.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arrived_at", "num_prefill_tokens", "num_decode_tokens"))
        writer.writerows((time, 8, 2) for time in arrivals)
    config = _config(root, patch, is_moe=moe, attn_dp=dp, moe_ep=dp if moe else 1,
                     policy=policy, trace=str(trace),
                     num_pipeline_stages=pp, num_layers=2 * pp,
                     analytical_backend=True)
    config.cluster_config.replica_scheduler_config.max_forwards_in_flight = depth
    sim = Simulator(config)
    cluster = sim._global_scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
    replica_id = next(iter(cluster._cluster.replicas))
    lanes = [cluster.get_replica_scheduler(replica_id, index) for index in range(dp)]
    return sim, cluster, lanes


def _run(root, depth, arrivals=(0.0, 0.1, 0.2)):
    """Observe actual formation, host ends, stage ends and every EP cohort."""
    forwards, iterations, attempts, rooms = {}, [], [], []
    now = [0.0]
    with pytest.MonkeyPatch.context() as patch:
        sim, _, lanes = _simulator(root, patch, depth, arrivals=arrivals)
        original_schedule = VLLMv1EngineReplicaScheduler.on_schedule
        original_end = VLLMv1EngineReplicaScheduler._end_iteration
        original_device_end = BatchStageEndEvent.handle_event
        original_room = ep_wave_inputs.prepare_ep_wave_inputs

        def schedule(lane, time):
            now[0] = time
            prior = lane._batch_creation_counter
            waits = lane._host_forward_wait if depth is not None else bool(lane._dummy_forwards_in_flight)
            batches = original_schedule(lane, time)
            attempts.append(dict(lane=lane._replica_local_id, time=time, prior=prior,
                                 waits=waits, formed=len(batches)))
            for batch in batches:
                forwards[batch.id] = dict(
                    lane=lane._replica_local_id, index=batch._forward_cohort_provisional_id,
                    formed=time, dummy=isinstance(batch, DummyForwardBatch),
                    requests=[request.id for request in batch.requests],
                    idle=batch.engine_idle_time,
                    ahead=(next(i for i, f in enumerate(lane._unfinished_forwards)
                                if f.batch_id == batch.id) if depth is not None else 0),
                )
            return batches

        def iteration_end(lane):
            iterations.append(dict(lane=lane._replica_local_id, time=now[0],
                                   count=lane._batch_creation_counter,
                                   unfinished=(len(lane._unfinished_forwards)
                                               if depth is not None else lane._dummy_forwards_in_flight),
                                   peers=[peer._batch_creation_counter for peer in lanes]))
            return original_end(lane)

        def device_end(event, *args):
            row = forwards[event._batch.id]
            row.update(start=event._batch_stage.scheduled_at, end=event.time,
                       cohort=event._batch._forward_cohort_id)
            return original_device_end(event, *args)

        def room(*args, **kwargs):
            result = original_room(*args, **kwargs)
            sources = result.source_batches
            rooms.append([(lane, batch._forward_cohort_provisional_id,
                           batch._forward_cohort_id) for lane, batch in sorted(sources.items())])
            return result

        patch.setattr(VLLMv1EngineReplicaScheduler, "on_schedule", schedule)
        patch.setattr(VLLMv1EngineReplicaScheduler, "_end_iteration", iteration_end)
        patch.setattr(BatchStageEndEvent, "handle_event", device_end)
        patch.setattr(ep_wave_inputs, "prepare_ep_wave_inputs", room)
        sim.run()
        requests = [dict(id=req.id, arrival=req.arrived_at, completed=req.completed)
                    for req in sim._all_requests]
        final = [dict(unfinished=(len(lane._unfinished_forwards) if depth is not None
                                  else lane._dummy_forwards_in_flight),
                      wave=lane._dp_wave_running, idle=lane.engine_loop_idle) for lane in lanes]
    result = dict(forwards=list(forwards.values()), iterations=iterations,
                  attempts=attempts, rooms=rooms, requests=requests, final=final)
    (root / "evidence.json").write_text(json.dumps(result, indent=1))
    return result


@pytest.mark.parametrize("depth", [None, 1, 19])
def test_valid_depth(depth):
    assert VllmV1SchedulerConfig(max_forwards_in_flight=depth).max_forwards_in_flight == depth
    assert VllmV1SchedulerConfig.__dataclass_fields__["max_forwards_in_flight"].metadata["omit_if_none"]


@pytest.mark.parametrize("depth", [0, -1, True, 2.0, "19"])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="max_forwards_in_flight"):
        VllmV1SchedulerConfig(max_forwards_in_flight=depth)


@pytest.mark.parametrize("shape,reason", [({"moe": False, "dp": 1}, "dummy passes"),
                                         ({"dp": 1}, "dummy passes"),
                                         ({"pp": 2}, "num_pipeline_stages=1")])
def test_constructor_rejects_unmeasured_lanes(tmp_path, shape, reason):
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(ValueError, match=f"max_forwards_in_flight.*{reason}"):
            _simulator(tmp_path, patch, 19, **shape)


@pytest.mark.parametrize("depth", [1, 3, 19])
def test_actual_runtime_bounds_host_returns_pairs_cohorts_and_drains(tmp_path, depth):
    run = _run(tmp_path, depth)
    assert all(req["completed"] for req in run["requests"])
    assert run["final"] == [dict(unfinished=0, wave=False, idle=True)] * 2
    assert max(row["unfinished"] for row in run["iterations"]) == depth
    assert all(row["unfinished"] <= depth and min(row["peers"]) >= row["count"]
               for row in run["iterations"])
    assert run["rooms"]
    assert all(len(room) == 2 and len({index for _, index, _ in room}) == 1
               and len({cohort for _, _, cohort in room}) == 1 for room in run["rooms"])
    for lane_id in range(2):
        rows = sorted((row for row in run["forwards"] if row["lane"] == lane_id),
                      key=lambda row: row["index"])
        assert [row["index"] for row in rows] == list(range(len(rows)))
        assert all(after["start"] >= before["end"] for before, after in zip(rows, rows[1:]))
        # Once n-K ends, the host leaves n and forms n+1 at that same time.
        for ended in run["iterations"]:
            if ended["lane"] == lane_id and ended["count"] > depth:
                oldest = rows[ended["count"] - depth - 1]
                assert ended["time"] >= oldest["end"]


def _host_pass(sim, lane, time):
    event = ReplicaScheduleEvent(time, lane._replica_id, lane._cluster_type, lane._replica_local_id)
    events = event.handle_event(sim._global_scheduler, sim._metric_store)
    return event._batches, [ev for ev in events if isinstance(ev, ReplicaScheduleEvent)]


def _drive_hosts(sim, lanes, time, rounds=1):
    formed = []
    for _ in range(rounds):
        for lane in lanes:
            batches, _ = _host_pass(sim, lane, time)
            formed.extend((lane, batch) for batch in batches)
    return formed


def test_idle_hosts_form_then_wait_and_launch_one_more_per_device_end(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        sim, _, lanes = _simulator(tmp_path, patch, 3, arrivals=(100.0,))
        for lane in lanes:
            lane._dp_wave_running = True
        formed = _drive_hosts(sim, lanes, 0.0, rounds=5)
        # n is scheduled before the launch wait: K+1 can be formed, but only
        # K iterations return before forward n-K ends.
        assert [lane._batch_creation_counter for lane in lanes] == [4, 4]
        assert [lane._dp_wave_steps for lane in lanes] == [3, 3]
        assert all(lane.on_schedule(0.5) == [] for lane in lanes)
        for lane in lanes:
            assert lane.on_dummy_forward_end(1.0)
        more = _drive_hosts(sim, lanes, 1.0, rounds=2)
        assert len(more) == 2
        assert [lane._batch_creation_counter for lane in lanes] == [5, 5]
        assert [lane._dp_wave_steps for lane in lanes] == [4, 4]
        assert len(formed) == 8


def test_wave_start_wakes_peers_and_batch_wait_stops_idle_lane_one_ahead(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        sim, _, (busy, idle) = _simulator(tmp_path, patch, 19, arrivals=(100.0,))
        request = Request(0.0, 8, 2)
        busy.add_request(request)
        [batch], wakes = _host_pass(sim, busy, 0.0)
        assert [event._replica_local_id for event in wakes] == [idle._replica_local_id]
        assert busy._blocking_batch_id == batch.id
        _drive_hosts(sim, [idle], 0.0, rounds=3)
        assert (busy._batch_creation_counter, idle._batch_creation_counter) == (1, 2)
        assert idle.on_schedule(0.5) == []
        # Device completion alone does not release the batch-output block.
        busy.on_device_forward_end(batch)
        assert busy.on_schedule(1.0) == []
        batch.on_batch_end(1.0)
        busy.on_batch_end(batch)
        _, wakes = _host_pass(sim, busy, 1.0)
        assert busy._batch_creation_counter == 2
        assert idle._replica_local_id in [event._replica_local_id for event in wakes]
        _host_pass(sim, idle, 1.0)
        assert idle._batch_creation_counter == 3


@pytest.mark.parametrize("wait", ["depth", "peer"])
def test_arrival_during_host_wait_belongs_to_the_following_forward(tmp_path, wait):
    with pytest.MonkeyPatch.context() as patch:
        sim, _, lanes = _simulator(tmp_path, patch, 1 if wait == "depth" else 19,
                                   arrivals=(100.0,))
        for lane in lanes:
            lane._dp_wave_running = True
        if wait == "depth":
            formed = _drive_hosts(sim, lanes, 0.0, rounds=3)
        else:
            formed = _drive_hosts(sim, [lanes[0]], 0.0)
        target = lanes[0]
        waited_count = target._batch_creation_counter
        request = Request(0.5, 8, 2)
        target.add_request(request)
        assert target.on_schedule(0.5) == []
        assert all(request not in batch.requests for _, batch in formed)
        if wait == "depth":
            for lane in lanes:
                lane.on_dummy_forward_end(1.0)
        else:
            _host_pass(sim, lanes[1], 1.0)
        batches, _ = _host_pass(sim, target, 1.0)
        [batch] = [batch for batch in batches if not isinstance(batch, DummyForwardBatch)]
        assert request in batch.requests
        assert batch._forward_cohort_provisional_id == waited_count


def test_wave_ends_with_device_work_and_next_prefill_queues_behind_it(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        sim, _, lanes = _simulator(tmp_path, patch, 19, arrivals=(100.0,))
        for lane in lanes:
            lane._dp_wave_running = True
        _drive_hosts(sim, lanes, 0.0, rounds=21)
        for device_end in range(1, 14):
            for lane in lanes:
                lane.on_dummy_forward_end(float(device_end))
            _drive_hosts(sim, lanes, float(device_end), rounds=2)
        assert all(not lane._dp_wave_running and lane.engine_loop_idle for lane in lanes)
        assert all(lane._dummy_forwards_in_flight == 19 for lane in lanes)
        request = Request(14.0, 8, 2)
        lanes[0].add_request(request)
        [batch], _ = _host_pass(sim, lanes[0], 14.0)
        assert batch._forward_cohort_provisional_id == 32
        assert len(lanes[0]._unfinished_forwards) == 20
    # The actual DES must preserve this queue on the stage.
    run = _run(tmp_path / "device", 19, arrivals=(0.0, 0.12, 0.24, 0.36))
    queued = []
    for lane in (0, 1):
        rows = sorted((row for row in run["forwards"] if row["lane"] == lane),
                      key=lambda row: row["index"])
        for previous, row in zip(rows, rows[1:]):
            prior_group = [peer for peer in run["forwards"]
                           if peer["index"] == previous["index"]]
            if not row["dummy"] and row["ahead"] and all(peer["dummy"] for peer in prior_group):
                queued.append(row)
                assert row["start"] == pytest.approx(previous["end"], abs=1e-9)
    assert queued


def test_nixl_report_and_idle_clock_follow_host_end_and_zero_cost_jumps(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        sim, cluster, original_lanes = _simulator(tmp_path, patch, 1, arrivals=(100.0,))
        lanes = []
        for old in original_lanes:
            lane = VLLMv1EngineReplicaScheduler(
                replica_config=old._replica_config, replica_scheduler_config=old._config,
                request_generator_config=sim._config.request_generator_config,
                replica=old._replica, predictor=old._predictor,
                cluster_scheduler=cluster, cluster_type=ClusterType.DECODE,
                replica_local_id=old._replica_local_id,
            )
            cluster._replica_schedulers[(old._replica_id, old._replica_local_id)] = lane
            lane._dp_wave_running = True
            lanes.append(lane)
        sim._global_scheduler._cluster_schedulers[ClusterType.DECODE] = cluster
        read = SimpleNamespace(request_id=1, transfer_time_ms=4.0)
        lanes[0].register_nixl_read(read)
        # Scheduling claims and formation posts at host time zero, even with
        # device work from an earlier dummy still unfinished.
        _drive_hosts(sim, lanes, 0.0)
        assert lanes[0]._nixl_reads_posted == [(0.004, read)]
        _drive_hosts(sim, lanes, 0.001, rounds=2)
        assert all(lane._host_forward_wait for lane in lanes)
        for lane in lanes:
            lane.on_dummy_forward_end(0.004)
        _drive_hosts(sim, lanes, 0.004, rounds=2)
        # The next pass found the READ at formation, but its n-K wait still
        # holds the host, so it cannot report or advance the idle clock yet.
        assert lanes[0].consume_found_nixl_reads() == []
        assert lanes[0]._nixl_reads_found_by_dummy_forward == [read]
        assert lanes[0]._last_iteration_end == 0.004
        for lane in lanes:
            lane.on_dummy_forward_end(0.006)
        # Use on_schedule directly to observe reports before the event drains
        # them through the real KV-transfer completion handler.
        count = lanes[0]._batch_creation_counter
        assert lanes[0].on_schedule(0.006) == []
        assert lanes[0]._batch_creation_counter == count
        assert lanes[0].consume_found_nixl_reads() == [read]
        assert lanes[0]._last_iteration_end == 0.006

    with pytest.MonkeyPatch.context() as patch:
        sim, _, lanes = _simulator(tmp_path / "zero", patch, 19, arrivals=(100.0,))
        for lane in lanes:
            lane._dp_wave_running = True
        lanes[0]._nixl_reads_posted = [(1.0, read)]
        _drive_hosts(sim, lanes, 0.0, rounds=21)
        assert [lane._batch_creation_counter for lane in lanes] == [20, 20]
        # Fast host returns must not classify a positive-cost device forward
        # as zero. Only actual device ends at formation time enable the jump.
        assert all(lane.next_pass_start(0.0) == 0.0 for lane in lanes)
        for lane in lanes:
            lane.on_dummy_forward_end(0.0)
        for lane in lanes:
            _, events = _host_pass(sim, lane, 0.0)
            assert any(event.time == 1.0 for event in events)
            assert lane._last_iteration_end == 0.0
        assert [lane._batch_creation_counter for lane in lanes] == [20, 20]



def test_real_forward_posts_reads_at_formation_and_not_again_at_device_start(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        sim, cluster, old_lanes = _simulator(tmp_path, patch, 19, arrivals=(100.0,))
        old = old_lanes[0]
        lane = VLLMv1EngineReplicaScheduler(
            replica_config=old._replica_config, replica_scheduler_config=old._config,
            request_generator_config=sim._config.request_generator_config,
            replica=old._replica, predictor=old._predictor,
            cluster_scheduler=cluster, cluster_type=ClusterType.DECODE,
            replica_local_id=old._replica_local_id,
        )
        cluster._replica_schedulers[(old._replica_id, old._replica_local_id)] = lane
        lane._dp_wave_running = True
        [dummy] = lane.on_schedule(0.0)
        old_lanes[1]._batch_creation_counter = lane._batch_creation_counter
        request = Request(0.001, 8, 2)
        request._num_processed_tokens = 8
        request._is_prefill_complete = True
        lane.add_request(request)
        read = SimpleNamespace(request_id=99, transfer_time_ms=4.0)
        lane.register_nixl_read(read)
        [batch] = lane.on_schedule(0.001)
        assert not isinstance(batch, DummyForwardBatch)
        assert lane._unfinished_forwards[0].batch_id == dummy.id
        assert lane._nixl_reads_posted == [(0.005, read)]
        # The stage invocation does not find or repost READs already owned by
        # the formed host pass, even after their wire time has ended.
        lane.on_engine_step_start(0.020, batch)
        assert lane._nixl_reads_posted == [(0.005, read)]
        assert lane._nixl_reads_found_by_batch == {}


def test_run_ahead_load_reports_follow_formed_forwards_and_held_completions(tmp_path):
    """Host steps own reports even while older forwards occupy device rooms."""
    records = []
    with pytest.MonkeyPatch.context() as patch:
        sim, cluster, lanes = _simulator(
            tmp_path, patch, 19, arrivals=(0.0, 0.12, 0.24, 0.36),
            policy=VllmLoadBalancingClusterSchedulerConfig,
        )
        original_admit = cluster.on_replica_batch_scheduled
        original_complete = cluster.on_replica_batch_end
        original_report = cluster._load_balancer.report
        active = []

        def report(time, lane_id, key, load):
            balancer = cluster._load_balancer
            row = active[-1]
            row.update(key=key, previous_step=balancer.last_report_step,
                       changed=load != balancer.engine_counts[lane_id])
            original_report(time, lane_id, key, load)
            row.update(last_step=balancer.last_report_step,
                       applied=balancer.engine_counts[lane_id] == load)

        def observe(kind, original, time, replica_id, lane_id, batch):
            lane = lanes[lane_id]
            held = cluster._held_key[lane_id]
            row = dict(kind=kind, time=time, lane=lane_id, held=held,
                       ahead=len(lane._unfinished_forwards),
                       expected=(batch._forward_cohort_provisional_id
                                 if kind == "admission" else
                                 held if held is not None else lane._batch_creation_counter))
            active.append(row)
            original(time, replica_id, lane_id, batch)
            active.pop()
            row["held_after"] = cluster._held_key[lane_id]
            records.append(row)

        def admit(time, replica_id, lane_id, batch):
            observe("admission", original_admit, time, replica_id, lane_id, batch)

        def complete(time, replica_id, lane_id, batch):
            observe("completion", original_complete, time, replica_id, lane_id, batch)

        patch.setattr(cluster._load_balancer, "report", report)
        patch.setattr(cluster, "on_replica_batch_scheduled", admit)
        patch.setattr(cluster, "on_replica_batch_end", complete)
        sim.run()
        assert all(request.completed for request in sim._all_requests)
        assert cluster._held_key == [None, None]

    (tmp_path / "load_reports.json").write_text(json.dumps(records, indent=1))
    for lane_id in range(2):
        admissions = [row for row in records
                      if row["kind"] == "admission" and row["lane"] == lane_id]
        assert len(admissions) >= 2
        assert all(after["expected"] > before["expected"]
                   for before, after in zip(admissions, admissions[1:]))
    for row in records:
        if row["kind"] == "admission":
            # PP1's real batch queue fills even when K's device queue has room.
            assert "key" not in row
            assert row["held_after"] == row["expected"]
        else:
            assert row["key"] == row["expected"]
            assert row["held_after"] is None
    assert any(row["kind"] == "admission" and row["ahead"] > 1 for row in records)
    assert any(row["kind"] == "completion" and row["held"] is not None for row in records)


def test_run_ahead_decode_queue_reports_host_keys_and_keeps_stale_counts(tmp_path, caplog) -> None:
    """Native batch_queue_size owns publication; K owns only device capacity.

    coordinator.py:402-415 applies an older report's counts without moving the
    latest step boundary. This intentionally preserves the native stale rule.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        scheduler = _policy_scheduler(
            patch, cluster_type=ClusterType.DECODE,
            replica_scheduler_config=VllmV1SchedulerConfig(
                max_forwards_in_flight=19, max_concurrent_batches=2,
                num_blocks=128, block_size=16, batch_size_cap=4,
                max_tokens_in_batch=16, enable_chunked_prefill=True,
            ),
        )
        replica_id = scheduler._serving_replica_id
        lanes = [scheduler.get_replica_scheduler(replica_id, lane_id)
                 for lane_id in range(2)]
        balancer = scheduler._load_balancer
        reports = []
        original_report = balancer.report

        def report(time, lane_id, key, load):
            reports.append((lane_id, key))
            original_report(time, lane_id, key, load)

        patch.setattr(balancer, "report", report)
        for lane in lanes:
            lane._dp_wave_running = True
        for _ in range(3):
            for lane in lanes:
                lane.on_schedule(0.0)

        def add_decode_request(lane, time):
            request = _request()
            request._num_processed_tokens = request.num_prefill_tokens
            request._is_prefill_complete = True
            request._prefill_completed_at = 0.0
            request.on_arrival(time, ClusterType.DECODE)
            lane.add_request(request)
            return request

        requests = [add_decode_request(lane, 0.1) for lane in lanes]
        first = [lane.on_schedule(0.1)[0] for lane in lanes]
        assert [batch._forward_cohort_provisional_id for batch in first] == [3, 3]
        # Three older dummy forwards remain queued; the first real batch still
        # leaves an output slot, so both lanes publish its host key immediately.
        assert all(len(lane._unfinished_forwards) == 4 for lane in lanes)
        assert reports == [(0, 3), (1, 3)]
        assert scheduler._held_key == [None, None]
        second = [lane.on_schedule(0.1)[0] for lane in lanes]
        assert [batch._forward_cohort_provisional_id for batch in second] == [4, 4]
        assert scheduler._held_key == [4, 4]
        assert reports == [(0, 3), (1, 3)]
        assert all(len(lane._unfinished_forwards) == 5 for lane in lanes)
        for time in (0.2, 0.3, 0.4):
            for lane in lanes:
                lane.on_dummy_forward_end(time)

        def complete(lane_id, batch, time):
            lane = lanes[lane_id]
            lane.on_device_forward_end(batch)
            batch.on_batch_end(time, ClusterType.DECODE)
            lane.on_batch_end(batch)
            scheduler.on_replica_batch_end(time, replica_id, lane_id, batch)

        complete(0, first[0], 0.5)
        assert scheduler._held_key == [None, 4]
        assert reports[-1] == (0, 4)
        complete(0, second[0], 0.6)
        # Without a held admission, completion reports the next host forward,
        # even though no device room was started by this test.
        assert reports[-1] == (0, 5)
        assert requests[0].completed
        assert balancer.last_report_step == 5
        add_decode_request(lanes[1], 0.65)
        complete(1, first[1], 0.7)
        assert reports[-1] == (1, 4)
        assert scheduler._held_key == [None, None]
        assert "out-of-order step 4" in caplog.text
        assert balancer.last_report_step == 5
        assert balancer.engine_counts[1] == lanes[1].get_request_load()
        assert balancer.engine_counts[1].waiting == 1
        complete(1, second[1], 0.8)
        assert reports[-1] == (1, 5)
        assert requests[1].completed
        assert all(not lane._unfinished_forwards for lane in lanes)


def test_unset_forward_times_match_read_only_base(tmp_path):
    base = Path("/data/ycfeng/Frontier/.worktrees/b26-base")
    script = Path(__file__).resolve()
    # Load the test observer from this file while production imports resolve
    # only against the chosen checkout; the base has no new config field.
    runner = tmp_path / "baseline.py"
    runner.write_text("import importlib.util, sys\n"
                      f"spec=importlib.util.spec_from_file_location('run_ahead_checks', {str(script)!r})\n"
                      "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                      "module._run(__import__('pathlib').Path(sys.argv[1]), None)\n")
    import os
    for name, repo in (("base", base), ("current", script.parents[2])):
        result = subprocess.run([sys.executable, str(runner), str(tmp_path / name)],
                                cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    before = json.loads((tmp_path / "base/evidence.json").read_text())
    after = json.loads((tmp_path / "current/evidence.json").read_text())
    assert [(row["lane"], row["index"], row["start"], row["end"])
            for row in before["forwards"]] == [
                (row["lane"], row["index"], row["start"], row["end"])
                for row in after["forwards"]]
