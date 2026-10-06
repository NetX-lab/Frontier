from types import SimpleNamespace

import pytest

from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils.sync_state import initialize_sync_waiting_rooms


@pytest.mark.parametrize("sync_kind", ["prefill", "decode"])
def test_a_lone_lane_wakes_its_idle_sibling_engine_and_waits_for_its_dummy_forward(sync_kind):
    from frontier.entities import Batch, DummyForwardBatch, Request
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent
    from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
        RoundRobinClusterScheduler,
    )
    from frontier.types import ClusterType

    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = getattr(ClusterType, sync_kind.upper())
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(model_config=SimpleNamespace(is_moe=True))
    )
    initialize_sync_waiting_rooms(scheduler)
    scheduler._forward_sync_state = ForwardSyncState()
    scheduler._replica_dp_size = 2
    setattr(scheduler, f"_uses_shared_{sync_kind}_layer_path", lambda *_: True)
    sibling_engine = SimpleNamespace(engine_loop_idle=True)
    scheduler._replica_schedulers = {
        (0, 0): SimpleNamespace(engine_loop_idle=False),
        (0, 1): sibling_engine,
    }
    batch = Batch(0, [Request(0.0, 16, 4)], [16], is_moe=True)
    dummy = DummyForwardBatch(0, 0, forward_index=0)
    for lane_id, lane_batch in enumerate((batch, dummy)):
        lane_batch.set_global_id(lane_id)
        lane_batch._forward_cohort_id = 0
        lane_batch._forward_cohort_provisional_id = 0
    # The woken engine starts its stage forward after the lone lane did.
    batch._forward_launch_start_time = 0.0
    dummy._forward_launch_start_time = 0.5
    completed = []

    def wave_ready(**kwargs):
        completed.append(kwargs["layer_id"])
        assert kwargs["cohort_batches"] == {0: batch, 1: dummy}
        return []

    setattr(scheduler, f"_on_{sync_kind}_ep_wave_ready", wave_ready)
    sync = getattr(scheduler, f"on_{sync_kind}_sync")
    for layer_id in range(4):
        events = sync(float(layer_id), 0, 0, batch, 0, "pre_moe", layer_id, 0.0)
        if layer_id == 0:
            assert [type(event) for event in events] == [ReplicaScheduleEvent]
            assert events[0]._replica_local_id == 1
            # The woken engine issues its dummy forward and is no longer idle.
            sibling_engine.engine_loop_idle = False
        else:
            assert events == []
        assert completed == list(range(layer_id))
        assert sync(float(layer_id), 0, 0, dummy, 1, "pre_moe", layer_id, 0.0) == []
        assert completed == list(range(layer_id + 1))
        # vLLM's per-forward DP all-reduce: both lanes launch from the later start.
        assert batch._forward_launch_start_time == dummy._forward_launch_start_time == 0.5
        assert scheduler._forward_sync_state._open_steps == {}
        assert batch._forward_cohort_id == dummy._forward_cohort_id == layer_id


def make_batch(batch_id, *, step_id=None, provisional_id=None):
    batch = SimpleNamespace(id=batch_id, global_id=batch_id)
    if step_id is not None:
        batch._forward_cohort_id = step_id
    if provisional_id is not None:
        batch._forward_cohort_provisional_id = provisional_id
    return batch


def room_lookup(rooms):
    return lambda step_id: rooms.get(step_id)


def test_sibling_lanes_join_one_open_step():
    state = ForwardSyncState()
    rooms = {7: {"batches": {0: make_batch(1)}}}
    first = make_batch(1, step_id=7, provisional_id=7)
    second = make_batch(2, step_id=7, provisional_id=7)

    first_result = state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=first,
        lane_id=0,
        layer_id=3,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )
    second_result = state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=second,
        lane_id=1,
        layer_id=3,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )

    assert first_result == 7
    assert second_result == 7
    assert second._forward_cohort_id == 7


def test_late_real_batch_gets_a_fresh_step_after_step_close():
    state = ForwardSyncState()
    batch = make_batch(11, step_id=4, provisional_id=4)
    state.close_step(
        replica_id=0,
        stage_id=1,
        layer_id=2,
        sync_stage="pre_moe",
        provisional_id=4,
    )

    result = state.resolve_step(
        replica_id=0,
        stage_id=1,
        batch=batch,
        lane_id=0,
        layer_id=2,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert result == 5
    assert batch._forward_cohort_id == 5


def test_closed_hint_gets_fresh_step_id():
    state = ForwardSyncState()
    first = make_batch(30, step_id=5, provisional_id=5)
    state.close_step(
        replica_id=0,
        stage_id=0,
        layer_id=1,
        sync_stage="pre_moe",
        provisional_id=5,
    )
    late = make_batch(31, step_id=5, provisional_id=5)

    step_id = state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=late,
        lane_id=1,
        layer_id=1,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert step_id == 6
    assert late._forward_cohort_id == 6


def test_prefill_and_decode_share_monotonic_identity_allocator():
    state = ForwardSyncState()
    first = make_batch(40, step_id=2, provisional_id=2)
    second = make_batch(41, step_id=2, provisional_id=2)
    state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=first,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )
    state.close_step(
        replica_id=0,
        stage_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        provisional_id=2,
    )

    step_id = state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=second,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert step_id == 3


def test_conflicting_live_lane_fails_fast():
    state = ForwardSyncState()
    first = make_batch(50, step_id=3, provisional_id=3)
    second = make_batch(51, step_id=3, provisional_id=3)
    rooms = {3: {"batches": {0: first}}}
    state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=first,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )

    with pytest.raises(ValueError, match="two open sync cohorts"):
        state.resolve_step(
            replica_id=0,
            stage_id=0,
            batch=second,
            lane_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            room_lookup=room_lookup(rooms),
        )


def test_open_step_with_missing_room_fails_fast():
    state = ForwardSyncState()
    batch = make_batch(60, step_id=3, provisional_id=3)
    state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=batch,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup({3: {"batches": {}}}),
    )

    with pytest.raises(RuntimeError, match="missing waiting room"):
        state.resolve_step(
            replica_id=0,
            stage_id=0,
            batch=batch,
            lane_id=1,
            layer_id=0,
            sync_stage="pre_moe",
            room_lookup=room_lookup({}),
        )


def test_same_batch_reentry_fails_fast():
    state = ForwardSyncState()
    batch = make_batch(61, step_id=4, provisional_id=4)
    rooms = {4: {"batches": {0: batch}}}
    state.resolve_step(
        replica_id=0,
        stage_id=0,
        batch=batch,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )

    with pytest.raises(ValueError, match="two open sync cohorts"):
        state.resolve_step(
            replica_id=0,
            stage_id=0,
            batch=batch,
            lane_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            room_lookup=room_lookup(rooms),
        )


def test_completed_step_state_stays_bounded():
    state = ForwardSyncState()
    for step_id in range(1000):
        batch = make_batch(step_id, step_id=step_id, provisional_id=step_id)
        resolved = state.resolve_step(
            replica_id=0,
            stage_id=0,
            batch=batch,
            lane_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            room_lookup=room_lookup({step_id: {"batches": {}}}),
        )
        assert resolved == step_id
        state.close_step(
            replica_id=0,
            stage_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            provisional_id=step_id,
        )

    assert state._open_steps == {}
    assert state._next_step_id_by_replica == {0: 1000}
    assert set(state.__dict__) == {
        "_open_steps",
        "_next_step_id_by_replica",
    }
