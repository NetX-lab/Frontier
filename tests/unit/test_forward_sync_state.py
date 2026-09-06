from types import SimpleNamespace

import pytest

from frontier.scheduler.utils.forward_sync_state import ForwardSyncState


def make_batch(batch_id, *, step_id=None, provisional_id=None, idle=False):
    batch = SimpleNamespace(
        id=batch_id,
        global_id=batch_id,
        is_idle=idle,
    )
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
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=first,
        lane_id=0,
        layer_id=3,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )
    second_result = state.resolve_step(
        sync_kind="prefill",
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
        sync_kind="decode",
        replica_id=0,
        stage_id=1,
        layer_id=2,
        sync_stage="pre_moe",
        provisional_id=4,
    )

    result = state.resolve_step(
        sync_kind="decode",
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


def test_closed_step_suppresses_late_idle_placeholder():
    state = ForwardSyncState()
    real = make_batch(11, step_id=4, provisional_id=4)
    state.close_step(
        sync_kind="decode",
        replica_id=0,
        stage_id=1,
        layer_id=2,
        sync_stage="pre_moe",
        provisional_id=4,
    )
    late_idle = make_batch(12, step_id=4, provisional_id=4, idle=True)

    result = state.resolve_step(
        sync_kind="decode",
        replica_id=0,
        stage_id=1,
        batch=late_idle,
        lane_id=1,
        layer_id=2,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert result is None
    assert state.open_steps("decode") == {}


def test_closed_step_late_idle_result_is_optional_step_id():
    state = ForwardSyncState()
    batch = make_batch(13, step_id=6, provisional_id=6, idle=True)
    state.close_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        layer_id=1,
        sync_stage="pre_moe",
        provisional_id=6,
    )

    result = state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=batch,
        lane_id=1,
        layer_id=1,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert result is None


def test_idle_placeholder_can_be_replaced_in_open_step():
    state = ForwardSyncState()
    idle = make_batch(20, step_id=9, provisional_id=9, idle=True)
    real = make_batch(21, step_id=9, provisional_id=9)
    rooms = {9: {"batches": {1: idle}}}

    result = state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=real,
        lane_id=1,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )

    assert result == 9


def test_closed_hint_gets_fresh_step_id():
    state = ForwardSyncState()
    first = make_batch(30, step_id=5, provisional_id=5)
    state.close_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        layer_id=1,
        sync_stage="pre_moe",
        provisional_id=5,
    )
    late = make_batch(31, step_id=5, provisional_id=5)

    step_id = state.resolve_step(
        sync_kind="prefill",
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
    prefill = make_batch(40, step_id=2, provisional_id=2)
    decode = make_batch(41, step_id=2, provisional_id=2)
    state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=prefill,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )
    state.close_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        provisional_id=2,
    )

    step_id = state.resolve_step(
        sync_kind="decode",
        replica_id=0,
        stage_id=0,
        batch=decode,
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
        sync_kind="decode",
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
            sync_kind="decode",
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
        sync_kind="decode",
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
            sync_kind="decode",
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
        sync_kind="decode",
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
            sync_kind="decode",
            replica_id=0,
            stage_id=0,
            batch=batch,
            lane_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            room_lookup=room_lookup(rooms),
        )


def test_late_idle_event_does_not_replace_real_batch_in_new_step():
    state = ForwardSyncState()
    real = make_batch(62, step_id=10, provisional_id=10)
    old_idle = make_batch(63, step_id=10, provisional_id=10, idle=True)
    rooms = {10: {"batches": {1: real}}}
    state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=real,
        lane_id=1,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )
    old_idle_step = state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=old_idle,
        lane_id=1,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup(rooms),
    )

    assert old_idle_step == 10
    assert rooms[10]["batches"][1] is real


def test_completed_step_state_stays_bounded():
    state = ForwardSyncState()
    for step_id in range(1000):
        batch = make_batch(step_id, step_id=step_id, provisional_id=step_id)
        resolved = state.resolve_step(
            sync_kind="decode",
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
            sync_kind="decode",
            replica_id=0,
            stage_id=0,
            layer_id=0,
            sync_stage="pre_moe",
            provisional_id=step_id,
        )

    assert state.open_steps("decode") == {}
    assert state._next_step_id_by_replica == {0: 1000}
    assert set(state.__dict__) == {
        "_open_steps_by_kind",
        "_next_step_id_by_replica",
    }
