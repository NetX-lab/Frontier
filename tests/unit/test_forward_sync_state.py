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

    assert first_result == (7, False)
    assert second_result == (7, False)
    assert second._forward_cohort_id == 7


def test_completed_event_is_suppressed_for_same_batch():
    state = ForwardSyncState()
    batch = make_batch(11, step_id=4, provisional_id=4)
    state.close_step(
        sync_kind="decode",
        replica_id=0,
        stage_id=1,
        layer_id=2,
        sync_stage="pre_moe",
        provisional_id=4,
        step_id=4,
        source_batches={0: batch},
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

    assert result == (4, True)


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

    assert result == (9, False)


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
        step_id=5,
        source_batches={0: first},
    )
    late = make_batch(31, step_id=5, provisional_id=5)

    step_id, duplicate = state.resolve_step(
        sync_kind="prefill",
        replica_id=0,
        stage_id=0,
        batch=late,
        lane_id=1,
        layer_id=1,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert (step_id, duplicate) == (6, False)
    assert late._forward_cohort_id == 6


def test_prefill_and_decode_keep_separate_completion_maps():
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
        step_id=2,
        source_batches={0: prefill},
    )

    step_id, duplicate = state.resolve_step(
        sync_kind="decode",
        replica_id=0,
        stage_id=0,
        batch=decode,
        lane_id=0,
        layer_id=0,
        sync_stage="pre_moe",
        room_lookup=room_lookup({}),
    )

    assert (step_id, duplicate) == (3, False)


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
