"""TDD coverage for simulator-only GDN state slot ownership."""

import pytest

from frontier.attention.gdn.state import GatedDeltaNetStateSlotManager


def test_state_slots_allocate_wait_resume_and_release_without_tensor_storage():
    manager = GatedDeltaNetStateSlotManager(capacity=2)
    first = manager.allocate("request-1")
    assert first.request_id == "request-1"
    assert first.slot_id == 0
    assert manager.has_slot("request-1")

    manager.retain("request-1")
    assert manager.resume("request-1") is first
    second = manager.allocate("request-2")
    assert second.slot_id == 1
    with pytest.raises(MemoryError, match="state slots exhausted"):
        manager.allocate("request-3")
    manager.release("request-1")
    third = manager.allocate("request-3")
    assert third.slot_id == 0
    assert manager.active_request_ids == ("request-2", "request-3")
    assert not hasattr(first, "tensor")


def test_state_slot_lifecycle_is_idempotent_for_waiting_and_release():
    manager = GatedDeltaNetStateSlotManager(capacity=1)
    slot = manager.allocate(7)
    assert manager.retain(7) is slot
    assert manager.retain(7) is slot
    assert manager.release(7) is slot
    assert manager.release(7) is None
    with pytest.raises(KeyError, match="no retained GDN state slot"):
        manager.resume(7)


def test_state_slot_reuse_preserves_lowest_available_id():
    manager = GatedDeltaNetStateSlotManager(capacity=5)
    for request_id in range(4):
        assert manager.allocate(request_id).slot_id == request_id
    for request_id in (3, 0, 2):
        manager.release(request_id)
    assert [manager.allocate(request_id).slot_id for request_id in (5, 6, 7, 8)] == [0, 2, 3, 4]
