"""Simulator-only lifecycle for fixed GDN request state slots."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Hashable


@dataclass(frozen=True)
class GatedDeltaNetStateSlot:
    """An ownership token for one request's state reservation."""

    request_id: Hashable
    slot_id: int


class GatedDeltaNetStateSlotManager:
    """Allocate, retain, resume, and release fixed state reservations."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        self._capacity = capacity
        self._slots_by_request: dict[Hashable, GatedDeltaNetStateSlot] = {}
        self._free_slot_ids = list(range(capacity))

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active_request_ids(self) -> tuple[Hashable, ...]:
        return tuple(self._slots_by_request)

    def has_slot(self, request_id: Hashable) -> bool:
        return request_id in self._slots_by_request

    def allocate(self, request_id: Hashable) -> GatedDeltaNetStateSlot:
        if request_id in self._slots_by_request:
            raise ValueError(f"request {request_id!r} already owns a GDN state slot")
        if not self._free_slot_ids:
            raise MemoryError("GDN state slots exhausted")
        slot = GatedDeltaNetStateSlot(request_id, heappop(self._free_slot_ids))
        self._slots_by_request[request_id] = slot
        return slot

    def retain(self, request_id: Hashable) -> GatedDeltaNetStateSlot:
        """Retain a waiting request's slot without changing ownership."""

        try:
            return self._slots_by_request[request_id]
        except KeyError as exc:
            raise KeyError(
                f"no retained GDN state slot for request {request_id!r}"
            ) from exc

    def resume(self, request_id: Hashable) -> GatedDeltaNetStateSlot:
        """Resume a request with its original slot identity."""

        return self.retain(request_id)

    def release(self, request_id: Hashable) -> GatedDeltaNetStateSlot | None:
        slot = self._slots_by_request.pop(request_id, None)
        if slot is None:
            return None
        heappush(self._free_slot_ids, slot.slot_id)
        return slot
