"""Stage-local admission ownership for dense work and MoE EP waves."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass


FULL_STAGE_WORLD = "FULL_STAGE_WORLD"
EP_WAVE = "EP_WAVE"
_IDLE = "IDLE"


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an exact non-negative int")
    return value


@dataclass(frozen=True)
class StageAdmissionTicket:
    """Immutable operation identity assigned by one stage context."""

    replica_id: int
    stage_id: int
    admission_seq: int
    operation_id: Hashable
    scope: str
    participant_ep_ids: tuple[int, ...]


class StageExecutionContext:
    """Own one physical pipeline stage's mutually-exclusive operation scope.

    The context is intentionally independent of event timing and child lane
    queues.  A complete operation first enters the ready FIFO, then the owner
    admits it atomically.  EP child schedulers may start only after their
    wave's ticket has been acquired, and the ticket remains active through the
    wave-level combine/cleanup boundary.
    """

    def __init__(self, *, replica_id: int, stage_id: int, ep_size: int) -> None:
        self._replica_id = _require_non_negative_int(replica_id, "replica_id")
        self._stage_id = _require_non_negative_int(stage_id, "stage_id")
        self._ep_size = _require_non_negative_int(ep_size, "ep_size")
        if self._ep_size <= 0:
            raise ValueError("ep_size must be an exact positive int")

        self._next_admission_seq = 0
        self._ready_fifo: deque[StageAdmissionTicket] = deque()
        self._active_ticket: StageAdmissionTicket | None = None
        self._operation_ids: set[Hashable] = set()

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def ep_size(self) -> int:
        return self._ep_size

    @property
    def is_idle(self) -> bool:
        return self._active_ticket is None

    @property
    def active_scope(self) -> str | None:
        return None if self._active_ticket is None else self._active_ticket.scope

    @property
    def active_operation_id(self) -> Hashable | None:
        return (
            None
            if self._active_ticket is None
            else self._active_ticket.operation_id
        )

    @property
    def active_participant_ep_ids(self) -> tuple[int, ...] | None:
        return (
            None
            if self._active_ticket is None
            else self._active_ticket.participant_ep_ids
        )

    @property
    def queued_tickets(self) -> tuple[StageAdmissionTicket, ...]:
        return tuple(self._ready_fifo)

    def _validate_operation_id(self, operation_id: Hashable) -> None:
        try:
            hash(operation_id)
        except TypeError as exc:
            raise ValueError("operation_id must be hashable") from exc
        if operation_id in self._operation_ids:
            raise ValueError(
                "operation_id is already queued or active in this stage context: "
                f"{operation_id!r}"
            )

    def _new_ticket(
        self,
        *,
        operation_id: Hashable,
        scope: str,
        participant_ep_ids: tuple[int, ...],
    ) -> StageAdmissionTicket:
        self._validate_operation_id(operation_id)
        ticket = StageAdmissionTicket(
            replica_id=self._replica_id,
            stage_id=self._stage_id,
            admission_seq=self._next_admission_seq,
            operation_id=operation_id,
            scope=scope,
            participant_ep_ids=participant_ep_ids,
        )
        self._next_admission_seq += 1
        self._ready_fifo.append(ticket)
        self._operation_ids.add(operation_id)
        return ticket

    def enqueue_full_stage(self, *, operation_id: Hashable) -> StageAdmissionTicket:
        """Queue one dense/full-stage operation for ordered admission."""

        return self._new_ticket(
            operation_id=operation_id,
            scope=FULL_STAGE_WORLD,
            participant_ep_ids=(),
        )

    def _normalize_participants(
        self, participant_ep_ids: Iterable[int]
    ) -> tuple[int, ...]:
        if isinstance(participant_ep_ids, (str, bytes)):
            raise ValueError("EP participant set must be an iterable of ints")
        try:
            participants = tuple(participant_ep_ids)
        except TypeError as exc:
            raise ValueError("EP participant set must be iterable") from exc
        if any(
            isinstance(ep_id, bool)
            or not isinstance(ep_id, int)
            or ep_id < 0
            for ep_id in participants
        ):
            raise ValueError("EP participant IDs must be exact non-negative ints")
        expected = tuple(range(self._ep_size))
        if participants != expected:
            raise ValueError(
                "EP_WAVE requires the complete EP participant set "
                f"{expected}, got {participants}"
            )
        return participants

    def enqueue_ep_wave(
        self,
        *,
        operation_id: Hashable,
        participant_ep_ids: Iterable[int],
    ) -> StageAdmissionTicket:
        """Queue one complete Replica-local EP wave."""

        participants = self._normalize_participants(participant_ep_ids)
        return self._new_ticket(
            operation_id=operation_id,
            scope=EP_WAVE,
            participant_ep_ids=participants,
        )

    def transition_active_scope(
        self,
        ticket: StageAdmissionTicket,
        *,
        operation_id: Hashable,
        scope: str,
        participant_ep_ids: Iterable[int] = (),
    ) -> StageAdmissionTicket:
        """Replace the active layer operation without releasing the stage.

        Shared co-location/PDD batches remain admitted to one pipeline stage while
        they walk layers in lockstep.  Their attention/dense and MoE operations
        therefore need a scope transition, not a momentary ``IDLE`` window that
        could let another operation claim the stage.  The transition keeps the
        parent ownership active, allocates a fresh admission sequence for the
        dependent layer operation, and leaves unrelated ready operations queued.

        This is deliberately an active-owner transition: callers must first have
        acquired ``ticket`` and must replace their stored ticket with the returned
        value.  It does not allow a child lane to bypass the parent FIFO or create
        a partial EP wave.
        """

        self._validate_ticket(ticket)
        if self._active_ticket != ticket:
            raise ValueError(
                "cannot transition a stage admission ticket that is not active"
            )
        if scope == FULL_STAGE_WORLD:
            participants = ()
        elif scope == EP_WAVE:
            participants = self._normalize_participants(participant_ep_ids)
        else:
            raise ValueError(f"unknown stage admission scope: {scope!r}")

        self._validate_operation_id(operation_id)
        next_ticket = StageAdmissionTicket(
            replica_id=self._replica_id,
            stage_id=self._stage_id,
            admission_seq=self._next_admission_seq,
            operation_id=operation_id,
            scope=scope,
            participant_ep_ids=participants,
        )
        self._next_admission_seq += 1
        self._operation_ids.remove(ticket.operation_id)
        self._operation_ids.add(operation_id)
        self._active_ticket = next_ticket
        return next_ticket

    def _validate_ticket(self, ticket: StageAdmissionTicket) -> None:
        if not isinstance(ticket, StageAdmissionTicket):
            raise ValueError("stage admission requires a StageAdmissionTicket")
        if ticket.replica_id != self._replica_id or ticket.stage_id != self._stage_id:
            raise ValueError(
                "stage admission ticket belongs to a different replica/stage context"
            )
        if ticket.scope not in (FULL_STAGE_WORLD, EP_WAVE):
            raise ValueError(f"unknown stage admission scope: {ticket.scope!r}")
        if ticket.operation_id not in self._operation_ids:
            raise ValueError(
                "stage admission ticket is no longer queued or active: "
                f"{ticket.operation_id!r}"
            )

    def try_acquire(self, ticket: StageAdmissionTicket) -> bool:
        """Acquire the FIFO-head ticket if this stage is currently idle."""

        self._validate_ticket(ticket)
        if self._active_ticket is not None:
            return False
        if not self._ready_fifo or self._ready_fifo[0] != ticket:
            return False
        self._ready_fifo.popleft()
        self._active_ticket = ticket
        return True

    def owns(self, ticket: StageAdmissionTicket) -> bool:
        """Return whether this context already owns ``ticket`` for this wave."""

        self._validate_ticket(ticket)
        return self._active_ticket == ticket

    def release(self, ticket: StageAdmissionTicket) -> None:
        """Release exactly the operation currently owning this stage."""

        self._validate_ticket(ticket)
        if self._active_ticket != ticket:
            raise ValueError(
                "cannot release a stage admission ticket that is not the active operation"
            )
        self._active_ticket = None
        self._operation_ids.remove(ticket.operation_id)

    def cancel(self, ticket: StageAdmissionTicket) -> None:
        """Cancel a queued-but-not-acquired operation after atomic preflight failure."""

        self._validate_ticket(ticket)
        if self._active_ticket == ticket:
            raise ValueError("cannot cancel the active stage operation")
        try:
            self._ready_fifo.remove(ticket)
        except ValueError as exc:
            raise ValueError("stage admission ticket is not queued") from exc
        self._operation_ids.remove(ticket.operation_id)


__all__ = [
    "EP_WAVE",
    "FULL_STAGE_WORLD",
    "StageAdmissionTicket",
    "StageExecutionContext",
]
