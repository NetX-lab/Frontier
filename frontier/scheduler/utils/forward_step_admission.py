"""Stage-ticket transitions for shared forward-step EP waves."""

from typing import Any

from frontier.scheduler.replica_stage_scheduler.stage_execution_context import EP_WAVE, FULL_STAGE_WORLD


def promote_to_ep_wave(
    scheduler: Any,
    *,
    source_batches: dict[int, Any],
    replica_id: int,
    stage_id: int,
    layer_id: int,
    step_id: int,
    participant_ep_ids: tuple[int, ...],
) -> None:
    """Replace active lane owners with one Replica-local EP wave ticket."""

    live_batches = [batch for batch in source_batches.values() if not batch.is_idle]
    tickets = []
    for source_batch in live_batches:
        ticket = getattr(source_batch, "_stage_admission_ticket", None)
        if ticket is None:
            if getattr(scheduler, "_stage_execution_contexts", None) is None:
                return
            raise ValueError("cohort EP promotion requires a stage admission ticket for every live batch")
        if ticket not in tickets:
            tickets.append(ticket)
    if not tickets:
        return
    context = scheduler.get_stage_execution_context(replica_id, stage_id)
    if len(tickets) == 1 and tickets[0].scope == EP_WAVE:
        wave_ticket = tickets[0]
    elif len(tickets) == 1:
        owner_batch = next(
            source_batch for source_batch in live_batches
            if getattr(source_batch, "_stage_admission_ticket", None) == tickets[0]
        )
        scheduler.transition_stage_admission_for_layer(
            owner_batch,
            stage_id=stage_id,
            layer_id=layer_id,
            operation_kind="ffn",
            scope=EP_WAVE,
            participant_ep_ids=participant_ep_ids,
        )
        wave_ticket = owner_batch._stage_admission_ticket
    else:
        if any(ticket.scope != FULL_STAGE_WORLD for ticket in tickets):
            raise ValueError("cohort EP promotion requires full-stage owner tickets")
        wave_ticket = context.replace_full_stage_owners_with_ep_wave(
            tickets,
            operation_id=("shared_ep_wave", int(replica_id), int(stage_id), int(step_id), int(layer_id)),
            participant_ep_ids=participant_ep_ids,
        )
    for source_batch in live_batches:
        source_batch._stage_admission_ticket = wave_ticket
        history = getattr(source_batch, "_stage_admission_scope_history", None)
        if history is None:
            history = []
            source_batch._stage_admission_scope_history = history
        history.append({
            "stage_id": int(stage_id),
            "layer_id": int(layer_id),
            "scope": EP_WAVE,
            "admission_seq": int(wave_ticket.admission_seq),
            "participant_ep_ids": tuple(wave_ticket.participant_ep_ids),
        })


def restore_full_stage_owners(
    scheduler: Any,
    *,
    source_batches: dict[int, Any],
    replica_id: int,
    stage_id: int,
    layer_id: int,
    operation_kind: str,
) -> bool:
    """Restore one full-stage ticket per request-owner lane."""

    live_batches = [batch for batch in source_batches.values() if not batch.is_idle]
    if not live_batches:
        return False
    tickets = []
    for batch in live_batches:
        ticket = getattr(batch, "_stage_admission_ticket", None)
        if ticket is None:
            if getattr(scheduler, "_stage_execution_contexts", None) is None:
                return False
            raise ValueError("cohort full-stage restoration requires a stage admission ticket for every live batch")
        tickets.append(ticket)
    wave_tickets = list(dict.fromkeys(tickets))
    if len(wave_tickets) != 1 or wave_tickets[0].scope != EP_WAVE:
        return False
    context = scheduler.get_stage_execution_context(replica_id, stage_id)
    owners = context.replace_ep_wave_with_full_stage_owners(
        wave_tickets[0],
        operation_ids=[
            ("shared_layer", int(batch.id), int(batch.schedule_epoch), int(stage_id), int(layer_id), operation_kind, FULL_STAGE_WORLD)
            for batch in live_batches
        ],
    )
    for source_batch, owner in zip(live_batches, owners):
        source_batch._stage_admission_ticket = owner
    return True
