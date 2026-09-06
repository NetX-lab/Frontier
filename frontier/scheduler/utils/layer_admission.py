"""Stage admission transitions at shared model layer boundaries."""

from typing import Any

def transition_layer_admission(
    scheduler: Any,
    batch: Any,
    *,
    stage_id: int,
    layer_id: int,
    operation_kind: str,
    scope: str,
    participant_ep_ids: tuple[int, ...] = (),
) -> None:
    """Replace a batch's active stage ticket at a layer boundary."""

    if type(stage_id) is not int or stage_id < 0:
        raise ValueError("stage_id must be an exact non-negative int")
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError("layer_id must be an exact non-negative int")
    if operation_kind not in ("attention", "ffn"):
        raise ValueError(
            "operation_kind must be 'attention' or 'ffn', "
            f"got {operation_kind!r}",
        )
    ticket = getattr(batch, "_stage_admission_ticket", None)
    contexts = getattr(scheduler, "_stage_execution_contexts", None)
    if ticket is None and contexts is None:
        return
    if ticket is None:
        raise ValueError("shared layer operation is missing its stage admission ticket")
    context = scheduler.get_stage_execution_context(ticket.replica_id, stage_id)
    operation_id = (
        "shared_layer",
        int(batch.id),
        int(batch.schedule_epoch),
        int(stage_id),
        int(layer_id),
        operation_kind,
        scope,
    )
    next_ticket = context.transition_active_scope(
        ticket,
        operation_id=operation_id,
        scope=scope,
        participant_ep_ids=participant_ep_ids,
    )
    batch._stage_admission_ticket = next_ticket
    history = getattr(batch, "_stage_admission_scope_history", None)
    if history is None:
        history = []
        batch._stage_admission_scope_history = history
    history.append(
        {
            "stage_id": int(stage_id),
            "layer_id": int(layer_id),
            "scope": scope,
            "admission_seq": int(next_ticket.admission_seq),
            "participant_ep_ids": tuple(next_ticket.participant_ep_ids),
        }
    )
