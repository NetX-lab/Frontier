"""Planning helpers for advancing an EP dispatch collective."""

from __future__ import annotations

from typing import Any, NamedTuple


class EPDispatchAdvance(NamedTuple):
    """One validated EP lane ready to enter expert execution."""

    ep_id: int
    batch: Any
    ready_time: float


def prepare_dispatch_advance(
    *,
    ep_batches: dict[int, Any],
    time: float,
) -> tuple[EPDispatchAdvance, ...]:
    """Validate dispatch output and calculate per-lane expert ready times."""

    if not ep_batches:
        raise ValueError("EP dispatch collective reached with empty ep_batches")
    prepared: list[EPDispatchAdvance] = []
    for ep_id, batch in ep_batches.items():
        expert_compute_time = getattr(batch, "expert_compute_time", None)
        if expert_compute_time is None:
            raise ValueError(
                f"Missing expert_compute_time for EP batch {batch.id} "
                f"(ep_id={ep_id})"
            )
        prepared.append(
            EPDispatchAdvance(
                ep_id=int(ep_id),
                batch=batch,
                ready_time=float(time + expert_compute_time),
            )
        )
    return tuple(prepared)
