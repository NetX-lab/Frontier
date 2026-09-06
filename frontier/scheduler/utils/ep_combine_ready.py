"""Entry-point routing for EP AllToAll combine readiness.

The cluster scheduler owns runtime state and the concrete handler implementation.
This module owns the public dispatch boundary so event callbacks remain small and
the EP-specific control flow can be moved here incrementally without changing
callers or monkeypatch targets.
"""

from typing import Any


def handle_combine_ready(
    scheduler: Any,
    *,
    time: float,
    replica_id: int,
    stage_id: int,
    batch: Any,
    ep_id: int,
):
    """Dispatch one EP combine-ready callback to the scheduler-owned handler."""

    handler = getattr(scheduler, "_handle_ep_alltoall_combine_ready", None)
    if not callable(handler):
        raise TypeError(
            "Base cluster scheduler must provide "
            "_handle_ep_alltoall_combine_ready()"
        )
    return handler(
        time=time,
        replica_id=replica_id,
        stage_id=stage_id,
        batch=batch,
        ep_id=ep_id,
    )
