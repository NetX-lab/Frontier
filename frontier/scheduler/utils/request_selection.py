"""Pure request-selection helpers used by scheduler transition handlers."""

from typing import Any, Iterable


def collect_active_requests(batches: Iterable[Any]) -> list[Any]:
    """Return unfinished requests once, preserving batch/request order.

    Only requests the batch still executes are returned. A request preempted
    after the batch formed has re-entered the waiting queue under a new
    execution epoch, and the batch's remaining layers no longer run for it.
    """

    active_requests = []
    seen_request_ids = set()
    for batch in batches:
        if batch.is_idle:
            continue
        for request in batch.current_execution_requests:
            if request.completed or request.id in seen_request_ids:
                continue
            seen_request_ids.add(request.id)
            active_requests.append(request)
    return active_requests
