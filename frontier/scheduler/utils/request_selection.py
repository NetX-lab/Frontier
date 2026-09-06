"""Pure request-selection helpers used by scheduler transition handlers."""

from typing import Any, Iterable


def collect_active_requests(batches: Iterable[Any]) -> list[Any]:
    """Return unfinished requests once, preserving batch/request order."""

    active_requests = []
    seen_request_ids = set()
    for batch in batches:
        if batch.is_idle:
            continue
        for request in batch.requests:
            if request.completed or request.id in seen_request_ids:
                continue
            seen_request_ids.add(request.id)
            active_requests.append(request)
    return active_requests
