"""Scheduler request populations exposed to serving load balancers."""

from typing import NamedTuple


class RequestLoad(NamedTuple):
    """Count waiting and admitted requests, including unscheduled running work."""

    waiting: int
    running: int
