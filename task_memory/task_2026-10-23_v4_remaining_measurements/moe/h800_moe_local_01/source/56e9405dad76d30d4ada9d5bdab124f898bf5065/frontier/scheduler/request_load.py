"""Scheduler request populations exposed to serving load balancers."""

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestLoad:
    """Request populations with KV pressure sampled at the same instant.

    Equality and iteration expose only waiting/running counts. Native engines
    publish when those counts change, so a KV-only change must not refresh the
    coordinator's snapshot. Read ``kv_cache_usage`` to compare pressure itself.
    """

    waiting: int
    running: int
    kv_cache_usage: float = field(default=0.0, compare=False)

    def __iter__(self) -> Iterator[int]:
        yield self.waiting
        yield self.running
