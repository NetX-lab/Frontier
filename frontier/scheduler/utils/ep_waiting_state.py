"""Waiting rooms for replica-local expert-parallel collectives."""

from collections import defaultdict


def _new_ep_waiting_room():
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
        )
    )


class EPWaitingState:
    """Own EP dispatch and combine waiting rooms for one cluster scheduler."""

    def __init__(self) -> None:
        self.allgather = _new_ep_waiting_room()
        self.alltoall_dispatch = _new_ep_waiting_room()
