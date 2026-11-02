"""Runtime state owner for DECODE_FFN M2N transfers."""

from collections import deque


class M2NTransferState:
    """Own M2N waiting, ready, and raw-batch inventories."""

    def __init__(self) -> None:
        self.waiting_by_layer = {}
        self.ready_groups = deque()
        self.raw_batches = {}
