"""Runtime state owner for DECODE_ATTN A2F/F2A transfers."""

from typing import Any


class AttentionTransferState:
    """Own attention transfer rooms, idle lanes, and barrier sequence."""

    def __init__(self) -> None:
        self.a2f_waiting_by_layer = {}
        self.f2a_waiting_by_round = {}
        self.batch_queue = []
        self.idle_expected_lanes = set()
        self.barrier_round_counter = 0


def initialize_attention_transfer_state(scheduler: Any) -> None:
    """Initialize DECODE_ATTN transfer queues and lane metadata."""

    scheduler._attention_transfer_state = AttentionTransferState()
    expected_lanes = [
        (replica_id, None)
        for replica_id in list(scheduler._cluster.replicas.keys())
    ]
    scheduler._a2f_expected_lanes = expected_lanes
    scheduler._a2f_group_micro_batches = len(expected_lanes)
    scheduler._f2a_expected_lanes = list(expected_lanes)
    scheduler._f2a_group_micro_batches = len(expected_lanes)
