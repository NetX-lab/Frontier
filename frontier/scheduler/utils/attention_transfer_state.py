"""Runtime state owner for DECODE_ATTN A2F/F2A transfers."""


class AttentionTransferState:
    """Own attention transfer rooms, idle lanes, and barrier sequence."""

    def __init__(self) -> None:
        self.a2f_waiting_by_layer = {}
        self.f2a_waiting_by_round = {}
        self.idle_expected_lanes = None
        self.barrier_round_counter = 0
