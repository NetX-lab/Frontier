"""Compatibility views for scheduler-owned runtime state."""

from frontier.scheduler.utils.attention_transfer_state import AttentionTransferState
from frontier.scheduler.utils.ep_waiting_state import EPWaitingState
from frontier.scheduler.utils.m2n_state import M2NTransferState


class SchedulerStateViews:
    """Expose legacy scheduler attributes while state lives in dedicated owners."""

    def _get_attention_transfer_state(self) -> AttentionTransferState:
        state = getattr(self, "_attention_transfer_state", None)
        if state is None:
            state = AttentionTransferState()
            self._attention_transfer_state = state
        return state

    _a2f_waiting_by_layer = property(
        lambda self: self._get_attention_transfer_state().a2f_waiting_by_layer,
        lambda self, value: setattr(self._get_attention_transfer_state(), "a2f_waiting_by_layer", value),
    )
    _f2a_waiting_by_round = property(
        lambda self: self._get_attention_transfer_state().f2a_waiting_by_round,
        lambda self, value: setattr(self._get_attention_transfer_state(), "f2a_waiting_by_round", value),
    )
    _decode_attn_idle_expected_lanes = property(
        lambda self: self._get_attention_transfer_state().idle_expected_lanes,
        lambda self, value: setattr(self._get_attention_transfer_state(), "idle_expected_lanes", value),
    )
    _decode_attn_barrier_round_counter = property(
        lambda self: self._get_attention_transfer_state().barrier_round_counter,
        lambda self, value: setattr(self._get_attention_transfer_state(), "barrier_round_counter", value),
    )
    _af_batch_queue = property(
        lambda self: self._get_attention_transfer_state().batch_queue,
        lambda self, value: setattr(self._get_attention_transfer_state(), "batch_queue", value),
    )

    def _get_m2n_state(self) -> M2NTransferState:
        state = getattr(self, "_m2n_state", None)
        if state is None:
            state = M2NTransferState()
            self._m2n_state = state
        return state

    _m2n_waiting_by_layer = property(
        lambda self: self._get_m2n_state().waiting_by_layer,
        lambda self, value: setattr(self._get_m2n_state(), "waiting_by_layer", value),
    )
    _m2n_ready_groups = property(
        lambda self: self._get_m2n_state().ready_groups,
        lambda self, value: setattr(self._get_m2n_state(), "ready_groups", value),
    )
    _raw_batch_waiting_for_m2n_back = property(
        lambda self: self._get_m2n_state().raw_batches,
        lambda self, value: setattr(self._get_m2n_state(), "raw_batches", value),
    )

    def _get_ep_waiting_state(self) -> EPWaitingState:
        state = getattr(self, "_ep_waiting_state", None)
        if state is None:
            state = EPWaitingState()
            self._ep_waiting_state = state
        return state

    _ep_allgather_waiting_room = property(
        lambda self: self._get_ep_waiting_state().allgather,
        lambda self, value: setattr(self._get_ep_waiting_state(), "allgather", value),
    )
    _ep_alltoall_dispatch_waiting_room = property(
        lambda self: self._get_ep_waiting_state().alltoall_dispatch,
        lambda self, value: setattr(self._get_ep_waiting_state(), "alltoall_dispatch", value),
    )
