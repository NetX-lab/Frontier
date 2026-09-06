from collections import defaultdict, deque

from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.utils.attention_transfer_state import AttentionTransferState
from frontier.scheduler.utils.ep_waiting_state import EPWaitingState


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        return []


def test_ep_waiting_rooms_are_owned_by_ep_state():
    scheduler = object.__new__(_ConcreteClusterScheduler)
    allgather = defaultdict(deque)

    scheduler._ep_allgather_waiting_room = allgather

    assert isinstance(scheduler._ep_waiting_state, EPWaitingState)
    assert scheduler._ep_waiting_state.allgather is allgather
    assert scheduler._ep_allgather_waiting_room is allgather


def test_attention_return_queue_is_owned_by_attention_state():
    scheduler = object.__new__(_ConcreteClusterScheduler)
    batch_queue = []

    scheduler._af_batch_queue = batch_queue

    assert isinstance(scheduler._attention_transfer_state, AttentionTransferState)
    assert scheduler._attention_transfer_state.batch_queue is batch_queue
    assert scheduler._af_batch_queue is batch_queue
