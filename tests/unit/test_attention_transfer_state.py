from types import SimpleNamespace

from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.utils.attention_transfer_state import AttentionTransferState
from frontier.scheduler.utils.pdaf_attention import get_f2a_expected_lanes
from frontier.types import ClusterType


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def test_attention_transfer_state_starts_with_empty_idle_lane_inventory():
    state = AttentionTransferState()

    assert state.idle_expected_lanes == set()


def test_f2a_expected_lanes_accept_uninitialized_idle_inventory():
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._attention_transfer_state = AttentionTransferState()
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._replica_schedulers = {(0, None): object()}
    scheduler._replica_scheduler_count = 1

    assert get_f2a_expected_lanes(scheduler, replica_id=0) == [(0, None)]
