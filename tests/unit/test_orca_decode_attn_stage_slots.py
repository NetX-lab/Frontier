from types import SimpleNamespace

from frontier.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from frontier.types import ClusterType


def _request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=request_id,
        completed_layer_count=0,
        current_decode_token_index=1,
        is_finished_for_cluster=lambda _cluster_type: False,
    )


def _scheduler() -> OrcaReplicaScheduler:
    scheduler = object.__new__(OrcaReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_id = 0
    scheduler._replica_local_id = 0
    scheduler._max_batch_size = 1
    scheduler._micro_batch_size = 1
    scheduler._af_pipeline_num_micro_batch = 2
    scheduler._decode_attn_active_stage_slots = set()
    scheduler._preempted_requests = []
    scheduler._request_queue = [_request(0), _request(1)]
    scheduler._allocation_map = {}
    scheduler._max_blocks_per_sequence = 1
    scheduler._num_running_batches = 0
    scheduler.can_allocate = lambda _num_blocks: True
    scheduler.allocate = lambda request_id, num_blocks: scheduler._allocation_map.__setitem__(
        request_id,
        num_blocks,
    )
    scheduler._get_request_next_num_tokens = lambda _request: 1

    next_batch_id = iter(range(3))
    scheduler._create_batch = lambda requests, num_tokens: SimpleNamespace(
        id=next(next_batch_id),
        requests=list(requests),
        num_tokens=list(num_tokens),
        afd_stage_idx=None,
    )
    return scheduler


def test_decode_attn_orca_assigns_unique_stage_slots_and_reuses_completed_slot() -> None:
    scheduler = _scheduler()

    first = scheduler._get_next_batch(is_micro_batch=True)
    second = scheduler._get_next_batch(is_micro_batch=True)

    assert [first.afd_stage_idx, second.afd_stage_idx] == [0, 1]
    assert scheduler._decode_attn_active_stage_slots == {0, 1}

    scheduler._num_running_batches = 2
    scheduler.on_batch_end(first)
    replacement = scheduler._get_next_batch(is_micro_batch=True)

    assert replacement.afd_stage_idx == 0
    assert scheduler._decode_attn_active_stage_slots == {0, 1}
