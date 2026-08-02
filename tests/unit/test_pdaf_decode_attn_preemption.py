"""Regression tests for DECODE_ATTN same-iteration preemption state."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from frontier.entities.request import Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=request_id,
        priority=0,
        arrived_at=float(request_id),
        completed=False,
        completed_layer_count=0,
        af_roundtrip_inflight=False,
        current_decode_token_index=1,
        record_preemption=lambda *args: None,
        advance_runtime_epoch=lambda: None,
        on_enter_waiting_queue=lambda *args: None,
        _preempted=False,
    )


def _decode_attn_scheduler() -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._scheduling_policy = "fcfs"
    scheduler._enable_preemption = True
    scheduler._micro_batch_size = 2
    scheduler._af_pipeline_num_micro_batch = 1
    scheduler._num_stages = 1
    scheduler._replica_is_moe = True
    scheduler._continuation_request_ids = set()
    scheduler._running_requests = []
    scheduler._waiting_requests = []
    scheduler._request_queue = []
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._af_pending_micro_batches = deque()
    scheduler._batch_creation_counter = 0
    scheduler._current_schedule_time = 0.0
    scheduler._decode_attn_active_cohort_states = {}
    scheduler._decode_attn_next_cohort_id = 0
    scheduler._scheduled_num_computed_tokens_by_request = {}
    scheduler._current_iteration_token_budget = 2
    scheduler._config = SimpleNamespace(num_blocks=1, block_size=1)
    scheduler._get_scheduler_num_computed_tokens = lambda request: 0

    def free_request_resources(request: SimpleNamespace) -> None:
        if request.id in scheduler._allocation_map:
            scheduler._allocation_map.pop(request.id)
            scheduler._num_allocated_blocks = max(
                0, scheduler._num_allocated_blocks - 1
            )

    scheduler._free_request_resources = free_request_resources
    scheduler._can_allocate_request = lambda request, tokens: (
        (request.id == 0 and request.id in scheduler._allocation_map)
        or (request.id == 1 and scheduler._num_allocated_blocks == 0)
    )

    def allocate_request(request: SimpleNamespace, tokens: int) -> None:
        if request.id not in scheduler._allocation_map:
            scheduler._allocation_map[request.id] = 1
            scheduler._num_allocated_blocks += 1

    scheduler._allocate_request = allocate_request
    scheduler._create_batch = lambda requests, tokens: SimpleNamespace(
        requests=list(requests),
        num_tokens=list(tokens),
        set_global_id=lambda value: None,
    )
    return scheduler


def test_decode_attn_preemption_rolls_victim_out_of_current_batch() -> None:
    """A same-iteration victim cannot remain in both Batch and waiting queue."""

    scheduler = _decode_attn_scheduler()
    victim = _request(0)
    successor = _request(1)
    scheduler._running_requests[:] = [victim, successor]
    scheduler._allocation_map[victim.id] = 1
    scheduler._num_allocated_blocks = 1

    batch = scheduler._schedule_decode_attn_only()

    assert batch is not None
    assert [request.id for request in batch.requests] == [1]
    assert [request.id for request in scheduler._waiting_requests] == [0]
    assert [request.id for request in scheduler._running_requests] == [1]
    assert scheduler._allocation_map == {1: 1}
    assert {request.id for request in batch.requests}.isdisjoint(
        request.id for request in scheduler._waiting_requests
    )


def test_decode_attn_without_preemption_keeps_scheduled_request() -> None:
    """The rollback contract must not alter a legal no-preemption iteration."""

    scheduler = _decode_attn_scheduler()
    request = _request(0)
    scheduler._can_allocate_request = lambda candidate, tokens: True
    scheduler._running_requests[:] = [request]

    batch = scheduler._schedule_decode_attn_only()

    assert batch is not None
    assert [scheduled.id for scheduled in batch.requests] == [0]
    assert scheduler._waiting_requests == []
    assert [running.id for running in scheduler._running_requests] == [0]


def test_decode_attn_preemption_preserves_handoff_token_progress() -> None:
    """DECODE_ATTN preemption must not rewind the completed PREFILL handoff."""

    scheduler = _decode_attn_scheduler()
    victim = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    victim._is_prefill_complete = True
    victim._num_processed_tokens = 17
    victim._num_handoff_emitted_decode_tokens = 1
    victim._current_decode_token_index = 2
    victim._completed_layer_count = 0
    victim._scheduled = True
    scheduler._get_scheduler_num_computed_tokens = (
        lambda request: request.num_processed_tokens
    )
    scheduler._running_requests[:] = [victim]
    scheduler._allocation_map[victim.id] = 1
    scheduler._num_allocated_blocks = 1
    scheduler._scheduled_num_computed_tokens_by_request[victim.id] = 17

    before = {
        "processed_tokens": victim.num_processed_tokens,
        "is_prefill_complete": victim.is_prefill_complete,
        "emitted_decode_tokens": victim.num_emitted_decode_tokens,
        "remaining_decode_tokens": victim.remaining_decode_tokens,
        "decode_token_index": victim.current_decode_token_index,
    }

    scheduler._preempt_request(victim, [])

    after = {
        "processed_tokens": victim.num_processed_tokens,
        "is_prefill_complete": victim.is_prefill_complete,
        "emitted_decode_tokens": victim.num_emitted_decode_tokens,
        "remaining_decode_tokens": victim.remaining_decode_tokens,
        "decode_token_index": victim.current_decode_token_index,
    }

    assert after == before
    assert victim.get_tokens_at_preemption(ClusterType.DECODE_ATTN) == [17]
    assert victim.id not in scheduler._allocation_map
    assert victim.id not in scheduler._scheduled_num_computed_tokens_by_request
    assert scheduler._num_allocated_blocks == 0


def test_non_pdaf_decode_preemption_keeps_restart_reset_contract() -> None:
    """The generic DECODE path still resets Request progress on preemption."""

    scheduler = _decode_attn_scheduler()
    scheduler._cluster_type = ClusterType.DECODE
    victim = _request(0)
    victim._num_processed_tokens = 17
    scheduler._running_requests[:] = [victim]
    scheduler._allocation_map[victim.id] = 1
    scheduler._num_allocated_blocks = 1

    scheduler._preempt_request(victim, [])

    assert victim._num_processed_tokens == 0


def test_decode_attn_admission_records_handoff_waiting_time() -> None:
    """Initial DECODE_ATTN admission must close the request waiting interval."""

    scheduler = _decode_attn_scheduler()
    scheduler._current_schedule_time = 5.0
    scheduler._can_allocate_request = lambda candidate, tokens: True
    request = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    request._is_prefill_complete = True
    request._num_processed_tokens = 16
    request.on_disaggregated_decode_handoff(1.0, ClusterType.DECODE_ATTN)
    request.on_arrival(1.0, ClusterType.DECODE_ATTN)
    scheduler._waiting_requests[:] = [request]

    batch = scheduler._schedule_decode_attn_only()

    assert batch is not None
    assert request.get_cluster_waiting_time(ClusterType.DECODE_ATTN) == 4.0
    assert request._is_waiting[ClusterType.DECODE_ATTN] is False


def test_decode_attn_handoff_allocation_commits_the_preflight_block_count() -> None:
    """Transferred prompt KV blocks must remain charged after admission."""

    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._config = SimpleNamespace(num_blocks=3, block_size=16)
    scheduler._max_model_len = 64
    scheduler._watermark_blocks = 0
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._scheduled_num_computed_tokens_by_request = {}

    first = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    first._is_prefill_complete = True
    first._num_processed_tokens = 16
    second = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    second._is_prefill_complete = True
    second._num_processed_tokens = 16

    assert scheduler._can_allocate_request(first, 1) is True
    scheduler._allocate_request(first, 1)

    assert scheduler._allocation_map[first.id] == 2
    assert scheduler._num_allocated_blocks == 2
    assert scheduler._can_allocate_request(second, 1) is False


def test_decode_attn_preempted_handoff_resumes_without_token_loss() -> None:
    """A preempted handoff request can resume and commit its final output token."""

    scheduler = _decode_attn_scheduler()
    scheduler._current_schedule_time = 5.0
    scheduler._config = SimpleNamespace(num_blocks=35, block_size=1)
    scheduler._max_model_len = 64
    scheduler._watermark_blocks = 0
    del scheduler._can_allocate_request
    del scheduler._allocate_request
    del scheduler._free_request_resources

    victim = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=3)
    victim._is_prefill_complete = True
    victim._num_processed_tokens = 17
    victim._num_handoff_emitted_decode_tokens = 1
    victim._current_decode_token_index = 2
    victim.on_batch_schedule(4.0, ClusterType.DECODE_ATTN)
    successor = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=3)
    successor._is_prefill_complete = True
    successor._num_processed_tokens = 17
    successor._num_handoff_emitted_decode_tokens = 1
    successor._current_decode_token_index = 2
    successor.on_batch_schedule(4.0, ClusterType.DECODE_ATTN)
    scheduler._running_requests[:] = [victim, successor]
    scheduler._allocation_map = {victim.id: 17, successor.id: 17}
    scheduler._num_allocated_blocks = 34
    scheduler._get_scheduler_num_computed_tokens = (
        lambda request: request.num_processed_tokens
    )

    pressure_batch = scheduler._schedule_decode_attn_only()

    assert pressure_batch is not None
    assert [request.id for request in pressure_batch.requests] == [successor.id]
    assert scheduler._waiting_requests == [victim]
    assert scheduler._running_requests == [successor]
    assert scheduler._allocation_map == {successor.id: 18}
    assert scheduler._num_allocated_blocks == 18
    assert victim.get_preemption_count(ClusterType.DECODE_ATTN) == 1
    assert victim.num_processed_tokens == 17
    assert victim.num_emitted_decode_tokens == 2
    assert victim.remaining_decode_tokens == 1

    scheduler._free_request_resources(successor)
    scheduler._running_requests.remove(successor)
    scheduler._current_schedule_time = 7.0

    resumed_batch = scheduler._schedule_decode_attn_only()

    assert resumed_batch is not None
    assert resumed_batch.requests == [victim]
    assert scheduler._waiting_requests == []
    assert scheduler._running_requests == [victim]
    assert victim.get_cluster_waiting_time(ClusterType.DECODE_ATTN) == 2.0
    assert scheduler._allocation_map == {victim.id: 18}
    assert scheduler._num_allocated_blocks == 18
    assert scheduler._num_allocated_blocks <= scheduler._config.num_blocks

    victim.on_batch_schedule(7.0, ClusterType.DECODE_ATTN)
    victim.on_batch_end(8.0, 1, ClusterType.DECODE_ATTN)

    assert victim.num_processed_tokens == 18
    assert victim.num_emitted_decode_tokens == 3
    assert victim.remaining_decode_tokens == 0
    assert victim.current_decode_token_index == 3
    assert victim.completed is True
