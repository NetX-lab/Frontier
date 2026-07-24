"""Regression tests for DECODE_ATTN same-iteration preemption state."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

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
