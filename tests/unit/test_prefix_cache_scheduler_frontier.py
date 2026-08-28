"""Regression coverage for chunked Prefix cache allocation progress."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities.request import Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


@pytest.mark.parametrize("frontier", (-1, 1.5, True))
def test_prefix_manager_rejects_invalid_scheduler_frontier(frontier) -> None:
    manager = KVCacheManager(
        block_size=16,
        num_gpu_blocks=2,
        enable_caching=True,
        caching_hash_algo="builtin",
        num_preallocate_tokens=0,
    )
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=1,
        block_hash_ids=[11],
    )

    with pytest.raises(
        ValueError,
        match="scheduler_num_computed_tokens must be a non-negative integer",
    ):
        manager.can_allocate_slots(
            request,
            1,
            scheduler_num_computed_tokens=frontier,
        )


def test_chunked_prefill_binds_next_prefix_hash_at_scheduler_frontier() -> None:
    """A scheduled chunk advances allocation before Request progress commits."""
    manager = KVCacheManager(
        block_size=16,
        num_gpu_blocks=2,
        enable_caching=True,
        caching_hash_algo="builtin",
        num_preallocate_tokens=0,
    )
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=32,
        num_decode_tokens=1,
        block_hash_ids=[11, 22],
    )
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=16, num_blocks=2)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._replica_id = 0
    scheduler._replica_local_id = None
    scheduler._active_schedule_iteration_id = 0
    scheduler._prefix_cache_identity_event_seq = 0
    scheduler._current_schedule_time = 0.0
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._scheduled_num_computed_tokens_by_request = {}

    assert scheduler._get_scheduler_num_computed_tokens(request) == 0
    assert scheduler._can_allocate_request(request, 16)
    scheduler._allocate_request(request, 16)
    assert manager.block_pool.get_cached_block(11) is not None
    assert request.num_processed_tokens == 0

    scheduler._advance_scheduler_num_computed_tokens(request, 16)
    assert scheduler._get_scheduler_num_computed_tokens(request) == 16
    assert scheduler._can_allocate_request(request, 16)
    scheduler._allocate_request(request, 16)

    assert manager.block_pool.get_cached_block(22) is not None
    assert request.num_processed_tokens == 0


def test_preempted_prefix_recovery_allows_one_scheduled_cache_hit() -> None:
    recovery_request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=1,
        block_hash_ids=[11],
    )
    recovery_request._scheduled = True
    recovery_request._preempted = True

    recovery_request.on_cache_hit(16)

    assert recovery_request.num_processed_tokens == 16
    assert recovery_request.num_prefill_tokens_cached == 16

    duplicate_request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=1,
        block_hash_ids=[11],
    )
    duplicate_request._scheduled = True
    duplicate_request._preempted = False
    with pytest.raises(ValueError, match="already scheduled"):
        duplicate_request.on_cache_hit(16)
