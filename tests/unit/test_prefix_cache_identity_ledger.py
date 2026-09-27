"""Regression tests for Prefix cache physical-block identity evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities.request import Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.kv_cache.kv_cache_block_pool import BlockPool
from frontier.scheduler.replica_scheduler import (
    vllm_v1_prefix_cache as prefix_cache_module,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _request(*, hashes: list[int], num_prefill_tokens: int = 4) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=1,
        block_hash_ids=hashes,
    )


def _manager(num_gpu_blocks: int = 2) -> KVCacheManager:
    return KVCacheManager(
        block_size=2,
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        caching_hash_algo="builtin",
        num_preallocate_tokens=0,
    )


def test_allocation_result_records_binding_creator_and_epoch() -> None:
    manager = _manager()
    creator = _request(hashes=[11, 22])

    result = manager.allocate_slots(creator, 4)

    assert result is not None
    assert [block.block_id for block in result.new_blocks] == [0, 1]
    assert [
        (
            binding.block_hash,
            binding.block_id,
            binding.creator_request_id,
            binding.binding_epoch,
        )
        for binding in result.new_bindings
    ] == [
        (11, 0, creator.id, 1),
        (22, 1, creator.id, 1),
    ]
    assert result.evicted_bindings == ()


@pytest.mark.parametrize("num_tokens", (0, -1, 1.5, True))
def test_kv_cache_manager_rejects_invalid_allocation_tokens(num_tokens) -> None:
    manager = _manager()
    request = _request(hashes=[11, 22])

    with pytest.raises(ValueError, match="num_tokens must be a positive integer"):
        manager.allocate_slots(request, num_tokens)


@pytest.mark.parametrize("num_blocks", (-1, 1.5, True))
def test_block_pool_rejects_invalid_allocation_count(num_blocks) -> None:
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True)

    with pytest.raises(
        ValueError,
        match="num_blocks must be a non-negative integer",
    ):
        pool.get_new_blocks(num_blocks)


def test_committed_full_hit_admission_records_reuse_eviction_and_rebinding(
    monkeypatch,
) -> None:
    manager = _manager()
    creator = _request(hashes=[11, 22])
    first_result = manager.allocate_slots(creator, 4)
    assert first_result is not None
    manager.free(creator)

    consumer = _request(hashes=[11, 22])
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=2, num_blocks=2)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._replica_id = 3
    scheduler._replica_local_id = None
    scheduler._active_schedule_iteration_id = 7
    scheduler._current_schedule_time = 1.25
    scheduler._prefix_cache_identity_event_seq = 0
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        prefix_cache_module,
        "_log_frontier_vllm_v1_schedule_decision",
        lambda event: events.append(dict(event)),
    )

    admission = scheduler._prepare_prefix_cache_admission(consumer)

    assert [block.block_hash for block in admission.raw_hit_blocks] == [11, 22]
    assert [block.block_hash for block in admission.effective_hit_blocks] == [11]
    assert admission.raw_cached_tokens == 4
    assert admission.effective_cached_tokens == 2
    assert admission.num_new_tokens == 2
    assert admission.full_hit_backoff_applied is True

    allocation = scheduler._allocate_request(
        consumer,
        admission.num_new_tokens,
        new_computed_blocks=list(admission.effective_hit_blocks),
        prefix_cache_admission=admission,
    )

    assert allocation is not None
    assert [event["event"] for event in events] == [
        "prefix_cache_admission",
        "prefix_cache_allocation",
    ]
    admission_event, allocation_event = events
    assert admission_event["replica_local_id"] is None
    assert allocation_event["replica_local_id"] is None
    assert admission_event["identity_event_seq"] == 0
    assert allocation_event["identity_event_seq"] == 1
    assert admission_event["query_hashes"] == [11, 22]
    assert admission_event["raw_hit_blocks"] == [
        {
            "query_index": 0,
            "block_hash": 11,
            "block_id": 0,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        },
        {
            "query_index": 1,
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        },
    ]
    assert admission_event["admitted_hit_blocks"] == [
        admission_event["raw_hit_blocks"][0]
    ]
    assert admission_event["full_hit_backoff_applied"] is True
    assert allocation_event["reused_blocks"] == [
        {
            "block_hash": 11,
            "block_id": 0,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        }
    ]
    assert allocation_event["new_block_ids"] == [1]
    assert allocation_event["evicted_bindings"] == [
        {
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        }
    ]
    assert allocation_event["new_bindings"] == [
        {
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(consumer.id),
            "binding_epoch": 2,
        }
    ]


@pytest.mark.parametrize(
    "num_prefill_tokens, admitted_hashes",
    [(3, [11]), (5, [11, 22])],
    ids=["prompt_inside_the_second_block", "prompt_inside_the_third_block"],
)
def test_a_hit_beyond_the_prompt_leaves_its_last_token_to_compute(
    num_prefill_tokens, admitted_hashes
) -> None:
    # A clipped trace row keeps the hashes of its longer original prompt.
    # vLLM caps a hit at num_tokens - 1 (kv_cache_manager.get_computed_blocks).
    manager = _manager(num_gpu_blocks=4)
    creator = _request(hashes=[11, 22, 33], num_prefill_tokens=6)
    assert manager.allocate_slots(creator, 6) is not None
    manager.free(creator)
    consumer = _request(hashes=[11, 22, 33], num_prefill_tokens=num_prefill_tokens)
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=2, num_blocks=4)

    admission = scheduler._prepare_prefix_cache_admission(consumer)

    assert [block.block_hash for block in admission.raw_hit_blocks] == [11, 22, 33]
    assert [block.block_hash for block in admission.effective_hit_blocks] == admitted_hashes
    assert admission.effective_cached_tokens == 2 * len(admitted_hashes)
    assert admission.num_new_tokens == num_prefill_tokens - 2 * len(admitted_hashes)
    assert admission.full_hit_backoff_applied is True
    assert manager.prefix_cache_stats.hits == len(admitted_hashes)
