"""Allocation and lifetime checks for the existing manager's grouped path."""

from types import SimpleNamespace
from dataclasses import replace
from math import ceil

import pytest

from frontier.attention.memory import (
    AttentionKVCacheCapacity,
    AttentionKVCacheComponent,
    AttentionKVCacheGroup,
    AttentionKVCacheSelection,
    get_attention_kv_cache_capacity,
)
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.config import ReplicaConfig
from frontier.kv_cache import base_kv_cache_manager
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager


def capacity():
    return AttentionKVCacheCapacity(
        family_id="test_grouped", block_size=8, num_layers=1,
        components=(
            AttentionKVCacheComponent("history", (0,), 16, stored_rows=8, precision="float16", content_size_bytes=16),
            AttentionKVCacheComponent("window", (0,), 16, stored_rows=4, precision="float32", content_size_bytes=16),
        ),
        groups=(
            AttentionKVCacheGroup("history", 8, (("history", 0),)),
            AttentionKVCacheGroup(
                "window", 4, (("window", 0),),
                retention="sliding_window", window_tokens=5,
            ),
        ),
        reserved_blocks=1, zero_new_blocks=True,
    )


def manager(blocks=20):
    return KVCacheManager(8, blocks, False, "builtin", 0, cache_capacity=capacity())


def test_grouped_allocation_uses_one_pool_and_exposes_new_ids():
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    allocated = cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    tables = cache.get_group_block_ids(request.id)
    assert {k: len(v) for k, v in tables.items()} == {"history": 1, "window": 2}
    live = {block for table in tables.values() for block in table if block is not None}
    assert len(live) == 3 and 0 not in live
    assert allocated.zero_required_block_ids == tuple(b.block_id for b in allocated.new_blocks)
    assert dict(allocated.group_new_block_ids) == tables
    assert cache.num_used_blocks == cache.get_num_blocks_for_request(request) == 3
    assert cache.num_used_blocks + cache.block_pool.get_num_free_blocks() + 1 == 20
    assert cache.usage == pytest.approx(3 / 19)


def test_demand_query_is_pure_and_reclaim_preserves_partial_page():
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    before = cache.get_group_block_ids(request.id)
    assert cache.can_allocate_slots(request, 4, scheduler_num_computed_tokens=8)
    assert cache.get_group_block_ids(request.id) == before
    # At settled frontier 7, skipped tokens=7-5+1=3: no whole page is safe.
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=7)
    assert cache.get_group_block_ids(request.id) == before
    # Frontier 8 permits the first four-token window page to be retired.
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    assert cache.get_group_block_ids(request.id)["window"] == (None, before["window"][1])
    assert cache.get_group_block_ids(request.id)["history"] == before["history"]
    assert cache.num_used_blocks == 2
    allocated = cache.allocate_slots(request, 4, scheduler_num_computed_tokens=8)
    assert len(allocated.new_blocks) == 2
    assert len(allocated.zero_required_block_ids) == 2
    assert cache.get_num_blocks_for_request(request) == 4


def test_safe_release_survives_failed_demand_and_id_cleanup_is_idempotent():
    cache = manager(5)
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    released = cache.get_group_block_ids(request.id)
    assert not cache.can_allocate_slots(request, 16, scheduler_num_computed_tokens=8)
    assert cache.allocate_slots(request, 16, scheduler_num_computed_tokens=8) is None
    assert cache.get_group_block_ids(request.id) == released
    assert cache.num_used_blocks == 2
    cache.free_request_id(request.id)
    cache.free_request_id(request.id)
    assert cache.get_group_block_ids(request.id) == {}
    assert cache.num_used_blocks == 0 and cache.block_pool.get_num_free_blocks() == 4


def test_admission_caps_window_history_but_step_allocation_is_uncapped():
    cache = manager()
    # history ceil(80/8)=10; window cap ceil((5-1+8)/4)+1=4.
    assert cache.get_num_admission_blocks(80, max_model_len=80, max_in_flight_tokens=8) == 14
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    assert cache.get_num_blocks_to_allocate(request, 80, scheduler_num_computed_tokens=0) == 30
    assert not cache.can_allocate_slots(request, 80, scheduler_num_computed_tokens=0)


def test_free_pool_reassignment_requires_zeroing_but_held_pages_do_not():
    cache = manager(4)
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    first = cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    first_ids = set(first.zero_required_block_ids)
    held = cache.allocate_slots(request, 1, scheduler_num_computed_tokens=0)
    assert held.new_blocks == () and held.zero_required_block_ids == ()
    cache.free(request)
    second = cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    assert set(second.zero_required_block_ids) == first_ids


def test_grouped_contract_rejects_prefix_and_preallocation():
    for caching, preallocation in ((True, 0), (False, 8)):
        with pytest.raises(ValueError, match="grouped"):
            KVCacheManager(8, 20, caching, "builtin", preallocation, cache_capacity=capacity())


def test_pool_failure_leaves_ownership_unchanged(monkeypatch):
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    before = cache.get_group_block_ids(request.id)
    free_before = cache.block_pool.get_num_free_blocks()
    original = cache.block_pool.get_new_blocks

    def fail(_count):
        raise RuntimeError("injected pool allocation failure")

    monkeypatch.setattr(cache.block_pool, "get_new_blocks", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cache.allocate_slots(request, 4, scheduler_num_computed_tokens=8)
    assert cache.get_group_block_ids(request.id) == before
    assert cache.block_pool.get_num_free_blocks() == free_before
    monkeypatch.setattr(cache.block_pool, "get_new_blocks", original)
    assert cache.allocate_slots(request, 4, scheduler_num_computed_tokens=8) is not None


def test_allocation_failure_rolls_back_only_new_assignment(monkeypatch):
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 8, scheduler_num_computed_tokens=0)
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    before = cache.get_group_block_ids(request.id)
    assert before["window"][0] is None
    free_before = cache.block_pool.get_num_free_blocks()
    result_type = base_kv_cache_manager.KVCacheAllocationResult
    assigned = []

    def fail_after_pool_assignment(**kwargs):
        assigned.extend(kwargs["new_blocks"])
        assert len(assigned) == 2
        assert all(block.ref_cnt == 1 for block in assigned)
        raise RuntimeError("injected result construction failure")

    monkeypatch.setattr(
        base_kv_cache_manager, "KVCacheAllocationResult", fail_after_pool_assignment
    )
    with pytest.raises(RuntimeError, match="injected result construction"):
        cache.allocate_slots(request, 4, scheduler_num_computed_tokens=8)
    assert all(block.ref_cnt == 0 for block in assigned)
    assert cache.get_group_block_ids(request.id) == before
    assert cache.block_pool.get_num_free_blocks() == free_before
    assert cache.num_used_blocks == 2
    assert cache.num_used_blocks + free_before + 1 == cache.num_gpu_blocks
    monkeypatch.setattr(base_kv_cache_manager, "KVCacheAllocationResult", result_type)
    assert cache.allocate_slots(request, 4, scheduler_num_computed_tokens=8) is not None


def test_grouped_query_never_creates_request_ownership():
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    free_before = cache.block_pool.get_num_free_blocks()
    for _ in range(3):
        assert cache.get_num_blocks_to_allocate(request, 8) == 3
        assert cache.can_allocate_slots(request, 8)
    assert cache._group_tables == cache._settled_tokens == {}
    assert cache.req_to_blocks == {}
    assert cache.block_pool.get_num_free_blocks() == free_before


def test_two_requests_keep_disjoint_groups_and_cleanup_only_the_owner():
    cache = manager()
    first = SimpleNamespace(id=7, num_processed_tokens=0)
    second = SimpleNamespace(id=8, num_processed_tokens=0)
    cache.allocate_slots(first, 8)
    cache.allocate_slots(second, 8)
    second_tables = cache.get_group_block_ids(second.id)
    first_ids = {b for table in cache.get_group_block_ids(first.id).values() for b in table}
    second_ids = {b for table in second_tables.values() for b in table}
    assert len(first_ids | second_ids) == 6
    cache.reclaim_skipped_blocks(first, settled_num_computed_tokens=8)
    cache.free_request_id(first.id)
    assert cache.get_group_block_ids(second.id) == second_tables
    assert cache.get_num_blocks_for_request(second) == cache.num_used_blocks == 3
    cache.free_request_id(second.id)
    assert cache.num_used_blocks == 0
    assert cache.block_pool.get_num_free_blocks() + 1 == cache.num_gpu_blocks


def test_reclaim_rejects_progress_regression_and_allocation_before_settled():
    cache = manager()
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 8)
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    before = cache.get_group_block_ids(request.id)
    with pytest.raises(ValueError, match="cannot move backwards"):
        cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=7)
    with pytest.raises(ValueError, match="cannot precede settled"):
        cache.can_allocate_slots(request, 1, scheduler_num_computed_tokens=7)
    assert cache.get_group_block_ids(request.id) == before


@pytest.fixture(scope="module")
def source_capacity():
    replica = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
        indexer_kv_cache_dtype="fp8",
    )
    return get_attention_kv_cache_capacity(
        DEEPSEEK_V4_ATTENTION_FAMILY, replica.model_config, block_size=256,
        selection=AttentionKVCacheSelection.from_replica(replica, block_size=256),
    )


@pytest.mark.parametrize("tokens", [3, 4, 5, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_source_group_boundaries_preserve_logical_positions(source_capacity, tokens):
    cache = KVCacheManager(256, 512, False, "builtin", 0, cache_capacity=source_capacity)
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, tokens, scheduler_num_computed_tokens=0)
    # Independent pinned source logical block/window pairs, not resolver output.
    spans = (256, 64, 64, 4, 8)
    windows = (None, 128, 128, 8, 128)
    before = cache.get_group_block_ids(request.id)
    expected_slots = tuple(ceil(tokens / span) for span in spans)
    assert tuple(len(table) for table in before.values()) == expected_slots
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=tokens)
    after = cache.get_group_block_ids(request.id)
    expected_live = 0
    for table, original, span, window in zip(after.values(), before.values(), spans, windows):
        skipped = 0 if window is None else max(0, tokens - window + 1) // span
        assert table[:skipped] == (None,) * skipped
        assert table[skipped:] == original[skipped:]
        expected_live += len(table) - skipped
    assert cache.num_used_blocks == expected_live
    expected_new = sum(ceil((tokens + 1) / span) - ceil(tokens / span) for span in spans)
    assert cache.get_num_blocks_to_allocate(
        request, 1, scheduler_num_computed_tokens=tokens
    ) == expected_new
    result = cache.allocate_slots(request, 1, scheduler_num_computed_tokens=tokens)
    assert len(result.zero_required_block_ids) == expected_new
    assert cache.num_used_blocks == expected_live + expected_new
    assert cache.num_used_blocks + cache.block_pool.get_num_free_blocks() + 1 == 512


def test_source_admission_accounts_window_and_one_inflight_chunk(source_capacity):
    cache = KVCacheManager(256, 512, False, "builtin", 0, cache_capacity=source_capacity)
    # History16 + two SWA7 + CSA state67 + HCA state49 pages.
    assert cache.get_num_admission_blocks(
        4096, max_model_len=4096, max_in_flight_tokens=256
    ) == 146


def test_extra_retained_tokens_delay_whole_page_reclamation():
    original = capacity()
    extended = replace(original, groups=(original.groups[0], replace(
        original.groups[1], extra_retained_tokens=4,
    )))
    cache = KVCacheManager(8, 20, False, "builtin", 0, cache_capacity=extended)
    request = SimpleNamespace(id=7, num_processed_tokens=0)
    cache.allocate_slots(request, 12)
    before = cache.get_group_block_ids(request.id)["window"]
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=8)
    assert cache.get_group_block_ids(request.id)["window"] == before
    cache.reclaim_skipped_blocks(request, settled_num_computed_tokens=12)
    assert cache.get_group_block_ids(request.id)["window"] == (None, *before[1:])
