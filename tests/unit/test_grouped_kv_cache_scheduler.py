"""Scheduler wiring for the approved grouped KV-cache allocation contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.memory import (
    AttentionKVCacheCapacity,
    AttentionKVCacheComponent,
    AttentionKVCacheGroup,
)
from frontier.config import (
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Replica, Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType


def _capacity() -> AttentionKVCacheCapacity:
    logical_sizes = (256, 64, 64, 4, 8)
    components = tuple(
        AttentionKVCacheComponent(
            f"component_{index}",
            (index,),
            16,
            stored_rows=logical_size,
            precision="float32" if index >= 3 else "fp8",
            content_size_bytes=16,
        )
        for index, logical_size in enumerate(logical_sizes)
    )
    groups = (
        AttentionKVCacheGroup("compressed", 256, (("component_0", 0),)),
        AttentionKVCacheGroup(
            "swa_even",
            64,
            (("component_1", 1),),
            retention="sliding_window",
            window_tokens=128,
        ),
        AttentionKVCacheGroup(
            "swa_odd",
            64,
            (("component_2", 2),),
            retention="sliding_window",
            window_tokens=128,
        ),
        AttentionKVCacheGroup(
            "csa_state",
            4,
            (("component_3", 3),),
            retention="sliding_window",
            window_tokens=8,
        ),
        AttentionKVCacheGroup(
            "hca_state",
            8,
            (("component_4", 4),),
            retention="sliding_window",
            window_tokens=128,
        ),
    )
    return AttentionKVCacheCapacity(
        family_id="grouped_fixture",
        block_size=256,
        num_layers=5,
        components=components,
        groups=groups,
        reserved_blocks=1,
        zero_new_blocks=True,
    )


def _request(
    request_id: int,
    *,
    prefill: int = 256,
    decode: int = 257,
    processed: int = 0,
    prefill_complete: bool = False,
):
    return SimpleNamespace(
        id=request_id,
        num_prefill_tokens=prefill,
        num_decode_tokens=decode,
        num_processed_tokens=processed,
        num_context_tokens=processed,
        is_recomputing=False,
        is_prefill_complete=prefill_complete,
    )


def _scheduler(*, blocks: int = 400, watermark: int = 3):
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._config = SimpleNamespace(
        block_size=256,
        num_blocks=blocks,
        max_tokens_in_batch=64,
    )
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._max_model_len = 4096
    scheduler._watermark_blocks = watermark
    scheduler._kv_cache_manager = KVCacheManager(
        256,
        blocks,
        False,
        "builtin",
        0,
        cache_capacity=_capacity(),
    )
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._scheduled_num_computed_tokens_by_request = {}
    scheduler._pending_cache_zero_block_ids = {}
    scheduler._active_batch_request_counts = {}
    scheduler._gdn_state_slot_manager = None
    scheduler._current_iteration_token_budget = 64
    scheduler._max_num_running_reqs = 8
    scheduler._max_num_scheduled_tokens = 64
    scheduler._current_schedule_time = 0.0
    scheduler._active_schedule_iteration_id = 0
    scheduler._replica_id = 0
    scheduler._replica_local_id = None
    scheduler._running_requests = []
    scheduler._request_queue = []
    scheduler._preempted_requests = []
    scheduler._waiting_requests = []
    scheduler._max_concurrent_batches = 1
    scheduler._uses_output_placeholders = False
    return scheduler


@pytest.mark.parametrize(
    ("tokens", "expected"),
    ((4, 5), (8, 6), (64, 27), (128, 53), (256, 105)),
)
def test_grouped_step_demand_preserves_each_logical_block_boundary(
    tokens: int, expected: int
) -> None:
    scheduler = _scheduler()
    request = _request(1)

    assert scheduler._kv_cache_manager.get_num_blocks_to_allocate(
        request,
        tokens,
        scheduler_num_computed_tokens=0,
    ) == expected


def test_grouped_reclaim_uses_completed_progress_not_optimistic_frontier() -> None:
    scheduler = _scheduler()
    request = _request(1)
    scheduler._allocate_request(request, 256, scheduler_num_computed_tokens=0)
    scheduler._scheduled_num_computed_tokens_by_request[request.id] = 512
    request.num_processed_tokens = 257
    request.num_context_tokens = 257
    request.is_prefill_complete = True

    scheduler._can_allocate_request(request, 1)

    tables = scheduler._kv_cache_manager.get_group_block_ids(request.id)
    assert sum(block is None for block in tables["swa_even"]) == 2
    assert sum(block is None for block in tables["swa_odd"]) == 2
    assert sum(block is None for block in tables["csa_state"]) == 62
    assert sum(block is None for block in tables["hca_state"]) == 16
    assert scheduler._kv_cache_manager._settled_tokens[request.id] == 256
    assert scheduler.memory_usage_percent == pytest.approx(23 / 399 * 100)


def test_grouped_waiting_admission_accounts_for_concurrent_request_peak() -> None:
    probe = _scheduler()
    first = _request(1)
    admission_blocks = probe._get_grouped_admission_blocks(first)
    scheduler = _scheduler(blocks=admission_blocks + 6, watermark=3)
    first = _request(1)
    second = _request(2)

    scheduler._check_request_fits_kv_pool(first)
    assert scheduler._can_allocate_request(first, 4)
    scheduler._allocate_request(first, 4, scheduler_num_computed_tokens=0)
    assert scheduler._kv_cache_manager.get_num_blocks_for_request(first) == 5
    assert not scheduler._can_allocate_request(second, 4)
    assert scheduler._kv_cache_manager.get_group_block_ids(second.id) == {}


def test_grouped_waiting_admission_also_checks_uncapped_step_demand() -> None:
    scheduler = _scheduler(blocks=20, watermark=1)
    request = _request(1, prefill=4, decode=1)

    scheduler._check_request_fits_kv_pool(request)
    assert scheduler._get_grouped_admission_blocks(request) == 5
    assert scheduler._kv_cache_manager.get_num_blocks_to_allocate(
        request,
        256,
        scheduler_num_computed_tokens=0,
    ) == 105
    assert not scheduler._can_allocate_request(
        request,
        256,
        scheduler_num_computed_tokens=0,
    )
    assert scheduler._kv_cache_manager.get_group_block_ids(request.id) == {}


def test_grouped_scheduler_rejects_prefix_reuse_input_before_query() -> None:
    scheduler = _scheduler()
    request = _request(1)

    with pytest.raises(ValueError, match="computed prefix blocks"):
        scheduler._can_allocate_request(
            request,
            4,
            new_computed_blocks=[object()],
            scheduler_num_computed_tokens=0,
        )
    assert scheduler._kv_cache_manager.get_group_block_ids(request.id) == {}


def test_grouped_scheduler_rejects_second_inflight_step_and_cleans_orphan() -> None:
    scheduler = _scheduler()
    request = _request(1)
    scheduler._allocate_request(request, 8, scheduler_num_computed_tokens=0)
    scheduler._active_batch_request_counts[request.id] = 1

    assert not scheduler._can_allocate_request(
        request, 8, scheduler_num_computed_tokens=8
    )
    with pytest.raises(RuntimeError, match="one in-flight batch"):
        scheduler._allocate_request(
            request, 8, scheduler_num_computed_tokens=8
        )
    scheduler._running_requests.clear()
    scheduler._free_request_resources_by_id(request.id)
    assert scheduler._allocation_map == {}
    assert scheduler._kv_cache_manager.num_used_blocks == 0
    assert (
        scheduler._kv_cache_manager.block_pool.get_num_free_blocks()
        + scheduler._kv_cache_manager.block_pool.reserved_blocks
        == scheduler._config.num_blocks
    )


def test_grouped_allocation_emits_new_and_zeroing_ids(monkeypatch) -> None:
    import frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy as policy

    events = []
    monkeypatch.setattr(policy, "schedule_decision_logging_enabled", lambda: True)
    monkeypatch.setattr(
        policy, "_log_frontier_vllm_v1_schedule_decision", events.append
    )
    scheduler = _scheduler()
    request = _request(1)

    allocation = scheduler._allocate_request(
        request, 8, scheduler_num_computed_tokens=0
    )

    event = events[-1]
    assigned = {
        block_id
        for block_ids in event["group_new_block_ids"].values()
        for block_id in block_ids
    }
    assert event["event"] == "grouped_cache_allocation"
    assert assigned == set(allocation.zero_required_block_ids)
    assert assigned == {block.block_id for block in allocation.new_blocks}
    assert 0 not in assigned
    assert "prefix_cache_hits" not in event


class _ClusterScheduler:
    def __init__(self, replica_id: int) -> None:
        self._context = StageExecutionContext(
            replica_id=replica_id,
            stage_id=0,
            ep_size=16,
        )
        self.lanes = []

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        assert (replica_id, stage_id) == (self._context.replica_id, 0)
        return self._context

    def get_replica_scheduler(self, replica_id: int, lane_id: int):
        assert replica_id == self._context.replica_id
        return self.lanes[lane_id]

    def on_replica_batch_scheduled(self, time, replica_id, lane_id, batch):
        assert replica_id == self._context.replica_id
        assert batch.scheduled_at == time


def _v4_schedulers(
    *, blocks=200, chunk=64, policy="fcfs", enable_preemption=True,
    max_concurrent_batches=None,
):
    replica_config = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        device="h800",
        network_device="h800_dgx",
        attn_tensor_parallel_size=8,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16,
        attention_backend="FLASHMLA_SPARSE_DSV4",
        kv_cache_dtype="fp8",
        indexer_kv_cache_dtype="fp8",
    )
    request_config = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, request_config, ClusterType.MONOLITHIC)
    cluster = _ClusterScheduler(replica.id)
    for lane in range(2):
        cluster.lanes.append(VLLMv1EngineReplicaScheduler(
            replica_config=replica_config,
            replica_scheduler_config=VllmV1SchedulerConfig(
                max_concurrent_batches=max_concurrent_batches,
                num_blocks=blocks,
                scheduling_policy=policy,
                enable_preemption=enable_preemption,
                num_blocks_mode="explicit",
                block_size=256,
                max_tokens_in_batch=chunk,
                long_prefill_token_threshold=chunk,
                enable_chunked_prefill=True,
                max_model_len=256,
                batch_size_cap=8,
            ),
            request_generator_config=request_config,
            replica=replica,
            predictor=object(),
            cluster_scheduler=cluster,
            cluster_type=ClusterType.MONOLITHIC,
            replica_local_id=lane,
        ))
    return cluster.lanes


def test_explicit_queue_capacity_preserves_grouped_single_flight_guard():
    with pytest.raises(NotImplementedError, match="Explicit queue capacity requires"):
        _v4_schedulers(max_concurrent_batches=2)


def test_real_v4_scheduler_reuses_planner_layout_with_reserved_pool() -> None:
    scheduler = _v4_schedulers()[0]

    assert scheduler._kv_cache_manager.cache_capacity is scheduler._kv_cache_capacity
    assert scheduler._kv_cache_manager.enable_caching is False
    assert scheduler._is_grouped_kv_allocation()
    assert not scheduler._is_prefix_caching_enabled()
    assert scheduler._get_available_kv_blocks() == 199


def test_real_v4_scheduler_retires_pages_only_after_settled_decode() -> None:
    scheduler = _v4_schedulers(blocks=80, chunk=189)[0]
    manager = scheduler._kv_cache_manager
    pool = manager.block_pool
    # 189 allocated tokens need 1 compressed, 3+3 SWA, 48 CSA and 24 HCA pages.
    # At settled 190 the SWA cutoff is 63; only settled 191 retires page zero.
    expected = (
        (189, 0, 0, (0, 0, 0, 0, 0), 79, 189),
        (1, 189, 189, (0, 0, 0, 45, 7), 27, 191),
        (1, 190, 191, (0, 0, 0, 45, 7), 27, 192),
        (1, 191, 192, (0, 1, 1, 46, 8), 23, 193),
    )
    groups = ("compressed", "swa_0", "swa_1", "csa_state_0", "hca_state_0")
    sizes = (1, 3, 3, 48, 24)
    all_ids = set(range(1, 80))
    previous_cycle_ids = set()
    for cycle in range(2):
        request = Request(float(cycle * 10), 190, 3)
        scheduler.add_request(request)
        initial_tables = None
        for step, (query, context, processed_before, retired, live, processed_after) in enumerate(expected):
            now = float(cycle * 10 + step * 2)
            assert request.num_processed_tokens == processed_before
            batches = scheduler.on_schedule(now)
            assert len(batches) == 1
            batch = batches[0]
            assert batch.num_tokens == [query]
            assert batch.num_context_tokens == [context]
            tables = manager.get_group_block_ids(request.id)
            assert set(tables) == set(groups)
            for group, size, skipped in zip(groups, sizes, retired):
                assert len(tables[group]) == size
                assert tables[group][:skipped] == (None,) * skipped
                assert all(block is not None for block in tables[group][skipped:])
            if initial_tables is None:
                initial_tables = tables
            for group, skipped in zip(groups, retired):
                assert tables[group][skipped:] == initial_tables[group][skipped:]
            ids = [block for table in tables.values() for block in table if block is not None]
            assert len(ids) == len(set(ids)) == live
            assert set(ids) <= all_ids
            assert [block.ref_cnt for block in pool.blocks] == [
                int(block_id in ids) for block_id in range(80)
            ]
            assert manager.get_num_blocks_for_request(request) == manager.num_used_blocks == live
            assert pool.get_num_free_blocks() + live + pool.reserved_blocks == 80
            assert scheduler.memory_usage_percent == pytest.approx(live / 79 * 100)
            if step == 0:
                assert set(ids) == all_ids
                if cycle:
                    assert set(ids) == previous_cycle_ids
                previous_cycle_ids = set(ids)
            # A second admission attempt cannot reclaim pages while this batch is in flight.
            assert scheduler.on_schedule(now + 0.25) == []
            assert manager.get_group_block_ids(request.id) == tables
            batch.on_batch_end(now + 1.0, ClusterType.MONOLITHIC)
            scheduler.on_batch_end(batch)
            assert request.num_processed_tokens == processed_after
        assert request.completed
        assert manager.get_group_block_ids(request.id) == {}
        assert manager.num_used_blocks == 0
        assert pool.get_num_free_blocks() == 79
        assert scheduler.memory_usage_percent == 0.0
        assert scheduler.num_running_batches == 0
        assert not scheduler.has_unfinished_requests
