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
from frontier.entities import Replica
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


@pytest.mark.parametrize("reported_blocks", (None, 0, 105, 999))
def test_grouped_reclaim_uses_completed_progress_not_optimistic_frontier(
    reported_blocks: int | None,
) -> None:
    scheduler = _scheduler()
    request = _request(1)
    scheduler._allocate_request(request, 256, scheduler_num_computed_tokens=0)
    if reported_blocks is None:
        scheduler._allocation_map.pop(request.id)
    else:
        scheduler._allocation_map[request.id] = reported_blocks
    scheduler._scheduled_num_computed_tokens_by_request[request.id] = 512
    request.num_processed_tokens = 257
    request.num_context_tokens = 257
    request.is_prefill_complete = True

    assert scheduler._can_allocate_request(request, 1)

    tables = scheduler._kv_cache_manager.get_group_block_ids(request.id)
    assert sum(block is None for block in tables["swa_even"]) == 2
    assert sum(block is None for block in tables["swa_odd"]) == 2
    assert sum(block is None for block in tables["csa_state"]) == 62
    assert sum(block is None for block in tables["hca_state"]) == 16
    assert scheduler._kv_cache_manager._settled_tokens[request.id] == 256
    assert scheduler._allocation_map[request.id] == 23
    assert scheduler.memory_usage_percent == pytest.approx(23 / 399 * 100)


def test_grouped_owned_request_continues_without_allocation_report() -> None:
    request = _request(1)
    admission_blocks = _scheduler()._get_grouped_admission_blocks(request)
    scheduler = _scheduler(blocks=admission_blocks + 6, watermark=3)

    assert scheduler._can_allocate_request(request, 4)
    scheduler._allocate_request(request, 4, scheduler_num_computed_tokens=0)
    assert scheduler._kv_cache_manager.has_grouped_allocation(request.id)
    scheduler._allocation_map.pop(request.id)
    assert scheduler._get_available_kv_blocks() - admission_blocks < 3
    assert scheduler._kv_cache_manager.get_num_blocks_to_allocate(
        request, 4, scheduler_num_computed_tokens=4
    ) == 1

    assert scheduler._can_allocate_request(
        request, 4, scheduler_num_computed_tokens=4
    )
    scheduler._allocate_request(request, 4, scheduler_num_computed_tokens=4)

    assert scheduler._kv_cache_manager.get_num_blocks_for_request(request) == 6
    assert scheduler._allocation_map[request.id] == 6


@pytest.mark.parametrize("reported_blocks", (None, 0, 105))
def test_grouped_released_request_admission_ignores_stale_allocation_report(
    reported_blocks: int | None,
) -> None:
    scheduler = _scheduler(blocks=110, watermark=3)
    request = _request(1)
    scheduler._allocate_request(request, 256, scheduler_num_computed_tokens=0)
    request.num_processed_tokens = 257
    request.num_context_tokens = 257
    request.is_prefill_complete = True
    scheduler._kv_cache_manager.free_request_id(request.id)
    assert not scheduler._kv_cache_manager.has_grouped_allocation(request.id)
    if reported_blocks is None:
        scheduler._allocation_map.pop(request.id)
    else:
        scheduler._allocation_map[request.id] = reported_blocks
    free_blocks = scheduler._get_available_kv_blocks()
    assert free_blocks - scheduler._get_grouped_admission_blocks(request) >= 3
    assert scheduler._kv_cache_manager.get_num_blocks_to_allocate(
        request, 1, scheduler_num_computed_tokens=256
    ) == 110
    assert free_blocks == 109

    assert not scheduler._can_allocate_request(
        request, 1, scheduler_num_computed_tokens=256
    )

    assert scheduler._kv_cache_manager.get_group_block_ids(request.id) == {}
    assert scheduler._kv_cache_manager._settled_tokens == {}
    assert scheduler._get_available_kv_blocks() == free_blocks


@pytest.mark.parametrize("reported_blocks", (None, 0, 5))
def test_grouped_waiting_admission_accounts_for_concurrent_request_peak(
    reported_blocks: int | None,
) -> None:
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
    if reported_blocks is not None:
        scheduler._allocation_map[second.id] = reported_blocks
    first_tables = scheduler._kv_cache_manager.get_group_block_ids(first.id)
    free_blocks = scheduler._get_available_kv_blocks()
    assert scheduler._kv_cache_manager.can_allocate_slots(second, 4)

    assert not scheduler._can_allocate_request(second, 4)

    assert scheduler._kv_cache_manager.get_group_block_ids(second.id) == {}
    assert not scheduler._kv_cache_manager.has_grouped_allocation(second.id)
    assert scheduler._kv_cache_manager._settled_tokens == {}
    assert scheduler._kv_cache_manager.get_group_block_ids(first.id) == first_tables
    assert scheduler._get_available_kv_blocks() == free_blocks


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

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        assert (replica_id, stage_id) == (self._context.replica_id, 0)
        return self._context


def test_real_v4_scheduler_reuses_planner_layout_with_reserved_pool() -> None:
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
    scheduler = VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=200,
            num_blocks_mode="explicit",
            block_size=256,
            max_tokens_in_batch=64,
            batch_size_cap=8,
        ),
        request_generator_config=request_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id),
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert scheduler._kv_cache_manager.cache_capacity is scheduler._kv_cache_capacity
    assert scheduler._kv_cache_manager.enable_caching is False
    assert scheduler._is_grouped_kv_allocation()
    assert not scheduler._is_prefix_caching_enabled()
    assert scheduler._get_available_kv_blocks() == 199
