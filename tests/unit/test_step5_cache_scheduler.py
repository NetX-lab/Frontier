"""Step cache planning and request lifetime through Frontier's real V1 scheduler.

These CPU regressions use the 92-layer text configuration and source-defined
cache bytes. Completion callbacks are controlled; they do not execute the
model, measure GPU allocations, or establish native or predictive E2E support.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from frontier.attention.memory import AttentionKVCacheSelection
from frontier.config import (
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.config.speculative_decoding_config import SpeculativeDecodingConfig
from frontier.entities import Cluster, Request
from frontier.errors import FrontierMemoryOOMError
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_scheduler import vllm_v1_decision_log
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_BYTES = 40_599_958_208
BF16_BLOCK_BYTES = 1_224_980
SUMMARY_BYTES_PER_SLOT = 94_484
DEVICE_BYTES = 80 * 1024**3
SWA_GROUPS = ("swa_0", "swa_1", "swa_2")


@pytest.fixture(autouse=True)
def simulation_globals(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    for name, value in (
        ("SYS_ARCH", "co-location"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("CUDAGRAPH_CAPTURE_SIZES", None), ("KV_CONNECTOR", "none"),
        ("ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC", False),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
        ("QUANTIZATION_MANAGER", None),
    ):
        monkeypatch.setattr(global_vars, name, value)


@pytest.fixture
def replica_config():
    return ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800",
        network_device="h800_dgx", num_pipeline_stages=1,
        attn_tensor_parallel_size=16, attn_dp=1,
        moe_tensor_parallel_size=16, moe_expert_parallel_size=1,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )


def scheduler(replica_config, tmp_path, *, request_config=None, **fields):
    options = dict(
        num_blocks=0, num_blocks_mode="memory_planner", block_size=64,
        max_model_len=8192, batch_size_cap=1, max_tokens_in_batch=512,
        enable_chunked_prefill=True, watermark_blocks_fraction=0,
    )
    options.update(fields)
    if request_config is None:
        request_config = SyntheticRequestGeneratorConfig(
            num_requests=2,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=1025, decode_tokens=3, max_tokens=8192,
            ),
        )
    cluster_config = ClusterConfig(
        cluster_type=ClusterType.MONOLITHIC, num_replicas=1,
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(**options),
    )
    cluster = Cluster(
        cluster_config,
        MetricsConfig(output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
                      write_json_trace=False),
        request_config,
    )
    owner = RoundRobinClusterScheduler(
        config=cluster_config, cluster=cluster,
        request_generator_config=request_config, predictor=object(),
    )
    result = owner.get_replica_scheduler(next(iter(cluster.replicas)), 0)
    assert isinstance(result, VLLMv1EngineReplicaScheduler)
    return result


def complete(scheduler, batch, time):
    """Deliver a controlled completion through the real Batch and Request APIs."""
    batch.on_batch_end(time, ClusterType.MONOLITHIC)
    scheduler.on_batch_end(batch)


def assert_pool_conserved(scheduler, requests):
    cache = scheduler._kv_cache_manager
    pool = cache.block_pool
    assigned = [
        block_id
        for request in requests
        for table in cache.get_group_block_ids(request.id).values()
        for block_id in table if block_id is not None
    ]
    assert len(assigned) == len(set(assigned)) == cache.num_used_blocks
    assert 0 not in assigned
    assert all(pool.blocks[block_id].ref_cnt == 1 for block_id in assigned)
    assert pool.reserved_blocks == 1
    assert cache.num_used_blocks + pool.get_num_free_blocks() + 1 == cache.num_gpu_blocks
    assert scheduler.num_allocated_blocks == cache.num_used_blocks
    assert scheduler.memory_usage_percent == pytest.approx(cache.usage * 100)


@pytest.mark.parametrize(("dtype", "raw_page", "stride", "planned"), [
    ("auto", 49_152, 1_224_980, 29_967),
    ("bfloat16", 49_152, 1_224_980, 29_967),
    ("fp8", 24_576, 659_732, 55_642),
    ("fp8_e4m3", 24_576, 659_732, 55_642),
])
def test_tp16_planner_binds_four_groups_to_one_physical_pool(
    replica_config, tmp_path, dtype, raw_page, stride, planned,
):
    configured = replace(replica_config, kv_cache_dtype=dtype)
    actual = scheduler(configured, tmp_path)
    planner = MemoryPlanner(configured, actual._replica, ClusterType.MONOLITHIC,
                            max_num_seqs=1)
    capacity = actual._kv_cache_manager.cache_capacity
    assert capacity is actual._kv_cache_capacity
    assert capacity.num_layers == actual._replica.num_layers == 92
    assert configured.world_size == 16
    assert AttentionKVCacheSelection.from_replica(
        configured, block_size=64,
    ).attention_tensor_parallel_size == 16
    assert planner.get_parameter_memory_per_device_bytes() == PARAMETER_BYTES
    assert planner.get_num_blocks(block_size=64) == actual._config.num_blocks == planned
    assert capacity.physical_block_size_bytes == stride
    assert capacity.fixed_bytes_per_request_slot == SUMMARY_BYTES_PER_SLOT
    assert [group.group_id for group in capacity.groups] == [*SWA_GROUPS, "sparse"]
    assert [group.logical_block_size for group in capacity.groups] == [64] * 4
    swa, sparse, summary, tail = capacity.components
    assert swa.page_size_bytes == sparse.page_size_bytes == raw_page
    assert swa.content_size_bytes == sparse.content_size_bytes == raw_page
    assert summary.page_size_bytes == 2048
    assert tail.page_size_bytes == 2060
    assert len(swa.owner_layer_ids) == 69
    assert sparse.owner_layer_ids == summary.owner_layer_ids == tail.owner_layer_ids == tuple(range(3, 92, 4))
    for index, group in enumerate(capacity.groups):
        expected_raw = "swa_raw_kv" if index < 3 else "sparse_raw_kv"
        assert tuple(owner for component, owner in group.component_owners
                     if component == expected_raw) == tuple(range(index, 92, 4))
    # Raw pages are unpadded in the supported 64/64 tuple. The pool pays one
    # raw group stride plus sparse sidepages, not four independent raw views.
    raw_stride = 1_130_496 if raw_page == 49_152 else 565_248
    assert [capacity.get_group_size_bytes(group) for group in capacity.groups] == [
        raw_stride, raw_stride, raw_stride, stride,
    ]
    assert stride - raw_stride == SUMMARY_BYTES_PER_SLOT
    assert capacity.zero_new_blocks is False
    assert actual._kv_cache_manager.enable_caching is False
    assert actual._kv_cache_manager.block_pool.get_num_free_blocks() == planned - 1
    with pytest.raises(ValueError, match="no additive common block"):
        _ = capacity.resident_bytes_per_common_block
    assert_pool_conserved(actual, [])


def test_scheduler_reclaims_only_completed_swa_pages_and_keeps_sparse_history(
    replica_config, tmp_path,
):
    actual = scheduler(replica_config, tmp_path)
    request = Request(0.0, 1025, 3)
    actual.add_request(request)
    cache = actual._kv_cache_manager
    initial_free = cache.block_pool.get_num_free_blocks()
    assert actual.num_pending_requests == 1
    assert not cache.has_grouped_allocation(request.id)
    assert_pool_conserved(actual, [])

    # Each row is a worked 64-token page boundary, with a 512-token SWA
    # window. The last sampled output never becomes another cache input, and
    # decode context excludes the sampled token that is the step's query.
    steps = (
        (512, 0, 512, 512, 8, 0, 32),
        (512, 512, 1024, 1024, 16, 0, 64),
        (1, 1024, 1025, 1026, 17, 8, 44),
        (1, 1025, 1026, 1027, 17, 8, 44),
        (1, 1026, 1027, 1028, 17, 8, 44),
    )
    previous_sparse = ()
    for index, (query, context, scheduled, completed, pages, retired, used) in enumerate(steps):
        time = float(2 * index + 1)
        before = request.num_processed_tokens
        [batch] = actual.on_schedule(time)
        assert batch.requests == [request]
        assert batch.num_tokens == [query]
        assert batch.num_context_tokens == [context]
        assert actual._scheduled_num_computed_tokens_by_request[request.id] == scheduled
        assert request.num_processed_tokens == before
        tables = cache.get_group_block_ids(request.id)
        assert tables["sparse"][:len(previous_sparse)] == previous_sparse
        assert len(tables["sparse"]) == pages
        assert all(block is not None for block in tables["sparse"])
        for group in SWA_GROUPS:
            assert len(tables[group]) == pages
            assert tables[group][:retired] == (None,) * retired
            assert all(block is not None for block in tables[group][retired:])
        assert cache.num_used_blocks == used
        assert_pool_conserved(actual, [request])
        # A second schedule while the PP1 output is pending cannot advance
        # the frontier or retire pages against its uncompleted input width.
        assert actual.on_schedule(time + .5) == []
        assert cache.get_group_block_ids(request.id) == tables
        assert request.num_processed_tokens == before
        assert actual._scheduled_num_computed_tokens_by_request[request.id] == scheduled
        previous_sparse = tables["sparse"]
        complete(actual, batch, time + 1)
        assert request.num_processed_tokens == completed
        assert actual.num_running_batches == 0
        if index < len(steps) - 1:
            assert not request.completed
            assert cache.get_group_block_ids(request.id) == tables
            assert_pool_conserved(actual, [request])

    assert request.completed
    assert request.num_emitted_decode_tokens == 3
    assert cache.get_group_block_ids(request.id) == {}
    assert not cache.has_grouped_allocation(request.id)
    assert request.id not in actual._scheduled_num_computed_tokens_by_request
    assert request.id not in cache._settled_tokens
    assert actual.num_pending_requests == 0
    assert cache.block_pool.get_num_free_blocks() == initial_free
    assert_pool_conserved(actual, [])
    assert actual.is_empty()
    assert actual.on_schedule(12.0) == []


@pytest.mark.parametrize(("max_num_seqs", "blocks", "remainder"), [
    (1, 29_967, 382_976),
    (8, 29_966, 946_568),
    (64, 29_962, 555_384),
])
def test_automatic_planning_reserves_summary_scratch_once_per_request_slot(
    replica_config, tmp_path, max_num_seqs, blocks, remainder,
):
    actual = scheduler(replica_config, tmp_path, batch_size_cap=max_num_seqs)
    capacity = actual._kv_cache_manager.cache_capacity
    assert actual._admitted_request_capacity == max_num_seqs
    assert actual._config.num_blocks == blocks
    assert capacity.fixed_bytes_per_request_slot == SUMMARY_BYTES_PER_SLOT
    assert capacity.physical_block_size_bytes == BF16_BLOCK_BYTES
    # 72 GiB is the configured 90% H800 budget. The known Step
    # shard leaves one shared-pool extent and M summary scratch slots.
    requested = 77_309_411_328
    assert requested - PARAMETER_BYTES - blocks * BF16_BLOCK_BYTES - remainder == (
        max_num_seqs * SUMMARY_BYTES_PER_SLOT
    )
    assert 0 <= remainder < BF16_BLOCK_BYTES
    assert {component.component_id for component in capacity.components} == {
        "swa_raw_kv", "sparse_raw_kv", "region_summary", "open_region_state",
    }
    assert_pool_conserved(actual, [])


def test_planner_reserves_largest_scheduler_ownership_cap_without_more_token_groups(
    replica_config, tmp_path,
):
    actual = scheduler(
        replica_config, tmp_path, batch_size_cap=1,
        enable_phase_aware_thinking_profile=True,
        hidden_phase_batch_size_cap=8, final_phase_batch_size_cap=64,
    )
    assert actual._admitted_request_capacity == 64
    assert actual._config.num_blocks == 29_962
    assert actual._kv_cache_capacity.fixed_bytes_per_request_slot == SUMMARY_BYTES_PER_SLOT
    assert len(actual._kv_cache_capacity.groups) == 4


@pytest.mark.parametrize(("blocks", "remainder"), [(2, 0), (17, 0), (17, 1_224_979)])
def test_automatic_planner_floor_keeps_null_and_fixed_scratch_in_the_same_budget(
    replica_config, tmp_path, blocks, remainder,
):
    requested = PARAMETER_BYTES + SUMMARY_BYTES_PER_SLOT + blocks * BF16_BLOCK_BYTES + remainder
    utilization = requested / DEVICE_BYTES
    assert int(DEVICE_BYTES * utilization) == requested
    actual = scheduler(replica_config, tmp_path, gpu_memory_utilization=utilization)
    assert actual._config.num_blocks == blocks
    assert actual._kv_cache_manager.block_pool.get_num_free_blocks() == blocks - 1
    assert_pool_conserved(actual, [])


@pytest.mark.parametrize(("cache_budget", "available"), [
    (0, 0), (1_224_979, 1_224_979), (1_224_980, 1_224_980),
    (2_449_959, 2_449_959),
])
def test_automatic_planner_rejects_no_allocatable_block_after_fixed_scratch_and_null(
    replica_config, tmp_path, cache_budget, available,
):
    requested = PARAMETER_BYTES + SUMMARY_BYTES_PER_SLOT + cache_budget
    with pytest.raises(FrontierMemoryOOMError) as raised:
        scheduler(replica_config, tmp_path, gpu_memory_utilization=requested / DEVICE_BYTES)
    assert raised.value.reason == "insufficient_kv_cache_budget"
    details = raised.value.details
    assert details["parameter_memory_per_device_bytes"] == PARAMETER_BYTES
    assert details["fixed_cache_reservation_bytes"] == SUMMARY_BYTES_PER_SLOT
    assert details["physical_block_size_bytes"] == BF16_BLOCK_BYTES
    assert details["available_kv_cache_memory_bytes"] == available
    assert details["reserved_blocks"] == 1
    assert details["cluster_type"] == "MONOLITHIC"


@pytest.mark.parametrize("weight_budget", [PARAMETER_BYTES - 1, PARAMETER_BYTES])
def test_automatic_planner_rejects_parameter_shard_before_creating_cache_pool(
    replica_config, tmp_path, weight_budget,
):
    with pytest.raises(FrontierMemoryOOMError) as raised:
        scheduler(replica_config, tmp_path, gpu_memory_utilization=weight_budget / DEVICE_BYTES)
    assert raised.value.reason == "parameter_memory_exceeds_requested_budget"
    assert raised.value.details["parameter_memory_per_device_bytes"] == PARAMETER_BYTES
    assert raised.value.details["requested_memory_bytes"] == weight_budget


@pytest.mark.parametrize(("blocks", "watermark_fraction", "fits"), [
    (4, 0.0, False), (5, 0.0, True), (5, 0.2, False), (6, 0.2, True),
])
def test_request_admission_consumes_four_group_ids_from_one_pool_after_null_and_watermark(
    replica_config, tmp_path, blocks, watermark_fraction, fits,
):
    requested = PARAMETER_BYTES + SUMMARY_BYTES_PER_SLOT + blocks * BF16_BLOCK_BYTES
    actual = scheduler(replica_config, tmp_path,
                       gpu_memory_utilization=requested / DEVICE_BYTES,
                       watermark_blocks_fraction=watermark_fraction,
                       max_tokens_in_batch=64)
    assert actual._config.num_blocks == blocks
    cache = actual._kv_cache_manager
    request = Request(0.0, 64, 1)
    if not fits:
        with pytest.raises(FrontierMemoryOOMError) as raised:
            actual.add_request(request)
        assert raised.value.reason == "request_exceeds_kv_cache"
        assert raised.value.details["reserved_blocks"] == 1
        assert raised.value.details["allocatable_blocks"] == blocks - 1
        assert not cache.has_grouped_allocation(request.id)
        assert actual.num_pending_requests == 0
        assert_pool_conserved(actual, [])
        return
    actual.add_request(request)
    [batch] = actual.on_schedule(1.0)
    assert batch.num_tokens == [64]
    assert {group: len(table) for group, table in cache.get_group_block_ids(request.id).items()} == {
        "swa_0": 1, "swa_1": 1, "swa_2": 1, "sparse": 1,
    }
    assert cache.num_used_blocks == 4
    assert cache.block_pool.get_num_free_blocks() == blocks - 5
    assert_pool_conserved(actual, [request])
    complete(actual, batch, 2.0)
    assert request.completed
    assert cache.block_pool.get_num_free_blocks() == blocks - 1
    assert_pool_conserved(actual, [])


def test_real_scheduler_logs_only_new_group_ids_without_pool_zeroing(
    replica_config, tmp_path, monkeypatch,
):
    log_path = tmp_path / "schedule_decisions.jsonl"
    logger = logging.Logger("step5_cache_scheduler_test", logging.INFO)
    handler = logging.FileHandler(log_path)
    logger.addHandler(handler)
    monkeypatch.setattr(
        vllm_v1_decision_log, "_frontier_vllm_v1_sched_decision_logger", logger,
    )
    try:
        actual = scheduler(replica_config, tmp_path, max_tokens_in_batch=64)
        request = Request(0.0, 64, 3)
        actual.add_request(request)
        cache = actual._kv_cache_manager
        initial_free = cache.block_pool.get_num_free_blocks()
        expected_ids = []
        for index, query in enumerate((64, 1, 1)):
            [batch] = actual.on_schedule(float(2 * index + 1))
            assert batch.requests == [request]
            assert batch.num_tokens == [query]
            tables = cache.get_group_block_ids(request.id)
            if index < 2:
                expected_ids.append({group: [table[-1]] for group, table in tables.items()})
            else:
                expected_ids.append({group: [] for group in tables})
            assert_pool_conserved(actual, [request])
            complete(actual, batch, float(2 * index + 2))
        assert request.completed
        assert cache.block_pool.get_num_free_blocks() == initial_free
        assert_pool_conserved(actual, [])
    finally:
        logger.removeHandler(handler)
        handler.close()

    events = [
        event for line in log_path.read_text().splitlines()
        if (event := json.loads(line))["event"] == "grouped_cache_allocation"
    ]
    assert len(events) == 3
    for event, ids, tokens in zip(events, expected_ids, (64, 1, 1)):
        assert event["source"] == "frontier"
        assert event["scheduler"] == "vllm_v1"
        assert event["request_id"] == str(request.id)
        assert event["num_tokens"] == tokens
        assert event["grouped_cache_schema_version"] == 1
        assert event["group_new_block_ids"] == ids
        # This is simulator allocation metadata. It does not establish native
        # zeroing, sparse-summary initialization, or device contents.
        assert event["zero_required_block_ids"] == []
        assigned = [block for blocks in ids.values() for block in blocks]
        assert len(assigned) == len(set(assigned))
        assert 0 not in assigned
        assert "prefix_cache_hits" not in event
    assert events[0]["available_blocks"] == initial_free - 4
    assert events[1]["available_blocks"] == events[2]["available_blocks"] == initial_free - 8


@pytest.mark.parametrize(("max_num_seqs", "waves"), [
    (1, (
        ((0,), (64,), (0,), 8),
        ((1,), (64,), (0,), 8),
        ((1,), (1,), (64,), 4),
        ((1,), (1,), (65,), 4),
        ((2,), (64,), (0,), 8),
    )),
    (2, (
        ((0, 1), (64, 64), (0, 0), 4),
        ((2, 1), (64, 1), (0, 64), 0),
        ((1,), (1,), (65,), 4),
    )),
])
def test_scheduler_limits_request_ownership_and_frees_only_completed_owners(
    replica_config, tmp_path, max_num_seqs, waves,
):
    # Thirteen physical IDs leave twelve allocatable IDs after the null.
    # With M=2, the second wave fills the pool while the first owner is gone.
    blocks = 13
    requested = (PARAMETER_BYTES + max_num_seqs * SUMMARY_BYTES_PER_SLOT
                 + blocks * BF16_BLOCK_BYTES)
    actual = scheduler(
        replica_config, tmp_path, batch_size_cap=max_num_seqs,
        max_tokens_in_batch=128, gpu_memory_utilization=requested / DEVICE_BYTES,
    )
    requests = [Request(0.0, 64, outputs) for outputs in (1, 3, 1)]
    for request in requests:
        actual.add_request(request)
    cache = actual._kv_cache_manager
    assert actual._config.num_blocks == blocks
    assert actual._admitted_request_capacity == max_num_seqs
    assert_pool_conserved(actual, [])
    second_tables = {}
    for index, (owners, queries, contexts, free) in enumerate(waves):
        before = {request.id: request.num_processed_tokens for request in requests}
        [batch] = actual.on_schedule(float(2 * index + 1))
        assert batch.requests == [requests[owner] for owner in owners]
        assert tuple(batch.num_tokens) == queries
        assert tuple(batch.num_context_tokens) == contexts
        assert actual.get_request_load().running <= max_num_seqs
        assert cache.block_pool.get_num_free_blocks() == free
        assert {request.id: request.num_processed_tokens for request in requests} == before
        assert_pool_conserved(actual, requests)
        for request, query in zip(batch.requests, queries):
            expected_frontier = 64 if query == 64 else before[request.id]
            assert actual._scheduled_num_computed_tokens_by_request[request.id] == expected_frontier
        next_second_tables = cache.get_group_block_ids(requests[1].id)
        for group, previous in second_tables.items():
            assert next_second_tables[group][:len(previous)] == previous
        second_tables = next_second_tables
        assigned = {
            request.id: [block for table in cache.get_group_block_ids(request.id).values()
                         for block in table if block is not None]
            for request in batch.requests
        }
        complete(actual, batch, float(2 * index + 2))
        for request in batch.requests:
            if request.completed:
                assert cache.get_group_block_ids(request.id) == {}
                assert request.id not in actual._scheduled_num_computed_tokens_by_request
                assert request.id not in cache._settled_tokens
                assert all(cache.block_pool.blocks[block].ref_cnt == 0 for block in assigned[request.id])
            else:
                assert cache.get_num_blocks_for_request(request) == len(assigned[request.id])
        if not requests[1].completed:
            # Releasing first or third cannot change the second owner's pages.
            assert cache.get_group_block_ids(requests[1].id) == second_tables
        else:
            second_tables = {}
        assert_pool_conserved(actual, requests)

    assert all(request.completed for request in requests)
    assert [request.num_emitted_decode_tokens for request in requests] == [1, 3, 1]
    assert actual._scheduled_num_computed_tokens_by_request == {}
    assert cache._group_tables == cache._settled_tokens == {}
    assert cache.block_pool.get_num_free_blocks() == blocks - 1
    assert actual.is_empty()


@pytest.mark.parametrize("fields", [
    {"num_blocks_mode": "explicit", "num_blocks": 2},
    {"num_blocks": 2},
    {"num_blocks_mode": "memory_planner_profiled"},
    {"num_blocks_mode": "memory_planner_profiled",
     "enable_runtime_non_kv_cache_overhead_profiling": True},
    {"non_kv_cache_overhead_bytes": 1},
], ids=("explicit_blocks", "prepopulated_blocks", "profiled_mode",
        "runtime_profile", "additional_overhead"))
def test_scheduler_rejects_unverified_block_counts_or_double_charged_scratch(
    replica_config, tmp_path, fields,
):
    with pytest.raises(NotImplementedError, match="Fixed cache scratch requires automatic memory_planner"):
        scheduler(replica_config, tmp_path, **fields)


@pytest.mark.parametrize(("fields", "message"), [
    ({"attention_backend": None}, "STEP5_DSA/STEP5_FLASH_ATTN pair"),
    ({"attention_backend": "FLASH_ATTN"}, "STEP5_DSA/STEP5_FLASH_ATTN pair"),
    ({"kv_cache_dtype": None}, "BF16 or E4M3"),
    ({"kv_cache_dtype": "float16"}, "BF16 or E4M3"),
    ({"kv_cache_dtype": "fp8_e5m2"}, "BF16 or E4M3"),
    ({"indexer_kv_cache_dtype": "fp8"}, "no configurable indexer cache dtype"),
    ({"device": "a100"}, "H800 or H200 Hopper"),
], ids=("missing_backend", "wrong_backend", "missing_cache_dtype", "fp16_cache",
        "e5m2_cache", "indexer_dtype", "non_hopper"))
def test_scheduler_rejects_missing_or_unsupported_cache_selection(
    replica_config, tmp_path, fields, message,
):
    with pytest.raises(NotImplementedError, match=message):
        scheduler(replace(replica_config, **fields), tmp_path)


@pytest.mark.parametrize(("fields", "message"), [
    ({"num_preallocate_tokens": 64}, "does not support block preallocation"),
    ({"block_size": 24}, "multiple of 16 tokens"),
    ({"batch_size_cap": 0}, "effective scheduler request capacity must be positive"),
], ids=("preallocation", "kernel_alignment", "missing_capacity"))
def test_scheduler_rejects_unsupported_grouped_cache_options(
    replica_config, tmp_path, fields, message,
):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        scheduler(replica_config, tmp_path, **fields)


def test_scheduler_rejects_pipeline_parallel_before_cache_creation(replica_config, tmp_path):
    with pytest.raises(NotImplementedError, match="Step parameter storage requires PP1"):
        scheduler(replace(replica_config, num_pipeline_stages=2), tmp_path)


@pytest.mark.parametrize("fields", (
    {"method": "eagle"},
    {"method": "deepseek_mtp", "mtp_n_predict": 1, "mtp_num_layers": 3},
), ids=("eagle", "enabled_mtp"))
def test_scheduler_rejects_speculative_or_mtp_request_path(replica_config, tmp_path, fields):
    configured = replace(
        replica_config, speculative_decoding_config=SpeculativeDecodingConfig(
            enabled=True, **fields,
        ),
    )
    with pytest.raises(NotImplementedError, match="excludes speculative decoding/MTP"):
        scheduler(configured, tmp_path)


def test_public_block_planner_requires_request_capacity_for_fixed_scratch(replica_config, tmp_path):
    actual = scheduler(replica_config, tmp_path)
    planner = MemoryPlanner(replica_config, actual._replica, ClusterType.MONOLITHIC)
    with pytest.raises(ValueError, match="requires max_num_seqs for fixed cache scratch"):
        planner.get_num_blocks(block_size=64)


def test_real_scheduler_rejects_prefix_even_with_valid_trace_metadata(replica_config, tmp_path):
    trace = tmp_path / "prefix_trace.csv"
    trace.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        "0.0,64,3,session1,page1\n", encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="Grouped KV cache allocation does not support prefix caching"):
        scheduler(
            replica_config, tmp_path, enable_prefix_caching=True,
            request_config=TraceRequestGeneratorConfig(trace_file=str(trace), max_tokens=8192),
        )


def test_complete_request_logs_cache_state_at_actual_batch_end(
    replica_config, tmp_path, monkeypatch,
):
    import frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy as policy
    import frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation as allocation

    events = []
    monkeypatch.setattr(policy, "schedule_decision_logging_enabled", lambda: True)
    monkeypatch.setattr(policy, "_log_frontier_vllm_v1_schedule_decision", events.append)
    monkeypatch.setattr(allocation, "schedule_decision_logging_enabled", lambda: True)
    actual = scheduler(replica_config, tmp_path, max_tokens_in_batch=64)
    request = Request(0.0, 64, 3)
    actual.add_request(request)
    for index in range(3):
        [scheduled] = actual.on_schedule(float(index * 2 + 1))
        complete(actual, scheduled, float(index * 2 + 2))
    allocations = [row for row in events if row["event"] == "grouped_cache_allocation"]
    ends = [row for row in events if row["event"] == "grouped_cache_batch_end"]
    releases = [row for row in events if row["event"] == "grouped_cache_release"]
    assert len(allocations) == len(ends) == 3 and len(releases) == 1
    assert [row["simulation_time"] for row in ends] == [2, 4, 6]
    assert [row["grouped_cache_state"]["settled_num_computed_tokens"] for row in ends] == [64, 65, 66]
    assert [row["grouped_cache_state"]["num_emitted_decode_tokens"] for row in ends] == [1, 2, 3]
    for row in allocations + ends + releases:
        state = row["grouped_cache_state"]
        assert state["pool_used_blocks"] + state["pool_free_blocks"] + state["pool_reserved_blocks"] == 29967
        assert state["fixed_cache_reservation_bytes"] == SUMMARY_BYTES_PER_SLOT
        assert state["physical_block_size_bytes"] == BF16_BLOCK_BYTES
    release = releases[0]
    assert release["simulation_time"] == 6 and release["batch_id"] == ends[-1]["batch_id"]
    state = release["grouped_cache_state"]
    assert state["group_block_ids"] == {} and state["block_ref_counts"] == {}
    assert not state["has_grouped_allocation"]
    assert state["scheduled_num_computed_tokens"] is None
    assert state["reclamation_num_computed_tokens"] is None
    assert state["request_allocated_blocks"] == state["scheduler_allocated_blocks"] == state["pool_used_blocks"] == 0
    assert state["pool_free_blocks"] == 29966


@pytest.mark.parametrize("fault", [
    "missing_release", "padding_frontier", "scratch_release", "sparse_release",
    "duplicate_owner", "wrong_end_time", "stale_scheduled_frontier", "wrong_new_ids",
])
def test_complete_controlled_cache_lifetime_rejects_incomplete_or_invalid_evidence(
    replica_config, tmp_path, monkeypatch, fault,
):
    from copy import deepcopy
    import frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy as policy
    import frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation as allocation
    from tests.integration.run_step5_structural import check_cache_lifetime

    events = []
    monkeypatch.setattr(policy, "schedule_decision_logging_enabled", lambda: True)
    monkeypatch.setattr(policy, "_log_frontier_vllm_v1_schedule_decision", events.append)
    monkeypatch.setattr(allocation, "schedule_decision_logging_enabled", lambda: True)
    actual = scheduler(replica_config, tmp_path)
    request = Request(0.0, 4105, 9)
    actual.add_request(request)
    stages = []
    for index in range(17):
        start = float(index * 2 + 1)
        [scheduled] = actual.on_schedule(start)
        stages.append({"batch_id": scheduled.id, "request_ids": [str(request.id)],
                       "request_num_tokens": scheduled.num_tokens,
                       "stage_start_ts": start, "stage_end_ts": start + 1})
        complete(actual, scheduled, start + 1)
    # JSON round-trip uses the same representation as the production log.
    events = json.loads(json.dumps(events))
    completion = {"completed_at": 34, "request_id": request.id}
    result = check_cache_lifetime(events, stages, completion)
    assert result == {"cache_state_events": 35, "kv_input_tokens": 4113,
                      "pool_num_blocks": 29967, "released_free_blocks": 29966,
                      "fixed_cache_reservation_bytes": 94484}
    altered = deepcopy(events)
    allocations = [row for row in altered if row["event"] == "grouped_cache_allocation"]
    ends = [row for row in altered if row["event"] == "grouped_cache_batch_end"]
    release = next(row for row in altered if row["event"] == "grouped_cache_release")
    if fault == "missing_release":
        altered.remove(release)
    elif fault == "padding_frontier":
        ends[-1]["grouped_cache_state"]["settled_num_computed_tokens"] += 7
    elif fault == "scratch_release":
        release["grouped_cache_state"]["fixed_cache_reservation_bytes"] = 0
    elif fault == "sparse_release":
        ends[-2]["grouped_cache_state"]["group_block_ids"]["sparse"][0] = None
    elif fault == "duplicate_owner":
        tables = allocations[0]["grouped_cache_state"]["group_block_ids"]
        tables["swa_1"][0] = tables["swa_0"][0]
    elif fault == "wrong_end_time":
        ends[-1]["simulation_time"] -= 1
    elif fault == "stale_scheduled_frontier":
        release["grouped_cache_state"]["scheduled_num_computed_tokens"] = 4113
    else:
        allocations[0]["group_new_block_ids"]["sparse"] = []
    with pytest.raises(ValueError):
        check_cache_lifetime(altered, stages, completion)


def test_disabled_state_log_does_not_walk_group_tables(replica_config, tmp_path, monkeypatch):
    import frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation as allocation
    from unittest.mock import Mock

    monkeypatch.setattr(allocation, "schedule_decision_logging_enabled", lambda: False)
    actual = scheduler(replica_config, tmp_path)
    actual._get_grouped_cache_state = Mock(side_effect=AssertionError("state logging is disabled"))
    request = Request(0.0, 64, 1)
    actual.add_request(request)
    [scheduled] = actual.on_schedule(1)
    complete(actual, scheduled, 2)
    actual._get_grouped_cache_state.assert_not_called()
    assert request.completed
