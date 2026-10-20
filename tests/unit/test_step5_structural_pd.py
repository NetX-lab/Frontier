"""Structural Step P/D handoff through the family payload and real V1 lanes.

Hand-computed bytes use the Step-5-Preview text configuration at attention
TP8, block size 16 and a BF16 cache: one local KV head of width 192, so a raw
K+V page is 16 * 192 * 2 * 2 = 12288 bytes per layer and rank. Sparse layers
keep two FP8 region summaries of width 256 per page (512 bytes) and one
open-region accumulator page: FP32 numerator and denominator of width 256, an
FP32 maximum and an int64 last slot, (2 * 256 + 1) * 4 + 8 = 2060 bytes.
"""

from math import ceil
from types import SimpleNamespace

import pytest

from frontier.attention.memory import (
    AttentionKVCacheSelection,
    get_attention_kv_transfer_payload,
)
from frontier.attention.step5_memory import resolve_step5_kv_cache_capacity
from frontier.config import (
    AnalyticalKVCacheTransferConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.entities import Batch, Replica, Request
from frontier.events.kv_cache_transfer_start_event import KVCacheTransferStartEvent
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.metrics.op_trace_utils import build_kv_cache_transfer_meta
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType

BLOCK_SIZE = 16
TENSOR_PARALLEL_SIZE = 8
RAW_PAGE_BYTES = 12288
SUMMARY_PAGE_BYTES = 512
OPEN_REGION_PAGE_BYTES = 2060
SWA_LAYERS = 69
SPARSE_LAYERS = 23
# cdiv(512-token window, 16-token page) + 1, as vLLM NIXL clips window groups.
SWA_TRANSFER_PAGES = 33
NUM_CACHE_GROUPS = 4


@pytest.fixture(autouse=True)
def simulation_globals(monkeypatch):
    for name, value in (
        ("SYS_ARCH", "pd-disaggregation"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("CUDAGRAPH_CAPTURE_SIZES", None), ("KV_CONNECTOR", "nixl"),
        ("ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC", False),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
        ("QUANTIZATION_MANAGER", None),
    ):
        monkeypatch.setattr(global_vars, name, value)


class _StageContexts:
    def __init__(self, replica_id: int) -> None:
        self._context = StageExecutionContext(replica_id=replica_id, stage_id=0, ep_size=32)

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        return self._context


class _Metrics:
    def __init__(self) -> None:
        self.starts = []
        self.ends = []

    def on_kv_cache_transfer_start(self, *args) -> None:
        self.starts.append(args)

    def on_kv_cache_transfer_end(self, *args) -> None:
        self.ends.append(args)

    def on_replica_schedule(self, *args, **kwargs) -> None:
        pass


class _GlobalScheduler:
    def __init__(self, source, target) -> None:
        self._clusters = {ClusterType.PREFILL: source, ClusterType.DECODE: target}

    def get_cluster_scheduler(self, cluster_type):
        return self._clusters[cluster_type]

    def get_cluster_logical_time(self, _cluster_type):
        return 0.0


def _replica_config() -> ReplicaConfig:
    return ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        num_pipeline_stages=1, attn_tensor_parallel_size=TENSOR_PARALLEL_SIZE, attn_dp=4,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=32,
        total_expert_num=352, router_topk=8,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )


@pytest.fixture(scope="module")
def step_config():
    return _replica_config()


@pytest.fixture(scope="module")
def capacity(step_config):
    return resolve_step5_kv_cache_capacity(
        step_config.model_config, BLOCK_SIZE,
        AttentionKVCacheSelection.from_replica(step_config, block_size=BLOCK_SIZE),
    )


def _payload(step_config, capacity, prompt_tokens, *, tensor_parallel_size=TENSOR_PARALLEL_SIZE):
    return get_attention_kv_transfer_payload(
        step_config.model_config, capacity, prompt_tokens=prompt_tokens,
        tensor_parallel_size=tensor_parallel_size,
        target_reserved_blocks=NUM_CACHE_GROUPS * ceil(prompt_tokens / BLOCK_SIZE),
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "pages", "open_pages", "per_rank_bytes", "logical_unique_bytes"),
    [
        # 4105 = 256 full pages + 9 tokens; 4105 % 8 = 1 leaves one open region.
        (4105, 257, 1,
         69 * 33 * 12288 + 23 * 257 * 12288 + 23 * 257 * 512 + 23 * 2060,
         4 * 69 * 33 * 12288 + 4 * 23 * 257 * 12288 + 23 * 257 * 512 + 23 * 2060),
        # 4096 = 256 full pages; every region is closed.
        (4096, 256, 0,
         69 * 33 * 12288 + 23 * 256 * 12288 + 23 * 256 * 512,
         4 * 69 * 33 * 12288 + 4 * 23 * 256 * 12288 + 23 * 256 * 512),
    ],
)
def test_component_bytes_match_hand_computation(
    step_config, capacity, prompt_tokens, pages, open_pages, per_rank_bytes,
    logical_unique_bytes,
):
    payload = _payload(step_config, capacity, prompt_tokens)
    components = {component.component_id: component for component in payload.components}

    assert {
        name: (len(c.owner_layer_ids), c.page_count, c.page_size_bytes, c.rank_replication)
        for name, c in components.items()
    } == {
        # KV heads 4 at TP8: one local head, repeated on 2 ranks.
        "swa_raw_kv": (SWA_LAYERS, SWA_TRANSFER_PAGES, RAW_PAGE_BYTES, 2),
        # Full history is never divided by the 8-token region size.
        "sparse_raw_kv": (SPARSE_LAYERS, pages, RAW_PAGE_BYTES, 2),
        "region_summary": (SPARSE_LAYERS, pages, SUMMARY_PAGE_BYTES, TENSOR_PARALLEL_SIZE),
        "open_region_state": (
            SPARSE_LAYERS, open_pages, OPEN_REGION_PAGE_BYTES, TENSOR_PARALLEL_SIZE,
        ),
    }
    assert payload.transfer_bytes_per_rank == per_rank_bytes
    assert payload.transfer_bytes_per_lane == TENSOR_PARALLEL_SIZE * per_rank_bytes
    assert payload.logical_unique_bytes == logical_unique_bytes
    assert payload.target_reserved_blocks == NUM_CACHE_GROUPS * pages
    assert payload.target_resident_bytes_per_rank == (
        NUM_CACHE_GROUPS * pages * capacity.physical_block_size_bytes
    )
    assert capacity.physical_block_size_bytes == SPARSE_LAYERS * (
        RAW_PAGE_BYTES + SUMMARY_PAGE_BYTES + OPEN_REGION_PAGE_BYTES
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "swa_pages", "sparse_pages", "open_pages"),
    [
        (7, 1, 1, 1), (8, 1, 1, 0), (9, 1, 1, 1),
        (15, 1, 1, 1), (16, 1, 1, 0), (17, 2, 2, 1),
        (527, 33, 33, 1), (528, 33, 33, 0), (529, 33, 34, 1),
    ],
)
def test_window_clip_and_open_region_follow_page_and_region_boundaries(
    step_config, capacity, prompt_tokens, swa_pages, sparse_pages, open_pages,
):
    pages = {
        component.component_id: component.page_count
        for component in _payload(step_config, capacity, prompt_tokens).components
    }
    assert pages == {
        "swa_raw_kv": swa_pages, "sparse_raw_kv": sparse_pages,
        "region_summary": sparse_pages, "open_region_state": open_pages,
    }


# Step checks its 64 Q heads first; every TP that divides them also divides
# or is a multiple of its 4 KV heads, so fractional layouts fail on Q heads.
@pytest.mark.parametrize("tensor_parallel_size", [3, 6])
def test_fractional_heads_fail_clearly(step_config, capacity, tensor_parallel_size):
    with pytest.raises(ValueError, match="Step Q heads must divide tensor_parallel_size"):
        _payload(step_config, capacity, 4105, tensor_parallel_size=tensor_parallel_size)


@pytest.mark.parametrize("sys_arch", ["co-location", "pd-disaggregation"])
def test_thinking_mode_fails_clearly(step_config, sys_arch):
    from frontier.config.config import SimulationConfig

    config = object.__new__(SimulationConfig)
    config.sys_arch = sys_arch
    config.enable_thinking_mode = True
    config.thinking_depth = 2
    config.tool_call_latency = 0.0
    config.thinking_round_prefill_tokens = None
    config.thinking_round_decode_tokens = None
    config.cluster_config = SimpleNamespace(replica_config=step_config)

    with pytest.raises(ValueError, match="not supported for Step structural execution"):
        config._validate_thinking_mode_config()


def test_trace_reports_rank_lane_unique_and_resident_bytes_separately(
    step_config, capacity,
):
    payload = _payload(step_config, capacity, 4105)
    batch = SimpleNamespace(requests=[SimpleNamespace(num_prefill_tokens=4105)])

    meta = build_kv_cache_transfer_meta(
        batch, step_config, ClusterType.PREFILL, 103_687_956, transfer_payload=payload,
    )

    assert meta["attention_family"] == "step5_attention"
    assert meta["byte_scope"] == "per_tp_rank_pages"
    assert meta["native_wire_bytes_known"] is False
    assert sum(meta["tensor_size_bytes"].values()) == meta["transfer_size_bytes"] == 103_687_956
    assert meta["transfer_bytes_per_lane"] == 829_503_648
    assert meta["logical_unique_bytes"] == 405_530_388
    assert meta["target_reserved_blocks"] == 1028
    assert meta["target_resident_bytes_per_rank"] == 1028 * 341_780
    with pytest.raises(ValueError, match="do not match the page payload"):
        build_kv_cache_transfer_meta(
            batch, step_config, ClusterType.PREFILL, 103_687_955, transfer_payload=payload,
        )


def _lane(cluster_type: ClusterType):
    replica_config = _replica_config()
    request_config = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, request_config, cluster_type)
    lane = VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=0, num_blocks_mode="memory_planner", block_size=BLOCK_SIZE,
            max_model_len=8192, max_tokens_in_batch=512, batch_size_cap=1,
            enable_chunked_prefill=True, watermark_blocks_fraction=0,
        ),
        request_generator_config=request_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_StageContexts(replica.id),
        cluster_type=cluster_type,
        replica_local_id=0,
    )
    return lane, replica_config


def _harness():
    source_lane, source_config = _lane(ClusterType.PREFILL)
    target_lane, _ = _lane(ClusterType.DECODE)
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_connector="nixl")
    )
    source = SimpleNamespace(
        _config=SimpleNamespace(
            replica_config=source_config,
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
        ),
        _kv_cache_transfer_predictor=predictor,
        get_replica_scheduler=lambda replica_id, local_id: source_lane,
    )
    target = object.__new__(RoundRobinClusterScheduler)
    target._cluster_type = ClusterType.DECODE
    target._cluster = SimpleNamespace(replicas={target_lane._replica_id: target_lane._replica})
    target._replica_dp_size = 1
    target._request_counter = 0
    target._replica_schedulers = {(target_lane._replica_id, 0): target_lane}
    return source_lane, target_lane, _GlobalScheduler(source, target), predictor


def _request_owned_by_prefill(lane, prompt_tokens: int) -> Request:
    request = Request(0.0, prompt_tokens, 9)
    lane._allocate_request(request, prompt_tokens)
    request._num_processed_tokens = prompt_tokens
    request._is_prefill_complete = True
    request._prefill_completed_at = 0.5
    lane._pending_kv_transfer_requests.add(request.id)
    return request


def _start_event(request: Request, source_lane) -> KVCacheTransferStartEvent:
    return KVCacheTransferStartEvent(
        1.0,
        source_replica_id=source_lane._replica_id,
        source_replica_local_id=0,
        target_cluster_type=ClusterType.DECODE,
        batch=Batch(
            replica_id=source_lane._replica_id, requests=[request],
            num_tokens=[request.num_prefill_tokens], is_moe=True,
        ),
        kv_cache_size_bytes=None,
        transfer_time_ms=None,
        kv_connector="nixl",
        source_batch_stage_id=0,
    )


@pytest.fixture
def harness():
    return _harness()


def test_handoff_reserves_every_decode_group_and_prices_one_rank(
    harness, step_config, capacity,
):
    source, target, scheduler, predictor = harness
    request = _request_owned_by_prefill(source, 4105)
    metrics = _Metrics()

    [end_event] = _start_event(request, source).handle_event(scheduler, metrics)
    info = end_event._transfer_info
    expected = _payload(step_config, capacity, 4105)

    assert info.source_prompt_common_blocks == 257
    # A fresh reservation at frontier 4104 holds 257 pages in each of 4 groups.
    assert info.target_reserved_common_blocks == target._allocation_map[request.id] == 1028
    assert info.transfer_payload == expected
    assert info.kv_cache_size_bytes == expected.transfer_bytes_per_rank == 103_687_956
    assert info.transfer_time_ms == predictor.get_transfer_time(
        ClusterType.PREFILL, ClusterType.DECODE, None, 103_687_956,
    )

    end_event._transfer_info.transfer_payload = _payload(step_config, capacity, 4104)
    with pytest.raises(RuntimeError, match="NIXL payload changed"):
        end_event.handle_event(scheduler, metrics)
    end_event._transfer_info.transfer_payload = expected

    end_event.handle_event(scheduler, metrics)
    assert request in target._waiting_requests
    assert request.id not in source._allocation_map
    assert len(metrics.starts) == len(metrics.ends) == 1


def test_first_decode_step_keeps_the_window_page_its_recompute_reads(harness):
    # D recomputes token 526 from frontier 526; its 512-token window starts at
    # token 15 in page 0. Retiring on the whole 527-token prompt would free it.
    source, target, scheduler, _ = harness
    request = _request_owned_by_prefill(source, 527)
    metrics = _Metrics()
    [end_event] = _start_event(request, source).handle_event(scheduler, metrics)
    end_event.handle_event(scheduler, metrics)
    swa_group = next(
        group for group in target._kv_cache_capacity.groups
        if group.retention == "sliding_window"
    )
    assert KVCacheManager._num_skipped_group_blocks(swa_group, 527) == 1
    assert KVCacheManager._num_skipped_group_blocks(swa_group, 526) == 0

    _, scheduled, _ = target._schedule_decode_waiting_requests(512)

    assert scheduled == [request]
    tables = target._kv_cache_manager.get_group_block_ids(request.id)
    for group in target._kv_cache_capacity.groups:
        assert len(tables[group.group_id]) == 33
        assert tables[group.group_id][0] is not None
