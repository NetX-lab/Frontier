"""Structural GLM P/D handoff through real V1 cache allocators and events."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.config import (
    AnalyticalKVCacheTransferConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.config import global_vars
from frontier.entities import Batch, ExecutionTime, Replica, Request, StageExecutionTime
from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent
from frontier.events.kv_cache_transfer_start_event import KVCacheTransferStartEvent
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.metrics.op_trace_utils import build_kv_cache_transfer_meta
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import (
    LORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.scheduler.utils.dense_metrics import build_prefill_metrics_execution_time
from frontier.types import ClusterType


class _StageContexts:
    def __init__(self, replica_id: int) -> None:
        self._context = StageExecutionContext(
            replica_id=replica_id, stage_id=0, ep_size=8
        )

    def get_stage_execution_context(self, replica_id: int, stage_id: int):
        assert (replica_id, stage_id) == (self._context.replica_id, 0)
        return self._context


class _Metrics:
    def __init__(self) -> None:
        self.starts = []
        self.ends = []
        self.replica_schedules = []

    def on_kv_cache_transfer_start(self, *args) -> None:
        self.starts.append(args)

    def on_kv_cache_transfer_end(self, *args) -> None:
        self.ends.append(args)

    def on_replica_schedule(self, *args, **kwargs) -> None:
        self.replica_schedules.append((args, kwargs))


class _GlobalScheduler:
    def __init__(self, source, target) -> None:
        self._clusters = {
            ClusterType.PREFILL: source,
            ClusterType.DECODE: target,
        }

    def get_cluster_scheduler(self, cluster_type):
        return self._clusters[cluster_type]

    def get_cluster_logical_time(self, _cluster_type):
        return 0.0


def _replica_config() -> ReplicaConfig:
    return ReplicaConfig(
        model_name="zai-org/GLM-5.3",
        device="h800",
        network_device="h800_dgx",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        total_expert_num=256,
        router_topk=8,
    )


def _lane(
    cluster_type: ClusterType,
    *,
    num_blocks: int = 8,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = False,
):
    replica_config = _replica_config()
    request_config = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, request_config, cluster_type)
    lane = VLLMv1EngineReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=num_blocks,
            num_blocks_mode="explicit",
            block_size=64,
            max_model_len=max_model_len,
            max_tokens_in_batch=256,
            batch_size_cap=8,
            enable_prefix_caching=enable_prefix_caching,
        ),
        request_generator_config=request_config,
        replica=replica,
        predictor=object(),
        cluster_scheduler=_StageContexts(replica.id),
        cluster_type=cluster_type,
        replica_local_id=0,
    )
    return lane, replica_config


def _source_cluster(lane, replica_config, predictor):
    return SimpleNamespace(
        _config=SimpleNamespace(
            replica_config=replica_config,
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
        ),
        _kv_cache_transfer_predictor=predictor,
        get_replica_scheduler=lambda replica_id, local_id: lane,
    )


def _target_cluster(lane, policy=RoundRobinClusterScheduler):
    scheduler = object.__new__(policy)
    scheduler._cluster_type = ClusterType.DECODE
    scheduler._cluster = SimpleNamespace(replicas={lane._replica_id: lane._replica})
    scheduler._replica_dp_size = 1
    scheduler._request_counter = 0
    scheduler._replica_schedulers = {(lane._replica_id, 0): lane}
    return scheduler


def _request_owned_by_prefill(lane, prompt_tokens: int) -> Request:
    request = Request(0.0, prompt_tokens, 3)
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
            replica_id=source_lane._replica_id,
            requests=[request],
            num_tokens=[request.num_prefill_tokens],
            is_moe=True,
        ),
        kv_cache_size_bytes=None,
        transfer_time_ms=None,
        kv_connector="nixl",
        source_batch_stage_id=0,
    )


def _harness(*, target_blocks: int = 8, target_max_model_len: int = 4096):
    source_lane, source_config = _lane(ClusterType.PREFILL)
    target_lane, _ = _lane(
        ClusterType.DECODE,
        num_blocks=target_blocks,
        max_model_len=target_max_model_len,
    )
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_connector="nixl")
    )
    source = _source_cluster(source_lane, source_config, predictor)
    target = _target_cluster(target_lane)
    return source_lane, target_lane, _GlobalScheduler(source, target), predictor


# The D reservation holds exactly the source prompt pages, as vLLM's remote
# load does, with no block for the next token.
@pytest.mark.parametrize(("prompt_tokens", "prompt_blocks"), [(64, 1), (65, 2)])
def test_structural_transfer_counts_only_source_backed_prompt_pages(
    prompt_tokens: int, prompt_blocks: int
) -> None:
    source, target, scheduler, _ = _harness()
    request = _request_owned_by_prefill(source, prompt_tokens)
    metrics = _Metrics()

    [end_event] = _start_event(request, source).handle_event(scheduler, metrics)
    info = end_event._transfer_info

    assert source._allocation_map[request.id] == prompt_blocks
    assert target._allocation_map[request.id] == prompt_blocks
    assert info.source_prompt_common_blocks == prompt_blocks
    assert info.target_reserved_common_blocks == prompt_blocks
    assert info.kv_cache_size_bytes == (
        prompt_blocks * source._kv_cache_capacity.resident_bytes_per_common_block
    )
    assert info.payload_kind == "structural_whole_prompt_pages_per_worker_lane"

    arrival_events = end_event.handle_event(scheduler, metrics)

    assert len(arrival_events) == 1
    assert request in target._waiting_requests
    assert request.id not in source._allocation_map
    assert request.id not in source._pending_kv_transfer_requests
    assert info.target_ready_time == end_event.time
    assert info.source_release_time == end_event.time
    assert len(metrics.starts) == len(metrics.ends) == 1


def test_generic_nixl_sizing_is_fail_closed() -> None:
    _, _, _, predictor = _harness()
    request = Request(0.0, 64, 3)

    with pytest.raises(NotImplementedError, match="reservation-backed"):
        predictor.get_kv_cache_size_for_request(request, _replica_config())


def test_structural_start_rejects_source_prefix_before_side_effects() -> None:
    source, target, scheduler, _ = _harness()
    source, source_config = _lane(
        ClusterType.PREFILL, enable_prefix_caching=True
    )
    predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(kv_connector="nixl")
    )
    source_cluster = _source_cluster(source, source_config, predictor)
    scheduler._clusters[ClusterType.PREFILL] = source_cluster
    request = _request_owned_by_prefill(source, 64)
    metrics = _Metrics()

    with pytest.raises(NotImplementedError, match="source prefix caching"):
        _start_event(request, source).handle_event(scheduler, metrics)

    assert source._kv_transfer_start_claimed_request_ids == set()
    assert target._allocation_map == {}
    assert scheduler._clusters[ClusterType.DECODE]._request_counter == 0
    assert metrics.starts == []


def test_decode_policy_without_target_selection_fails_as_configuration_error() -> None:
    source, target, scheduler, _ = _harness()
    lor_target = _target_cluster(target, policy=LORClusterScheduler)
    scheduler._clusters[ClusterType.DECODE] = lor_target
    request = _request_owned_by_prefill(source, 64)

    # Lane validation belongs to every DECODE policy; selection does not.
    lor_target.validate_kv_transfer_target_scope(
        source._kv_cache_capacity, request
    )
    with pytest.raises(ValueError, match="LORClusterScheduler does not"):
        _start_event(request, source).handle_event(scheduler, _Metrics())

    assert target._allocation_map == {}


def test_failed_payload_rolls_back_the_reserved_decode_frontier(monkeypatch) -> None:
    source, target, scheduler, predictor = _harness()
    request = _request_owned_by_prefill(source, 128)

    def fail_payload(**_):
        raise ValueError("payload construction failed")

    monkeypatch.setattr(
        predictor, "get_structural_nixl_transfer_info_for_request", fail_payload
    )
    with pytest.raises(ValueError, match="payload construction failed"):
        _start_event(request, source).handle_event(scheduler, _Metrics())

    assert target._allocation_map == {}
    assert target._scheduled_num_computed_tokens_by_request == {}
    assert target._kv_transfer_receiving_request_ids == set()
    assert source._kv_transfer_start_claimed_request_ids == set()


def test_structural_transfer_trace_uses_dsa_cache_components(monkeypatch) -> None:
    replica_config = _replica_config()
    request = Request(0.0, 65, 3)
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[request.num_prefill_tokens],
        is_moe=True,
    )
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")

    meta = build_kv_cache_transfer_meta(
        batch,
        replica_config,
        ClusterType.PREFILL,
        transfer_size_bytes=2 * 5_928_192,
    )

    assert meta["payload_kind"] == "structural_whole_prompt_pages_per_worker_lane"
    assert meta["byte_scope"] == "per_worker_lane_resident_pages"
    assert meta["source_prompt_common_blocks"] == 2
    assert meta["dtype"] == "MIXED"
    assert meta["tensor_dtypes"] == {
        "main_mla": "BF16",
        "indexer_k": "FP8",
        "indexer_scale": "FP32",
    }
    assert meta["tensor_shape"] == {
        "main_mla": [2, 78, 64, 576],
        "indexer_k": [2, 21, 64, 128],
        "indexer_scale": [2, 21, 64, 1],
    }
    assert meta["tensor_size_bytes"] == {
        "main_mla": 11_501_568,
        "indexer_k": 344_064,
        "indexer_scale": 10_752,
    }
    assert sum(meta["tensor_size_bytes"].values()) == meta["transfer_size_bytes"]
    assert meta["num_kv_heads"] is None
    assert meta["head_dim"] is None
    assert meta["native_wire_bytes_known"] is False


def test_structural_transfer_trace_rejects_non_page_payload(monkeypatch) -> None:
    replica_config = _replica_config()
    request = Request(0.0, 64, 3)
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[request.num_prefill_tokens],
        is_moe=True,
    )
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")

    with pytest.raises(ValueError, match="complete common blocks"):
        build_kv_cache_transfer_meta(
            batch,
            replica_config,
            ClusterType.PREFILL,
            transfer_size_bytes=5_928_191,
        )


def _timing_layer(*, family: str, variant: str, op_times: dict[str, float]):
    return ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        global_layer_id=0,
        attention_family_id=family,
        attention_variant_id=variant,
        op_times=op_times,
    )


def test_dense_metrics_preserve_dsa_attention_when_adding_dense_ffn() -> None:
    attention_only = _timing_layer(
        family="dsa_attention",
        variant="full_index",
        op_times={
            "attn_dsa_indexer": 2.0,
            "attn_dsa_mla": 3.0,
            "attn_tensor_parallel_allreduce": 5.0,
        },
    )
    dense_probe = _timing_layer(
        family="mla",
        variant="mla",
        op_times={
            "attn_kv_cache_save": 11.0,
            "attn_prefill": 13.0,
            "mlp_up_proj": 17.0,
            "mlp_act": 19.0,
            "mlp_down_proj": 23.0,
            "mlp_tensor_parallel_allreduce": 29.0,
        },
    )
    predictor = SimpleNamespace(
        _enable_dummy_mode=True,
        _predict_dsa_dummy_attention_time=(
            lambda _layer_id: attention_only.attention_time_component
        ),
        predict_stage_execution_time=Mock(
            return_value=StageExecutionTime((dense_probe,))
        )
    )
    model = SimpleNamespace(
        is_moe=True,
        num_layers=2,
        get_num_moe_layers=lambda: 1,
        is_moe_layer=lambda layer_id: layer_id == 1,
    )

    corrected = build_prefill_metrics_execution_time(
        original_execution_time=StageExecutionTime((attention_only,)),
        sample_batch=object(),
        predictor=predictor,
        stage_id=0,
        cluster_type=ClusterType.PREFILL,
        model_config=model,
    ).layer_execution_times[0]

    assert corrected.attention_family_id == "dsa_attention"
    assert corrected.attn_dsa_indexer_time == 2.0
    assert corrected.attn_dsa_mla_time == 3.0
    assert corrected.attention_kv_cache_save_execution_time == 0.0
    assert corrected.attention_prefill_execution_time == 0.0
    assert corrected.attention_all_reduce_time == 5.0
    assert corrected.mlp_layer_up_proj_execution_time == 17.0
    assert corrected.mlp_layer_act_execution_time == 19.0
    assert corrected.mlp_layer_down_proj_execution_time == 23.0
    assert corrected.mlp_all_reduce_time == 29.0


def test_duplicate_and_late_events_reject_before_metrics_or_release() -> None:
    source, target, scheduler, _ = _harness()
    request = _request_owned_by_prefill(source, 64)
    metrics = _Metrics()
    event = _start_event(request, source)
    [end_event] = event.handle_event(scheduler, metrics)
    allocated = dict(target._allocation_map)

    with pytest.raises(RuntimeError, match="already started"):
        event.handle_event(scheduler, metrics)
    with pytest.raises(RuntimeError, match="Duplicate NIXL transfer start"):
        _start_event(request, source).handle_event(scheduler, metrics)

    assert len(metrics.starts) == 1
    assert target._allocation_map == allocated

    end_event.handle_event(scheduler, metrics)
    with pytest.raises(RuntimeError, match="no receiving NIXL reservation"):
        KVCacheTransferEndEvent(end_event.time, end_event._transfer_info).handle_event(
            scheduler, metrics
        )
    assert len(metrics.ends) == 1
    assert request.id not in source._allocation_map


def test_target_identity_error_keeps_prefill_and_metrics_untouched() -> None:
    source, target, scheduler, _ = _harness()
    request = _request_owned_by_prefill(source, 65)
    metrics = _Metrics()
    [end_event] = _start_event(request, source).handle_event(scheduler, metrics)
    end_event._transfer_info.target_reserved_common_blocks += 1

    with pytest.raises(RuntimeError, match="reservation changed"):
        end_event.handle_event(scheduler, metrics)

    assert len(metrics.ends) == 0
    assert request.id in source._allocation_map
    assert request.id in source._pending_kv_transfer_requests
    assert request.id in target._kv_transfer_receiving_request_ids


def test_payload_identity_error_keeps_prefill_and_metrics_untouched() -> None:
    source, target, scheduler, _ = _harness()
    request = _request_owned_by_prefill(source, 65)
    metrics = _Metrics()
    [end_event] = _start_event(request, source).handle_event(scheduler, metrics)
    end_event._transfer_info.kv_cache_size_bytes -= 1

    with pytest.raises(RuntimeError, match="NIXL payload changed"):
        end_event.handle_event(scheduler, metrics)

    assert len(metrics.ends) == 0
    assert request.id in source._allocation_map
    assert request.id in source._pending_kv_transfer_requests
    assert request.id in target._kv_transfer_receiving_request_ids


def test_target_request_limits_reject_before_claim_or_placement() -> None:
    source, target, scheduler, _ = _harness(target_max_model_len=128)
    request = _request_owned_by_prefill(source, 65)
    request._num_decode_tokens = 65
    metrics = _Metrics()

    with pytest.raises(ValueError, match="max_model_len"):
        _start_event(request, source).handle_event(scheduler, metrics)

    assert source._kv_transfer_start_claimed_request_ids == set()
    assert scheduler._clusters[ClusterType.DECODE]._request_counter == 0
    assert target._allocation_map == {}
    assert metrics.starts == []


def test_near_full_target_preserves_fifo_and_prefill_ownership() -> None:
    source, target, scheduler, _ = _harness(target_blocks=4)
    occupied = Request(0.0, 127, 3)
    target._allocate_request(occupied, 127)
    # vLLM reserves ceil(prompt / 64) blocks for a remote load: three for
    # the head and two for the tail, exactly the source prompt pages.
    head = _request_owned_by_prefill(source, 192)
    tail = _request_owned_by_prefill(source, 128)
    metrics = _Metrics()
    head_event = _start_event(head, source)
    tail_event = _start_event(tail, source)

    head_event.handle_event(scheduler, metrics)
    tail_event.handle_event(scheduler, metrics)

    assert target._pending_kv_transfer_start_events == [head_event, tail_event]
    assert set(source._allocation_map) == {head.id, tail.id}
    assert metrics.starts == []

    target._free_request_resources(occupied)
    target._retry_pending_kv_transfer_starts(2.0)

    assert target._popped_output_events == [head_event]
    assert target._pending_kv_transfer_start_events == [tail_event]
    assert target._allocation_map[head.id] == 3
    assert tail.id not in target._allocation_map

    [head_end] = target._popped_output_events.pop(0).handle_event(
        scheduler, metrics
    )
    assert head_end._transfer_info.source_prompt_common_blocks == 3
    assert head_end._transfer_info.target_reserved_common_blocks == 3
    assert set(source._allocation_map) == {head.id, tail.id}

    head_end.handle_event(scheduler, metrics)
    assert head in target._waiting_requests
    assert set(source._allocation_map) == {tail.id}
    assert target._pending_kv_transfer_start_events == [tail_event]

    target._free_request_resources(head)
    target._retry_pending_kv_transfer_starts(head_end.time + 1.0)
    assert target._popped_output_events == [tail_event]
    assert target._pending_kv_transfer_start_events == []

    [tail_end] = target._popped_output_events.pop(0).handle_event(
        scheduler, metrics
    )
    tail_end.handle_event(scheduler, metrics)
    assert tail_end._transfer_info.source_prompt_common_blocks == 2
    assert tail_end._transfer_info.target_reserved_common_blocks == 2
    assert source._allocation_map == {}
    assert len(metrics.starts) == len(metrics.ends) == 2

    target._free_request_resources(tail)
    assert target._allocation_map == {}


def test_d_ready_reservation_progresses_before_blocked_growth_head() -> None:
    source, target, scheduler, _ = _harness(target_blocks=4)
    old = Request(0.0, 127, 20)
    old._num_processed_tokens = 127
    old._is_prefill_complete = True
    old._prefill_completed_at = 0.5
    old.on_arrival(0.5, ClusterType.DECODE)
    target.add_request(old)
    assert target._schedule_decode_waiting_requests(256)[1] == [old]

    old._num_processed_tokens = 128
    target._scheduled_num_computed_tokens_by_request[old.id] = 128
    # Its two-block reservation fills the pool, so the old head cannot grow.
    incoming = _request_owned_by_prefill(source, 128)
    metrics = _Metrics()
    [end_event] = _start_event(incoming, source).handle_event(scheduler, metrics)
    preempted = []
    assert target._schedule_running_requests(256, preempted)[1] == []
    assert preempted == [old]

    end_event.handle_event(scheduler, metrics)
    assert target._waiting_requests == [old, incoming]
    assert target._kv_transfer_ready_request_ids == {incoming.id}
    assert target._schedule_decode_waiting_requests(0)[1] == []
    assert target._kv_transfer_ready_request_ids == {incoming.id}

    _, admitted, _ = target._schedule_decode_waiting_requests(256)
    assert admitted == [incoming]
    # Its first D step recomputes the last prompt token in the reserved blocks.
    assert target._allocation_map[incoming.id] == 2
    assert target._waiting_requests == [old]
    assert target._kv_transfer_ready_request_ids == set()

    target._running_requests.remove(incoming)
    target._free_request_resources(incoming)
    _, resumed, _ = target._schedule_decode_waiting_requests(256)
    assert resumed == [old]
    assert target._waiting_requests == []

    target._running_requests.remove(old)
    target._free_request_resources(old)
    assert target._allocation_map == {}
    assert target._kv_transfer_receiving_request_ids == set()
    assert target._kv_transfer_ready_request_ids == set()
    assert source._allocation_map == {}


def test_d_ready_admission_preserves_order_within_each_class() -> None:
    _, target, _, _ = _harness(target_blocks=8)
    ordinary_a = Request(0.0, 64, 3)
    ready_a = Request(0.1, 64, 3)
    ordinary_b = Request(0.2, 64, 3)
    ready_b = Request(0.3, 64, 3)
    for request in (ordinary_a, ready_a, ordinary_b, ready_b):
        request._num_processed_tokens = request.num_prefill_tokens
        request._is_prefill_complete = True
        request._prefill_completed_at = 0.5
        request.on_arrival(0.5, ClusterType.DECODE)
    for request in (ready_a, ready_b):
        assert target.reserve_kv_transfer_target(request) == 1
        target.mark_kv_transfer_target_ready(request)
    target._waiting_requests = [ordinary_a, ready_a, ordinary_b, ready_b]

    _, admitted, _ = target._schedule_decode_waiting_requests(1)

    assert admitted == [ready_a]
    assert target._waiting_requests == [ready_b, ordinary_a, ordinary_b]
    assert target._kv_transfer_ready_request_ids == {ready_b.id}


def test_d_ready_marker_without_allocation_rejects_before_reordering() -> None:
    _, target, _, _ = _harness()
    request = Request(0.0, 64, 3)
    request._num_processed_tokens = request.num_prefill_tokens
    request._is_prefill_complete = True
    request._prefill_completed_at = 0.5
    request.on_arrival(0.5, ClusterType.DECODE)
    target._waiting_requests = [request]
    target._kv_transfer_ready_request_ids.add(request.id)

    with pytest.raises(RuntimeError, match="no reserved allocation"):
        target._schedule_decode_waiting_requests(1)


def test_decode_waiting_order_is_unchanged_without_d_ready_reservations() -> None:
    _, target, _, _ = _harness(target_blocks=8)
    first = Request(0.0, 64, 3)
    second = Request(0.1, 64, 3)
    for request in (first, second):
        request._num_processed_tokens = request.num_prefill_tokens
        request._is_prefill_complete = True
        request._prefill_completed_at = 0.5
        request.on_arrival(0.5, ClusterType.DECODE)
    target._waiting_requests = [first, second]

    _, admitted, _ = target._schedule_decode_waiting_requests(1)

    assert admitted == [first]
    assert target._waiting_requests == [second]
    assert target._kv_transfer_ready_request_ids == set()


def test_legacy_transfer_dicts_remain_byte_compatible() -> None:
    request = Request(0.0, 16, 2)
    batch = Batch(replica_id=0, requests=[request], num_tokens=[16], is_moe=False)
    event = KVCacheTransferStartEvent(
        1.0,
        source_replica_id=0,
        source_replica_local_id=0,
        target_cluster_type=ClusterType.DECODE,
        batch=batch,
        kv_cache_size_bytes=1024,
        transfer_time_ms=0.5,
        source_batch_stage_id=0,
    )

    assert "kv_connector" not in event.to_dict()
    info = event.handle_event(None, _Metrics())[0]._transfer_info
    assert "kv_connector" not in info.to_dict()
