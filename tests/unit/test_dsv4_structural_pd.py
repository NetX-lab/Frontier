"""Grouped V4 NIXL lifecycle through real allocators, requests and events."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from frontier.config import AnalyticalKVCacheTransferConfig, ReplicaConfig, SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig
from frontier.config import global_vars
from frontier.entities import Replica, Request
from frontier.execution_time_predictor.deepseek_v4 import serving_context
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import AnalyticalKVCacheTransferPredictor, get_grouped_nixl_payload
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import StageExecutionContext
from frontier.types import ClusterType
from tests.unit.test_glm53_structural_pd import _GlobalScheduler, _Metrics, _source_cluster, _start_event, _target_cluster


@pytest.fixture(autouse=True)
def _nixl_context(monkeypatch):
    for name, value in (
        ("SYS_ARCH", "pd-disaggregation"), ("SIMULATION_MODE", "offline"),
        ("KV_CONNECTOR", "nixl"), ("USE_CUDA_GRAPH", False),
        ("DECODE_CUDA_GRAPH_MODE", "none"), ("CUDAGRAPH_CAPTURE_SIZES", None),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False), ("QUANTIZATION_MANAGER", None),
    ):
        monkeypatch.setattr(global_vars, name, value)


def _lane(role, *, blocks=4000):
    config = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h200", network_device="h200_dgx",
        attn_tensor_parallel_size=8, attn_dp=4, moe_tensor_parallel_size=1, moe_expert_parallel_size=32,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8", indexer_kv_cache_dtype="fp8",
    )
    generator = SyntheticRequestGeneratorConfig()
    replica = Replica(config, generator, role)
    context = StageExecutionContext(replica_id=replica.id, stage_id=0, ep_size=32)
    cluster = SimpleNamespace(
        get_stage_execution_context=lambda replica_id, stage_id: context,
        on_replica_batch_scheduled=lambda *args: None,
    )
    lane = VLLMv1EngineReplicaScheduler(
        replica_config=config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=blocks, num_blocks_mode="explicit", block_size=256,
            max_model_len=7169, max_tokens_in_batch=7169, batch_size_cap=8,
        ),
        request_generator_config=generator, replica=replica, predictor=object(),
        cluster_scheduler=cluster, cluster_type=role, replica_local_id=0,
    )
    return lane, config


def _harness():
    source, config = _lane(ClusterType.PREFILL)
    target, _ = _lane(ClusterType.DECODE)
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(kv_connector="nixl"))
    scheduler = _GlobalScheduler(_source_cluster(source, config, predictor), _target_cluster(target))
    return source, target, scheduler, predictor


def _prefill(source, prompt=6144, output=1024):
    request = Request(0.0, prompt, output)
    source.add_request(request)
    # Run the real per-owner scheduling iteration, not the four-owner DP busy
    # loop: no EP compute or peer idle forwards are executed in this component test.
    [batch] = source._schedule_iteration(0.0)
    assert batch.num_tokens == [prompt]
    initial = source._kv_cache_manager.get_group_block_ids(request.id)
    batch.on_batch_end(0.5, ClusterType.PREFILL)
    source.on_batch_end(batch)
    assert request.id in source._pending_kv_transfer_requests
    assert request.num_processed_tokens == prompt
    assert request.num_processed_decode_tokens == 0
    return request, initial


def _counts(tables):
    return [sum(block is not None for block in table) for table in tables.values()]


def _conserves(lane):
    manager = lane._kv_cache_manager
    ids = [block for request_id in lane._allocation_map for table in manager.get_group_block_ids(request_id).values() for block in table if block is not None]
    assert len(ids) == len(set(ids)) == manager.num_used_blocks == lane._num_allocated_blocks
    pool = manager.block_pool
    assert pool.get_num_free_blocks() + len(ids) + pool.reserved_blocks == lane._config.num_blocks
    assert [block.ref_cnt for block in pool.blocks] == [int(i in ids) for i in range(lane._config.num_blocks)]


@pytest.mark.parametrize(("prompt", "real_metrics"), [(191, False), (6144, False), (6145, False), (6144, True)])
def test_real_grouped_handoff_and_all_outputs_conserve(prompt, real_metrics, record_property, tmp_path):
    source, target, scheduler, _ = _harness()
    request, initial = _prefill(source, prompt)
    if prompt == 6144:
        assert list(map(len, initial.values())) == [24, 96, 96, 1536, 768]
    metrics = _Metrics()
    if real_metrics:
        from frontier.config import ClusterConfig, MetricsConfig
        from frontier.metrics.metrics_store import MetricsStore
        from frontier.metrics.trace_store import TraceStore

        cluster_configs = {
            role: ClusterConfig(
                cluster_type=role, replica_config=source._replica_config,
                replica_scheduler_config=source._config,
            )
            for role in (ClusterType.PREFILL, ClusterType.DECODE)
        }
        traces = TraceStore(str(tmp_path), enabled=True)
        traces.initialize({"case": "grouped_nixl_real_metrics"})
        metrics = MetricsStore(
            SimpleNamespace(metrics_config=MetricsConfig(
                output_dir=str(tmp_path), write_metrics=True, store_plots=False,
                enable_op_level_tracing=True,
            )),
            cluster_configs, traces,
        )
    start = _start_event(request, source)
    [end] = start.handle_event(scheduler, metrics)
    info = end._transfer_info
    if prompt == 6144:
        assert _counts(info.source_group_block_ids) == [24, 2, 2, 2, 16]
        assert _counts(info.target_group_block_ids) == [24, 2, 2, 2, 16]
        assert list(map(len, info.transferred_group_block_ids.values())) == [24, 2, 2, 2, 16]
        assert [source._kv_cache_capacity.get_group_size_bytes(group) for group in source._kv_cache_capacity.groups] == [1_002_240, 823_680, 786_240, 870_912, 656_640]
        assert info.kv_cache_size_bytes == 39_521_664
    assert info.kv_cache_size_bytes == sum(
        len(info.transferred_group_block_ids[group.group_id]) * source._kv_cache_capacity.get_group_size_bytes(group)
        for group in source._kv_cache_capacity.groups
    )
    assert info.source_prompt_common_blocks is info.target_reserved_common_blocks is None
    assert target._get_scheduler_num_computed_tokens(request) == prompt - 1
    assert target._waiting_requests == []
    assert target._schedule_iteration(0.75) == []
    assert request.id in source._allocation_map
    assert start.to_dict()["target_group_block_ids"] == info.to_dict()["target_group_block_ids"]
    _conserves(source)
    _conserves(target)
    end.handle_event(scheduler, metrics)
    assert info.target_ready_time == info.source_release_time == end.time
    assert source._allocation_map == {}
    assert source._pending_kv_transfer_requests == set()
    assert request.num_processed_decode_tokens == 0
    for output in range(1024):
        [batch] = target._schedule_iteration(end.time + output * 2)
        assert batch.num_tokens == [1]
        assert batch.num_context_tokens == [prompt - 1 + output]
        assert target._schedule_iteration(end.time + output * 2 + 0.25) == []
        _conserves(target)
        batch.on_batch_end(end.time + output * 2 + 1, ClusterType.DECODE)
        computed_frontier = target._get_scheduler_num_computed_tokens(request)
        target.on_batch_end(batch)
    assert request.completed
    assert request.num_processed_decode_tokens == 1024
    assert computed_frontier == prompt + 1023
    assert target._allocation_map == {}
    assert target._scheduled_num_computed_tokens_by_request == {}
    _conserves(source)
    _conserves(target)
    with pytest.raises(RuntimeError, match="no receiving"):
        end.handle_event(scheduler, metrics)
    if real_metrics:
        assert metrics._kv_cache_transfer_metrics["transfer_count"] == 1
        assert metrics._kv_cache_transfer_metrics["total_data_transferred"] == info.kv_cache_size_bytes
        [trace] = traces._buffer
        assert trace.name == "kv_cache_transfer" and trace.type == "TRANSFER"
        assert trace.meta["transfer_size_bytes"] == info.kv_cache_size_bytes
        assert trace.meta["payload_kind"] == info.payload_kind
        assert trace.meta["block_size"] == 256
        assert trace.meta["tensor_shape"] == {}
        assert sum(trace.meta["tensor_size_bytes"].values()) == info.kv_cache_size_bytes
        assert [group["source_held_pages"] for group in trace.meta["cache_groups"]] == [24, 2, 2, 2, 16]
        assert [group["target_reserved_pages"] for group in trace.meta["cache_groups"]] == [24, 2, 2, 2, 16]
        assert [group["transferred_pages"] for group in trace.meta["cache_groups"]] == [24, 2, 2, 2, 16]
        assert sum(group["payload_bytes"] for group in trace.meta["cache_groups"]) == info.kv_cache_size_bytes
        assert sum(component["resident_bytes"] for component in trace.meta["cache_components"]) == info.kv_cache_size_bytes
        assert trace.meta["source_group_block_ids"] == {key: list(value) for key, value in info.source_group_block_ids.items()}
        assert trace.meta["target_group_block_ids"] == {key: list(value) for key, value in info.target_group_block_ids.items()}
        assert trace.meta["transferred_group_block_ids"] == {key: list(value) for key, value in info.transferred_group_block_ids.items()}
        assert trace.meta["native_wire_bytes_known"] is False
        traces.close()
        written = [json.loads(line) for line in (tmp_path / "op_traces.jsonl").read_text().splitlines()]
        assert len(written) == 2 and written[1]["meta"] == trace.meta
        record_property("grouped_nixl_real_metrics_trace", json.dumps(written[1]))
    else:
        assert len(metrics.starts) == len(metrics.ends) == 1
    record_property("grouped_nixl_lifecycle", json.dumps({
        "prompt": prompt, "outputs": request.num_processed_decode_tokens,
        "initial_source_groups": list(map(len, initial.values())),
        "handoff_source_groups": _counts(info.source_group_block_ids),
        "destination_reserved_groups": _counts(info.target_group_block_ids),
        "transferred_groups": list(map(len, info.transferred_group_block_ids.values())),
        "payload_bytes_per_worker_lane": info.kv_cache_size_bytes,
        "first_context": prompt - 1, "last_context": prompt + 1022,
        "computed_frontier": computed_frontier,
        "source_free_blocks": source._kv_cache_manager.block_pool.get_num_free_blocks(),
        "target_free_blocks": target._kv_cache_manager.block_pool.get_num_free_blocks(),
        "pool_blocks": target._config.num_blocks, "reserved_null_blocks": 1,
        "native_wire_bytes_known": False,
        "real_metrics_callback": real_metrics,
    }))


@pytest.mark.parametrize("owner", ["source", "target"])
@pytest.mark.parametrize("fault", ["missing", "empty", "zero", "wrong_length", "duplicate", "wrong_null"])
def test_payload_rejects_missing_zero_or_incorrect_groups(owner, fault):
    source, target, scheduler, _ = _harness()
    request, _ = _prefill(source)
    [end] = _start_event(request, source).handle_event(scheduler, _Metrics())
    info = end._transfer_info
    source_groups = dict(info.source_group_block_ids)
    target_groups = dict(info.target_group_block_ids)
    tables = source_groups if owner == "source" else target_groups
    key = "compressed"
    if fault == "missing":
        tables.pop(key)
    elif fault == "empty":
        tables[key] = ()
    elif fault == "zero":
        tables[key] = (0,) + tables[key][1:]
    elif fault == "wrong_length":
        tables[key] = tables[key][:-1]
    elif fault == "duplicate":
        tables[key] = (tables[key][1],) + tables[key][1:]
    else:
        tables["swa_0"] = (tables[key][0],) + tables["swa_0"][1:]
    with pytest.raises(ValueError, match="NIXL"):
        get_grouped_nixl_payload(source._kv_cache_capacity, request.num_prefill_tokens, source_groups, target_groups)


@pytest.mark.parametrize("fault", ["source", "target", "selected", "bytes"])
def test_completion_rejects_changed_group_snapshot_before_metrics_or_release(fault):
    source, target, scheduler, _ = _harness()
    request, _ = _prefill(source)
    metrics = _Metrics()
    [end] = _start_event(request, source).handle_event(scheduler, metrics)
    info = end._transfer_info
    if fault == "bytes":
        info.kv_cache_size_bytes += 1
    else:
        tables = {"source": info.source_group_block_ids, "target": info.target_group_block_ids, "selected": info.transferred_group_block_ids}[fault]
        tables["compressed"] = tuple(reversed(tables["compressed"]))
    with pytest.raises(RuntimeError, match="changed"):
        end.handle_event(scheduler, metrics)
    assert metrics.ends == []
    assert request.id in source._allocation_map
    assert request.id in target._kv_transfer_receiving_request_ids
    assert request.num_processed_decode_tokens == 0
    _conserves(source)
    _conserves(target)


def test_failed_payload_rolls_back_real_grouped_reservation(monkeypatch):
    source, target, scheduler, predictor = _harness()
    request, _ = _prefill(source)
    monkeypatch.setattr(predictor, "get_structural_nixl_transfer_info_for_request", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad payload")))
    with pytest.raises(ValueError, match="bad payload"):
        _start_event(request, source).handle_event(scheduler, _Metrics())
    assert target._allocation_map == {}
    assert target._kv_transfer_receiving_request_ids == set()
    assert target._scheduled_num_computed_tokens_by_request == {}
    assert source._kv_transfer_start_claimed_request_ids == set()
    _conserves(source)
    _conserves(target)


def test_capacity_refusal_retains_p_then_reserves_after_actual_free():
    source, target, scheduler, _ = _harness()
    blocker = Request(0.0, 6144, 1024)
    target._allocate_request(blocker, 7167, scheduler_num_computed_tokens=0)
    blocker._num_processed_tokens = 7167
    blocker._is_prefill_complete = True
    target._scheduled_num_computed_tokens_by_request[blocker.id] = 7166
    target.add_request(blocker)
    request, _ = _prefill(source, output=1)
    start = _start_event(request, source)
    metrics = _Metrics()
    start.handle_event(scheduler, metrics)
    assert metrics.starts == []
    assert request.id not in target._allocation_map
    assert request.id in source._allocation_map
    assert target.has_pending_kv_transfer_starts()
    [last] = target._schedule_iteration(2.0)
    last.on_batch_end(3.0, ClusterType.DECODE)
    target.on_batch_end(last)
    assert blocker.completed
    assert blocker.id not in target._allocation_map
    assert target.consume_popped_output_events() == [start]
    assert not target.has_pending_kv_transfer_starts()
    assert start.time == 3.0
    [end] = start.handle_event(scheduler, metrics)
    end.handle_event(scheduler, metrics)
    [tail] = target._schedule_iteration(4.0)
    tail.on_batch_end(5.0, ClusterType.DECODE)
    target.on_batch_end(tail)
    assert request.completed
    assert target._allocation_map == {}
    _conserves(source)
    _conserves(target)


def test_grouped_transfer_info_rejects_scalar_shortcut_and_empty_group():
    source, target, scheduler, _ = _harness()
    request, _ = _prefill(source)
    [end] = _start_event(request, source).handle_event(scheduler, _Metrics())
    with pytest.raises(ValueError, match="scalar"):
        replace(end._transfer_info, source_prompt_common_blocks=24)
    groups = dict(end._transfer_info.transferred_group_block_ids, compressed=())
    with pytest.raises(ValueError, match="positive"):
        replace(end._transfer_info, transferred_group_block_ids=groups)


def test_grouped_reservation_rejects_capacity_mismatch_before_placement():
    source, target, scheduler, _ = _harness()
    request, _ = _prefill(source)
    target._kv_cache_capacity = replace(target._kv_cache_capacity, zero_new_blocks=False)
    with pytest.raises(ValueError, match="capacities must match"):
        _start_event(request, source).handle_event(scheduler, _Metrics())
    assert target._allocation_map == {}


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("connector", ["none", "p2p_nccl"])
def test_other_grouped_pd_connectors_remain_unsupported(monkeypatch, role, connector):
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", connector)
    with pytest.raises(NotImplementedError, match="NIXL PREFILL/DECODE"):
        _lane(role)


@pytest.mark.parametrize("dummy", [False, True])
def test_grouped_transport_does_not_admit_v4_execution(dummy):
    predictor = SimpleNamespace(_enable_dummy_mode=dummy)
    with pytest.raises(NotImplementedError, match="eager co-location without PD" if dummy else "measured/native execution remains unqualified"):
        serving_context(predictor, None, 0)


@pytest.mark.parametrize("owner", ["source", "target"])
def test_completion_rejects_actual_same_count_group_ownership_change(owner):
    source, target, scheduler, _ = _harness()
    request, _ = _prefill(source)
    metrics = _Metrics()
    [end] = _start_event(request, source).handle_event(scheduler, metrics)
    lane = source if owner == "source" else target
    table = lane._kv_cache_manager._group_tables[request.id]["compressed"]
    table.reverse()
    with pytest.raises(RuntimeError, match="ownership changed"):
        end.handle_event(scheduler, metrics)
    assert metrics.ends == []
    assert request.id in source._allocation_map
    assert request.id in target._kv_transfer_receiving_request_ids
    _conserves(source)
    _conserves(target)
