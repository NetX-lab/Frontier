"""NIXL transfers every prompt page but recomputes the last prompt token."""

import json

import pytest

from frontier.attention.dsa_features import build_dsa_features
from frontier.config import (
    AnalyticalKVCacheTransferConfig,
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.entities import Cluster, Request
from frontier.events.kv_cache_transfer_start_event import KVCacheTransferStartEvent
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType
from tests.unit.test_glm53_structural_pd import (
    _GlobalScheduler,
    _Metrics,
    _replica_config,
)


def _cluster(role):
    config = ClusterConfig(
        cluster_type=role,
        replica_config=_replica_config(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=128,
            num_blocks_mode="explicit",
            block_size=64,
            max_model_len=7168,
            max_tokens_in_batch=8192,
            batch_size_cap=1,
            enable_chunked_prefill=False,
        ),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=False
        ),
    )
    generator = SyntheticRequestGeneratorConfig()
    cluster = Cluster(config, MetricsConfig(write_metrics=False), generator)
    scheduler = RoundRobinClusterScheduler(
        config=config,
        cluster=cluster,
        request_generator_config=generator,
        kv_cache_transfer_predictor=AnalyticalKVCacheTransferPredictor(
            AnalyticalKVCacheTransferConfig(kv_connector="nixl")
        ),
    )
    return scheduler, next(iter(scheduler._replica_schedulers.values()))


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_nixl_full_prompt_transfer_samples_1024_outputs_at_native_contexts(
    monkeypatch, record_property, mode
):
    for name, value in (
        ("SYS_ARCH", "pd-disaggregation"), ("SIMULATION_MODE", mode),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("CUDAGRAPH_CAPTURE_SIZES", None), ("KV_CONNECTOR", "nixl"),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
        ("QUANTIZATION_MANAGER", None),
    ):
        monkeypatch.setattr(global_vars, name, value)
    source, p = _cluster(ClusterType.PREFILL)
    target, d = _cluster(ClusterType.DECODE)
    scheduler = _GlobalScheduler(source, target)
    metrics = _Metrics()
    request = Request(0.1, 6144, 1024)
    p.add_request(request)
    [prefill] = p.on_schedule(0.2)
    assert prefill.num_tokens == [6144]
    prefill.on_batch_end(0.5, ClusterType.PREFILL)
    p.on_batch_end(prefill)
    assert request.num_processed_tokens == 6144
    assert request.num_emitted_decode_tokens == 0
    assert request.first_decode_token_completed_at == 0

    [end] = KVCacheTransferStartEvent(
        1.0,
        source_replica_id=p._replica_id,
        source_replica_local_id=0,
        target_cluster_type=ClusterType.DECODE,
        batch=prefill,
        kv_cache_size_bytes=None,
        transfer_time_ms=None,
        kv_connector="nixl",
        source_batch_stage_id=0,
    ).handle_event(scheduler, metrics)
    info = end._transfer_info
    assert info.source_prompt_common_blocks == info.target_reserved_common_blocks == 96
    assert info.kv_cache_size_bytes == 569_106_432
    assert p._allocation_map[request.id] == d._allocation_map[request.id] == 96
    assert d._get_scheduler_num_computed_tokens(request) == 6143
    assert d.on_schedule(end.time - 0.001) == []
    end.handle_event(scheduler, metrics)
    assert request.id in d._kv_transfer_ready_request_ids
    assert p._allocation_map == {}
    assert len(metrics.starts) == len(metrics.ends) == 1

    scheduler_contexts, batch_contexts, key_lengths = [], [], []
    first_batch = None
    for step in range(1024):
        time = end.time + (step + 1) * 0.01
        scheduler_contexts.append(d._get_scheduler_num_computed_tokens(request))
        [batch] = d.on_schedule(time)
        assert batch.requests == [request]
        assert batch.num_tokens == [1]
        # This correction does not change the first-tail phase/graph contract.
        assert batch.request_is_decoding == [True]
        assert batch.num_prefill_tokens == 0
        assert batch.get_decode_cuda_graph_runtime_mode() == "NONE"
        features = build_dsa_features(
            batch.num_tokens, batch.num_context_tokens,
            target._config.replica_config.model_config.index_topk,
        )
        batch_contexts.append(batch.num_context_tokens[0])
        key_lengths.append(features.values["max_key_len"])
        assert features.path == "decode" and features.workload == "decode"
        batch.on_batch_end(time + 0.005, ClusterType.DECODE)
        assert request.num_emitted_decode_tokens == step + 1
        assert request.completed == (step == 1023)
        computed = d._get_scheduler_num_computed_tokens(request)
        d.on_batch_end(batch)
        if first_batch is None:
            first_batch = batch
    assert request.num_processed_tokens == 7168
    assert request.num_emitted_decode_tokens == 1024
    assert computed == 7167
    assert d._allocation_map == {}
    assert d._running_requests == []
    assert d._scheduled_num_computed_tokens_by_request == {}
    assert p._pending_kv_transfer_requests == set()
    evidence = {
        "mode": mode, "steps": len(batch_contexts),
        "source_prompt_pages": info.source_prompt_common_blocks,
        "transfer_bytes_per_worker_lane": info.kv_cache_size_bytes,
        "scheduler_contexts": scheduler_contexts,
        "batch_contexts": batch_contexts, "key_lengths": key_lengths,
        "final_computed": computed, "final_request_length": request.num_processed_tokens,
        "outputs": request.num_emitted_decode_tokens, "released": True,
    }
    record_property("nixl_context_loop", json.dumps(evidence))
    assert scheduler_contexts == list(range(6143, 7167))
    assert batch_contexts == list(range(6143, 7167))
    assert key_lengths == list(range(6144, 7168))
    assert first_batch.num_context_tokens == [6143]
