"""Allocation identities reach one synthetic before-forward execution scope."""

from dataclasses import FrozenInstanceError

import pytest

from frontier.entities import Request, StageExecutionTime
from frontier.types import ClusterType
from test_dsv4_source_execution import predictor
from test_grouped_kv_cache_scheduler import _v4_schedulers


def test_scheduled_batches_keep_new_zeroing_ids_without_replay():
    scheduler = _v4_schedulers(blocks=80, chunk=189)[0]
    first_metadata = None
    for cycle in range(2):
        request = Request(float(cycle * 10), 190, 3)
        scheduler.add_request(request)
        for step in range(4):
            now = float(cycle * 10 + step * 2)
            batch, = scheduler.on_schedule(now)
            metadata = batch.cache_block_zeroing
            if step == 0:
                assert tuple(sorted(metadata.block_ids)) == tuple(range(1, 80))
                # Approved packed-layout oracle: max group sum is 1,002,240B.
                assert metadata.physical_block_size_bytes == 1002240
                assert metadata.physical_reservation_extent_bytes == 79176960
                with pytest.raises(FrozenInstanceError):
                    metadata.block_ids = ()
                if first_metadata is None:
                    first_metadata = metadata
            else:
                assert metadata is None
            assert scheduler.on_schedule(now + 0.25) == []
            batch.on_batch_end(now + 1.0, ClusterType.MONOLITHIC)
            scheduler.on_batch_end(batch)
        assert request.completed
    assert first_metadata.block_ids == tuple(range(1, 80))


@pytest.mark.parametrize("tp_rank", [0, 7])
def test_real_allocation_has_one_before_forward_scope_per_worker(predictor, tp_rank):
    from frontier.execution_time_predictor.deepseek_v4 import (
        component_ledger, serving_context, timing_operator_times,
    )

    scheduler = _v4_schedulers(blocks=80, chunk=189)[0]
    scheduler.add_request(Request(0.0, 190, 3))
    batch, = scheduler.on_schedule(0.0)
    metadata = batch.cache_block_zeroing
    layers = [predictor._predict_dsv4_dummy_structural_execution(
        batch, layer, tp_rank, include_ffn=False, include_stage_owned=layer == 0,
        **serving_context(predictor, batch, layer),
    ) for layer in range(43)]
    stage = StageExecutionTime(layers)
    assert stage.op_times["kv_cache_block_zero"] == 2.0
    assert sum("kv_cache_block_zero" in layer.op_times for layer in layers) == 1
    phase = timing_operator_times(layers[0], "attention")
    assert phase[0] == ("COMPUTE", "kv_cache_block_zero", 2.0)
    assert phase[1][1] == "dsv4_embedding"
    assert sum(component_ledger(stage).values()) == pytest.approx(stage.total_time * 1000)
    calls = layers[0]._dsv4_workloads["kv_cache_block_zero"]
    assert calls["block_ids"] == tuple(range(1, 80))
    assert calls["block_assignment_count"] == calls["unique_block_count"] == 79
    assert calls["physical_reservation_extent_bytes"] == 79176960
    assert calls["native_store_traffic_bytes"] == "UNKNOWN"
    assert calls["native_h2d_cost"] == "UNKNOWN"
    repeated = predictor._predict_dsv4_dummy_structural_execution(
        batch, 0, tp_rank, include_ffn=False, include_stage_owned=True,
        **serving_context(predictor, batch, 0),
    )
    assert repeated.op_times == layers[0].op_times
    assert batch.cache_block_zeroing is metadata


def test_compression_page_growth_is_new_work_but_retention_is_not(predictor):
    from frontier.execution_time_predictor.deepseek_v4 import serving_context

    scheduler = _v4_schedulers(blocks=80, chunk=4)[0]
    request = Request(0.0, 5, 2)
    scheduler.add_request(request)
    for step, expected in enumerate(((1, 2, 3, 4, 5), (6,), ())):
        batch, = scheduler.on_schedule(float(step * 2))
        metadata = batch.cache_block_zeroing
        assert (() if metadata is None else metadata.block_ids) == expected
        timing = predictor._predict_dsv4_dummy_structural_execution(
            batch, 0, 0, include_ffn=False,
            **serving_context(predictor, batch, 0),
        )
        assert timing.op_times.get("kv_cache_block_zero", 0.0) == (2.0 if expected else 0.0)
        batch.on_batch_end(float(step * 2 + 1), ClusterType.MONOLITHIC)
        scheduler.on_batch_end(batch)
    assert request.completed


def test_multiple_requests_share_one_group_ordered_assignment_stream():
    scheduler = _v4_schedulers(blocks=30, chunk=8)[0]
    requests = [Request(0.0, 4, 1), Request(0.0, 4, 1)]
    for request in requests:
        scheduler.add_request(request)
    batch, = scheduler.on_schedule(0.0)
    assert batch.num_tokens == [4, 4]
    assert batch.cache_block_zeroing.block_ids == (1, 6, 2, 7, 3, 8, 4, 9, 5, 10)
    assert batch.cache_block_zeroing.physical_reservation_extent_bytes == 10022400


def test_priority_rollback_keeps_the_assignment_of_a_preempted_request():
    scheduler = _v4_schedulers(blocks=40, chunk=16, policy="priority")[0]
    victim, survivor = Request(0.0, 4, 24), Request(0.0, 190, 3)
    scheduler.add_request(victim)
    scheduler.add_request(survivor)
    metadata = None
    for step in range(11):
        batch, = scheduler.on_schedule(float(step * 2))
        if step == 0:
            victim.set_priority(10)
            survivor.set_priority(0)
        if step == 9:
            assert batch.requests == [survivor]
            assert victim.get_total_preemption_count() == 1
            metadata = batch.cache_block_zeroing
            assert metadata.block_ids == (24, 27, 10, 15, 28, 29, 1)
        if step == 10:
            assert 24 in batch.cache_block_zeroing.block_ids
            assert metadata.block_ids == (24, 27, 10, 15, 28, 29, 1)
        batch.on_batch_end(float(step * 2 + 1), ClusterType.MONOLITHIC)
        scheduler.on_batch_end(batch)


def test_failed_waiting_admission_adds_no_work_before_later_reuse():
    scheduler = _v4_schedulers(blocks=80, chunk=189)[0]
    first, waiting = Request(0.0, 190, 3), Request(0.0, 190, 3)
    scheduler.add_request(first)
    scheduler.add_request(waiting)
    for step in range(5):
        batch, = scheduler.on_schedule(float(step * 2))
        assert batch.requests == [first if step < 4 else waiting]
        if step in (1, 2, 3):
            assert batch.cache_block_zeroing is None
        else:
            assert set(batch.cache_block_zeroing.block_ids) == set(range(1, 80))
        batch.on_batch_end(float(step * 2 + 1), ClusterType.MONOLITHIC)
        scheduler.on_batch_end(batch)


def test_request_free_dp_peer_does_not_inherit_cache_zero_work(predictor):
    from frontier.entities import DummyForwardBatch
    from frontier.execution_time_predictor.deepseek_v4 import serving_context

    active, idle = _v4_schedulers(blocks=80, chunk=4)
    active.add_request(Request(0.0, 5, 2))
    active_batch, = active.on_schedule(0.0)
    idle_batch, = idle.on_schedule(0.0)
    assert active_batch.cache_block_zeroing is not None
    assert isinstance(idle_batch, DummyForwardBatch)
    assert idle_batch.cache_block_zeroing is None
    timing = predictor._predict_dsv4_dummy_structural_execution(
        idle_batch, 0, 0, include_ffn=False,
        **serving_context(predictor, idle_batch, 0),
    )
    assert "kv_cache_block_zero" not in timing.op_times


def test_repeated_assignment_ids_do_not_inflate_unique_reservation_extent():
    from frontier.entities.batch import KVCacheBlockZeroing

    metadata = KVCacheBlockZeroing((3, 7, 3), 1002240)
    assert metadata.block_ids == (3, 7, 3)
    assert metadata.physical_reservation_extent_bytes == 2004480


@pytest.mark.parametrize("expand", [False, True])
def test_real_stage_trace_places_cache_zero_before_embedding(predictor, tmp_path, expand):
    from types import SimpleNamespace
    from frontier.config import (
        ClusterConfig, MetricsConfig, SimulationConfig, global_vars,
        get_quantization_manager,
    )
    from frontier.entities import BatchStage
    from frontier.execution_time_predictor.deepseek_v4 import serving_context
    from frontier.metrics.metrics_store import MetricsStore

    global_vars.reset_global_vars()
    get_quantization_manager().configure_from_model_config(predictor._model_config)
    scheduler = _v4_schedulers(blocks=80, chunk=4)[0]
    scheduler.add_request(Request(0.0, 5, 2))
    batch, = scheduler.on_schedule(0.0)
    cluster = ClusterConfig(replica_config=scheduler._replica_config)
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(
        output_dir=str(tmp_path), write_metrics=True, store_plots=False,
        store_operation_metrics=True, store_utilization_metrics=False,
        enable_op_level_tracing=True, enable_per_layer_expansion=expand,
        num_requests_to_trace_per_layer=1 if expand else 0))
    sink = SimpleNamespace(events=[])
    sink.log_event = sink.events.append
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, sink)
    layers = [predictor._predict_dsv4_dummy_structural_execution(
        batch, layer, 0, include_ffn=False, include_stage_owned=layer == 0,
        **serving_context(predictor, batch, layer),
    ) for layer in range(43)]
    stage = StageExecutionTime(layers)
    batch_stage = BatchStage(batch.id, 0, 0, stage.total_time, stage.model_time,
                             batch.requests, batch.num_tokens, batch.request_is_decoding,
                             ClusterType.MONOLITHIC)
    batch_stage.on_schedule(1.0)
    store.on_replica_stage_schedule(time=1.0, replica_id=0, stage_id=0,
                                   batch_stage=batch_stage, execution_time=stage,
                                   cluster_type=ClusterType.MONOLITHIC, replica_local_id=0)
    physical = [event for event in sink.events if event.type in {"COMPUTE", "COMM"}]
    zero, = [event for event in physical if event.name == "kv_cache_block_zero"]
    embedding, = [event for event in physical if event.name == "dsv4_embedding"]
    assert physical[0] is zero
    assert zero.duration_ms == 2.0
    assert zero.ts_start + zero.duration_ms / 1000 <= embedding.ts_start + 1e-12
    assert sum(event.duration_ms for event in physical) == pytest.approx(sum(stage.op_times.values()))
    assert sum(event.duration_ms for event in sink.events) == pytest.approx(stage.total_time * 1000)
    call, = zero.meta["source_tensor_calls"]
    assert call["block_ids"] == (1, 2, 3, 4, 5)
    assert call["physical_reservation_extent_bytes"] == 5011200
    assert call["native_store_traffic_bytes"] == "UNKNOWN"
    assert "synthetic" in zero.meta["cost_source"]
    from frontier.metrics.constants import OperationMetrics
    metric = store._operation_metrics[ClusterType.MONOLITHIC][OperationMetrics.KV_CACHE_BLOCK_ZERO]
    assert metric.sum == 2.0
    global_vars.reset_global_vars()


def test_legacy_model_metrics_do_not_gain_empty_cache_zero_tables(tmp_path):
    from frontier.config import ClusterConfig, MetricsConfig, ReplicaConfig, SimulationConfig, global_vars
    from frontier.metrics.constants import OperationMetrics
    from frontier.metrics.metrics_store import MetricsStore

    global_vars.reset_global_vars()
    cluster = ClusterConfig(replica_config=ReplicaConfig(model_name="meta-llama/Llama-2-7b-hf"))
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(
        output_dir=str(tmp_path), write_metrics=True, store_plots=False))
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster})
    assert OperationMetrics.KV_CACHE_BLOCK_ZERO not in store._operation_metrics[ClusterType.MONOLITHIC]
    assert OperationMetrics.KV_CACHE_BLOCK_ZERO not in store._operation_metrics_per_batch[ClusterType.MONOLITHIC]
    global_vars.reset_global_vars()
