"""Real stage-event first-call contracts; all costs are unadmitted CPU fixtures."""

from types import SimpleNamespace

import pytest

from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.config import global_vars
from frontier.config.model_config import BaseModelConfig
from frontier.config.replica_scheduler_config import VllmV1SchedulerConfig
from frontier.entities import Batch, DummyForwardBatch, EPBatchGroup
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.entities.batch import KVCacheBlockZeroing
from frontier.execution_time_predictor.kv_block_zeroing_host_time import KVBlockZeroingHostTime
from frontier.execution_time_predictor.step5 import step5_row_features
from frontier.execution_time_predictor.step5_cost import step5_batch_features, step5_batch_workload
from frontier.profiling.step5.inputs import Step5ProfileInput
from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.types import ClusterType, MeasurementType
from test_monolithic_mixed_forward_sync import (
    _build_scheduler, _decode_batch, _global, _MetricsStore, _prefill_batch,
)
from test_step5_mrv2_inputs import producer_row


def _harness(role, *, dense_layers=frozenset(), request_capacity=32):
    scheduler, predictor, context, stages = _build_scheduler(
        cluster_type=role, dense_layers=dense_layers,
    )
    predictor.requires_forward_start_agreement = True
    config = VllmV1SchedulerConfig(batch_size_cap=request_capacity)
    for engine in scheduler._replica_schedulers.values():
        engine._config = config
        engine._max_num_running_reqs = request_capacity
    predictor._model_config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    predictor._cluster_type = role
    predictor._block_size = 16
    predictor._step5_max_model_len = 8192
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8, moe_tensor_parallel_size=1, moe_expert_parallel_size=1,
        moe_all2all_backend=None,
    )
    predictor._step5_attention_workload = lambda batch: step5_batch_workload(predictor, batch)
    scheduler.get_replica = lambda _: SimpleNamespace(is_moe=True)
    observed = []
    original = predictor.predict_stage_execution_time

    def predict(batch, stage_id, cluster_type=None, **kwargs):
        if not isinstance(batch, EPBatchGroup):
            workload = step5_batch_workload(predictor, batch)
            observed.append((batch, workload))
        value = original(batch, stage_id, cluster_type, **kwargs)
        value.forward_preparation_time = 0.
        value.final_residual_time = 0.
        value.forward_launch_stall_time = lambda _: 0.
        return value

    predictor.predict_stage_execution_time = predict
    return scheduler, predictor, context, stages, observed


def _enqueue(stage, batch, lane):
    batch.set_global_id(lane)
    batch._forward_cohort_id = lane
    batch._stage_owner_replica_local_id = lane
    stage.add_batch(batch)


def _event(role, lane, time=1.):
    return ReplicaStageScheduleEvent(time, 0, 0, role, lane)


@pytest.fixture(autouse=True)
def _globals(monkeypatch):
    monkeypatch.setattr(global_vars, "SYS_ARCH", "pd-disaggregation")
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [1, 2, 4, 8, 16, 24, 32])


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("case,mode,widths", [
    ("decode_dummy", "FULL", [4, 4]),
    ("decode_real", "FULL", [4, 4]),
    ("prefill_dummy", "PIECEWISE", [16, 16]),
    ("prefill_real", "PIECEWISE", [16, 16]),
    ("miss_dummy", "NONE", [40, 1]),
    ("eager_real", "NONE", [3, 1]),
])
def test_stage_event_first_prediction_binds_all_useful_and_dummy_lanes(role, case, mode, widths,
                                                                   monkeypatch):
    scheduler, predictor, context, stages, observed = _harness(role)
    if case == "eager_real":
        monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "none")
    batches = {
        "decode_dummy": [_decode_batch(3), DummyForwardBatch(0, 0, 0)],
        "decode_real": [_decode_batch(3), _decode_batch(1)],
        "prefill_dummy": [_prefill_batch(13), DummyForwardBatch(0, 0, 0)],
        "prefill_real": [_prefill_batch(13), _prefill_batch(5)],
        "miss_dummy": [_prefill_batch(40), DummyForwardBatch(0, 0, 0)],
        "eager_real": [_prefill_batch(3), _decode_batch(1)],
    }[case]
    snapshots = [(list(b.num_tokens), list(b.num_context_tokens), list(b.requests)) for b in batches]
    for lane, batch in enumerate(batches):
        _enqueue(stages[lane], batch, lane)
    # Existing local graph metadata is not an agreement marker.
    batches[0].decode_cuda_graph_metadata = SimpleNamespace(runtime_mode="FULL")
    _event(role, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert observed == [], "no lane may predict before all group members arrive"
    assert _event(role, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False)) == []
    resumes = _event(role, 1, time=2.).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert observed == [] and len(resumes) == 2
    assert all(isinstance(event, ReplicaStageScheduleEvent) and event.time == 2. for event in resumes)
    assert context.forward_start_resolved and context.forward_group_sealed
    for event in resumes:
        follow = event.handle_event(_global(scheduler), _MetricsStore(reporting=False))
        assert len(follow) == 1
        assert event.handle_event(_global(scheduler), _MetricsStore(reporting=False)) == []
    assert [batch for batch, _ in observed] == batches
    assert [w.graph_mode for _, w in observed] == [mode.lower()] * 2
    assert [w.physical_query_rows for _, w in observed] == widths
    for lane, (batch, workload) in enumerate(observed):
        assert (batch.num_tokens, batch.num_context_tokens, batch.requests) == snapshots[lane]
        assert stages[lane].held_forward_start_batch is None and stages[lane].is_busy
        assert context.is_active(batch._stage_admission_ticket)
        if isinstance(batch, DummyForwardBatch):
            dummy = batch.mrv2_dummy_descriptor
            assert isinstance(dummy, MRv2DummyDescriptor)
            assert workload.query_lengths == dummy.query_lens
            assert workload.has_attention_metadata and not batch.requests
            assert dummy.max_query_len is None
            assert dummy.num_reqs == (widths[lane] if mode == "FULL" else None if mode == "PIECEWISE" else 1)
            assert dummy.query_lens == ((1,) * widths[lane] if mode == "FULL" else (widths[lane],))
            features, _ = step5_batch_features(predictor, batch, 3)
            cell = Step5ProfileInput.from_mrv2_dummy(3, dummy, block_table_columns=512)
            source = producer_row(predictor._model_config, cell, duration=.5, count=17)
            assert features == step5_row_features(source), "runtime dummy must match the actual producer geometry"


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
def test_empty_lane_wakeup_and_next_group_reset_are_exactly_once(role):
    from frontier.events.replica_schedule_event import ReplicaScheduleEvent
    scheduler, _, context, stages, observed = _harness(role)
    scheduler.get_replica_scheduler(0, 1).engine_loop_idle = True
    real = _decode_batch(1)
    _enqueue(stages[0], real, 0)
    [wake] = _event(role, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert isinstance(wake, ReplicaScheduleEvent) and wake._replica_local_id == 1
    assert _event(role, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False)) == []
    # Repeated wake requests cannot issue a second engine event.
    assert not context.claim_forward_start_wakeup(1)
    dummy = DummyForwardBatch(0, 0, 0)
    _enqueue(stages[1], dummy, 1)
    events = _event(role, 1).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    for event in events:
        event.handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert len(observed) == 2
    group = real.forward_start_descriptor.group_id
    for batch, stage in zip((real, dummy), stages.values()):
        context.release(batch._stage_admission_ticket)
        stage.on_stage_end()
    assert context.is_idle and not context.forward_start_resolved
    assert not context.forward_group_sealed and context.queued_tickets == ()
    next_real = _decode_batch(1)
    _enqueue(stages[0], next_real, 0)
    [wake] = _event(role, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert isinstance(wake, ReplicaScheduleEvent)
    assert next_real._forward_cohort_provisional_id > group
    assert len(observed) == 2


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("arrival_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("policy,capacity,left,right,mode,widths", [
    ("full_and_piecewise", 3, ("decode", 2), ("decode", 1), "FULL", [2, 2]),
    ("full_and_piecewise", 4, ("decode", 3), ("decode", 1), "FULL", [4, 4]),
    ("full_and_piecewise", 3, ("decode", 3), ("decode", 1), "PIECEWISE", [4, 4]),
    ("full_decode_only", 3, ("decode", 3), ("decode", 1), "NONE", [3, 1]),
    ("full_and_piecewise", 3, ("prefill", 7), ("decode", 2), "PIECEWISE", [8, 8]),
    ("full_and_piecewise", 3, ("prefill", 7), ("dummy", 1), "PIECEWISE", [8, 8]),
    ("piecewise", 3, ("decode", 3), ("decode", 1), "PIECEWISE", [4, 4]),
    ("none", 3, ("decode", 3), ("decode", 1), "NONE", [3, 1]),
    ("full_and_piecewise", 3, ("prefill", 9), ("dummy", 1), "NONE", [9, 1]),
    ("full_decode_only", 3, ("prefill", 7), ("decode", 2), "NONE", [7, 2]),
])
def test_stage_first_call_redispatch_uses_request_supported_graph_candidates(
    role, arrival_order, policy, capacity, left, right, mode, widths, monkeypatch,
):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", policy)
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [8, 2, 1, 4])
    scheduler, _, context, stages, observed = _harness(role, request_capacity=capacity)
    assert all(engine._config.batch_size_cap == engine._max_num_running_reqs == capacity
               for engine in scheduler._replica_schedulers.values())
    builders = {"decode": _decode_batch, "prefill": _prefill_batch,
                "dummy": lambda _: DummyForwardBatch(0, 0, 0)}
    batches = [builders[kind](tokens) for kind, tokens in (left, right)]
    snapshots = [(list(batch.num_tokens), list(batch.num_context_tokens), list(batch.requests))
                 for batch in batches]
    for lane, batch in enumerate(batches):
        _enqueue(stages[lane], batch, lane)
    metrics = _MetricsStore(reporting=False)
    _event(role, arrival_order[0]).handle_event(_global(scheduler), metrics)
    assert observed == []
    resumes = _event(role, arrival_order[1]).handle_event(_global(scheduler), metrics)
    assert len(resumes) == 2 and observed == []
    tickets = [batch._stage_admission_ticket for batch in batches]
    for event in resumes:
        event.handle_event(_global(scheduler), metrics)
    assert [batch for batch, _ in observed] == batches
    assert [workload.graph_mode for _, workload in observed] == [mode.lower()] * 2
    assert [workload.physical_query_rows for _, workload in observed] == widths
    for lane, (batch, workload) in enumerate(observed):
        assert (batch.num_tokens, batch.num_context_tokens, batch.requests) == snapshots[lane]
        assert batch._stage_admission_ticket == tickets[lane] and context.is_active(tickets[lane])
        assert batch._stage_owner_replica_local_id == lane
        descriptor = batch.forward_start_descriptor
        assert descriptor.graph_mode == mode and descriptor.max_query_len is None
        assert descriptor.uniform_token_count == (1 if mode == "FULL" else None)
        assert descriptor.num_reqs == (
            widths[lane] if mode == "FULL" else None if mode == "PIECEWISE" else len(batch.num_tokens)
        )
        if mode == "FULL":
            assert descriptor.num_reqs <= capacity
        if isinstance(batch, DummyForwardBatch):
            assert batch.request_ids == [] and batch.request_execution_signatures == []
            assert batch.mrv2_dummy_descriptor.query_lens == workload.query_lengths
        else:
            assert workload.query_lengths == tuple(snapshots[lane][0])
            assert workload.sequence_lengths == tuple(
                query + context for query, context in zip(snapshots[lane][0], snapshots[lane][1])
            )


def test_native_none_ep_aggregate_keeps_local_widths():
    from frontier.scheduler.utils.batch_builders import build_virtual_global_batch
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start

    global_vars.DECODE_CUDA_GRAPH_MODE = "none"
    real, dummy = _prefill_batch(9), DummyForwardBatch(0, 0, 0)
    bind_forward_start([real, dummy], group_id=1, request_capacity=32)
    result = prepare_ep_wave_inputs(
        source_batches={0: real, 1: dummy}, batch=real, step_id_getter=lambda _: 1,
        aggregate_batch_builder=build_virtual_global_batch,
        cluster_type=ClusterType.PREFILL,
    )
    assert result.aggregate_batch.total_num_tokens == 10
    assert result.aggregate_batch.get_effective_total_tokens_for_compute(ClusterType.PREFILL) == 10
    assert result.total_step_tokens == 10


def test_missing_first_call_descriptor_fails_clearly():
    _, predictor, _, _, _ = _harness(ClusterType.DECODE)
    with pytest.raises(ValueError, match="forward-start descriptor"):
        step5_batch_workload(predictor, _decode_batch(1))


def test_held_dispatched_batch_preserves_its_query_context_after_preemption():
    scheduler, _, _, stages, observed = _harness(ClusterType.DECODE)
    real, dummy = _decode_batch(1), DummyForwardBatch(0, 0, 0)
    _enqueue(stages[0], real, 0)
    _event(ClusterType.DECODE, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    request = real.requests[0]
    request._is_prefill_complete = False
    request._num_processed_tokens = 0
    _enqueue(stages[1], dummy, 1)
    resumes = _event(ClusterType.DECODE, 1).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    for event in resumes:
        event.handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert observed[0][0] is real
    assert observed[0][1].query_lengths == (1,) and observed[0][1].sequence_lengths == (9,)
    assert observed[0][1].phase == "decode" and real.request_is_decoding == [True]


def test_prefill_one_token_extension_uses_full_descriptor_without_changing_owned_credit():
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start

    _, predictor, _, _, _ = _harness(ClusterType.PREFILL)
    real = _prefill_batch(16)
    real._num_tokens = [1]
    real._total_num_tokens = real._num_prefill_tokens = 1
    real._num_context_tokens = [15]
    dummy = DummyForwardBatch(0, 0, 0)
    bind_forward_start([real, dummy], group_id=1, request_capacity=32)
    features, workload = step5_batch_features(predictor, real, 3)
    assert workload.phase == "decode" and workload.graph_mode == "full"
    assert features["batch_num_prefill_tokens"] == 0 and features["batch_num_decode_tokens"] == 1
    assert real.num_prefill_tokens == 1 and real.num_decode_tokens == 0


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("policy,requests,mode,widths", [
    ("full_and_piecewise", 1, "FULL", [1, 1]),
    ("full_and_piecewise", 3, "PIECEWISE", [4, 4]),
    ("full_decode_only", 3, "NONE", [3, 1]),
])
def test_empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival(
    role, policy, requests, mode, widths, monkeypatch,
):
    from frontier.config import ReplicaConfig, SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig
    from frontier.entities import Replica
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler

    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", policy)
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [1, 2, 4, 8])
    scheduler, predictor, _, stages, observed = _harness(role)
    replica_config = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16, total_expert_num=352, router_topk=8,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    generator = SyntheticRequestGeneratorConfig()
    replica = Replica(replica_config, generator, role)
    replica._id = 0
    config = VllmV1SchedulerConfig(
        num_blocks=0, num_blocks_mode="memory_planner", max_model_len=8192,
        block_size=16, max_tokens_in_batch=32, batch_size_cap=3,
        watermark_blocks_fraction=0,
    )
    engines = [VLLMv1EngineReplicaScheduler(
        replica_config=replica_config, replica_scheduler_config=config,
        request_generator_config=generator, replica=replica, predictor=predictor,
        cluster_scheduler=scheduler, cluster_type=role, replica_local_id=lane,
    ) for lane in range(2)]
    scheduler._replica_schedulers = {(0, lane): engine for lane, engine in enumerate(engines)}
    scheduler.get_replica = lambda _: replica
    # An already-dispatched real lane has started the native DP busy-loop wave.
    for engine in engines:
        engine._dp_wave_running = True
        profile = engine._get_iteration_scheduler_profile()
        assert profile["round_class"] is None and profile["max_num_running_reqs"] == 3
    assert all(engine._max_num_running_reqs == config.batch_size_cap == 3 for engine in engines)
    real = _decode_batch(requests)
    snapshot = (list(real.num_tokens), list(real.num_context_tokens), list(real.requests))
    _enqueue(stages[0], real, 0)
    metrics = _MetricsStore(reporting=False)
    [wake] = _event(role, 0).handle_event(_global(scheduler), metrics)
    [arrival] = wake.handle_event(_global(scheduler), metrics)
    assert isinstance(arrival._batch, DummyForwardBatch) and arrival._batch.requests == []
    [schedule] = arrival.handle_event(_global(scheduler), metrics)
    resumes = schedule.handle_event(_global(scheduler), metrics)
    assert len(resumes) == 2 and observed == []
    for event in resumes:
        event.handle_event(_global(scheduler), metrics)
    assert len(observed) == 2 and engines[1]._dummy_forwards_in_flight == 1
    assert [workload.graph_mode for _, workload in observed] == [mode.lower()] * 2
    assert [workload.physical_query_rows for _, workload in observed] == widths
    assert (real.num_tokens, real.num_context_tokens, real.requests) == snapshot
    dummy = observed[1][0]
    assert dummy.requests == [] and dummy.request_execution_signatures == []
    for batch, _ in observed:
        descriptor = batch.forward_start_descriptor
        assert descriptor.max_query_len is None
        assert descriptor.uniform_token_count == (1 if mode == "FULL" else None)
        assert descriptor.num_reqs == (
            descriptor.num_tokens if mode == "FULL" else
            None if mode == "PIECEWISE" else len(batch.num_tokens)
        )
        if mode == "FULL":
            assert descriptor.num_reqs <= config.batch_size_cap


@pytest.mark.parametrize("policy", ["full_and_piecewise", "full_decode_only", "piecewise"])
def test_p_role_requires_honest_cuda_event_dummy_coverage(policy):
    from frontier.execution_time_predictor.step5_cost import step5_measurements_for_role

    assert step5_measurements_for_role(ClusterType.PREFILL, policy) == (
        MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT,
    )


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("dense_layers", [frozenset(), frozenset((0, 1, 2))])
def test_real_event_layer_lifecycle_releases_group_and_credits_no_dummy_requests(role, dense_layers):
    from frontier.events.batch_stage_end_event import BatchStageEndEvent

    scheduler, _, context, stages, _ = _harness(role, dense_layers=dense_layers)
    real = _prefill_batch(13) if role is ClusterType.PREFILL else _decode_batch(3)
    dummy = DummyForwardBatch(0, 0, 0)
    for lane, batch in enumerate((real, dummy)):
        _enqueue(stages[lane], batch, lane)
    metrics = _MetricsStore(reporting=False)
    scheduler._create_prefill_corrected_execution_time_for_metrics = lambda *args: None
    events = []
    for lane in range(2):
        events.extend(_event(role, lane).handle_event(_global(scheduler), metrics))
    ends = []
    while events:
        event = min(events, key=lambda item: (item.time, item.id))
        events.remove(event)
        if isinstance(event, BatchStageEndEvent):
            ends.append(event)
        else:
            events.extend(event.handle_event(_global(scheduler), metrics))
    assert len(ends) == 2 and {event._batch.id for event in ends} == {real.id, dummy.id}
    assert dummy.requests == [] and dummy.num_tokens == [1]
    assert dummy.request_ids == [] and dummy.request_execution_signatures == []
    if role is ClusterType.DECODE:
        assert [request.completed_layer_count for request in real.requests] == [4] * 3
    metrics.on_batch_stage_end = lambda *_: None
    scheduler.get_replica_scheduler(0, 1).on_dummy_forward_end = lambda _: False
    scheduler.get_replica_scheduler(0, 1).consume_paced_dummy_end = lambda _: False
    for event in ends:
        event.handle_event(_global(scheduler), metrics)
    assert context.is_idle and not context.forward_start_resolved
    assert context.queued_tickets == () and not context.forward_group_sealed
    assert all(not stage.is_busy and stage.held_forward_start_batch is None for stage in stages.values())


@pytest.mark.parametrize("query,policy,method", [
    (1, "full_and_piecewise", MeasurementType.CUDA_EVENT),
    (16, "full_and_piecewise", MeasurementType.CUDA_EVENT),
    (40, "none", MeasurementType.KERNEL_ONLY),
])
def test_agreed_event_batch_actual_producer_fit_cold_warm_count_lookup(tmp_path, monkeypatch,
                                                                  query, policy, method):
    from test_step5_measured_training import _manager, _train, _write
    from test_step5_measured_replay import _predictor, _bundle_identity
    from frontier.execution_time_predictor.step5_cost import predict_step5_attention_time

    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", policy)
    scheduler, spy, _, stages, _ = _harness(ClusterType.DECODE)
    real = _decode_batch(1) if query == 1 else _prefill_batch(query)
    dummy = DummyForwardBatch(0, 0, 0)
    for lane, batch in enumerate((real, dummy)):
        _enqueue(stages[lane], batch, lane)
    _event(ClusterType.DECODE, 0).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    resumes = _event(ClusterType.DECODE, 1).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    graph = real.get_decode_cuda_graph_runtime_mode().lower()
    real_cell = Step5ProfileInput(3, tuple(real.num_tokens), tuple(real.num_context_tokens),
                                  "decode" if query == 1 else "prefill", graph,
                                  block_table_columns=512)
    dummy_cell = Step5ProfileInput.from_mrv2_dummy(3, dummy.mrv2_dummy_descriptor,
                                                 block_table_columns=512)
    rows = [producer_row(spy._model_config, cell, duration=duration, count=count)
            for cell, duration, count in ((real_cell, .25, 23), (dummy_cell, .5, 17))]
    path = _write(tmp_path, rows, method)
    manager = _manager(tmp_path / "cache", method)
    cold = _train(manager, spy._model_config, path)
    manager = _manager(tmp_path / "cache", method)
    monkeypatch.setattr(manager, "_create_estimator_and_params", lambda *_: pytest.fail("warm refit"))
    warm = _train(manager, spy._model_config, path)
    assert cold["step5_layer_e2e"]._frontier_model_hash == warm["step5_layer_e2e"]._frontier_model_hash
    identity = _bundle_identity(spy._model_config, warm)
    receipt = manager.publish_model_bundle(identity, warm)
    loaded = manager.load_model_bundle(receipt["path"], identity)
    predictor = _predictor(spy._model_config, loaded, method)
    predictor.requires_forward_start_agreement = True
    for row, batch in zip(rows, (real, dummy)):
        features, _ = step5_batch_features(predictor, batch, 3)
        assert features == step5_row_features(row)
        cost = predict_step5_attention_time(predictor, batch, 3)
        assert cost.total_time() == row["time_stats"]["step5_layer_e2e"]["mean"]
        assert cost.kernel_count == row["time_stats"]["step5_layer_e2e"]["kernel_count"]
        assert cost.operator_costs["step5_layer_e2e"].exact_lookup
    root = loaded["step5_layer_e2e"]
    original_coverage = root._frontier_step5_coverage
    for present, missing in ((real, dummy), (dummy, real)):
        features, _ = step5_batch_features(predictor, present, 3)
        from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS

        root._frontier_step5_coverage = {tuple(features[name] for name in STEP5_FEATURE_COLUMNS)}
        assert predict_step5_attention_time(predictor, present, 3).total_time() > 0
        with pytest.raises(ValueError, match="coverage is unavailable"):
            predict_step5_attention_time(predictor, missing, 3)
    root._frontier_step5_coverage = original_coverage
    # The same admitted batches also continue through the actual first-call path.
    costs = []
    original_predict = spy.predict_stage_execution_time

    def predict_with_shared_cost(batch, stage_id, cluster_type=None, **kwargs):
        costs.append(predict_step5_attention_time(predictor, batch, 3))
        return original_predict(batch, stage_id, cluster_type, **kwargs)

    spy.predict_stage_execution_time = predict_with_shared_cost
    for event in resumes:
        assert len(event.handle_event(_global(scheduler), _MetricsStore(reporting=False))) == 1
    assert [cost.total_time() for cost in costs] == [.25, .5]
    assert [cost.kernel_count for cost in costs] == [23, 17]


def test_unknown_policy_and_descriptor_mislabeling_reject_before_cost(monkeypatch):
    from dataclasses import replace
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start

    _, predictor, _, _, _ = _harness(ClusterType.DECODE)
    real, dummy = _decode_batch(1), DummyForwardBatch(0, 0, 0)
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "unknown")
    with pytest.raises(ValueError, match="Unsupported CUDA graph policy"):
        bind_forward_start([real, dummy], group_id=1, request_capacity=32)
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    bind_forward_start([real, dummy], group_id=1, request_capacity=32)
    dummy.forward_start_descriptor = replace(dummy.forward_start_descriptor, num_reqs=2)
    with pytest.raises(ValueError, match="agreed MRv2 InputBatch descriptor"):
        step5_batch_workload(predictor, dummy)
    real.forward_start_descriptor = SimpleNamespace(graph_mode="FULL", num_tokens=1)
    with pytest.raises(ValueError, match="forward-start descriptor"):
        step5_batch_workload(predictor, real)


def _first_attention(role, batches, arrivals, host_time_ms):
    """Admit two lanes at their arrival times; return each lane's first sync event time."""
    scheduler, predictor, _, stages, _ = _harness(role)
    if host_time_ms is not None:
        predictor._kv_block_zeroing_host_time = KVBlockZeroingHostTime("fixture", host_time_ms, ())
    engine_steps = {}
    for lane in range(2):
        scheduler.get_replica_scheduler(0, lane).on_engine_step_start = (
            lambda time, batch: engine_steps.__setitem__(batch.id, time)
        )
    for lane, batch in enumerate(batches):
        _enqueue(stages[lane], batch, lane)
    _event(role, 0, arrivals[0]).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    resumes = _event(role, 1, arrivals[1]).handle_event(_global(scheduler), _MetricsStore(reporting=False))
    assert [event.time for event in resumes] == [arrivals[1]] * 2
    first = {}
    for event in resumes:
        [follow] = event.handle_event(_global(scheduler), _MetricsStore(reporting=False))
        first[follow._batch.id] = follow.time
    return [first[batch.id] for batch in batches], [engine_steps.get(batch.id) for batch in batches]


def _resetting(batch):
    batch.cache_block_zeroing = KVCacheBlockZeroing((3,), 1024)
    return batch


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("arrivals,start", [
    ((1.0, 1.0), 1.009), ((1.0, 1.004), 1.009), ((1.0, 1.02), 1.02),
])
def test_kv_block_zeroing_host_time_delays_every_lane_to_the_slowest_reset(role, arrivals, start,
                                                                          monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "none")
    make = _prefill_batch if role is ClusterType.PREFILL else _decode_batch
    # Host times 9/0 ms and first attention 1/5 ms: both lanes are ready at start + 5 ms.
    batches = [_resetting(make(1)), make(5)]
    first, engine_steps = _first_attention(role, batches, arrivals, 9.0)
    assert first == pytest.approx([start + .001, start + .005])
    for batch in batches:
        assert batch._forward_launch_start_time == pytest.approx(start)
        assert batch._forward_start_host_delay == pytest.approx(start - arrivals[1])
        stage_start = (batch._prefill_stage_start_time if role is ClusterType.PREFILL
                       else batch._decode_stage_start_time)
        assert stage_start == arrivals[1]
    # The DECODE engine step posts its NIXL READs after the launch.
    expected_steps = [pytest.approx(start)] * 2 if role is ClusterType.DECODE else [None] * 2
    assert engine_steps == expected_steps


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("mode", ["none", "full_and_piecewise"])
def test_kv_block_zeroing_host_time_shifts_a_dummy_peer_and_needs_a_resetting_batch(role, mode,
                                                                                   monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    make = _prefill_batch if role is ClusterType.PREFILL else _decode_batch

    def run(host_time_ms, resetting):
        real = make(3)
        if resetting:
            _resetting(real)
        dummy = DummyForwardBatch(0, 0, 0)
        first, _ = _first_attention(role, [real, dummy], (1.0, 1.0), host_time_ms)
        return first, dummy

    baseline, _ = run(None, True)
    assert run(9.0, False)[0] == baseline
    shifted, dummy = run(9.0, True)
    assert shifted == pytest.approx([time + .009 for time in baseline])
    assert dummy.stage_start_time == 1.0
    assert dummy._forward_launch_start_time == pytest.approx(1.009)
