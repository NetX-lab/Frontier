"""Real metrics/trace regression seams; CPU fixtures do not admit native timing."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from frontier.attention.families import DSA_ATTENTION_FAMILY, STEP5_ATTENTION_FAMILY
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.config import ClusterConfig, MetricsConfig, ReplicaConfig, SimulationConfig, global_vars
from frontier.entities import BatchStage, ExecutionTime, Request, StageExecutionTime
from frontier.entities.time_components import AttentionOperatorTimes, OperatorCost
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.metrics.trace_store import TraceStore
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_execution_time_op_times import _base_execution_kwargs
from tests.unit.test_metrics_stage_execution_time import _TraceStore


ROOT = "step5_layer_e2e"
PROOF = OperatorCost(.6244, 23., MeasurementType.KERNEL_ONLY,
                     "cpu_fixture_duration", "cpu_fixture_count", "cpu_fixture", True)


@pytest.fixture(autouse=True)
def reset_globals():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _root_layer(layer_id=0, *, proof=PROOF, times=None, costs=None, **kwargs):
    values = {name: False if isinstance(value, bool) else 0.
              for name, value in _base_execution_kwargs().items()}
    values.update(num_layers_per_pipeline_stage=1, global_layer_id=layer_id,
                  attention_family_id="step5_attention", attention_variant_id="swa",
                  attention_operator_times=AttentionOperatorTimes(
                      {ROOT: proof.duration_ms} if times is None else times,
                      costs={ROOT: proof} if costs is None else costs,
                      diagnostics={"step5_qkv_projection": .4, "step5_swa_attention": .5}),
                  **kwargs)
    return ExecutionTime(**values)


@pytest.mark.parametrize("wrap_stage", [False, True])
def test_measured_root_component_ledger_charges_only_the_physical_root(wrap_stage):
    layer = _root_layer()
    execution = StageExecutionTime([layer]) if wrap_stage else layer
    ledger = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(execution)
    assert ledger[ROOT] == .6244
    assert math.fsum(ledger.values()) == pytest.approx(execution.total_time * 1e3)
    assert not {"step5_qkv_projection", "step5_swa_attention"}.intersection(ledger)


def test_measured_root_mixed_stage_keeps_stage_owner_and_gap_once():
    first = _root_layer(attn_norm_time=.01, op_times={"emb": .11, "input_cast": .12})
    first.add_kernel_gap(.001, first.attention_time_component.kernel_count, {})
    second = _root_layer(1, proof=replace(PROOF, duration_ms=.2, kernel_count=7),
                         op_times={"emb": 99., "input_cast": 99., "final_residual": 99.})
    dense_values = {name: False if isinstance(value, bool) else 0.
                    for name, value in _base_execution_kwargs().items()}
    dense_values.update(num_layers_per_pipeline_stage=1, global_layer_id=2,
                        attention_family_id="dense_attention", attention_variant_id="standard",
                        attention_prefill_execution_time=.3)
    owner = _root_layer(proof=replace(PROOF, duration_ms=99.),
                        op_times={"emb": .11, "input_cast": .12, "final_residual": .13},
                        pipeline_parallel_communication_time=.14, schedule_time=.15)
    stage = StageExecutionTime([first, second, ExecutionTime(**dense_values)], stage_execution_time=owner)
    ledger = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(stage)
    assert ledger[ROOT] == .8244
    assert ledger["attention_prefill_execution_time"] == .3
    assert ledger["kernel_gap_time"] == .023
    assert ledger["emb"] == .11 and ledger["input_cast"] == .12 and ledger["final_residual"] == .13
    assert ledger["pipeline_parallel_communication_time"] == .14 and ledger["schedule_time"] == .15
    assert math.fsum(ledger.values()) == pytest.approx(stage.total_time * 1e3)
    assert stage.layer_execution_times[0].attention_time_component.kernel_count == 23
    assert stage.layer_execution_times[1].attention_operator_times.costs[ROOT].kernel_count == 7


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("measurement", [MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT])
def test_schedule_publishes_root_operation_and_actual_trace_proof(tmp_path, expand, measurement):
    replica = ReplicaConfig(model_name="stepfun-ai/Step-5-Preview", device="h800",
                            attn_tensor_parallel_size=8, attn_dp=1,
                            moe_tensor_parallel_size=8, moe_expert_parallel_size=1)
    cluster = ClusterConfig(replica_config=replica)
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(
        output_dir=str(tmp_path), write_metrics=True, store_plots=False,
        store_operation_metrics=True, enable_op_level_tracing=True,
        enable_per_layer_expansion=expand, num_requests_to_trace_per_layer=1,
        store_frontier_stage_batch_ledger=True))
    sink = _TraceStore()
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, sink)
    proof = replace(PROOF, measurement_type=measurement)
    second_proof = replace(proof, duration_ms=.2, kernel_count=7,
                           duration_model_hash="second_duration", kernel_count_model_hash="second_count",
                           exact_lookup=False)
    layers = [layer.as_single_layer(
        global_layer_id=layer.global_layer_id, attention_family_id=layer.attention_family_id,
        attention_variant_id=replica.model_config.get_layer_attention_specs()[layer.global_layer_id].variant_id,
    ) for layer in (_root_layer(proof=proof), _root_layer(1, proof=second_proof))]
    stage = StageExecutionTime(layers)
    batch = BatchStage(batch_id=7, replica_id=0, pipeline_stage=0,
                       execution_time=stage.total_time, model_execution_time=stage.model_time,
                       requests=[Request(arrived_at=0., num_prefill_tokens=8, num_decode_tokens=1)],
                       num_tokens=[8], request_is_decoding=[False], cluster_type=ClusterType.MONOLITHIC)
    batch.on_schedule(1.)
    store.on_replica_stage_schedule(time=1., replica_id=0, stage_id=0, batch_stage=batch,
                                    execution_time=stage, cluster_type=ClusterType.MONOLITHIC)
    metric = OperationMetrics(ROOT)
    series = store._operation_metrics_per_batch[ClusterType.MONOLITHIC][metric]
    assert [value for _, value in series._data_series] == [.6244, .2]
    assert [event.name for event in sink.events] == [ROOT, ROOT]
    cursor = 1.
    for event, layer, cost in zip(sink.events, stage.layer_execution_times, (proof, second_proof)):
        assert event.duration_ms == cost.duration_ms and event.ts_start == pytest.approx(cursor)
        cursor += cost.duration_ms * 1e-3
        assert event.meta["operator_cost"] == {
            "duration_ms": cost.duration_ms, "kernel_count": cost.kernel_count,
            "measurement_type": cost.measurement_type.value,
            "duration_model_hash": cost.duration_model_hash,
            "kernel_count_model_hash": cost.kernel_count_model_hash,
            "evidence_kind": cost.evidence_kind, "exact_lookup": cost.exact_lookup,
            "includes_kernel_gap": cost.includes_kernel_gap,
        }
        assert event.meta["evidence_kind"] == "source_structural_metadata"
        assert event.meta["tensor_shape"] == {"input": [8, replica.model_config.embedding_dim],
                                               "output": [8, replica.model_config.embedding_dim]}
        assert event.meta["source_geometry"]["global_layer_id"] == layer.global_layer_id
        assert event.meta["attention_operator_names"] == [ROOT]
    disk_sink = TraceStore(str(tmp_path), enabled=True)
    disk_sink.initialize({"evidence_kind": "cpu_fixture"})
    for event in sink.events:
        disk_sink.log_event(event)
    disk_sink.close()
    persisted = [json.loads(line) for line in (tmp_path / "op_traces.jsonl").read_text().splitlines()][1:]
    assert [event["meta"]["operator_cost"] for event in persisted] == [event.meta["operator_cost"] for event in sink.events]
    row = store._pending_frontier_stage_batch_ledger_rows[id(batch)]["execution_time"]
    assert row["component_ledger_ms"][ROOT] == .8244
    assert math.fsum(row["component_ledger_ms"].values()) == pytest.approx(row["total_time_ms"])


@pytest.mark.parametrize("fault", ["missing_cost", "duration", "count", "zero_count", "mixed", "other_family", "model_projection", "legacy_child"])
def test_root_projection_rejects_malformed_or_overlapping_physical_scopes(fault):
    kwargs = {}
    if fault == "missing_cost": kwargs["costs"] = {}
    if fault == "duration": kwargs["costs"] = {ROOT: replace(PROOF, duration_ms=.1)}
    if fault == "count": kwargs["costs"] = {ROOT: replace(PROOF, kernel_count=float("nan"))}
    if fault == "zero_count": kwargs["costs"] = {ROOT: replace(PROOF, kernel_count=0.)}
    if fault == "mixed": kwargs["times"] = {ROOT: .6244, "step5_swa_attention": .5}
    if fault == "other_family": kwargs["times"] = {ROOT: .6244, "gdn_core_prefill": .5}
    if fault == "model_projection": kwargs["attention_layer_pre_proj_execution_time"] = .1
    if fault == "legacy_child": kwargs["attention_prefill_execution_time"] = .1
    with pytest.raises(ValueError):
        get_attention_trace_op_times(_root_layer(**kwargs), STEP5_ATTENTION_FAMILY)


def test_root_projection_keeps_per_layer_duration_without_changing_proof():
    layer = _root_layer()
    assert [(op.name, duration) for op, duration in get_attention_trace_op_times(
        layer, STEP5_ATTENTION_FAMILY, per_layer_count=2)] == [(ROOT, .3122)]
    assert layer.attention_operator_times.costs[ROOT] == PROOF


def test_root_projection_uses_registered_role_not_operator_name():
    root = next(operator for operator in STEP5_ATTENTION_FAMILY.operators if operator.is_inclusive_root)
    family = replace(STEP5_ATTENTION_FAMILY, family_id="renamed_attention",
                     operators=(replace(root, name="renamed_inclusive_root"),))
    source = SimpleNamespace(attention_operator_times=AttentionOperatorTimes(
        {"renamed_inclusive_root": .6244}, costs={"renamed_inclusive_root": PROOF}),
        attention_time_component=_root_layer().attention_time_component)
    assert [(operator.name, duration) for operator, duration in get_attention_trace_op_times(
        source, family)] == [("renamed_inclusive_root", .6244)]


def test_unregistered_root_scope_is_rejected_before_metrics():
    with pytest.raises(ValueError, match="Unsupported ExecutionTime operator timing"):
        _root_layer(times={"unregistered_root": .6244})


@pytest.mark.parametrize("fault", [None, "missing", "extra", "legacy_child"])
def test_legacy_inclusive_scalar_scopes_keep_projection_and_rejection(monkeypatch, fault):
    import frontier.entities.time_components as components
    from frontier.attention.ops import ProjectionOwnership
    import frontier.attention.families as families

    family = replace(DSA_ATTENTION_FAMILY, dsa_frozen=False, execution_enabled=True,
                     operators=tuple(replace(operator, projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE)
                                     for operator in DSA_ATTENTION_FAMILY.operators),
                     disjoint_model_projection_attrs=("attention_layer_pre_proj_execution_time",
                                                      "attention_layer_post_proj_execution_time"))
    original = components.iter_execution_enabled_families(include_structural=True)
    monkeypatch.setattr(components, "iter_execution_enabled_families", lambda **_: tuple(
        family if item.family_id == family.family_id else item for item in original))
    monkeypatch.setitem(families._FAMILIES_BY_ID, family.family_id, family)
    caches = (components._canonical_operator_execution_time_attrs,
              components._attention_operator_execution_time_attrs)
    for cache in caches:
        cache.cache_clear()
    try:
        values = {name: False if isinstance(value, bool) else 0.
                  for name, value in _base_execution_kwargs().items()}
        times = {"attn_dsa_indexer": .1, "attn_dsa_mla": .2}
        if fault == "missing": times.pop("attn_dsa_indexer")
        if fault == "extra": times["attn_prefill"] = .3
        values.update(num_layers_per_pipeline_stage=1, global_layer_id=0,
                      attention_family_id=family.family_id, attention_variant_id=family.supported_variants[0],
                      op_times=times)
        if fault == "legacy_child": values["attention_prefill_execution_time"] = .3
        layer = ExecutionTime(**values)
        if fault is not None:
            with pytest.raises(ValueError):
                get_attention_trace_op_times(layer, family)
            return
        assert [(op.name, duration) for op, duration in get_attention_trace_op_times(layer, family)] == list(times.items())
        ledger = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(StageExecutionTime([layer]))
        assert ledger["attn_dsa_indexer"] == .1 and ledger["attn_dsa_mla"] == .2
        assert math.fsum(ledger.values()) == pytest.approx(.3)
    finally:
        for cache in caches:
            cache.cache_clear()


def test_root_operation_registry_does_not_add_empty_legacy_dense_tables(tmp_path):
    cluster = ClusterConfig(replica_config=ReplicaConfig(model_name="meta-llama/Llama-2-7b-hf"))
    store = MetricsStore(SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(
        output_dir=str(tmp_path), store_plots=False)), {ClusterType.MONOLITHIC: cluster})
    assert OperationMetrics(ROOT) not in store._operation_metrics[ClusterType.MONOLITHIC]
