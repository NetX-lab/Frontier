"""Independent mixed-layer reporting oracles through the real metrics store."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.attention import LayerAttentionSpec
from frontier.attention.families import get_attention_family
from frontier.config import global_vars
from frontier.config.config import ClusterConfig, MetricsConfig, ReplicaConfig, SimulationConfig
from frontier.entities import BatchStage, Request, StageExecutionTime
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.scheduler.utils.dense_metrics import build_prefill_metrics_execution_time
from frontier.scheduler.utils.expert_parallel import get_ep_phase_times_ms, predict_ep_wave_phase_times
from frontier.types import ClusterType
from tests.unit.test_stage_finalized_contract import _layer, _mixed_layers


@pytest.fixture(autouse=True)
def _isolated_simulation_globals():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


class _TraceSink:
    def __init__(self):
        self.events = []

    def log_event(self, event):
        self.events.append(event)


def _store(tmp_path, **flags):
    values = dict(
        output_dir=str(tmp_path),
        store_plots=False,
        store_utilization_metrics=False,
        store_operation_metrics=False,
        store_frontier_stage_batch_ledger=False,
        enable_op_level_tracing=True,
    )
    values.update(flags)
    replica = ReplicaConfig(
        device="a100", network_device="a100_pairwise_nvlink",
        model_name="meta-llama/Llama-2-7b-hf",
    )
    # The synthetic mixed-stage oracle needs valid MLA dimensions for metadata.
    replica.model_config = replace(
        replica.model_config, use_mla=True, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=8, qk_head_dim=16, v_head_dim=8,
    )
    cluster = ClusterConfig(replica_config=replica)
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(**values))
    sink = _TraceSink()
    return MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, sink), sink


def _mixed_stage():
    specifications = (
        (LayerAttentionSpec(4, "dense_attention", "standard"), "attn_prefill", 2.0),
        (LayerAttentionSpec(7, "dense_attention", "standard"), "attn_prefill", 3.0),
        (LayerAttentionSpec(9, "latent_mla_attention", "mla"), "attn_mla_prefill", 5.0),
        (LayerAttentionSpec(12, "gated_delta_net", "qwen3_5"), "gdn_core_prefill", 7.0),
        (LayerAttentionSpec(14, "gated_delta_net", "qwen3_5"), "gdn_core_prefill", 11.0),
    )
    layers = tuple(_layer(
        spec.global_layer_id,
        attention_family_id=spec.family_id,
        attention_variant_id=spec.variant_id,
        op_times={
            **{operator.name: 0.0 for operator in get_attention_family(spec.family_id).e2e_trace_ops()},
            name: duration,
        },
    ) for spec, name, duration in specifications)
    return StageExecutionTime(layers, stage_execution_time=layers[0])


def _schedule(store, stage):
    batch = BatchStage(
        batch_id=73, replica_id=0, pipeline_stage=0,
        execution_time=stage.total_time, model_execution_time=stage.model_time,
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], cluster_type=ClusterType.MONOLITHIC,
    )
    batch.on_schedule(1.0)
    store.on_replica_stage_schedule(
        time=1.0, replica_id=0, stage_id=0, batch_stage=batch,
        execution_time=stage, cluster_type=ClusterType.MONOLITHIC,
    )
    return batch


def test_expanded_traces_preserve_actual_global_ids_and_family_metadata(tmp_path):
    store, sink = _store(tmp_path, enable_per_layer_expansion=True)
    stage = _mixed_stage()
    _schedule(store, stage)

    assert [(event.layer_id, event.name, event.duration_ms) for event in sink.events] == [
        (4, "attn_prefill", 2.0), (7, "attn_prefill", 3.0),
        (9, "attn_mla_prefill", 5.0), (12, "gdn_core_prefill", 7.0),
        (14, "gdn_core_prefill", 11.0),
    ]
    for event, layer in zip(sink.events, stage.layer_execution_times):
        assert event.meta["global_layer_id"] == layer.global_layer_id
        assert event.meta["attention_family_id"] == layer.attention_family_id
        assert event.meta["attention_variant_id"] == layer.attention_variant_id
        assert event.replica_id == 0
        assert event.batch_id == 73


def test_aggregate_traces_group_family_variant_and_operation(tmp_path):
    store, sink = _store(tmp_path, enable_per_layer_expansion=False)
    _schedule(store, _mixed_stage())

    assert [(event.name, event.duration_ms) for event in sink.events] == [
        ("attn_prefill", 5.0), ("attn_mla_prefill", 5.0), ("gdn_core_prefill", 18.0),
    ]
    assert [event.layer_id for event in sink.events] == [-1, -1, -1]
    assert [event.meta["global_layer_ids"] for event in sink.events] == [[4, 7], [9], [12, 14]]
    assert [event.meta["attention_family_id"] for event in sink.events] == [
        "dense_attention", "latent_mla_attention", "gated_delta_net",
    ]
    assert [event.meta["attention_variant_id"] for event in sink.events] == [
        "standard", "mla", "qwen3_5",
    ]
    assert sum(event.duration_ms for event in sink.events) == pytest.approx(28.0)


def test_requested_operator_metrics_do_not_require_utilization(tmp_path):
    store, sink = _store(tmp_path, store_operation_metrics=True, enable_op_level_tracing=False)
    _schedule(store, _mixed_stage())

    series = store._operation_metrics_per_batch[ClusterType.MONOLITHIC]
    assert series[OperationMetrics.ATTN_PREFILL]._data_series == [(73, 2.0), (73, 3.0)]
    assert series[OperationMetrics("attn_mla_prefill")]._data_series == [(73, 5.0)]
    assert series[OperationMetrics.GDN_CORE_PREFILL]._data_series == [(73, 7.0), (73, 11.0)]
    assert not sink.events


def test_aggregate_trace_keeps_distinct_variants_separate(tmp_path):
    store, sink = _store(tmp_path)
    standard = _mixed_stage().layer_execution_times[0]
    mfa = standard.as_single_layer(
        global_layer_id=6, attention_family_id="dense_attention", attention_variant_id="mfa",
    )
    _schedule(store, StageExecutionTime((standard, mfa), stage_execution_time=standard))

    assert len(sink.events) == 2
    assert [event.duration_ms for event in sink.events] == [2.0, 2.0]
    assert [event.meta["global_layer_ids"] for event in sink.events] == [[4], [6]]
    assert [event.meta["attention_variant_id"] for event in sink.events] == ["standard", "mfa"]


def test_dense_metrics_adapter_predicts_each_actual_dense_layer_and_preserves_owner():
    first = _layer(7, op_times={"attn_prefill": 2.0}, schedule_time=13.0)
    routed = _layer(8, is_moe=True, op_times={"attn_prefill": 3.0})
    last = _layer(9, op_times={"attn_prefill": 5.0})
    original = StageExecutionTime((first, routed, last), stage_execution_time=first)
    dense_outputs = {
        7: _layer(7, op_times={"attn_prefill": 2.0, "mlp_up_proj": 11.0}, schedule_time=999.0),
        9: _layer(9, op_times={"attn_prefill": 5.0, "mlp_up_proj": 17.0}, schedule_time=999.0),
    }
    predictor = Mock()
    predictor.predict_stage_execution_time.side_effect = lambda *args, **kwargs: StageExecutionTime(
        (dense_outputs[kwargs["layer_id"]],),
    )
    model = SimpleNamespace(is_moe=True, num_layers=10, get_moe_layer_ids=lambda: [8], is_moe_layer=lambda layer_id: layer_id == 8)
    batch = object()

    corrected = build_prefill_metrics_execution_time(
        original_execution_time=original, sample_batch=batch, predictor=predictor,
        stage_id=2, cluster_type=ClusterType.PREFILL, model_config=model,
    )

    assert corrected.global_layer_ids == (7, 8, 9)
    assert corrected.model_time_ms == 38.0
    assert corrected.schedule_time == 13.0
    assert corrected.total_time == pytest.approx(0.051)
    assert original.model_time_ms == 10.0
    assert [call.kwargs["layer_id"] for call in predictor.predict_stage_execution_time.call_args_list] == [7, 9]
    for call in predictor.predict_stage_execution_time.call_args_list:
        assert call.args == (batch, 2)
        assert call.kwargs["num_layers"] == 1
        assert call.kwargs["cluster_type"] == ClusterType.PREFILL
    assert corrected.layer_execution_times[1].op_times == routed.op_times


@pytest.mark.parametrize("capture_lane_timings", [False, True])
def test_ep_wave_retains_actual_predictions_only_when_requested(capture_lane_timings):
    stages = {
        ep_id: StageExecutionTime((_layer(7, is_moe=True, op_times={
            "moe_gating_linear": 2.0,
            "moe_grouped_gemm": 5.0 + ep_id,
            "expert_parallel_alltoall_dispatch": 3.0,
            "expert_parallel_alltoall_combine": 4.0,
        }),))
        for ep_id in (0, 1)
    }
    batches = {ep_id: SimpleNamespace(ep_id=ep_id, lane_workload=object()) for ep_id in (0, 1)}
    predictor = Mock()
    predictor.predict_stage_execution_time.side_effect = lambda batch, *args, **kwargs: stages[batch.ep_id]

    phases = predict_ep_wave_phase_times(
        layer_workload=SimpleNamespace(participant_ep_ids=(0, 1)), source_batch=object(),
        stage_id=2, layer_id=7, cluster_type=ClusterType.PREFILL,
        predictor=predictor, lane_builder=lambda **kwargs: batches[kwargs["ep_id"]],
        phase_getter=get_ep_phase_times_ms, workload_logger=Mock(),
        trace_identity={}, batch_id=73, capture_lane_timings=capture_lane_timings,
    )

    assert phases.pre_dispatch_times_ms == (2.0, 2.0)
    assert phases.routed_compute_times_ms == (5.0, 6.0)
    assert phases.dispatch_times_ms == (3.0, 3.0)
    assert phases.combine_times_ms == (4.0, 4.0)
    if capture_lane_timings:
        assert [record.ep_id for record in phases.lane_records] == [0, 1]
        for record in phases.lane_records:
            assert record.execution_time is stages[record.ep_id]
            assert record.batch is batches[record.ep_id]
    else:
        assert phases.lane_records == ()


def test_ledger_projects_every_family_and_charges_owner_once(tmp_path):
    store, _ = _store(tmp_path)
    mixed_ledger = store._build_frontier_stage_batch_component_ledger(_mixed_stage())
    assert mixed_ledger["attention_prefill_execution_time"] == 5.0
    assert mixed_ledger["attn_mla_prefill_time"] == 5.0
    assert mixed_ledger["gdn_core_prefill"] == 18.0
    assert sum(mixed_ledger.values()) == 28.0

    first, second = _mixed_layers()
    second = second.as_single_layer(
        global_layer_id=9, attention_family_id="latent_mla_attention", attention_variant_id="mla",
    )
    for layer in (first, second):
        layer._replace_operator_time_values({
            **{operator.name: 0.0 for operator in get_attention_family(layer.attention_family_id).e2e_trace_ops()},
            **layer.op_times,
        })
    stage = StageExecutionTime((first, second), stage_execution_time=first)
    ledger = store._build_frontier_stage_batch_component_ledger(stage)
    assert ledger["pipeline_parallel_communication_time"] == 11.0
    assert ledger["schedule_time"] == 13.0
    assert ledger["decode_draft_proposer_time"] == 17.0
    assert ledger["mtp_terminal_overshoot_time"] == 19.0
    assert ledger["moe_grouped_gemm_time"] == 31.0
    assert ledger["moe_gating_routing_topk_time"] == 37.0
    assert sum(ledger.values()) == pytest.approx(298.0)
    assert store._build_frontier_stage_batch_diagnostic_component_ledger(stage) == {
        "pp_stage_boundary_handoff_time": 23.0,
    }
    assert stage.model_time_ms == 285.0
    assert stage.diagnostic_total_time_ms == pytest.approx(321.0)
    assert store._build_frontier_stage_batch_component_ledger(stage) == ledger


def test_requested_ledger_completes_without_utilization_metrics(tmp_path):
    store, _ = _store(
        tmp_path, enable_op_level_tracing=False, store_frontier_stage_batch_ledger=True,
    )
    stage = _mixed_stage()
    batch = _schedule(store, stage)
    assert len(store._pending_frontier_stage_batch_ledger_rows) == 1

    store.on_batch_stage_end(
        batch_stage=batch, time=1.25, replica_id=0, stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert not store._pending_frontier_stage_batch_ledger_rows
    assert not store._pending_frontier_stage_batch_ledger_rows_by_key
    assert not store._pending_frontier_stage_batch_ledger_row_keys
    assert len(store._frontier_stage_batch_ledger_rows) == 1
    row = store._frontier_stage_batch_ledger_rows[0]
    assert row["stage_start_ts"] == 1.0
    assert row["stage_end_ts"] == 1.25
    assert row["execution_time"]["total_time_ms"] == 28.0


@pytest.mark.parametrize("write_metrics", [False, True])
def test_disabled_reporting_does_not_build_projection_records(tmp_path, monkeypatch, write_metrics):
    store, sink = _store(tmp_path, write_metrics=write_metrics, enable_op_level_tracing=False)
    for name in (
        "_build_op_trace_context", "_build_frontier_stage_batch_component_ledger",
        "_push_stage_layer_operation_metrics",
    ):
        monkeypatch.setattr(store, name, Mock(side_effect=AssertionError("Disabled projection was built")))
    _schedule(store, _mixed_stage())

    assert not sink.events
    assert not store._pending_frontier_stage_batch_ledger_rows
    assert not store._frontier_stage_batch_ledger_rows
    assert all(not len(series) for series in store._operation_metrics_per_batch[ClusterType.MONOLITHIC].values())


def test_ep_lane_reporting_preserves_work_and_barrier_gaps(tmp_path):
    from frontier.entities.batch import EPBatchGroup
    from frontier.moe_ep_workload import EPLaneWorkload
    from frontier.scheduler.utils.expert_parallel import EPWaveLaneTiming, EPWavePhaseTimes, EPWaveTiming

    store, sink = _store(tmp_path, store_operation_metrics=True,
                         store_frontier_stage_batch_ledger=True)
    replica = store._cluster_configs[ClusterType.MONOLITHIC].replica_config
    replica.model_config = replace(replica.model_config, is_moe=True, num_experts=2,
                                   num_experts_per_tok=1)
    request = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)
    records = []
    for ep_id, (pre, routed) in enumerate(((1.0, 5.0), (3.0, 2.0))):
        batch = EPBatchGroup(
            requests=[request], num_tokens=[8], replica_id=0, ep_id=ep_id, time=1.0,
            source_batch_ids=[73], cluster_type=ClusterType.MONOLITHIC, is_moe=True,
            lane_workload=EPLaneWorkload(ep_id, 2, 2, (ep_id,), (4,), 4, 1),
        )
        batch.moe_pre_routing_effective_total_tokens = 8
        layer = _layer(7, is_moe=True, schedule_time=999.0, op_times={
            "moe_gating_linear": pre, "moe_grouped_gemm": routed,
            "expert_parallel_alltoall_dispatch": 2.0,
            "expert_parallel_alltoall_combine": 4.0, "add_ffn_residual": 1.0,
        })
        records.append(EPWaveLaneTiming(ep_id, batch, StageExecutionTime((layer,))))
    plan = SimpleNamespace(
        phase_times=EPWavePhaseTimes((7.0, 6.0), (1.0, 3.0), (2.0, 2.0),
                                    (5.0, 2.0), (4.0, 4.0), (1.0, 1.0), tuple(records)),
        timing=EPWaveTiming(5.0, 1.005, 9.0, 1.014, 1.0, 1.015),
    )
    store.on_ep_wave_schedule(plan, time=1.0, replica_id=0, stage_id=2,
                             cluster_type=ClusterType.MONOLITHIC)

    assert len(sink.events) == 10
    for ep_id, expected_work in enumerate((13.0, 12.0)):
        events = [event for event in sink.events if event.meta["ep_id"] == ep_id]
        assert [event.ts_start for event in events] == pytest.approx([1, 1.003, 1.005, 1.010, 1.014])
        assert sum(event.duration_ms for event in events) == expected_work
        assert all(event.meta["global_layer_ids"] == [7] for event in events)
        assert all(event.meta["stage_id"] == 2 for event in events)
        assert all(event.type != "OVERHEAD" for event in events)
    series = store._operation_metrics_per_batch[ClusterType.MONOLITHIC]
    assert [value for _, value in series[OperationMetrics.MOE_GROUPED_GEMM]._data_series] == [5.0, 2.0]
    assert not store._frontier_stage_batch_ledger_rows
    assert len(store._frontier_ep_wave_lane_ledger_rows) == 2
    store._write_frontier_stage_batch_ledger()
    import json
    from pathlib import Path
    rows = [json.loads(line) for line in (Path(store._config.output_dir) / "frontier_ep_wave_lane_ledger.jsonl").read_text().splitlines()]
    assert [row["ep_id"] for row in rows] == [0, 1]
    assert rows[1]["phases"]["routed_compute"]["duration_ms"] == 2.0
    assert rows[0]["phases"]["combine"]["start_time_s"] == pytest.approx(1.010)


@pytest.mark.parametrize("wave_first", [False, True])
def test_ep_and_attention_share_source_batch_expansion_decision(tmp_path, wave_first):
    from frontier.entities.batch import EPBatchGroup
    from frontier.moe_ep_workload import EPLaneWorkload
    from frontier.scheduler.utils.expert_parallel import EPWaveLaneTiming, EPWavePhaseTimes, EPWaveTiming

    store, sink = _store(
        tmp_path, enable_per_layer_expansion=True, num_requests_to_trace_per_layer=1,
    )
    replica = store._cluster_configs[ClusterType.MONOLITHIC].replica_config
    replica.model_config = replace(replica.model_config, is_moe=True,
                                   num_experts=1, num_experts_per_tok=1)
    request = Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)
    lane = EPBatchGroup(
        requests=[request], num_tokens=[8], replica_id=0, ep_id=0, time=1.0,
        source_batch_ids=[73], cluster_type=ClusterType.MONOLITHIC, is_moe=True,
        lane_workload=EPLaneWorkload(0, 1, 1, (0,), (8,), 8, 1),
    )
    lane.moe_pre_routing_effective_total_tokens = 8
    layer = _layer(7, is_moe=True, op_times={
        "moe_gating_linear": 1.0, "expert_parallel_alltoall_dispatch": 0.0,
        "expert_parallel_alltoall_combine": 0.0,
    })
    plan = SimpleNamespace(
        phase_times=EPWavePhaseTimes((1.0,), (1.0,), (0.0,), (0.0,), (0.0,), (0.0,),
                                    (EPWaveLaneTiming(0, lane, StageExecutionTime((layer,))),)),
        timing=EPWaveTiming(1.0, 1.001, 0.0, 1.001, 0.0, 1.001),
    )

    def attention(batch_id, current_request):
        stage = _mixed_stage()
        batch = BatchStage(
            batch_id=batch_id, replica_id=0, pipeline_stage=0,
            execution_time=stage.total_time, model_execution_time=stage.model_time,
            requests=[current_request], num_tokens=[8], cluster_type=ClusterType.MONOLITHIC,
        )
        batch.on_schedule(1.0)
        store.on_replica_stage_schedule(
            time=1.0, replica_id=0, stage_id=0, batch_stage=batch,
            execution_time=stage, cluster_type=ClusterType.MONOLITHIC,
        )

    def wave():
        store.on_ep_wave_schedule(plan, time=1.0, replica_id=0, stage_id=0,
                                 cluster_type=ClusterType.MONOLITHIC)

    if wave_first:
        wave()
        attention(73, request)
    else:
        attention(73, request)
        wave()
    assert sorted(event.layer_id for event in sink.events) == [4, 7, 7, 9, 12, 14]

    sink.events.clear()
    attention(74, Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1))
    assert len(sink.events) == 3
    assert all(event.layer_id == -1 for event in sink.events)
