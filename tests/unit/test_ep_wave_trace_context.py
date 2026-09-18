"""Phase-local EP trace context construction and event value contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.config import MetricsConfig, get_quantization_manager
from frontier.entities import Request, StageExecutionTime
from frontier.entities.batch import EPBatchGroup
from frontier.metrics import ep_wave_metrics, op_trace_utils
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.utils.expert_parallel import (
    EPWaveLaneTiming, EPWavePhaseTimes, EPWaveTiming,
)
from frontier.types import ClusterType
from tests.unit.predictor_cache_fixtures import predictor_fixture_config
from tests.unit.test_stage_finalized_contract import _layer


def _reporting_case(*, tracing=True, operations=True, ledger=True, zero_work=False):
    get_quantization_manager().load_config(None)
    replica = predictor_fixture_config(total_experts=2)["replica_config"]
    batch = EPBatchGroup(
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], replica_id=0, ep_id=0, time=1.0,
        source_batch_ids=[73], cluster_type=ClusterType.MONOLITHIC, is_moe=True,
        lane_workload=EPLaneWorkload(0, 1, 2, (0, 1), (4, 12), 16, 2),
    )
    batch.moe_pre_routing_effective_total_tokens = 8
    times = {
        "moe_gating_linear": 1.0, "moe_gating_routing_topk": 2.0,
        "moe_grouped_gemm": 3.0, "moe_tensor_parallel_allreduce": 4.0,
        "add_ffn_residual": 5.0,
        "expert_parallel_alltoall_dispatch": 0.0,
        "expert_parallel_alltoall_combine": 0.0,
    }
    if zero_work:
        times = dict.fromkeys(times, 0.0)
    layer = _layer(7, is_moe=True, op_times=times)
    phases = tuple(layer._moe_phase_time(phase) for phase in (
        "pre_dispatch", "dispatch", "routed_compute", "combine", "post_combine",
    ))
    plan = SimpleNamespace(
        phase_times=EPWavePhaseTimes(
            (phases[0] + phases[2] + phases[4],), *((value,) for value in phases),
            (EPWaveLaneTiming(0, batch, StageExecutionTime((layer,))),),
        ),
        timing=(EPWaveTiming(0.0, 1.0, 0.0, 1.0, 0.0, 1.0) if zero_work
                else EPWaveTiming(3.0, 1.003, 7.0, 1.010, 5.0, 1.015)),
    )
    store = SimpleNamespace(
        _config=MetricsConfig(
            enable_op_level_tracing=tracing, store_operation_metrics=operations,
            store_frontier_stage_batch_ledger=ledger,
        ),
        _cluster_configs={ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica)},
        trace_store=SimpleNamespace(log_event=Mock()),
        _should_expand_layers=Mock(return_value=True), _push_metric=Mock(),
        _frontier_ep_wave_lane_ledger_rows=[],
    )
    return store, plan


@pytest.mark.parametrize("tracing,operations,ledger", [
    (False, False, False), (False, True, False),
    (False, False, True), (True, True, True),
])
@pytest.mark.parametrize("zero_work", [False, True])
def test_ep_context_is_built_only_once_per_traced_positive_phase(
    monkeypatch, tracing, operations, ledger, zero_work,
):
    store, plan = _reporting_case(
        tracing=tracing, operations=operations, ledger=ledger, zero_work=zero_work,
    )
    context_spy = Mock(wraps=ep_wave_metrics.OpTraceContext)
    monkeypatch.setattr(ep_wave_metrics, "OpTraceContext", context_spy)

    ep_wave_metrics.record_ep_wave(
        store, plan, time=1.0, replica_id=0, stage_id=2,
        cluster_type=ClusterType.MONOLITHIC,
    )

    positive_phases = 3 if tracing and not zero_work else 0
    assert context_spy.call_count == positive_phases
    assert store._push_metric.call_count == (5 if operations and not zero_work else 0)
    assert len(store._frontier_ep_wave_lane_ledger_rows) == int(ledger)
    events = [call.args[0] for call in store.trace_store.log_event.call_args_list]
    assert len(events) == (5 if positive_phases else 0)
    if not events:
        return

    assert [event.ts_start for event in events] == pytest.approx([
        1.0, 1.001, 1.003, 1.006, 1.010,
    ])
    assert [event.duration_ms for event in events] == [1, 2, 3, 4, 5]
    replica = store._cluster_configs[ClusterType.MONOLITHIC].replica_config
    for event in events:
        routed = event.meta["phase"] == "routed_compute"
        tokens = 16 if routed else 8
        context = op_trace_utils.OpTraceContext(
            ClusterType.MONOLITHIC, replica.model_config, replica,
            8, tokens, tokens, tokens, routed,
        )
        expected = op_trace_utils.compute_op_trace_meta(event.name, event.type, context)
        assert {key: event.meta[key] for key in expected} == expected
        assert event.meta["parallel_context"] == op_trace_utils.build_parallel_context(context)
        assert event.meta["effective_total_tokens_compute"] == tokens
        assert event.meta["tokens_are_post_routing"] is routed
        assert event.meta["global_layer_ids"] == [7]
        assert event.layer_id == 7
    events[0].meta["parallel_context"]["PP"] = 99
    assert events[1].meta["parallel_context"]["PP"] == 1
