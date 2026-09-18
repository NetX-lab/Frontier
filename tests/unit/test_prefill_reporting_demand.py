"""Reporting demand must control payload work without changing completion timing."""

from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.metrics.metrics_store import MetricsStore
from tests.unit.test_pdaf_prefill_model_time import (
    _LayerExecutionTime,
    _make_final_sync_scheduler,
)


@pytest.mark.parametrize("write,trace,ops,ledger,summary,utilization", list(product((False, True), repeat=6)))
def test_stage_payload_demand_includes_every_consumer(write, trace, ops, ledger, summary, utilization):
    store = object.__new__(MetricsStore)
    store._config = SimpleNamespace(
        write_metrics=write, enable_op_level_tracing=trace,
        store_operation_metrics=ops, store_frontier_stage_batch_ledger=ledger,
        store_frontier_stage_batch_ledger_summary=summary,
        store_utilization_metrics=utilization,
    )
    expected = trace or (write and (ops or ledger or summary))
    assert store.stage_execution_reporting_enabled is expected
    assert store.ep_wave_reporting_enabled is expected


def test_disabled_reporting_avoids_stage_payload_allocation_and_prediction(monkeypatch):
    import tests.unit.test_pdaf_prefill_model_time as fixture_module

    stage_constructor = Mock(wraps=fixture_module.StageExecutionTime)
    monkeypatch.setattr(fixture_module, "StageExecutionTime", stage_constructor)
    outcomes = []
    for reporting in (True, False):
        stage_constructor.reset_mock()
        batch = SimpleNamespace(
            id=8, is_idle=False, _prefill_stage_start_time=10.0,
            _prefill_model_execution_components_ms_by_stage={0: [1.0, 2.0]},
            schedule_epoch=0, request_execution_signatures=[],
            request_mutation_signatures=[], thinking_round_start_times=[],
        )
        execution = _LayerExecutionTime(attention_ms=1.0, post_attention_ms=2.0, pipeline_ms=0.5)
        scheduler, batch_stage = _make_final_sync_scheduler(
            batch=batch, execution_time=execution, num_layers=2,
        )
        metrics = Mock(stage_execution_reporting_enabled=reporting)
        events = scheduler.on_prefill_sync_collective(
            time=10.01, replica_id=0, stage_id=0, batch_global_id=9,
            sync_stage="post_moe", layer_id=1, metrics_store=metrics,
        )
        assert scheduler._predictor.calls == ([(1, 1), (2, 0)] if reporting else [(1, 1)])
        assert stage_constructor.call_count == int(reporting)
        assert scheduler._create_prefill_corrected_execution_time_for_metrics.call_count == int(reporting)
        metrics.on_replica_stage_schedule.assert_called_once()
        payload = metrics.on_replica_stage_schedule.call_args.args[4]
        assert (payload is not None) is reporting
        outcomes.append((
            [event.time for event in events],
            batch_stage.override_execution_time.call_args,
            batch_stage.override_model_execution_time.call_args,
        ))
    assert outcomes[0] == outcomes[1]
