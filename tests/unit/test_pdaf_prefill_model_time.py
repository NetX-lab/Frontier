import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def test_prefill_final_sync_records_elapsed_model_time_not_full_stage_prediction() -> None:
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._cluster_type = ClusterType.PREFILL

    class _ExecutionTime:
        pipeline_time = 2.0
        model_time = 15.0
        total_time = 20.0

        def get_single_layer_attention_time(self) -> float:
            return 1.0

        def get_single_layer_post_attention_time(self) -> float:
            return 1.0

        def get_single_layer_dp_input_allreduce_time(self) -> float:
            return 0.0

        def get_single_layer_dp_output_allreduce_time(self) -> float:
            return 0.0

    class _Predictor:
        _num_layers_per_pipeline_stage = 32

        def predict_stage_execution_time(self, _batch, _stage_id, *, cluster_type, num_layers, layer_id):
            assert cluster_type is ClusterType.PREFILL
            if num_layers == 32:
                return SimpleNamespace(
                    pipeline_time=2.0,
                    model_time=449.0,
                    total_time=454.0,
                )
            return _ExecutionTime()

    batch = SimpleNamespace(
        id=7,
        is_idle=False,
        _prefill_stage_start_time=0.0,
        _prefill_model_execution_components_ms_by_stage={0: [15.0]},
        schedule_epoch=0,
        request_execution_signatures=[],
        request_mutation_signatures=[],
        thinking_round_start_times=[],
    )
    batch_stage = Mock()
    stage_scheduler = SimpleNamespace(
        is_last_stage=True,
        _execution_time_predictor=_Predictor(),
        predict_and_create_stage=Mock(return_value=(batch_stage, None)),
    )

    scheduler._predictor = _Predictor()
    scheduler._prefill_sync_waiting_room = {
        0: {0: {9: {31: {"post_moe": {"batches": {0: batch}}}}}}
    }
    scheduler.get_replica_stage_scheduler = Mock(return_value=stage_scheduler)
    scheduler.get_replica = Mock(return_value=SimpleNamespace(dp_size=1))
    scheduler._create_prefill_corrected_execution_time_for_metrics = Mock(
        return_value=_ExecutionTime()
    )
    scheduler._should_trigger_kv_transfer = Mock(return_value=False)

    scheduler.on_prefill_sync_collective(
        time=0.015,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="post_moe",
        layer_id=31,
        metrics_store=Mock(),
    )

    # 15 ms of elapsed model work plus the 2 ms pipeline handoff remains the
    # stage's model time; the 449 ms full-stage prediction is not re-counted.
    batch_stage.override_model_execution_time.assert_called_once_with(
        pytest.approx(0.017)
    )


class _LayerExecutionTime:
    def __init__(
        self,
        *,
        attention_ms: float,
        post_attention_ms: float,
        pipeline_ms: float,
    ) -> None:
        self._attention_ms = attention_ms
        self._post_attention_ms = post_attention_ms
        self.pipeline_time = pipeline_ms
        self.model_time = (
            attention_ms + post_attention_ms + pipeline_ms
        ) * 1e-3
        self.total_time = self.model_time

    def get_single_layer_attention_time(self) -> float:
        return self._attention_ms

    def get_single_layer_post_attention_time(self) -> float:
        return self._post_attention_ms

    def get_single_layer_dp_input_allreduce_time(self) -> float:
        return 0.0

    def get_single_layer_dp_output_allreduce_time(self) -> float:
        return 0.0


class _LayerPredictor:
    def __init__(self, layer_times: dict[int, _LayerExecutionTime]) -> None:
        self._layer_times = layer_times
        self._num_layers_per_pipeline_stage = len(layer_times)
        self.calls: list[tuple[int, int]] = []

    def predict_stage_execution_time(
        self,
        _batch,
        _stage_id,
        cluster_type=None,
        *,
        num_layers,
        layer_id=None,
    ):
        assert cluster_type is ClusterType.PREFILL
        assert num_layers == 1
        assert layer_id is not None
        self.calls.append((num_layers, layer_id))
        return self._layer_times[layer_id]


def _make_final_sync_scheduler(
    *,
    batch,
    execution_time: _LayerExecutionTime,
    num_layers: int,
) -> tuple[_ConcreteClusterScheduler, Mock]:
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._cluster_type = ClusterType.PREFILL

    predictor = _LayerPredictor({
        layer_id: execution_time for layer_id in range(num_layers)
    })
    batch_stage = Mock()
    stage_scheduler = SimpleNamespace(
        is_last_stage=True,
        _execution_time_predictor=predictor,
        predict_and_create_stage=Mock(return_value=(batch_stage, None)),
    )

    scheduler._predictor = predictor
    scheduler._prefill_sync_waiting_room = {
        0: {
            0: {
                9: {
                    num_layers - 1: {
                        "post_moe": {"batches": {0: batch}}
                    }
                }
            }
        }
    }
    scheduler.get_replica_stage_scheduler = Mock(return_value=stage_scheduler)
    scheduler.get_replica = Mock(return_value=SimpleNamespace(dp_size=1))
    scheduler._create_prefill_corrected_execution_time_for_metrics = Mock(
        return_value=execution_time
    )
    scheduler._should_trigger_kv_transfer = Mock(return_value=False)
    return scheduler, batch_stage


def test_prefill_final_sync_uses_component_ledger_without_timestamp_residue() -> None:
    components_ms = [7.0, 8.0] * 32
    stage_start_time = 2.040120574549602
    explicit_component_time = math.fsum(components_ms) * 1e-3
    final_sync_time = stage_start_time
    for component_ms in components_ms:
        final_sync_time += component_ms * 1e-3
    synchronization_wait = 0.125
    final_sync_time += synchronization_wait

    assert final_sync_time - stage_start_time != explicit_component_time

    batch = SimpleNamespace(
        id=8,
        is_idle=False,
        _prefill_stage_start_time=stage_start_time,
        _prefill_model_execution_components_ms_by_stage={0: components_ms},
        schedule_epoch=0,
        request_execution_signatures=[],
        request_mutation_signatures=[],
        thinking_round_start_times=[],
    )
    execution_time = _LayerExecutionTime(
        attention_ms=7.0,
        post_attention_ms=8.0,
        pipeline_ms=1.0,
    )
    execution_time.total_time = 21.0 * 1e-3
    scheduler, batch_stage = _make_final_sync_scheduler(
        batch=batch,
        execution_time=execution_time,
        num_layers=32,
    )

    events = scheduler.on_prefill_sync_collective(
        time=final_sync_time,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="post_moe",
        layer_id=31,
        metrics_store=Mock(),
    )

    stage_cpu_overhead = execution_time.total_time - execution_time.model_time
    final_stage_increment = (
        execution_time.pipeline_time * 1e-3 + stage_cpu_overhead
    )
    completion_time = final_sync_time + final_stage_increment
    expected_wall_time = completion_time - stage_start_time
    reconstructed_wall_time = (
        final_sync_time
        - stage_start_time
        + final_stage_increment
    )

    assert reconstructed_wall_time != expected_wall_time
    assert len(events) == 1
    assert events[0].time == completion_time
    batch_stage.override_model_execution_time.assert_called_once_with(0.481)
    batch_stage.override_execution_time.assert_called_once_with(
        expected_wall_time
    )
    assert expected_wall_time > 0.481 + stage_cpu_overhead


def test_prefill_final_sync_fails_fast_without_component_ledger() -> None:
    batch = SimpleNamespace(
        id=9,
        is_idle=False,
        _prefill_stage_start_time=10.0,
        schedule_epoch=0,
        request_execution_signatures=[],
        request_mutation_signatures=[],
        thinking_round_start_times=[],
    )
    execution_time = _LayerExecutionTime(
        attention_ms=1.0,
        post_attention_ms=1.0,
        pipeline_ms=1.0,
    )
    scheduler, _ = _make_final_sync_scheduler(
        batch=batch,
        execution_time=execution_time,
        num_layers=1,
    )

    with pytest.raises(
        ValueError,
        match="missing PREFILL model-execution component ledger",
    ):
        scheduler.on_prefill_sync_collective(
            time=10.002,
            replica_id=0,
            stage_id=0,
            batch_global_id=9,
            sync_stage="post_moe",
            layer_id=0,
            metrics_store=Mock(),
        )


def test_prefill_sync_records_heterogeneous_layer_components_once() -> None:
    layer_times = {
        0: _LayerExecutionTime(
            attention_ms=1.25,
            post_attention_ms=2.5,
            pipeline_ms=0.75,
        ),
        1: _LayerExecutionTime(
            attention_ms=3.75,
            post_attention_ms=4.5,
            pipeline_ms=0.75,
        ),
    }
    predictor = _LayerPredictor(layer_times)
    stage_start_time = 10.0
    batch = SimpleNamespace(
        id=10,
        global_id=9,
        is_idle=False,
        total_num_tokens=8,
        _prefill_stage_start_time=stage_start_time,
        _prefill_model_execution_components_ms_by_stage={0: [1.25]},
        schedule_epoch=0,
        request_execution_signatures=[],
        request_mutation_signatures=[],
        thinking_round_start_times=[],
    )
    batch_stage = Mock()
    stage_scheduler = SimpleNamespace(
        is_last_stage=True,
        _execution_time_predictor=predictor,
        predict_and_create_stage=Mock(return_value=(batch_stage, None)),
    )
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._predictor = predictor
    scheduler.get_replica_stage_scheduler = Mock(return_value=stage_scheduler)
    scheduler.get_replica = Mock(return_value=SimpleNamespace(dp_size=1))
    scheduler._create_prefill_corrected_execution_time_for_metrics = Mock(
        return_value=layer_times[1]
    )
    scheduler._should_trigger_kv_transfer = Mock(return_value=False)

    def set_waiting_room(layer_id: int, sync_stage: str) -> None:
        scheduler._prefill_sync_waiting_room = {
            0: {
                0: {
                    9: {
                        layer_id: {
                            sync_stage: {"batches": {0: batch}}
                        }
                    }
                }
            }
        }

    set_waiting_room(0, "pre_moe")
    events = scheduler.on_prefill_sync_collective(
        time=stage_start_time + 0.00125,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="pre_moe",
        layer_id=0,
        metrics_store=Mock(),
    )

    set_waiting_room(0, "post_moe")
    events = scheduler.on_prefill_sync_collective(
        time=events[0].time,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="post_moe",
        layer_id=0,
        metrics_store=Mock(),
    )

    set_waiting_room(1, "pre_moe")
    events = scheduler.on_prefill_sync_collective(
        time=events[0].time,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="pre_moe",
        layer_id=1,
        metrics_store=Mock(),
    )

    set_waiting_room(1, "post_moe")
    scheduler.on_prefill_sync_collective(
        time=events[0].time,
        replica_id=0,
        stage_id=0,
        batch_global_id=9,
        sync_stage="post_moe",
        layer_id=1,
        metrics_store=Mock(),
    )

    expected_components_ms = [1.25, 2.5, 3.75, 4.5]
    assert batch._prefill_model_execution_components_ms_by_stage == {
        0: expected_components_ms
    }
    expected_model_time = (
        math.fsum(expected_components_ms) + layer_times[1].pipeline_time
    ) * 1e-3
    batch_stage.override_model_execution_time.assert_called_once_with(
        expected_model_time
    )


def test_prefill_stage_schedule_resets_component_ledger_for_pipeline_stage() -> None:
    execution_time = _LayerExecutionTime(
        attention_ms=1.25,
        post_attention_ms=2.5,
        pipeline_ms=0.75,
    )
    predictor = _LayerPredictor({0: execution_time})
    batch = SimpleNamespace(
        id=11,
        global_id=9,
        requests=[],
        num_prefill_tokens=8,
        num_decode_tokens=0,
        _prefill_model_execution_components_ms_by_stage={99: [123.0]},
    )
    stage_scheduler = SimpleNamespace(
        is_busy=False,
        get_queue_batches=Mock(return_value=[batch]),
        pop_batch_if_not_busy=Mock(return_value=batch),
        consume_last_stale_drop_count=Mock(return_value=0),
        _execution_time_predictor=predictor,
    )
    cluster_scheduler = SimpleNamespace(
        get_replica_stage_scheduler=Mock(return_value=stage_scheduler),
        get_replica=Mock(
            return_value=SimpleNamespace(
                is_moe=True,
                dp_size=2,
                num_moe_expert_parallel_size=1,
            )
        ),
    )
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler)
    )

    events = ReplicaStageScheduleEvent(
        time=42.0,
        replica_id=0,
        stage_id=3,
        cluster_type=ClusterType.PREFILL,
        replica_local_id=0,
    ).handle_event(global_scheduler, Mock())

    assert len(events) == 1
    assert batch._prefill_stage_start_time == 42.0
    assert batch._prefill_model_execution_components_ms_by_stage == {
        3: [1.25]
    }
