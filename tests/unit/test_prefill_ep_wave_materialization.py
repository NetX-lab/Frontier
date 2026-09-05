from __future__ import annotations

from collections import defaultdict
import io
from types import SimpleNamespace

import numpy as np
import pytest

from frontier.entities import Batch, Request
from frontier.entities.batch import EPBatchGroup
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
from frontier.events.prefill_sync_event import PrefillSyncEvent
from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    EP_WAVE,
    FULL_STAGE_WORLD,
    StageExecutionContext,
)
from frontier.types import ClusterType


class _ModelConfig:
    is_moe = True

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id == 4


class _ExecutionTime:
    def __init__(
        self,
        *,
        pre_dispatch_ms: float,
        dispatch_ms: float,
        routed_compute_ms: float,
        combine_ms: float,
        post_combine_ms: float = 0.0,
    ) -> None:
        self._pre_dispatch_ms = pre_dispatch_ms
        self._dispatch_ms = dispatch_ms
        self._routed_compute_ms = routed_compute_ms
        self._combine_ms = combine_ms
        self._post_combine_ms = post_combine_ms
        self.expert_parallel_communication_time = dispatch_ms + combine_ms

    def get_single_layer_post_attention_time(self) -> float:
        return (
            self._pre_dispatch_ms
            + self._dispatch_ms
            + self._routed_compute_ms
            + self._combine_ms
            + self._post_combine_ms
        )

    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return self._pre_dispatch_ms

    def get_single_layer_moe_dispatch_time(self) -> float:
        return self._dispatch_ms

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return self._routed_compute_ms

    def get_single_layer_moe_combine_time(self) -> float:
        return self._combine_ms

    def get_single_layer_moe_post_combine_time(self) -> float:
        return self._post_combine_ms


class _LanePredictor:
    _num_layers_per_pipeline_stage = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[int, int]]] = []
        self.include_attention_calls: list[bool] = []

    def predict_stage_execution_time(
        self,
        batch,
        _stage_id,
        *,
        cluster_type,
        num_layers,
        layer_id,
        include_attention=True,
    ):
        assert cluster_type is ClusterType.PREFILL
        assert num_layers == 1
        self.include_attention_calls.append(include_attention)
        per_expert_tokens = dict(getattr(batch, "per_expert_tokens", {}))
        self.calls.append((layer_id, per_expert_tokens))
        if layer_id == 3:
            return _ExecutionTime(
                pre_dispatch_ms=0.0,
                dispatch_ms=0.0,
                routed_compute_ms=2.0,
                combine_ms=0.0,
            )
        return _ExecutionTime(
            **{
                0: {
                    "pre_dispatch_ms": 2.0,
                    "dispatch_ms": 1.0,
                    "routed_compute_ms": 0.0,
                    "combine_ms": 1.0,
                    "post_combine_ms": 2.0,
                },
                7: {
                    "pre_dispatch_ms": 1.0,
                    "dispatch_ms": 2.0,
                    "routed_compute_ms": 4.0,
                    "combine_ms": 3.0,
                    "post_combine_ms": 2.0,
                },
                4: {
                    "pre_dispatch_ms": 1.0,
                    "dispatch_ms": 2.0,
                    "routed_compute_ms": 4.0,
                    "combine_ms": 3.0,
                    "post_combine_ms": 2.0,
                },
            }[sum(per_expert_tokens.values())]
        )


def _scheduler(
    *, lane_capacity: int = 1
) -> tuple[RoundRobinClusterScheduler, _LanePredictor, Batch]:
    predictor = _LanePredictor()
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=_ModelConfig(),
            total_expert_num=4,
            moe_expert_parallel_size=2,
            moe_tensor_parallel_size=1,
            router_topk=1,
        )
    )
    scheduler._predictor = predictor
    predictor._prefill_routing_details = {
        0: {
            4: {0: 0.0, 1: 0.0, 2: 0.25, 3: 0.75},
        }
    }
    scheduler._prefill_sync_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: {"batches": {}, "arrival_times": {}}
                    )
                )
            )
        )
    )
    scheduler.get_replica_stage_scheduler = lambda *_args: SimpleNamespace(
        _execution_time_predictor=predictor,
    )
    request = Request(arrived_at=0.0, num_prefill_tokens=4, num_decode_tokens=0)
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(9)
    batch.time = 0.0
    context = StageExecutionContext(
        replica_id=0,
        stage_id=0,
        ep_size=2,
        full_stage_capacity=lane_capacity,
    )
    ticket = context.enqueue_full_stage(
        operation_id=("stage_batch", batch.id, batch.schedule_epoch)
    )
    assert context.try_acquire(ticket) is True
    scheduler._stage_execution_contexts = {(0, 0): context}
    batch._stage_admission_ticket = ticket
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch._prefill_stage_start_time = 0.0
    return scheduler, predictor, batch


@pytest.mark.parametrize("operation", ("promote", "restore"))
def test_cohort_materialization_rejects_live_batch_without_admission_ticket(
    operation: str,
) -> None:
    scheduler, _predictor, batch_zero = _scheduler(lane_capacity=2)
    context = scheduler._stage_execution_contexts[(0, 0)]
    batch_zero._forward_cohort_id = 19
    batch_zero._stage_owner_replica_local_id = 0

    batch_one = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=2, num_decode_tokens=0)],
        [2],
        is_moe=True,
    )
    batch_one.set_global_id(20)
    batch_one._forward_cohort_id = 19
    batch_one._stage_owner_replica_local_id = 1
    source_batches = {0: batch_zero, 1: batch_one}

    if operation == "restore":
        batch_zero._stage_admission_ticket = context.transition_active_scope(
            batch_zero._stage_admission_ticket,
            operation_id=("ep_wave", 19),
            scope=EP_WAVE,
            participant_ep_ids=(0, 1),
        )
        operation_args = {
            "source_batches": source_batches,
            "replica_id": 0,
            "stage_id": 0,
            "layer_id": 4,
            "cohort_id": 19,
            "operation_kind": "ffn",
        }
        expected_message = "cohort full-stage restoration requires"
        method = scheduler._restore_cohort_full_stage_owners
    else:
        operation_args = {
            "source_batches": source_batches,
            "replica_id": 0,
            "stage_id": 0,
            "layer_id": 4,
            "cohort_id": 19,
            "participant_ep_ids": (0, 1),
        }
        expected_message = "cohort EP promotion requires"
        method = scheduler._promote_cohort_to_ep_wave

    with pytest.raises(ValueError, match=expected_message):
        method(**operation_args)


def test_prefill_moe_layer_materializes_global_distribution_once_and_waits_for_slowest_ep(
    monkeypatch,
):
    scheduler, predictor, batch = _scheduler()
    import frontier.logger as frontier_logging

    log_stream = io.StringIO()
    monkeypatch.setattr(frontier_logging._default_handler, "stream", log_stream)

    events = scheduler._on_prefill_ep_wave_ready(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=4,
    )

    assert len(events) == 1
    assert isinstance(events[0], PrefillSyncCollectiveEvent)
    assert events[0].time == pytest.approx(0.014)
    assert predictor.calls == [
        (4, {0: 0, 1: 0}),
        (4, {2: 1, 3: 3}),
    ]
    assert predictor.include_attention_calls == [False, False]
    assert batch._prefill_model_execution_components_ms_by_stage[0] == [1.0, 13.0]
    assert batch._prefill_ep_wave_lane_times_ms == (4.0, 7.0)
    assert batch._stage_admission_ticket.scope == EP_WAVE
    assert batch._stage_admission_scope_history[-1]["participant_ep_ids"] == (0, 1)
    room = scheduler._prefill_sync_waiting_room[0][0][9][4]["post_moe"]
    assert room["batches"] == {0: batch}
    captured = log_stream.getvalue().splitlines()
    workload_lines = [line for line in captured if "[EP-WORKLOAD]" in line]
    assert len(workload_lines) == 2
    assert "[EP-WORKLOAD][PREFILL]" in workload_lines[0]
    assert "ep_id=0" in workload_lines[0]
    assert "per_expert_tokens={0: 0, 1: 0}" in workload_lines[0]
    assert "lane_compute_ms=4.000000" in workload_lines[0]
    assert "routed_compute_ms=0.000000" in workload_lines[0]
    assert "lane_comm_ms=2.000000" in workload_lines[0]
    barrier_lines = [line for line in captured if "[EP-BARRIER]" in line]
    assert len(barrier_lines) == 2
    assert "[EP-BARRIER][PREFILL]" in barrier_lines[0]
    assert "phase=dispatch" in barrier_lines[0]
    assert "expected_ep_ids=[0, 1]" in barrier_lines[0]
    assert "arrived_ep_ids=[0, 1]" in barrier_lines[0]
    assert "max_lane_time_ms=2.000000" in barrier_lines[0]
    assert "barrier_time_ms=4.000000" in barrier_lines[0]
    assert "barrier_end_time_s=0.005000" in barrier_lines[0]
    assert "[EP-BARRIER][PREFILL]" in barrier_lines[1]
    assert "phase=combine" in barrier_lines[1]
    assert "expected_ep_ids=[0, 1]" in barrier_lines[1]
    assert "arrived_ep_ids=[0, 1]" in barrier_lines[1]
    assert "max_lane_time_ms=4.000000" in barrier_lines[1]
    assert "barrier_time_ms=7.000000" in barrier_lines[1]
    assert "barrier_end_time_s=0.012000" in barrier_lines[1]


def test_prefill_ep_wave_aggregates_attention_dp_lanes_once():
    scheduler, predictor, batch_zero = _scheduler(lane_capacity=2)
    context = scheduler._stage_execution_contexts[(0, 0)]
    batch_one = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=4, num_decode_tokens=0)],
        [3],
        is_moe=True,
    )
    batch_zero._forward_cohort_id = 11
    batch_one._forward_cohort_id = 11
    batch_zero.set_global_id(22)
    batch_one.set_global_id(23)
    batch_one.time = 0.0
    batch_one._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch_one._prefill_stage_start_time = 0.0
    ticket = context.enqueue_full_stage(operation_id=("stage_batch", batch_one.id, 0))
    assert context.try_acquire(ticket) is True
    batch_one._stage_admission_ticket = ticket

    events = scheduler._on_prefill_ep_wave_ready(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch_zero,
        cohort_batches={0: batch_zero, 1: batch_one},
        layer_id=4,
    )

    assert len(events) == 1
    assert events[0]._batch_global_id == 11
    assert predictor.calls == [
        (4, {0: 0, 1: 0}),
        (4, {2: 2, 3: 5}),
    ]
    assert batch_zero._prefill_ep_wave_workload.routing_token_count == 7
    assert batch_one._prefill_ep_wave_workload.routing_token_count == 7


def test_prefill_placeholder_stays_replaceable_until_idle_event_is_consumed():
    scheduler, _predictor, batch_zero = _scheduler(lane_capacity=2)
    scheduler._replica_dp_size = 2
    context = scheduler._stage_execution_contexts[(0, 0)]
    batch_zero._forward_cohort_id = 11
    batch_zero._stage_owner_replica_local_id = 0

    batch_one = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=3, num_decode_tokens=0)],
        [3],
        is_moe=True,
    )
    batch_one._forward_cohort_id = 11
    batch_one.set_global_id(23)
    batch_one.time = 0.0
    batch_one._stage_owner_replica_local_id = 1
    batch_one._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch_one._prefill_stage_start_time = 0.0
    ticket = context.enqueue_full_stage(operation_id=("stage_batch", batch_one.id, 0))
    assert context.try_acquire(ticket) is True
    batch_one._stage_admission_ticket = ticket

    empty_lane = lambda: SimpleNamespace(
        get_replica_stage_scheduler=lambda _stage_id: SimpleNamespace(
            is_busy=False,
            is_empty=lambda: True,
        )
    )
    scheduler._replica_schedulers = {(0, 1): empty_lane()}
    first_events = scheduler.on_prefill_sync(
        0.001,
        0,
        0,
        batch_zero,
        0,
        "pre_moe",
        4,
        0.0,
    )

    assert len(first_events) == 1
    assert isinstance(first_events[0], PrefillSyncEvent)
    room = scheduler._prefill_sync_waiting_room[0][0][11][4]["pre_moe"]
    assert room["batches"][0] is batch_zero
    assert room["batches"][1].is_idle

    second_events = scheduler.on_prefill_sync(
        0.002,
        0,
        0,
        batch_one,
        1,
        "pre_moe",
        4,
        0.0,
    )

    assert len(second_events) == 1
    assert isinstance(second_events[0], PrefillSyncCollectiveEvent)
    assert scheduler._prefill_sync_waiting_room[0][0][11][4]["pre_moe"] == {}
    assert batch_zero._prefill_ep_wave_workload.routing_token_count == 7
    assert batch_one._prefill_ep_wave_workload.routing_token_count == 7
    assert first_events[0].handle_event(
        SimpleNamespace(get_cluster_scheduler=lambda _cluster_type: scheduler),
        SimpleNamespace(),
    ) == []

    context.release(batch_zero._stage_admission_ticket)
    batch_zero.__dict__.pop("_stage_admission_ticket", None)
    batch_one.__dict__.pop("_stage_admission_ticket", None)

    late_batch = Batch(
        0,
        [Request(arrived_at=0.003, num_prefill_tokens=4, num_decode_tokens=0)],
        [4],
        is_moe=True,
    )
    late_batch._forward_cohort_id = 11
    late_batch._stage_owner_replica_local_id = 1
    late_batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    late_batch._prefill_stage_start_time = 0.003
    late_ticket = context.enqueue_full_stage(
        operation_id=("stage_batch", late_batch.id, late_batch.schedule_epoch)
    )
    assert context.try_acquire(late_ticket) is True
    late_batch._stage_admission_ticket = late_ticket
    scheduler._replica_schedulers[(0, 0)] = empty_lane()
    late_events = scheduler.on_prefill_sync(
        0.003,
        0,
        0,
        late_batch,
        1,
        "pre_moe",
        4,
        0.0,
    )
    assert len(late_events) == 1
    assert isinstance(late_events[0], PrefillSyncEvent)
    assert late_batch._forward_cohort_id != 11
    fresh_cohort_id = late_batch._forward_cohort_id
    late_room = scheduler._prefill_sync_waiting_room[0][0][fresh_cohort_id][4][
        "pre_moe"
    ]
    assert late_room["batches"][1] is late_batch
    assert late_room["batches"][0].is_idle
    late_collective_events = late_events[0].handle_event(
        SimpleNamespace(get_cluster_scheduler=lambda _cluster_type: scheduler),
        SimpleNamespace(),
    )
    assert len(late_collective_events) == 1
    assert isinstance(late_collective_events[0], PrefillSyncCollectiveEvent)
    assert scheduler._prefill_sync_waiting_room[0][0][fresh_cohort_id][4][
        "post_moe"
    ]["batches"][1] is late_batch


def test_prefill_dense_layer_emits_one_completion_per_attention_dp_owner():
    scheduler, predictor, batch_zero = _scheduler(lane_capacity=2)
    scheduler._replica_dp_size = 2
    context = scheduler._stage_execution_contexts[(0, 0)]
    batch_zero._forward_cohort_id = 13
    batch_zero._stage_owner_replica_local_id = 0

    batch_one = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=3, num_decode_tokens=0)],
        [3],
        is_moe=True,
    )
    batch_one._forward_cohort_id = 13
    batch_one.set_global_id(31)
    batch_one._stage_owner_replica_local_id = 1
    batch_one._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch_one._prefill_stage_start_time = 0.0
    ticket = context.enqueue_full_stage(operation_id=("stage_batch", batch_one.id, 0))
    assert context.try_acquire(ticket) is True
    batch_one._stage_admission_ticket = ticket

    events = scheduler._on_prefill_ep_wave_ready(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch_zero,
        cohort_batches={0: batch_zero, 1: batch_one},
        layer_id=3,
    )

    assert len(events) == 2
    assert all(isinstance(event, DenseLayerCompleteEvent) for event in events)
    assert {event._batch for event in events} == {batch_zero, batch_one}
    assert predictor.calls == [(3, {}), (3, {})]
    assert batch_zero._stage_admission_ticket.scope == FULL_STAGE_WORLD
    assert batch_one._stage_admission_ticket.scope == FULL_STAGE_WORLD
    assert batch_zero._stage_admission_ticket.operation_id != batch_one._stage_admission_ticket.operation_id


def test_prefill_ep_wave_accepts_numpy_timestamp_from_non_dummy_predictor():
    scheduler, _predictor, batch = _scheduler()

    events = scheduler._on_prefill_ep_wave_ready(
        time=np.float64(0.001),
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=4,
    )

    assert len(events) == 1
    assert events[0].time == pytest.approx(0.014)


def test_prefill_dense_layer_bypasses_ep_materializer(monkeypatch):
    scheduler, predictor, batch = _scheduler()
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}

    def fail_materializer(**_kwargs):
        raise AssertionError("dense layer must not materialize EP workload")

    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler.materialize_layer_ep_workload",
        fail_materializer,
    )

    events = scheduler._on_prefill_ep_wave_ready(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=3,
    )

    assert len(events) == 1
    assert isinstance(events[0], DenseLayerCompleteEvent)
    assert events[0].time == pytest.approx(0.003)
    assert predictor.calls == [(3, {})]
    assert batch._stage_admission_ticket.scope == FULL_STAGE_WORLD


def test_shared_ep_lane_preserves_source_pre_routing_tokens_for_zero_lane() -> None:
    request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=0)
    lane = EPBatchGroup(
        requests=[request],
        num_tokens=[0],
        replica_id=0,
        ep_id=1,
        time=0.0,
        source_batch_ids=[1],
        lane_workload=EPLaneWorkload(
            ep_id=1,
            moe_expert_parallel_size=2,
            total_expert_num=4,
            owned_expert_ids=(2, 3),
            local_token_counts=(0, 0),
            routed_token_count=0,
            router_topk=1,
        ),
        cluster_type=ClusterType.PREFILL,
        is_moe=True,
    )
    lane.moe_pre_routing_effective_total_tokens = 8

    assert lane.total_num_tokens == 0
    assert lane.get_effective_total_tokens_for_compute(ClusterType.PREFILL) == 8
