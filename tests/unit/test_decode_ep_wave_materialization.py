from __future__ import annotations

from collections import defaultdict
import io
from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent
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
    num_layers = 3

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id == 2


class _ExecutionTime:
    def __init__(
        self,
        *,
        pre_dispatch_ms: float,
        dispatch_ms: float,
        routed_compute_ms: float,
        combine_ms: float,
    ) -> None:
        self._pre_dispatch_ms = pre_dispatch_ms
        self._dispatch_ms = dispatch_ms
        self._routed_compute_ms = routed_compute_ms
        self._combine_ms = combine_ms
        self.expert_parallel_communication_time = dispatch_ms + combine_ms
        self.pipeline_time = 0.0
        self.total_time = 0.0
        self.model_time = 0.0
        self.decode_draft_proposer_time = 0.0

    def get_single_layer_post_attention_time(self) -> float:
        return (
            self._pre_dispatch_ms
            + self._dispatch_ms
            + self._routed_compute_ms
            + self._combine_ms
        )

    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return self._pre_dispatch_ms

    def get_single_layer_moe_dispatch_time(self) -> float:
        return self._dispatch_ms

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return self._routed_compute_ms

    def get_single_layer_moe_combine_time(self) -> float:
        return self._combine_ms

    def get_single_layer_attention_time(self) -> float:
        return 2.0


class _LanePredictor:
    _num_layers_per_pipeline_stage = 3

    def __init__(self) -> None:
        self._decode_routing_details = {
            0: {2: {0: 0.25, 1: 0.0, 2: 0.0, 3: 0.75}}
        }
        self.calls: list[tuple[int, dict[int, int]]] = []
        self.include_attention_calls: list[bool] = []

    def predict_stage_execution_time(
        self,
        batch,
        _stage_id,
        cluster_type=None,
        num_layers=None,
        layer_id=None,
        include_attention=True,
        **_kwargs,
    ):
        assert cluster_type is ClusterType.DECODE
        assert num_layers == 1
        per_expert = dict(getattr(batch, "per_expert_tokens", {}))
        self.calls.append((layer_id, per_expert))
        self.include_attention_calls.append(include_attention)
        return _ExecutionTime(
            **{
                0: {
                    "pre_dispatch_ms": 0.0,
                    "dispatch_ms": 0.0,
                    "routed_compute_ms": 3.0,
                    "combine_ms": 0.0,
                },
                1: {
                    "pre_dispatch_ms": 0.5,
                    "dispatch_ms": 0.25,
                    "routed_compute_ms": 1.5,
                    "combine_ms": 0.75,
                },
                2: {
                    "pre_dispatch_ms": 0.5,
                    "dispatch_ms": 0.25,
                    "routed_compute_ms": 1.5,
                    "combine_ms": 0.75,
                },
                3: {
                    "pre_dispatch_ms": 2.0,
                    "dispatch_ms": 1.0,
                    "routed_compute_ms": 3.0,
                    "combine_ms": 3.0,
                },
                4: {
                    "pre_dispatch_ms": 2.0,
                    "dispatch_ms": 1.0,
                    "routed_compute_ms": 3.0,
                    "combine_ms": 3.0,
                },
            }[sum(per_expert.values())]
        )


def _scheduler() -> tuple[RoundRobinClusterScheduler, _LanePredictor, Batch]:
    predictor = _LanePredictor()
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=_ModelConfig(),
            total_expert_num=4,
            moe_expert_parallel_size=2,
            moe_tensor_parallel_size=1,
            router_topk=1,
            attn_dp=1,
        )
    )
    scheduler._predictor = predictor
    scheduler._decode_sync_waiting_room = defaultdict(
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
    request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=4)
    request._is_prefill_complete = True
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(7)
    batch.time = 0.0
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    ticket = context.enqueue_full_stage(
        operation_id=("stage_batch", batch.id, batch.schedule_epoch)
    )
    assert context.try_acquire(ticket) is True
    scheduler._stage_execution_contexts = {(0, 0): context}
    batch._stage_admission_ticket = ticket
    return scheduler, predictor, batch


def test_decode_moe_layer_uses_local_ep_wave_and_slowest_lane_barrier(monkeypatch) -> None:
    scheduler, predictor, batch = _scheduler()
    import frontier.logger as frontier_logging

    log_stream = io.StringIO()
    monkeypatch.setattr(frontier_logging._default_handler, "stream", log_stream)

    events = scheduler._on_decode_ep_wave_ready(
        time=0.01,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=2,
    )

    assert len(events) == 1
    assert isinstance(events[0], DecodeSyncCollectiveEvent)
    assert events[0].time == pytest.approx(0.019)
    assert predictor.calls == [
        (2, {0: 1, 1: 0}),
        (2, {2: 0, 3: 3}),
    ]
    assert batch._decode_ep_wave_lane_times_ms == (2.0, 5.0)
    assert not hasattr(batch, "_decode_ep_wave_post_moe_comm_time_s")
    assert batch._stage_admission_ticket.scope == EP_WAVE
    assert batch._stage_admission_scope_history[-1]["participant_ep_ids"] == (0, 1)
    room = scheduler._decode_sync_waiting_room[0][0][7][2]["post_moe"]
    assert room["batches"] == {0: batch}
    captured = log_stream.getvalue().splitlines()
    workload_lines = [line for line in captured if "[EP-WORKLOAD]" in line]
    assert len(workload_lines) == 2
    assert "[EP-WORKLOAD][DECODE]" in workload_lines[0]
    assert "ep_id=0" in workload_lines[0]
    assert "lane_compute_ms=2.000000" in workload_lines[0]
    assert "routed_compute_ms=1.500000" in workload_lines[0]
    assert "lane_comm_ms=1.000000" in workload_lines[0]
    barrier_lines = [line for line in captured if "[EP-BARRIER]" in line]
    assert len(barrier_lines) == 2
    assert "[EP-BARRIER][DECODE]" in barrier_lines[0]
    assert "phase=dispatch" in barrier_lines[0]
    assert "expected_ep_ids=[0, 1]" in barrier_lines[0]
    assert "arrived_ep_ids=[0, 1]" in barrier_lines[0]
    assert "max_lane_time_ms=2.000000" in barrier_lines[0]
    assert "barrier_time_ms=3.000000" in barrier_lines[0]
    assert "barrier_end_time_s=0.013000" in barrier_lines[0]
    assert "[EP-BARRIER][DECODE]" in barrier_lines[1]
    assert "phase=combine" in barrier_lines[1]
    assert "expected_ep_ids=[0, 1]" in barrier_lines[1]
    assert "arrived_ep_ids=[0, 1]" in barrier_lines[1]
    assert "max_lane_time_ms=3.000000" in barrier_lines[1]
    assert "barrier_time_ms=6.000000" in barrier_lines[1]
    assert "barrier_end_time_s=0.019000" in barrier_lines[1]


def test_decode_moe_ep_lane_prediction_excludes_attention() -> None:
    scheduler, predictor, batch = _scheduler()

    scheduler._on_decode_ep_wave_ready(
        time=0.01,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=2,
    )

    assert predictor.include_attention_calls == [False, False]


def test_decode_dense_layer_bypasses_ep_materializer(monkeypatch) -> None:
    scheduler, predictor, batch = _scheduler()

    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler.materialize_layer_ep_workload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense decode layer must not materialize EP workload")
        ),
    )

    events = scheduler._on_decode_ep_wave_ready(
        time=0.01,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=1,
    )

    assert len(events) == 1
    assert isinstance(events[0], DenseLayerCompleteEvent)
    assert events[0].time == pytest.approx(0.013)
    assert predictor.calls == [(1, {})]
    assert batch._stage_admission_ticket.scope == FULL_STAGE_WORLD


def test_decode_ep_collective_communication_is_added_once() -> None:
    scheduler, predictor, batch = _scheduler()
    predictor._num_layers_per_pipeline_stage = 4
    scheduler._config.replica_config.model_config.num_layers = 4
    scheduler.get_replica_stage_scheduler = lambda *_args: SimpleNamespace(
        _execution_time_predictor=predictor,
        is_last_stage=False,
    )

    wave_events = scheduler._on_decode_ep_wave_ready(
        time=0.01,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=2,
    )
    assert wave_events[0].time == pytest.approx(0.019)

    transition_events = scheduler.on_decode_sync_collective(
        time=wave_events[0].time,
        replica_id=0,
        stage_id=0,
        batch_global_id=batch.global_id,
        sync_stage="post_moe",
        layer_id=2,
        metrics_store=SimpleNamespace(),
    )

    assert len(transition_events) == 1
    assert transition_events[0].time == pytest.approx(0.021)
    assert batch._stage_admission_ticket.scope == FULL_STAGE_WORLD
