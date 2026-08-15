from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.entities.batch import EPBatchGroup
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
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
    def __init__(self, post_attention_ms: float) -> None:
        self._post_attention_ms = post_attention_ms

    def get_single_layer_post_attention_time(self) -> float:
        return self._post_attention_ms


class _LanePredictor:
    _num_layers_per_pipeline_stage = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[int, int]]] = []

    def predict_stage_execution_time(
        self,
        batch,
        _stage_id,
        *,
        cluster_type,
        num_layers,
        layer_id,
    ):
        assert cluster_type is ClusterType.PREFILL
        assert num_layers == 1
        per_expert_tokens = dict(getattr(batch, "per_expert_tokens", {}))
        self.calls.append((layer_id, per_expert_tokens))
        return _ExecutionTime(
            {0: 2.0, 4: 7.0}[sum(per_expert_tokens.values())]
        )


def _scheduler() -> tuple[RoundRobinClusterScheduler, _LanePredictor, Batch]:
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
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    ticket = context.enqueue_full_stage(
        operation_id=("stage_batch", batch.id, batch.schedule_epoch)
    )
    assert context.try_acquire(ticket) is True
    scheduler._stage_execution_contexts = {(0, 0): context}
    batch._stage_admission_ticket = ticket
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch._prefill_stage_start_time = 0.0
    return scheduler, predictor, batch


def test_prefill_moe_layer_materializes_global_distribution_once_and_waits_for_slowest_ep():
    scheduler, predictor, batch = _scheduler()

    events = scheduler._on_prefill_ep_wave_ready(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch,
        layer_id=4,
    )

    assert len(events) == 1
    assert isinstance(events[0], PrefillSyncCollectiveEvent)
    assert events[0].time == pytest.approx(0.008)
    assert predictor.calls == [
        (4, {0: 0, 1: 0}),
        (4, {2: 1, 3: 3}),
    ]
    assert batch._prefill_model_execution_components_ms_by_stage[0] == [1.0, 7.0]
    assert batch._stage_admission_ticket.scope == EP_WAVE
    assert batch._stage_admission_scope_history[-1]["participant_ep_ids"] == (0, 1)
    room = scheduler._prefill_sync_waiting_room[0][0][9][4]["post_moe"]
    assert room["batches"] == {0: batch}


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
        per_expert_tokens={2: 0, 3: 0},
        cluster_type=ClusterType.PREFILL,
        is_moe=True,
    )
    lane.moe_pre_routing_effective_total_tokens = 8

    assert lane.total_num_tokens == 0
    assert lane.get_effective_total_tokens_for_compute(ClusterType.PREFILL) == 8
