from __future__ import annotations

from collections import defaultdict
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

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id == 2


class _ExecutionTime:
    def __init__(self, post_attention_ms: float, ep_comm_ms: float) -> None:
        self._post_attention_ms = post_attention_ms
        self.expert_parallel_communication_time = ep_comm_ms

    def get_single_layer_post_attention_time(self) -> float:
        return self._post_attention_ms


class _LanePredictor:
    _num_layers_per_pipeline_stage = 3

    def __init__(self) -> None:
        self._decode_routing_details = {
            0: {2: {0: 0.25, 1: 0.0, 2: 0.0, 3: 0.75}}
        }
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
        assert cluster_type is ClusterType.DECODE
        assert num_layers == 1
        per_expert = dict(getattr(batch, "per_expert_tokens", {}))
        self.calls.append((layer_id, per_expert))
        return _ExecutionTime(
            post_attention_ms={0: 3.0, 1: 3.0, 2: 3.0, 3: 9.0, 4: 9.0}[
                sum(per_expert.values())
            ],
            ep_comm_ms={0: 0.0, 1: 1.0, 2: 1.0, 3: 4.0, 4: 4.0}[
                sum(per_expert.values())
            ],
        )

    def _get_expert_parallel_communication_time(self, batch) -> float:
        return {1: 1.0, 2: 1.0, 3: 4.0, 4: 4.0}[
            sum(batch.per_expert_tokens.values())
        ]


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
            attn_data_parallel_size=1,
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


def test_decode_moe_layer_uses_local_ep_wave_and_slowest_lane_barrier() -> None:
    scheduler, predictor, batch = _scheduler()

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
    assert batch._decode_ep_wave_lane_times_ms == (3.0, 9.0)
    assert batch._decode_ep_wave_post_moe_comm_time_s == pytest.approx(0.004)
    assert batch._stage_admission_ticket.scope == EP_WAVE
    assert batch._stage_admission_scope_history[-1]["participant_ep_ids"] == (0, 1)
    room = scheduler._decode_sync_waiting_room[0][0][7][2]["post_moe"]
    assert room["batches"] == {0: batch}


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
