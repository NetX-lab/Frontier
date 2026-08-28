from __future__ import annotations

from types import SimpleNamespace
from collections import defaultdict
from unittest.mock import Mock

import pytest

from frontier.entities import Batch, Request
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
from frontier.events.decode_sync_event import DecodeSyncEvent
from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent
from frontier.events.prefill_sync_event import PrefillSyncEvent
from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType


class _MixedModelConfig:
    is_moe = True
    num_layers = 3

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id == 2


class _ExecutionTime:
    def __init__(self, post_attention_ms: float) -> None:
        self._post_attention_ms = post_attention_ms
        self.expert_parallel_communication_time = 0.0

    def get_single_layer_post_attention_time(self) -> float:
        return self._post_attention_ms

    def get_single_layer_attention_time(self) -> float:
        return 9.0

    def get_single_layer_attention_scope_time(self) -> float:
        return 9.0


class _LayerPredictor:
    _num_layers_per_pipeline_stage = 3

    def __init__(self) -> None:
        self.calls: list[int] = []
        self._monolithic_routing_details = {0: {2: {0: 1.0}}}

    def predict_stage_execution_time(
        self, _batch, _stage_id, cluster_type, num_layers, layer_id, **_kwargs
    ):
        assert num_layers == 1
        self.calls.append(layer_id)
        return _ExecutionTime(4.0 if layer_id == 1 else 9.0)


class _StageOffsetPredictor:
    _num_layers_per_pipeline_stage = 16

    def __init__(self) -> None:
        self.calls: list[tuple[int, ClusterType, int, int]] = []

    def predict_stage_execution_time(
        self,
        _batch,
        stage_id,
        cluster_type=None,
        *,
        num_layers,
        layer_id,
        **_kwargs,
    ):
        self.calls.append((stage_id, cluster_type, num_layers, layer_id))
        return _ExecutionTime(1.0)


def _scheduler(cluster_type: ClusterType):
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=_MixedModelConfig(),
            attn_dp=1,
        )
    )
    scheduler._predictor = SimpleNamespace(
        _prefill_routing_details={0: {2: {0: 1.0}}},
        _decode_routing_details={0: {2: {0: 1.0}}},
        _monolithic_routing_details={0: {2: {0: 1.0}}},
    )
    return scheduler


@pytest.mark.parametrize(
    ("cluster_type", "num_prefill_tokens", "num_decode_tokens", "event_type"),
    [
        (ClusterType.MONOLITHIC, 4, 0, PrefillSyncEvent),
        (ClusterType.MONOLITHIC, 0, 4, DecodeSyncEvent),
    ],
)
def test_shared_pipeline_stage_one_starts_at_global_layer_offset(
    cluster_type: ClusterType,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    event_type,
) -> None:
    predictor = _StageOffsetPredictor()
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
    )
    if num_decode_tokens > 0:
        request._is_prefill_complete = True
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(9)
    stage_scheduler = SimpleNamespace(
        is_busy=False,
        get_queue_batches=Mock(return_value=[batch]),
        pop_batch_if_not_busy=Mock(return_value=batch),
        consume_last_stale_drop_count=Mock(return_value=0),
        _execution_time_predictor=predictor,
    )
    cluster_scheduler = SimpleNamespace(
        get_replica_stage_scheduler=Mock(return_value=stage_scheduler),
        get_replica=Mock(return_value=SimpleNamespace(is_moe=True)),
    )
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler)
    )

    events = ReplicaStageScheduleEvent(
        time=1.0,
        replica_id=0,
        stage_id=1,
        cluster_type=cluster_type,
        replica_local_id=None,
    ).handle_event(global_scheduler, Mock())

    assert len(events) == 1
    assert isinstance(events[0], event_type)
    assert events[0]._layer_id == 16
    assert predictor.calls == [(1, cluster_type, 1, 16)]


def _room():
    return defaultdict(
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


def test_monolithic_prefill_guard_only_admits_moe_layers() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)

    assert scheduler._uses_shared_prefill_ep_wave(None, 2) is True
    assert scheduler._uses_shared_prefill_ep_wave(None, 1) is False
    assert scheduler._uses_shared_prefill_layer_protocol(None, 1) is True


def test_monolithic_decode_guard_only_admits_moe_layers() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)

    assert scheduler._uses_shared_decode_ep_wave(None, 2) is True
    assert scheduler._uses_shared_decode_ep_wave(None, 1) is False
    assert scheduler._uses_shared_decode_layer_protocol(None, 1) is True


def test_monolithic_prefill_dense_layer_uses_full_stage_protocol_without_ep_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)
    predictor = _LayerPredictor()
    scheduler._predictor = predictor
    scheduler._prefill_sync_waiting_room = _room()
    scheduler._config.replica_config.total_expert_num = 2
    scheduler._config.replica_config.moe_expert_parallel_size = 1
    scheduler._config.replica_config.router_topk = 1
    request = Request(0.0, 4, 0)
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(3)
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}

    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler.materialize_layer_ep_workload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense layer must not materialize EP workload")
        ),
    )

    events = scheduler.on_prefill_sync(
        time=0.001,
        replica_id=0,
        stage_id=0,
        batch=batch,
        replica_local_id=0,
        sync_stage="pre_moe",
        layer_id=1,
        stage_execution_time=0.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], DenseLayerCompleteEvent)
    assert events[0].time == pytest.approx(0.005)
    assert predictor.calls == [1]


def test_monolithic_decode_dense_layer_uses_full_stage_protocol_without_ep_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)
    predictor = _LayerPredictor()
    scheduler._predictor = predictor
    scheduler._decode_sync_waiting_room = _room()
    scheduler.get_replica_stage_scheduler = lambda *_args: SimpleNamespace(
        _execution_time_predictor=predictor,
        is_last_stage=False,
    )
    scheduler.get_replica = lambda _replica_id: SimpleNamespace(ep_size=1)
    scheduler._config.replica_config.total_expert_num = 2
    scheduler._config.replica_config.moe_expert_parallel_size = 1
    scheduler._config.replica_config.router_topk = 1
    request = Request(0.0, 0, 4)
    request._is_prefill_complete = True
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(5)
    batch.decode_sync_global_id = 5

    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler.materialize_layer_ep_workload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense layer must not materialize EP workload")
        ),
    )

    events = scheduler.on_decode_sync(
        time=0.002,
        replica_id=0,
        stage_id=0,
        batch=batch,
        replica_local_id=0,
        sync_stage="pre_moe",
        layer_id=1,
        stage_execution_time=0.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], DenseLayerCompleteEvent)
    assert events[0].time == pytest.approx(0.006)
    assert predictor.calls == [1]


def test_dense_prefill_completion_advances_to_next_layer_without_collective() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)
    predictor = _LayerPredictor()
    scheduler._predictor = predictor
    scheduler._prefill_sync_waiting_room = _room()
    request = Request(0.0, 4, 0)
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(3)
    batch._prefill_stage_start_time = 0.0
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0, 4.0]}

    events = scheduler.on_dense_layer_complete(
        0.005,
        0,
        0,
        batch,
        1,
        "prefill",
        object(),
    )

    assert len(events) == 1
    assert isinstance(events[0], PrefillSyncEvent)
    assert events[0]._layer_id == 2
    assert events[0].time == pytest.approx(0.014)
    assert 1 not in scheduler._prefill_sync_waiting_room[0][0][3]


def test_dense_decode_completion_advances_to_next_layer_without_collective() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)
    predictor = _LayerPredictor()
    scheduler._predictor = predictor
    scheduler._decode_sync_waiting_room = _room()
    scheduler.get_replica_stage_scheduler = lambda *_args: SimpleNamespace(
        _execution_time_predictor=predictor,
        is_last_stage=False,
    )
    scheduler.get_replica = lambda _replica_id: SimpleNamespace(ep_size=1)
    request = Request(0.0, 0, 4)
    request._is_prefill_complete = True
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(5)
    batch.decode_sync_global_id = 5
    batch._decode_stage_start_time = 0.0

    events = scheduler.on_dense_layer_complete(
        0.006,
        0,
        0,
        batch,
        1,
        "decode",
        object(),
    )

    assert len(events) == 1
    assert isinstance(events[0], DecodeSyncEvent)
    assert events[0]._layer_id == 2
    assert events[0].time == pytest.approx(0.015)
    assert 1 not in scheduler._decode_sync_waiting_room[0][0][5]
