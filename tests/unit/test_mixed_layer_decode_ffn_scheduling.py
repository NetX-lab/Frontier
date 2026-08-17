"""Regression tests for mixed dense/MoE DECODE_FFN layer propagation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, Request
from frontier.entities.batch import DenseFFNBatchGroup, EPBatchGroup
from frontier.entities.batch_stage import BatchStage
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.ep_alltoall_dispatch_ready_event import (
    EPAllToAllDispatchReadyEvent,
)
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.moe_ep_workload import LayerEPWorkload
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    EP_WAVE,
    FULL_STAGE_WORLD,
    StageExecutionContext,
)
from frontier.types import ClusterType


def test_dense_ffn_batch_group_has_no_ep_lane_identity() -> None:
    source = Path("frontier/entities/batch.py").read_text(encoding="utf-8")
    scheduler_source = Path(
        "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "lane_id: int" not in source[source.index("class DenseFFNBatchGroup"):]
    assert "lane_id=0" not in scheduler_source
    assert "get_full_stage_replica_scheduler" in scheduler_source


class _ConcreteDisaggregationPredictor(
    SklearnDisaggregationExecutionTimePredictor
):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _ZeroAttributes:
    """Return zero for timing fields that are irrelevant to branch selection."""

    def __getattr__(self, _name: str) -> float:
        return 0.0


class _QueuedBatchSink:
    def __init__(self) -> None:
        self._m2n_immediate_batch_queue = []
        self._activation_bytes_allocated = 0

    def add_batch_to_m2n_queue(self, batch: Batch) -> None:
        self._m2n_immediate_batch_queue.append(batch)
        self._activation_bytes_allocated += batch.activation_bytes


class _ActivationCommitFailingSink:
    """Inject failure after the immediate queue write has completed."""

    def __init__(self) -> None:
        self._m2n_immediate_batch_queue = []
        self._activation_memory = 0

    @property
    def _activation_bytes_allocated(self) -> int:
        return self._activation_memory

    @_activation_bytes_allocated.setter
    def _activation_bytes_allocated(self, _value: int) -> None:
        raise RuntimeError("injected activation counter commit failure")


class _SyntheticEPBatch:
    def __init__(self) -> None:
        self.id = 700
        self.global_id = -1
        self.total_num_tokens = 1
        self.activation_bytes = 0

    def set_global_id(self, global_id: int) -> None:
        self.global_id = global_id


@pytest.fixture(scope="module")
def mixed_model_config():
    return BaseModelConfig.create_from_name("step-moe-noquant")


@pytest.fixture(scope="module")
def dense_model_config():
    return BaseModelConfig.create_from_name("llama2_7b_dense_example")


@pytest.fixture(scope="module")
def generic_moe_model_config():
    return BaseModelConfig.create_from_name("mixtral_8x7b_moe")


def _decode_request() -> Request:
    request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=2)
    request._is_prefill_complete = True
    return request


def _source_batch(*, layer_id: int, afd_stage_idx: int = 2) -> Batch:
    request = _decode_request()
    request._completed_layer_count = layer_id
    batch = Batch(replica_id=0, requests=[request], num_tokens=[1], is_moe=True)
    batch.afd_stage_idx = afd_stage_idx
    batch.decode_attn_original_replica_id = 0
    batch.decode_attn_original_replica_local_id = 0
    batch.time = 0.0
    return batch


def _transfer_info(*, layer_id: int, afd_stage_idx: int = 2):
    return SimpleNamespace(
        layer_id=layer_id,
        afd_stage_idx=afd_stage_idx,
        target_ffn_replica_id=0,
        activation_size_bytes=128,
    )


def _branch_scheduler(model_config, *, layer_id: int):
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_ready_groups = deque(
        [[(_source_batch(layer_id=layer_id), _transfer_info(layer_id=layer_id))]]
    )
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=model_config,
            moe_expert_parallel_size=1,
            local_expert_num=1,
            total_expert_num=1,
            router_topk=1,
        )
    )
    scheduler._predictor = SimpleNamespace(
        _decode_ffn_routing_details={0: {layer_id: {0: 1.0}}}
    )
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._replica_ep_size = 1
    scheduler._batch_group_creation_counter = 0
    scheduler._raw_batch_waiting_for_m2n_back = {}

    dense_result = [(0, 0)]
    scheduler._schedule_dense_ffn_from_m2n_group = Mock(return_value=dense_result)
    ep_batch = _SyntheticEPBatch()
    scheduler._distribute_tokens_within_ep_replica = Mock(return_value=ep_batch)
    queue_sink = _QueuedBatchSink()
    scheduler.get_replica_scheduler = Mock(return_value=queue_sink)
    return scheduler, dense_result


@pytest.mark.parametrize("layer_id", [3, 60])
def test_mixed_dense_layer_uses_dense_ffn_scheduler(
    mixed_model_config, layer_id: int
) -> None:
    scheduler, dense_result = _branch_scheduler(
        mixed_model_config, layer_id=layer_id
    )

    result = scheduler.schedule_ffn_with_m2n_immediate()

    assert result == dense_result
    assert scheduler._schedule_dense_ffn_from_m2n_group.call_count == 1
    assert scheduler._distribute_tokens_within_ep_replica.call_count == 0


@pytest.mark.parametrize("layer_id", [4, 59])
def test_mixed_moe_layer_keeps_ep_ffn_scheduler(
    mixed_model_config, layer_id: int
) -> None:
    scheduler, _ = _branch_scheduler(mixed_model_config, layer_id=layer_id)

    result = scheduler.schedule_ffn_with_m2n_immediate()

    assert result == [(0, 0)]
    assert scheduler._schedule_dense_ffn_from_m2n_group.call_count == 0
    assert scheduler._distribute_tokens_within_ep_replica.call_count == 1


def test_pure_dense_layer_keeps_dense_ffn_scheduler(dense_model_config) -> None:
    scheduler, dense_result = _branch_scheduler(dense_model_config, layer_id=0)

    result = scheduler.schedule_ffn_with_m2n_immediate()

    assert result == dense_result
    assert scheduler._schedule_dense_ffn_from_m2n_group.call_count == 1
    assert scheduler._distribute_tokens_within_ep_replica.call_count == 0


def _builder_scheduler(model_config):
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=model_config,
            router_topk=1,
            total_expert_num=1,
            moe_expert_parallel_size=1,
            local_expert_num=1,
        )
    )
    scheduler._raw_batch_waiting_for_m2n_back = {}
    scheduler._batch_group_creation_counter = 0
    scheduler._ep_routed_token_allocation_cache = {}
    queue_sink = _QueuedBatchSink()
    scheduler._full_stage_replica_schedulers = {0: queue_sink}
    scheduler.get_full_stage_replica_scheduler = Mock(return_value=queue_sink)
    return scheduler, queue_sink


def test_ep_builder_uses_shared_layer_materializer(monkeypatch, mixed_model_config) -> None:
    """DECODE_FFN must consume one shared global-to-local EP materialization."""

    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=mixed_model_config,
            router_topk=2,
            moe_expert_parallel_size=2,
            local_expert_num=2,
            total_expert_num=4,
        )
    )
    scheduler._raw_batch_waiting_for_m2n_back = {}
    scheduler._batch_group_creation_counter = 0
    scheduler._ep_routed_token_allocation_cache = {}

    source_batch = _source_batch(layer_id=4)
    source_batch._num_tokens = [3]
    source_batch._total_num_tokens = 3
    transfer_info = _transfer_info(layer_id=4, afd_stage_idx=2)

    captured = {}

    def fake_materializer(**kwargs):
        captured.update(kwargs)
        return LayerEPWorkload(
            target_replica_id=0,
            global_layer_id=4,
            routing_token_count=3,
            router_topk=2,
            total_routed_assignments=6,
            global_per_expert_tokens={0: 1, 1: 1, 2: 2, 3: 2},
            per_ep_per_expert_tokens={
                0: {0: 1, 1: 1},
                1: {2: 2, 3: 2},
            },
            per_ep_routed_tokens={0: 2, 1: 4},
            participant_ep_ids=(0, 1),
            expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
        )

    import frontier.scheduler.cluster_scheduler.base_cluster_scheduler as base_scheduler

    monkeypatch.setattr(
        base_scheduler,
        "materialize_layer_ep_workload",
        fake_materializer,
        raising=False,
    )

    plan = scheduler._prepare_ep_batch_group_plan(
        [(source_batch, transfer_info)],
        replica_id=0,
        ep_id=1,
        expert_global_ids=[2, 3],
        layer_global_id=4,
        routing_details={0: {4: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}}},
    )

    assert captured == {
        "routing_ratios": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        "target_replica_id": 0,
        "global_layer_id": 4,
        "routing_token_count": 3,
        "router_topk": 2,
        "total_expert_num": 4,
        "moe_expert_parallel_size": 2,
        "expert_to_ep": {0: 0, 1: 0, 2: 1, 3: 1},
    }
    assert plan.per_expert_tokens == ((2, 2), (3, 2))


def test_ep_wave_schedule_materializes_one_shared_workload_for_all_lanes(
    monkeypatch,
    mixed_model_config,
) -> None:
    scheduler, _ = _builder_scheduler(mixed_model_config)
    source_batch = _source_batch(layer_id=4)
    source_batch._num_tokens = [3]
    source_batch._total_num_tokens = 3
    scheduler._m2n_ready_groups = deque(
        [[(source_batch, _transfer_info(layer_id=4, afd_stage_idx=2))]]
    )
    scheduler._predictor = SimpleNamespace(
        _decode_ffn_routing_details={
            0: {4: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}}
        }
    )
    scheduler._replica_ep_size = 2
    scheduler._config.replica_config.router_topk = 2
    scheduler._config.replica_config.total_expert_num = 4
    scheduler._config.replica_config.local_expert_num = 2
    scheduler._config.replica_config.moe_expert_parallel_size = 2

    queue_sinks = [_QueuedBatchSink(), _QueuedBatchSink()]
    scheduler.get_replica_scheduler = Mock(
        side_effect=lambda _replica_id, ep_id: queue_sinks[ep_id]
    )
    materializer_calls = []

    def fake_materializer(**kwargs):
        materializer_calls.append(kwargs)
        return LayerEPWorkload(
            target_replica_id=0,
            global_layer_id=4,
            routing_token_count=3,
            router_topk=2,
            total_routed_assignments=6,
            global_per_expert_tokens={0: 0, 1: 0, 2: 3, 3: 3},
            per_ep_per_expert_tokens={
                0: {0: 0, 1: 0},
                1: {2: 3, 3: 3},
            },
            per_ep_routed_tokens={0: 0, 1: 6},
            participant_ep_ids=(0, 1),
            expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
        )

    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler.materialize_layer_ep_workload",
        fake_materializer,
    )

    result = scheduler.schedule_ffn_with_m2n_immediate()

    assert result == [(0, 0), (0, 1)]
    assert len(materializer_calls) == 1
    assert [len(sink._m2n_immediate_batch_queue) for sink in queue_sinks] == [1, 1]


def test_dense_builder_propagates_decode_ffn_layer_metadata(
    mixed_model_config,
) -> None:
    scheduler, queue_sink = _builder_scheduler(mixed_model_config)
    source_batch = _source_batch(layer_id=3, afd_stage_idx=2)
    transfer_info = _transfer_info(layer_id=3, afd_stage_idx=2)
    ready_groups = deque([[(source_batch, transfer_info)]])

    result = scheduler._schedule_dense_ffn_from_m2n_group(
        ready_groups, Mock()
    )

    assert result == [(0, None)]
    assert len(queue_sink._m2n_immediate_batch_queue) == 1
    dense_batch = queue_sink._m2n_immediate_batch_queue[0]
    assert isinstance(dense_batch, DenseFFNBatchGroup)
    assert getattr(dense_batch, "decode_ffn_layer_id", None) == 3
    assert dense_batch.afd_stage_idx == 2


def test_dense_ffn_batch_uses_full_stage_parent_scope(
    mixed_model_config,
) -> None:
    scheduler, queue_sink = _builder_scheduler(mixed_model_config)
    context = StageExecutionContext(replica_id=0, stage_id=2, ep_size=2)
    scheduler.get_stage_execution_context = Mock(return_value=context)
    source_batch = _source_batch(layer_id=3, afd_stage_idx=2)

    scheduler._schedule_dense_ffn_from_m2n_group(
        deque([[(source_batch, _transfer_info(layer_id=3, afd_stage_idx=2))]]),
        Mock(),
    )

    dense_batch = queue_sink._m2n_immediate_batch_queue[0]
    ticket = dense_batch._stage_admission_ticket
    assert ticket.scope == FULL_STAGE_WORLD
    assert ticket.participant_ep_ids == ()
    assert context.queued_tickets == (ticket,)


def test_dense_builder_rejects_mismatched_afd_stage_idx(
    mixed_model_config,
) -> None:
    scheduler, _ = _builder_scheduler(mixed_model_config)
    group = [
        (_source_batch(layer_id=3, afd_stage_idx=1), _transfer_info(layer_id=3)),
        (_source_batch(layer_id=3, afd_stage_idx=2), _transfer_info(layer_id=3)),
    ]

    with pytest.raises(ValueError, match="afd_stage_idx mismatch"):
        scheduler._schedule_dense_ffn_from_m2n_group(deque([group]), Mock())


def test_moe_ready_group_validation_is_atomic_before_consumption(
    mixed_model_config,
) -> None:
    """Malformed MoE groups must not consume the ready queue or group id."""

    malformed_batch = _source_batch(layer_id=4)
    transfer_info = _transfer_info(layer_id=4)
    transfer_info.target_ffn_replica_id = 9
    group = [
        (
            malformed_batch,
            transfer_info,
        )
    ]
    ready_groups = deque([group])
    queue_sink = _QueuedBatchSink()
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_ready_groups = ready_groups
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=mixed_model_config,
            moe_expert_parallel_size=1,
            local_expert_num=1,
            total_expert_num=1,
            router_topk=1,
        )
    )
    scheduler._predictor = SimpleNamespace(
        _decode_ffn_routing_details={0: {4: {0: 1.0}}}
    )
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._replica_ep_size = 1
    scheduler._batch_group_creation_counter = 7
    scheduler._raw_batch_waiting_for_m2n_back = {}
    scheduler._full_stage_replica_schedulers = {0: queue_sink}
    scheduler.get_full_stage_replica_scheduler = Mock(return_value=queue_sink)

    with pytest.raises(ValueError, match="target replica is not available"):
        scheduler.schedule_ffn_with_m2n_immediate()

    assert tuple(ready_groups) == (group,)
    assert scheduler._batch_group_creation_counter == 7
    assert scheduler._raw_batch_waiting_for_m2n_back == {}
    assert queue_sink._m2n_immediate_batch_queue == []


def test_dense_ready_group_validation_is_atomic_before_consumption(
    mixed_model_config,
) -> None:
    """Malformed dense groups must not consume the ready queue or mutate state."""

    malformed_batch = _source_batch(layer_id=3)
    delattr(malformed_batch, "afd_stage_idx")
    group = [
        (
            malformed_batch,
            _transfer_info(layer_id=3),
        )
    ]
    ready_groups = deque([group])
    queue_sink = _QueuedBatchSink()
    scheduler, _ = _builder_scheduler(mixed_model_config)
    scheduler._batch_group_creation_counter = 7
    scheduler._m2n_ready_groups = ready_groups
    scheduler.get_replica_scheduler = Mock(return_value=queue_sink)

    with pytest.raises(ValueError, match="afd_stage_idx missing"):
        scheduler._schedule_dense_ffn_from_m2n_group(ready_groups, Mock())

    assert tuple(ready_groups) == (group,)
    assert scheduler._batch_group_creation_counter == 7
    assert scheduler._raw_batch_waiting_for_m2n_back == {}
    assert queue_sink._m2n_immediate_batch_queue == []


def _atomicity_scheduler(
    model_config,
    *,
    layer_id: int,
    ep_size: int = 1,
    queue_factory=None,
):
    """Build a minimally complete scheduler for DECODE_FFN atomicity tests."""
    scheduler = object.__new__(RoundRobinClusterScheduler)
    source_batch = _source_batch(layer_id=layer_id)
    transfer_info = _transfer_info(layer_id=layer_id)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_ready_groups = deque([[(source_batch, transfer_info)]])
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=model_config,
            router_topk=1,
            moe_expert_parallel_size=ep_size,
            local_expert_num=1,
            total_expert_num=ep_size,
        )
    )
    scheduler._predictor = SimpleNamespace(
        _decode_ffn_routing_details={
            0: {layer_id: {expert_id: 1.0 / ep_size for expert_id in range(ep_size)}}
        }
    )
    scheduler._cluster = SimpleNamespace(replicas={0: object()})
    scheduler._replica_ep_size = ep_size
    scheduler._batch_group_creation_counter = 0
    scheduler._raw_batch_waiting_for_m2n_back = {}
    scheduler._ep_routed_token_allocation_cache = {}

    if queue_factory is None:
        queue_factory = lambda _ep_id: _QueuedBatchSink()
    queue_sinks = {ep_id: queue_factory(ep_id) for ep_id in range(ep_size)}
    scheduler._full_stage_replica_schedulers = {0: queue_sinks[0]}
    scheduler.get_full_stage_replica_scheduler = Mock(
        return_value=queue_sinks[0]
    )
    scheduler.get_replica_scheduler = Mock(
        side_effect=lambda _replica_id, ep_id: queue_sinks[ep_id]
    )
    return scheduler, source_batch, transfer_info, queue_sinks


def test_decode_ffn_wave_materialization_attaches_one_parent_ticket(
    mixed_model_config,
) -> None:
    scheduler, _, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config,
        layer_id=4,
        ep_size=2,
    )
    context = StageExecutionContext(replica_id=0, stage_id=2, ep_size=2)
    scheduler.get_stage_execution_context = Mock(return_value=context)

    affected = scheduler.schedule_ffn_with_m2n_immediate()

    assert affected == [(0, 0), (0, 1)]
    lane_batches = [
        queue_sinks[ep_id]._m2n_immediate_batch_queue[0]
        for ep_id in (0, 1)
    ]
    tickets = [batch._stage_admission_ticket for batch in lane_batches]
    assert tickets[0] == tickets[1]
    assert tickets[0].scope == EP_WAVE
    assert tickets[0].participant_ep_ids == (0, 1)
    assert context.queued_tickets == (tickets[0],)


def _atomicity_snapshot(
    scheduler, source_batch, queue_sinks, *, include_entity_ids: bool = True
):
    cache = getattr(scheduler, "_ep_routed_token_allocation_cache", None)
    snapshot = {
        "ready_identity": id(scheduler._m2n_ready_groups),
        "ready_items": tuple(id(group) for group in scheduler._m2n_ready_groups),
        "raw_identity": id(scheduler._raw_batch_waiting_for_m2n_back),
        "raw_items": tuple(
            (key, id(value))
            for key, value in scheduler._raw_batch_waiting_for_m2n_back.items()
        ),
        "counter": (
            type(scheduler._batch_group_creation_counter),
            scheduler._batch_group_creation_counter,
        ),
        "cache_present": hasattr(scheduler, "_ep_routed_token_allocation_cache"),
        "cache_identity": id(cache) if cache is not None else None,
        "cache_items": repr(cache),
        "queues": tuple(
            (
                id(queue),
                tuple(
                    id(batch)
                    for batch in getattr(
                        queue,
                        "_m2n_immediate_batch_queue",
                        getattr(queue, "batches", []),
                    )
                ),
                getattr(queue, "_activation_bytes_allocated", None),
            )
            for queue in queue_sinks.values()
        ),
        "source_routing": getattr(source_batch, "_num_routing_tokens", None),
    }
    if include_entity_ids:
        snapshot.update(batch_id=Batch._id, request_id=Request._id)
    return snapshot


def _assert_atomicity_snapshot_unchanged(
    before, scheduler, source_batch, queue_sinks, *, include_entity_ids: bool = True
):
    assert before == _atomicity_snapshot(
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=include_entity_ids,
    )


def _capture_decode_ffn_exception(scheduler):
    try:
        scheduler.schedule_ffn_with_m2n_immediate()
    except Exception as exc:
        return exc
    return None


def _corrupt_decode_ffn_source_batch(
    case, scheduler, source_batch, transfer_info
):
    expected_message = {
        "source_type": "source batch",
        "requests_not_list": "requests",
        "request_not_request": "Request",
        "empty_source": "must not be empty",
        "num_tokens_not_list": "num_tokens",
        "num_tokens_bool": "num_tokens",
        "num_tokens_float": "num_tokens",
        "num_tokens_negative": "num_tokens",
        "request_token_length_mismatch": "length",
        "total_num_tokens_bool": "total_num_tokens",
        "total_num_tokens_negative": "total_num_tokens",
        "total_num_tokens_mismatch": "total_num_tokens",
    }[case]

    if case == "source_type":
        invalid_source = SimpleNamespace(
            id=source_batch.id,
            requests=source_batch.requests,
            num_tokens=source_batch.num_tokens,
            total_num_tokens=source_batch.total_num_tokens,
            afd_stage_idx=source_batch.afd_stage_idx,
            decode_attn_original_replica_id=0,
            decode_attn_original_replica_local_id=0,
            time=0.0,
            _num_routing_tokens=-1,
        )
        scheduler._m2n_ready_groups = deque(
            [[(invalid_source, transfer_info)]]
        )
        return invalid_source, expected_message
    if case == "requests_not_list":
        source_batch._requests = tuple(source_batch._requests)
    elif case == "request_not_request":
        source_batch._requests = [object()]
    elif case == "empty_source":
        source_batch._requests = []
        source_batch._num_tokens = []
        source_batch._total_num_tokens = 0
    elif case == "num_tokens_not_list":
        source_batch._num_tokens = tuple(source_batch._num_tokens)
    elif case == "num_tokens_bool":
        source_batch._num_tokens = [True]
        source_batch._total_num_tokens = 1
    elif case == "num_tokens_float":
        source_batch._num_tokens = [0.5]
        source_batch._total_num_tokens = 0.5
    elif case == "num_tokens_negative":
        source_batch._num_tokens = [-1]
        source_batch._total_num_tokens = -1
    elif case == "request_token_length_mismatch":
        source_batch._num_tokens = [1, 1]
        source_batch._total_num_tokens = 2
    elif case == "total_num_tokens_bool":
        source_batch._total_num_tokens = True
    elif case == "total_num_tokens_negative":
        source_batch._total_num_tokens = -1
    elif case == "total_num_tokens_mismatch":
        source_batch._total_num_tokens = 2
    else:
        raise AssertionError(f"Unhandled source corruption case: {case}")
    return source_batch, expected_message


@pytest.mark.parametrize("layer_id", [3, 4])
@pytest.mark.parametrize(
    "case",
    [
        "source_type",
        "requests_not_list",
        "request_not_request",
        "empty_source",
        "num_tokens_not_list",
        "num_tokens_bool",
        "num_tokens_float",
        "num_tokens_negative",
        "request_token_length_mismatch",
        "total_num_tokens_bool",
        "total_num_tokens_negative",
        "total_num_tokens_mismatch",
    ],
)
def test_decode_ffn_rejects_malformed_source_before_target_lookup_or_construction(
    mixed_model_config, layer_id, case
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    source_batch, expected_message = _corrupt_decode_ffn_source_batch(
        case,
        scheduler,
        source_batch,
        transfer_info,
    )
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    exc = _capture_decode_ffn_exception(scheduler)

    scheduler.get_replica_scheduler.assert_not_called()
    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(
        before, scheduler, source_batch, queue_sinks
    )
    assert type(exc) is ValueError
    assert expected_message in str(exc)


@pytest.mark.parametrize("layer_id", [3, 4])
@pytest.mark.parametrize(
    ("owner", "field_name", "bad_value", "expected_message"),
    [
        ("transfer", "layer_id", True, "layer_id"),
        ("transfer", "layer_id", 1.0, "layer_id"),
        ("transfer", "layer_id", -1, "layer_id"),
        ("transfer", "layer_id", "1", "layer_id"),
        ("transfer", "afd_stage_idx", True, "transfer afd_stage_idx"),
        ("transfer", "afd_stage_idx", 1.0, "transfer afd_stage_idx"),
        ("transfer", "afd_stage_idx", -1, "transfer afd_stage_idx"),
        ("transfer", "afd_stage_idx", "1", "transfer afd_stage_idx"),
        ("batch", "afd_stage_idx", True, "source batch afd_stage_idx"),
        ("batch", "afd_stage_idx", 1.0, "source batch afd_stage_idx"),
        ("batch", "afd_stage_idx", -1, "source batch afd_stage_idx"),
        ("batch", "afd_stage_idx", "1", "source batch afd_stage_idx"),
        (
            "transfer",
            "target_ffn_replica_id",
            True,
            "target_ffn_replica_id",
        ),
        (
            "transfer",
            "target_ffn_replica_id",
            0.0,
            "target_ffn_replica_id",
        ),
        (
            "transfer",
            "target_ffn_replica_id",
            -1,
            "target_ffn_replica_id",
        ),
        (
            "transfer",
            "target_ffn_replica_id",
            "0",
            "target_ffn_replica_id",
        ),
    ],
)
def test_decode_ffn_rejects_nonexact_metadata_before_target_lookup_or_construction(
    mixed_model_config,
    layer_id,
    owner,
    field_name,
    bad_value,
    expected_message,
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    target = transfer_info if owner == "transfer" else source_batch
    setattr(target, field_name, bad_value)
    if field_name == "target_ffn_replica_id" and bad_value is True:
        scheduler._cluster.replicas = {1: object()}
        scheduler._predictor._decode_ffn_routing_details = {
            1: {layer_id: {0: 1.0}}
        }
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    exc = _capture_decode_ffn_exception(scheduler)

    scheduler.get_replica_scheduler.assert_not_called()
    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(
        before, scheduler, source_batch, queue_sinks
    )
    assert type(exc) is ValueError
    assert expected_message in str(exc)


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_transfer_and_source_stage_mismatch_before_target_lookup(
    mixed_model_config, layer_id
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    transfer_info.afd_stage_idx = source_batch.afd_stage_idx + 1
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    exc = _capture_decode_ffn_exception(scheduler)

    scheduler.get_replica_scheduler.assert_not_called()
    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(
        before, scheduler, source_batch, queue_sinks
    )
    assert type(exc) is ValueError
    assert "afd_stage_idx mismatch" in str(exc)


@pytest.mark.parametrize("layer_id", [3, 4])
@pytest.mark.parametrize(
    ("corruption", "expected_message"),
    [
        ("request", "Request"),
        ("transfer_stage", "transfer afd_stage_idx"),
    ],
)
def test_decode_ffn_validates_later_group_entries_before_target_lookup_or_construction(
    mixed_model_config,
    layer_id,
    corruption,
    expected_message,
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    later_source = _source_batch(layer_id=layer_id)
    later_transfer = _transfer_info(layer_id=layer_id)
    if corruption == "request":
        later_source._requests = [object()]
    elif corruption == "transfer_stage":
        later_transfer.afd_stage_idx = True
    else:
        raise AssertionError(f"Unhandled later-entry corruption: {corruption}")
    scheduler._m2n_ready_groups = deque(
        [[(source_batch, transfer_info), (later_source, later_transfer)]]
    )
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    exc = _capture_decode_ffn_exception(scheduler)

    scheduler.get_replica_scheduler.assert_not_called()
    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(
        before, scheduler, source_batch, queue_sinks
    )
    assert type(exc) is ValueError
    assert expected_message in str(exc)


@pytest.mark.parametrize("bad_layer_id", [True, 1.0, -1, "1"])
def test_get_ffn_layer_id_rejects_nonexact_value_directly(bad_layer_id) -> None:
    group = [(object(), _transfer_info(layer_id=bad_layer_id))]

    with pytest.raises(ValueError, match="exact non-negative int"):
        RoundRobinClusterScheduler._get_ffn_layer_id_from_group(group)


@pytest.mark.parametrize("bad_counter", [True, 0.0, -1, "bad"])
@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_invalid_group_counter_before_construction(
    mixed_model_config, bad_counter, layer_id
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    scheduler._batch_group_creation_counter = bad_counter
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(ValueError, match="exact non-negative int"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_requires_exact_ready_group_deque_before_construction(
    mixed_model_config, container, layer_id
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    scheduler._m2n_ready_groups = container(scheduler._m2n_ready_groups)
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(RuntimeError, match="exact deque"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_raw_batch_collision_before_construction(
    mixed_model_config, layer_id
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    foreign_batch = object()
    scheduler._raw_batch_waiting_for_m2n_back[source_batch.id] = foreign_batch
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(ValueError, match="already registered"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


@pytest.mark.parametrize("bad_id", [True, 0.0, -1, "1"])
@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_nonexact_source_batch_id_before_construction(
    mixed_model_config, bad_id, layer_id
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    source_batch._id = bad_id
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(ValueError, match="source batch id"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_activation_size_before_construction(
    mixed_model_config, layer_id
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    transfer_info.activation_size_bytes = 0.5
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(ValueError, match="activation_size_bytes"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_duplicate_source_ids_before_construction(
    mixed_model_config, layer_id
) -> None:
    scheduler, source_batch, transfer_info, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=layer_id
    )
    duplicate_batch = _source_batch(layer_id=layer_id)
    duplicate_batch._id = source_batch.id
    scheduler._m2n_ready_groups = deque(
        [[(source_batch, transfer_info), (duplicate_batch, _transfer_info(layer_id=layer_id))]]
    )
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(ValueError, match="duplicate source batch IDs"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)
    scheduler.get_replica_scheduler.assert_not_called()


def test_decode_ffn_second_ep_construction_failure_does_not_commit_scheduler_state(
    mixed_model_config,
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=4, ep_size=2
    )
    original_create = scheduler._create_batch_group
    create_calls = 0

    def fail_on_second_creation(*args, **kwargs):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 2:
            raise RuntimeError("injected second EP construction failure")
        return original_create(*args, **kwargs)

    scheduler._create_batch_group = fail_on_second_creation
    before = _atomicity_snapshot(
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )

    with pytest.raises(RuntimeError, match="second EP construction"):
        scheduler.schedule_ffn_with_m2n_immediate()

    assert create_calls == 2
    _assert_atomicity_snapshot_unchanged(
        before,
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_commit_failure_stops_followup_state_after_queue_write(
    mixed_model_config,
    layer_id,
) -> None:
    """A queue-write commit failure must stop all later scheduler mutations."""

    scheduler, _, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config,
        layer_id=layer_id,
        queue_factory=lambda _ep_id: _ActivationCommitFailingSink(),
    )
    ready_group = scheduler._m2n_ready_groups[0]
    queue_sink = queue_sinks[0]

    with pytest.raises(RuntimeError, match="activation counter commit failure"):
        scheduler.schedule_ffn_with_m2n_immediate()

    assert tuple(scheduler._m2n_ready_groups) == (ready_group,)
    assert scheduler._raw_batch_waiting_for_m2n_back == {}
    assert scheduler._batch_group_creation_counter == 0
    assert len(queue_sink._m2n_immediate_batch_queue) == 1
    assert queue_sink._activation_bytes_allocated == 0


def test_decode_ffn_preflights_all_ep_target_queues_before_construction(
    mixed_model_config,
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config, layer_id=4, ep_size=2
    )
    lookup_calls = 0

    def fail_second_queue_lookup(_replica_id, ep_id):
        nonlocal lookup_calls
        lookup_calls += 1
        if ep_id == 1:
            raise KeyError("missing EP target queue")
        return queue_sinks[ep_id]

    scheduler.get_replica_scheduler = Mock(side_effect=fail_second_queue_lookup)
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(RuntimeError, match="target replica scheduler"):
        scheduler.schedule_ffn_with_m2n_immediate()

    assert lookup_calls == 2
    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)


@pytest.mark.parametrize("layer_id", [3, 4])
def test_decode_ffn_rejects_test_only_batch_list_queue_before_construction(
    mixed_model_config,
    layer_id,
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config,
        layer_id=layer_id,
        queue_factory=lambda _ep_id: SimpleNamespace(batches=[]),
    )
    create_group = Mock(wraps=scheduler._create_batch_group)
    scheduler._create_batch_group = create_group
    before = _atomicity_snapshot(scheduler, source_batch, queue_sinks)

    with pytest.raises(RuntimeError, match="exact immediate batch queue"):
        scheduler.schedule_ffn_with_m2n_immediate()

    create_group.assert_not_called()
    _assert_atomicity_snapshot_unchanged(before, scheduler, source_batch, queue_sinks)


def test_decode_ffn_moe_final_log_failure_occurs_before_commit(
    mixed_model_config,
    monkeypatch,
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config,
        layer_id=4,
    )
    logger = Mock()

    def fail_final_log(message):
        if "Affected EP lanes" in message:
            raise RuntimeError("injected MoE final log failure")

    logger.info.side_effect = fail_final_log
    monkeypatch.setattr(
        "frontier.logger.get_cluster_logger",
        lambda *_args, **_kwargs: logger,
    )
    before = _atomicity_snapshot(
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )

    with pytest.raises(RuntimeError, match="MoE final log failure"):
        scheduler.schedule_ffn_with_m2n_immediate()

    _assert_atomicity_snapshot_unchanged(
        before,
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )


def test_decode_ffn_dense_final_log_failure_occurs_before_commit(
    mixed_model_config,
) -> None:
    scheduler, source_batch, _, queue_sinks = _atomicity_scheduler(
        mixed_model_config,
        layer_id=3,
    )
    logger = Mock()

    def fail_final_log(message):
        if "[FFN-GROUP][DENSE]" in message:
            raise RuntimeError("injected dense final log failure")

    logger.info.side_effect = fail_final_log
    before = _atomicity_snapshot(
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )

    with pytest.raises(RuntimeError, match="dense final log failure"):
        scheduler._schedule_dense_ffn_from_m2n_group(
            scheduler._m2n_ready_groups,
            logger,
        )

    _assert_atomicity_snapshot_unchanged(
        before,
        scheduler,
        source_batch,
        queue_sinks,
        include_entity_ids=False,
    )


def test_ep_builder_propagates_decode_ffn_layer_metadata(
    mixed_model_config,
) -> None:
    scheduler, _ = _builder_scheduler(mixed_model_config)
    source_batch = _source_batch(layer_id=4, afd_stage_idx=2)
    routing_details = {0: {4: {0: 1.0}}}

    ep_batch = scheduler._distribute_tokens_within_ep_replica(
        [(source_batch, _transfer_info(layer_id=4, afd_stage_idx=2))],
        replica_id=0,
        ep_id=0,
        expert_global_ids=[0],
        layer_global_id=4,
        routing_details=routing_details,
    )

    assert getattr(ep_batch, "decode_ffn_layer_id", None) == 4
    assert ep_batch.afd_stage_idx == 2
    assert scheduler._raw_batch_waiting_for_m2n_back == {}


def _stage_scheduler(predictor: Mock) -> ReplicaStageScheduler:
    predictor._num_layers_per_pipeline_stage = 61
    predictor.predict_stage_execution_time.return_value = SimpleNamespace(
        total_time=1.0,
        model_time=0.8,
    )
    return ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=0,
    )


def test_replica_stage_scheduler_passes_decode_ffn_layer_id() -> None:
    predictor = Mock()
    scheduler = _stage_scheduler(predictor)
    batch = _source_batch(layer_id=4)
    batch.decode_ffn_layer_id = 4

    scheduler.predict_and_create_stage(batch)

    kwargs = predictor.predict_stage_execution_time.call_args.kwargs
    assert kwargs.get("layer_id") == 4


def test_replica_stage_scheduler_rejects_missing_decode_ffn_layer_id() -> None:
    predictor = Mock()
    scheduler = _stage_scheduler(predictor)
    batch = _source_batch(layer_id=4)

    with pytest.raises(ValueError, match="decode_ffn_layer_id"):
        scheduler.predict_and_create_stage(batch)


def test_replica_stage_scheduler_rejects_missing_layer_when_skipping_prediction() -> None:
    predictor = Mock()
    scheduler = _stage_scheduler(predictor)
    batch = _source_batch(layer_id=4)

    with pytest.raises(ValueError, match="decode_ffn_layer_id"):
        scheduler.predict_and_create_stage(batch, skip_get_execution_time=True)


def test_dense_ffn_batch_stage_tokens_are_post_routing() -> None:
    predictor = Mock()
    scheduler = _stage_scheduler(predictor)
    dense_batch = DenseFFNBatchGroup(
        requests=[_decode_request()],
        num_tokens=[1],
        replica_id=0,
        time=0.0,
        source_batch_ids=[1],
        cluster_type=ClusterType.DECODE_FFN,
    )
    dense_batch.decode_ffn_layer_id = 3

    batch_stage, _ = scheduler.predict_and_create_stage(dense_batch)

    assert batch_stage.tokens_are_post_routing is True


def test_post_routing_batch_stage_does_not_mutate_original_requests() -> None:
    """Synthetic FFN stages leave request accounting to the cluster return path."""
    request = SimpleNamespace(
        id=1,
        runtime_epoch=0,
        on_batch_stage_schedule=Mock(),
        on_batch_stage_end=Mock(),
    )
    batch_stage = BatchStage(
        batch_id=10,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.25,
        model_execution_time=0.10,
        requests=[request],
        num_tokens=[1],
        cluster_type=ClusterType.DECODE_FFN,
        tokens_are_post_routing=True,
    )

    batch_stage.on_schedule(1.0)
    batch_stage.on_stage_end(1.25)

    request.on_batch_stage_schedule.assert_not_called()
    request.on_batch_stage_end.assert_not_called()


def test_regular_batch_stage_accounts_requests_once() -> None:
    request = SimpleNamespace(
        id=2,
        runtime_epoch=0,
        is_prefill_complete=True,
        on_batch_stage_schedule=Mock(),
        on_batch_stage_end=Mock(),
    )
    batch_stage = BatchStage(
        batch_id=11,
        replica_id=0,
        pipeline_stage=0,
        execution_time=0.25,
        model_execution_time=0.10,
        requests=[request],
        num_tokens=[1],
        cluster_type=ClusterType.DECODE_ATTN,
        tokens_are_post_routing=False,
    )

    batch_stage.on_schedule(2.0)
    batch_stage.on_stage_end(2.25)

    request.on_batch_stage_schedule.assert_called_once_with(
        2.0, ClusterType.DECODE_ATTN
    )
    request.on_batch_stage_end.assert_called_once_with(
        2.25, 0.25, 0.10, ClusterType.DECODE_ATTN
    )


def _run_decode_ffn_stage_event(monkeypatch, batch: Batch):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "is_disaggregated_mode", lambda: True)

    batch_stage = SimpleNamespace(execution_time=0.01, on_schedule=Mock())
    execution_time = SimpleNamespace(
        get_single_layer_moe_pre_dispatch_time=lambda: 0.0,
        get_single_layer_moe_post_dispatch_compute_time=lambda: 1.0,
        expert_parallel_communication_time=0.0,
    )
    stage_scheduler = Mock()
    stage_scheduler.get_queue_batches.return_value = [batch]
    stage_scheduler.pop_batch_if_not_busy.return_value = batch
    stage_scheduler.consume_last_stale_drop_count.return_value = 0
    stage_scheduler.predict_and_create_stage.return_value = (
        batch_stage,
        execution_time,
    )
    stage_scheduler.is_last_stage = True
    stage_scheduler.is_busy = False

    cluster_scheduler = Mock()
    cluster_scheduler.get_replica_stage_scheduler.return_value = stage_scheduler
    cluster_scheduler.get_replica.return_value = SimpleNamespace(
        is_moe=True,
        dp_size=1,
        num_moe_expert_parallel_size=2,
    )
    global_scheduler = Mock()
    global_scheduler.get_cluster_scheduler.return_value = cluster_scheduler

    event = ReplicaStageScheduleEvent(
        time=1.0,
        replica_id=0,
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=0,
    )
    return event.handle_event(global_scheduler, Mock())


def test_mixed_dense_ffn_event_uses_direct_stage_lifecycle(monkeypatch) -> None:
    dense_batch = DenseFFNBatchGroup(
        requests=[_decode_request()],
        num_tokens=[1],
        replica_id=0,
        time=0.0,
        source_batch_ids=[1],
        cluster_type=ClusterType.DECODE_FFN,
    )
    dense_batch.set_global_id(10)
    dense_batch.decode_ffn_layer_id = 3

    events = _run_decode_ffn_stage_event(monkeypatch, dense_batch)

    assert len(events) == 1
    assert isinstance(events[0], BatchStageEndEvent)
    assert not any(
        isinstance(event, EPAllToAllDispatchReadyEvent) for event in events
    )


def test_mixed_moe_ffn_event_keeps_ep_dispatch_lifecycle(monkeypatch) -> None:
    ep_batch = EPBatchGroup(
        requests=[_decode_request()],
        num_tokens=[1],
        replica_id=0,
        ep_id=0,
        time=0.0,
        source_batch_ids=[1],
        per_expert_tokens={0: 1},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    ep_batch.set_global_id(10)
    ep_batch.decode_ffn_layer_id = 4

    events = _run_decode_ffn_stage_event(monkeypatch, ep_batch)

    assert len(events) == 1
    assert isinstance(events[0], EPAllToAllDispatchReadyEvent)


def test_mixed_moe_ffn_event_rejects_untyped_batch(monkeypatch) -> None:
    batch = _source_batch(layer_id=4)
    batch.set_global_id(10)
    batch.decode_ffn_layer_id = 4

    with pytest.raises(ValueError, match="requires EPBatchGroup"):
        _run_decode_ffn_stage_event(monkeypatch, batch)


def _trained_predictor(model_config, *, isolate_branch: bool = True):
    predictor = _ConcreteDisaggregationPredictor.__new__(
        _ConcreteDisaggregationPredictor
    )
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.DECODE_FFN
    predictor._replica_config = SimpleNamespace(
        model_config=model_config,
        total_expert_num=48,
        moe_expert_parallel_size=1,
        moe_tensor_parallel_size=1,
    )
    predictor._moe_ep_size = predictor._replica_config.moe_expert_parallel_size
    predictor._get_cluster_replica_config = lambda _cluster_type: predictor._replica_config
    predictor._get_cluster_model_architecture_profile = (
        lambda _cluster_type: model_config.get_model_architecture_profile()
    )
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda _batch, _cluster: False
    predictor._select_measurement_type_for_batch = lambda _batch: "decode"
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._get_communication_time = lambda *_args: _ZeroAttributes()
    predictor._get_overhead_time = lambda *_args: _ZeroAttributes()
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda *_args: 0.0
    predictor._get_add_layer_act_execution_time = lambda *_args: 0.0
    predictor._get_expert_parallel_communication_time = lambda *_args: 0.0
    predictor._predict_one_op_time = lambda _name, value, *_args: value

    def predict_moe(*_args, **_kwargs):
        return _ZeroAttributes()

    def predict_mlp(*_args, **_kwargs):
        return _ZeroAttributes()

    predictor.predict_moe_layer_time = predict_moe
    predictor.predict_mlp_layer_time = predict_mlp

    def zero_attention_params():
        params = {
            "attention_rope_execution_time": 0.0,
            "attention_kv_cache_save_execution_time": 0.0,
            "attention_decode_execution_time": 0.0,
            "attention_prefill_execution_time": 0.0,
            "attention_layer_pre_proj_execution_time": 0.0,
            "attention_layer_post_proj_execution_time": 0.0,
            "attn_norm_time": 0.0,
        }
        return params

    if isolate_branch:
        predictor._get_zero_attn_params = zero_attention_params
    return predictor


@pytest.mark.parametrize(
    ("layer_id", "expected_is_moe"),
    [(3, False), (4, True), (59, True), (60, False)],
)
def test_trained_decode_ffn_predictor_classifies_each_layer(
    mixed_model_config, layer_id: int, expected_is_moe: bool
) -> None:
    predictor = _trained_predictor(mixed_model_config)
    batch = SimpleNamespace(id=900 + layer_id)

    execution_time = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=layer_id,
    )

    assert execution_time._is_moe is expected_is_moe


def test_trained_dense_decode_ffn_constructs_execution_time_with_one_is_moe_source(
    dense_model_config,
) -> None:
    predictor = _trained_predictor(dense_model_config, isolate_branch=False)
    batch = SimpleNamespace(id=1000)

    try:
        execution_time = predictor.predict_stage_execution_time(
            batch,
            stage_id=0,
            cluster_type=ClusterType.DECODE_FFN,
            num_layers=1,
            layer_id=0,
        )
    except TypeError as exc:
        pytest.fail(
            "Dense DECODE_FFN must construct ExecutionTime with one is_moe source; "
            f"got {exc}"
        )

    assert execution_time._is_moe is False


def test_trained_dense_decode_ffn_excludes_post_attention_layernorm(
    dense_model_config,
) -> None:
    predictor = _trained_predictor(dense_model_config)
    predictor.predict_mlp_layer_time = lambda *_args, **_kwargs: SimpleNamespace(
        mlp_norm_time=0.75,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
    )

    execution_time = predictor.predict_stage_execution_time(
        SimpleNamespace(id=1001),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=0,
    )

    assert execution_time.mlp_norm_time == 0.0


def test_trained_moe_decode_ffn_excludes_post_attention_layernorm(
    generic_moe_model_config,
) -> None:
    predictor = _trained_predictor(generic_moe_model_config)
    predictor._get_mlp_norm_layer_act_execution_time = lambda *_args: 0.75

    execution_time = predictor.predict_stage_execution_time(
        SimpleNamespace(id=1002),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=0,
    )

    assert execution_time.mlp_norm_time == 0.0


@pytest.mark.xfail(
    strict=False,
    reason="D7 Option B: reference dummy predictor classifies mixed layers at model level",
)
def test_dummy_decode_ffn_predictor_reproducer_for_mixed_dense_layer(
    mixed_model_config,
) -> None:
    predictor = _ConcreteDisaggregationPredictor.__new__(
        _ConcreteDisaggregationPredictor
    )
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config = lambda _cluster_type: SimpleNamespace(
        model_config=mixed_model_config,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        num_pipeline_stages=1,
    )

    execution_time = predictor.predict_stage_execution_time(
        SimpleNamespace(),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=3,
    )

    assert execution_time._is_moe is False
