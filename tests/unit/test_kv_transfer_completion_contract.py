#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent
from frontier.events.replica_schedule_event import ReplicaScheduleEvent
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (
    FasterTransformerReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.lightllm_replica_scheduler import (
    LightLLMReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class _Request:
    def __init__(
        self,
        request_id: int,
        *,
        completed: bool = False,
        is_prefill_complete: bool = False,
        num_decode_tokens: int = 0,
    ) -> None:
        self.id = request_id
        self.completed = completed
        self.is_prefill_complete = is_prefill_complete
        self.num_decode_tokens = num_decode_tokens
        self.transfer_events: list[tuple[float, float]] = []

    def on_kv_cache_transfer_complete(
        self, time: float, transfer_duration_s: float
    ) -> None:
        self.transfer_events.append((time, transfer_duration_s))


class _ConcreteBaseScheduler(BaseReplicaScheduler):
    def on_batch_end(self, batch) -> None:
        raise NotImplementedError

    def _get_next_batch(self):
        raise NotImplementedError


class _ContractSourceScheduler:
    memory_usage_percent = 12.5

    def __init__(
        self,
        *,
        num_pending_requests: int = 0,
        num_running_batches: int = 0,
        should_schedule_after_kv_transfer_completion: bool | None = None,
    ) -> None:
        self.completed_request_ids: list[int] = []
        self.num_pending_requests = num_pending_requests
        self.num_running_batches = num_running_batches
        self.should_schedule_after_kv_transfer_completion_calls = 0
        self._should_schedule_after_kv_transfer_completion = (
            should_schedule_after_kv_transfer_completion
        )

    def complete_kv_transfer_for_requests(self, requests) -> None:
        self.completed_request_ids.extend(request.id for request in requests)

    def should_schedule_after_kv_transfer_completion(self) -> bool:
        self.should_schedule_after_kv_transfer_completion_calls += 1
        if self._should_schedule_after_kv_transfer_completion is not None:
            return self._should_schedule_after_kv_transfer_completion
        return self.num_pending_requests > 0 and self.num_running_batches == 0


class _SourceClusterScheduler:
    def __init__(self, replica_scheduler) -> None:
        self.replica_scheduler = replica_scheduler

    def get_dp_replica_scheduler(self, replica_id: int, dp_id: int):
        assert replica_id == 3
        assert dp_id == 1
        return self.replica_scheduler


class _TargetClusterScheduler:
    def __init__(self) -> None:
        self.arrival_batches = []

    def on_kv_cache_arrival(self, time, batch, transfer_info):
        self.arrival_batches.append((time, batch, transfer_info))
        return ["arrival-event"]


class _GlobalScheduler:
    def __init__(self, source_cluster, target_cluster) -> None:
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster

    def get_cluster_scheduler(self, cluster_type: ClusterType):
        if cluster_type == ClusterType.PREFILL:
            return self.source_cluster
        if cluster_type == ClusterType.DECODE:
            return self.target_cluster
        raise AssertionError(f"Unexpected cluster_type={cluster_type}")

    def get_cluster_logical_time(self, cluster_type: ClusterType) -> float:
        assert cluster_type == ClusterType.PREFILL
        return 1.5


class _MetricsStore:
    def __init__(self) -> None:
        self.kv_transfer_end_calls = []
        self.replica_schedule_calls = []

    def on_kv_cache_transfer_end(self, *args, **kwargs) -> None:
        self.kv_transfer_end_calls.append((args, kwargs))

    def on_replica_schedule(self, *args, **kwargs) -> None:
        self.replica_schedule_calls.append((args, kwargs))


class _HookFailingBatch:
    id = 909
    schedule_epoch = 0
    request_execution_signatures = []
    request_mutation_signatures = []
    thinking_round_start_times = []

    def on_cluster_stage_end(self, time, cluster_type) -> None:
        raise RuntimeError("stage hook failed")


class _HookReplicaScheduler:
    def on_cluster_stage_end(self, batch) -> None:
        raise AssertionError("batch hook should fail before replica hook")


class _HookClusterScheduler:
    def get_dp_replica_scheduler(self, replica_id: int, dp_id: int):
        assert replica_id == 0
        assert dp_id == 0
        return _HookReplicaScheduler()


class _HookGlobalScheduler:
    def get_cluster_scheduler(self, cluster_type: ClusterType):
        assert cluster_type == ClusterType.DECODE_ATTN
        return _HookClusterScheduler()


def test_cluster_batch_end_fails_fast_when_stage_hook_fails() -> None:
    event = ClusterBatchEndEvent(
        time=1.0,
        replica_id=0,
        batch=_HookFailingBatch(),
        cluster_type=ClusterType.DECODE_ATTN,
        dp_id=0,
    )

    with pytest.raises(RuntimeError, match="stage hook failed"):
        event.handle_event(_HookGlobalScheduler(), _MetricsStore())


def test_kv_transfer_end_uses_public_scheduler_completion_contract() -> None:
    request = _Request(101)
    batch = SimpleNamespace(
        id=7,
        global_id=17,
        requests=[request],
        request_ids=[request.id],
    )
    transfer_info = SimpleNamespace(
        transfer_start_time=1.0,
        transfer_end_time=None,
        kv_cache_size_bytes=4096,
        target_cluster_type=ClusterType.DECODE,
        source_cluster_type=ClusterType.PREFILL,
        source_replica_id=3,
        source_dp_id=1,
        transfer_time_ms=2.0,
        batch=batch,
    )
    source_replica_scheduler = _ContractSourceScheduler()
    source_cluster_scheduler = _SourceClusterScheduler(source_replica_scheduler)
    target_cluster_scheduler = _TargetClusterScheduler()
    scheduler = _GlobalScheduler(source_cluster_scheduler, target_cluster_scheduler)
    metrics_store = _MetricsStore()

    event = KVCacheTransferEndEvent(1.25, transfer_info)
    arrival_events = event.handle_event(scheduler, metrics_store)

    assert arrival_events == ["arrival-event"]
    assert source_replica_scheduler.completed_request_ids == [101]
    assert request.transfer_events == [(1.25, 0.25)]
    assert len(metrics_store.kv_transfer_end_calls) == 1
    assert len(metrics_store.replica_schedule_calls) == 1


def test_kv_transfer_end_reschedules_source_when_pending_work_remains() -> None:
    request = _Request(111)
    batch = SimpleNamespace(
        id=8,
        global_id=18,
        requests=[request],
        request_ids=[request.id],
    )
    transfer_info = SimpleNamespace(
        transfer_start_time=1.0,
        transfer_end_time=None,
        kv_cache_size_bytes=4096,
        target_cluster_type=ClusterType.DECODE,
        source_cluster_type=ClusterType.PREFILL,
        source_replica_id=3,
        source_dp_id=1,
        transfer_time_ms=2.0,
        batch=batch,
    )
    source_replica_scheduler = _ContractSourceScheduler(
        num_pending_requests=2,
        num_running_batches=0,
    )
    source_cluster_scheduler = _SourceClusterScheduler(source_replica_scheduler)
    target_cluster_scheduler = _TargetClusterScheduler()
    scheduler = _GlobalScheduler(source_cluster_scheduler, target_cluster_scheduler)
    metrics_store = _MetricsStore()

    event = KVCacheTransferEndEvent(1.25, transfer_info)
    arrival_events = event.handle_event(scheduler, metrics_store)

    assert arrival_events[0] == "arrival-event"
    assert len(arrival_events) == 2
    reschedule_event = arrival_events[1]
    assert isinstance(reschedule_event, ReplicaScheduleEvent)
    assert reschedule_event.time == 1.5
    assert source_replica_scheduler.completed_request_ids == [request.id]
    assert (
        source_replica_scheduler.should_schedule_after_kv_transfer_completion_calls
        == 1
    )


def test_kv_transfer_end_uses_scheduler_reschedule_contract() -> None:
    request = _Request(112)
    batch = SimpleNamespace(
        id=11,
        global_id=21,
        requests=[request],
        request_ids=[request.id],
    )
    transfer_info = SimpleNamespace(
        transfer_start_time=1.0,
        transfer_end_time=None,
        kv_cache_size_bytes=4096,
        target_cluster_type=ClusterType.DECODE,
        source_cluster_type=ClusterType.PREFILL,
        source_replica_id=3,
        source_dp_id=1,
        transfer_time_ms=2.0,
        batch=batch,
    )
    source_replica_scheduler = _ContractSourceScheduler(
        num_pending_requests=0,
        num_running_batches=0,
        should_schedule_after_kv_transfer_completion=True,
    )
    source_cluster_scheduler = _SourceClusterScheduler(source_replica_scheduler)
    target_cluster_scheduler = _TargetClusterScheduler()
    scheduler = _GlobalScheduler(source_cluster_scheduler, target_cluster_scheduler)
    metrics_store = _MetricsStore()

    event = KVCacheTransferEndEvent(1.25, transfer_info)
    arrival_events = event.handle_event(scheduler, metrics_store)

    assert source_replica_scheduler.completed_request_ids == [request.id]
    assert (
        source_replica_scheduler.should_schedule_after_kv_transfer_completion_calls
        == 1
    )
    assert arrival_events[0] == "arrival-event"
    assert len(arrival_events) == 2
    assert isinstance(arrival_events[1], ReplicaScheduleEvent)


def test_base_scheduler_rejects_unsupported_kv_transfer_completion() -> None:
    scheduler = object.__new__(_ConcreteBaseScheduler)

    with pytest.raises(NotImplementedError, match="KV transfer completion"):
        scheduler.complete_kv_transfer_for_requests([])


@pytest.mark.parametrize(
    "scheduler_cls",
    [VLLMv1EngineReplicaScheduler, SarathiReplicaScheduler],
)
def test_replica_scheduler_completion_releases_allocation_and_pending_state(
    scheduler_cls,
) -> None:
    request = _Request(202)
    scheduler = object.__new__(scheduler_cls)
    scheduler._pending_kv_transfer_requests = {request.id}
    scheduler._allocation_map = {request.id: 2}
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._replica_id = 3
    scheduler._dp_id = 1
    freed_request_ids = []

    def _free_request_resources(freed_request) -> None:
        freed_request_ids.append(freed_request.id)
        scheduler._allocation_map.pop(freed_request.id, None)

    scheduler._free_request_resources = _free_request_resources

    scheduler.complete_kv_transfer_for_requests([request])

    assert freed_request_ids == [request.id]
    assert scheduler._allocation_map == {}
    assert scheduler._pending_kv_transfer_requests == set()


@pytest.mark.parametrize(
    "scheduler_cls",
    [VLLMv1EngineReplicaScheduler, SarathiReplicaScheduler],
)
def test_replica_scheduler_completion_requires_pending_transfer_state(
    scheduler_cls,
) -> None:
    request = _Request(303)
    scheduler = object.__new__(scheduler_cls)
    scheduler._pending_kv_transfer_requests = set()
    scheduler._allocation_map = {request.id: 1}
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._replica_id = 3
    scheduler._dp_id = 1

    with pytest.raises(ValueError, match="without pending transfer state"):
        scheduler.complete_kv_transfer_for_requests([request])


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        VLLMReplicaScheduler,
        LightLLMReplicaScheduler,
        FasterTransformerReplicaScheduler,
    ],
)
def test_legacy_scheduler_completion_releases_allocation_and_pending_state(
    scheduler_cls,
) -> None:
    request = _Request(404)
    scheduler = object.__new__(scheduler_cls)
    scheduler._pending_kv_transfer_requests = {request.id}
    scheduler._allocation_map = {request.id: 2}
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._replica_id = 3
    scheduler._dp_id = 1
    freed_request_ids = []

    def _free_request_resources(freed_request) -> None:
        freed_request_ids.append(freed_request.id)
        scheduler._allocation_map.pop(freed_request.id, None)

    scheduler._free_request_resources = _free_request_resources

    scheduler.complete_kv_transfer_for_requests([request])

    assert freed_request_ids == [request.id]
    assert scheduler._allocation_map == {}
    assert scheduler._pending_kv_transfer_requests == set()


@pytest.mark.parametrize(
    "scheduler_cls,preempted_attr",
    [
        (VLLMReplicaScheduler, "_preempted_requests"),
        (LightLLMReplicaScheduler, "_preempted_requests"),
        (FasterTransformerReplicaScheduler, "_preempted_batches"),
    ],
)
def test_legacy_scheduler_prefill_completion_waits_for_kv_transfer(
    scheduler_cls,
    preempted_attr,
) -> None:
    request = _Request(
        505,
        completed=False,
        is_prefill_complete=True,
        num_decode_tokens=2,
    )
    batch = SimpleNamespace(
        id=77,
        requests=[request],
        all_requests_completed=False,
    )
    scheduler = object.__new__(scheduler_cls)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._num_running_batches = 1
    scheduler._pending_kv_transfer_requests = set()
    setattr(scheduler, preempted_attr, [])

    if scheduler_cls is FasterTransformerReplicaScheduler:
        scheduler._pending_free_map = {}

    scheduler.on_batch_end(batch)

    assert scheduler._num_running_batches == 0
    assert scheduler._pending_kv_transfer_requests == {request.id}
    assert getattr(scheduler, preempted_attr) == []
    scheduler._request_queue = []
    assert scheduler.num_pending_requests == 0


def test_kv_transfer_end_does_not_reschedule_source_for_transfer_only_state() -> None:
    request = _Request(515)
    batch = SimpleNamespace(
        id=9,
        global_id=19,
        requests=[request],
        request_ids=[request.id],
    )
    transfer_info = SimpleNamespace(
        transfer_start_time=1.0,
        transfer_end_time=None,
        kv_cache_size_bytes=4096,
        target_cluster_type=ClusterType.DECODE,
        source_cluster_type=ClusterType.PREFILL,
        source_replica_id=3,
        source_dp_id=1,
        transfer_time_ms=2.0,
        batch=batch,
    )
    source_replica_scheduler = _ContractSourceScheduler(
        num_pending_requests=0,
        num_running_batches=0,
    )
    source_cluster_scheduler = _SourceClusterScheduler(source_replica_scheduler)
    target_cluster_scheduler = _TargetClusterScheduler()
    scheduler = _GlobalScheduler(source_cluster_scheduler, target_cluster_scheduler)
    metrics_store = _MetricsStore()

    event = KVCacheTransferEndEvent(1.25, transfer_info)
    arrival_events = event.handle_event(scheduler, metrics_store)

    assert arrival_events == ["arrival-event"]
    assert source_replica_scheduler.completed_request_ids == [request.id]


def test_faster_transformer_prefill_preempted_batch_skips_transferred_requests() -> None:
    transferred_request = _Request(
        606,
        completed=False,
        is_prefill_complete=True,
        num_decode_tokens=2,
    )
    unfinished_prefill = _Request(
        607,
        completed=False,
        is_prefill_complete=False,
        num_decode_tokens=2,
    )
    batch = SimpleNamespace(
        id=88,
        requests=[transferred_request, unfinished_prefill],
        all_requests_completed=False,
    )
    scheduler = object.__new__(FasterTransformerReplicaScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._pending_kv_transfer_requests = set()
    scheduler._replica_id = 0
    scheduler._replica_is_moe = False

    def _get_request_next_num_tokens(request) -> int:
        assert request is unfinished_prefill
        return 3

    scheduler._get_request_next_num_tokens = _get_request_next_num_tokens
    scheduler._create_batch = lambda requests, num_tokens: SimpleNamespace(
        requests=requests,
        request_ids=[request.id for request in requests],
        num_tokens=num_tokens,
    )

    next_batch = scheduler._generate_next_batch_from_preempted(batch)

    assert next_batch.requests == [unfinished_prefill]
    assert transferred_request.id not in next_batch.request_ids


@pytest.mark.parametrize(
    "scheduler_cls",
    [VLLMReplicaScheduler, LightLLMReplicaScheduler],
)
def test_request_preempting_legacy_scheduler_counts_preempted_pending_work(
    scheduler_cls,
) -> None:
    scheduler = object.__new__(scheduler_cls)
    scheduler._request_queue = [_Request(701)]
    scheduler._preempted_requests = [_Request(702), _Request(703)]

    assert scheduler.num_pending_requests == 3


def test_faster_transformer_counts_preempted_batch_pending_work() -> None:
    unfinished_prefill = _Request(
        801,
        completed=False,
        is_prefill_complete=False,
        num_decode_tokens=2,
    )
    transfer_bound_request = _Request(
        802,
        completed=False,
        is_prefill_complete=True,
        num_decode_tokens=2,
    )
    completed_request = _Request(
        803,
        completed=True,
        is_prefill_complete=True,
        num_decode_tokens=2,
    )
    preempted_batch = SimpleNamespace(
        requests=[unfinished_prefill, transfer_bound_request, completed_request]
    )
    scheduler = object.__new__(FasterTransformerReplicaScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._request_queue = [_Request(804)]
    scheduler._preempted_batches = [preempted_batch]
    scheduler._pending_kv_transfer_requests = {transfer_bound_request.id}

    assert scheduler.num_pending_requests == 2
