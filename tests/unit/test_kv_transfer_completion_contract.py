#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class _Request:
    def __init__(self, request_id: int) -> None:
        self.id = request_id
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
    num_pending_requests = 0
    num_running_batches = 0

    def __init__(self) -> None:
        self.completed_request_ids: list[int] = []

    def complete_kv_transfer_for_requests(self, requests) -> None:
        self.completed_request_ids.extend(request.id for request in requests)


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
