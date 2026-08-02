from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from frontier.config import global_vars
from frontier.entities import Request
from frontier.entities.batch import DenseFFNBatchGroup, EPBatchGroup
from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def _request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=2,
    )
    request._is_prefill_complete = True
    return request


def _batch(ep_id: int) -> EPBatchGroup:
    batch = EPBatchGroup(
        requests=[_request()],
        num_tokens=[1],
        replica_id=0,
        ep_id=ep_id,
        time=0.0,
        source_batch_ids=[1],
        per_expert_tokens={ep_id: 1},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    batch.set_global_id(10)
    batch.decode_ffn_layer_id = 4
    return batch


def _combine_batch(
    ep_id: int,
    *,
    source_batch_ids: list[int],
    execution_time: float = 0.01,
    activation_bytes: int = 100,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=100 + ep_id,
        source_batch_ids=list(source_batch_ids),
        per_expert_tokens={},
        execution_time=execution_time,
        activation_bytes=activation_bytes,
    )


def _raw_batch(batch_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=batch_id,
        requests=[],
        request_runtime_epochs=[],
        time=1.0,
    )


def _combine_scheduler(
    batches: dict[int, SimpleNamespace],
    *,
    raw_batches: dict[int, SimpleNamespace] | None = None,
):
    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    room = {"batches": batches}
    scheduler._ep_allgather_waiting_room = {0: {0: {10: room}}}
    scheduler._raw_batch_waiting_for_m2n_back = dict(raw_batches or {})

    stage_schedulers = {ep_id: Mock() for ep_id in batches}
    replica_schedulers = {
        ep_id: SimpleNamespace(
            decrement_num_running_batches=Mock(),
            release_activation_memory_bytes=Mock(),
            memory_usage_percent=25.0 + ep_id,
            num_running_batches=1,
        )
        for ep_id in batches
    }
    scheduler.get_dp_replica_stage_scheduler = Mock(
        side_effect=lambda _replica_id, ep_id, _stage_id: stage_schedulers[ep_id]
    )
    scheduler.get_dp_replica_scheduler = Mock(
        side_effect=lambda _replica_id, ep_id: replica_schedulers[ep_id]
    )
    scheduler._create_m2n_transfer_events_for_aggregated_batch = Mock(
        return_value=[]
    )
    return scheduler, room, stage_schedulers, replica_schedulers


def _run_ep_stage(
    monkeypatch: pytest.MonkeyPatch,
    batch: EPBatchGroup,
    *,
    stage_id: int,
    full_stage_time_s: float,
) -> None:
    monkeypatch.setattr(global_vars, "is_disaggregated_mode", lambda: True)
    monkeypatch.setattr(
        global_vars,
        "get_monolithic_moe_stage_aggregation",
        lambda: False,
    )
    batch_stage = SimpleNamespace(
        id=stage_id,
        execution_time=full_stage_time_s,
        on_schedule=Mock(),
    )
    execution_time = SimpleNamespace(
        get_single_layer_moe_pre_dispatch_time=lambda: 2.0,
        get_single_layer_moe_post_dispatch_compute_time=lambda: 1.0,
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
    cluster_scheduler.get_dp_replica_stage_scheduler.return_value = stage_scheduler
    cluster_scheduler.get_replica.return_value = SimpleNamespace(
        is_moe=True,
        dp_size=1,
        num_moe_expert_parallel_size=2,
    )
    global_scheduler = Mock()
    global_scheduler.get_cluster_scheduler.return_value = cluster_scheduler

    events = ReplicaStageScheduleEvent(
        time=1.0,
        replica_id=0,
        stage_id=stage_id,
        cluster_type=ClusterType.DECODE_FFN,
        dp_id=batch.ep_id,
    ).handle_event(global_scheduler, Mock())

    assert events[0].time == pytest.approx(1.002)


def test_ep_dispatch_preserves_full_stage_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = {ep_id: _batch(ep_id) for ep_id in range(2)}
    for batch in batches.values():
        _run_ep_stage(
            monkeypatch,
            batch,
            stage_id=0,
            full_stage_time_s=0.010,
        )

    scheduler = object.__new__(_ConcreteClusterScheduler)
    scheduler._ep_alltoall_dispatch_waiting_room = {
        0: {0: {10: {"batches": batches}}}
    }
    ready_events = scheduler.on_ep_alltoall_dispatch_collective_schedule(
        time=1.005,
        replica_id=0,
        stage_id=0,
        batch_global_id=10,
    )

    assert [event.time for event in ready_events] == pytest.approx([1.006, 1.006])
    for batch in batches.values():
        assert batch.execution_time == pytest.approx(0.010)


def test_ep_dispatch_collective_validates_all_lanes_before_mutation() -> None:
    batches = {ep_id: _batch(ep_id) for ep_id in range(2)}
    batches[0].expert_compute_time = 0.25
    batches[1].expert_compute_time = None
    for batch in batches.values():
        batch.time = 1.0

    scheduler = object.__new__(_ConcreteClusterScheduler)
    room = {"batches": batches}
    scheduler._ep_alltoall_dispatch_waiting_room = {0: {0: {10: room}}}

    with pytest.raises(ValueError, match="Missing expert_compute_time"):
        scheduler.on_ep_alltoall_dispatch_collective_schedule(
            time=2.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
        )

    assert scheduler._ep_alltoall_dispatch_waiting_room[0][0][10] is room
    assert [batch.time for batch in batches.values()] == [1.0, 1.0]


def test_ep_combine_collective_validates_source_ids_before_mutation() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10]),
        1: _combine_batch(1, source_batch_ids=[11]),
    }
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="source_batch_ids mismatch"):
        scheduler.on_ep_alltoall_combine_collective_schedule(
            time=5.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
            metrics_store=metrics_store,
        )

    assert scheduler._ep_allgather_waiting_room[0][0][10] is room
    for stage_scheduler in stage_schedulers.values():
        stage_scheduler.on_stage_end.assert_not_called()
    for replica_scheduler in replica_schedulers.values():
        replica_scheduler.decrement_num_running_batches.assert_not_called()
        replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    metrics_store.flush_frontier_stage_batch_ledger_row.assert_not_called()


def test_ep_combine_collective_reschedules_only_non_empty_stage_lanes() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10]),
        1: _combine_batch(1, source_batch_ids=[10]),
    }
    raw = _raw_batch(10)
    scheduler, room, stage_schedulers, _ = _combine_scheduler(
        batches,
        raw_batches={10: raw},
    )
    stage_schedulers[0].is_empty.return_value = False
    stage_schedulers[1].is_empty.return_value = True

    events = scheduler.on_ep_alltoall_combine_collective_schedule(
        time=5.0,
        replica_id=0,
        stage_id=0,
        batch_global_id=10,
        metrics_store=Mock(),
    )

    schedule_events = [
        event for event in events if isinstance(event, ReplicaStageScheduleEvent)
    ]
    assert [event._dp_id for event in schedule_events] == [0]


@pytest.mark.parametrize(
    ("source_batch_ids", "error_match"),
    [
        ([], "empty source_batch_ids"),
        ([10, 10], "duplicate source_batch_ids"),
    ],
    ids=["empty", "duplicate"],
)
def test_ep_combine_collective_rejects_invalid_source_id_cohort_before_lookup_or_mutation(
    source_batch_ids: list[int],
    error_match: str,
) -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=source_batch_ids),
        1: _combine_batch(1, source_batch_ids=source_batch_ids),
    }
    raw = _raw_batch(10)
    raw_request = SimpleNamespace(
        runtime_epoch=0,
        on_batch_stage_end=Mock(),
    )
    raw.requests = [raw_request]
    raw.request_runtime_epochs = [0]
    raw_inventory = {10: raw}
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches,
        raw_batches=raw_inventory,
    )
    raw_inventory_spy = Mock(wraps=raw_inventory)
    scheduler._raw_batch_waiting_for_m2n_back = raw_inventory_spy
    metrics_store = Mock()

    with pytest.raises(ValueError, match=error_match):
        try:
            scheduler.on_ep_alltoall_combine_collective_schedule(
                time=5.0,
                replica_id=0,
                stage_id=0,
                batch_global_id=10,
                metrics_store=metrics_store,
            )
        finally:
            assert scheduler._ep_allgather_waiting_room[0][0][10] is room
            assert raw_inventory == {10: raw}
            assert raw.time == 1.0
            raw_inventory_spy.get.assert_not_called()
            raw_inventory_spy.pop.assert_not_called()
            scheduler.get_dp_replica_stage_scheduler.assert_not_called()
            scheduler.get_dp_replica_scheduler.assert_not_called()
            scheduler._create_m2n_transfer_events_for_aggregated_batch.assert_not_called()
            raw_request.on_batch_stage_end.assert_not_called()
            for stage_scheduler in stage_schedulers.values():
                stage_scheduler.on_stage_end.assert_not_called()
            for replica_scheduler in replica_schedulers.values():
                replica_scheduler.decrement_num_running_batches.assert_not_called()
                replica_scheduler.release_activation_memory_bytes.assert_not_called()
            metrics_store.on_batch_end.assert_not_called()
            metrics_store.on_replica_schedule.assert_not_called()
            metrics_store.flush_frontier_stage_batch_ledger_row.assert_not_called()


def test_ep_combine_collective_validates_token_conservation_before_mutation() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10]),
        1: _combine_batch(1, source_batch_ids=[10]),
    }
    for ep_id, batch in batches.items():
        batch.total_num_tokens = 1
        batch.per_expert_tokens = {ep_id: 2}
    raw = _raw_batch(10)
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="Token conservation violated"):
        scheduler.on_ep_alltoall_combine_collective_schedule(
            time=5.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
            metrics_store=metrics_store,
        )

    assert scheduler._ep_allgather_waiting_room[0][0][10] is room
    assert scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    for stage_scheduler in stage_schedulers.values():
        stage_scheduler.on_stage_end.assert_not_called()
    for replica_scheduler in replica_schedulers.values():
        replica_scheduler.decrement_num_running_batches.assert_not_called()
        replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    metrics_store.flush_frontier_stage_batch_ledger_row.assert_not_called()
    scheduler._create_m2n_transfer_events_for_aggregated_batch.assert_not_called()


def test_ep_combine_collective_requires_execution_time_from_every_lane() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10], execution_time=0.01),
        1: _combine_batch(1, source_batch_ids=[10], execution_time=0.0),
    }
    batches[1].per_expert_tokens = {1: 1}
    batches[1].total_num_tokens = 1
    raw = _raw_batch(10)
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="execution_time"):
        scheduler.on_ep_alltoall_combine_collective_schedule(
            time=5.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
            metrics_store=metrics_store,
        )

    assert scheduler._ep_allgather_waiting_room[0][0][10] is room
    assert scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    for stage_scheduler in stage_schedulers.values():
        stage_scheduler.on_stage_end.assert_not_called()
    for replica_scheduler in replica_schedulers.values():
        replica_scheduler.decrement_num_running_batches.assert_not_called()
        replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    scheduler._create_m2n_transfer_events_for_aggregated_batch.assert_not_called()


def test_ep_combine_collective_validates_all_raw_batches_before_mutation() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10, 11]),
        1: _combine_batch(1, source_batch_ids=[10, 11]),
    }
    raw = _raw_batch(10)
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="Missing raw batch for id=11"):
        scheduler.on_ep_alltoall_combine_collective_schedule(
            time=5.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
            metrics_store=metrics_store,
        )

    assert scheduler._ep_allgather_waiting_room[0][0][10] is room
    assert scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    for stage_scheduler in stage_schedulers.values():
        stage_scheduler.on_stage_end.assert_not_called()
    for replica_scheduler in replica_schedulers.values():
        replica_scheduler.decrement_num_running_batches.assert_not_called()
        replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    metrics_store.flush_frontier_stage_batch_ledger_row.assert_not_called()
    scheduler._create_m2n_transfer_events_for_aggregated_batch.assert_not_called()


def test_ep_combine_collective_prepares_all_transfer_events_before_mutation() -> None:
    batches = {
        0: _combine_batch(0, source_batch_ids=[10, 11]),
        1: _combine_batch(1, source_batch_ids=[10, 11]),
    }
    raw_10 = _raw_batch(10)
    raw_11 = _raw_batch(11)
    raw_10.requests = [SimpleNamespace(runtime_epoch=0, on_batch_stage_end=Mock())]
    raw_11.requests = [SimpleNamespace(runtime_epoch=0, on_batch_stage_end=Mock())]
    raw_10.request_runtime_epochs = [0]
    raw_11.request_runtime_epochs = [0]
    scheduler, room, stage_schedulers, replica_schedulers = _combine_scheduler(
        batches,
        raw_batches={10: raw_10, 11: raw_11},
    )
    scheduler._create_m2n_transfer_events_for_aggregated_batch.side_effect = [
        [],
        ValueError("transfer prep rejected"),
    ]
    metrics_store = Mock()

    with pytest.raises(ValueError, match="transfer prep rejected"):
        scheduler.on_ep_alltoall_combine_collective_schedule(
            time=5.0,
            replica_id=0,
            stage_id=0,
            batch_global_id=10,
            metrics_store=metrics_store,
        )

    assert scheduler._ep_allgather_waiting_room[0][0][10] is room
    assert scheduler._raw_batch_waiting_for_m2n_back == {
        10: raw_10,
        11: raw_11,
    }
    assert [raw_10.time, raw_11.time] == [1.0, 1.0]
    raw_10.requests[0].on_batch_stage_end.assert_not_called()
    raw_11.requests[0].on_batch_stage_end.assert_not_called()
    for stage_scheduler in stage_schedulers.values():
        stage_scheduler.on_stage_end.assert_not_called()
    for replica_scheduler in replica_schedulers.values():
        replica_scheduler.decrement_num_running_batches.assert_not_called()
        replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    metrics_store.flush_frontier_stage_batch_ledger_row.assert_not_called()
    assert (
        scheduler._create_m2n_transfer_events_for_aggregated_batch.call_args_list
        == [call(raw_10, 5.0), call(raw_11, 5.0)]
    )


def test_ep_batch_records_each_pipeline_stage_once() -> None:
    batch = _batch(ep_id=0)

    batch.record_decode_ffn_stage_execution_time_once(0, 0.010)
    batch.record_decode_ffn_stage_execution_time_once(0, 0.010)
    batch.record_decode_ffn_stage_execution_time_once(1, 0.020)

    assert batch.execution_time == pytest.approx(0.030)


def _dense_ffn_completion_fixture(
    *,
    source_batch_ids: list[int],
    execution_time: float,
    raw_batches: dict[int, SimpleNamespace],
):
    batch = DenseFFNBatchGroup(
        requests=[_request()],
        num_tokens=[1],
        replica_id=0,
        lane_id=0,
        time=1.0,
        source_batch_ids=source_batch_ids,
        cluster_type=ClusterType.DECODE_FFN,
    )
    batch.execution_time = execution_time
    batch.activation_bytes = 100
    batch.on_cluster_stage_end = Mock(wraps=batch.on_cluster_stage_end)

    replica_scheduler = SimpleNamespace(
        on_cluster_stage_end=Mock(),
        decrement_num_running_batches=Mock(),
        release_activation_memory_bytes=Mock(),
        memory_usage_percent=25.0,
    )
    cluster_scheduler = SimpleNamespace(
        get_dp_replica_scheduler=Mock(return_value=replica_scheduler),
        _m2n_transfer_predictor=Mock(),
        _config=SimpleNamespace(replica_config=SimpleNamespace()),
        _raw_batch_waiting_for_m2n_back=dict(raw_batches),
    )
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=cluster_scheduler)
    )
    event = ClusterBatchEndEvent(
        time=5.0,
        replica_id=0,
        batch=batch,
        cluster_type=ClusterType.DECODE_FFN,
        dp_id=0,
    )
    return event, batch, cluster_scheduler, replica_scheduler, global_scheduler


def _event_raw_batch(
    batch_id: int,
    *,
    decode_attn_original_replica_id: int | None = 0,
) -> SimpleNamespace:
    request = SimpleNamespace(
        on_batch_stage_end=Mock(),
        completed=False,
        completed_layer_count=4,
    )
    return SimpleNamespace(
        id=batch_id,
        requests=[request],
        time=1.0,
        decode_attn_original_replica_id=decode_attn_original_replica_id,
        decode_attn_original_dp_id=0,
        afd_stage_idx=0,
    )


def test_decode_ffn_completion_validates_all_raw_batches_before_hooks() -> None:
    raw = _event_raw_batch(10)
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10, 11],
        execution_time=0.01,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="source_batch_id=11"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


def test_decode_ffn_completion_rejects_duplicate_source_ids_before_hooks() -> None:
    raw = _event_raw_batch(10)
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10, 10],
        execution_time=0.01,
        raw_batches={10: raw},
    )
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.return_value = (
        4096,
        2.5,
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="duplicate source_batch_ids"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    raw.requests[0].on_batch_stage_end.assert_not_called()
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()


def test_decode_ffn_completion_validates_execution_time_before_hooks() -> None:
    raw = _event_raw_batch(10)
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10],
        execution_time=0.0,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="Invalid DECODE_FFN execution_time"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


def test_decode_ffn_completion_validates_routing_before_all_mutation() -> None:
    raw = _event_raw_batch(10, decode_attn_original_replica_id=None)
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10],
        execution_time=0.01,
        raw_batches={10: raw},
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="missing decode_attn_original routing"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    raw.requests[0].on_batch_stage_end.assert_not_called()
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


def test_decode_ffn_completion_prepares_transfer_before_all_mutation() -> None:
    raw = _event_raw_batch(10)
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10],
        execution_time=0.01,
        raw_batches={10: raw},
    )
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.side_effect = (
        ValueError("predictor input rejected")
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="predictor input rejected"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    raw.requests[0].on_batch_stage_end.assert_not_called()
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.assert_called_once()


def test_decode_ffn_completion_constructs_transfer_event_before_all_mutation() -> None:
    raw = _event_raw_batch(10)
    raw.afd_stage_idx = None
    (
        event,
        batch,
        cluster_scheduler,
        replica_scheduler,
        global_scheduler,
    ) = _dense_ffn_completion_fixture(
        source_batch_ids=[10],
        execution_time=0.01,
        raw_batches={10: raw},
    )
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.return_value = (
        4096,
        2.5,
    )
    metrics_store = Mock()

    with pytest.raises(ValueError, match="afd_stage_idx must be set"):
        event.handle_event(global_scheduler, metrics_store)

    assert cluster_scheduler._raw_batch_waiting_for_m2n_back == {10: raw}
    assert raw.time == 1.0
    raw.requests[0].on_batch_stage_end.assert_not_called()
    batch.on_cluster_stage_end.assert_not_called()
    replica_scheduler.on_cluster_stage_end.assert_not_called()
    replica_scheduler.decrement_num_running_batches.assert_not_called()
    replica_scheduler.release_activation_memory_bytes.assert_not_called()
    metrics_store.on_batch_end.assert_not_called()
    metrics_store.on_replica_schedule.assert_not_called()
    cluster_scheduler._m2n_transfer_predictor.get_transfer_info.assert_called_once()
