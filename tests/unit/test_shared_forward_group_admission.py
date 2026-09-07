"""Regression coverage for unequal online lane histories and stage membership."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import RoundRobinClusterScheduler
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import ReplicaStageScheduler
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import EP_WAVE, StageExecutionContext
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.types import ClusterType


def make_stage(context, lane, cluster_type=ClusterType.DECODE):
    return ReplicaStageScheduler(
        replica_id=0, stage_id=0, is_last_stage=True, is_moe=True,
        execution_time_predictor=SimpleNamespace(), cluster_type=cluster_type,
        replica_local_id=lane, stage_execution_context=context,
    )


def make_batch(lane, counter):
    batch = Batch(0, [Request(0.0, 16, 4)], [1], is_moe=True)
    batch.set_global_id(counter * 2 + lane)
    batch._forward_cohort_id = counter
    batch._forward_cohort_provisional_id = counter
    batch._stage_owner_replica_local_id = lane
    return batch


@pytest.mark.parametrize("cluster_type", [ClusterType.PREFILL, ClusterType.DECODE, ClusterType.MONOLITHIC])
@pytest.mark.parametrize("first_lane", [0, 1])
def test_admitted_lanes_share_identity_after_unequal_batch_histories(cluster_type, first_lane):
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=8, full_stage_capacity=2)
    stages = [make_stage(context, lane, cluster_type) for lane in range(2)]
    prior = make_batch(0, 0)
    stages[0].add_batch(prior)
    assert stages[0].pop_batch_if_not_busy() is prior
    context.release(prior._stage_admission_ticket)
    stages[0].on_stage_end()

    batches = [make_batch(0, 1), make_batch(1, 0)]
    for lane in (first_lane, 1 - first_lane):
        stages[lane].add_batch(batches[lane])
        assert stages[lane].pop_batch_if_not_busy() is batches[lane]

    assert batches[0]._forward_cohort_provisional_id == batches[1]._forward_cohort_provisional_id
    assert batches[0]._forward_cohort_provisional_id != prior._forward_cohort_provisional_id
    assert [batch.global_id for batch in batches] == [2, 1]


def test_started_group_blocks_new_lane_through_ep_restore_and_partial_release():
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2, full_stage_capacity=4)
    first = context.enqueue_full_stage(operation_id="first")
    second = context.enqueue_full_stage(operation_id="second")
    assert context.try_acquire(first)
    group = context.bind_forward_group(first)
    assert context.try_acquire(second)
    assert context.bind_forward_group(second) == group
    wave = context.replace_full_stage_owners_with_ep_wave(
        (first, second), operation_id="wave", participant_ep_ids=(0, 1),
    )
    late = context.enqueue_full_stage(operation_id="late")
    assert not context.try_acquire(late)
    owners = context.replace_ep_wave_with_full_stage_owners(wave, operation_ids=("next0", "next1"))
    assert not context.try_acquire(late)
    context.release(owners[0])
    assert not context.try_acquire(late)
    context.release(owners[1])
    assert context.try_acquire(late)
    assert context.bind_forward_group(late) > group
    context.release(late)
    assert context.is_idle
    assert not context.forward_group_sealed
    assert context.queued_tickets == ()


@pytest.mark.parametrize("sync_kind", ["prefill", "decode"])
def test_next_group_queue_does_not_block_current_group_idle_participation(sync_kind):
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2, full_stage_capacity=2)
    stages = [make_stage(context, lane, getattr(ClusterType, sync_kind.upper())) for lane in range(2)]
    batch = make_batch(0, 0)
    stages[0].add_batch(batch)
    assert stages[0].pop_batch_if_not_busy() is batch
    wave = context.transition_active_scope(
        batch._stage_admission_ticket, operation_id="first_ffn", scope=EP_WAVE,
        participant_ep_ids=(0, 1),
    )
    batch._stage_admission_ticket, = context.replace_ep_wave_with_full_stage_owners(
        wave, operation_ids=("next_layer",),
    )
    stages[1].add_batch(make_batch(1, 0))
    assert stages[1].pop_batch_if_not_busy() is None

    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = getattr(ClusterType, sync_kind.upper())
    scheduler._forward_sync_state = ForwardSyncState()
    scheduler._stage_execution_contexts = {(0, 0): context}
    scheduler._replica_dp_size = 2
    rooms = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}}))
    )))
    setattr(scheduler, f"_{sync_kind}_sync_waiting_room", rooms)
    setattr(scheduler, f"_uses_shared_{sync_kind}_layer_path", lambda *_: True)
    scheduler._replica_schedulers = {
        (0, lane): SimpleNamespace(get_replica_stage_scheduler=lambda _, stage=stage: stage)
        for lane, stage in enumerate(stages)
    }
    completed = []
    setattr(scheduler, f"_on_{sync_kind}_ep_wave_ready", lambda **kwargs: completed.append(kwargs) or [])
    events = getattr(scheduler, f"on_{sync_kind}_sync")(
        1.0, 0, 0, batch, 0, "pre_moe", 1, 0.0,
    )
    assert len(events) == 1
    global_scheduler = SimpleNamespace(get_cluster_scheduler=lambda _: scheduler)
    assert events[0].handle_event(global_scheduler, None) == []
    assert len(completed) == 1
    assert completed[0]["cohort_batches"][1].is_idle
    assert not stages[1].is_empty()
    assert not stages[1].is_busy
    assert scheduler._forward_sync_state.open_steps(sync_kind) == {}
