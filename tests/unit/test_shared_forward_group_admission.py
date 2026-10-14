"""Regression coverage for unequal online lane histories and stage membership."""

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import ReplicaStageScheduler
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import StageExecutionContext
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


@pytest.mark.parametrize("first_lane", [0, 1])
def test_idle_lane_is_admitted_behind_a_busy_lane_queued_ticket(first_lane):
    """Under PP a busy lane can hold the FIFO head; the other lane must still join."""

    other_lane = 1 - first_lane
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2, full_stage_capacity=2)
    stages = [make_stage(context, lane, ClusterType.MONOLITHIC) for lane in range(2)]
    first_now, first_next = make_batch(first_lane, 0), make_batch(first_lane, 1)
    # Distinct provisional ids, so that sharing a bound group is observable.
    other_now, other_next = make_batch(other_lane, 5), make_batch(other_lane, 6)
    stages[first_lane].add_batch(first_now)
    assert stages[first_lane].pop_batch_if_not_busy() is first_now
    stages[first_lane].add_batch(first_next)
    stages[other_lane].add_batch(other_now)
    stages[other_lane].add_batch(other_next)
    assert context.queued_tickets[0] == first_next._stage_admission_ticket

    assert stages[other_lane].pop_batch_if_not_busy() is other_now
    assert other_now._forward_cohort_provisional_id == first_now._forward_cohort_provisional_id

    wave = context.replace_full_stage_owners_with_ep_wave(
        (first_now._stage_admission_ticket, other_now._stage_admission_ticket),
        operation_id="wave", participant_ep_ids=(0, 1),
    )
    owners = context.replace_ep_wave_with_full_stage_owners(wave, operation_ids=("restored0", "restored1"))
    for owner in owners:
        context.release(owner)
    for stage in stages:
        stage.on_stage_end()

    assert stages[first_lane].pop_batch_if_not_busy() is first_next
    assert stages[other_lane].pop_batch_if_not_busy() is other_next
    assert first_next._forward_cohort_provisional_id == other_next._forward_cohort_provisional_id
    assert first_next._forward_cohort_provisional_id > first_now._forward_cohort_provisional_id
    assert context.queued_tickets == ()


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


def test_a_new_lane_joins_the_bound_group_only_until_it_is_sealed():
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2, full_stage_capacity=2)
    assert context.joinable_forward_group_id == 0
    first = context.enqueue_full_stage(operation_id="first")
    assert context.try_acquire(first)
    group = context.bind_forward_group(first)
    assert context.joinable_forward_group_id == group
    wave = context.replace_full_stage_owners_with_ep_wave(
        (first,), operation_id="wave", participant_ep_ids=(0, 1),
    )
    assert context.joinable_forward_group_id == group + 1
    (owner,) = context.replace_ep_wave_with_full_stage_owners(wave, operation_ids=("next",))
    context.release(owner)
    assert context.is_idle
    assert context.joinable_forward_group_id == group + 1
    later = context.enqueue_full_stage(operation_id="later")
    assert context.try_acquire(later)
    assert context.bind_forward_group(later) == group + 1
