from __future__ import annotations

import pytest

from frontier.entities import Batch, EPBatchGroup, Request
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
    EP_WAVE,
    StageExecutionContext,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
)
from frontier.types import ClusterType


def test_ep_wave_owns_stage_before_dense_can_start() -> None:
    context = StageExecutionContext(replica_id=2, stage_id=4, ep_size=2)
    wave = context.enqueue_ep_wave(operation_id=11, participant_ep_ids=(0, 1))
    dense = context.enqueue_full_stage(operation_id=12)

    assert context.try_acquire(dense) is False
    assert context.try_acquire(wave) is True
    assert context.active_scope == EP_WAVE
    assert context.active_operation_id == 11

    context.release(wave)
    assert context.try_acquire(dense) is True
    assert context.active_scope == FULL_STAGE_WORLD
    context.release(dense)
    assert context.is_idle


def test_ep_wave_requires_complete_replica_local_participant_set() -> None:
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=3)

    with pytest.raises(ValueError, match="complete EP participant set"):
        context.enqueue_ep_wave(operation_id=1, participant_ep_ids=(0, 1))

    with pytest.raises(ValueError, match="complete EP participant set"):
        context.enqueue_ep_wave(operation_id=2, participant_ep_ids=(0, 1, 1))

    with pytest.raises(ValueError, match="complete EP participant set"):
        context.enqueue_ep_wave(operation_id=3, participant_ep_ids=(0, 1, 3))


def test_admission_fifo_cannot_skip_an_earlier_ready_wave() -> None:
    context = StageExecutionContext(replica_id=0, stage_id=1, ep_size=2)
    first = context.enqueue_ep_wave(operation_id=20, participant_ep_ids=(0, 1))
    second = context.enqueue_ep_wave(operation_id=21, participant_ep_ids=(0, 1))

    assert context.try_acquire(second) is False
    assert context.try_acquire(first) is True
    context.release(first)
    assert context.try_acquire(second) is True
    context.release(second)


def test_release_requires_the_active_operation_ticket() -> None:
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=1)
    wave = context.enqueue_ep_wave(operation_id=30, participant_ep_ids=(0,))
    context.try_acquire(wave)

    wrong = context.enqueue_full_stage(operation_id=31)
    with pytest.raises(ValueError, match="active operation"):
        context.release(wrong)

    context.release(wave)


def test_stage_contexts_are_isolated_by_stage_identity() -> None:
    stage_zero = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    stage_one = StageExecutionContext(replica_id=0, stage_id=1, ep_size=2)
    wave_zero = stage_zero.enqueue_ep_wave(operation_id=40, participant_ep_ids=(0, 1))
    dense_one = stage_one.enqueue_full_stage(operation_id=41)

    assert stage_zero.try_acquire(wave_zero) is True
    assert stage_one.try_acquire(dense_one) is True
    assert stage_zero.active_scope == EP_WAVE
    assert stage_one.active_scope == FULL_STAGE_WORLD


def test_duplicate_operation_id_is_rejected_until_previous_ticket_releases() -> None:
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=1)
    wave = context.enqueue_ep_wave(operation_id=50, participant_ep_ids=(0,))

    with pytest.raises(ValueError, match="operation_id"):
        context.enqueue_full_stage(operation_id=50)

    assert context.try_acquire(wave) is True
    with pytest.raises(ValueError, match="operation_id"):
        context.enqueue_ep_wave(operation_id=50, participant_ep_ids=(0,))


def test_child_stage_schedulers_share_parent_ep_wave_ownership() -> None:
    predictor = object()
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    lane_zero = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.DECODE_FFN,
        dp_id=0,
        stage_execution_context=context,
    )
    lane_one = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.DECODE_FFN,
        dp_id=1,
        stage_execution_context=context,
    )

    wave = context.enqueue_ep_wave(operation_id=60, participant_ep_ids=(0, 1))
    request_zero = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    request_one = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    batch_zero = Batch(0, [request_zero], [1], is_moe=True)
    batch_one = Batch(0, [request_one], [1], is_moe=True)
    batch_zero.set_global_id(60)
    batch_one.set_global_id(60)
    batch_zero._stage_admission_ticket = wave
    batch_one._stage_admission_ticket = wave
    lane_zero.add_batch(batch_zero)
    lane_one.add_batch(batch_one)

    assert lane_zero.pop_batch_if_not_busy() is batch_zero
    assert context.active_operation_id == 60
    assert lane_one.pop_batch_if_not_busy() is batch_one

    dense = context.enqueue_full_stage(operation_id=61)
    dense_batch = Batch(0, [request_zero], [1], is_moe=False)
    dense_batch.set_global_id(61)
    dense_batch._stage_admission_ticket = dense
    lane_zero.add_batch(dense_batch)
    assert lane_zero.pop_batch_if_not_busy() is None

    lane_zero.on_stage_end()
    lane_one.on_stage_end()
    context.release(wave)
    assert lane_zero.pop_batch_if_not_busy() is dense_batch


def test_cluster_scheduler_releases_parent_ticket_after_wave_cleanup() -> None:
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    ticket = context.enqueue_ep_wave(operation_id=70, participant_ep_ids=(0, 1))
    assert context.try_acquire(ticket) is True
    batch = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)],
        [1],
        is_moe=True,
    )
    batch.afd_stage_idx = 0
    batch._stage_admission_ticket = ticket

    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._stage_execution_contexts = {(0, 0): context}
    scheduler.release_stage_admission_for_batch(batch)

    assert context.is_idle
    assert not hasattr(batch, "_stage_admission_ticket")


def test_shared_domain_source_batch_gets_full_stage_ticket_before_queue_insert() -> None:
    predictor = object()
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    stage = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.PREFILL,
        dp_id=0,
        stage_execution_context=context,
    )
    request = Request(arrived_at=0.0, num_prefill_tokens=4, num_decode_tokens=0)
    batch = Batch(0, [request], [4], is_moe=True)
    batch.set_global_id(80)

    stage.add_batch(batch)

    ticket = getattr(batch, "_stage_admission_ticket", None)
    assert ticket is not None
    assert ticket.scope == FULL_STAGE_WORLD
    assert stage.pop_batch_if_not_busy() is batch
    assert context.active_operation_id == ("stage_batch", batch.id, batch.schedule_epoch)


def test_decode_ffn_ep_batch_without_wave_ticket_fails_fast() -> None:
    predictor = object()
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    stage = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.DECODE_FFN,
        dp_id=0,
        stage_execution_context=context,
    )
    request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    batch = EPBatchGroup(
        requests=[request],
        num_tokens=[1],
        replica_id=0,
        ep_id=0,
        time=0.0,
        source_batch_ids=[1],
        per_expert_tokens={0: 1},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    batch.set_global_id(81)

    with pytest.raises(ValueError, match="complete EP_WAVE admission ticket"):
        stage.add_batch(batch)
