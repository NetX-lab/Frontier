from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, EPBatchGroup, Request
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
    EP_WAVE,
    StageExecutionContext,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
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
        replica_local_id=0,
        stage_execution_context=context,
    )
    lane_one = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=1,
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


def test_stale_ep_wave_lane_drops_siblings_with_the_same_parent_ticket() -> None:
    """A stale lane must not leak an invalid parent ticket to a sibling lane."""

    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    ticket = context.enqueue_ep_wave(operation_id=71, participant_ep_ids=(0, 1))
    stale_request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    live_request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    stale_lane = Batch(0, [stale_request], [1], is_moe=True)
    live_lane = Batch(0, [live_request], [1], is_moe=True)
    stale_lane.set_global_id(71)
    live_lane.set_global_id(71)
    stale_lane._stage_admission_ticket = ticket
    live_lane._stage_admission_ticket = ticket

    stage = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=object(),
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=0,
        stage_execution_context=context,
    )
    stage.add_batch(stale_lane)
    stage.add_batch(live_lane)
    stale_request._execution_epoch += 1

    # A stale lane invalidates the whole EP wave, so its sibling is dropped
    # instead of carrying an admission ticket that has already been released.
    assert stage.pop_batch_if_not_busy() is None
    assert stage.consume_last_stale_drop_count() == 2
    assert context.is_idle
    assert context.queued_tickets == ()


def test_stale_schedule_epoch_drops_siblings_with_the_same_parent_ticket() -> None:
    """A stale queue snapshot must not leak its parent EP-wave ticket."""

    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    ticket = context.enqueue_ep_wave(operation_id=72, participant_ep_ids=(0, 1))
    stale_request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    live_request = Request(arrived_at=0.0, num_prefill_tokens=0, num_decode_tokens=1)
    stale_lane = Batch(0, [stale_request], [1], is_moe=True)
    live_lane = Batch(0, [live_request], [1], is_moe=True)
    stale_lane.set_global_id(72)
    live_lane.set_global_id(72)
    stale_lane._stage_admission_ticket = ticket
    live_lane._stage_admission_ticket = ticket

    stage = ReplicaStageScheduler(
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=object(),
        cluster_type=ClusterType.DECODE_FFN,
        replica_local_id=0,
        stage_execution_context=context,
    )
    stage.add_batch(stale_lane)
    stage.add_batch(live_lane)
    stale_lane._schedule_epoch += 1

    assert stage.pop_batch_if_not_busy() is None
    assert stage.consume_last_stale_drop_count() == 2
    assert context.is_idle
    assert context.queued_tickets == ()


def test_stale_stage_end_releases_the_captured_active_ticket() -> None:
    """A stale stage-end event must not leave its parent stage permanently busy."""

    class _ProbeClusterScheduler(BaseClusterScheduler):
        def schedule(self):
            return []

    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=1)
    ticket = context.enqueue_full_stage(operation_id=("stage_batch", 73, 0))
    assert context.try_acquire(ticket)
    stage_end_calls = []
    cluster_scheduler = object.__new__(_ProbeClusterScheduler)
    cluster_scheduler._stage_execution_contexts = {(0, 0): context}
    cluster_scheduler.get_replica_stage_scheduler = lambda *_args: SimpleNamespace(
        on_stage_end=lambda: stage_end_calls.append(True)
    )

    batch = Batch(
        0,
        [Request(arrived_at=0.0, num_prefill_tokens=1, num_decode_tokens=0)],
        [1],
        is_moe=False,
    )
    batch.set_global_id(73)
    batch._stage_admission_ticket = ticket
    batch_stage = SimpleNamespace(
        id=1,
        on_stage_end=lambda *_args: pytest.fail(
            "stale batch stage must not end"
        ),
    )
    event = BatchStageEndEvent(
        time=1.0,
        replica_id=0,
        stage_id=0,
        is_last_stage=True,
        batch=batch,
        batch_stage=batch_stage,
        cluster_type=ClusterType.PREFILL,
        replica_local_id=None,
    )
    batch._schedule_epoch += 1
    scheduler = SimpleNamespace(
        get_cluster_scheduler=lambda *_args: cluster_scheduler
    )
    metrics_store = SimpleNamespace(
        on_batch_stage_end=lambda *_args: pytest.fail(
            "stale batch stage must not write metrics"
        )
    )

    assert event.handle_event(scheduler, metrics_store) == []
    assert stage_end_calls == [True]
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
        replica_local_id=0,
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
        replica_local_id=0,
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


def test_active_stage_scope_can_transition_between_layer_operations() -> None:
    """A lockstep shared-domain batch must expose the current layer scope."""

    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=2)
    attention = context.enqueue_full_stage(operation_id=("batch", 1, "attention", 0))
    assert context.try_acquire(attention) is True

    moe = context.transition_active_scope(
        attention,
        operation_id=("batch", 1, "moe", 0),
        scope=EP_WAVE,
        participant_ep_ids=(0, 1),
    )
    assert moe.scope == EP_WAVE
    assert moe.admission_seq > attention.admission_seq
    assert context.active_scope == EP_WAVE
    assert context.active_operation_id == ("batch", 1, "moe", 0)

    dense = context.enqueue_full_stage(operation_id=("batch", 2, "dense", 0))
    assert context.try_acquire(dense) is False

    dense_layer = context.transition_active_scope(
        moe,
        operation_id=("batch", 1, "dense", 1),
        scope=FULL_STAGE_WORLD,
    )
    assert dense_layer.scope == FULL_STAGE_WORLD
    assert context.active_scope == FULL_STAGE_WORLD
    context.release(dense_layer)
    assert context.try_acquire(dense) is True
    context.release(dense)


def test_shared_layer_transition_distinguishes_attention_and_dense_ffn() -> None:
    """Consecutive dense layers need two distinct FULL_STAGE_WORLD operations."""

    class _ProbeScheduler(BaseClusterScheduler):
        def schedule(self):
            return []

    scheduler = object.__new__(_ProbeScheduler)
    context = StageExecutionContext(replica_id=0, stage_id=0, ep_size=1)
    scheduler._stage_execution_contexts = {(0, 0): context}
    batch = SimpleNamespace(id=0, schedule_epoch=1)
    ticket = context.enqueue_full_stage(operation_id=("stage_batch", 0, 1))
    assert context.try_acquire(ticket) is True
    batch._stage_admission_ticket = ticket

    scheduler.transition_stage_admission_for_layer(
        batch,
        stage_id=0,
        layer_id=0,
        operation_kind="ffn",
        scope=FULL_STAGE_WORLD,
    )
    scheduler.transition_stage_admission_for_layer(
        batch,
        stage_id=0,
        layer_id=1,
        operation_kind="attention",
        scope=FULL_STAGE_WORLD,
    )
    scheduler.transition_stage_admission_for_layer(
        batch,
        stage_id=0,
        layer_id=1,
        operation_kind="ffn",
        scope=FULL_STAGE_WORLD,
    )

    assert context.active_operation_id == (
        "shared_layer",
        0,
        1,
        0,
        1,
        "ffn",
        FULL_STAGE_WORLD,
    )


def test_shared_moe_stage_context_uses_replica_local_ep_size() -> None:
    class _ProbeScheduler(BaseClusterScheduler):
        def schedule(self):
            return []

    scheduler = object.__new__(_ProbeScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=SimpleNamespace(is_moe=True),
            moe_expert_parallel_size=4,
            num_pipeline_stages=1,
        )
    )
    scheduler._cluster = SimpleNamespace(
        replicas={0: SimpleNamespace(num_pipeline_stages=1)}
    )

    contexts = scheduler._build_stage_execution_contexts()

    assert contexts[(0, 0)].ep_size == 4
