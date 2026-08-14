from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.decode_sync_event import DecodeSyncEvent
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _ExecutionTime:
    expert_parallel_communication_time = 0.0
    pipeline_time = 0.0
    total_time = 0.0
    model_time = 0.0
    decode_draft_proposer_time = 0.0
    mtp_terminal_overshoot_time = 0.0

    def get_single_layer_attention_time(self) -> float:
        return 0.0


class _Predictor:
    def __init__(self, num_layers_per_pipeline_stage: int) -> None:
        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._execution_time = _ExecutionTime()

    def predict_stage_execution_time(self, *args, **kwargs):
        return self._execution_time


class _BatchStage:
    _next_id = 0

    def __init__(self) -> None:
        self.id = _BatchStage._next_id
        _BatchStage._next_id += 1
        self.scheduled_at = None
        self.execution_time = None
        self.model_execution_time = None

    def on_schedule(self, time: float) -> None:
        self.scheduled_at = time

    def override_execution_time(self, execution_time: float) -> None:
        self.execution_time = execution_time

    def override_model_execution_time(self, model_execution_time: float) -> None:
        self.model_execution_time = model_execution_time


class _StageScheduler:
    def __init__(self, predictor: _Predictor, *, is_last_stage: bool) -> None:
        self._execution_time_predictor = predictor
        self.is_last_stage = is_last_stage

    def predict_and_create_stage(self, batch, skip_get_execution_time=False):
        assert skip_get_execution_time is True
        return _BatchStage(), None


class _MetricsStore:
    def __init__(self) -> None:
        self.stage_schedule_calls = []

    def on_replica_stage_schedule(self, *args, **kwargs) -> None:
        self.stage_schedule_calls.append((args, kwargs))


class _DecodeSyncScheduler:
    def __init__(
        self,
        *,
        total_layers: int,
        num_layers_per_pipeline_stage: int,
        pipeline_parallel_size: int,
        cluster_type: ClusterType = ClusterType.DECODE,
        participant_count: int = 1,
        shared_domain_sync: bool = False,
    ) -> None:
        predictor = _Predictor(num_layers_per_pipeline_stage)
        self._cluster_type = cluster_type
        self._shared_domain_sync = shared_domain_sync
        self._config = SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    is_moe=True,
                    num_layers=total_layers,
                ),
                moe_expert_parallel_size=participant_count,
            )
        )
        self._replica = SimpleNamespace(id=1, dp_size=participant_count)
        self._stage_schedulers = {
            stage_id: _StageScheduler(
                predictor,
                is_last_stage=stage_id == pipeline_parallel_size - 1,
            )
            for stage_id in range(pipeline_parallel_size)
        }
        self._decode_sync_waiting_room = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(dict))
                )
            )
        )

    def add_post_moe_collective(
        self,
        *,
        stage_id: int,
        batch_global_id: int,
        layer_id: int,
        batches: dict[int, Batch],
    ) -> None:
        self._decode_sync_waiting_room[1][stage_id][batch_global_id][layer_id][
            "post_moe"
        ] = {
            "batches": batches,
            "arrival_times": {participant_id: 0.0 for participant_id in batches},
        }

    def get_replica_stage_scheduler(
        self,
        replica_id: int,
        dp_id: int,
        stage_id: int,
    ) -> _StageScheduler:
        assert replica_id == 1
        return self._stage_schedulers[stage_id]

    def get_replica(self, replica_id: int):
        assert replica_id == 1
        return self._replica

    def _get_decode_sync_participant_count(self, replica, batch) -> int:
        return self._replica.dp_size

    def _is_monolithic_decode_shared_domain_sync(self, batch: Batch) -> bool:
        return self._shared_domain_sync

    def _create_virtual_global_batch(
        self,
        sample_batch: Batch,
        total_global_tokens: int,
        total_global_prefill_tokens: int,
    ) -> Batch:
        return sample_batch

    def _record_mtp_terminal_completion_delay(
        self,
        batch: Batch,
        terminal_delay_s: float,
    ) -> None:
        assert terminal_delay_s == 0.0

    def _create_corrected_execution_time_for_metrics(
        self,
        original_execution_time,
        actual_execution_time_ms: float,
        original_start_time: float,
    ):
        return SimpleNamespace()

    def _accumulate_monolithic_decode_shared_domain_related_wait_ms(
        self,
        **kwargs,
    ) -> float:
        return 0.0

    def _pop_monolithic_decode_shared_domain_related_wait_ms(
        self,
        **kwargs,
    ) -> float:
        return 0.0

    def _build_monolithic_decode_shared_domain_trace_execution_time(
        self,
        execution_time,
        *,
        related_wait_ms: float,
    ):
        assert related_wait_ms == 0.0
        return execution_time


def _request(completed_layer_count: int = 0, *, completed: bool = False) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=1,
        num_decode_tokens=4,
        num_processed_tokens=1,
    )
    request._is_prefill_complete = True
    request._completed_layer_count = completed_layer_count
    request._completed = completed
    return request


def _batch(requests: list[Request], *, is_idle: bool = False) -> Batch:
    batch = Batch(
        replica_id=1,
        requests=requests,
        num_tokens=[1] * len(requests),
        is_idle=is_idle,
        is_moe=True,
    )
    batch.set_global_id(41)
    # This fixture represents a completed canonical EP_WAVE.  The collective
    # completion path no longer accepts the retired aggregate/DP scalar input.
    batch._decode_ep_wave_lane_times_ms = (0.0,)
    batch._decode_ep_wave_post_moe_comm_time_s = 0.0
    return batch


def _run_terminal_collective(
    scheduler: _DecodeSyncScheduler,
    metrics_store: _MetricsStore,
    *,
    stage_id: int,
    layer_id: int,
    batches: dict[int, Batch],
    batch_global_id: int = 41,
    time: float = 1.0,
):
    scheduler.add_post_moe_collective(
        stage_id=stage_id,
        batch_global_id=batch_global_id,
        layer_id=layer_id,
        batches=batches,
    )
    return BaseClusterScheduler.on_decode_sync_collective(
        scheduler,
        time=time,
        replica_id=1,
        stage_id=stage_id,
        batch_global_id=batch_global_id,
        sync_stage="post_moe",
        layer_id=layer_id,
        metrics_store=metrics_store,
    )


def test_pp2_terminal_collectives_account_all_94_layers() -> None:
    request = _request(completed_layer_count=46)
    batch = _batch([request])
    scheduler = _DecodeSyncScheduler(
        total_layers=94,
        num_layers_per_pipeline_stage=47,
        pipeline_parallel_size=2,
    )
    metrics_store = _MetricsStore()

    first_stage_events = _run_terminal_collective(
        scheduler,
        metrics_store,
        stage_id=0,
        layer_id=46,
        batches={0: batch},
        time=1.0,
    )
    for _ in range(46):
        request.mb_on_step_layer_count_increment()
    second_stage_events = _run_terminal_collective(
        scheduler,
        metrics_store,
        stage_id=1,
        layer_id=46,
        batches={0: batch},
        time=2.0,
    )

    assert len(first_stage_events) == 1
    assert len(second_stage_events) == 1
    assert all(
        isinstance(event, BatchStageEndEvent)
        for event in first_stage_events + second_stage_events
    )
    assert request.completed_layer_count == 94


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_monolithic_shared_domain_terminal_collectives_account_all_layers(
    pipeline_parallel_size: int,
) -> None:
    total_layers = 8
    layers_per_stage = total_layers // pipeline_parallel_size
    request = _request(completed_layer_count=layers_per_stage - 1)
    lane_zero_batch = _batch([request])
    lane_one_batch = _batch([request])
    scheduler = _DecodeSyncScheduler(
        total_layers=total_layers,
        num_layers_per_pipeline_stage=layers_per_stage,
        pipeline_parallel_size=pipeline_parallel_size,
        cluster_type=ClusterType.MONOLITHIC,
        participant_count=2,
        shared_domain_sync=True,
    )
    metrics_store = _MetricsStore()

    for stage_id in range(pipeline_parallel_size):
        if stage_id > 0:
            for _ in range(layers_per_stage - 1):
                request.mb_on_step_layer_count_increment()
        _run_terminal_collective(
            scheduler,
            metrics_store,
            stage_id=stage_id,
            layer_id=layers_per_stage - 1,
            batches={0: lane_zero_batch, 1: lane_one_batch},
            time=float(stage_id + 1),
        )

    assert request.completed_layer_count == total_layers


def test_terminal_collective_increments_duplicate_active_request_once() -> None:
    request = _request(completed_layer_count=3)
    first_batch = _batch([request])
    duplicate_lane_batch = _batch([request])
    idle_batch = _batch([], is_idle=True)
    scheduler = _DecodeSyncScheduler(
        total_layers=8,
        num_layers_per_pipeline_stage=4,
        pipeline_parallel_size=2,
        participant_count=3,
    )
    metrics_store = _MetricsStore()

    events = _run_terminal_collective(
        scheduler,
        metrics_store,
        stage_id=0,
        layer_id=3,
        batches={0: first_batch, 1: duplicate_lane_batch, 2: idle_batch},
    )

    assert request.completed_layer_count == 4
    assert len(events) == 2


def test_terminal_collective_skips_completed_request_in_mixed_batch() -> None:
    active_request = _request(completed_layer_count=3)
    completed_request = _request(completed_layer_count=7, completed=True)
    batch = _batch([active_request, completed_request])
    scheduler = _DecodeSyncScheduler(
        total_layers=8,
        num_layers_per_pipeline_stage=4,
        pipeline_parallel_size=2,
    )
    metrics_store = _MetricsStore()

    _run_terminal_collective(
        scheduler,
        metrics_store,
        stage_id=0,
        layer_id=3,
        batches={0: batch},
    )

    assert active_request.completed_layer_count == 4
    assert completed_request.completed_layer_count == 7


def test_replayed_terminal_collective_does_not_increment_again() -> None:
    request = _request(completed_layer_count=3)
    batch = _batch([request])
    scheduler = _DecodeSyncScheduler(
        total_layers=8,
        num_layers_per_pipeline_stage=4,
        pipeline_parallel_size=2,
    )
    metrics_store = _MetricsStore()
    scheduler.add_post_moe_collective(
        stage_id=0,
        batch_global_id=41,
        layer_id=3,
        batches={0: batch},
    )

    first_events = BaseClusterScheduler.on_decode_sync_collective(
        scheduler,
        time=1.0,
        replica_id=1,
        stage_id=0,
        batch_global_id=41,
        sync_stage="post_moe",
        layer_id=3,
        metrics_store=metrics_store,
    )
    replay_events = BaseClusterScheduler.on_decode_sync_collective(
        scheduler,
        time=1.0,
        replica_id=1,
        stage_id=0,
        batch_global_id=41,
        sync_stage="post_moe",
        layer_id=3,
        metrics_store=metrics_store,
    )

    assert len(first_events) == 1
    assert replay_events == []
    assert request.completed_layer_count == 4


@pytest.mark.parametrize("completed_layer_count", [8, 9])
def test_post_moe_collective_rejects_missing_prior_decode_step_reset(
    completed_layer_count: int,
) -> None:
    request = _request(completed_layer_count=completed_layer_count)
    batch = _batch([request])
    scheduler = _DecodeSyncScheduler(
        total_layers=8,
        num_layers_per_pipeline_stage=4,
        pipeline_parallel_size=2,
    )
    metrics_store = _MetricsStore()

    with pytest.raises(
        ValueError,
        match=(
            r"request_id=.*completed_layer_count=.*total_layers=.*"
            r"current_decode_token_index=.*spec_last_committed_tokens=.*"
            r"possible missing prior decode-step reset"
        ),
    ):
        _run_terminal_collective(
            scheduler,
            metrics_store,
            stage_id=0,
            layer_id=3,
            batches={0: batch},
        )


def test_stale_batch_stage_end_does_not_mutate_or_emit() -> None:
    request = _request(completed_layer_count=8)
    batch = _batch([request])
    batch_stage = SimpleNamespace(
        id=17,
        on_stage_end=lambda *_args, **_kwargs: pytest.fail(
            "stale batch stage must not end"
        ),
    )
    event = BatchStageEndEvent(
        time=1.0,
        replica_id=1,
        stage_id=1,
        is_last_stage=True,
        batch=batch,
        batch_stage=batch_stage,
        cluster_type=ClusterType.DECODE,
        dp_id=0,
    )
    batch._schedule_epoch += 1
    scheduler = SimpleNamespace(
        get_cluster_scheduler=lambda *_args, **_kwargs: pytest.fail(
            "stale batch stage must not look up its scheduler"
        )
    )
    metrics_store = SimpleNamespace(
        on_batch_stage_end=lambda *_args, **_kwargs: pytest.fail(
            "stale batch stage must not write metrics"
        )
    )

    assert event.handle_event(scheduler, metrics_store) == []
    assert request.completed_layer_count == 8


def test_non_terminal_collective_keeps_existing_increment_and_next_layer_event() -> None:
    request = _request(completed_layer_count=0)
    batch = _batch([request])
    scheduler = _DecodeSyncScheduler(
        total_layers=8,
        num_layers_per_pipeline_stage=4,
        pipeline_parallel_size=2,
    )
    metrics_store = _MetricsStore()
    scheduler.add_post_moe_collective(
        stage_id=0,
        batch_global_id=41,
        layer_id=0,
        batches={0: batch},
    )

    events = BaseClusterScheduler.on_decode_sync_collective(
        scheduler,
        time=1.0,
        replica_id=1,
        stage_id=0,
        batch_global_id=41,
        sync_stage="post_moe",
        layer_id=0,
        metrics_store=metrics_store,
    )

    assert request.completed_layer_count == 1
    assert len(events) == 1
    assert isinstance(events[0], DecodeSyncEvent)
    assert events[0].time == pytest.approx(1.0)
