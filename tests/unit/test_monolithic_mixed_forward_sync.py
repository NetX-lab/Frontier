"""Regressions for shared MONOLITHIC forwards with local mixed batches."""

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import RoundRobinClusterScheduler
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils.sync_state import initialize_sync_waiting_rooms
from frontier.types import ClusterType


def make_batch(phase):
    requests, tokens = [], []
    if phase in ("decode", "mixed"):
        request = Request(0.0, 16, 8)
        request._is_prefill_complete = True
        request._num_processed_tokens = 17
        requests.append(request)
        tokens.append(1)
    if phase in ("prefill", "mixed"):
        requests.append(Request(0.0, 16, 8))
        tokens.append(16)
    batch = Batch(0, requests, tokens, is_moe=True)
    batch.set_global_id(17)
    batch._forward_cohort_provisional_id = 17
    return batch


@pytest.mark.parametrize("left", ["prefill", "decode", "mixed"])
@pytest.mark.parametrize("right", ["prefill", "decode", "mixed"])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_all_local_phases_join_one_wave(left, right, order):
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._config = SimpleNamespace(replica_config=SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True)))
    scheduler._forward_sync_state = ForwardSyncState()
    scheduler._replica_dp_size = 2
    initialize_sync_waiting_rooms(scheduler)
    scheduler._uses_shared_prefill_layer_path = lambda *_: True
    scheduler._uses_shared_decode_layer_path = lambda *_: True
    scheduler._replica_schedulers = {
        (0, lane): SimpleNamespace(get_replica_stage_scheduler=lambda _: SimpleNamespace(
            is_busy=True)) for lane in range(2)
    }
    waves = []
    scheduler._on_prefill_ep_wave_ready = lambda **kwargs: waves.append(kwargs) or []
    scheduler._on_decode_ep_wave_ready = scheduler._on_prefill_ep_wave_ready
    batches = [make_batch(left), make_batch(right)]
    for index, lane in enumerate(order):
        batch = batches[lane]
        enter = scheduler.on_prefill_sync if batch.num_prefill_tokens else scheduler.on_decode_sync
        enter(0.1 + index * 0.02, 0, 0, batch, lane, "pre_moe", 0, 0.0)
    assert len(waves) == 1
    assert waves[0]["time"] == pytest.approx(0.12)
    assert waves[0]["cohort_batches"] == dict(enumerate(batches))
    assert ForwardSyncState.get_step_id(batches[0]) == ForwardSyncState.get_step_id(batches[1])


@pytest.mark.parametrize("phases", [("decode", "prefill"), ("mixed", "prefill"), ("mixed", "decode"), ("mixed", "mixed")])
def test_completion_preserves_each_source_and_its_attention_time(phases):
    from frontier.scheduler.utils.forward_collective import complete_forward_collective

    batches = [make_batch(phase) for phase in phases]
    calls, restores = [], []

    def predict(batch, stage_id, cluster_type=None, *, layer_id=None, **kwargs):
        calls.append((batch, layer_id, tuple(batch.num_tokens),
                      tuple(r.num_processed_tokens for r in batch.requests)))
        milliseconds = 2.0 if batch is batches[0] else 7.0
        return SimpleNamespace(get_single_layer_attention_scope_time=lambda: milliseconds)

    predictor = SimpleNamespace(_num_layers_per_pipeline_stage=3,
                                predict_stage_execution_time=predict)
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._config = SimpleNamespace(replica_config=SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True, num_layers=3)))
    scheduler._predictor = predictor
    scheduler.get_replica_stage_scheduler = lambda *_: SimpleNamespace(
        _execution_time_predictor=predictor)
    scheduler._restore_forward_step_full_stage_owners = lambda **kw: restores.append(kw) or True
    initialize_sync_waiting_rooms(scheduler)
    for lane, batch in enumerate(batches):
        batch._stage_owner_replica_local_id = lane
        batch._prefill_model_execution_components_ms_by_stage = {0: []}
        batch._decode_model_execution_components_ms_by_stage = {0: []}
    sources = dict(enumerate(batches))
    scheduler._prefill_sync_waiting_room[0][0][17][0]["post_moe"]["batches"] = sources
    events = complete_forward_collective(scheduler, 1.0, 0, 0, 17, 0, None)
    assert len(restores) == 1
    assert restores[0]["source_batches"] == sources
    assert [e.time for e in events] == pytest.approx([1.002, 1.007])
    assert [e._batch for e in events] == batches
    assert [c[0] for c in calls if c[1] == 1] == batches
    assert all(c[2] == tuple(c[0].num_tokens) for c in calls)
    for batch in batches:
        assert all(r.completed_layer_count == int(r.is_prefill_complete) for r in batch.requests)
        assert [r.num_processed_tokens for r in batch.requests] == [17 if r.is_prefill_complete else 0 for r in batch.requests]
    assert not scheduler._prefill_sync_waiting_room[0][0]
    with pytest.raises(KeyError):
        complete_forward_collective(scheduler, 1.0, 0, 0, 17, 0, None)


def test_mixed_batch_finishes_requests_once_without_rewriting_old_ttft():
    batch = make_batch("mixed")
    decode, prefill = batch.requests
    decode._prefill_completed_at = 0.05
    decode._current_decode_token_index = 1
    batch.on_schedule(0.1, ClusterType.MONOLITHIC)
    batch.on_batch_end(0.2, ClusterType.MONOLITHIC)
    assert decode.num_processed_tokens == 18
    assert decode.prefill_completed_at == 0.05
    assert prefill.num_processed_tokens == 17
    assert prefill.prefill_completed_at == 0.2
    assert prefill.is_prefill_complete
    assert decode.completed_layer_count == prefill.completed_layer_count == 0


def test_ep_inputs_reject_cross_lane_request_duplication():
    from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs

    batch = make_batch("mixed")
    with pytest.raises(ValueError, match="two EP source lanes"):
        prepare_ep_wave_inputs(
            source_batches={0: batch, 1: batch}, batch=batch,
            step_id_getter=ForwardSyncState.get_step_id,
            aggregate_batch_builder=lambda *_: pytest.fail("duplicate input reached prediction"),
        )
