"""Fail-fast invariants for PD-AF cluster scheduler runtime state."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from frontier.entities import Batch, EPBatchGroup, Request
from frontier.entities.m2n_transfer_info import M2NTransferInfo
from frontier.events.cluster_schedule_event import ClusterScheduleEvent
from frontier.events.ep_alltoall_dispatch_collective_event import (
    EPAllToAllDispatchCollectiveEvent,
)
from frontier.events.m2n_transfer_end_event import M2NTransferEndEvent
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.model_architectures import MODEL_ARCHITECTURE_REGISTRY
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class _ConcreteClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def test_pdaf_source_replica_ordinal_maps_sticky_to_target_ffn_replica() -> None:
    target_ids = [4, 7, 9]

    assert BaseClusterScheduler._map_source_attn_replica_to_ffn_replica(
        0, target_ids
    ) == 4
    assert BaseClusterScheduler._map_source_attn_replica_to_ffn_replica(
        1, target_ids
    ) == 7
    assert BaseClusterScheduler._map_source_attn_replica_to_ffn_replica(
        2, target_ids
    ) == 9
    assert BaseClusterScheduler._map_source_attn_replica_to_ffn_replica(
        3, target_ids
    ) == 4


def test_pdaf_source_replica_mapping_allows_idle_target_ffn_replicas() -> None:
    target_ids = [4, 7, 9]

    assigned = {
        BaseClusterScheduler._map_source_attn_replica_to_ffn_replica(
            source_ordinal, target_ids
        )
        for source_ordinal in range(2)
    }

    assert assigned == {4, 7}
    assert 9 not in assigned


def _scheduler() -> _ConcreteClusterScheduler:
    return object.__new__(_ConcreteClusterScheduler)


def _transfer_info(**overrides) -> M2NTransferInfo:
    fields = {
        "batch": SimpleNamespace(id=12, global_id=77, requests=[]),
        "activation_size_bytes": 128,
        "source_cluster_type": ClusterType.DECODE_ATTN,
        "target_cluster_type": ClusterType.DECODE_FFN,
        "source_replica_id": 0,
        "source_replica_local_id": None,
        "transfer_time_ms": 500.0,
        "transfer_start_time": 0.5,
        "layer_id": 4,
        "afd_stage_idx": 1,
    }
    fields.update(overrides)
    return M2NTransferInfo(**fields)


def _real_pdaf_batch(
    *,
    replica_id: int,
    global_id: int,
    lane: tuple[int, int],
    layer_id: int,
    afd_stage_idx: int,
    barrier_round_id: int,
    expected_lanes=((0, None), (1, None)),
    is_moe: bool = True,
) -> Batch:
    """Build a real Batch with the minimum AFD identity metadata for barrier tests."""

    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=2,
    )
    request._is_prefill_complete = True
    request._current_decode_token_index = 2
    request._completed_layer_count = layer_id
    batch = Batch(
        replica_id=replica_id,
        requests=[request],
        num_tokens=[1],
        is_moe=is_moe,
    )
    batch.set_global_id(global_id)
    batch.afd_stage_idx = afd_stage_idx
    batch.decode_attn_original_replica_id = lane[0]
    batch.decode_attn_original_replica_local_id = lane[1]
    batch.decode_attn_barrier_round_id = barrier_round_id
    batch.decode_attn_barrier_expected_lanes = expected_lanes
    batch.decode_ffn_layer_id = layer_id
    return batch


def _room_snapshot(room):
    return {
        "batches": tuple(
            sorted((ep_id, id(batch)) for ep_id, batch in room["batches"].items())
        ),
        "arrival_times": tuple(sorted(room["arrival_times"].items())),
    }


def test_decode_ffn_arrival_hook_failure_propagates_before_scheduler_progress() -> None:
    arrival_error = RuntimeError("arrival hook failed")
    request = SimpleNamespace(on_arrival=Mock(side_effect=arrival_error))
    batch = SimpleNamespace(
        requests=[request],
        is_idle=False,
        decode_attn_barrier_expected_lanes=((0, None), (1, None)),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(batch=batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False

    waiting_state_before = dict(scheduler._m2n_waiting_by_layer)
    ready_state_before = tuple(scheduler._m2n_ready_groups)

    with pytest.raises(RuntimeError, match="arrival hook failed"):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                batch,
                transfer_info,
                Mock(),
            )
        finally:
            assert not hasattr(batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer == waiting_state_before
            assert tuple(scheduler._m2n_ready_groups) == ready_state_before


@pytest.mark.parametrize(
    (
        "invalid_case",
        "transfer_overrides",
        "batch_overrides",
        "expected_lanes",
        "error_match",
    ),
    [
        ("missing_layer", {"layer_id": None}, {}, ((0, None),), "layer_id"),
        ("bool_layer", {"layer_id": True}, {}, ((0, None),), "layer_id"),
        (
            "missing_stage",
            {"afd_stage_idx": None},
            {},
            ((0, None),),
            "afd_stage_idx",
        ),
        (
            "bool_stage",
            {"afd_stage_idx": True},
            {},
            ((0, None),),
            "afd_stage_idx",
        ),
        (
            "bool_round",
            {},
            {"decode_attn_barrier_round_id": True},
            ((0, None),),
            "barrier_round_id",
        ),
        (
            "negative_round",
            {},
            {"decode_attn_barrier_round_id": -1},
            ((0, None),),
            "barrier_round_id",
        ),
        (
            "bool_source_replica",
            {"source_replica_id": True},
            {},
            ((1, None),),
            "source_replica_id|lane",
        ),
        (
            "bool_source_dp",
            {"source_replica_local_id": True},
            {},
            ((0, 1),),
            "source_replica_local_id|lane",
        ),
        (
            "unexpected_lane",
            {"source_replica_id": 9},
            {},
            ((0, None),),
            "lane",
        ),
    ],
)
def test_decode_ffn_rejects_malformed_receipt_before_lifecycle_side_effects(
    invalid_case: str,
    transfer_overrides: dict,
    batch_overrides: dict,
    expected_lanes: tuple[tuple[int, int], ...],
    error_match: str,
) -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch_fields = {
        "id": 12,
        "global_id": 77,
        "requests": [request],
        "is_idle": False,
        "afd_stage_metadata": None,
        "decode_attn_barrier_expected_lanes": expected_lanes,
        "decode_attn_barrier_round_id": 7,
    }
    batch_fields.update(batch_overrides)
    batch = SimpleNamespace(**batch_fields)
    transfer_info = _transfer_info(batch=batch, **transfer_overrides)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    logger = Mock()

    with pytest.raises((TypeError, ValueError), match=error_match):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                batch,
                transfer_info,
                logger,
            )
        finally:
            request.on_arrival.assert_not_called()
            assert not hasattr(batch, "decode_ffn_m2n_arrival_time"), invalid_case
            assert scheduler._m2n_waiting_by_layer == {}
            assert tuple(scheduler._m2n_ready_groups) == ()
            logger.info.assert_not_called()


def test_decode_ffn_accepts_exact_receipt_metadata() -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(batch=batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    logger = Mock()

    events = scheduler._handle_m2n_arrival_decode_ffn(
        1.25,
        batch,
        transfer_info,
        logger,
    )

    request.on_arrival.assert_called_once_with(1.25, ClusterType.DECODE_FFN)
    assert batch.decode_ffn_m2n_arrival_time == pytest.approx(1.25)
    assert scheduler._m2n_waiting_by_layer == {}
    assert list(scheduler._m2n_ready_groups) == [[(batch, transfer_info)]]
    assert len(events) == 1
    assert events[0].time == pytest.approx(1.25)


def test_decode_ffn_direct_arrival_rejects_mismatched_batch_identity_before_side_effects() -> None:
    passed_request = SimpleNamespace(id=34, on_arrival=Mock())
    passed_batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[passed_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    transfer_request = SimpleNamespace(id=35, on_arrival=Mock())
    transfer_batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[transfer_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    assert passed_batch is not transfer_batch
    assert passed_batch.id == transfer_batch.id
    assert passed_batch.global_id == transfer_batch.global_id
    transfer_info = _transfer_info(batch=transfer_batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False

    with pytest.raises(ValueError, match="batch.*transfer_info.batch|identity"):
        try:
            scheduler.on_m2n_arrival(1.25, passed_batch, transfer_info)
        finally:
            passed_request.on_arrival.assert_not_called()
            transfer_request.on_arrival.assert_not_called()
            assert not hasattr(passed_batch, "decode_ffn_m2n_arrival_time")
            assert not hasattr(transfer_batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer == {}
            assert tuple(scheduler._m2n_ready_groups) == ()


def test_decode_ffn_private_handler_rejects_wrong_direction_before_side_effects() -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
    )
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    logger = Mock()

    with pytest.raises(ValueError, match="target scheduler mismatch|DECODE_FFN"):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                batch,
                transfer_info,
                logger,
            )
        finally:
            request.on_arrival.assert_not_called()
            assert not hasattr(batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer == {}
            assert tuple(scheduler._m2n_ready_groups) == ()
            logger.info.assert_not_called()


@pytest.mark.parametrize(
    "cluster_type",
    [
        cluster_type
        for cluster_type in ClusterType
        if cluster_type not in {ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN}
    ],
)
def test_m2n_arrival_rejects_unsupported_cluster_target(
    cluster_type: ClusterType,
) -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = cluster_type
    batch = SimpleNamespace(
        id=12,
        requests=[SimpleNamespace(id=34)],
    )

    with pytest.raises(ValueError, match="M2N arrival is unsupported for cluster"):
        scheduler.on_m2n_arrival(1.0, batch, _transfer_info(batch=batch))


@pytest.mark.parametrize(
    (
        "scheduler_cluster_type",
        "transfer_source",
        "transfer_target",
        "handler_name",
    ),
    [
        (
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            "_handle_m2n_arrival_decode_ffn",
        ),
        (
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            "_handle_m2n_arrival_decode_attn",
        ),
    ],
)
def test_m2n_arrival_rejects_transfer_target_scheduler_mismatch_before_handler(
    scheduler_cluster_type: ClusterType,
    transfer_source: ClusterType,
    transfer_target: ClusterType,
    handler_name: str,
) -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = scheduler_cluster_type
    scheduler._m2n_waiting_by_layer = {"sentinel": object()}
    scheduler._m2n_ready_groups = deque(["sentinel"])
    handler = Mock(return_value=[])
    setattr(scheduler, handler_name, handler)
    batch = SimpleNamespace(
        id=12,
        requests=[SimpleNamespace(id=34)],
    )
    transfer_info = _transfer_info(
        batch=batch,
        source_cluster_type=transfer_source,
        target_cluster_type=transfer_target,
    )
    waiting_before = dict(scheduler._m2n_waiting_by_layer)
    ready_before = tuple(scheduler._m2n_ready_groups)

    with pytest.raises(ValueError, match="M2N target scheduler mismatch"):
        try:
            scheduler.on_m2n_arrival(1.0, batch, transfer_info)
        finally:
            handler.assert_not_called()
            assert scheduler._m2n_waiting_by_layer == waiting_before
            assert tuple(scheduler._m2n_ready_groups) == ready_before


@pytest.mark.parametrize(
    ("scheduler_cluster_type", "transfer_source", "handler_name"),
    [
        (
            ClusterType.DECODE_FFN,
            ClusterType.DECODE_ATTN,
            "_handle_m2n_arrival_decode_ffn",
        ),
        (
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
            "_handle_m2n_arrival_decode_attn",
        ),
    ],
)
def test_m2n_arrival_routes_valid_target_to_matching_handler(
    scheduler_cluster_type: ClusterType,
    transfer_source: ClusterType,
    handler_name: str,
) -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = scheduler_cluster_type
    expected_events = [object()]
    handler = Mock(return_value=expected_events)
    setattr(scheduler, handler_name, handler)
    if scheduler_cluster_type == ClusterType.DECODE_ATTN:
        request = Request(0.0, 0, 2)
        request._completed_layer_count = 4
        request._current_decode_token_index = 2
    else:
        request = SimpleNamespace(
            id=34,
            completed=False,
            completed_layer_count=4,
            current_decode_token_index=2,
            af_roundtrip_inflight=False,
        )
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        afd_stage_idx=1,
        replay_decode_token_index=2,
        decode_attn_original_replica_id=0,
        decode_attn_original_replica_local_id=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
        decode_attn_cohort_id=9,
        decode_attn_cohort_request_ids=(request.id,),
    )
    transfer_info = _transfer_info(
        batch=batch,
        source_cluster_type=transfer_source,
        target_cluster_type=scheduler_cluster_type,
    )
    if scheduler_cluster_type == ClusterType.DECODE_FFN:
        scheduler._ffn_expected_lanes = [(0, None)]
        scheduler._ffn_group_micro_batches = 1
        scheduler._m2n_waiting_by_layer = {}
    else:
        scheduler._config = SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=SimpleNamespace(num_layers=8),
            ),
        )
        scheduler._f2a_expected_lanes = [(0, None)]
        scheduler._decode_attn_idle_expected_lanes = set()
        scheduler._replica_scheduler_count = 1
        scheduler._f2a_waiting_by_round = {}
        scheduler._cluster = SimpleNamespace(replicas={0: SimpleNamespace()})
        scheduler._replica_schedulers = {
            (0, None): SimpleNamespace(
                _decode_attn_active_cohort_states={
                    9: {
                        "all_request_ids": {request.id},
                        "pending_request_ids": {request.id},
                        "active_stage_indices": {1},
                    },
                },
            ),
        }

    actual_events = scheduler.on_m2n_arrival(1.0, batch, transfer_info)

    assert actual_events is expected_events
    handler.assert_called_once()


@pytest.mark.parametrize(
    "invalid_cluster_type",
    [ClusterType.DECODE_FFN.value, None],
    ids=["raw-int", "none"],
)
def test_m2n_target_validation_rejects_nonexact_scheduler_cluster_type(
    invalid_cluster_type,
) -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = invalid_cluster_type

    with pytest.raises((TypeError, ValueError), match="ClusterType|cluster_type"):
        scheduler.validate_m2n_arrival_target(_transfer_info())


@pytest.mark.parametrize(
    "invalid_cluster_type",
    [ClusterType.DECODE_FFN.value, None],
    ids=["raw-int", "none"],
)
def test_m2n_target_validation_rejects_nonexact_transfer_cluster_type(
    invalid_cluster_type,
) -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    transfer_info = _transfer_info()
    transfer_info.target_cluster_type = invalid_cluster_type

    with pytest.raises((TypeError, ValueError), match="ClusterType|cluster_type"):
        scheduler.validate_m2n_arrival_target(transfer_info)


def test_m2n_target_validation_rejects_mutated_raw_source_cluster_type() -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    transfer_info = _transfer_info()
    transfer_info.source_cluster_type = ClusterType.DECODE_ATTN.value

    with pytest.raises(
        (TypeError, ValueError),
        match="ClusterType|source_cluster_type",
    ):
        scheduler.validate_m2n_arrival_target(transfer_info)


def test_m2n_transfer_end_rejects_wrong_registry_scheduler_before_side_effects() -> None:
    request = SimpleNamespace(
        id=34,
        on_m2n_transfer_complete=Mock(),
        on_inter_cluster_transfer_end=Mock(),
    )
    batch = SimpleNamespace(id=12, global_id=77, requests=[request])
    transfer_info = _transfer_info(
        batch=batch,
        target_cluster_type=ClusterType.DECODE_FFN,
        transfer_start_time=0.5,
        transfer_end_time=0.75,
    )
    wrong_cluster_scheduler = _scheduler()
    wrong_cluster_scheduler._cluster_type = ClusterType.DECODE_ATTN
    wrong_cluster_scheduler.on_m2n_arrival = Mock(return_value=[])
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=wrong_cluster_scheduler)
    )
    metrics_store = SimpleNamespace(on_m2n_transfer_end=Mock())
    transfer_end_time_before = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(1.0, transfer_info)

    with pytest.raises(ValueError, match="M2N target scheduler mismatch"):
        try:
            event.handle_event(global_scheduler, metrics_store)
        finally:
            metrics_store.on_m2n_transfer_end.assert_not_called()
            request.on_m2n_transfer_complete.assert_not_called()
            request.on_inter_cluster_transfer_end.assert_not_called()
            wrong_cluster_scheduler.on_m2n_arrival.assert_not_called()
            assert transfer_info.transfer_end_time == transfer_end_time_before


def test_m2n_transfer_end_rejects_malformed_decode_ffn_receipt_before_side_effects() -> None:
    request = SimpleNamespace(
        id=34,
        on_arrival=Mock(),
        on_m2n_transfer_complete=Mock(),
        on_inter_cluster_transfer_end=Mock(),
    )
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(
        batch=batch,
        layer_id=None,
        transfer_start_time=0.5,
        transfer_end_time=0.75,
    )
    target_scheduler = _scheduler()
    target_scheduler._cluster_type = ClusterType.DECODE_FFN
    target_scheduler._m2n_waiting_by_layer = {}
    target_scheduler._m2n_ready_groups = deque()
    target_scheduler._is_periodic_scheduling_enabled = False
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=target_scheduler)
    )
    metrics_store = SimpleNamespace(on_m2n_transfer_end=Mock())
    transfer_end_time_before = transfer_info.transfer_end_time
    event = M2NTransferEndEvent(1.0, transfer_info)

    with pytest.raises(ValueError, match="layer_id"):
        try:
            event.handle_event(global_scheduler, metrics_store)
        finally:
            assert transfer_info.transfer_end_time == transfer_end_time_before
            metrics_store.on_m2n_transfer_end.assert_not_called()
            request.on_m2n_transfer_complete.assert_not_called()
            request.on_inter_cluster_transfer_end.assert_not_called()
            request.on_arrival.assert_not_called()
            assert target_scheduler._m2n_waiting_by_layer == {}
            assert tuple(target_scheduler._m2n_ready_groups) == ()


def test_decode_ffn_empty_metadata_rejects_unconfigured_lane_before_side_effects() -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=(),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(
        batch=batch,
        source_replica_id=99,
        source_replica_local_id=99,
    )
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = [(0, None)]
    scheduler._ffn_group_micro_batches = 1
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    logger = Mock()

    with pytest.raises(ValueError, match="Unexpected lane.*DECODE_FFN"):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                batch,
                transfer_info,
                logger,
            )
        finally:
            request.on_arrival.assert_not_called()
            assert not hasattr(batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer == {}
            assert tuple(scheduler._m2n_ready_groups) == ()


def test_decode_ffn_empty_metadata_accepts_configured_lane() -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=(),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(batch=batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = [(0, None)]
    scheduler._ffn_group_micro_batches = 1
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False

    events = scheduler._handle_m2n_arrival_decode_ffn(
        1.25,
        batch,
        transfer_info,
        Mock(),
    )

    request.on_arrival.assert_called_once_with(1.25, ClusterType.DECODE_FFN)
    assert batch.decode_ffn_m2n_arrival_time == pytest.approx(1.25)
    assert scheduler._m2n_waiting_by_layer == {}
    assert list(scheduler._m2n_ready_groups) == [[(batch, transfer_info)]]
    assert len(events) == 1
    assert events[0].time == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("scheduler_lanes", "error_match"),
    [
        (None, "scheduler lane topology"),
        ([], "scheduler lane topology"),
        ([[0, 0]], "exact 2-tuples"),
        ([(0, None), (0, None)], "duplicate"),
        ([(0, None), (-1, 0)], "replica_id"),
        ([(0, None), (1, True)], "dp_id"),
    ],
    ids=["none", "empty", "list-lane", "duplicate", "negative", "bool"],
)
def test_decode_ffn_empty_metadata_rejects_malformed_scheduler_lane_topology(
    scheduler_lanes,
    error_match: str,
) -> None:
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=(),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(batch=batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = scheduler_lanes
    scheduler._ffn_group_micro_batches = 1
    scheduler._m2n_waiting_by_layer = {}

    with pytest.raises(ValueError, match=error_match):
        scheduler._validate_decode_ffn_m2n_receipt(batch, transfer_info)


def test_decode_ffn_rejects_inconsistent_room_lane_contract_before_side_effects() -> None:
    first_request = SimpleNamespace(id=34, on_arrival=Mock())
    first_batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[first_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None), (1, None)),
        decode_attn_barrier_round_id=7,
    )
    first_transfer_info = _transfer_info(batch=first_batch)
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = [(0, None), (1, None), (2, None)]
    scheduler._ffn_group_micro_batches = 3
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    logger = Mock()

    first_events = scheduler._handle_m2n_arrival_decode_ffn(
        1.0,
        first_batch,
        first_transfer_info,
        logger,
    )
    assert first_events == []
    first_request.on_arrival.assert_called_once_with(1.0, ClusterType.DECODE_FFN)
    assert first_batch.decode_ffn_m2n_arrival_time == pytest.approx(1.0)

    group_key = (4, 1, 7)
    room = scheduler._m2n_waiting_by_layer[group_key]
    assert tuple(room["lanes_rr_order"]) == ((0, None),)
    assert tuple(room["per_lane_queues"]) == ((0, None),)
    assert tuple(room["per_lane_queues"][(0, None)]) == (
        (first_batch, first_transfer_info),
    )
    ready_groups_before = tuple(scheduler._m2n_ready_groups)

    second_request = SimpleNamespace(
        id=35,
        on_arrival=Mock(),
        on_m2n_transfer_complete=Mock(),
        on_inter_cluster_transfer_end=Mock(),
    )
    second_batch = SimpleNamespace(
        id=13,
        global_id=78,
        requests=[second_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None), (2, None)),
        decode_attn_barrier_round_id=7,
    )
    second_transfer_info = _transfer_info(
        batch=second_batch,
        source_replica_id=2,
        source_replica_local_id=None,
        transfer_start_time=0.6,
        transfer_end_time=0.9,
    )
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler)
    )
    metrics_store = SimpleNamespace(on_m2n_transfer_end=Mock())
    transfer_end_time_before = second_transfer_info.transfer_end_time

    with pytest.raises(
        ValueError,
        match="Inconsistent.*expected lane contract|expected lane contract.*inconsistent",
    ):
        try:
            M2NTransferEndEvent(1.1, second_transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert second_transfer_info.transfer_end_time == transfer_end_time_before
            metrics_store.on_m2n_transfer_end.assert_not_called()
            second_request.on_m2n_transfer_complete.assert_not_called()
            second_request.on_inter_cluster_transfer_end.assert_not_called()
            second_request.on_arrival.assert_not_called()
            assert not hasattr(second_batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer[group_key] is room
            assert tuple(room["lanes_rr_order"]) == ((0, None),)
            assert tuple(room["per_lane_queues"]) == ((0, None),)
            assert tuple(room["per_lane_queues"][(0, None)]) == (
                (first_batch, first_transfer_info),
            )
            assert tuple(scheduler._m2n_ready_groups) == ready_groups_before


def test_decode_ffn_accepts_consistent_room_lane_contract() -> None:
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = [(0, None), (1, None)]
    scheduler._ffn_group_micro_batches = 2
    scheduler._m2n_waiting_by_layer = {}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    lane_contract = ((0, None), (1, None))

    first_request = SimpleNamespace(id=34, on_arrival=Mock())
    first_batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[first_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=lane_contract,
        decode_attn_barrier_round_id=7,
    )
    first_transfer_info = _transfer_info(batch=first_batch)
    first_events = scheduler._handle_m2n_arrival_decode_ffn(
        1.0,
        first_batch,
        first_transfer_info,
        Mock(),
    )

    second_request = SimpleNamespace(id=35, on_arrival=Mock())
    second_batch = SimpleNamespace(
        id=13,
        global_id=78,
        requests=[second_request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=tuple(reversed(lane_contract)),
        decode_attn_barrier_round_id=7,
    )
    second_transfer_info = _transfer_info(
        batch=second_batch,
        source_replica_id=1,
    )
    second_events = scheduler._handle_m2n_arrival_decode_ffn(
        1.1,
        second_batch,
        second_transfer_info,
        Mock(),
    )

    assert first_events == []
    assert len(second_events) == 1
    assert second_events[0].time == pytest.approx(1.1)
    first_request.on_arrival.assert_called_once_with(1.0, ClusterType.DECODE_FFN)
    second_request.on_arrival.assert_called_once_with(1.1, ClusterType.DECODE_FFN)
    assert scheduler._m2n_waiting_by_layer == {}
    assert list(scheduler._m2n_ready_groups) == [
        [(first_batch, first_transfer_info), (second_batch, second_transfer_info)]
    ]


def test_decode_ffn_rejects_existing_room_without_lane_contract_before_side_effects() -> None:
    request = SimpleNamespace(id=34, on_arrival=Mock())
    batch = SimpleNamespace(
        id=12,
        global_id=77,
        requests=[request],
        is_idle=False,
        afd_stage_metadata=None,
        decode_attn_barrier_expected_lanes=((0, None),),
        decode_attn_barrier_round_id=7,
    )
    transfer_info = _transfer_info(batch=batch)
    group_key = (4, 1, 7)
    corrupt_room = {
        "per_lane_queues": defaultdict(deque),
        "lanes_rr_order": deque(),
        "rr_cursor": 0,
    }
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._ffn_expected_lanes = [(0, None)]
    scheduler._ffn_group_micro_batches = 1
    scheduler._m2n_waiting_by_layer = {group_key: corrupt_room}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False

    with pytest.raises(
        (RuntimeError, ValueError),
        match="missing.*expected lane contract|expected lane contract.*missing",
    ):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                batch,
                transfer_info,
                Mock(),
            )
        finally:
            request.on_arrival.assert_not_called()
            assert not hasattr(batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer == {group_key: corrupt_room}
            assert tuple(corrupt_room["lanes_rr_order"]) == ()
            assert dict(corrupt_room["per_lane_queues"]) == {}
            assert tuple(scheduler._m2n_ready_groups) == ()


def test_decode_ffn_rejects_corrupt_existing_queue_before_lifecycle_mutation() -> None:
    """A malformed queued item must fail before arrival or queue mutation."""

    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_request = incoming_batch.requests[0]
    incoming_request.on_arrival = Mock()
    incoming_transfer = _transfer_info(
        batch=incoming_batch,
        source_replica_id=1,
        source_replica_local_id=None,
        layer_id=4,
        afd_stage_idx=1,
    )
    group_key = (4, 1, 7)
    corrupt_entry = object()
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([corrupt_entry])},
        ),
        "lanes_rr_order": deque([(0, None)]),
        "rr_cursor": 0,
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {group_key: room}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    waiting_room_before = scheduler._m2n_waiting_by_layer[group_key]
    queue_before = tuple(room["per_lane_queues"][(0, None)])
    lanes_before = tuple(room["lanes_rr_order"])

    with pytest.raises(
        (AttributeError, TypeError, RuntimeError, ValueError),
        match="waiting|queued|Batch|entry|tuple|M2N",
    ):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                incoming_batch,
                incoming_transfer,
                Mock(),
            )
        finally:
            incoming_request.on_arrival.assert_not_called()
            assert not hasattr(incoming_batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer[group_key] is waiting_room_before
            assert tuple(room["per_lane_queues"][(0, None)]) == queue_before
            assert tuple(room["lanes_rr_order"]) == lanes_before
            assert tuple(scheduler._m2n_ready_groups) == ()


def test_decode_ffn_rejects_stale_existing_queue_entry_before_barrier_mix() -> None:
    """A real queued batch from another stage/round must not join this barrier."""

    stale_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=9,
        barrier_round_id=99,
    )
    stale_transfer = _transfer_info(
        batch=stale_batch,
        source_replica_id=0,
        source_replica_local_id=None,
        layer_id=4,
        afd_stage_idx=9,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_request = incoming_batch.requests[0]
    incoming_request.on_arrival = Mock()
    incoming_transfer = _transfer_info(
        batch=incoming_batch,
        source_replica_id=1,
        source_replica_local_id=None,
        layer_id=4,
        afd_stage_idx=1,
    )
    group_key = (4, 1, 7)
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(stale_batch, stale_transfer)])},
        ),
        "lanes_rr_order": deque([(0, None)]),
        "rr_cursor": 0,
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._m2n_waiting_by_layer = {group_key: room}
    scheduler._m2n_ready_groups = deque()
    scheduler._is_periodic_scheduling_enabled = False
    queue_before = tuple(room["per_lane_queues"][(0, None)])
    lanes_before = tuple(room["lanes_rr_order"])

    with pytest.raises((RuntimeError, ValueError), match="stage|round|layer|waiting|cohort"):
        try:
            scheduler._handle_m2n_arrival_decode_ffn(
                1.25,
                incoming_batch,
                incoming_transfer,
                Mock(),
            )
        finally:
            incoming_request.on_arrival.assert_not_called()
            assert not hasattr(incoming_batch, "decode_ffn_m2n_arrival_time")
            assert scheduler._m2n_waiting_by_layer[group_key] is room
            assert tuple(room["per_lane_queues"][(0, None)]) == queue_before
            assert tuple(room["lanes_rr_order"]) == lanes_before
            assert tuple(scheduler._m2n_ready_groups) == ()


def _a2f_waiting_room_scheduler(room: dict) -> _ConcreteClusterScheduler:
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=SimpleNamespace(is_moe=True, num_layers=4),
        ),
        decode_ffn_cluster_num_replicas=1,
    )
    scheduler._a2f_expected_lanes = [(0, None), (1, None)]
    scheduler._replica_schedulers = {
        (0, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
        (1, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
    }
    scheduler._decode_attn_idle_expected_lanes = {(1, None)}
    scheduler._a2f_waiting_by_layer = {(4, 1): room}
    scheduler._decode_attn_barrier_round_counter = 10
    scheduler._m2n_transfer_predictor = SimpleNamespace(
        get_transfer_info=Mock(return_value=(128, 0.5)),
    )
    scheduler._is_periodic_scheduling_enabled = False
    return scheduler


def _a2f_room_snapshot(room: dict) -> tuple:
    return (
        id(room),
        id(room["per_lane_queues"]),
        tuple(
            (
                lane,
                id(queue),
                tuple((layer_id, id(queued_batch)) for layer_id, queued_batch in queue),
            )
            for lane, queue in sorted(room["per_lane_queues"].items())
        ),
    )


def _a2f_batch_barrier_snapshot(batch: Batch) -> tuple:
    return (
        batch.decode_attn_barrier_round_id,
        batch.decode_attn_barrier_expected_lanes,
        batch.afd_stage_idx,
        batch.time,
    )


def _dense_a2f_scheduler() -> tuple[_ConcreteClusterScheduler, dict]:
    """Build a one-lane dense A-to-F scheduler fixture."""

    room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lane_contract": ((1, None),),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    scheduler._config.replica_config.model_config.is_moe = False
    scheduler._a2f_expected_lanes = [(1, None)]
    scheduler._replica_schedulers = {
        (1, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
    }
    scheduler._a2f_waiting_by_layer = {}
    scheduler._decode_attn_idle_expected_lanes = {(1, None)}
    return scheduler, room


def _dense_a2f_batch() -> Batch:
    batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
        expected_lanes=((1, None),),
    )
    batch._is_moe = False
    return batch


def _dense_a2f_state_snapshot(scheduler, batch: Batch, room: dict) -> tuple:
    return (
        id(scheduler._a2f_waiting_by_layer),
        tuple(
            (group_key, id(group_room))
            for group_key, group_room in scheduler._a2f_waiting_by_layer.items()
        ),
        _a2f_room_snapshot(room),
        set(scheduler._decode_attn_idle_expected_lanes),
        scheduler._decode_attn_barrier_round_counter,
        _a2f_batch_barrier_snapshot(batch),
        tuple(
            (
                id(request),
                request.completed,
                request.completed_layer_count,
            )
            for request in batch.requests
        ),
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "original_lane_mismatch",
        "original_lane_bool",
        "original_lane_float",
        "request_layer_mismatch",
        "decode_ffn_layer_mismatch",
        "no_active_request",
        "idle_batch",
        "is_moe_mismatch",
        "non_request_payload",
    ],
)
def test_decode_attn_a2f_dense_rejects_invalid_batch_before_side_effects(
    invalid_case: str,
) -> None:
    """Dense A-to-F admission must validate the whole Batch before release."""

    scheduler, room = _dense_a2f_scheduler()
    batch = _dense_a2f_batch()
    request = batch.requests[0]
    if invalid_case == "original_lane_mismatch":
        batch.decode_attn_original_replica_id = 0
    elif invalid_case == "original_lane_bool":
        batch.decode_attn_original_replica_id = True
    elif invalid_case == "original_lane_float":
        batch.decode_attn_original_replica_local_id = 0.0
    elif invalid_case == "request_layer_mismatch":
        request._completed_layer_count = 99
    elif invalid_case == "decode_ffn_layer_mismatch":
        batch.decode_ffn_layer_id = 99
    elif invalid_case == "no_active_request":
        request._completed = True
    elif invalid_case == "idle_batch":
        batch._is_idle = True
        batch._requests = []
        batch._num_tokens = []
        batch._total_num_tokens = 0
    elif invalid_case == "is_moe_mismatch":
        batch._is_moe = True
    elif invalid_case == "non_request_payload":
        batch._requests = [
            SimpleNamespace(id=request.id, completed=False, completed_layer_count=4)
        ]
    else:
        raise AssertionError(f"Unhandled invalid case: {invalid_case}")

    state_before = _dense_a2f_state_snapshot(scheduler, batch, room)
    predictor = scheduler._m2n_transfer_predictor.get_transfer_info
    with (
        patch(
            "frontier.events.m2n_transfer_start_event.M2NTransferStartEvent"
        ) as transfer_event,
        patch("frontier.events.replica_schedule_event.ReplicaScheduleEvent") as schedule_event,
        pytest.raises((RuntimeError, TypeError, ValueError), match="A.?F|Batch|Replica|lane|layer|request|idle|moe|is_moe|exact|full-stage"),
    ):
        scheduler.on_decode_attn_a2f_ready(
            1.25,
            batch,
            replica_id=1,
            dp_id=None,
            layer_id=4,
            logger=Mock(),
        )

    assert _dense_a2f_state_snapshot(scheduler, batch, room) == state_before
    predictor.assert_not_called()
    transfer_event.assert_not_called()
    schedule_event.assert_not_called()


@pytest.mark.parametrize("active_state", [{}, {9: {"afd_stage_idx": 1, "af_phase": "ffn_inflight", "current_layer_id": 4}}])
def test_decode_attn_a2f_dense_rejects_missing_local_attn_topology_without_fallback(
    active_state: dict,
) -> None:
    """An empty/non-local-attn stage must not fall back to configured lanes."""

    scheduler, room = _dense_a2f_scheduler()
    batch = _dense_a2f_batch()
    batch.decode_attn_cohort_id = 9
    batch.decode_attn_cohort_request_ids = tuple(request.id for request in batch.requests)
    scheduler._replica_schedulers = {
        (1, None): SimpleNamespace(_decode_attn_active_cohort_states=active_state),
    }
    state_before = _dense_a2f_state_snapshot(scheduler, batch, room)
    with pytest.raises((RuntimeError, ValueError), match="active|cohort|local_attn|topology"):
        scheduler.on_decode_attn_a2f_ready(
            1.25,
            batch,
            replica_id=1,
            dp_id=None,
            layer_id=4,
            logger=Mock(),
        )
    assert _dense_a2f_state_snapshot(scheduler, batch, room) == state_before
    scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


def _dense_a2f_cohort_fixture() -> tuple[_ConcreteClusterScheduler, dict, Batch, dict]:
    scheduler, room = _dense_a2f_scheduler()
    batch = _dense_a2f_batch()
    batch.decode_attn_cohort_id = 9
    batch.decode_attn_cohort_request_ids = tuple(request.id for request in batch.requests)
    cohort_state = {
        "all_request_ids": {request.id for request in batch.requests},
        "pending_request_ids": {request.id for request in batch.requests},
        "af_phase": "local_attn",
        "current_layer_id": 4,
        "active_stage_indices": {1},
        "stage_phases": {1: "local_attn"},
        "stage_current_layer_ids": {1: 4},
        "afd_stage_idx": 1,
    }
    scheduler._replica_schedulers = {
        (1, None): SimpleNamespace(_decode_attn_active_cohort_states={9: cohort_state}),
    }
    return scheduler, room, batch, cohort_state


def _assert_dense_a2f_atomic_state(
    scheduler,
    room,
    batch,
    cohort_state,
    before,
) -> None:
    assert _dense_a2f_state_snapshot(scheduler, batch, room) == before[0]
    assert cohort_state == before[1]


def test_decode_attn_a2f_dense_event_constructor_failure_is_atomic() -> None:
    scheduler, room, batch, cohort_state = _dense_a2f_cohort_fixture()
    before = (
        _dense_a2f_state_snapshot(scheduler, batch, room),
        deepcopy(cohort_state),
    )
    with patch(
        "frontier.events.m2n_transfer_start_event.M2NTransferStartEvent",
        side_effect=RuntimeError("event ctor boom"),
    ) as transfer_event:
        with pytest.raises(RuntimeError, match="event ctor boom"):
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
    _assert_dense_a2f_atomic_state(scheduler, room, batch, cohort_state, before)
    assert scheduler._m2n_transfer_predictor.get_transfer_info.call_count == 1
    assert transfer_event.call_count == 1


def test_decode_attn_a2f_accepts_numpy_real_event_time() -> None:
    """Numerical predictor scalars must satisfy the timestamp contract."""
    import numpy as np

    scheduler, _room, batch, _cohort_state = _dense_a2f_cohort_fixture()
    events = scheduler.on_decode_attn_a2f_ready(
        np.float64(1.25),
        batch,
        replica_id=1,
        dp_id=None,
        layer_id=4,
        logger=Mock(),
    )

    assert events
    assert all(type(event.time) is float for event in events)


def test_decode_attn_a2f_dense_cohort_setter_failure_is_atomic() -> None:
    scheduler, room, batch, cohort_state = _dense_a2f_cohort_fixture()
    before = (
        _dense_a2f_state_snapshot(scheduler, batch, room),
        deepcopy(cohort_state),
    )
    with patch.object(
        scheduler,
        "_set_decode_attn_batch_cohort_phase",
        side_effect=RuntimeError("cohort setter boom"),
    ):
        with pytest.raises(RuntimeError, match="cohort setter boom"):
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
    _assert_dense_a2f_atomic_state(scheduler, room, batch, cohort_state, before)


def test_decode_attn_a2f_dense_cohort_apply_failure_is_atomic() -> None:
    scheduler, room, batch, cohort_state = _dense_a2f_cohort_fixture()
    before = (
        _dense_a2f_state_snapshot(scheduler, batch, room),
        deepcopy(cohort_state),
    )
    with patch.object(
        scheduler,
        "_apply_decode_attn_batch_cohort_phase",
        side_effect=RuntimeError("cohort apply boom"),
    ):
        with pytest.raises(RuntimeError, match="cohort apply boom"):
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
    _assert_dense_a2f_atomic_state(scheduler, room, batch, cohort_state, before)


def test_decode_attn_a2f_moe_event_constructor_failure_is_atomic() -> None:
    queued_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(4, queued_batch)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    before_room = _a2f_room_snapshot(room)
    before_round = scheduler._decode_attn_barrier_round_counter
    before_batches = {
        id(batch): _a2f_batch_barrier_snapshot(batch)
        for batch in (queued_batch, incoming_batch)
    }
    with patch(
        "frontier.events.m2n_transfer_start_event.M2NTransferStartEvent",
        side_effect=RuntimeError("event ctor boom"),
    ) as transfer_event:
        with pytest.raises(RuntimeError, match="event ctor boom"):
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                incoming_batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
    assert _a2f_room_snapshot(room) == before_room
    assert scheduler._a2f_waiting_by_layer[(4, 1)] is room
    assert scheduler._decode_attn_barrier_round_counter == before_round
    for batch in (queued_batch, incoming_batch):
        assert _a2f_batch_barrier_snapshot(batch) == before_batches[id(batch)]
    assert transfer_event.call_count == 1


def test_decode_attn_a2f_moe_cohort_apply_failure_is_atomic() -> None:
    queued_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    cohort_request_ids = tuple(
        request.id
        for cohort_batch in (queued_batch, incoming_batch)
        for request in cohort_batch.requests
    )
    for cohort_batch in (queued_batch, incoming_batch):
        cohort_batch.decode_attn_cohort_id = 9
        cohort_batch.decode_attn_cohort_request_ids = cohort_request_ids
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(4, queued_batch)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    cohort_states_by_lane = {
        lane: {
            "all_request_ids": set(cohort_request_ids),
            "pending_request_ids": set(cohort_request_ids),
            "af_phase": "local_attn",
            "current_layer_id": 4,
            "active_stage_indices": {1},
            "stage_phases": {1: "local_attn"},
            "stage_current_layer_ids": {1: 4},
        }
        for lane in ((0, None), (1, None))
    }
    scheduler._replica_schedulers = {
        lane: SimpleNamespace(
            _decode_attn_active_cohort_states={9: cohort_state}
        )
        for lane, cohort_state in cohort_states_by_lane.items()
    }
    before_room = _a2f_room_snapshot(room)
    before_round = scheduler._decode_attn_barrier_round_counter
    before_batches = {
        id(batch): _a2f_batch_barrier_snapshot(batch)
        for batch in (queued_batch, incoming_batch)
    }
    before_cohort_states = deepcopy(cohort_states_by_lane)
    with patch.object(
        scheduler,
        "_apply_decode_attn_batch_cohort_phase",
        side_effect=RuntimeError("cohort apply boom"),
    ):
        with pytest.raises(RuntimeError, match="cohort apply boom"):
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                incoming_batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
    assert _a2f_room_snapshot(room) == before_room
    assert scheduler._a2f_waiting_by_layer[(4, 1)] is room
    assert scheduler._decode_attn_barrier_round_counter == before_round
    for batch in (queued_batch, incoming_batch):
        assert _a2f_batch_barrier_snapshot(batch) == before_batches[id(batch)]
    assert cohort_states_by_lane == before_cohort_states


def test_decode_attn_a2f_rejects_corrupt_existing_queue_before_mutation() -> None:
    """A malformed A→F queued batch must not consume the room or idle lane."""

    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    corrupt_entry = object()
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(99, corrupt_entry)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    room_before = _a2f_room_snapshot(room)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)
    predictor = scheduler._m2n_transfer_predictor.get_transfer_info

    with pytest.raises(
        (AttributeError, RuntimeError, TypeError, ValueError),
        match="waiting|queued|Batch|entry|layer|stage|A.?F",
    ):
        try:
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                incoming_batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
        finally:
            assert _a2f_room_snapshot(room) == room_before
            assert scheduler._decode_attn_idle_expected_lanes == idle_before
            assert predictor.call_count == 0
            assert scheduler._decode_attn_barrier_round_counter == 10


def test_decode_attn_a2f_rejects_stale_existing_queue_before_event_mix() -> None:
    """A real queued batch from another layer/stage must not emit an A→F event."""

    stale_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=99,
        afd_stage_idx=9,
        barrier_round_id=99,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(99, stale_batch)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    room_before = _a2f_room_snapshot(room)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)
    predictor = scheduler._m2n_transfer_predictor.get_transfer_info
    stale_stage_before = stale_batch.afd_stage_idx
    stale_round_before = stale_batch.decode_attn_barrier_round_id

    with pytest.raises((RuntimeError, ValueError), match="layer|stage|round|waiting|cohort"):
        try:
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                incoming_batch,
                replica_id=1,
                dp_id=None,
                layer_id=4,
                logger=Mock(),
            )
        finally:
            assert _a2f_room_snapshot(room) == room_before
            assert scheduler._decode_attn_idle_expected_lanes == idle_before
            assert predictor.call_count == 0
            assert scheduler._decode_attn_barrier_round_counter == 10
            assert stale_batch.afd_stage_idx == stale_stage_before
            assert stale_batch.decode_attn_barrier_round_id == stale_round_before
            assert scheduler._a2f_waiting_by_layer[(4, 1)] is room


@pytest.mark.parametrize(
    ("field", "value", "kwargs", "error_match"),
    [
        ("layer_id", True, {}, "layer_id"),
        ("layer_id", 4.0, {}, "layer_id"),
        ("afd_stage_idx", True, {}, "afd_stage_idx"),
        ("afd_stage_idx", 1.0, {}, "afd_stage_idx"),
        ("replica_id", True, {"dp_id": 0}, "replica_id"),
            ("dp_id", 0.0, {"replica_id": 1}, "replica_local_id"),
    ],
)
def test_decode_attn_a2f_rejects_coercible_topology_before_mutation(
    field: str,
    value,
    kwargs: dict,
    error_match: str,
) -> None:
    """A→F admission must reject bool/float topology values before state changes."""

    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    room_before = _a2f_room_snapshot(room)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)
    round_before = scheduler._decode_attn_barrier_round_counter

    call_kwargs = {
        "replica_id": 1,
        "dp_id": 0,
        "layer_id": 4,
        "logger": Mock(),
    }
    if field in {"layer_id", "replica_id", "dp_id"}:
        call_kwargs[field] = value
    else:
        setattr(incoming_batch, field, value)
    call_kwargs.update(kwargs)

    with pytest.raises((TypeError, RuntimeError, ValueError), match=error_match):
        try:
            scheduler.on_decode_attn_a2f_ready(
                1.25,
                incoming_batch,
                **call_kwargs,
            )
        finally:
            assert _a2f_room_snapshot(room) == room_before
            assert scheduler._decode_attn_idle_expected_lanes == idle_before
            assert scheduler._decode_attn_barrier_round_counter == round_before


def test_decode_attn_a2f_rejects_existing_room_without_lane_contract() -> None:
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {"per_lane_queues": defaultdict(deque)}
    scheduler = _a2f_waiting_room_scheduler(room)
    room_before = _a2f_room_snapshot(room)
    room_map_before = dict(scheduler._a2f_waiting_by_layer)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="expected lane contract|expected_lane_contract|waiting-room",
    ):
        scheduler.on_decode_attn_a2f_ready(
            1.25,
            incoming_batch,
            replica_id=1,
            dp_id=None,
            layer_id=4,
            logger=Mock(),
        )

    assert _a2f_room_snapshot(room) == room_before
    assert scheduler._a2f_waiting_by_layer == room_map_before
    assert scheduler._a2f_waiting_by_layer[(4, 1)] is room
    assert scheduler._decode_attn_idle_expected_lanes == idle_before
    scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


@pytest.mark.parametrize(
    "invalid_lane",
    [(True, 0), (1.0, 0), (1, False), (1, 0.0)],
)
def test_decode_attn_a2f_rejects_coercible_expected_lane_topology(
    invalid_lane,
) -> None:
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    scheduler._a2f_expected_lanes = [(0, None), invalid_lane]
    room_before = _a2f_room_snapshot(room)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="lane topology|replica_id|dp_id|exact non-negative int",
    ):
        scheduler.on_decode_attn_a2f_ready(
            1.25,
            incoming_batch,
            replica_id=1,
            dp_id=None,
            layer_id=4,
            logger=Mock(),
        )

    assert _a2f_room_snapshot(room) == room_before
    assert scheduler._decode_attn_idle_expected_lanes == idle_before
    assert scheduler._decode_attn_barrier_round_counter == 10
    scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


@pytest.mark.parametrize("workflow", ["moe", "dense"])
@pytest.mark.parametrize(
    ("predictor_result", "predictor_error", "error_match"),
    [
        (None, RuntimeError("predictor boom"), "predictor boom"),
        (None, None, "exact.*tuple|transfer result"),
        ((128, float("nan")), None, "finite|transfer_time"),
        ((-1, 0.5), None, "activation_size|non-negative"),
        ((128, -0.5), None, "transfer_time|non-negative"),
    ],
)
def test_decode_attn_a2f_predictor_failure_preserves_runtime_state(
    workflow: str,
    predictor_result,
    predictor_error,
    error_match: str,
) -> None:
    queued_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
        is_moe=workflow == "moe",
    )
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(4, queued_batch)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    if workflow == "dense":
        scheduler._config.replica_config.model_config.is_moe = False
        scheduler._a2f_expected_lanes = [(1, None)]
        scheduler._replica_schedulers = {
            (1, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
        }
        scheduler._a2f_waiting_by_layer = {}
        scheduler._decode_attn_idle_expected_lanes = {(1, None)}
    predictor = Mock(return_value=predictor_result)
    if predictor_error is not None:
        predictor.side_effect = predictor_error
    scheduler._m2n_transfer_predictor.get_transfer_info = predictor

    room_before = _a2f_room_snapshot(room)
    room_map_before = dict(scheduler._a2f_waiting_by_layer)
    idle_before = set(scheduler._decode_attn_idle_expected_lanes)
    round_before = scheduler._decode_attn_barrier_round_counter
    batch_snapshots_before = {
        id(batch): _a2f_batch_barrier_snapshot(batch)
        for batch in (queued_batch, incoming_batch)
    }

    with (
        patch(
            "frontier.events.m2n_transfer_start_event.M2NTransferStartEvent"
        ) as transfer_event,
        patch(
            "frontier.events.replica_schedule_event.ReplicaScheduleEvent"
        ) as schedule_event,
        pytest.raises(
            (RuntimeError, TypeError, ValueError),
            match=error_match,
        ),
    ):
        scheduler.on_decode_attn_a2f_ready(
            1.25,
            incoming_batch,
            replica_id=1,
            dp_id=None,
            layer_id=4,
            logger=Mock(),
        )

    assert _a2f_room_snapshot(room) == room_before
    assert scheduler._a2f_waiting_by_layer == room_map_before
    assert scheduler._decode_attn_idle_expected_lanes == idle_before
    assert scheduler._decode_attn_barrier_round_counter == round_before
    for batch in (queued_batch, incoming_batch):
        assert _a2f_batch_barrier_snapshot(batch) == batch_snapshots_before[id(batch)]
    transfer_event.assert_not_called()
    schedule_event.assert_not_called()


def test_decode_attn_a2f_moe_incomplete_barrier_commits_incoming_and_idle() -> None:
    incoming_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=78,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
        expected_lanes=((0, None), (1, None), (2, None)),
    )
    scheduler = _a2f_waiting_room_scheduler(
        {
            "per_lane_queues": defaultdict(deque),
            "expected_lane_contract": ((0, None), (1, None), (2, None)),
        }
    )
    scheduler._a2f_expected_lanes = [(0, None), (1, None), (2, None)]
    scheduler._replica_schedulers = {
        lane: SimpleNamespace(_decode_attn_active_cohort_states={})
        for lane in scheduler._a2f_expected_lanes
    }
    scheduler._decode_attn_idle_expected_lanes = {(2, None)}

    events = scheduler.on_decode_attn_a2f_ready(
        1.25,
        incoming_batch,
        replica_id=0,
        dp_id=None,
        layer_id=4,
        logger=Mock(),
    )

    room = scheduler._a2f_waiting_by_layer[(4, 1)]
    assert events == []
    assert room["expected_lane_contract"] == ((0, None), (1, None), (2, None))
    assert tuple(room["per_lane_queues"][(0, None)]) == ((4, incoming_batch),)
    idle_entry = tuple(room["per_lane_queues"][(2, None)])
    assert len(idle_entry) == 1
    assert idle_entry[0][0] == 4
    assert idle_entry[0][1].is_idle is True
    assert scheduler._decode_attn_idle_expected_lanes == {(2, None)}
    assert scheduler._decode_attn_barrier_round_counter == 10
    scheduler._m2n_transfer_predictor.get_transfer_info.assert_not_called()


def test_decode_attn_a2f_complete_barrier_preserves_lane_fifo() -> None:
    first_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=76,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=6,
    )
    second_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=77,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
    )
    room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([(4, first_batch), (4, second_batch)])},
        ),
        "expected_lane_contract": ((0, None), (1, None)),
    }
    scheduler = _a2f_waiting_room_scheduler(room)

    events = scheduler.on_decode_attn_a2f_ready(
        1.25,
        incoming_batch,
        replica_id=1,
        dp_id=None,
        layer_id=4,
        logger=Mock(),
    )

    assert len(events) == 4
    assert tuple(room["per_lane_queues"][(0, None)]) == ((4, second_batch),)
    assert tuple(room["per_lane_queues"][(1, None)]) == ()
    assert first_batch.decode_attn_barrier_round_id == 10
    assert incoming_batch.decode_attn_barrier_round_id == 10
    assert second_batch.decode_attn_barrier_round_id == 7
    assert scheduler._decode_attn_barrier_round_counter == 11
    assert scheduler._m2n_transfer_predictor.get_transfer_info.call_count == 2
    assert scheduler._a2f_waiting_by_layer[(4, 1)] is room


def test_decode_attn_a2f_single_batch_complete_drain_removes_waiting_room() -> None:
    incoming_batch = _real_pdaf_batch(
        replica_id=0,
        global_id=78,
        lane=(0, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
        expected_lanes=((0, None),),
    )
    room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lane_contract": ((0, None),),
    }
    scheduler = _a2f_waiting_room_scheduler(room)
    scheduler._a2f_expected_lanes = [(0, None)]
    scheduler._replica_schedulers = {
        (0, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
    }
    scheduler._decode_attn_idle_expected_lanes = {(0, None)}

    events = scheduler.on_decode_attn_a2f_ready(
        1.25,
        incoming_batch,
        replica_id=0,
        dp_id=None,
        layer_id=4,
        logger=Mock(),
    )

    assert len(events) == 2
    assert scheduler._a2f_waiting_by_layer == {}
    assert scheduler._decode_attn_idle_expected_lanes == set()
    assert incoming_batch.decode_attn_barrier_round_id == 10
    assert incoming_batch.decode_attn_barrier_expected_lanes == ((0, None),)
    assert scheduler._decode_attn_barrier_round_counter == 11


def test_decode_attn_a2f_dense_accepts_zero_cost_predictor_result() -> None:
    incoming_batch = _real_pdaf_batch(
        replica_id=1,
        global_id=78,
        lane=(1, None),
        layer_id=4,
        afd_stage_idx=1,
        barrier_round_id=7,
        expected_lanes=((1, None),),
        is_moe=False,
    )
    scheduler = _a2f_waiting_room_scheduler(
        {
            "per_lane_queues": defaultdict(deque),
            "expected_lane_contract": ((1, None),),
        }
    )
    scheduler._config.replica_config.model_config.is_moe = False
    scheduler._a2f_expected_lanes = [(1, None)]
    scheduler._replica_schedulers = {
        (1, None): SimpleNamespace(_decode_attn_active_cohort_states={}),
    }
    scheduler._a2f_waiting_by_layer = {}
    scheduler._decode_attn_idle_expected_lanes = {(1, None)}
    scheduler._m2n_transfer_predictor.get_transfer_info.return_value = (0, 0.0)

    events = scheduler.on_decode_attn_a2f_ready(
        1.25,
        incoming_batch,
        replica_id=1,
        dp_id=None,
        layer_id=4,
        logger=Mock(),
    )

    assert len(events) == 2
    assert scheduler._a2f_waiting_by_layer == {}
    assert scheduler._decode_attn_idle_expected_lanes == set()
    assert incoming_batch.decode_attn_barrier_round_id == 10
    assert incoming_batch.decode_attn_barrier_expected_lanes == ((1, None),)
    assert scheduler._decode_attn_barrier_round_counter == 11


def _decode_attn_return_fixture(
    *,
    completed_layer_count: int = 0,
    expected_lanes=((0, None),),
):
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=2,
    )
    request._is_prefill_complete = True
    request._current_decode_token_index = 2
    request._completed_layer_count = completed_layer_count
    request._af_roundtrip_inflight = True

    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[1],
        is_moe=True,
    )
    batch._id = 12
    batch.set_global_id(77)
    batch.afd_stage_idx = 1
    batch.decode_attn_original_replica_id = 0
    batch.decode_attn_original_replica_local_id = None
    batch.replay_decode_token_index = 2
    batch.decode_attn_barrier_round_id = 7
    batch.decode_attn_barrier_expected_lanes = expected_lanes
    batch.decode_attn_cohort_id = 9
    batch.decode_attn_cohort_request_ids = (request.id,)

    cohort_state = {
        "all_request_ids": {request.id},
        "pending_request_ids": {request.id},
        "af_phase": "ffn_inflight",
        "current_layer_id": completed_layer_count,
        "active_stage_indices": {1},
        "stage_phases": {1: "ffn_inflight"},
        "stage_current_layer_ids": {1: completed_layer_count},
    }
    replica_scheduler = SimpleNamespace(
        _decode_attn_active_cohort_states={9: cohort_state},
    )
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=SimpleNamespace(num_layers=4),
        )
    )
    scheduler._f2a_expected_lanes = [(0, None)]
    scheduler._decode_attn_idle_expected_lanes = set()
    scheduler._replica_scheduler_count = 1
    scheduler._f2a_waiting_by_round = {}
    scheduler._af_batch_queue = []
    scheduler._is_periodic_scheduling_enabled = False
    scheduler._cluster = SimpleNamespace(replicas={0: SimpleNamespace()})
    scheduler._replica_schedulers = {(0, None): replica_scheduler}

    transfer_info = _transfer_info(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        source_replica_id=0,
        source_replica_local_id=None,
        layer_id=completed_layer_count,
        afd_stage_idx=1,
        transfer_start_time=0.5,
        transfer_end_time=0.75,
    )
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler),
    )
    metrics_store = SimpleNamespace(on_m2n_transfer_end=Mock())

    request.on_m2n_transfer_complete = Mock(
        wraps=request.on_m2n_transfer_complete,
    )
    request.on_inter_cluster_transfer_end = Mock(
        wraps=request.on_inter_cluster_transfer_end,
    )
    return (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    )


def _snapshot_f2a_waiting_rooms(rooms: dict) -> tuple:
    snapshot = []
    for round_key, room in sorted(rooms.items()):
        if type(room) is not dict:
            snapshot.append(
                (
                    round_key,
                    id(room),
                    type(room).__name__,
                    repr(room),
                )
            )
            continue
        per_lane_queues = room.get("per_lane_queues")
        if isinstance(per_lane_queues, dict):
            queue_snapshot = tuple(
                (
                    lane,
                    id(queue),
                    tuple(id(batch) for batch in queue),
                )
                for lane, queue in sorted(per_lane_queues.items())
            )
        else:
            queue_snapshot = (type(per_lane_queues).__name__, repr(per_lane_queues))
        snapshot.append(
            (
                round_key,
                id(room),
                room.get("expected_lanes", "missing"),
                id(per_lane_queues),
                queue_snapshot,
            )
        )
    return tuple(snapshot)


def _snapshot_decode_attn_return_state(
    scheduler,
    batch,
    request,
    transfer_info,
    cohort_state,
) -> dict:
    return {
        "transfer_end_time": transfer_info.transfer_end_time,
        "m2n_time_ffn_to_attn": request._m2n_transfer_time_ffn_to_attn,
        "af_roundtrip_inflight": request.af_roundtrip_inflight,
        "completed_layer_count": request.completed_layer_count,
        "af_common_layer_count": batch._af_common_layer_count,
        "rooms": _snapshot_f2a_waiting_rooms(scheduler._f2a_waiting_by_round),
        "queue": tuple(id(queued) for queued in scheduler._af_batch_queue),
        "cohort_state": deepcopy(cohort_state),
    }


def _assert_decode_attn_return_state_unchanged(
    before: dict,
    scheduler,
    batch,
    request,
    transfer_info,
    metrics_store,
    cohort_state,
) -> None:
    assert transfer_info.transfer_end_time == before["transfer_end_time"]
    assert request._m2n_transfer_time_ffn_to_attn == before["m2n_time_ffn_to_attn"]
    assert request.af_roundtrip_inflight is before["af_roundtrip_inflight"]
    assert request.completed_layer_count == before["completed_layer_count"]
    assert batch._af_common_layer_count == before["af_common_layer_count"]
    assert (
        _snapshot_f2a_waiting_rooms(scheduler._f2a_waiting_by_round)
        == before["rooms"]
    )
    assert tuple(id(queued) for queued in scheduler._af_batch_queue) == before["queue"]
    assert cohort_state == before["cohort_state"]
    metrics_store.on_m2n_transfer_end.assert_not_called()
    request.on_m2n_transfer_complete.assert_not_called()
    request.on_inter_cluster_transfer_end.assert_not_called()


def _append_decode_attn_return_request(
    batch: Batch,
    cohort_state: dict,
    *,
    completed: bool = False,
    completed_layer_count: int = 0,
    roundtrip_inflight=True,
) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=0,
        num_decode_tokens=2,
    )
    request._is_prefill_complete = True
    request._current_decode_token_index = 2
    request._completed_layer_count = completed_layer_count
    request._completed = completed
    request._af_roundtrip_inflight = roundtrip_inflight
    request.on_m2n_transfer_complete = Mock(
        wraps=request.on_m2n_transfer_complete,
    )
    request.on_inter_cluster_transfer_end = Mock(
        wraps=request.on_inter_cluster_transfer_end,
    )
    batch._requests.append(request)
    batch._num_tokens.append(1)
    batch._total_num_tokens += 1
    batch.decode_attn_cohort_request_ids = tuple(
        batch_request.id for batch_request in batch.requests
    )
    cohort_state["all_request_ids"].add(request.id)
    cohort_state["pending_request_ids"].add(request.id)
    return request


def _install_decode_attn_cohort_registries_by_lane(scheduler, fixtures) -> None:
    cohort_states_by_lane = {}
    for fixture in fixtures:
        _, batch, _, _, _, _, cohort_state = fixture
        lane = (
            batch.decode_attn_original_replica_id,
            batch.decode_attn_original_replica_local_id,
        )
        cohort_states = cohort_states_by_lane.setdefault(lane, {})
        cohort_id = 9 + len(cohort_states)
        batch.decode_attn_cohort_id = cohort_id
        cohort_states[cohort_id] = cohort_state
    scheduler._replica_schedulers = {
        lane: SimpleNamespace(_decode_attn_active_cohort_states=cohort_states)
        for lane, cohort_states in cohort_states_by_lane.items()
    }


def _snapshot_decode_attn_cohort_registries_by_lane(scheduler) -> tuple:
    return tuple(
        (
            lane,
            id(replica_scheduler),
            id(replica_scheduler._decode_attn_active_cohort_states),
            tuple(
                (
                    cohort_id,
                    id(cohort_state),
                    deepcopy(cohort_state),
                )
                for cohort_id, cohort_state in sorted(
                    replica_scheduler._decode_attn_active_cohort_states.items()
                )
            ),
        )
        for lane, replica_scheduler in sorted(
            scheduler._replica_schedulers.items()
        )
    )


def _assert_decode_attn_cohort_state(
    cohort_state: dict,
    *,
    request_ids,
    phase: str,
    layer_id: int,
) -> None:
    expected_request_ids = set(request_ids)
    assert cohort_state == {
        "all_request_ids": expected_request_ids,
        "pending_request_ids": expected_request_ids,
        "af_phase": phase,
        "current_layer_id": layer_id,
        "active_stage_indices": {1},
        "stage_phases": {1: phase},
        "stage_current_layer_ids": {1: layer_id},
    }


def _snapshot_decode_attn_multi_request_return_state(
    scheduler,
    batch,
    transfer_info,
    metrics_store,
    cohort_state,
) -> dict:
    return {
        "transfer_end_time": transfer_info.transfer_end_time,
        "metrics_calls": tuple(metrics_store.on_m2n_transfer_end.call_args_list),
        "requests": tuple(
            {
                "request": request,
                "m2n_time_ffn_to_attn": request._m2n_transfer_time_ffn_to_attn,
                "roundtrip_type": type(request._af_roundtrip_inflight),
                "roundtrip_value": request._af_roundtrip_inflight,
                "completed_layer_count": request.completed_layer_count,
                "completed": request.completed,
                "transfer_complete_calls": tuple(
                    request.on_m2n_transfer_complete.call_args_list
                ),
                "transfer_end_calls": tuple(
                    request.on_inter_cluster_transfer_end.call_args_list
                ),
            }
            for request in batch.requests
        ),
        "af_common_layer_count": batch._af_common_layer_count,
        "rooms": _snapshot_f2a_waiting_rooms(scheduler._f2a_waiting_by_round),
        "queue": tuple(id(queued) for queued in scheduler._af_batch_queue),
        "cohort_state": deepcopy(cohort_state),
    }


def _assert_decode_attn_multi_request_return_state_unchanged(
    before: dict,
    scheduler,
    batch,
    transfer_info,
    metrics_store,
    cohort_state,
) -> None:
    assert transfer_info.transfer_end_time == before["transfer_end_time"]
    assert tuple(metrics_store.on_m2n_transfer_end.call_args_list) == before[
        "metrics_calls"
    ]
    assert len(batch.requests) == len(before["requests"])
    for request, request_before in zip(batch.requests, before["requests"]):
        assert request is request_before["request"]
        assert (
            request._m2n_transfer_time_ffn_to_attn
            == request_before["m2n_time_ffn_to_attn"]
        )
        assert type(request._af_roundtrip_inflight) is request_before["roundtrip_type"]
        assert request._af_roundtrip_inflight is request_before["roundtrip_value"]
        assert request.completed_layer_count == request_before["completed_layer_count"]
        assert request.completed is request_before["completed"]
        assert tuple(request.on_m2n_transfer_complete.call_args_list) == request_before[
            "transfer_complete_calls"
        ]
        assert tuple(
            request.on_inter_cluster_transfer_end.call_args_list
        ) == request_before["transfer_end_calls"]
    assert batch._af_common_layer_count == before["af_common_layer_count"]
    assert (
        _snapshot_f2a_waiting_rooms(scheduler._f2a_waiting_by_round)
        == before["rooms"]
    )
    assert tuple(id(queued) for queued in scheduler._af_batch_queue) == before["queue"]
    assert cohort_state == before["cohort_state"]


def test_decode_attn_direct_arrival_rejects_same_ids_different_batch_before_handler() -> None:
    (
        scheduler,
        transfer_batch,
        _,
        transfer_info,
        _,
        _,
        _,
    ) = _decode_attn_return_fixture()
    passed_batch = Batch(
        replica_id=0,
        requests=transfer_batch.requests,
        num_tokens=[1],
        is_moe=True,
    )
    passed_batch._id = transfer_batch.id
    passed_batch.set_global_id(transfer_batch.global_id)
    assert passed_batch is not transfer_batch
    assert passed_batch.id == transfer_batch.id
    assert passed_batch.global_id == transfer_batch.global_id
    handler = Mock(return_value=[])
    scheduler._handle_m2n_arrival_decode_attn = handler

    with pytest.raises(ValueError, match="batch.*transfer_info.batch|identity"):
        try:
            scheduler.on_m2n_arrival(1.0, passed_batch, transfer_info)
        finally:
            handler.assert_not_called()


@pytest.mark.parametrize(
    ("owner", "field_name", "invalid_value", "error_match"),
    [
        ("batch", "decode_attn_original_replica_id", None, "original_replica_id"),
        ("batch", "decode_attn_original_replica_id", True, "original_replica_id"),
        ("batch", "decode_attn_original_replica_id", -1, "original_replica_id"),
        ("batch", "decode_attn_original_replica_id", 0.0, "original_replica_id"),
        ("batch", "decode_attn_original_replica_local_id", True, "original_replica_local_id"),
        ("batch", "decode_attn_original_replica_local_id", -1, "original_replica_local_id"),
        ("batch", "decode_attn_original_replica_local_id", 0.0, "original_replica_local_id"),
        ("batch", "_global_id", True, "global_id"),
        ("batch", "_global_id", -1, "global_id"),
        ("batch", "_global_id", 77.0, "global_id"),
        ("batch", "afd_stage_idx", None, "afd_stage_idx"),
        ("batch", "afd_stage_idx", True, "afd_stage_idx"),
        ("batch", "afd_stage_idx", -1, "afd_stage_idx"),
        ("batch", "afd_stage_idx", 1.0, "afd_stage_idx"),
        ("batch", "decode_attn_barrier_round_id", True, "barrier_round_id"),
        ("batch", "decode_attn_barrier_round_id", -1, "barrier_round_id"),
        ("batch", "decode_attn_barrier_round_id", 7.0, "barrier_round_id"),
        ("batch", "replay_decode_token_index", True, "decode_token_index"),
        ("batch", "replay_decode_token_index", -1, "decode_token_index"),
        ("batch", "replay_decode_token_index", 2.0, "decode_token_index"),
        ("batch", "replay_decode_token_index", 3, "decode_token_index"),
        ("batch", "decode_attn_cohort_id", True, "cohort_id"),
        ("batch", "decode_attn_cohort_id", -1, "cohort_id"),
        ("batch", "decode_attn_cohort_id", 9.0, "cohort_id"),
        ("batch", "decode_attn_cohort_id", "9", "cohort_id"),
        ("batch", "decode_attn_cohort_id", object(), "cohort_id"),
        ("transfer", "source_replica_id", True, "source_replica_id"),
        ("transfer", "source_replica_id", -1, "source_replica_id"),
        ("transfer", "source_replica_local_id", True, "source_replica_local_id"),
        ("transfer", "source_replica_local_id", -1, "source_replica_local_id"),
        ("transfer", "layer_id", None, "layer_id"),
        ("transfer", "layer_id", True, "layer_id"),
        ("transfer", "layer_id", -1, "layer_id"),
        ("transfer", "layer_id", 0.0, "layer_id"),
        ("transfer", "afd_stage_idx", None, "afd_stage_idx"),
        ("transfer", "afd_stage_idx", True, "afd_stage_idx"),
        ("transfer", "afd_stage_idx", -1, "afd_stage_idx"),
        ("transfer", "afd_stage_idx", 1.0, "afd_stage_idx"),
    ],
)
def test_decode_attn_transfer_end_rejects_malformed_receipt_before_all_side_effects(
    owner: str,
    field_name: str,
    invalid_value,
    error_match: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    target = batch if owner == "batch" else transfer_info
    setattr(target, field_name, invalid_value)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((TypeError, ValueError), match=error_match):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    ("binding_case", "error_match"),
    [
        pytest.param(
            "missing_cohort",
            "cohort_id|active cohort",
            id="missing-cohort",
        ),
        pytest.param(
            "unknown_cohort",
            "cohort|active|unknown|not found",
            id="unknown-cohort",
        ),
        pytest.param(
            "request_not_pending",
            "pending|cohort|request",
            id="request-not-pending",
        ),
        pytest.param(
            "cohort_request_ids_mismatch",
            "cohort|all_request_ids|request IDs",
            id="cohort-request-ids-mismatch",
        ),
        pytest.param(
            "inactive_stage",
            "active stage|afd_stage_idx|cohort",
            id="inactive-stage",
        ),
        pytest.param(
            "missing_active_stage_indices",
            "active stage|active_stage_indices|missing",
            id="missing-active-stage-indices",
        ),
    ],
)
def test_decode_attn_transfer_end_rejects_receipt_not_bound_to_active_cohort_before_all_side_effects(
    binding_case: str,
    error_match: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()

    if binding_case == "missing_cohort":
        batch.decode_attn_cohort_id = None
    elif binding_case == "unknown_cohort":
        batch.decode_attn_cohort_id = 10
    elif binding_case == "request_not_pending":
        sibling_request_id = request.id + 10_000
        cohort_state["all_request_ids"] = {request.id, sibling_request_id}
        cohort_state["pending_request_ids"] = {sibling_request_id}
        batch.decode_attn_cohort_request_ids = (
            request.id,
            sibling_request_id,
        )
    elif binding_case == "cohort_request_ids_mismatch":
        batch.decode_attn_cohort_request_ids = (request.id + 10_000,)
    elif binding_case == "inactive_stage":
        cohort_state["active_stage_indices"] = {0}
    elif binding_case == "missing_active_stage_indices":
        cohort_state.pop("active_stage_indices")
    else:
        raise AssertionError(f"Unhandled cohort binding case: {binding_case}")

    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((RuntimeError, ValueError), match=error_match):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_rejects_non_request_incoming_receipt_before_all_side_effects() -> None:
    incoming_fixture = _decode_attn_return_fixture(
        expected_lanes=((0, None), (0, 1)),
    )
    queued_fixture = _decode_attn_return_fixture(
        expected_lanes=((0, None), (0, 1)),
    )
    scheduler = incoming_fixture[0]
    scheduler._f2a_expected_lanes = [(0, None), (0, 1)]
    scheduler._replica_scheduler_count = 2
    global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler),
    )
    for fixture, dp_id in ((incoming_fixture, 0), (queued_fixture, 1)):
        _, fixture_batch, _, fixture_transfer, _, _, _ = fixture
        fixture_batch.decode_attn_original_replica_local_id = dp_id
        fixture_transfer.source_replica_local_id = dp_id
    _install_decode_attn_cohort_registries_by_lane(
        scheduler,
        (incoming_fixture, queued_fixture),
    )
    assert M2NTransferEndEvent(0.9, queued_fixture[3]).handle_event(
        global_scheduler,
        queued_fixture[5],
    ) == []

    _, batch, original_request, transfer_info, _, metrics_store, _ = (
        incoming_fixture
    )
    fake_request = SimpleNamespace(
        id=original_request.id,
        num_prefill_tokens=original_request.num_prefill_tokens,
        num_decode_tokens=original_request.num_decode_tokens,
        num_processed_tokens=original_request.num_processed_tokens,
        completed=False,
        af_roundtrip_inflight=True,
        _af_roundtrip_inflight=True,
        completed_layer_count=original_request.completed_layer_count,
        _completed_layer_count=original_request._completed_layer_count,
        current_decode_token_index=original_request.current_decode_token_index,
        _current_decode_token_index=original_request._current_decode_token_index,
        _m2n_transfer_time_ffn_to_attn=(
            original_request._m2n_transfer_time_ffn_to_attn
        ),
    )

    def complete_m2n_transfer(transfer_time: float, is_attn_to_ffn: bool) -> None:
        assert is_attn_to_ffn is False
        fake_request._m2n_transfer_time_ffn_to_attn += transfer_time
        fake_request.af_roundtrip_inflight = False
        fake_request._af_roundtrip_inflight = False

    def increment_layer() -> None:
        fake_request.completed_layer_count += 1
        fake_request._completed_layer_count += 1

    fake_request.on_m2n_transfer_complete = Mock(
        side_effect=complete_m2n_transfer,
    )
    fake_request.on_inter_cluster_transfer_end = Mock()
    fake_request.mb_on_step_layer_count_increment = Mock(
        side_effect=increment_layer,
    )
    batch._requests[0] = fake_request

    fixtures_in_assertion_order = (incoming_fixture, queued_fixture)
    before_by_fixture = tuple(
        _snapshot_decode_attn_multi_request_return_state(
            scheduler,
            fixture_batch,
            fixture_transfer_info,
            fixture_metrics_store,
            fixture_cohort_state,
        )
        for (
            _,
            fixture_batch,
            _,
            fixture_transfer_info,
            _,
            fixture_metrics_store,
            fixture_cohort_state,
        ) in fixtures_in_assertion_order
    )
    cohort_registries_before = _snapshot_decode_attn_cohort_registries_by_lane(
        scheduler
    )
    returned_events = None

    with pytest.raises(
        ValueError,
        match="incoming|receipt|exact Request|request type",
    ):
        try:
            returned_events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            for fixture, fixture_before in zip(
                fixtures_in_assertion_order,
                before_by_fixture,
            ):
                (
                    _,
                    fixture_batch,
                    _,
                    fixture_transfer_info,
                    _,
                    fixture_metrics_store,
                    fixture_cohort_state,
                ) = fixture
                _assert_decode_attn_multi_request_return_state_unchanged(
                    fixture_before,
                    scheduler,
                    fixture_batch,
                    fixture_transfer_info,
                    fixture_metrics_store,
                    fixture_cohort_state,
                )
            assert (
                _snapshot_decode_attn_cohort_registries_by_lane(scheduler)
                == cohort_registries_before
            )
            fake_request.mb_on_step_layer_count_increment.assert_not_called()
            assert returned_events is None


def test_decode_attn_transfer_end_accepts_active_cohort_microbatch_subset() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    sibling_request_id = request.id + 10_000
    cohort_state["all_request_ids"] = {request.id, sibling_request_id}
    cohort_state["pending_request_ids"] = {request.id}
    cohort_state["active_stage_indices"] = {1, 2}
    cohort_state["stage_phases"] = {1: "ffn_inflight", 2: "ffn_inflight"}
    cohort_state["stage_current_layer_ids"] = {1: 0, 2: 0}
    batch.decode_attn_cohort_request_ids = (request.id, sibling_request_id)

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert scheduler._af_batch_queue == [batch]
    assert cohort_state["all_request_ids"] == {request.id, sibling_request_id}
    assert cohort_state["pending_request_ids"] == {request.id}
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
def test_decode_attn_transfer_end_rejects_false_roundtrip_at_any_request_position_before_commit(
    invalid_index: int,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    _append_decode_attn_return_request(batch, cohort_state)
    _append_decode_attn_return_request(batch, cohort_state)
    batch.requests[invalid_index]._af_roundtrip_inflight = False
    before = _snapshot_decode_attn_multi_request_return_state(
        scheduler,
        batch,
        transfer_info,
        metrics_store,
        cohort_state,
    )

    with pytest.raises(ValueError, match="roundtrip|F-to-A|F->A"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_multi_request_return_state_unchanged(
                before,
                scheduler,
                batch,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize("roundtrip_state", [False, 1, 1.0, None])
def test_decode_attn_transfer_end_rejects_invalid_active_request_roundtrip_before_mutation(
    roundtrip_state,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    request._af_roundtrip_inflight = roundtrip_state
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="roundtrip|inflight|exact bool",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_rejects_invalid_completed_request_roundtrip_before_mutation() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    stale_request = Request(0.0, 0, 2)
    stale_request._is_prefill_complete = True
    stale_request._completed = True
    stale_request._completed_layer_count = 3
    stale_request._af_roundtrip_inflight = False
    batch._requests.append(stale_request)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )
    stale_before = (
        stale_request.af_roundtrip_inflight,
        stale_request._m2n_transfer_time_ffn_to_attn,
        stale_request.completed_layer_count,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="roundtrip|inflight|exact bool",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )
            assert (
                stale_request.af_roundtrip_inflight,
                stale_request._m2n_transfer_time_ffn_to_attn,
                stale_request.completed_layer_count,
            ) == stale_before


def test_decode_attn_transfer_end_rejects_completed_request_outside_active_cohort_before_mutation() -> None:
    (
        scheduler,
        batch,
        _,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    foreign_request = Request(0.0, 0, 2)
    foreign_request._is_prefill_complete = True
    foreign_request._completed = True
    foreign_request._completed_layer_count = 3
    foreign_request._af_roundtrip_inflight = True
    foreign_request.on_m2n_transfer_complete = Mock(
        wraps=foreign_request.on_m2n_transfer_complete,
    )
    foreign_request.on_inter_cluster_transfer_end = Mock(
        wraps=foreign_request.on_inter_cluster_transfer_end,
    )
    batch._requests.append(foreign_request)
    before = _snapshot_decode_attn_multi_request_return_state(
        scheduler,
        batch,
        transfer_info,
        metrics_store,
        cohort_state,
    )

    with pytest.raises(ValueError, match="cohort|request IDs|all_request_ids"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_multi_request_return_state_unchanged(
                before,
                scheduler,
                batch,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "invalid_roundtrip_state",
    [
        pytest.param(None, id="none"),
        pytest.param(0, id="zero-int"),
        pytest.param(1, id="one-int"),
        pytest.param(0.0, id="zero-float"),
        pytest.param(1.0, id="one-float"),
        pytest.param("true", id="string"),
        pytest.param(object(), id="object"),
    ],
)
def test_decode_attn_transfer_end_rejects_non_exact_roundtrip_state_before_commit(
    invalid_roundtrip_state,
) -> None:
    (
        scheduler,
        batch,
        _,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    invalid_request = _append_decode_attn_return_request(
        batch,
        cohort_state,
        roundtrip_inflight=invalid_roundtrip_state,
    )
    before = _snapshot_decode_attn_multi_request_return_state(
        scheduler,
        batch,
        transfer_info,
        metrics_store,
        cohort_state,
    )

    with pytest.raises((TypeError, ValueError), match="exact bool|roundtrip|F-to-A|F->A"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert invalid_request in batch.requests
            _assert_decode_attn_multi_request_return_state_unchanged(
                before,
                scheduler,
                batch,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize("contract_source", ["metadata", "existing_room"])
def test_decode_attn_transfer_end_rejects_lane_contract_outside_scheduler_topology_before_mutation(
    contract_source: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (0, 9)))
    round_key = (0, 1, 1, 7)
    if contract_source == "existing_room":
        batch.decode_attn_barrier_expected_lanes = ()
        scheduler._f2a_waiting_by_round[round_key] = {
            "per_lane_queues": defaultdict(deque),
            "expected_lanes": ((0, None), (0, 9)),
        }
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="topology|outside|expected lanes",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_preflight_accepts_ordered_lane_subset_contract() -> None:
    scheduler, batch, _, transfer_info, _, _, _ = _decode_attn_return_fixture(
        expected_lanes=((1, None), (0, None)),
    )
    scheduler._f2a_expected_lanes = [(0, None), (1, None), (2, None)]
    scheduler._cluster.replicas = {
        0: SimpleNamespace(),
        1: SimpleNamespace(),
        2: SimpleNamespace(),
    }

    scheduler.preflight_m2n_arrival(batch, transfer_info)

    assert batch.decode_attn_barrier_expected_lanes == ((1, None), (0, None))


@pytest.mark.skip(reason="Cross-Replica full-stage returns no longer form a legacy local-DP lane barrier")
def test_decode_attn_ordered_lane_subset_controls_full_release_order() -> None:
    fixtures = [
        _decode_attn_return_fixture(expected_lanes=((1, None), (0, None)))
        for _ in range(2)
    ]
    scheduler = fixtures[0][0]
    scheduler._f2a_expected_lanes = [(0, None), (1, None), (2, None)]
    scheduler._replica_scheduler_count = 3
    scheduler._cluster.replicas = {
        0: SimpleNamespace(),
        1: SimpleNamespace(),
        2: SimpleNamespace(),
    }

    for source_replica_id, fixture in zip((1, 0), fixtures):
        _, batch, _, transfer_info, _, _, _ = fixture
        batch.decode_attn_original_replica_id = source_replica_id
        batch.decode_attn_original_replica_local_id = None
        transfer_info.source_replica_id = source_replica_id
        transfer_info.source_replica_local_id = None
    _install_decode_attn_cohort_registries_by_lane(scheduler, fixtures)

    returned_events = []
    for fixture in fixtures:
        _, _, _, transfer_info, _, metrics_store, _ = fixture
        shared_global_scheduler = SimpleNamespace(
            get_cluster_scheduler=Mock(return_value=scheduler),
        )
        returned_events.append(
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                shared_global_scheduler,
                metrics_store,
            )
        )

    lane_0_batch = fixtures[0][1]
    lane_1_batch = fixtures[1][1]
    assert returned_events[0] == []
    assert len(returned_events[1]) == 1
    assert isinstance(returned_events[1][0], ClusterScheduleEvent)
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [lane_1_batch, lane_0_batch]


@pytest.mark.parametrize(
    "invalid_roundtrip_state",
    [
        pytest.param(False, id="false"),
        pytest.param(None, id="none"),
        pytest.param(0, id="zero-int"),
        pytest.param(1, id="one-int"),
    ],
)
def test_decode_attn_transfer_end_validates_completed_request_roundtrip_before_commit(
    invalid_roundtrip_state,
) -> None:
    (
        scheduler,
        batch,
        _,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    completed_request = _append_decode_attn_return_request(
        batch,
        cohort_state,
        completed=True,
        completed_layer_count=3,
        roundtrip_inflight=invalid_roundtrip_state,
    )
    before = _snapshot_decode_attn_multi_request_return_state(
        scheduler,
        batch,
        transfer_info,
        metrics_store,
        cohort_state,
    )

    with pytest.raises((TypeError, ValueError), match="exact bool|roundtrip|F-to-A|F->A"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert completed_request.completed is True
            _assert_decode_attn_multi_request_return_state_unchanged(
                before,
                scheduler,
                batch,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_preserves_legal_multi_active_request_contract() -> None:
    (
        scheduler,
        batch,
        first_request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    second_request = _append_decode_attn_return_request(batch, cohort_state)

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    metrics_store.on_m2n_transfer_end.assert_called_once()
    for request in (first_request, second_request):
        assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
        assert request.af_roundtrip_inflight is False
        assert request.completed_layer_count == 1
        request.on_m2n_transfer_complete.assert_called_once_with(0.5, False)
        request.on_inter_cluster_transfer_end.assert_called_once()
    assert batch._af_common_layer_count == 1
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(first_request.id, second_request.id),
        phase="local_attn",
        layer_id=1,
    )
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)
    assert events[0].time == pytest.approx(1.0)


def test_decode_attn_transfer_end_preserves_legal_active_and_completed_request_contract() -> None:
    (
        scheduler,
        batch,
        active_request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    completed_request = _append_decode_attn_return_request(
        batch,
        cohort_state,
        completed=True,
        completed_layer_count=3,
    )
    completed_request._m2n_transfer_time_ffn_to_attn = 7.0

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert active_request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert active_request.af_roundtrip_inflight is False
    assert active_request.completed_layer_count == 1
    active_request.on_m2n_transfer_complete.assert_called_once_with(0.5, False)
    active_request.on_inter_cluster_transfer_end.assert_called_once()
    assert completed_request._m2n_transfer_time_ffn_to_attn == pytest.approx(7.5)
    assert completed_request.af_roundtrip_inflight is False
    assert completed_request.completed_layer_count == 3
    completed_request.on_m2n_transfer_complete.assert_called_once_with(0.5, False)
    completed_request.on_inter_cluster_transfer_end.assert_called_once()
    assert batch._af_common_layer_count == 1
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(active_request.id, completed_request.id),
        phase="local_attn",
        layer_id=1,
    )
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)
    assert events[0].time == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("expected_lanes", "error_match"),
    [
        (((0, None), (0, None)), "duplicate"),
        (((0, True),), "dp_id|expected lane"),
        (((True, 0),), "replica_id|expected lane"),
        (((0, -1),), "dp_id|expected lane"),
        (((-1, 0),), "replica_id|expected lane"),
        (([0, 0],), "2-tuples|expected lane"),
    ],
)
def test_decode_attn_transfer_end_rejects_malformed_lane_contract_without_side_effects(
    expected_lanes,
    error_match: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=expected_lanes)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((TypeError, ValueError), match=error_match):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    ("field_name", "mismatched_value", "error_match"),
    [
        ("layer_id", 1, "layer_id|layer"),
        ("afd_stage_idx", 2, "afd_stage_idx|stage"),
        ("source_replica_local_id", 1, "source.*lane|source_replica_local_id"),
    ],
)
def test_decode_attn_transfer_end_rejects_batch_transfer_mismatch_without_side_effects(
    field_name: str,
    mismatched_value: int,
    error_match: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    setattr(transfer_info, field_name, mismatched_value)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(ValueError, match=error_match):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_rejects_unexpected_lane_without_creating_room() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    batch.decode_attn_original_replica_local_id = 9
    transfer_info.source_replica_local_id = 9
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(ValueError, match="lane|topology"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_private_handler_rejects_malformed_receipt_before_mutation() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        _,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    batch.decode_attn_original_replica_id = None
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((TypeError, ValueError), match="original_replica_id"):
        try:
            scheduler._handle_m2n_arrival_decode_attn(
                1.0,
                batch,
                transfer_info,
                Mock(),
            )
        finally:
            assert transfer_info.transfer_end_time == before["transfer_end_time"]
            assert request.completed_layer_count == before["completed_layer_count"]
            assert batch._af_common_layer_count == before["af_common_layer_count"]
            assert (
                _snapshot_f2a_waiting_rooms(scheduler._f2a_waiting_by_round)
                == before["rooms"]
            )
            assert tuple(scheduler._af_batch_queue) == ()
            assert cohort_state == before["cohort_state"]
            metrics_store.on_m2n_transfer_end.assert_not_called()
            request.on_m2n_transfer_complete.assert_not_called()
            request.on_inter_cluster_transfer_end.assert_not_called()


def test_decode_attn_transfer_end_rejects_mixed_request_layers_without_side_effects() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    second_request = Request(0.0, 0, 2)
    second_request._is_prefill_complete = True
    second_request._current_decode_token_index = 2
    second_request._completed_layer_count = 1
    second_request._af_roundtrip_inflight = True
    batch._requests.append(second_request)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((AssertionError, ValueError), match="layer|consistent"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )
            assert second_request.completed_layer_count == 1


def test_decode_attn_transfer_end_rejects_mixed_request_tokens_without_replay_identity() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    batch.replay_decode_token_index = None
    second_request = Request(0.0, 0, 2)
    second_request._is_prefill_complete = True
    second_request._current_decode_token_index = 3
    second_request._completed_layer_count = 0
    second_request._af_roundtrip_inflight = True
    batch._requests.append(second_request)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(ValueError, match="decode token|decode_token|mixed"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )
            assert second_request.completed_layer_count == 0


def test_decode_attn_transfer_end_rejects_existing_round_contract_before_mutation() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    round_key = (0, 1, 1, 7)
    existing_room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lanes": ((0, None), (0, 1)),
    }
    scheduler._f2a_expected_lanes = [(0, None), (0, 1)]
    scheduler._replica_scheduler_count = 2
    scheduler._f2a_waiting_by_round[round_key] = existing_room
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(ValueError, match="expected lanes|contract"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert scheduler._f2a_waiting_by_round[round_key] is existing_room
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "corrupt_room",
    [
        {"per_lane_queues": defaultdict(deque)},
        {"per_lane_queues": [], "expected_lanes": ((0, None),)},
        {"per_lane_queues": {(0, None): []}, "expected_lanes": ((0, None),)},
    ],
)
def test_decode_attn_transfer_end_rejects_corrupt_existing_room_before_mutation(
    corrupt_room: dict,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    round_key = (0, 1, 1, 7)
    scheduler._f2a_waiting_by_round[round_key] = corrupt_room
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises((RuntimeError, TypeError, ValueError), match="waiting room|expected lanes|queue"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert scheduler._f2a_waiting_by_round[round_key] is corrupt_room
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_rejects_corrupt_queued_batch_before_mutation() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (0, 1)))
    batch.decode_attn_original_replica_local_id = 1
    transfer_info.source_replica_local_id = 1
    scheduler._f2a_expected_lanes = [(0, None), (0, 1)]
    scheduler._replica_scheduler_count = 2
    scheduler._replica_schedulers[(0, 1)] = (
        scheduler._replica_schedulers[(0, None)]
    )
    round_key = (0, 1, 1, 7)
    corrupt_room = {
        "per_lane_queues": defaultdict(
            deque,
            {(0, None): deque([object()])},
        ),
        "expected_lanes": ((0, None), (0, 1)),
    }
    scheduler._f2a_waiting_by_round[round_key] = corrupt_room
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="queued|waiting room|Batch",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert scheduler._f2a_waiting_by_round[round_key] is corrupt_room
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    ("corruption", "invalid_value", "error_match"),
    [
        ("cohort_id", True, "cohort_id"),
        ("cohort_id", -1, "cohort_id"),
        ("cohort_id", 9.0, "cohort_id"),
        ("cohort_id", "9", "cohort_id"),
        ("cohort_id", object(), "cohort_id"),
        (
            "cohort_id",
            None,
            "cohort_id|active cohort",
        ),
        (
            "cohort_id",
            999_999,
            "cohort|active|unknown|not found",
        ),
        (
            "cohort_request_ids_mismatch",
            None,
            "cohort|all_request_ids|request IDs",
        ),
        (
            "foreign_completed_request",
            None,
            "cohort|all_request_ids|request IDs|outside",
        ),
        (
            "active_request_not_pending",
            None,
            "pending|cohort|request",
        ),
        (
            "duplicate_request_ids",
            None,
            "duplicate|unique|request IDs",
        ),
        (
            "inactive_stage",
            None,
            "active stage|afd_stage_idx|cohort",
        ),
        (
            "non_request_object",
            None,
            "queued request|exact Request|request type",
        ),
        ("roundtrip", True, "roundtrip|inflight"),
        ("roundtrip", 0, "roundtrip|inflight|exact bool"),
    ],
)
@pytest.mark.skip(reason="The retired attention-DP multi-lane queue contract is no longer supported")
def test_decode_attn_transfer_end_rejects_corrupt_real_queued_batch_before_mutation(
    corruption: str,
    invalid_value,
    error_match: str,
) -> None:
    lane0_fixture = _decode_attn_return_fixture(
        expected_lanes=((0, None), (1, None)),
    )
    lane1_fixture = _decode_attn_return_fixture(
        expected_lanes=((0, None), (1, None)),
    )
    scheduler = lane0_fixture[0]
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]
    scheduler._replica_scheduler_count = 2
    scheduler._cluster.replicas = {0: SimpleNamespace(), 1: SimpleNamespace()}
    shared_global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler),
    )
    for fixture, source_replica_id in ((lane0_fixture, 0), (lane1_fixture, 1)):
        _, fixture_batch, _, fixture_transfer, _, _, _ = fixture
        fixture_batch.decode_attn_original_replica_id = source_replica_id
        fixture_batch.decode_attn_original_replica_local_id = None
        fixture_transfer.source_replica_id = source_replica_id
        fixture_transfer.source_replica_local_id = None
    _install_decode_attn_cohort_registries_by_lane(
        scheduler,
        (lane0_fixture, lane1_fixture),
    )

    assert M2NTransferEndEvent(1.0, lane0_fixture[3]).handle_event(
        shared_global_scheduler,
        lane0_fixture[5],
    ) == []
    queued_batch = lane0_fixture[1]
    queued_request = lane0_fixture[2]
    queued_cohort_state = lane0_fixture[6]
    if corruption == "cohort_id":
        queued_batch.decode_attn_cohort_id = invalid_value
    elif corruption == "cohort_request_ids_mismatch":
        queued_batch.decode_attn_cohort_request_ids = (
            queued_request.id + 10_000,
        )
    elif corruption == "foreign_completed_request":
        foreign_request = _append_decode_attn_return_request(
            queued_batch,
            queued_cohort_state,
            completed=True,
            completed_layer_count=3,
            roundtrip_inflight=False,
        )
        queued_cohort_state["all_request_ids"].remove(foreign_request.id)
        queued_cohort_state["pending_request_ids"].remove(foreign_request.id)
        queued_batch.decode_attn_cohort_request_ids = tuple(
            sorted(queued_cohort_state["all_request_ids"])
        )
    elif corruption == "active_request_not_pending":
        queued_cohort_state["pending_request_ids"].remove(queued_request.id)
    elif corruption == "duplicate_request_ids":
        queued_batch._requests.append(queued_request)
        queued_batch._num_tokens.append(1)
        queued_batch._total_num_tokens += 1
    elif corruption == "inactive_stage":
        queued_cohort_state["active_stage_indices"] = {0}
    elif corruption == "non_request_object":
        queued_batch._requests[0] = SimpleNamespace(
            id=queued_request.id,
            completed=False,
            af_roundtrip_inflight=False,
            _af_roundtrip_inflight=False,
            completed_layer_count=queued_request.completed_layer_count,
            _completed_layer_count=queued_request._completed_layer_count,
            current_decode_token_index=queued_request.current_decode_token_index,
            _current_decode_token_index=(
                queued_request._current_decode_token_index
            ),
            _m2n_transfer_time_ffn_to_attn=(
                queued_request._m2n_transfer_time_ffn_to_attn
            ),
            on_m2n_transfer_complete=Mock(),
            on_inter_cluster_transfer_end=Mock(),
        )
    elif corruption == "roundtrip":
        queued_request._af_roundtrip_inflight = invalid_value
    else:
        raise AssertionError(f"Unhandled queued corruption: {corruption}")

    transfer_info = lane1_fixture[3]
    metrics_store = lane1_fixture[5]
    fixtures_in_assertion_order = (lane1_fixture, lane0_fixture)
    before_by_fixture = tuple(
        _snapshot_decode_attn_multi_request_return_state(
            scheduler,
            fixture_batch,
            fixture_transfer_info,
            fixture_metrics_store,
            fixture_cohort_state,
        )
        for (
            _,
            fixture_batch,
            _,
            fixture_transfer_info,
            _,
            fixture_metrics_store,
            fixture_cohort_state,
        ) in fixtures_in_assertion_order
    )
    cohort_registries_before = _snapshot_decode_attn_cohort_registries_by_lane(
        scheduler
    )

    expected_exception = (
        ValueError
        if corruption == "non_request_object"
        else (RuntimeError, TypeError, ValueError)
    )
    with pytest.raises(
        expected_exception,
        match=error_match,
    ):
        try:
            M2NTransferEndEvent(2.0, transfer_info).handle_event(
                shared_global_scheduler,
                metrics_store,
            )
        finally:
            for fixture, fixture_before in zip(
                fixtures_in_assertion_order,
                before_by_fixture,
            ):
                (
                    _,
                    fixture_batch,
                    _,
                    fixture_transfer_info,
                    _,
                    fixture_metrics_store,
                    fixture_cohort_state,
                ) = fixture
                _assert_decode_attn_multi_request_return_state_unchanged(
                    fixture_before,
                    scheduler,
                    fixture_batch,
                    fixture_transfer_info,
                    fixture_metrics_store,
                    fixture_cohort_state,
                )
            assert (
                _snapshot_decode_attn_cohort_registries_by_lane(scheduler)
                == cohort_registries_before
            )


@pytest.mark.parametrize(
    "scheduler_lanes",
    [
        [(0.0, 0)],
        [(0, False)],
        [("0", 0)],
    ],
)
def test_decode_attn_transfer_end_rejects_coercible_scheduler_lane_topology(
    scheduler_lanes,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    scheduler._f2a_expected_lanes = scheduler_lanes
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="scheduler lane topology|exact",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_rejects_foreign_replica_lane_outside_global_topology_before_mutation() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (1, 999)))
    scheduler._cluster = SimpleNamespace(
        replicas={0: SimpleNamespace(), 1: SimpleNamespace()},
    )
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="expected lanes.*topology|outside.*topology|lane.*topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_end_preserves_legal_multi_replica_lane_metadata() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (1, None)))
    scheduler._cluster = SimpleNamespace(
        replicas={0: SimpleNamespace(), 1: SimpleNamespace()},
    )
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert batch.decode_attn_barrier_expected_lanes == ((0, None), (1, None))
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="local_attn",
        layer_id=1,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


@pytest.mark.parametrize("empty_configured_lanes", [None, [], ()])
def test_decode_attn_transfer_end_rejects_unknown_replica_during_empty_topology_fallback_before_mutation(
    empty_configured_lanes,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (999, 0)))
    scheduler._cluster = SimpleNamespace(
        replicas={0: SimpleNamespace(), 1: SimpleNamespace()},
    )
    scheduler._f2a_expected_lanes = empty_configured_lanes
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="replica.*inventory|unknown.*replica|replica.*topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize("empty_configured_lanes", [None, [], ()])
def test_decode_attn_transfer_end_preserves_known_multi_replica_metadata_during_empty_topology_fallback(
    empty_configured_lanes,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(expected_lanes=((0, None), (1, None)))
    scheduler._cluster = SimpleNamespace(
        replicas={0: SimpleNamespace(), 1: SimpleNamespace()},
    )
    scheduler._f2a_expected_lanes = empty_configured_lanes

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert batch.decode_attn_barrier_expected_lanes == ((0, None), (1, None))
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="local_attn",
        layer_id=1,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


@pytest.mark.parametrize("invalid_configured_lanes", [False, 0, "", {}, set()])
def test_decode_attn_transfer_end_rejects_falsey_non_container_configured_topology_before_mutation(
    invalid_configured_lanes,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    scheduler._f2a_expected_lanes = invalid_configured_lanes
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="scheduler lane topology|exact list or tuple|topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize("empty_configured_lanes", [None, [], ()])
def test_decode_attn_transfer_end_preserves_explicit_empty_configured_topology_fallback(
    empty_configured_lanes,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    scheduler._f2a_expected_lanes = empty_configured_lanes

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    assert cohort_state["af_phase"] == "local_attn"
    assert cohort_state["current_layer_id"] == 1
    assert cohort_state["all_request_ids"] == {request.id}
    assert cohort_state["pending_request_ids"] == {request.id}
    assert cohort_state["active_stage_indices"] == {1}
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


@pytest.mark.parametrize(
    ("attribute_name", "invalid_value"),
    [
        ("_decode_attn_active_serving_wave_expected_lanes", False),
        ("_decode_attn_active_serving_wave_expected_lanes", 0),
        ("_decode_attn_active_serving_wave_expected_lanes", ""),
        ("_decode_attn_active_serving_wave_expected_lanes", {}),
        ("_decode_attn_active_serving_wave_expected_lanes", set()),
        ("_decode_attn_active_serving_wave_expected_lanes", None),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", False),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", 0),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", ""),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", []),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", ()),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", set()),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", None),
    ],
)
def test_decode_attn_transfer_end_rejects_falsey_invalid_active_wave_topology_before_mutation(
    attribute_name: str,
    invalid_value,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    setattr(scheduler, attribute_name, invalid_value)
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="active wave.*topology|exact",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "invalid_source",
    [
        "expected_lanes_masked_by_request_map",
        "request_map_masked_by_expected_lanes",
    ],
)
def test_decode_attn_transfer_end_validates_all_active_wave_sources_before_precedence(
    invalid_source: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    if invalid_source == "expected_lanes_masked_by_request_map":
        scheduler._decode_attn_active_serving_wave_expected_lanes = False
        scheduler._decode_attn_active_serving_wave_request_ids_by_lane = {
            (0, None): {request.id},
        }
    elif invalid_source == "request_map_masked_by_expected_lanes":
        scheduler._decode_attn_active_serving_wave_expected_lanes = ((0, None),)
        scheduler._decode_attn_active_serving_wave_request_ids_by_lane = False
    else:
        raise AssertionError(f"Unhandled active-wave source: {invalid_source}")
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="active wave.*topology|exact",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    ("attribute_name", "empty_value"),
    [
        ("_decode_attn_active_serving_wave_expected_lanes", []),
        ("_decode_attn_active_serving_wave_expected_lanes", ()),
        ("_decode_attn_active_serving_wave_request_ids_by_lane", {}),
    ],
)
def test_decode_attn_transfer_end_preserves_exact_empty_active_wave_topology(
    attribute_name: str,
    empty_value,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    setattr(scheduler, attribute_name, empty_value)

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="local_attn",
        layer_id=1,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


@pytest.mark.parametrize(
    "invalid_cohort_states",
    [False, 0, "", [], (), set()],
)
def test_decode_attn_transfer_end_rejects_falsey_invalid_legacy_cohort_states_before_mutation(
    invalid_cohort_states,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    scheduler._replica_schedulers = {
        (0, None): SimpleNamespace(
            _decode_attn_active_cohort_states=invalid_cohort_states,
        ),
    }
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(RuntimeError, match="active cohort states.*exact dict"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "invalid_cohort_state",
    [False, 0, "", [], (), set()],
)
def test_decode_attn_transfer_end_rejects_falsey_invalid_legacy_cohort_state_before_mutation(
    invalid_cohort_state,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    scheduler._replica_schedulers = {
        (0, None): SimpleNamespace(
            _decode_attn_active_cohort_states={9: invalid_cohort_state},
        ),
    }
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(RuntimeError, match="active cohort state.*exact dict"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "legacy_state_mode",
    ["missing", "exact_empty_dict", "valid_state"],
)
def test_decode_attn_stage_slot_reader_accepts_legal_legacy_cohort_states(
    legacy_state_mode: str,
) -> None:
    scheduler = _decode_attn_return_fixture()[0]
    legacy_scheduler = SimpleNamespace()
    if legacy_state_mode == "exact_empty_dict":
        legacy_scheduler._decode_attn_active_cohort_states = {}
    elif legacy_state_mode == "valid_state":
        legacy_scheduler._decode_attn_active_cohort_states = {
            9: {
                "afd_stage_idx": 1,
                "af_phase": "ffn_inflight",
                "current_layer_id": 0,
            },
        }
    scheduler._replica_schedulers = {(0, None): legacy_scheduler}

    active_lanes = scheduler._get_decode_attn_stage_slot_active_lanes(
        1,
        replica_id=0,
        phase="ffn_inflight",
        layer_id=0,
    )

    expected_lanes = [(0, None)] if legacy_state_mode == "valid_state" else []
    assert active_lanes == expected_lanes
    if legacy_state_mode == "missing":
        assert not hasattr(
            legacy_scheduler,
            "_decode_attn_active_cohort_states",
        )


@pytest.mark.parametrize(
    "invalid_source",
    [
        "idle_lane_float",
        "idle_lane_negative",
        "replica_scheduler_mapping_type",
        "replica_scheduler_lane_bool",
        "active_stage_slot_bool",
        "active_wave_lane_float",
        "active_wave_lane_negative",
        "active_wave_request_lane_bool",
        "configured_lane_negative",
        "replica_scheduler_count_bool",
        "replica_scheduler_count_float",
        "replica_scheduler_count_zero",
    ],
)
def test_decode_attn_transfer_end_rejects_each_invalid_topology_source(
    invalid_source: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()

    if invalid_source == "idle_lane_float":
        scheduler._decode_attn_idle_expected_lanes = {(0.0, 0)}
    elif invalid_source == "idle_lane_negative":
        scheduler._decode_attn_idle_expected_lanes = {(-1, 0)}
    elif invalid_source == "replica_scheduler_mapping_type":
        scheduler._replica_schedulers = []
    elif invalid_source == "replica_scheduler_lane_bool":
        scheduler._replica_schedulers = {
            (0, False): SimpleNamespace(),
        }
    elif invalid_source == "active_stage_slot_bool":
        scheduler._replica_schedulers = {
            (0, None): SimpleNamespace(
                get_decode_attn_active_stage_slots=Mock(return_value={True}),
            ),
        }
    elif invalid_source == "active_wave_lane_float":
        scheduler._decode_attn_active_serving_wave_expected_lanes = (
            (0.0, 0),
        )
    elif invalid_source == "active_wave_lane_negative":
        scheduler._decode_attn_active_serving_wave_expected_lanes = (
            (0, -1),
        )
    elif invalid_source == "active_wave_request_lane_bool":
        scheduler._decode_attn_active_serving_wave_request_ids_by_lane = {
            (0, False): {request.id},
        }
    elif invalid_source == "configured_lane_negative":
        scheduler._f2a_expected_lanes = [(0, -1)]
    elif invalid_source == "replica_scheduler_count_bool":
        scheduler._f2a_expected_lanes = None
        scheduler._replica_scheduler_count = True
    elif invalid_source == "replica_scheduler_count_float":
        scheduler._f2a_expected_lanes = None
        scheduler._replica_scheduler_count = 1.0
    elif invalid_source == "replica_scheduler_count_zero":
        scheduler._f2a_expected_lanes = None
        scheduler._replica_scheduler_count = 0
    else:
        raise AssertionError(f"Unhandled topology source: {invalid_source}")

    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="lane|topology|stage|replica scheduler|replica_scheduler_count|exact",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "topology_source",
    [
        "idle_lanes",
        "replica_scheduler_map",
        "active_stage_slots",
        "active_wave_lanes",
        "active_wave_request_map",
        "configured_f2a_lanes",
        "replica_scheduler_count",
    ],
)
def test_decode_attn_transfer_end_preserves_each_legal_topology_source(
    topology_source: str,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()

    if topology_source == "idle_lanes":
        scheduler._decode_attn_idle_expected_lanes = {(1, None)}
    elif topology_source == "replica_scheduler_map":
        scheduler._replica_schedulers = {
            (0, None): SimpleNamespace(
                _decode_attn_active_cohort_states={9: cohort_state},
                get_decode_attn_active_stage_slots=Mock(return_value=()),
            ),
        }
    elif topology_source == "active_stage_slots":
        scheduler._replica_schedulers = {
            (0, None): SimpleNamespace(
                _decode_attn_active_cohort_states={9: cohort_state},
                get_decode_attn_active_stage_slots=Mock(return_value=(1,)),
            ),
        }
        scheduler._f2a_expected_lanes = [(0, 1)]
    elif topology_source == "active_wave_lanes":
        scheduler._decode_attn_active_serving_wave_expected_lanes = (
            (0, None),
        )
        scheduler._f2a_expected_lanes = [(0, 1)]
    elif topology_source == "active_wave_request_map":
        scheduler._decode_attn_active_serving_wave_request_ids_by_lane = {
            (0, None): {request.id},
        }
        scheduler._f2a_expected_lanes = [(0, 1)]
    elif topology_source == "configured_f2a_lanes":
        scheduler._f2a_expected_lanes = [(0, None)]
    elif topology_source == "replica_scheduler_count":
        scheduler._f2a_expected_lanes = None
        scheduler._replica_scheduler_count = 1
    else:
        raise AssertionError(f"Unhandled topology source: {topology_source}")

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.completed_layer_count == 1
    assert request.af_roundtrip_inflight is False
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="local_attn",
        layer_id=1,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)


def test_decode_attn_transfer_end_rejects_empty_existing_room_contract() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    round_key = (0, 1, 1, 7)
    corrupt_room = {
        "per_lane_queues": defaultdict(deque),
        "expected_lanes": (),
    }
    scheduler._f2a_waiting_by_round[round_key] = corrupt_room
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="expected lanes|contract|non-empty",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert scheduler._f2a_waiting_by_round[round_key] is corrupt_room
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_transfer_preflight_does_not_lazy_create_replica_state() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_schedulers = {(0, None): replica_scheduler}
    scheduler._f2a_expected_lanes = [(0, 1)]
    scheduler._replica_scheduler_count = 2
    assert not hasattr(replica_scheduler, "_decode_attn_active_cohort_states")
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(ValueError, match="lane|topology"):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )
            assert not hasattr(
                replica_scheduler,
                "_decode_attn_active_cohort_states",
            )


@pytest.mark.parametrize(
    "corrupt_room",
    [
        object(),
        {"per_lane_queues": defaultdict(deque)},
        {"per_lane_queues": defaultdict(deque), "expected_lanes": ()},
        {"per_lane_queues": [], "expected_lanes": ((0, None),)},
        {"per_lane_queues": {(0, None): []}, "expected_lanes": ((0, None),)},
        {
            "per_lane_queues": defaultdict(
                deque,
                {(0, None): deque([object()])},
            ),
            "expected_lanes": ((0, None),),
        },
    ],
)
def test_decode_attn_final_transfer_rejects_existing_room_before_mutation(
    corrupt_room,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(completed_layer_count=3)
    round_key = (0, 4, 1, 7)
    scheduler._f2a_waiting_by_round[round_key] = corrupt_room
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="final|waiting room|expected lanes|queue|Batch",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            assert scheduler._f2a_waiting_by_round[round_key] is corrupt_room
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


class _CoercibleStagePhase:
    def __str__(self) -> str:
        return "ffn_inflight"

    def __deepcopy__(self, memo):
        del memo
        return self


def _install_real_decode_attn_replica_state(
    scheduler,
    request,
    cohort_state,
    *,
    active_stage_indices=None,
    stage_phases=None,
    stage_layers=None,
) -> None:
    if active_stage_indices is None:
        active_stage_indices = {1}
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    cohort_state["all_request_ids"] = {request.id}
    cohort_state["pending_request_ids"] = {request.id}
    cohort_state["active_stage_indices"] = active_stage_indices
    cohort_state["stage_phases"] = (
        {1: "ffn_inflight"} if stage_phases is None else stage_phases
    )
    cohort_state["stage_current_layer_ids"] = (
        {1: 0} if stage_layers is None else stage_layers
    )
    replica_scheduler._decode_attn_active_cohort_states = {9: cohort_state}
    scheduler._replica_schedulers = {(0, None): replica_scheduler}
    scheduler._f2a_expected_lanes = [(0, 1)]
    scheduler._replica_scheduler_count = 2


@pytest.mark.parametrize("active_stage_indices", [{True}, {1.0}, {"1"}])
def test_decode_attn_transfer_rejects_coercible_active_stage_index_before_mutation(
    active_stage_indices,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    _install_real_decode_attn_replica_state(
        scheduler,
        request,
        cohort_state,
        active_stage_indices=active_stage_indices,
    )
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="active stage|exact|topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "stage_layers",
    [
        {True: 0},
        {1: False},
        {1: 0.0},
        {1: "0"},
    ],
)
def test_decode_attn_transfer_rejects_coercible_active_stage_layer_before_mutation(
    stage_layers,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    _install_real_decode_attn_replica_state(
        scheduler,
        request,
        cohort_state,
        stage_layers=stage_layers,
    )
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="stage layer|exact|topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


@pytest.mark.parametrize(
    "stage_phases",
    [
        {True: "ffn_inflight"},
        {1: _CoercibleStagePhase()},
    ],
)
def test_decode_attn_transfer_rejects_non_string_active_stage_phase_before_mutation(
    stage_phases,
) -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()
    _install_real_decode_attn_replica_state(
        scheduler,
        request,
        cohort_state,
        stage_phases=stage_phases,
    )
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(
        (RuntimeError, TypeError, ValueError),
        match="stage phase|exact|topology",
    ):
        try:
            M2NTransferEndEvent(1.0, transfer_info).handle_event(
                global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_active_stage_slots_preserve_exact_filter_contract() -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {
        9: {
            "pending_request_ids": {17},
            "af_phase": "ffn_inflight",
            "current_layer_id": 0,
            "active_stage_indices": {1},
            "stage_phases": {1: "ffn_inflight"},
            "stage_current_layer_ids": {1: 0},
        }
    }
    state_identity = id(replica_scheduler._decode_attn_active_cohort_states)
    nested_state_identity = id(
        replica_scheduler._decode_attn_active_cohort_states[9]
    )
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="ffn_inflight",
        layer_id=0,
    ) == (1,)
    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="local_attn",
        layer_id=0,
    ) == ()
    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="ffn_inflight",
        layer_id=1,
    ) == ()
    assert id(replica_scheduler._decode_attn_active_cohort_states) == state_identity
    assert (
        id(replica_scheduler._decode_attn_active_cohort_states[9])
        == nested_state_identity
    )
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_stage_phases",
        "missing_stage_layers",
        "incomplete_stage_phases",
        "incomplete_stage_layers",
        "extra_stage_phase",
        "extra_stage_layer",
        "invalid_stage_phase",
        "invalid_stage_layer",
    ],
)
def test_decode_attn_active_stage_slots_reject_incomplete_or_invalid_stage_maps_without_mutation(
    invalid_case: str,
) -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    cohort_state = {
        "pending_request_ids": {17},
        "af_phase": "mixed",
        "current_layer_id": 0,
        "active_stage_indices": {0, 1},
        "stage_phases": {0: "ffn_inflight", 1: "local_attn"},
        "stage_current_layer_ids": {0: 0, 1: 0},
    }
    if invalid_case == "missing_stage_phases":
        cohort_state.pop("stage_phases")
    elif invalid_case == "missing_stage_layers":
        cohort_state.pop("stage_current_layer_ids")
    elif invalid_case == "incomplete_stage_phases":
        cohort_state["stage_phases"].pop(1)
    elif invalid_case == "incomplete_stage_layers":
        cohort_state["stage_current_layer_ids"].pop(1)
    elif invalid_case == "extra_stage_phase":
        cohort_state["stage_phases"][2] = "local_attn"
    elif invalid_case == "extra_stage_layer":
        cohort_state["stage_current_layer_ids"][2] = 0
    elif invalid_case == "invalid_stage_phase":
        cohort_state["stage_phases"][1] = "mixed"
    elif invalid_case == "invalid_stage_layer":
        cohort_state["stage_current_layer_ids"][1] = False
    else:
        raise AssertionError(f"Unhandled invalid case: {invalid_case}")
    replica_scheduler._decode_attn_active_cohort_states = {9: cohort_state}
    state_identity = id(replica_scheduler._decode_attn_active_cohort_states)
    nested_state_identity = id(cohort_state)
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    with pytest.raises(
        RuntimeError,
        match="stage phase|stage layer|active stage|key set|exact",
    ):
        replica_scheduler.get_decode_attn_active_stage_slots(
            phase="local_attn",
            layer_id=0,
        )

    assert id(replica_scheduler._decode_attn_active_cohort_states) == state_identity
    assert id(replica_scheduler._decode_attn_active_cohort_states[9]) == nested_state_identity
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


def test_decode_attn_active_stage_slots_preserve_mixed_phase_and_layer_state() -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {
        9: {
            "pending_request_ids": {17},
            "af_phase": "mixed",
            "current_layer_id": 5,
            "active_stage_indices": {0, 1},
            "stage_phases": {0: "ffn_inflight", 1: "local_attn"},
            "stage_current_layer_ids": {0: 4, 1: 5},
        }
    }
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="ffn_inflight",
        layer_id=4,
    ) == (0,)
    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="local_attn",
        layer_id=5,
    ) == (1,)
    assert replica_scheduler.get_decode_attn_active_stage_slots(
        phase="local_attn",
        layer_id=4,
    ) == ()
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


def test_decode_attn_cluster_phase_prepare_rejects_incomplete_stage_maps_without_mutation() -> None:
    cohort_state = {
        "pending_request_ids": {17},
        "af_phase": "local_attn",
        "current_layer_id": 0,
        "active_stage_indices": {0, 1},
        "stage_phases": {0: "local_attn"},
        "stage_current_layer_ids": {0: 0},
    }
    replica_scheduler = SimpleNamespace(
        _decode_attn_active_cohort_states={9: cohort_state}
    )
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_schedulers = {(0, None): replica_scheduler}
    batch = SimpleNamespace(
        decode_attn_cohort_id=9,
        decode_attn_original_replica_id=0,
        decode_attn_original_replica_local_id=None,
        afd_stage_idx=0,
    )
    state_identity = id(cohort_state)
    state_before = deepcopy(cohort_state)

    with pytest.raises(RuntimeError, match="stage phase|stage layer|key set|active"):
        scheduler._set_decode_attn_batch_cohort_phase(
            batch,
            phase="ffn_inflight",
            replica_id=0,
            dp_id=None,
            layer_id=0,
        )

    assert id(replica_scheduler._decode_attn_active_cohort_states[9]) == state_identity
    assert cohort_state == state_before


def test_decode_attn_cluster_phase_update_preserves_untouched_stage_visibility() -> None:
    cohort_state = {
        "all_request_ids": {17},
        "pending_request_ids": {17},
        "af_phase": "local_attn",
        "current_layer_id": 0,
        "active_stage_indices": {0, 1},
        "stage_phases": {0: "local_attn", 1: "local_attn"},
        "stage_current_layer_ids": {0: 0, 1: 0},
    }
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {9: cohort_state}
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_schedulers = {(0, None): replica_scheduler}
    batch = SimpleNamespace(
        decode_attn_cohort_id=9,
        decode_attn_original_replica_id=0,
        decode_attn_original_replica_local_id=None,
        afd_stage_idx=0,
    )

    scheduler._set_decode_attn_batch_cohort_phase(
        batch,
        phase="ffn_inflight",
        replica_id=0,
        dp_id=None,
        layer_id=0,
    )

    assert cohort_state["af_phase"] == "mixed"
    assert cohort_state["stage_phases"] == {
        0: "ffn_inflight",
        1: "local_attn",
    }
    assert scheduler._get_decode_attn_stage_slot_active_lanes(
        1,
        phase="local_attn",
        layer_id=0,
    ) == [(0, None)]


@pytest.mark.parametrize("cohort_id", [True, -1, 9.0, "9"])
def test_decode_attn_active_stage_slots_reject_invalid_cohort_id_without_mutation(
    cohort_id,
) -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {
        cohort_id: {
            "pending_request_ids": {17},
            "active_stage_indices": {1},
        }
    }
    state_identity = id(replica_scheduler._decode_attn_active_cohort_states)
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    with pytest.raises(RuntimeError, match="cohort.*exact|cohort.*non-negative"):
        replica_scheduler.get_decode_attn_active_stage_slots()

    assert id(replica_scheduler._decode_attn_active_cohort_states) == state_identity
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


@pytest.mark.parametrize(
    "pending_request_ids",
    [None, [], (17,), {}, frozenset({17}), {True}, {-1}, {17.0}, {"17"}],
)
def test_decode_attn_active_stage_slots_reject_invalid_pending_request_ids_without_mutation(
    pending_request_ids,
) -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {
        9: {
            "pending_request_ids": pending_request_ids,
            "active_stage_indices": {1},
            **(
                {
                    "stage_phases": {1: "ffn_inflight"},
                    "stage_current_layer_ids": {1: 0},
                }
                if pending_request_ids
                else {}
            ),
        }
    }
    state_identity = id(replica_scheduler._decode_attn_active_cohort_states)
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    with pytest.raises(
        RuntimeError,
        match="pending request|request id|exact set|exact non-negative",
    ):
        replica_scheduler.get_decode_attn_active_stage_slots()

    assert id(replica_scheduler._decode_attn_active_cohort_states) == state_identity
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


@pytest.mark.parametrize(
    ("pending_request_ids", "expected_slots"),
    [(set(), ()), ({17}, (1,))],
)
def test_decode_attn_active_stage_slots_accept_exact_pending_request_id_sets_without_mutation(
    pending_request_ids,
    expected_slots,
) -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN
    replica_scheduler._decode_attn_active_cohort_states = {
        9: {
            "pending_request_ids": pending_request_ids,
            "active_stage_indices": {1},
            "stage_phases": {1: "ffn_inflight"},
            "stage_current_layer_ids": {1: 0},
        }
    }
    state_identity = id(replica_scheduler._decode_attn_active_cohort_states)
    state_before = deepcopy(replica_scheduler._decode_attn_active_cohort_states)

    assert replica_scheduler.get_decode_attn_active_stage_slots() == expected_slots
    assert id(replica_scheduler._decode_attn_active_cohort_states) == state_identity
    assert replica_scheduler._decode_attn_active_cohort_states == state_before


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"phase": False}, "phase filter"),
        ({"phase": 1}, "phase filter"),
        ({"layer_id": True}, "layer filter"),
        ({"layer_id": 0.0}, "layer filter"),
        ({"layer_id": "0"}, "layer filter"),
        ({"layer_id": -1}, "layer filter"),
    ],
)
def test_decode_attn_active_stage_slots_reject_coercible_filter_arguments(
    kwargs,
    error_match,
) -> None:
    replica_scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    replica_scheduler._cluster_type = ClusterType.DECODE_ATTN

    with pytest.raises(ValueError, match=error_match):
        replica_scheduler.get_decode_attn_active_stage_slots(**kwargs)


def _configure_legacy_decode_attn_fifo_fixture(
    fixture,
    *,
    source_replica_id: int,
    global_id: int,
    decode_token_index: int,
) -> None:
    _, batch, request, transfer_info, _, _, _ = fixture
    batch.set_global_id(global_id)
    batch.decode_attn_original_replica_id = source_replica_id
    batch.decode_attn_original_replica_local_id = None
    batch.decode_attn_barrier_round_id = None
    batch.decode_attn_barrier_expected_lanes = ()
    batch.replay_decode_token_index = decode_token_index
    request._current_decode_token_index = decode_token_index
    transfer_info.source_replica_id = source_replica_id
    transfer_info.source_replica_local_id = None


@pytest.mark.skip(reason="The retired attention-DP unscoped-round contract is no longer supported")
def test_decode_attn_legacy_unscoped_round_preserves_distinct_fifo_wave_identities() -> None:
    fixtures = [_decode_attn_return_fixture() for _ in range(4)]
    scheduler = fixtures[0][0]
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]
    scheduler._replica_scheduler_count = 2
    scheduler._cluster.replicas = {0: SimpleNamespace(), 1: SimpleNamespace()}
    shared_global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler),
    )
    configurations = [
        (0, 77, 2),
        (0, 78, 3),
        (1, 77, 2),
        (1, 78, 3),
    ]
    for fixture, (source_replica_id, global_id, decode_token_index) in zip(
        fixtures,
        configurations,
    ):
        _configure_legacy_decode_attn_fifo_fixture(
            fixture,
            source_replica_id=source_replica_id,
            global_id=global_id,
            decode_token_index=decode_token_index,
        )
    _install_decode_attn_cohort_registries_by_lane(scheduler, fixtures)

    returned_events = []
    for index, fixture in enumerate(fixtures):
        transfer_info = fixture[3]
        metrics_store = fixture[5]
        returned_events.append(
            M2NTransferEndEvent(1.0 + index, transfer_info).handle_event(
                shared_global_scheduler,
                metrics_store,
            )
        )

    lane0_wave1, lane0_wave2, lane1_wave1, lane1_wave2 = [
        fixture[1] for fixture in fixtures
    ]
    assert [len(events) for events in returned_events] == [0, 0, 1, 1]
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [
        lane0_wave1,
        lane1_wave1,
        lane0_wave2,
        lane1_wave2,
    ]


@pytest.mark.skip(reason="The retired attention-DP unscoped-round contract is no longer supported")
def test_decode_attn_legacy_unscoped_round_rejects_cross_lane_identity_mismatch() -> None:
    lane0_fixture = _decode_attn_return_fixture()
    lane1_fixture = _decode_attn_return_fixture()
    scheduler = lane0_fixture[0]
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]
    scheduler._replica_scheduler_count = 2
    scheduler._cluster.replicas = {0: SimpleNamespace(), 1: SimpleNamespace()}
    shared_global_scheduler = SimpleNamespace(
        get_cluster_scheduler=Mock(return_value=scheduler),
    )
    _configure_legacy_decode_attn_fifo_fixture(
        lane0_fixture,
        source_replica_id=0,
        global_id=77,
        decode_token_index=2,
    )
    _configure_legacy_decode_attn_fifo_fixture(
        lane1_fixture,
        source_replica_id=1,
        global_id=78,
        decode_token_index=3,
    )
    _install_decode_attn_cohort_registries_by_lane(
        scheduler,
        (lane0_fixture, lane1_fixture),
    )
    M2NTransferEndEvent(1.0, lane0_fixture[3]).handle_event(
        shared_global_scheduler,
        lane0_fixture[5],
    )
    _, batch, request, transfer_info, _, metrics_store, cohort_state = lane1_fixture
    before = _snapshot_decode_attn_return_state(
        scheduler,
        batch,
        request,
        transfer_info,
        cohort_state,
    )

    with pytest.raises(RuntimeError, match="identity|global IDs|decode token"):
        try:
            M2NTransferEndEvent(2.0, transfer_info).handle_event(
                shared_global_scheduler,
                metrics_store,
            )
        finally:
            _assert_decode_attn_return_state_unchanged(
                before,
                scheduler,
                batch,
                request,
                transfer_info,
                metrics_store,
                cohort_state,
            )


def test_decode_attn_legal_nonfinal_return_preserves_event_and_queue_contract() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture()

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 1
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [batch]
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="local_attn",
        layer_id=1,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    request.on_m2n_transfer_complete.assert_called_once_with(0.5, False)
    request.on_inter_cluster_transfer_end.assert_called_once()
    assert len(events) == 1
    assert isinstance(events[0], ClusterScheduleEvent)
    assert events[0].time == pytest.approx(1.0)


@pytest.mark.skip(reason="Per-Replica full-stage F-to-A returns no longer use local-DP FIFO lanes")
def test_decode_attn_legal_nonfinal_return_preserves_per_lane_fifo_release() -> None:
    fixtures = [
        _decode_attn_return_fixture(expected_lanes=((0, None), (1, None)))
        for _ in range(4)
    ]
    scheduler = fixtures[0][0]
    scheduler._f2a_expected_lanes = [(0, None), (1, None)]
    scheduler._replica_scheduler_count = 2
    scheduler._cluster.replicas = {0: SimpleNamespace(), 1: SimpleNamespace()}

    for index, fixture in enumerate(fixtures):
        _, batch, _, transfer_info, _, _, _ = fixture
        source_replica_id = 0 if index < 2 else 1
        batch.decode_attn_original_replica_id = source_replica_id
        batch.decode_attn_original_replica_local_id = None
        transfer_info.source_replica_id = source_replica_id
        transfer_info.source_replica_local_id = None
    stale_request = _append_decode_attn_return_request(
        fixtures[0][1],
        fixtures[0][6],
        completed=True,
        completed_layer_count=3,
        roundtrip_inflight=True,
    )
    fixtures[0][6]["pending_request_ids"].remove(stale_request.id)
    assert stale_request.id in fixtures[0][6]["all_request_ids"]
    assert stale_request.id not in fixtures[0][6]["pending_request_ids"]
    assert set(fixtures[0][1].decode_attn_cohort_request_ids) == fixtures[0][6][
        "all_request_ids"
    ]
    _install_decode_attn_cohort_registries_by_lane(scheduler, fixtures)

    returned_events = []
    for index, fixture in enumerate(fixtures):
        _, _, _, transfer_info, _, metrics_store, _ = fixture
        shared_global_scheduler = SimpleNamespace(
            get_cluster_scheduler=Mock(return_value=scheduler),
        )
        events = M2NTransferEndEvent(1.0 + index, transfer_info).handle_event(
            shared_global_scheduler,
            metrics_store,
        )
        returned_events.append(events)

    first_batch, second_batch, third_batch, fourth_batch = [
        fixture[1] for fixture in fixtures
    ]
    assert [request.completed_layer_count for _, _, request, *_ in fixtures] == [
        1,
        1,
        1,
        1,
    ]
    assert stale_request.completed_layer_count == 3
    assert returned_events[0] == []
    assert returned_events[1] == []
    assert len(returned_events[2]) == 1
    assert isinstance(returned_events[2][0], ClusterScheduleEvent)
    assert returned_events[2][0].time == pytest.approx(3.0)
    assert len(returned_events[3]) == 1
    assert isinstance(returned_events[3][0], ClusterScheduleEvent)
    assert returned_events[3][0].time == pytest.approx(4.0)
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == [
        first_batch,
        third_batch,
        second_batch,
        fourth_batch,
    ]


def test_decode_attn_legal_final_return_preserves_global_end_contract() -> None:
    (
        scheduler,
        batch,
        request,
        transfer_info,
        global_scheduler,
        metrics_store,
        cohort_state,
    ) = _decode_attn_return_fixture(completed_layer_count=3)

    events = M2NTransferEndEvent(1.0, transfer_info).handle_event(
        global_scheduler,
        metrics_store,
    )

    assert transfer_info.transfer_end_time == pytest.approx(1.0)
    assert request._m2n_transfer_time_ffn_to_attn == pytest.approx(0.5)
    assert request.af_roundtrip_inflight is False
    assert request.completed_layer_count == 4
    assert scheduler._f2a_waiting_by_round == {}
    assert scheduler._af_batch_queue == []
    _assert_decode_attn_cohort_state(
        cohort_state,
        request_ids=(request.id,),
        phase="ffn_inflight",
        layer_id=3,
    )
    metrics_store.on_m2n_transfer_end.assert_called_once()
    assert len(events) == 2
    assert isinstance(events[0], GlobalBatchEndEvent)
    assert events[0].time == pytest.approx(1.0)
    assert events[0]._batch is batch
    assert events[0]._replica_id == 0
    assert events[0]._replica_local_id is None
    assert isinstance(events[1], ClusterScheduleEvent)
    assert events[1].time == pytest.approx(1.0)


def _ep_scheduler(
    waiting_attr: str,
    room=None,
    *,
    batch_global_id=77,
    ep_size: int = 2,
) -> _ConcreteClusterScheduler:
    scheduler = _scheduler()
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            moe_expert_parallel_size=ep_size,
            model_config=SimpleNamespace(
                embedding_dim=4096,
                model_architecture_profile="step3_text",
                model_type="step3_text",
                get_model_architecture_profile=lambda: (
                    MODEL_ARCHITECTURE_REGISTRY.get("step3_text")
                ),
            ),
        )
    )
    scheduler.get_replica = Mock(return_value=SimpleNamespace(ep_size=ep_size))
    scheduler._predictor = Mock()
    scheduler._predictor.predict_alltoall_time.return_value = 0.0
    scheduler._predictor.predict_allgather_time.return_value = 0.0
    waiting_rooms = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
        )
    )
    if room is not None:
        waiting_rooms[3][4][batch_global_id] = room
    setattr(scheduler, waiting_attr, waiting_rooms)
    return scheduler


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize(
    ("invalid_case", "event_ep_id", "batch_ep_id", "existing_ep_id"),
    [
        ("duplicate", 0, 0, 0),
        ("negative", -1, -1, None),
        ("out_of_range", 2, 2, None),
        ("event_batch_mismatch", 0, 1, None),
        ("bool", True, True, None),
        ("float", 0.0, 0.0, None),
    ],
)
def test_ep_waiting_room_rejects_invalid_ep_id_without_mutation(
    method_name: str,
    waiting_attr: str,
    invalid_case: str,
    event_ep_id: int,
    batch_ep_id: int,
    existing_ep_id: int | None,
) -> None:
    room = {"batches": {}, "arrival_times": {}}
    if existing_ep_id is not None:
        room["batches"][existing_ep_id] = SimpleNamespace(
            id=10,
            global_id=77,
            ep_id=existing_ep_id,
            replica_id=3,
            total_num_tokens=1,
        )
        room["arrival_times"][existing_ep_id] = 0.5

    scheduler = _ep_scheduler(waiting_attr, room)
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=batch_ep_id,
        replica_id=3,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)

    with pytest.raises(ValueError, match="ep_id"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=event_ep_id,
            )
        finally:
            assert _room_snapshot(room) == snapshot_before, invalid_case


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
def test_ep_waiting_room_rejects_event_batch_replica_mismatch_without_mutation(
    method_name: str,
    waiting_attr: str,
) -> None:
    room = {"batches": {}, "arrival_times": {}}
    scheduler = _ep_scheduler(waiting_attr, room)
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=0,
        replica_id=2,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)

    with pytest.raises(ValueError, match="replica_id"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=0,
            )
        finally:
            assert _room_snapshot(room) == snapshot_before


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize(
    (
        "invalid_case",
        "event_replica_id",
        "stage_id",
        "batch_replica_id",
        "error_match",
    ),
    [
        ("bool_event_replica", True, 4, 1, "replica_id"),
        ("float_event_replica", 3.0, 4, 3, "replica_id"),
        ("negative_event_replica", -1, 4, -1, "replica_id"),
        ("bool_stage", 3, True, 3, "stage_id"),
        ("float_stage", 3, 4.0, 3, "stage_id"),
        ("negative_stage", 3, -1, 3, "stage_id"),
        ("bool_batch_replica_alias", 1, 4, True, "replica_id"),
        ("float_batch_replica_alias", 3, 4, 3.0, "replica_id"),
    ],
)
def test_ep_waiting_room_rejects_nonexact_replica_and_stage_ids_without_mutation(
    method_name: str,
    waiting_attr: str,
    invalid_case: str,
    event_replica_id,
    stage_id,
    batch_replica_id,
    error_match: str,
) -> None:
    scheduler = _ep_scheduler(waiting_attr)
    waiting_rooms = getattr(scheduler, waiting_attr)
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=0,
        replica_id=batch_replica_id,
        total_num_tokens=1,
    )
    assert dict(waiting_rooms) == {}

    with pytest.raises(ValueError, match=error_match):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=event_replica_id,
                stage_id=stage_id,
                batch=batch,
                ep_id=0,
            )
        finally:
            assert dict(waiting_rooms) == {}, invalid_case
            scheduler._predictor.predict_alltoall_time.assert_not_called()
            scheduler._predictor.predict_allgather_time.assert_not_called()


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize(
    ("corrupt_field", "corrupt_value"),
    [
        ("global_id", 78),
        ("ep_id", 1),
        ("replica_id", 2),
    ],
)
def test_ep_waiting_room_rejects_corrupt_existing_batch_before_side_effects(
    method_name: str,
    waiting_attr: str,
    corrupt_field: str,
    corrupt_value,
) -> None:
    existing_fields = {
        "id": 10,
        "global_id": 77,
        "ep_id": 0,
        "replica_id": 3,
        "total_num_tokens": 500,
    }
    existing_fields[corrupt_field] = corrupt_value
    existing_batch = SimpleNamespace(**existing_fields)
    room = {
        "batches": {0: existing_batch},
        "arrival_times": {0: 0.5},
    }
    scheduler = _ep_scheduler(waiting_attr, room)
    incoming_batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=1,
        replica_id=3,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)
    logger = Mock()

    with patch("frontier.logger.get_cluster_logger", return_value=logger):
        with pytest.raises(ValueError, match=corrupt_field):
            try:
                getattr(scheduler, method_name)(
                    time=1.0,
                    replica_id=3,
                    stage_id=4,
                    batch=incoming_batch,
                    ep_id=1,
                )
            finally:
                assert _room_snapshot(room) == snapshot_before
                scheduler._predictor.predict_alltoall_time.assert_not_called()
                scheduler._predictor.predict_allgather_time.assert_not_called()
                logger.info.assert_not_called()


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize("invalid_global_id", [True, -1, 77.0])
def test_ep_waiting_room_rejects_invalid_batch_global_id_without_mutation(
    method_name: str,
    waiting_attr: str,
    invalid_global_id,
) -> None:
    scheduler = _ep_scheduler(
        waiting_attr,
        batch_global_id=invalid_global_id,
    )
    batch = SimpleNamespace(
        id=11,
        global_id=invalid_global_id,
        ep_id=0,
        replica_id=3,
        total_num_tokens=1,
    )
    waiting_rooms = getattr(scheduler, waiting_attr)
    assert dict(waiting_rooms) == {}

    with pytest.raises(ValueError, match="global_id"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=0,
            )
        finally:
            assert dict(waiting_rooms) == {}


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize("invalid_expected_ep_size", [True, 0, -1, 2.0])
def test_ep_waiting_room_rejects_invalid_expected_ep_size_without_mutation(
    method_name: str,
    waiting_attr: str,
    invalid_expected_ep_size,
) -> None:
    room = {"batches": {}, "arrival_times": {}}
    scheduler = _ep_scheduler(
        waiting_attr,
        room,
        ep_size=invalid_expected_ep_size,
    )
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=0,
        replica_id=3,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)

    with pytest.raises(ValueError, match="expected_ep_size"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=0,
            )
        finally:
            assert _room_snapshot(room) == snapshot_before


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize("populated_side", ["batches", "arrival_times"])
def test_ep_waiting_room_rejects_existing_keyset_mismatch_without_mutation(
    method_name: str,
    waiting_attr: str,
    populated_side: str,
) -> None:
    existing_batch = SimpleNamespace(
        id=10,
        global_id=77,
        ep_id=0,
        replica_id=3,
        total_num_tokens=1,
    )
    room = {"batches": {}, "arrival_times": {}}
    if populated_side == "batches":
        room["batches"][0] = existing_batch
    else:
        room["arrival_times"][0] = 0.5
    scheduler = _ep_scheduler(waiting_attr, room, ep_size=3)
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=1,
        replica_id=3,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)

    with pytest.raises(ValueError, match="key|waiting|arrival"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=1,
            )
        finally:
            assert _room_snapshot(room) == snapshot_before


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize("existing_invalid_ep_id", [True, 0.0, -1, 2])
def test_ep_waiting_room_rejects_existing_nonexact_lane_set_without_mutation(
    method_name: str,
    waiting_attr: str,
    existing_invalid_ep_id: int,
) -> None:
    existing_batch = SimpleNamespace(
        id=10,
        global_id=77,
        ep_id=existing_invalid_ep_id,
        replica_id=3,
        total_num_tokens=1,
    )
    room = {
        "batches": {existing_invalid_ep_id: existing_batch},
        "arrival_times": {existing_invalid_ep_id: 0.5},
    }
    scheduler = _ep_scheduler(waiting_attr, room)
    batch = SimpleNamespace(
        id=11,
        global_id=77,
        ep_id=0,
        replica_id=3,
        total_num_tokens=1,
    )
    snapshot_before = _room_snapshot(room)

    with pytest.raises(ValueError, match="ep_id|lane|exact"):
        try:
            getattr(scheduler, method_name)(
                time=1.0,
                replica_id=3,
                stage_id=4,
                batch=batch,
                ep_id=0,
            )
        finally:
            assert _room_snapshot(room) == snapshot_before


def test_ep_dispatch_valid_exact_lanes_preserve_collective_payload_and_time() -> None:
    room = {"batches": {}, "arrival_times": {}}
    scheduler = _ep_scheduler(
        "_ep_alltoall_dispatch_waiting_room",
        room,
        batch_global_id=0,
    )
    first_batch = SimpleNamespace(
        id=10,
        global_id=0,
        ep_id=0,
        replica_id=3,
        total_num_tokens=100,
    )
    second_batch = SimpleNamespace(
        id=11,
        global_id=0,
        ep_id=1,
        replica_id=3,
        total_num_tokens=200,
    )

    first_events = scheduler.on_ep_alltoall_dispatch_ready(
        time=1.25,
        replica_id=3,
        stage_id=4,
        batch=first_batch,
        ep_id=0,
    )
    second_events = scheduler.on_ep_alltoall_dispatch_ready(
        time=1.5,
        replica_id=3,
        stage_id=4,
        batch=second_batch,
        ep_id=1,
    )

    assert first_events == []
    assert len(second_events) == 1
    assert isinstance(second_events[0], EPAllToAllDispatchCollectiveEvent)
    assert second_events[0].time == pytest.approx(1.5)
    assert second_events[0].to_dict()["replica_id"] == 3
    assert second_events[0].to_dict()["stage_id"] == 4
    assert second_events[0].to_dict()["batch_global_id"] == 0
    scheduler._predictor.predict_alltoall_time.assert_called_once_with(
        data_size_bytes=200 * 4096 * 2,
        num_devices=2,
        cluster_type=ClusterType.DECODE_FFN,
        comm_domain="EP",
    )


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
def test_ep_collective_predictor_failure_does_not_commit_last_lane(
    method_name: str,
    waiting_attr: str,
) -> None:
    """The final EP lane must be committed only after predictor success."""

    room = {"batches": {}, "arrival_times": {}}
    scheduler = _ep_scheduler(waiting_attr, room, batch_global_id=0)
    first_batch = SimpleNamespace(
        id=10,
        global_id=0,
        ep_id=0,
        replica_id=3,
        total_num_tokens=100,
    )
    second_batch = SimpleNamespace(
        id=11,
        global_id=0,
        ep_id=1,
        replica_id=3,
        total_num_tokens=200,
    )
    assert getattr(scheduler, method_name)(
        time=1.25,
        replica_id=3,
        stage_id=4,
        batch=first_batch,
        ep_id=0,
    ) == []
    room_before = _room_snapshot(room)
    scheduler._predictor.predict_alltoall_time.side_effect = RuntimeError(
        "predict boom"
    )

    with pytest.raises(RuntimeError, match="predict boom"):
        try:
            getattr(scheduler, method_name)(
                time=1.5,
                replica_id=3,
                stage_id=4,
                batch=second_batch,
                ep_id=1,
            )
        finally:
            assert _room_snapshot(room) == room_before
            assert set(room["batches"]) == {0}
            assert set(room["arrival_times"]) == {0}


@pytest.mark.parametrize(
    ("method_name", "waiting_attr"),
    [
        (
            "on_ep_alltoall_dispatch_ready",
            "_ep_alltoall_dispatch_waiting_room",
        ),
        (
            "on_ep_alltoall_combine_ready",
            "_ep_allgather_waiting_room",
        ),
    ],
)
@pytest.mark.parametrize("invalid_exec_time_ms", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_ep_collective_rejects_nonfinite_or_negative_predictor_time_before_commit(
    method_name: str,
    waiting_attr: str,
    invalid_exec_time_ms: float,
) -> None:
    """Invalid collective latency must not publish an event or consume a lane."""

    room = {"batches": {}, "arrival_times": {}}
    scheduler = _ep_scheduler(waiting_attr, room, batch_global_id=0)
    first_batch = SimpleNamespace(
        id=10,
        global_id=0,
        ep_id=0,
        replica_id=3,
        total_num_tokens=100,
    )
    second_batch = SimpleNamespace(
        id=11,
        global_id=0,
        ep_id=1,
        replica_id=3,
        total_num_tokens=200,
    )
    assert getattr(scheduler, method_name)(
        time=1.25,
        replica_id=3,
        stage_id=4,
        batch=first_batch,
        ep_id=0,
    ) == []
    room_before = _room_snapshot(room)
    scheduler._predictor.predict_alltoall_time.return_value = invalid_exec_time_ms

    with pytest.raises(ValueError, match="finite|non-negative|collective|latency|time"):
        try:
            getattr(scheduler, method_name)(
                time=1.5,
                replica_id=3,
                stage_id=4,
                batch=second_batch,
                ep_id=1,
            )
        finally:
            assert _room_snapshot(room) == room_before
            assert set(room["batches"]) == {0}
            assert set(room["arrival_times"]) == {0}


def _empty_m2n_room():
    return {
        "per_lane_queues": {},
        "lanes_rr_order": deque(),
        "rr_cursor": 0,
    }


def test_debug_m2n_waiting_groups_supports_layer_stage_key() -> None:
    state = BaseClusterScheduler._debug_m2n_waiting_groups_state(
        {(4, 1): _empty_m2n_room()}
    )

    assert state == [
        {
            "key": {"layer_id": 4, "afd_stage_idx": 1},
            "lanes_rr_order": [],
            "rr_cursor": 0,
            "lane_queues": [],
        }
    ]


def test_debug_m2n_waiting_groups_supports_round_scoped_key_without_fake_target() -> None:
    state = BaseClusterScheduler._debug_m2n_waiting_groups_state(
        {(4, 1, 7): _empty_m2n_room()}
    )

    assert state == [
        {
            "key": {
                "layer_id": 4,
                "afd_stage_idx": 1,
                "barrier_round_id": 7,
            },
            "lanes_rr_order": [],
            "rr_cursor": 0,
            "lane_queues": [],
        }
    ]


@pytest.mark.parametrize("invalid_key", [4, (4,), (4, 1, 7, 9)])
def test_debug_m2n_waiting_groups_rejects_invalid_key_shape(invalid_key) -> None:
    with pytest.raises(TypeError, match="waiting key"):
        BaseClusterScheduler._debug_m2n_waiting_groups_state(
            {invalid_key: _empty_m2n_room()}
        )


def test_ep_batch_group_preserves_pre_routing_tokens_for_zero_lane() -> None:
    """A zero-routed EP lane still predicts shared pre-routing FFN work."""
    batch = EPBatchGroup(
        requests=[Request(0.0, 0, 0)],
        num_tokens=[0],
        replica_id=0,
        ep_id=4,
        time=0.0,
        source_batch_ids=[17],
        per_expert_tokens={6: 0},
        cluster_type=ClusterType.DECODE_FFN,
        is_moe=True,
    )
    batch.moe_pre_routing_effective_total_tokens = 8

    assert batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_FFN) == 8
    assert batch.get_effective_total_tokens_rounded(ClusterType.DECODE_FFN) == 8


def test_ep_execution_time_resolves_zero_routed_lane_without_fabrication() -> None:
    scheduler = _scheduler()
    zero_lane = SimpleNamespace(
        execution_time=0.0,
        per_expert_tokens={0: 0, 1: 0},
    )
    active_lane = SimpleNamespace(
        execution_time=0.75,
        per_expert_tokens={2: 1},
    )

    assert scheduler._resolve_ep_execution_time({4: zero_lane, 5: active_lane}) == 0.75


def test_ep_execution_time_rejects_zero_time_for_nonzero_routed_lane() -> None:
    scheduler = _scheduler()
    invalid_lane = SimpleNamespace(execution_time=0.0, per_expert_tokens={0: 1})

    with pytest.raises(ValueError, match="zero execution_time.*routed tokens"):
        scheduler._resolve_ep_execution_time({4: invalid_lane})
