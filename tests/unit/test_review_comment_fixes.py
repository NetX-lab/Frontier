from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from frontier.entities import Batch
from frontier.scheduler.utils.m2n_events import build_aggregated_batch_transfer_events
from frontier.scheduler.utils.m2n_promotion import promote_decode_ffn_group
from frontier.scheduler.utils.prefill_collective import handle_prefill_sync_collective
from frontier.scheduler.utils.scheduler_diagnostics import (
    SchedulerDiagnostics,
    scheduler_is_empty,
)
from frontier.scheduler.utils.sync_entry import enter_decode_sync, enter_prefill_sync
from frontier.types import ClusterType


class _ExplodingRequest:
    @property
    def id(self):
        raise RuntimeError("request id is malformed")


def test_m2n_event_logging_surfaces_malformed_request_metadata() -> None:
    batch = SimpleNamespace(
        id=1,
        requests=[_ExplodingRequest()],
        decode_attn_original_replica_id=0,
        decode_attn_original_replica_local_id=0,
        afd_stage_idx=0,
    )
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.DECODE_FFN,
        _m2n_transfer_predictor=SimpleNamespace(
            get_transfer_info=lambda **_: (16, 1.0)
        ),
        _config=SimpleNamespace(replica_config=SimpleNamespace()),
        _get_current_layer_id_from_batch=lambda _: 0,
    )

    with pytest.raises(RuntimeError, match="request id is malformed"):
        build_aggregated_batch_transfer_events(
            scheduler,
            batch,
            0.0,
            source_replica_id=0,
            source_replica_local_id=0,
        )


class _BrokenParticipantBatches:
    def keys(self):
        raise RuntimeError("participant mapping is malformed")

    def values(self):
        return []


def test_prefill_collective_surfaces_malformed_participant_mapping() -> None:
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.PREFILL,
        _prefill_sync_waiting_room={
            0: {0: {1: {0: {"post_moe": {"batches": _BrokenParticipantBatches()}}}}}
        },
    )

    with pytest.raises(RuntimeError, match="participant mapping is malformed"):
        handle_prefill_sync_collective(
            scheduler,
            0.0,
            0,
            0,
            1,
            "post_moe",
            0,
            metrics_store=None,
        )


def test_diagnostics_do_not_materialize_unused_state() -> None:
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.PREFILL,
        _request_queue=[],
        _replica_schedulers={},
    )

    assert scheduler_is_empty(scheduler)
    state = SchedulerDiagnostics.collect(scheduler)

    assert state["raw_batch_waiting_map"] == {"status": "not_applicable"}
    assert "_attention_transfer_state" not in scheduler.__dict__
    assert "_m2n_state" not in scheduler.__dict__


def test_prefill_sync_fails_when_expected_lane_scheduler_is_missing() -> None:
    batch = Batch(replica_id=0, requests=[], num_tokens=[], is_idle=False, is_moe=True)
    batch.set_global_id(7)
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.PREFILL,
        _prefill_sync_waiting_room={
            0: {0: {7: {0: {"pre_moe": {"batches": {}, "arrival_times": {}}}}}}
        },
        _replica_dp_size=2,
        _replica_schedulers={},
        _uses_shared_prefill_layer_path=lambda *_args: True,
        _get_forward_step_id=lambda current_batch: current_batch.global_id,
        _resolve_forward_step=lambda **_kwargs: (7, False),
    )

    with pytest.raises(RuntimeError, match="Missing Replica scheduler"):
        enter_prefill_sync(
            scheduler,
            time=0.0,
            replica_id=0,
            stage_id=0,
            batch=batch,
            replica_local_id=0,
            sync_stage="pre_moe",
            layer_id=0,
            stage_execution_time=0.0,
        )


def test_decode_sync_fails_when_expected_lane_scheduler_is_missing() -> None:
    batch = Batch(replica_id=0, requests=[], num_tokens=[], is_idle=False, is_moe=True)
    batch.set_global_id(7)
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.DECODE,
        _decode_sync_waiting_room={
            0: {0: {7: {0: {"pre_moe": {"batches": {}, "arrival_times": {}}}}}}
        },
        _replica_dp_size=2,
        _replica_schedulers={},
        _uses_shared_decode_layer_path=lambda *_args: True,
        _get_forward_step_id=lambda current_batch: current_batch.global_id,
        _resolve_forward_step=lambda **_kwargs: (7, False),
    )

    with pytest.raises(RuntimeError, match="Missing Replica scheduler"):
        enter_decode_sync(
            scheduler,
            time=0.0,
            replica_id=0,
            stage_id=0,
            batch=batch,
            replica_local_id=0,
            sync_stage="pre_moe",
            layer_id=0,
            stage_execution_time=0.0,
        )


def test_ffn_promotion_requires_idle_lane_state_when_injection_is_enabled() -> None:
    lane_zero = (0, 0)
    lane_one = (1, 0)
    room = {
        "per_lane_queues": defaultdict(deque, {lane_zero: deque([("batch", "info")])}),
        "lanes_rr_order": deque([lane_zero]),
        "rr_cursor": 0,
        "expected_lane_contract": (lane_zero, lane_one),
    }
    scheduler = SimpleNamespace(
        _ffn_group_micro_batches=2,
        _m2n_waiting_by_layer={(4, 2): room},
        _m2n_ready_groups=deque(),
        _validate_decode_ffn_waiting_room=lambda **_: (lane_zero, lane_one),
        _normalize_m2n_lanes=lambda lanes, **_: list(lanes),
    )

    with pytest.raises(RuntimeError, match="_ffn_idle_lanes"):
        promote_decode_ffn_group(
            scheduler,
            0.0,
            (4, 2),
            room,
            logger=SimpleNamespace(info=lambda *_: None),
            allow_idle_injection=True,
        )
