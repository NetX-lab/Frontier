from collections import deque
from types import MethodType, SimpleNamespace

import pytest

from frontier.entities.batch import Batch
from frontier.entities.request import Request
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _make_decode_request(request_index: int, decode_token_index: int = 1) -> Request:
    request = Request(
        arrived_at=float(request_index),
        num_prefill_tokens=32,
        num_decode_tokens=3,
        num_processed_tokens=32,
    )
    request._is_prefill_complete = True
    request._current_decode_token_index = decode_token_index
    return request


def _make_writer_scheduler(
    requests: list[Request],
    *,
    num_stages: int = 2,
    initial_global_id: int = 17,
) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_id = 0
    scheduler._replica_local_id = 0
    scheduler._replica_is_moe = True
    scheduler._micro_batch_size = len(requests)
    scheduler._af_pipeline_num_micro_batch = num_stages
    scheduler._af_pending_micro_batches = deque()
    scheduler._running_requests = list(requests)
    scheduler._waiting_requests = []
    scheduler._continuation_request_ids = set()
    scheduler._batch_creation_counter = initial_global_id
    scheduler._active_batch_request_counts = {}

    scheduler._can_allocate_request = lambda _request, _num_tokens: True
    scheduler._allocate_request = lambda _request, _num_tokens: None

    def create_batch(
        self: VLLMv1EngineReplicaScheduler,
        batch_requests: list[Request],
        batch_tokens: list[int],
    ) -> Batch:
        batch = Batch(
            replica_id=self._replica_id,
            requests=batch_requests,
            num_tokens=batch_tokens,
            is_moe=self._replica_is_moe,
        )
        batch.set_global_id(self._batch_creation_counter)
        self._batch_creation_counter += 1
        return batch

    scheduler._create_batch = MethodType(create_batch, scheduler)
    return scheduler


def _collect_scheduled_siblings(
    scheduler: VLLMv1EngineReplicaScheduler,
) -> list[Batch]:
    first_batch = scheduler._schedule_decode_attn_only()
    assert first_batch is not None
    return [first_batch, *list(scheduler._af_pending_micro_batches)]


def test_batch_declares_core_decode_attn_scheduler_identity_fields() -> None:
    batch = Batch(
        replica_id=0,
        requests=[_make_decode_request(0)],
        num_tokens=[1],
        is_moe=True,
    )

    assert getattr(batch, "replay_decode_token_index", "missing") is None
    assert getattr(batch, "decode_attn_cohort_id", "missing") is None
    assert getattr(batch, "decode_attn_cohort_request_ids", "missing") is None


def test_writer_freezes_uniform_schedule_time_decode_token_index() -> None:
    requests = [_make_decode_request(index, 1) for index in range(4)]
    scheduler = _make_writer_scheduler(requests)

    siblings = _collect_scheduled_siblings(scheduler)

    assert [
        getattr(batch, "replay_decode_token_index", None) for batch in siblings
    ] == [1, 1]


def test_writer_freezes_each_stage_head_identity_for_mixed_online_decode_positions() -> None:
    requests = [
        _make_decode_request(0, 2),
        _make_decode_request(1, 1),
    ]
    scheduler = _make_writer_scheduler(requests)

    siblings = _collect_scheduled_siblings(scheduler)

    assert [
        batch.replay_decode_token_index for batch in siblings
    ] == [2, 1]


def test_writer_assigns_shared_global_id_to_stage_siblings() -> None:
    requests = [_make_decode_request(index, 1) for index in range(4)]
    scheduler = _make_writer_scheduler(requests, initial_global_id=17)

    siblings = _collect_scheduled_siblings(scheduler)

    assert [batch.global_id for batch in siblings] == [17, 17]
    assert scheduler._batch_creation_counter == 19


def test_writer_assigns_one_cohort_identity_to_stage_siblings() -> None:
    requests = [_make_decode_request(index, 1) for index in range(4)]
    scheduler = _make_writer_scheduler(requests)

    siblings = _collect_scheduled_siblings(scheduler)

    cohort_ids = {
        getattr(batch, "decode_attn_cohort_id", None) for batch in siblings
    }
    assert len(cohort_ids) == 1
    assert None not in cohort_ids
    expected_request_ids = tuple(request.id for request in requests)
    assert all(
        getattr(batch, "decode_attn_cohort_request_ids", None)
        == expected_request_ids
        for batch in siblings
    )


@pytest.mark.parametrize(
    ("num_requests", "num_stages", "expected_active_stages"),
    [
        (4, 2, {0, 1}),
        (1, 4, {0}),
        (4, 1, {0}),
    ],
    ids=["multi-stage", "fewer-requests-than-stages", "single-stage"],
)
def test_writer_initializes_exact_state_for_every_active_stage(
    num_requests: int,
    num_stages: int,
    expected_active_stages: set[int],
) -> None:
    requests = [_make_decode_request(index, 1) for index in range(num_requests)]
    scheduler = _make_writer_scheduler(requests, num_stages=num_stages)

    siblings = _collect_scheduled_siblings(scheduler)

    cohort_id = siblings[0].decode_attn_cohort_id
    cohort_state = scheduler._decode_attn_active_cohort_states[cohort_id]
    assert cohort_state["active_stage_indices"] == expected_active_stages
    assert cohort_state["stage_phases"] == {
        stage_idx: "local_attn" for stage_idx in expected_active_stages
    }
    assert cohort_state["stage_current_layer_ids"] == {
        stage_idx: 0 for stage_idx in expected_active_stages
    }


def test_stage_phase_update_preserves_untouched_local_stage_visibility() -> None:
    requests = [_make_decode_request(index, 1) for index in range(4)]
    scheduler = _make_writer_scheduler(requests, num_stages=2)
    stage_zero, stage_one = _collect_scheduled_siblings(scheduler)

    scheduler.set_decode_attn_cohort_phase_for_batch(
        stage_zero,
        phase="ffn_inflight",
        layer_id=0,
    )

    cohort_state = scheduler._decode_attn_active_cohort_states[
        stage_zero.decode_attn_cohort_id
    ]
    assert cohort_state["af_phase"] == "mixed"
    assert cohort_state["stage_phases"] == {
        0: "ffn_inflight",
        1: "local_attn",
    }
    assert cohort_state["stage_current_layer_ids"] == {0: 0, 1: 0}
    assert stage_one.afd_stage_idx == 1
    assert scheduler.get_decode_attn_active_stage_slots(
        phase="ffn_inflight",
        layer_id=0,
    ) == (0,)
    assert scheduler.get_decode_attn_active_stage_slots(
        phase="local_attn",
        layer_id=0,
    ) == (1,)


class _DecodeAttnScheduleHarness:
    def __init__(
        self,
        *,
        active_request_ids: set[int],
        pending_siblings: list[SimpleNamespace] | None = None,
        new_batch: SimpleNamespace | None = None,
    ) -> None:
        self._cluster_type = ClusterType.DECODE_ATTN
        self._replica_id = 0
        self._replica_local_id = 0
        self._num_running_batches = 0
        self._af_pipeline_num_micro_batch = 2
        self._af_immediate_batch_queue = []
        self._af_pending_micro_batches = deque(pending_siblings or [])
        self._decode_attn_active_request_ids = set(active_request_ids)
        self._continuation_request_ids = set()
        self._new_batch = new_batch
        self.get_next_batch_calls = 0

    def _get_next_batch(self, is_micro_batch: bool):
        assert is_micro_batch is True
        self.get_next_batch_calls += 1
        if self._af_pending_micro_batches:
            return self._af_pending_micro_batches.popleft()
        batch, self._new_batch = self._new_batch, None
        return batch


def test_active_cohort_blocks_unrelated_new_decode_attn_cohort() -> None:
    active_request = _make_decode_request(0, 1)
    unrelated_request = _make_decode_request(1, 1)
    unrelated_batch = SimpleNamespace(id=101, requests=[unrelated_request])
    scheduler = _DecodeAttnScheduleHarness(
        active_request_ids={active_request.id},
        new_batch=unrelated_batch,
    )

    scheduled = BaseReplicaScheduler.on_schedule(scheduler, time=1.0)

    assert scheduled == []
    assert scheduler.get_next_batch_calls == 0


def test_active_cohort_allows_pending_stage_sibling_to_drain() -> None:
    active_request = _make_decode_request(0, 1)
    pending_batch = SimpleNamespace(id=100, requests=[active_request])
    scheduler = _DecodeAttnScheduleHarness(
        active_request_ids={active_request.id},
        pending_siblings=[pending_batch],
    )

    scheduled = BaseReplicaScheduler.on_schedule(scheduler, time=1.0)

    assert scheduled == [pending_batch]
    assert scheduler.get_next_batch_calls == 1


def _make_completion_scheduler(
    requests: list[Request],
    *,
    cohort_id: int,
    pending_request_ids: set[int],
) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._num_running_batches = len(requests)
    scheduler._running_requests = list(requests)
    scheduler._active_batch_request_counts = {
        request.id: 1 for request in requests
    }
    scheduler._decode_attn_active_request_ids = {
        request.id for request in requests
    }
    scheduler._decode_attn_active_cohort_states = {
        cohort_id: {
            "all_request_ids": {request.id for request in requests},
            "pending_request_ids": set(pending_request_ids),
            "af_phase": "local_attn",
            "active_stage_indices": {0, 1},
            "stage_phases": {0: "local_attn", 1: "local_attn"},
            "stage_current_layer_ids": {0: 0, 1: 0},
        }
    }
    scheduler._scheduled_num_computed_tokens_by_request = {}
    return scheduler


def _make_cohort_completion_batch(
    request: Request,
    *,
    cohort_id: int,
    cohort_request_ids: tuple[int, ...],
) -> Batch:
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[1],
        is_moe=True,
    )
    batch.decode_attn_cohort_id = cohort_id
    batch.decode_attn_cohort_request_ids = cohort_request_ids
    return batch


def test_cohort_remains_active_until_every_request_finishes_the_step() -> None:
    requests = [_make_decode_request(index, 1) for index in range(2)]
    cohort_id = 7
    scheduler = _make_completion_scheduler(
        requests,
        cohort_id=cohort_id,
        pending_request_ids={request.id for request in requests},
    )
    batch = _make_cohort_completion_batch(
        requests[0],
        cohort_id=cohort_id,
        cohort_request_ids=tuple(request.id for request in requests),
    )

    scheduler.on_batch_end(batch)

    state = scheduler._decode_attn_active_cohort_states[cohort_id]
    assert state["pending_request_ids"] == {requests[1].id}
    assert scheduler._decode_attn_active_request_ids == {
        request.id for request in requests
    }


def test_last_request_completion_releases_and_removes_cohort() -> None:
    requests = [_make_decode_request(index, 1) for index in range(2)]
    cohort_id = 7
    scheduler = _make_completion_scheduler(
        requests,
        cohort_id=cohort_id,
        pending_request_ids={requests[1].id},
    )
    batch = _make_cohort_completion_batch(
        requests[1],
        cohort_id=cohort_id,
        cohort_request_ids=tuple(request.id for request in requests),
    )

    scheduler.on_batch_end(batch)

    assert scheduler._decode_attn_active_request_ids == set()
    assert cohort_id not in scheduler._decode_attn_active_cohort_states


def test_completed_member_does_not_pin_mixed_cohort() -> None:
    requests = [_make_decode_request(index, 1) for index in range(2)]
    requests[0]._completed = True
    cohort_id = 7
    scheduler = _make_completion_scheduler(
        requests,
        cohort_id=cohort_id,
        pending_request_ids={request.id for request in requests},
    )
    scheduler._free_request_resources = lambda _request: None
    batch = Batch(
        replica_id=0,
        requests=requests,
        num_tokens=[1, 1],
        is_moe=True,
    )
    batch.decode_attn_cohort_id = cohort_id
    batch.decode_attn_cohort_request_ids = tuple(
        request.id for request in requests
    )

    scheduler.on_batch_end(batch)

    assert scheduler._decode_attn_active_request_ids == set()
    assert cohort_id not in scheduler._decode_attn_active_cohort_states
    assert scheduler._running_requests == [requests[1]]
