"""Behavior coverage for one shared monolithic forward across mixed lanes.

A monolithic Replica runs prefill and decode on the same attention-DP lanes, so
one forward step can hold a prefill batch on one lane and a pure-decode batch on
another. These tests drive the real synchronization entry, the real EP wave, and
the real collective completion for every phase pairing, and check the properties
the shared lifecycle owes each source: one shared identity, one completion, one
ownership restoration, and a continuation predicted from the source's own batch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent
from frontier.events.decode_sync_event import DecodeSyncEvent
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent
from frontier.events.prefill_sync_event import PrefillSyncEvent
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    FULL_STAGE_WORLD,
    StageExecutionContext,
)
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils.sync_state import initialize_sync_waiting_rooms
from frontier.types import ClusterType

#: Layers owned by pipeline stage 0 in this fixture. Layer 0 is the layer the
#: tests synchronize on, so the next layer (1) stays inside the stage and the
#: continuation path is the interesting one; layer 3 is the stage tail.
_GLOBAL_ID_BASE = 0
STAGE_LAYERS = 4
TOTAL_LAYERS = 4
NUM_LANES = 2


class _ModelConfig:
    is_moe = True
    num_layers = TOTAL_LAYERS

    def __init__(self, dense_layers: frozenset[int] = frozenset()) -> None:
        self._dense_layers = dense_layers

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id not in self._dense_layers


class _ExecutionTime:
    """A single-layer prediction whose value is a function of its own batch.

    Every duration is derived from one token count, so a lane that continued on
    a peer's prediction rather than on its own produces a visibly wrong time
    instead of an equal one.
    """

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.pipeline_time = 0.0
        self.model_time = float(tokens)
        self.total_time = float(tokens)
        self.decode_draft_proposer_time = 0.0
        self.expert_parallel_communication_time = 0.0

    def get_single_layer_attention_scope_time(self) -> float:
        return float(self.tokens)

    def get_single_layer_post_attention_time(self) -> float:
        # The EP decomposition is checked for conservation, so this must equal
        # the sum of the five phase times below.
        return float(self.tokens) + 2.0

    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return 0.0

    def get_single_layer_moe_dispatch_time(self) -> float:
        return 1.0

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return float(self.tokens)

    def get_single_layer_moe_combine_time(self) -> float:
        return 1.0

    def get_single_layer_moe_post_combine_time(self) -> float:
        return 0.0


class _LanePredictor:
    _num_layers_per_pipeline_stage = STAGE_LAYERS

    def __init__(self) -> None:
        # (layer_id, the token count of the batch the call was made for).
        self.calls: list[tuple[int, int]] = []
        self._monolithic_routing_details = {
            0: {
                layer_id: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
                for layer_id in range(TOTAL_LAYERS)
            }
        }
        # The disaggregated roles read the same routing from their own table.
        self._prefill_routing_details = self._monolithic_routing_details
        self._decode_routing_details = self._monolithic_routing_details

    def predict_stage_execution_time(
        self,
        batch,
        _stage_id,
        cluster_type=None,
        *,
        num_layers,
        layer_id,
        include_ffn=True,
        include_attention=True,
    ):
        del cluster_type, num_layers, include_ffn, include_attention
        per_expert = getattr(batch, "per_expert_tokens", None)
        tokens = (
            sum(per_expert.values())
            if per_expert is not None
            else int(batch.total_num_tokens)
        )
        self.calls.append((layer_id, tokens))
        return _ExecutionTime(tokens)


def _request(prefill_tokens: int, decode_tokens: int, *, decoding: bool) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=decode_tokens,
    )
    if decoding:
        # Batch.num_prefill_tokens credits a request's tokens to prefill only
        # while its prefill is unfinished, so this is what makes a lane decode.
        request._is_prefill_complete = True
        request._num_processed_tokens = prefill_tokens
    return request


def _prefill_batch(tokens: int) -> Batch:
    batch = Batch(0, [_request(tokens, 4, decoding=False)], [tokens], is_moe=True)
    return batch


def _decode_batch(tokens: int) -> Batch:
    requests = [_request(8, 4, decoding=True) for _ in range(tokens)]
    return Batch(0, requests, [1] * tokens, is_moe=True)


def _mixed_batch(prefill_tokens: int, decode_requests: int) -> Batch:
    """A prefill-mode batch that also carries already-decoding requests."""

    requests = [_request(prefill_tokens, 4, decoding=False)]
    requests.extend(_request(8, 4, decoding=True) for _ in range(decode_requests))
    return Batch(
        0, requests, [prefill_tokens] + [1] * decode_requests, is_moe=True
    )


class _MetricsStore:
    def __init__(self, *, reporting: bool) -> None:
        self.ep_wave_reporting_enabled = reporting
        self.stage_execution_reporting_enabled = reporting
        self.wave_calls = 0
        self.stage_schedules: list[tuple] = []

    def on_ep_wave_schedule(self, *_args, **_kwargs):
        self.wave_calls += 1

    def on_replica_stage_schedule(self, *args, **_kwargs):
        self.stage_schedules.append(args)


def _build_scheduler(
    *,
    dense_layers: frozenset[int] = frozenset(),
    lane_capacity: int = NUM_LANES,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
):
    predictor = _LanePredictor()
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=_ModelConfig(dense_layers),
            total_expert_num=4,
            moe_expert_parallel_size=2,
            moe_tensor_parallel_size=1,
            router_topk=1,
            num_pipeline_stages=1,
        )
    )
    scheduler._predictor = predictor
    scheduler._forward_sync_state = ForwardSyncState()
    scheduler._replica_dp_size = NUM_LANES
    initialize_sync_waiting_rooms(scheduler)

    context = StageExecutionContext(
        replica_id=0, stage_id=0, ep_size=2, full_stage_capacity=lane_capacity
    )
    scheduler._stage_execution_contexts = {(0, 0): context}
    stages = {
        lane: ReplicaStageScheduler(
            replica_id=0,
            stage_id=0,
            is_last_stage=True,
            is_moe=True,
            execution_time_predictor=predictor,
            cluster_type=cluster_type,
            replica_local_id=lane,
            stage_execution_context=context,
        )
        for lane in range(NUM_LANES)
    }
    scheduler._replica_schedulers = {
        (0, lane): SimpleNamespace(
            get_replica_stage_scheduler=lambda _stage_id, stage=stage: stage
        )
        for lane, stage in stages.items()
    }
    scheduler.get_replica_stage_scheduler = (
        lambda _replica_id, lane, _stage_id: stages[lane if lane is not None else 0]
    )
    return scheduler, predictor, context, stages


def _admit(stages, batch: Batch, lane: int) -> None:
    """Admit one lane's batch through the real stage queue.

    Going through `add_batch` / `pop_batch_if_not_busy` is what marks the lane's
    stage busy and binds its forward group, so a sibling lane holding the other
    phase is treated as occupied rather than as a lane that can be filled with
    an idle batch.
    """

    batch.set_global_id(NUM_LANES * _GLOBAL_ID_BASE + lane)
    batch._stage_owner_replica_local_id = lane
    stages[lane].add_batch(batch)
    assert stages[lane].pop_batch_if_not_busy() is batch
    batch._prefill_model_execution_components_ms_by_stage = {0: [1.0]}
    batch._prefill_stage_start_time = 0.0
    batch._decode_stage_start_time = 0.0


def _global(scheduler):
    """The one global-scheduler method a cluster-internal event calls."""

    return SimpleNamespace(get_cluster_scheduler=lambda _: scheduler)


def _enter(scheduler, batch: Batch, lane: int, layer_id: int, metrics_store=None):
    """Enter one lane through the phase its own batch is in."""

    entry = (
        scheduler.on_prefill_sync
        if batch.num_prefill_tokens > 0
        else scheduler.on_decode_sync
    )
    return entry(
        0.0, 0, 0, batch, lane, "pre_moe", layer_id, 0.0, metrics_store=metrics_store
    )


def _run_forward(
    scheduler, stages, lane_batches: dict[int, Batch], layer_id: int, metrics_store
):
    """Admit every lane, then complete the resulting collective event."""

    for lane, batch in lane_batches.items():
        _admit(stages, batch, lane)
    events: list = []
    for lane, batch in lane_batches.items():
        events.extend(_enter(scheduler, batch, lane, layer_id, metrics_store))
    collective = [
        event
        for event in events
        if isinstance(event, (PrefillSyncCollectiveEvent, DecodeSyncCollectiveEvent))
    ]
    assert len(collective) == 1, events
    event = collective[0]
    # Completion goes through the real event handler, so the test exercises the
    # dispatch that routes a monolithic cohort into the shared forward path.
    return event, event.handle_event(_global(scheduler), metrics_store)


PHASE_BUILDERS = {
    "prefill": lambda tokens: _prefill_batch(tokens),
    "decode": lambda tokens: _decode_batch(tokens),
    "mixed": lambda tokens: _mixed_batch(tokens - 1, 1),
}


@pytest.mark.parametrize("lane_zero_phase", sorted(PHASE_BUILDERS))
@pytest.mark.parametrize("lane_one_phase", sorted(PHASE_BUILDERS))
def test_every_phase_pairing_reaches_one_shared_completion(
    lane_zero_phase: str, lane_one_phase: str
) -> None:
    scheduler, _predictor, _context, stages = _build_scheduler()
    metrics_store = _MetricsStore(reporting=False)
    batches = {
        0: PHASE_BUILDERS[lane_zero_phase](4),
        1: PHASE_BUILDERS[lane_one_phase](6),
    }
    event, follow_on = _run_forward(scheduler, stages, batches, 0, metrics_store)

    # One forward, one identity, one collective event for both lanes.
    assert batches[0]._forward_cohort_id == batches[1]._forward_cohort_id
    assert event._batch_global_id == batches[0]._forward_cohort_id
    # Each live lane continues on its own next-layer sync event.
    assert len(follow_on) == 2
    assert {event._batch.id for event in follow_on} == {
        batches[0].id,
        batches[1].id,
    }
    assert all(event._layer_id == 1 for event in follow_on)
    # The room is consumed: nothing is left waiting for a second completion.
    room = scheduler._forward_sync_waiting_room[0][0][event._batch_global_id][0]
    assert "post_moe" not in room


@pytest.mark.parametrize(
    "phases", [("prefill", "decode"), ("decode", "prefill"), ("mixed", "decode")]
)
def test_the_collective_event_class_follows_cohort_contents_not_arrival_order(
    phases: tuple[str, str],
) -> None:
    """Events order by `(time, event_type, id)`, so the class is a priority.

    It has to be a function of the cohort, not of which lane happened to close
    the room, or a mixed forward's ordering would depend on arrival order.
    """

    lane_zero_phase, lane_one_phase = phases
    observed = []
    for order in ((0, 1), (1, 0)):
        scheduler, _predictor, _context, stages = _build_scheduler()
        batches = {
            0: PHASE_BUILDERS[lane_zero_phase](4),
            1: PHASE_BUILDERS[lane_one_phase](6),
        }
        for lane, batch in batches.items():
            _admit(stages, batch, lane)
        events: list = []
        for lane in order:
            events.extend(_enter(scheduler, batches[lane], lane, 0, None))
        collective = [
            event
            for event in events
            if isinstance(
                event, (PrefillSyncCollectiveEvent, DecodeSyncCollectiveEvent)
            )
        ]
        assert len(collective) == 1
        observed.append(type(collective[0]))
    assert observed[0] is observed[1]
    # A cohort holding any prefill token keeps the prefill collective type it
    # already has today; only a cohort with none of them decodes.
    assert observed[0] is PrefillSyncCollectiveEvent


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_reversed_arrival_order_gives_the_same_identity_and_outcome(order) -> None:
    scheduler, _predictor, _context, stages = _build_scheduler()
    batches = {0: _prefill_batch(4), 1: _decode_batch(6)}
    for lane, batch in batches.items():
        _admit(stages, batch, lane)
    events: list = []
    for lane in order:
        events.extend(_enter(scheduler, batches[lane], lane, 0, None))
    collective = [
        event
        for event in events
        if isinstance(event, (PrefillSyncCollectiveEvent, DecodeSyncCollectiveEvent))
    ]
    assert len(collective) == 1
    assert batches[0]._forward_cohort_id == batches[1]._forward_cohort_id
    follow_on = collective[0].handle_event(_global(scheduler), None)
    assert sorted(event._batch.id for event in follow_on) == sorted(
        batch.id for batch in batches.values()
    )


def test_each_source_continues_on_its_own_predicted_duration() -> None:
    """A deliberate unequal-token fixture detects borrowed timing.

    `_ExecutionTime` makes every duration a function of the batch it was
    predicted for, so a lane that reused a peer's prediction lands at the
    peer's time instead of its own.
    """

    scheduler, predictor, _context, stages = _build_scheduler()
    batches = {0: _prefill_batch(4), 1: _decode_batch(6)}
    event, follow_on = _run_forward(scheduler, stages, batches, 0, None)

    continuation = {event._batch.id: event for event in follow_on}
    prefill_event = continuation[batches[0].id]
    decode_event = continuation[batches[1].id]
    assert isinstance(prefill_event, PrefillSyncEvent)
    assert isinstance(decode_event, DecodeSyncEvent)
    # 4 tokens -> 4 ms, 6 tokens -> 6 ms, both measured from the wave end.
    assert prefill_event.time == pytest.approx(event.time + 4e-3)
    assert decode_event.time == pytest.approx(event.time + 6e-3)
    assert prefill_event.time != decode_event.time
    # Each lane's own token count appears in its own next-layer prediction.
    next_layer_calls = [call for call in predictor.calls if call[0] == 1]
    assert sorted(next_layer_calls) == [(1, 4), (1, 6)]


@pytest.mark.parametrize(
    "cluster_type,build,tokens",
    [
        (ClusterType.PREFILL, _prefill_batch, (4, 9)),
        (ClusterType.DECODE, _decode_batch, (2, 7)),
    ],
)
def test_a_disaggregated_role_continues_each_lane_on_its_own_duration(
    cluster_type, build, tokens
) -> None:
    """The per-phase helpers the shared forward delegates to also serve the
    PDD PREFILL and DECODE roles, so their attention-DP lanes each continue on
    their own prediction too, rather than on the first lane's.
    """

    scheduler, predictor, _context, stages = _build_scheduler(cluster_type=cluster_type)
    batches = {lane: build(count) for lane, count in enumerate(tokens)}
    event, follow_on = _run_forward(scheduler, stages, batches, 0, None)

    continuation = {event._batch.id: event for event in follow_on}
    for lane, count in enumerate(tokens):
        assert continuation[batches[lane].id].time == pytest.approx(
            event.time + count * 1e-3
        )
    next_layer_calls = [call for call in predictor.calls if call[0] == 1]
    assert sorted(next_layer_calls) == [(1, count) for count in tokens]


def test_an_idle_participant_does_not_gain_requests_progress_or_a_continuation() -> None:
    scheduler, _predictor, _context, stages = _build_scheduler()
    batch = _prefill_batch(4)
    _admit(stages, batch, 0)
    # Lane 1 has no work at all, so the entry fills it with an idle batch.
    events = _enter(scheduler, batch, 0, 0, None)
    idle_entries = [event for event in events if event._batch.is_idle]
    assert len(idle_entries) == 1
    idle_batch = idle_entries[0]._batch
    assert idle_batch.requests == []
    assert idle_batch._forward_cohort_id == batch._forward_cohort_id

    follow_on = idle_entries[0].handle_event(_global(scheduler), None)
    collective = [
        event
        for event in follow_on
        if isinstance(event, (PrefillSyncCollectiveEvent, DecodeSyncCollectiveEvent))
    ]
    assert len(collective) == 1
    completion = collective[0].handle_event(_global(scheduler), None)
    # Only the real source continues; the idle lane produces no event, no
    # request and no completed work.
    assert [event._batch.id for event in completion] == [batch.id]
    assert idle_batch.requests == []
    assert not getattr(idle_batch, "_prefill_ep_wave_lane_times_ms", None)


def test_successive_forwards_release_owners_rooms_and_open_step_bindings() -> None:
    scheduler, _predictor, _context, stages = _build_scheduler()
    batches = {0: _prefill_batch(4), 1: _decode_batch(6)}
    for lane, batch in batches.items():
        _admit(stages, batch, lane)

    cohort_ids = []
    for layer_id in range(3):
        events: list = []
        for lane, batch in batches.items():
            events.extend(_enter(scheduler, batch, lane, layer_id, None))
        collective = [
            event
            for event in events
            if isinstance(
                event, (PrefillSyncCollectiveEvent, DecodeSyncCollectiveEvent)
            )
        ]
        assert len(collective) == 1
        cohort_ids.append(collective[0]._batch_global_id)
        follow_on = collective[0].handle_event(_global(scheduler), None)
        assert len(follow_on) == 2
        # Every layer consumes its own room and its own open-step binding.
        room = scheduler._forward_sync_waiting_room[0][0][cohort_ids[-1]][layer_id]
        assert "post_moe" not in room
        assert scheduler._forward_sync_state.open_steps("forward") == {}
        assert scheduler._forward_sync_state.open_steps("prefill") == {}
        assert scheduler._forward_sync_state.open_steps("decode") == {}
        # Ownership is restored to exactly one full-stage ticket per live lane.
        assert all(
            batch._stage_admission_ticket.scope is FULL_STAGE_WORLD
            for batch in batches.values()
        )
        assert batches[0]._forward_cohort_id == batches[1]._forward_cohort_id
    # Each layer opens and closes its own binding, so the step id advances once
    # per layer while the two lanes keep agreeing on it.
    assert cohort_ids == sorted(set(cohort_ids)) and len(cohort_ids) == 3


def test_a_decoding_request_inside_a_prefill_batch_advances_exactly_one_layer() -> None:
    scheduler, _predictor, _context, stages = _build_scheduler()
    mixed = _mixed_batch(4, 1)
    decoding = mixed.requests[1]
    prefilling = mixed.requests[0]
    batches = {0: mixed, 1: _decode_batch(2)}
    peer_requests = list(batches[1].requests)
    _run_forward(scheduler, stages, batches, 0, None)

    assert decoding.completed_layer_count == 1
    # A request still prefilling has no decode layer to credit.
    assert prefilling.completed_layer_count == 0
    assert all(request.completed_layer_count == 1 for request in peer_requests)


def test_a_dense_layer_labels_each_source_by_its_own_phase() -> None:
    scheduler, _predictor, _context, stages = _build_scheduler(
        dense_layers=frozenset({1})
    )
    batches = {0: _prefill_batch(4), 1: _decode_batch(6)}
    for lane, batch in batches.items():
        _admit(stages, batch, lane)
    events: list = []
    for lane, batch in batches.items():
        events.extend(_enter(scheduler, batch, lane, 1, None))

    # A dense layer inside a MoE model produces no EP collective at all.
    assert all(isinstance(event, DenseLayerCompleteEvent) for event in events)
    assert len(events) == 2
    modes = {event._batch.id: event._phase for event in events}
    assert modes == {batches[0].id: "prefill", batches[1].id: "decode"}
    # Only the prefill source keeps a prefill component ledger entry.
    assert batches[0]._prefill_model_execution_components_ms_by_stage[0] == [1.0, 6.0]
    assert batches[1]._prefill_model_execution_components_ms_by_stage[0] == [1.0]
    # Each dense source is timed from its own tokens: 4 + 2 and 6 + 2 ms.
    assert {event._batch.id: event.time for event in events} == {
        batches[0].id: pytest.approx(6e-3),
        batches[1].id: pytest.approx(8e-3),
    }


@pytest.mark.parametrize("lane_zero_phase", ["mixed", "prefill"])
def test_a_dense_layer_credits_only_the_requests_that_are_decoding(
    lane_zero_phase: str,
) -> None:
    """A dense layer completes per source, and each source credits its decoders.

    The decoding member of a mixed prefill-mode source advances by one layer,
    the request still prefilling does not, and the pure-decode peer lane
    advances by one through its own handler. A pure-prefill source is the
    control: it has nothing to credit, and nothing is credited.
    """

    scheduler, _predictor, _context, stages = _build_scheduler(
        dense_layers=frozenset({1})
    )
    batches = {0: PHASE_BUILDERS[lane_zero_phase](5), 1: _decode_batch(2)}
    for lane, batch in batches.items():
        _admit(stages, batch, lane)
    events: list = []
    for lane, batch in batches.items():
        events.extend(_enter(scheduler, batch, lane, 1, None))
    assert [type(event) for event in events] == [DenseLayerCompleteEvent] * 2
    # Completion runs through the real event handler, per source.
    for event in events:
        event.handle_event(_global(scheduler), None)

    decoding = [r for r in batches[0].requests if r.is_prefill_complete]
    prefilling = [r for r in batches[0].requests if not r.is_prefill_complete]
    assert len(decoding) == (1 if lane_zero_phase == "mixed" else 0)
    assert [r.completed_layer_count for r in decoding] == [1] * len(decoding)
    assert [r.completed_layer_count for r in prefilling] == [0]
    assert [r.completed_layer_count for r in batches[1].requests] == [1, 1]


def test_a_mixed_source_is_credited_once_per_layer_across_routed_and_dense() -> None:
    """`MoE -> dense -> MoE`: the decoding member counts 1, 2, 3 -- not 1, 1, 2.

    A routed layer completes once for the whole cohort and a dense layer once
    per source, so the two paths credit through different handlers. What the
    request sees must not depend on which one ran.
    """

    scheduler, _predictor, _context, stages = _build_scheduler(
        dense_layers=frozenset({1})
    )
    batches = {0: _mixed_batch(4, 1), 1: _decode_batch(2)}
    prefilling, decoding = batches[0].requests
    peers = list(batches[1].requests)
    for lane, batch in batches.items():
        _admit(stages, batch, lane)

    credits = []
    for layer_id in range(3):
        events: list = []
        for lane, batch in batches.items():
            events.extend(_enter(scheduler, batch, lane, layer_id, None))
        completions = [
            event
            for event in events
            if isinstance(
                event,
                (
                    PrefillSyncCollectiveEvent,
                    DecodeSyncCollectiveEvent,
                    DenseLayerCompleteEvent,
                ),
            )
        ]
        assert len(completions) == (2 if layer_id == 1 else 1), events
        for event in completions:
            event.handle_event(_global(scheduler), None)
        credits.append(decoding.completed_layer_count)
        assert prefilling.completed_layer_count == 0
        assert [peer.completed_layer_count for peer in peers] == [layer_id + 1] * 2
    assert credits == [1, 2, 3]


def test_a_disabled_metrics_store_costs_the_run_nothing() -> None:
    """Reporting is demand-driven at this boundary.

    A store with reporting off must produce the same simulated outcome and the
    same predictor calls as no store at all, and must record nothing. The
    reporting-enabled comparison needs a real predictor result and is covered
    by `tests/integration/test_monolithic_mixed_forward_runtime.py`.
    """

    outcomes = []
    store = _MetricsStore(reporting=False)
    for metrics_store in (None, store):
        scheduler, predictor, _context, stages = _build_scheduler()
        batches = {0: _prefill_batch(4), 1: _decode_batch(6)}
        event, follow_on = _run_forward(scheduler, stages, batches, 0, metrics_store)
        outcomes.append(
            (
                event.time,
                sorted(item.time for item in follow_on),
                sorted(predictor.calls),
            )
        )

    assert outcomes[0] == outcomes[1]
    assert store.wave_calls == 0
    assert store.stage_schedules == []
