"""Pin vLLM V1's iteration-to-publication mapping under pipeline parallelism.

These are expectations read from the reference (`.real-engine/vLLM-BS` at
`ea95f571e`), not from a Frontier run. They are what a Frontier schedule-time
report has to reproduce, and they are the reason the completion report alone
cannot: at depth one every iteration both schedules and applies, while above it
an iteration can publish a changed population without completing anything.
"""

from __future__ import annotations

from frontier.scheduler.request_load import RequestLoad
from tests.comparison.dp_placement_pp.reference_loop import (
    Iteration,
    ReferenceDeployment,
    ReferenceEngine,
    ScheduledBatch,
)


def run(depth: int, script: list[Iteration]) -> list[tuple]:
    """Return one engine's (step, scheduled, applied, load, published) rows."""

    engine = ReferenceEngine(depth)
    return [
        (r.step, r.scheduled, r.applied_output, tuple(r.load), r.published)
        for r in (engine.step(iteration) for iteration in script)
    ]


def test_without_a_batch_queue_every_iteration_applies_what_it_scheduled():
    rows = run(1, [
        Iteration(arrivals=2, batch=ScheduledBatch(admitted=2)),
        Iteration(batch=ScheduledBatch(finished=1)),
        Iteration(batch=ScheduledBatch(finished=1)),
    ])
    assert [row[2] for row in rows] == [True, True, True]
    assert [row[3] for row in rows] == [(0, 2), (0, 1), (0, 0)]


def test_a_cold_fill_publishes_an_admission_before_anything_completes():
    rows = run(2, [
        Iteration(arrivals=3, batch=ScheduledBatch(admitted=3)),
        Iteration(arrivals=1, batch=ScheduledBatch(admitted=1)),
    ])
    assert rows[0][1:] == (True, False, (0, 3), True)
    assert rows[1][1:] == (True, True, (0, 4), True)


def test_an_already_ready_output_is_applied_in_the_iteration_that_schedules():
    rows = run(3, [
        Iteration(arrivals=4, batch=ScheduledBatch(admitted=2)),
        Iteration(batch=ScheduledBatch(admitted=2), oldest_ready=True),
    ])
    assert rows[0][1:3] == (True, False)
    assert rows[1][1:] == (True, True, (0, 4), True)


def test_an_empty_scheduler_output_does_not_take_the_early_return():
    rows = run(3, [
        Iteration(arrivals=2, batch=ScheduledBatch(admitted=2)),
        Iteration(batch=ScheduledBatch(num_scheduled_tokens=0)),
    ])
    assert rows[1][1:] == (True, True, (0, 2), False)


def test_a_drain_publishes_without_scheduling():
    rows = run(2, [
        Iteration(arrivals=2, batch=ScheduledBatch(admitted=2, finished=2)),
        Iteration(batch=None),
    ])
    assert rows[1][1:] == (False, True, (0, 0), True)


def test_depth_three_allows_two_admissions_before_the_first_completion():
    rows = run(3, [
        Iteration(arrivals=6, batch=ScheduledBatch(admitted=2)),
        Iteration(batch=ScheduledBatch(admitted=2)),
        Iteration(batch=ScheduledBatch(admitted=2)),
    ])
    assert [row[2] for row in rows] == [False, False, True]
    assert [row[0] for row in rows] == [0, 1, 2]
    assert all(row[4] for row in rows)


def test_an_idle_engine_in_a_running_wave_runs_a_dummy_iteration():
    rows = run(2, [Iteration(), Iteration()])
    assert rows == [(0, False, False, (0, 0), False), (1, False, False, (0, 0), False)]


def test_peers_at_the_same_iteration_index_publish_under_one_key():
    deployment = ReferenceDeployment(num_engines=2, queue_depth=2)
    first = deployment.step(0.010, 0, Iteration(arrivals=3, batch=ScheduledBatch(admitted=3)))
    second = deployment.step(0.020, 1, Iteration(arrivals=3, batch=ScheduledBatch(admitted=3)))

    assert first.step == second.step
    # Equal keys apply without latching, so the frontend sees both lanes.
    assert deployment.route(0.080) == 0
    assert deployment.balancer.frontend_counts[1] == RequestLoad(0, 3)


def test_an_idle_peer_keeps_pace_through_dummy_iterations():
    """While the wave runs, an idle engine steps with its peers."""

    deployment = ReferenceDeployment(num_engines=2, queue_depth=2)
    deployment.step(0.005, 0, Iteration(arrivals=2, batch=ScheduledBatch(admitted=2)))
    deployment.step(0.005, 1, Iteration())
    deployment.step(0.006, 0, Iteration(batch=ScheduledBatch()))
    deployment.step(0.006, 1, Iteration())
    peer = deployment.step(0.020, 1, Iteration(arrivals=3, batch=ScheduledBatch(admitted=3)))

    assert deployment.engines[0].step_counter == 2
    assert peer.step == 2
