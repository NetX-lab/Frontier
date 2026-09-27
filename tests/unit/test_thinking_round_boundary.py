"""A hidden thinking round keeps its iteration timing until it is requeued.

The batch that ends a hidden round still records per-token metrics for that
round after the request enters its tool wait, so the round's scheduling state
must stay readable until the requeue resets it.
"""

from frontier.entities.request import Request, RequestRoundPlan
from frontier.types import ClusterType


def test_a_hidden_round_keeps_its_iteration_timing_until_requeue() -> None:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=8,
        num_decode_tokens=2,
        thinking_depth=2,
        thinking_round_plans=[RequestRoundPlan(3, 4), RequestRoundPlan(8, 2)],
    )
    request.bind_thinking_home_queue(ClusterType.MONOLITHIC, 0, None)
    request.on_batch_schedule(0.0, ClusterType.MONOLITHIC)
    request.on_batch_end(1.0, 3, ClusterType.MONOLITHIC)
    for step_end in (2.0, 3.0, 4.0):
        request.on_batch_schedule(step_end - 1.0, ClusterType.MONOLITHIC)
        request.on_batch_end(step_end, 1, ClusterType.MONOLITHIC)
    assert request.completed

    request.begin_thinking_tool_wait(4.0)

    assert request.has_started_decode
    assert request.latest_iteration_scheduling_delay == 0.0

    request.finish_thinking_tool_wait_and_requeue(4.5)
    assert not request.scheduled
    assert request.current_thinking_round_index == 1
