"""Each batch carries the vLLM engine loop's idle time before the iteration that forms it.

The CPU probe keys its rows by the same interval: from the end of the engine's
previous loop iteration or DP dummy forward to the step's schedule call
(``frontier/profiling/cpu_overhead/vllm_cpu_probe.py``).
"""

import math
from types import SimpleNamespace

import pytest

from frontier.entities import Request
from tests.unit.test_vllm_v1_skipped_waiting_order import _build_scheduler


def _batch() -> SimpleNamespace:
    return SimpleNamespace(engine_idle_time=0.0)


def test_engine_idle_time_runs_from_the_last_batch_output_schedule_or_dummy_forward() -> None:
    scheduler = _build_scheduler()
    scheduler._cluster_scheduler.on_replica_batch_scheduled = lambda *_args: None
    scheduler.add_request(Request(0.0, 8, 2))

    # The engine's first batch follows no iteration.
    [first] = scheduler.on_schedule(1.0)
    assert first.engine_idle_time == math.inf

    first.on_batch_end(1.02)
    scheduler.on_batch_end(first)
    after_output, same_pass = _batch(), _batch()
    scheduler._start_iterations([after_output, same_pass], 1.12)
    # vLLM forms each batch in its own loop iteration, so a second batch of one
    # schedule pass follows the first with no idle time.
    assert after_output.engine_idle_time == pytest.approx(0.1)
    assert same_pass.engine_idle_time == 0.0

    # A dummy pass ends the engine's idle time when its last forward ends.
    scheduler._dummy_forwards_in_flight = 2
    assert scheduler.on_dummy_forward_end(1.4) is False
    assert scheduler.on_dummy_forward_end(1.5) is True
    after_dummy = _batch()
    scheduler._start_iterations([after_dummy], 1.75)
    assert after_dummy.engine_idle_time == pytest.approx(0.25)
