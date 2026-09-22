"""vLLM V1's data-parallel engine iteration, replayed on the CPU.

Reference: vLLM v0.10.2 as checked out in `.real-engine/vLLM-BS` at
`ea95f571e`. `EngineCore.step_with_batch_queue` (`vllm/v1/engine/core.py`)
decides what one iteration does, `DPEngineCoreProc._maybe_publish_request_counts`
decides whether that iteration publishes its populations, and `run_busy_loop`
advances the step counter that orders the publications.

Why it exists: under pipeline parallelism an engine publishes counts at
iterations a completion report cannot express, and the mapping from iteration
to publication is what Step 9 changes in Frontier. Replaying a scripted history
here gives an expectation written from the reference instead of from a Frontier
run.

What it deliberately leaves out: the coordinator's latch and the frontend's
score. `VllmDPLoadBalancer` already models both from the same source, so
`ReferenceDeployment` feeds the real balancer rather than carrying a second
copy of them. Wave resets are outside Step 9's scope, so the step counter here
only increases; the coordinator compares keys and never reads their magnitude.
The counter also advances on an iteration the reference would skip while every
engine is idle, which shifts later key values but not their order or equality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer


@dataclass(frozen=True)
class ScheduledBatch:
    """What one `schedule()` result does to an engine's populations.

    `admitted` requests move from waiting to running when the batch is
    scheduled. `finished` requests leave running when the batch's output is
    applied, which under pipeline parallelism is a later iteration.
    `num_scheduled_tokens` is zero for the empty scheduler output a busy engine
    can still produce; the reference reads it as `model_executed`.
    """

    admitted: int = 0
    finished: int = 0
    num_scheduled_tokens: int = 1


@dataclass(frozen=True)
class Iteration:
    """The scripted inputs of one engine iteration.

    `batch` is what `schedule()` returned, or `None` when the engine had
    nothing to schedule and only drains a queued output. `oldest_ready` is
    whether the oldest queued batch's future had already completed, the
    reference's third condition for returning without applying an output.
    """

    arrivals: int = 0
    batch: ScheduledBatch | None = None
    oldest_ready: bool = False


@dataclass(frozen=True)
class IterationRecord:
    """What one iteration did, and whether it published its populations."""

    step: int
    scheduled: bool
    applied_output: bool
    load: RequestLoad
    published: bool


@dataclass(frozen=True)
class Publication:
    """One engine's changed counts, as the coordinator receives them."""

    time: float
    engine: int
    step: int
    load: RequestLoad


class ReferenceEngine:
    """One DP engine's busy loop over a scripted history.

    `queue_depth` is `batch_queue_size`, which the reference sets to the
    pipeline-parallel size. At depth one the blocking path runs every
    iteration, which is what the engine does when the batch queue is absent:
    the same mechanism, not a special case.
    """

    def __init__(self, queue_depth: int) -> None:
        if type(queue_depth) is not int or queue_depth < 1:
            raise ValueError(
                f"queue depth must be a positive int, got {queue_depth!r}"
            )
        self._depth = queue_depth
        self._queue: deque[ScheduledBatch] = deque()
        self.waiting = 0
        self.running = 0
        self.step_counter = 0
        self._last_load = RequestLoad(0, 0)

    @property
    def load(self) -> RequestLoad:
        return RequestLoad(self.waiting, self.running)

    def step(self, iteration: Iteration) -> IterationRecord:
        """Run one iteration and return what the engine would publish."""

        self.waiting += iteration.arrivals
        batch = iteration.batch
        if batch is None and not self._queue:
            raise ValueError(
                "the reference steps an engine only while it holds requests or "
                "a queued batch"
            )
        if batch is not None:
            if batch.admitted > self.waiting:
                raise ValueError(
                    f"cannot admit {batch.admitted} of {self.waiting} waiting "
                    "requests"
                )
            self.waiting -= batch.admitted
            self.running += batch.admitted
            self._queue.appendleft(batch)
            if (
                batch.num_scheduled_tokens > 0
                and len(self._queue) < self._depth
                and not iteration.oldest_ready
            ):
                return self._record(scheduled=True, applied_output=False)

        oldest = self._queue.pop()
        if oldest.finished > self.running:
            raise ValueError(
                f"cannot finish {oldest.finished} of {self.running} running "
                "requests"
            )
        self.running -= oldest.finished
        return self._record(scheduled=batch is not None, applied_output=True)

    def _record(self, *, scheduled: bool, applied_output: bool) -> IterationRecord:
        """Publish changed counts under the pre-increment step counter."""

        load = self.load
        published = load != self._last_load
        if published:
            self._last_load = load
        record = IterationRecord(
            step=self.step_counter,
            scheduled=scheduled,
            applied_output=applied_output,
            load=load,
            published=published,
        )
        self.step_counter += 1
        return record


class ReferenceDeployment:
    """DP engines behind one coordinator and one frontend.

    The engines are scripted here; the coordinator's latch, its publish
    deadlines and the frontend's score come from `VllmDPLoadBalancer`. Times
    are simulated seconds, as they are for the balancer.
    """

    def __init__(self, *, num_engines: int, queue_depth: int) -> None:
        self.engines = [ReferenceEngine(queue_depth) for _ in range(num_engines)]
        self.balancer = VllmDPLoadBalancer(num_engines)
        self.publications: list[Publication] = []

    def step(self, time: float, engine: int, iteration: Iteration) -> IterationRecord:
        record = self.engines[engine].step(iteration)
        if record.published:
            self.balancer.report(time, engine, record.step, record.load)
            self.publications.append(
                Publication(time, engine, record.step, record.load)
            )
        return record

    def route(self, time: float) -> int:
        return self.balancer.select(time)
