"""vLLM V1 internal DP selection and coordinator count publication.

Reference: vLLM v0.10.2, `vllm/v1/engine/core_client.py` (the frontend's engine
selection) and `vllm/v1/engine/coordinator.py` (the coordinator's count
publication). Every constant below is cited against that source.

What is modeled: which engine a request is routed to, given counts the frontend
observes with a delay, and when the coordinator publishes a new snapshot of
those counts. What is not modeled: IPC transport latency, more than one
frontend, elastic scaling, and the coordinator's warm-start publication phase.

Timers advance lazily. This object creates no events, so it cannot keep a
drained simulation alive; a deadline that has passed is applied the next time
the balancer is consulted.
"""

from frontier.logger import init_logger
from frontier.scheduler.request_load import RequestLoad

logger = init_logger(__name__)

# `score = waiting * 4 + running` (reference `core_client.py:1146`).
WAITING_SCORE_WEIGHT = 4
# `wait_for = stats_update_interval_ms if stats_changed else 5000`
# (reference `coordinator.py:195-198`), whose default is 100
# (`coordinator.py:116`, `:122`, `:130`).
CHANGED_PUBLISH_INTERVAL_MS = 100
UNCHANGED_PUBLISH_INTERVAL_MS = 5000
# `min_timeout = 50 if last_step_counts is None else 0`
# (reference `coordinator.py:201-203`).
SNAPSHOT_COLLECTION_WAIT_MS = 50


class VllmDPLoadBalancer:
    """Own one frontend's load estimates and its coordinator's report state.

    The frontend's estimate (`frontend_counts`) is a delayed copy of the
    coordinator's authoritative per-engine counts (`engine_counts`). Selection
    reads the estimate and reserves against it locally; a new snapshot replaces
    the estimate wholesale, exactly as `core_client.py:1073-1078` assigns
    `self.lb_engines = sliced_counts`.
    """

    def __init__(self, num_engines: int):
        self.engine_counts = [RequestLoad(0, 0) for _ in range(num_engines)]
        self.frontend_counts = list(self.engine_counts)
        self.last_step_counts: list[RequestLoad] | None = None
        self.last_report_step = -1
        self.stats_changed = False
        # The reference's coordinator starts with `last_publish_time = 0` while
        # the clock reads epoch milliseconds, so `wait_for - elapsed` is deeply
        # negative on the first iteration and the 50 ms collection wait decides
        # the first publish. Seeding the last publish one unchanged interval in
        # the past reproduces that first deadline without special-casing it.
        self.last_publish_ms = -UNCHANGED_PUBLISH_INTERVAL_MS
        self.next_publish_ms = SNAPSHOT_COLLECTION_WAIT_MS

    def _poll_deadline(self, now_ms: int) -> int:
        """Return the next publish time, as the reference's poll timeout does.

        The reference waits `max(min_timeout, wait_for - elapsed)` from now
        (`coordinator.py:205-206`), which is the same instant as
        `max(now + min_timeout, last_publish + wait_for)`.
        """

        interval = (
            CHANGED_PUBLISH_INTERVAL_MS
            if self.stats_changed
            else UNCHANGED_PUBLISH_INTERVAL_MS
        )
        collection_wait = (
            SNAPSHOT_COLLECTION_WAIT_MS if self.last_step_counts is None else 0
        )
        return max(now_ms + collection_wait, self.last_publish_ms + interval)

    def _advance(self, time: float, *, report_arriving: bool = False) -> int:
        """Apply every publish deadline that has passed, then adopt `time`.

        `report_arriving` breaks the tie at an exact deadline. A real poller
        returns the waiting message rather than timing out, so a report that
        lands on its deadline is processed before the publish it triggers.
        """

        now_ms = int(time * 1000)
        while (
            self.next_publish_ms < now_ms
            or self.next_publish_ms == now_ms
            and not report_arriving
        ):
            published_ms = self.next_publish_ms
            if self.last_step_counts is not None:
                # A snapshot latched for the previous step is published first
                # and consumed (reference `coordinator.py:208-211`).
                self.frontend_counts = self.last_step_counts
                self.last_step_counts = None
            else:
                self.frontend_counts = list(self.engine_counts)
                self.stats_changed = False
            self.last_publish_ms = published_ms
            self.next_publish_ms = self._poll_deadline(published_ms)
        return now_ms

    def report(self, time: float, engine: int, step: int, load: RequestLoad) -> None:
        """Record one engine's counts after one engine iteration.

        Ordering is advisory, as in the reference: a strictly newer step latches
        the previous counts when there are unpublished changes, an equal step is
        the expected path for a peer engine reporting the same forward, and an
        out-of-order step only warns (`coordinator.py:293-310`). The counts are
        applied either way; nothing here aborts a run.
        """

        if load == self.engine_counts[engine]:
            # The reference engine compares against its own `last_counts` and
            # emits no coordinator message when they match
            # (`core.py:1080-1087`). That value never diverges from the
            # coordinator's copy, which every report overwrites.
            return
        now_ms = self._advance(time, report_arriving=True)
        if step > self.last_report_step:
            if self.stats_changed:
                self.last_step_counts = list(self.engine_counts)
            self.last_report_step = step
        elif step < self.last_report_step:
            logger.warning(
                "Received DP load stats for out-of-order step %d from engine %d "
                "(expected >= %d); counts are still applied",
                step,
                engine,
                self.last_report_step,
            )
        self.engine_counts[engine] = load
        self.stats_changed = True
        self.next_publish_ms = self._poll_deadline(now_ms)

    def select(self, time: float) -> int:
        """Choose the first minimum-score engine and reserve local waiting load.

        The reference scans from the frontend's own start index with a strict
        `<`, so the lowest-index minimum wins, then increments that engine's
        waiting estimate by the frontend count (`core_client.py:1139-1153`).
        One modeled frontend means that reservation is exactly one request.
        """

        self._advance(time)
        engine = min(
            range(len(self.frontend_counts)),
            key=lambda index: (
                WAITING_SCORE_WEIGHT * self.frontend_counts[index].waiting
                + self.frontend_counts[index].running
            ),
        )
        load = self.frontend_counts[engine]
        self.frontend_counts[engine] = RequestLoad(load.waiting + 1, load.running)
        return engine
