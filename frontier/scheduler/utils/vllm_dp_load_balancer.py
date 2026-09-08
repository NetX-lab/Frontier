"""vLLM V1 internal DP selection and coordinator count publication.

Reference: vLLM v0.10.2 core_client.py and coordinator.py. Transport latency is
not modeled. Timer expirations are evaluated before their next observation.
"""

from math import isfinite

from frontier.scheduler.request_load import RequestLoad


class VllmDPLoadBalancer:
    """Own one frontend's load estimates and its coordinator's report state."""

    def __init__(self, num_engines: int):
        if num_engines < 1:
            raise ValueError("DP load balancing requires at least one engine")
        self.engine_counts = [RequestLoad(0, 0) for _ in range(num_engines)]
        self.frontend_counts = list(self.engine_counts)
        self.last_step_counts: list[RequestLoad] | None = None
        self.last_report_step = -1
        self.stats_changed = False
        self.last_publish_ms = -5000
        self.next_publish_ms = 50
        self.time_ms = 0

    def _poll_deadline(self, now_ms: int) -> int:
        interval = 100 if self.stats_changed else 5000
        collection_wait = 50 if self.last_step_counts is None else 0
        return max(now_ms + collection_wait, self.last_publish_ms + interval)

    def _advance(self, time: float, *, report_arriving: bool = False) -> int:
        if not isfinite(time) or time < 0:
            raise ValueError("DP load balancing requires finite nonnegative time")
        now_ms = int(time * 1000)
        if now_ms < self.time_ms:
            raise ValueError("DP load balancing time cannot move backwards")
        while (
            self.next_publish_ms < now_ms
            or self.next_publish_ms == now_ms and not report_arriving
        ):
            published_ms = self.next_publish_ms
            if self.last_step_counts is not None:
                self.frontend_counts = self.last_step_counts
                self.last_step_counts = None
            else:
                self.frontend_counts = list(self.engine_counts)
                self.stats_changed = False
            self.last_publish_ms = published_ms
            self.next_publish_ms = self._poll_deadline(published_ms)
        self.time_ms = now_ms
        return now_ms

    def report(self, time: float, engine: int, step: int, load: RequestLoad) -> None:
        """Publish changed engine counts after one completed engine step."""
        if not 0 <= engine < len(self.engine_counts):
            raise ValueError("DP load report references an unknown engine")
        if step < 0 or min(load) < 0:
            raise ValueError("DP load report requires nonnegative step and counts")
        if load == self.engine_counts[engine]:
            # The upstream engine emits no coordinator message in this case.
            return
        now_ms = self._advance(time, report_arriving=True)
        if step > self.last_report_step:
            if self.stats_changed:
                self.last_step_counts = list(self.engine_counts)
            self.last_report_step = step
        self.engine_counts[engine] = load
        self.stats_changed = True
        self.next_publish_ms = self._poll_deadline(now_ms)

    def select(self, time: float) -> int:
        """Choose the first minimum-score engine and reserve local waiting load."""
        self._advance(time)
        engine = min(
            range(len(self.frontend_counts)),
            key=lambda index: (
                4 * self.frontend_counts[index].waiting
                + self.frontend_counts[index].running
            ),
        )
        load = self.frontend_counts[engine]
        self.frontend_counts[engine] = RequestLoad(load.waiting + 1, load.running)
        return engine
