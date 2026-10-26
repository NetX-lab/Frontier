"""NIXL READ completion through DECODE engine steps (nixl_read_completion='engine_step').

vLLM's schedule pass hands the requests whose KV it should pull to the NIXL
connector. The worker posts those READs when it starts executing that pass's
step (`start_load_kv`), finds them complete at the start of a later step
(`get_finished`), and that step's output reports them to the scheduler
(`finished_recving`). With a batch queue deeper than one, the step that posts a
READ is not the next one to start, and the output that reports it is not the
next one to be popped, so each READ is kept under the step that owns it.
"""

from typing import List, Optional, Sequence

from frontier.entities.batch import Batch, DummyForwardBatch


class NixlReadCompletion:
    """Claim, post, find and report the NIXL READs of one DECODE lane."""

    def register_nixl_read(self, transfer_info) -> None:
        """Request the NIXL READ of one reserved request; the lane's next schedule pass claims it."""
        self._nixl_reads_requested.append(transfer_info)

    def _claim_nixl_reads(self, batches: Sequence[Batch]) -> None:
        """Hand the requested READs to the schedule pass that formed `batches`.

        Its first batch posts them when it starts. A pass that formed no batch
        still queues an entry that executes as the lane's next step, so the
        lane's next step start of any kind posts them.
        """
        if not self._nixl_reads_requested:
            return
        if batches:
            self._nixl_reads_claimed[batches[0].id] = self._nixl_reads_requested
        else:
            self._nixl_reads_unposted.extend(self._nixl_reads_requested)
        self._nixl_reads_requested = []

    def on_engine_step_start(self, time: float, step: Optional[Batch] = None) -> None:
        """Start one engine step: a batch, a dummy forward, or with no `step` a connector-only step.

        The step first finds the posted READs whose wire time has ended
        (`get_finished`) and keeps them until its output, then posts the READs
        it carries (`start_load_kv`), so the step that posts a READ never finds
        it. A connector-only step reports what it finds at once.
        """
        posted = []
        found = []
        for wire_end, transfer_info in self._nixl_reads_posted:
            if wire_end <= time:
                found.append(transfer_info)
            else:
                posted.append((wire_end, transfer_info))
        to_post = self._nixl_reads_unposted
        if step is None:
            self._nixl_reads_reported.extend(found)
        elif isinstance(step, DummyForwardBatch):
            self._nixl_reads_found_by_dummy_forward.extend(found)
        else:
            if found:
                self._nixl_reads_found_by_batch[step.id] = found
            to_post = to_post + self._nixl_reads_claimed.pop(step.id, [])
        posted.extend(
            (time + transfer_info.transfer_time_ms * 1e-3, transfer_info)
            for transfer_info in to_post
        )
        self._nixl_reads_posted = posted
        self._nixl_reads_unposted = []

    def _report_nixl_reads_of_batch(self, batch: Batch) -> None:
        """The engine takes a batch's output, which reports the READs its start found."""
        self._nixl_reads_reported.extend(self._nixl_reads_found_by_batch.pop(batch.id, ()))

    def _report_nixl_reads_of_dummy_forward(self) -> None:
        """The engine's dummy forward ended; its step reports the READs it found."""
        self._nixl_reads_reported.extend(self._nixl_reads_found_by_dummy_forward)
        self._nixl_reads_found_by_dummy_forward = []

    def consume_found_nixl_reads(self) -> List:
        """The found READs whose step output the engine took, to complete now."""
        reported, self._nixl_reads_reported = self._nixl_reads_reported, []
        return reported

    @property
    def next_nixl_read_end(self) -> Optional[float]:
        """Earliest wire end among the posted READs."""
        return min((wire_end for wire_end, _ in self._nixl_reads_posted), default=None)

    @property
    def _has_nixl_reads(self) -> bool:
        """Whether a request waits for its NIXL READ (`WAITING_FOR_REMOTE_KVS`)."""
        return bool(self._nixl_reads_posted) or self._has_nixl_reads_off_wire

    @property
    def _has_nixl_reads_off_wire(self) -> bool:
        """Whether a READ waits for a step to post it or for a step output to report it."""
        return bool(
            self._nixl_reads_requested
            or self._nixl_reads_claimed
            or self._nixl_reads_unposted
            or self._nixl_reads_found_by_batch
            or self._nixl_reads_found_by_dummy_forward
            or self._nixl_reads_reported
        )
