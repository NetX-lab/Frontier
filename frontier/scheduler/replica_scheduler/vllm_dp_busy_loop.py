"""vLLM DPEngineCoreProc's DP busy loop and host/device forward queue."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

from frontier.entities.batch import Batch, DummyForwardBatch

# vLLM's DPEngineCoreProc runs its DP finish-sync all-reduce after every 32nd
# busy-loop iteration of a wave (`_has_global_unfinished_reqs`).
VLLM_DP_SYNC_INTERVAL = 32


@dataclass
class DeviceForward:
    """One formed forward, retained only until its device work ends."""

    batch_id: int
    dummy: bool
    parts_remaining: int
    formed_at: float


class VllmDPBusyLoop:
    """The DP wave, per-forward host rendezvous, and dummy iterations."""

    def _init_dp_busy_loop(self) -> None:
        # vLLM's DPEngineCoreProc: after a step that executed no batch, an
        # attention-DP engine of a MoE model runs a dummy forward on every
        # pipeline stage while its DP wave runs (`run_busy_loop`), so the other
        # engines' MoE layers always find a partner.
        self._runs_dp_dummy_passes = (
            self._has_engine_batch_queue
            and self._replica_is_moe
            and self._replica_config.attn_dp > 1
        )
        self._dummy_pass_due = False
        self._max_forwards_in_flight = self._config.max_forwards_in_flight
        if self._runs_dp_forwards_ahead:
            if not self._runs_dp_dummy_passes:
                raise ValueError(
                    "max_forwards_in_flight requires a lane that runs MoE "
                    "attention-DP dummy passes"
                )
            if self._num_stages > 1:
                raise ValueError(
                    "max_forwards_in_flight requires num_pipeline_stages=1; "
                    "DP host run-ahead was measured only for one stage"
                )
            self._unfinished_forwards: Deque[DeviceForward] = deque()
        else:
            self._dummy_forwards_in_flight = 0
        self._dp_iteration_resume_at: Optional[float] = None
        self._last_zero_cost_dummy_end: Optional[float] = None
        self._dummy_pass_start: Optional[float] = None
        # Such an engine counts the busy-loop iterations of its DP wave, each
        # running one forward, a batch or a dummy forward. After every
        # VLLM_DP_SYNC_INTERVAL-th, its host waits in an all-reduce over the DP
        # group, which ends the wave once no engine has unfinished requests.
        # A batch that ends during the wait keeps its output, and its queue
        # slot, until a later iteration pops it and applies the output.
        self._dp_wave_running = False
        self._dp_wave_steps = 0
        self._iteration_open = False
        self._waits_at_dp_sync = False
        self._unfinished_at_dp_sync = False
        self._dp_sync_released_lane_ids: List[int] = []
        self._unpopped_outputs: Deque[Callable[[float], List]] = deque()
        self._popped_output_events: List = []
        self._iteration_running_limit = math.inf
        # End of the engine loop's latest iteration: a schedule call that formed
        # a batch, a batch output, or a dummy forward. None before the first.
        # Only the batch-queue roles run vLLM's engine loop.
        self._last_iteration_end: Optional[float] = None

    @property
    def _runs_dp_forwards_ahead(self) -> bool:
        return getattr(self, "_max_forwards_in_flight", None) is not None

    @property
    def _dummy_forwards_in_flight(self) -> int:
        if self._runs_dp_forwards_ahead:
            return sum(forward.parts_remaining for forward in self._unfinished_forwards
                       if forward.dummy)
        return self._dummy_parts_remaining

    @_dummy_forwards_in_flight.setter
    def _dummy_forwards_in_flight(self, parts: int) -> None:
        # The legacy path has one pass. The opt-in path uses only the FIFO.
        self._dummy_parts_remaining = parts

    @property
    def _host_forward_wait(self) -> bool:
        if not self._runs_dp_forwards_ahead:
            return bool(self._dummy_forwards_in_flight)
        # Forward n is already formed. Its host return waits for n-K,
        # so at most K forwards remain unfinished when the iteration ends.
        return (
            len(self._unfinished_forwards) > self._max_forwards_in_flight
            or any(lane._batch_creation_counter < self._batch_creation_counter
                   for lane in self._dp_group())
        )

    def _record_device_forward(self, batch: Batch, time: float) -> None:
        dummy = isinstance(batch, DummyForwardBatch)
        self._unfinished_forwards.append(DeviceForward(batch.id, dummy, 1, time))
        group = self._dp_group()
        if all(lane._batch_creation_counter >= self._batch_creation_counter
               for lane in group):
            self._dp_sync_released_lane_ids.extend(
                lane._replica_local_id for lane in group
                if lane is not self and lane._iteration_open
                and lane._batch_creation_counter == self._batch_creation_counter
            )

    def on_device_forward_end(self, batch: Batch) -> bool:
        """Release device capacity at the last stage, before any output pop."""
        if not self._runs_dp_forwards_ahead:
            return False
        forward = self._unfinished_forwards.popleft()
        assert forward.batch_id == batch.id
        return self._iteration_open and not self._host_forward_wait

    def next_dp_iteration_start(self, time: float) -> Optional[float]:
        """Follow-up pass after formation/release, through ReplicaScheduleEvent."""
        if not self._runs_dp_forwards_ahead:
            return None
        resume_at, self._dp_iteration_resume_at = self._dp_iteration_resume_at, None
        if resume_at is not None:
            return resume_at
        if (self._iteration_open and not self._host_forward_wait
                and self._blocking_batch_id is None and not self._waits_at_dp_sync):
            return time
        return None

    def on_engine_step_start(self, time: float, step: Optional[Batch] = None) -> None:
        """A run-ahead connector step starts when its host forms the forward."""
        if self._runs_dp_forwards_ahead and step is not None:
            if getattr(step, "_host_engine_step_started", False):
                return
            step._host_engine_step_started = True
        super().on_engine_step_start(time, step)

    def _run_dp_iterations(self, time: float) -> List[Batch]:
        """Run DPEngineCoreProc's busy-loop iterations from `time` until the host waits.

        Each iteration schedules at most one batch. It returns at once while
        the batch queue has room and its oldest batch is in flight; otherwise
        it pops the oldest batch, waiting for it if it has not ended, and runs
        a dummy forward if it scheduled nothing (`run_busy_loop`). An
        iteration that waited or ran a dummy forward ends when the engine's
        next pass starts.
        """
        if self._dummy_pass_due:
            self._dummy_pass_due = False
            return self._issue_dummy_pass(time)
        if self._iteration_open:
            self._iteration_open = False
            if self._runs_dp_forwards_ahead:
                self._last_iteration_end = time
                self._report_nixl_reads_of_dummy_forward()
            if self._end_iteration():
                return []
            if self._runs_dp_forwards_ahead and self._nixl_reads_reported:
                # ReplicaScheduleEvent applies these connector outputs inline
                # before the next host pass can schedule their ready requests.
                self._dp_iteration_resume_at = time
                return []
            if self._runs_dp_forwards_ahead and self._dp_wave_running:
                resume_at = self.next_pass_start(time)
                if resume_at > time:
                    self._dp_iteration_resume_at = resume_at
                    return []
        if not self._dp_wave_running and self.has_unfinished_requests:
            # An engine that receives a request starts a wave on every engine.
            for lane in self._dp_group():
                lane._dp_wave_running = True
                if self._runs_dp_forwards_ahead and lane is not self:
                    self._dp_sync_released_lane_ids.append(lane._replica_local_id)
        batches: List[Batch] = []
        while True:
            formed = self._schedule_iteration(time)
            batches.extend(formed)
            if self._unpopped_outputs:
                # The oldest batch ended while the host waited at the DP sync,
                # so this iteration pops it at once and applies its output.
                # The iteration ends with its own forward.
                apply_output = self._unpopped_outputs.popleft()
                self._popped_output_events.extend(apply_output(time))
                if not formed:
                    return batches + self._issue_dummy_pass(time)
            elif not formed or self._num_running_batches == self._max_concurrent_batches:
                if self._in_flight_batch_ids:
                    self._blocking_batch_id = self._in_flight_batch_ids[0]
                    self._dummy_pass_due = not formed
                    self._iteration_open = True
                    self._clear_followup_polls()
                elif self._dp_wave_running:
                    return self._issue_dummy_pass(time)
                return batches
            if self._runs_dp_forwards_ahead:
                self._iteration_open = True
                return batches
            if self._end_iteration():
                self._clear_followup_polls()
                return batches

    def _schedule_iteration(self, time: float) -> List[Batch]:
        """Schedule call of one busy-loop iteration: at most one batch."""
        self._iteration_running_limit = self._num_running_batches + 1
        batches = self._start_iterations(super().on_schedule(time), time)
        self._iteration_running_limit = math.inf
        self._claim_nixl_reads(batches)
        if self._runs_dp_forwards_ahead:
            for batch in batches:
                self.on_engine_step_start(time, batch)
        self._in_flight_batch_ids.extend(batch.id for batch in batches)
        return batches

    def _end_iteration(self) -> bool:
        """Count one ended busy-loop iteration; return whether the host then waits at the DP sync."""
        self._dp_wave_steps += 1
        if self._dp_wave_steps % VLLM_DP_SYNC_INTERVAL:
            return False
        self._waits_at_dp_sync = True
        self._unfinished_at_dp_sync = self.has_unfinished_requests
        group = self._dp_group()
        if all(lane._waits_at_dp_sync for lane in group):
            # The all-reduce returns once the last engine joins it, and it ends
            # the wave when no engine joined with unfinished requests.
            wave_continues = any(lane._unfinished_at_dp_sync for lane in group)
            for lane in group:
                lane._waits_at_dp_sync = False
                if not wave_continues:
                    lane._dp_wave_running = False
                    lane._dp_wave_steps = 0
            released = [lane._replica_local_id for lane in group if lane is not self]
            if self._runs_dp_forwards_ahead:
                self._dp_sync_released_lane_ids.extend(released)
            else:
                self._dp_sync_released_lane_ids = released
        return self._waits_at_dp_sync

    def consume_dp_sync_release(self) -> List[int]:
        """Lanes whose host this engine's latest pass released from the DP sync."""
        released, self._dp_sync_released_lane_ids = self._dp_sync_released_lane_ids, []
        return list(dict.fromkeys(released))

    def _dp_group(self) -> List["VLLMv1EngineReplicaScheduler"]:
        return [
            self._cluster_scheduler.get_replica_scheduler(self._replica_id, lane_id)
            for lane_id in range(self._replica_config.attn_dp)
        ]

    def _start_iterations(self, batches: List[Batch], time: float) -> List[Batch]:
        """Give each batch the engine's idle time before the iteration that forms it.

        Each batch is one loop iteration of vLLM's engine, so a second batch of
        the same call follows the first with no idle time. An engine's first
        batch follows no iteration.
        """
        for batch in batches:
            batch.engine_idle_time = (
                math.inf if self._last_iteration_end is None else time - self._last_iteration_end
            )
            if not self._runs_dp_forwards_ahead:
                self._last_iteration_end = time
            else:
                self._record_device_forward(batch, time)
        return batches

    def _clear_followup_polls(self) -> None:
        # No iteration runs while the engine waits; the output or the dummy
        # forward that ends the wait starts the next one.
        self._monolithic_pp_terminal_release_followup_poll_pending = False
        self._monolithic_pp_mtp_output_wait_followup_poll_pending = False

    @property
    def has_unfinished_requests(self) -> bool:
        """vLLM's `Scheduler.has_unfinished_requests`: a waiting or running request.

        A request whose NIXL READ is pending waits as `WAITING_FOR_REMOTE_KVS`.
        """
        return (
            bool(self._running_requests)
            or any(self._waiting_queues())
            or self._has_nixl_reads
        )

    @property
    def _only_posted_nixl_reads_unfinished(self) -> bool:
        """Whether the engine's only unfinished requests, if any, wait for posted NIXL READs."""
        return not (
            self._running_requests
            or any(self._waiting_queues())
            or self._has_nixl_reads_off_wire
        )

    @property
    def engine_loop_idle(self) -> bool:
        """Whether the host waits for work, excluding queued run-ahead dummies."""
        return (
            not self._in_flight_batch_ids
            and not self._host_forward_wait
            and not (self._runs_dp_forwards_ahead and self._iteration_open)
            and not self._waits_at_dp_sync
        )

    def _issue_dummy_pass(self, time: float) -> List[Batch]:
        """Start one dummy forward on every pipeline stage at once.

        `execute_dummy_batch` is one RPC to every worker of the engine. Each
        stage runs its part once its earlier work ends, with no PP transfer
        between parts. The legacy host waits for every part; run-ahead waits
        only for its launch depth and DP peers.
        """
        dummy_forwards = []
        for stage_id in range(self._num_stages):
            dummy_forward = DummyForwardBatch(
                self._replica_id, stage_id, self._batch_creation_counter
            )
            dummy_forward._cluster_type = self._cluster_type
            dummy_forward.decode_cuda_graph_metadata = (
                self._build_decode_cuda_graph_metadata(dummy_forward)
            )
            self._assign_lane_identity(dummy_forward, self._batch_creation_counter)
            dummy_forwards.append(dummy_forward)
        self._batch_creation_counter += 1
        if self._runs_dp_forwards_ahead:
            self._record_device_forward(dummy_forwards[0], time)
            self.on_engine_step_start(time, dummy_forwards[0])
        else:
            self._dummy_forwards_in_flight = self._num_stages
        self._dummy_pass_start = time
        self._iteration_open = True
        self._clear_followup_polls()
        return dummy_forwards

    def on_dummy_forward_end(self, time: float) -> bool:
        """Record one ended dummy forward; return whether the engine iterates again."""
        if self._runs_dp_forwards_ahead:
            forward = self._unfinished_forwards.popleft()
            self._last_zero_cost_dummy_end = time if time == forward.formed_at else None
            return self._iteration_open and not self._host_forward_wait
        self._dummy_forwards_in_flight -= 1
        if self._dummy_forwards_in_flight:
            return False
        self._last_iteration_end = time
        self._report_nixl_reads_of_dummy_forward()
        return True

    def next_pass_start(self, time: float) -> float:
        """Start of the engine's next pass after its dummy forward ended at `time`.

        An execution path may price an idle forward at zero
        (`predict_step_cost_execution`), so a group whose forwards are all
        dummy forwards spins at no simulated cost, and such a dummy forward
        ends when it starts. While no engine of the group then has a request
        to schedule and some wait for posted NIXL READs, no pass changes
        anything until the earliest READ's wire ends; the pass that starts
        then finds it (`get_finished`), as the next step of an idle engine
        does. An engine that receives work earlier wakes the others through
        its forward's first room.
        """
        if self._dummy_pass_start is None or time > self._dummy_pass_start:
            return time
        group = self._dp_group()
        if self._runs_dp_forwards_ahead and not all(
            lane._last_zero_cost_dummy_end == time for lane in group
        ):
            # A host return at formation says nothing about GPU cost. Only
            # device ends at that same formation time witness a zero-cost group.
            return time
        if not all(lane._only_posted_nixl_reads_unfinished for lane in group):
            return time
        return min(
            (lane.next_nixl_read_end for lane in group if lane._nixl_reads_posted),
            default=time,
        )

