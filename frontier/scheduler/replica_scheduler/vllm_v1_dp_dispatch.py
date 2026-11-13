"""Host dispatch agreement and device drain of profiled PREFILL dummy passes."""

from typing import Dict, List, Optional, Tuple

from frontier.types import ClusterType


class DataParallelDispatchBarrier:
    """Keep an all-dummy cohort's host iteration independent of GPU completion."""

    def _initialize_dp_dispatch(self) -> None:
        get_cadence = getattr(self._predictor, "get_dp_dummy_step_cadence_ms", None)
        cadence_ms = get_cadence() if get_cadence is not None else None
        if cadence_ms is not None and (
            not self._runs_dp_dummy_passes or self._cluster_type != ClusterType.PREFILL
        ):
            raise ValueError("DP dummy-step host cadence requires PREFILL DP dummy passes")
        self._dp_dummy_step_cadence: Optional[float] = (
            None if cadence_ms is None else cadence_ms * 1e-3
        )
        self._dp_dispatch_records: Dict[int, Tuple[float, bool]] = {}
        self._paced_dummy_passes: set[int] = set()
        self._host_returns_at: Optional[float] = None
        self._dp_dispatch_host_ticks: List[Tuple[int, float]] = []
        self._dummy_unfinished_snapshot: Optional[bool] = None

    def _record_dp_dispatch(self, ordinal: int, time: float, *, is_dummy: bool) -> None:
        if getattr(self, "_dp_dummy_step_cadence", None) is None:
            return
        self._dp_dispatch_records[ordinal] = (time, is_dummy)
        group = self._dp_group()
        if not all(ordinal in lane._dp_dispatch_records for lane in group):
            return
        records = [lane._dp_dispatch_records[ordinal] for lane in group]
        if all(dummy for _, dummy in records):
            host_return = max(issue_time for issue_time, _ in records) + self._dp_dummy_step_cadence
            for lane in group:
                lane._dummy_forwards_in_flight = 0
                lane._paced_dummy_passes.add(ordinal)
                lane._host_returns_at = host_return
                self._dp_dispatch_host_ticks.append((lane._replica_local_id, host_return))
        for lane in group:
            del lane._dp_dispatch_records[ordinal]

    def _record_dummy_dispatch(self, ordinal: int, time: float) -> None:
        if getattr(self, "_dp_dummy_step_cadence", None) is not None:
            self._dummy_unfinished_snapshot = self.has_unfinished_requests
            self._record_dp_dispatch(ordinal, time, is_dummy=True)

    def _dp_dispatch_host_ready(self, time: float) -> bool:
        host_return = getattr(self, "_host_returns_at", None)
        if host_return is None:
            return True
        if time < host_return:
            return False
        self._last_iteration_end = host_return
        self._host_returns_at = None
        return True

    def _take_dummy_unfinished_snapshot(self) -> Optional[bool]:
        if getattr(self, "_dp_dummy_step_cadence", None) is None:
            return None
        snapshot, self._dummy_unfinished_snapshot = self._dummy_unfinished_snapshot, None
        return snapshot

    def consume_dp_dispatch_host_ticks(self) -> List[Tuple[int, float]]:
        """Return and clear the host continuations completed by this recording lane."""
        ticks = getattr(self, "_dp_dispatch_host_ticks", [])
        self._dp_dispatch_host_ticks = []
        return ticks

    def consume_paced_dummy_end(self, ordinal: int) -> bool:
        """Drain a paced device pass without ending or resuming a host iteration."""
        if getattr(self, "_dp_dummy_step_cadence", None) is None or ordinal not in self._paced_dummy_passes:
            return False
        self._paced_dummy_passes.remove(ordinal)
        return True

    def _validate_blocking_dummy_end(self) -> None:
        if (
            getattr(self, "_dp_dummy_step_cadence", None) is not None
            and self._dummy_forwards_in_flight == 0
        ):
            raise ValueError("Unknown DP dummy pass ended without a blocking dummy forward")
