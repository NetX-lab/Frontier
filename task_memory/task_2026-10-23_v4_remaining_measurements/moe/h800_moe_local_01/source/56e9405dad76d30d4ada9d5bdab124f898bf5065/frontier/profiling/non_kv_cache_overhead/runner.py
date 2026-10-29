"""Single-rank orchestration for vLLM-aligned non-KV memory profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from frontier.profiling.non_kv_cache_overhead.memory_accounting import (
    MemoryProfilingResult,
    MemorySnapshot,
    memory_profiling,
)
from frontier.profiling.non_kv_cache_overhead.types import NonKVMemoryBreakdown


ProfileRunCallback = Callable[[], None]


@dataclass(frozen=True)
class SingleRankProfileInput:
    """Input payload for single-rank memory profiling run."""

    profile_run_callback: ProfileRunCallback
    weights_memory_bytes: int
    baseline_snapshot: Optional[MemorySnapshot] = None


@dataclass(frozen=True)
class SingleRankProfileOutput:
    """Output payload for single-rank memory profiling run."""

    breakdown: NonKVMemoryBreakdown
    raw_result: MemoryProfilingResult


def run_single_rank_profile(
    profile_input: SingleRankProfileInput,
) -> SingleRankProfileOutput:
    """Execute one vLLM-like profiling run on a representative rank.

    Flow:
      1. Capture baseline snapshot (or use provided baseline)
      2. Enter memory profiling context
      3. Execute one injected profile-run callback
      4. Return non-KV memory breakdown
    """
    if not isinstance(profile_input, SingleRankProfileInput):
        raise TypeError(
            "profile_input must be SingleRankProfileInput, "
            f"got={type(profile_input)}"
        )

    if not callable(profile_input.profile_run_callback):
        raise TypeError(
            "profile_run_callback must be callable, "
            f"got={type(profile_input.profile_run_callback)}"
        )

    weights_memory_bytes = int(profile_input.weights_memory_bytes)
    if weights_memory_bytes < 0:
        raise ValueError(
            "weights_memory_bytes must be >= 0, "
            f"got={profile_input.weights_memory_bytes!r}"
        )

    baseline_snapshot = profile_input.baseline_snapshot
    if baseline_snapshot is None:
        baseline_snapshot = MemorySnapshot()
    elif not isinstance(baseline_snapshot, MemorySnapshot):
        raise TypeError(
            "baseline_snapshot must be MemorySnapshot or None, "
            f"got={type(baseline_snapshot)}"
        )

    with memory_profiling(
        baseline_snapshot,
        weights_memory=weights_memory_bytes,
    ) as profile_result:
        profile_input.profile_run_callback()

    return SingleRankProfileOutput(
        breakdown=profile_result.to_breakdown(),
        raw_result=profile_result,
    )
