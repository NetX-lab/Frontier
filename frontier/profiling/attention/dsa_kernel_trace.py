"""Kernel-only timing of DSA attention scopes from a torch.profiler trace.

The vLLM runner wraps every measured scope in a ``record_function`` range
named by ``annotation_name``. A scope's kernel-only time is the summed device
duration of the kernels launched inside that range: each CUDA runtime or
driver launch on the annotating CPU thread is matched to its range by start
time, and to its kernels by correlation id. Memory copies and memsets are not
kernels and are excluded, as in ``RecordFunctionTracer``.

Scopes never nest, so one sorted pass per thread assigns every launch; the
whole trace is processed in ``O(n log n)``.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ANNOTATION_PREFIX = "frontier_dsa"
_SEPARATOR = "|"
_ANNOTATION_CATEGORY = "user_annotation"
_LAUNCH_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
_KERNEL_CATEGORY = "kernel"


@dataclass(frozen=True)
class ScopeKey:
    step_index: int
    global_layer_id: int
    scope: str


@dataclass(frozen=True)
class ScopeKernelTime:
    kernel_time_ms: float
    kernel_count: int


def annotation_name(step_index: int, global_layer_id: int, scope: str) -> str:
    return _SEPARATOR.join(
        (ANNOTATION_PREFIX, str(step_index), str(global_layer_id), scope)
    )


def parse_annotation_name(name: Any) -> ScopeKey | None:
    if not isinstance(name, str) or not name.startswith(ANNOTATION_PREFIX + _SEPARATOR):
        return None
    parts = name.split(_SEPARATOR)
    if len(parts) != 4:
        raise ValueError(f"Malformed DSA annotation {name!r}")
    _, step_index, layer_id, scope = parts
    return ScopeKey(int(step_index), int(layer_id), scope)


def _thread(event: dict[str, Any]) -> tuple[Any, Any]:
    return event.get("pid"), event.get("tid")


def aggregate_kernel_trace(
    events: Iterable[dict[str, Any]],
) -> dict[ScopeKey, ScopeKernelTime]:
    """Return the kernel-only time and kernel count of every DSA scope."""

    ranges: dict[tuple[Any, Any], list[tuple[float, float, ScopeKey]]] = defaultdict(list)
    launches: list[dict[str, Any]] = []
    kernels: dict[Any, list[float]] = defaultdict(list)
    for event in events:
        category = event.get("cat")
        if category == _ANNOTATION_CATEGORY:
            key = parse_annotation_name(event.get("name"))
            if key is not None:
                start = float(event["ts"])
                ranges[_thread(event)].append((start, start + float(event["dur"]), key))
        elif category in _LAUNCH_CATEGORIES:
            launches.append(event)
        elif category == _KERNEL_CATEGORY:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                kernels[correlation].append(float(event["dur"]))

    starts: dict[tuple[Any, Any], list[float]] = {}
    for thread, thread_ranges in ranges.items():
        thread_ranges.sort(key=lambda item: item[0])
        for previous, current in zip(thread_ranges, thread_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"DSA scopes overlap on one thread: {previous[2]} and {current[2]}"
                )
        starts[thread] = [item[0] for item in thread_ranges]

    totals_us: dict[ScopeKey, float] = {}
    counts: dict[ScopeKey, int] = {}
    for thread_ranges in ranges.values():
        for _, _, key in thread_ranges:
            if key in totals_us:
                raise ValueError(f"DSA scope {key} appears twice in one trace")
            totals_us[key] = 0.0
            counts[key] = 0

    for launch in launches:
        thread = _thread(launch)
        thread_starts = starts.get(thread)
        if not thread_starts:
            continue
        timestamp = float(launch["ts"])
        position = bisect.bisect_right(thread_starts, timestamp) - 1
        if position < 0:
            continue
        start, end, key = ranges[thread][position]
        if timestamp > end:
            continue
        correlation = launch.get("args", {}).get("correlation")
        for duration in kernels.get(correlation, ()):
            totals_us[key] += duration
            counts[key] += 1

    empty = sorted(
        (key for key, count in counts.items() if count == 0),
        key=lambda key: (key.step_index, key.global_layer_id, key.scope),
    )
    if empty:
        raise ValueError(
            f"{len(empty)} DSA scopes launched no kernel, first {empty[0]}; the "
            "profiler trace is incomplete or a launch was not correlated"
        )
    return {
        key: ScopeKernelTime(kernel_time_ms=totals_us[key] * 1e-3, kernel_count=counts[key])
        for key in totals_us
    }


def load_trace_events(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


__all__ = [
    "ANNOTATION_PREFIX",
    "ScopeKernelTime",
    "ScopeKey",
    "aggregate_kernel_trace",
    "annotation_name",
    "load_trace_events",
    "parse_annotation_name",
]
