"""Correlate Kineto device activity with CPU scopes, without importing torch."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


_GPU_LAUNCH_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch",
                        "cuGraphLaunch", "cudaMemcpy", "cuMemcpy")


@dataclass(frozen=True)
class _CpuScope:
    name: str
    thread: tuple
    start: float
    end: float

    def contains(self, other):
        return (self.thread == other.thread
                and self.start <= other.start and other.end <= self.end)


def _duration(event):
    duration = event.get("dur")
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not isfinite(duration) or duration < 0):
        raise ValueError(f"Invalid Kineto duration for {event.get('name')!r}: {duration!r}")
    return duration


def _cpu_scope(event):
    start = event.get("ts")
    if (not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isfinite(start) or "pid" not in event or "tid" not in event):
        raise ValueError(f"CPU scope requires finite ts and pid/tid: {event.get('name')!r}")
    return _CpuScope(event.get("name", ""), (event["pid"], event["tid"]),
                     start, start + _duration(event))


def parse_device_activity(path, operator_names, iteration_prefix="step5_iteration/"):
    """Return per-iteration operator and layer GPU-active durations, in milliseconds.

    ``operator_names`` are the expected active outer owners in every iteration;
    inactive contract operators must be omitted by the caller. Root names must
    be unique. CPU annotations and launches must have Kineto ``pid``/``tid``,
    ``ts``/``dur`` and launch/device links must use ``args.correlation``.

    Each kernel or gpu_memcpy trace record contributes once, even when runtime
    and driver aliases share its correlation. Association uses CPU launch
    containment, never device timestamps. Nested operator annotations charge
    only their outermost owner. Layer duration is the sum of these device
    records, including copies; it is neither elapsed latency nor an interval
    union when device work overlaps. Preparation launched outside the iteration
    roots and device records without CPU launch links are excluded, with
    ``skipped_kernel_count`` and ``skipped_memory_copy_count`` reporting their
    totals. ``skipped_uncorrelated_activity_count`` is the subset with absent
    correlations or CPU launches, not proven preparation. Measured kernel,
    graph and memcpy launches require correlated activity; event/sync APIs do
    not. Ambiguous ownership and missing positive activity for an expected
    owner raise ValueError.
    """
    operators = tuple(operator_names)
    if (not operators or any(not isinstance(name, str) or not name for name in operators)
            or len(set(operators)) != len(operators) or "step5_layer_e2e" in operators):
        raise ValueError("operator_names must contain distinct active operator names")
    if not isinstance(iteration_prefix, str) or not iteration_prefix:
        raise ValueError("iteration_prefix must be a nonempty string")

    events = json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]
    roots = {}
    roots_by_thread = defaultdict(list)
    annotations = []
    launches = []
    device_by_correlation = defaultdict(list)
    for event in events:
        category = event.get("cat")
        if category == "user_annotation":
            name = event.get("name", "")
            if name.startswith(iteration_prefix):
                scope = _cpu_scope(event)
                if name in roots:
                    raise ValueError(f"Duplicate iteration annotation: {name!r}")
                roots[name] = scope
                roots_by_thread[scope.thread].append(scope)
            elif name in operators:
                annotations.append(_cpu_scope(event))
        elif category in ("cuda_runtime", "cuda_driver"):
            launches.append(event)
        elif category in ("kernel", "gpu_memcpy"):
            correlation = event.get("args", {}).get("correlation")
            device_by_correlation[correlation].append(event)
    if not roots:
        raise ValueError(f"No iteration annotations with prefix {iteration_prefix!r}")

    starts_by_thread = {}
    for thread, scopes in roots_by_thread.items():
        scopes.sort(key=lambda scope: scope.start)
        if any(left.end > right.start for left, right in zip(scopes, scopes[1:])):
            raise ValueError(f"Overlapping iteration annotations on CPU thread {thread!r}")
        starts_by_thread[thread] = [scope.start for scope in scopes]

    def iteration_for(scope):
        index = bisect_right(starts_by_thread.get(scope.thread, ()), scope.start) - 1
        if index >= 0:
            root = roots_by_thread[scope.thread][index]
            if root.contains(scope):
                return root.name
            if scope.start < root.end:
                raise ValueError(f"CPU scope/launch {scope.name!r} crosses iteration boundary: {root.name!r}")
        return None

    operators_by_iteration = defaultdict(list)
    for annotation in annotations:
        iteration = iteration_for(annotation)
        if iteration is not None:
            operators_by_iteration[iteration].append(annotation)

    owners_by_correlation = defaultdict(set)
    unmeasured_correlations = set()
    for event in launches:
        correlation = event.get("args", {}).get("correlation")
        launch = _cpu_scope(event)
        iteration = iteration_for(launch)
        if iteration is None:
            if correlation is not None:
                unmeasured_correlations.add(correlation)
            continue
        if correlation is None or correlation not in device_by_correlation:
            if launch.name.startswith(_GPU_LAUNCH_PREFIXES):
                raise ValueError(f"Measured GPU launch {launch.name!r} in {iteration!r} "
                                 f"has no matching correlation/device activity: {correlation!r}")
            continue
        candidates = [scope for scope in operators_by_iteration[iteration]
                      if scope.contains(launch)]
        outermost = [scope for scope in candidates
                     if all(scope.contains(other) for other in candidates)]
        owners = {scope.name for scope in outermost}
        if len(owners) != 1:
            raise ValueError(f"GPU correlation {correlation!r} in {iteration!r} has "
                             f"{'ambiguous' if candidates else 'no'} operator owner")
        owners_by_correlation[correlation].add((iteration, owners.pop()))

    per_iteration = {name: {"step5_layer_e2e": 0.} for name in roots}
    kernel_names, copy_names = set(), set()
    kernel_count = copy_count = 0
    kernel_time_ms = copy_time_ms = 0.
    skipped_kernel_count = skipped_copy_count = skipped_uncorrelated_count = 0
    for correlation, activity in device_by_correlation.items():
        owners = owners_by_correlation[correlation]
        if not owners:
            skipped_kernel_count += sum(event["cat"] == "kernel" for event in activity)
            skipped_copy_count += sum(event["cat"] == "gpu_memcpy" for event in activity)
            if correlation not in unmeasured_correlations:
                skipped_uncorrelated_count += len(activity)
            continue
        if len(owners) > 1:
            raise ValueError(f"GPU correlation {correlation!r} has conflicting "
                             f"iteration/operator owners: {sorted(owners)!r}")
        if correlation in unmeasured_correlations:
            raise ValueError(f"GPU correlation {correlation!r} is shared by measured and unmeasured launches")
        iteration, owner = next(iter(owners))
        times = per_iteration[iteration]
        for event in activity:
            duration_ms = _duration(event) / 1000
            times[owner] = times.get(owner, 0.) + duration_ms
            times["step5_layer_e2e"] += duration_ms
            if event["cat"] == "kernel":
                kernel_names.add(event["name"])
                kernel_count += 1
                kernel_time_ms += duration_ms
            else:
                copy_names.add(event["name"])
                copy_count += 1
                copy_time_ms += duration_ms
    for iteration, times in per_iteration.items():
        missing = [name for name in operators if times.get(name, 0.) <= 0]
        if missing:
            raise ValueError(f"Iteration {iteration!r} is missing GPU activity for active operators: {missing!r}")

    return {"per_iteration": per_iteration,
            "kernel_names": sorted(kernel_names), "kernel_count": kernel_count,
            "kernel_time_ms": kernel_time_ms,
            "memory_copy_names": sorted(copy_names), "memory_copy_count": copy_count,
            "memory_copy_time_ms": copy_time_ms,
            "skipped_kernel_count": skipped_kernel_count,
            "skipped_memory_copy_count": skipped_copy_count,
            "skipped_uncorrelated_activity_count": skipped_uncorrelated_count}
