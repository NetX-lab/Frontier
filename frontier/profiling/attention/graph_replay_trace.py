"""Join Kineto replay activities to captured operator scopes and eager ranges.

Graph activities use launch correlation and the CUPTI clone bridge; eager
kernels, copies and memsets use the innermost CPU annotation containing their
launch. Container scopes are declared by the integrator's scope tree: they
group child scopes and must own no kernel themselves, while other scopes may
own kernels outside their children. Sorted range indices keep trace attribution
O(n log n), plus the captured dependency edges visited per graph launch. Copies
and memsets remain in the intervals and graph wait calculation, but contribute
no operator kernel time. A step hold launches exactly one sleep kernel; it
is retained as an unowned dependency interval and excluded from all costs.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Collection, Iterable

from frontier.profiling.attention.dsa_kernel_trace import (
    ScopeKernelTime,
    ScopeKey,
    load_trace_events,
    parse_annotation_name,
)
from frontier.profiling.attention.graph_capture import CaptureRecord, GraphNode, ScopeOwner

STEP_ANNOTATION_PREFIX = "frontier_dsa_step"
HOLD_ANNOTATION_PREFIX = "frontier_dsa_hold"
HOLD_ACTIVITY = "hold"
_KERNEL_ACTIVITY = "kernel"
GRID, BLOCK = "grid", "block"
_LAUNCH_DIMENSIONS = 3
_SEPARATOR = "|"
_ANNOTATION_CATEGORY = "user_annotation"
_LAUNCH_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
_ACTIVITIES = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}
_ACTIVITY_NODES = frozenset(_ACTIVITIES.values())
_US_TO_MS = 1e-3


@dataclass(frozen=True)
class ReplayInterval:
    step_index: int
    owner: ScopeOwner | None
    kind: str
    activity: str
    name: str
    start_us: float
    end_us: float
    stream: Any
    graph_id: int | None
    graph_node_id: int | None
    original_node_id: int | None
    capture: int | None
    correlation: Any
    launch_end_us: float
    graph_gap_us: float | None
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None


@dataclass(frozen=True)
class ReplayAttribution:
    scope_times: dict[ScopeKey, ScopeKernelTime]
    layer_windows: dict[tuple[int, int], tuple[float, float]]
    intervals: tuple[ReplayInterval, ...]
    outside_kernel_counts: dict[int, int]
    unattributed_launch_count: int


def step_annotation_name(step_index: int) -> str:
    return _SEPARATOR.join((STEP_ANNOTATION_PREFIX, str(step_index)))


def hold_annotation_name(step_index: int) -> str:
    return _SEPARATOR.join((HOLD_ANNOTATION_PREFIX, str(step_index)))


def parse_step_annotation(name: Any) -> int | None:
    if not isinstance(name, str) or not name.startswith(STEP_ANNOTATION_PREFIX):
        return None
    parts = name.split(_SEPARATOR)
    if len(parts) != 2 or parts[0] != STEP_ANNOTATION_PREFIX:
        raise ValueError(f"Malformed DSA step annotation {name!r}")
    try:
        return int(parts[1])
    except ValueError as error:
        raise ValueError(f"Malformed DSA step annotation {name!r}") from error


def _thread(event: dict[str, Any]) -> tuple[Any, Any]:
    return event.get("pid"), event.get("tid")


@dataclass(frozen=True)
class _Range:
    start: float
    end: float
    key: Any


def _range(event: dict[str, Any], key: Any) -> _Range:
    start = float(event["ts"])
    end = start + float(event["dur"])
    if end < start:
        raise ValueError(f"Negative annotation duration for {key!r}")
    return _Range(start, end, key)


class _RangeIndex:
    """Nonoverlapping step ranges or flattened innermost scope ranges."""

    def __init__(self, ranges: list[_Range], nested: bool) -> None:
        ordered = sorted(ranges, key=lambda row: (row.start, -row.end))
        stack: list[_Range] = []
        boundaries: list[tuple[float, int, int, _Range]] = []
        for index, row in enumerate(ordered):
            while stack and row.start >= stack[-1].end:
                stack.pop()
            if stack:
                parent = stack[-1]
                if not nested or row.end > parent.end:
                    raise ValueError(f"{'Partially overlapping scope' if nested else 'Overlapping step'} ranges on one thread: {parent.key!r} and {row.key!r}")
                if row.start == parent.start and row.end == parent.end:
                    raise ValueError(f"Ambiguous identical scope ranges on one thread: {parent.key!r} and {row.key!r}")
            stack.append(row)
            if row.end > row.start:
                # Ends precede starts at seams; wider ranges enter first.
                boundaries.append((row.start, 1, index, row))
                boundaries.append((row.end, 0, -index, row))
        self.starts: list[float] = []
        self.ranges: list[_Range] = []
        stack = []
        for timestamp, edge, _, row in sorted(boundaries, key=lambda item: item[:3]):
            if stack and self.starts and timestamp > self.starts[-1]:
                active = stack[-1]
                self.ranges.append(_Range(self.starts[-1], timestamp, active.key))
            if edge:
                stack.append(row)
            else:
                stack.pop()
            # Save every boundary, then index only the nonempty active spans.
            self.starts.append(timestamp)
        self.starts = [row.start for row in self.ranges]

    def at(self, timestamp: float) -> Any:
        position = bisect.bisect_right(self.starts, timestamp) - 1
        if position >= 0 and timestamp < self.ranges[position].end:
            return self.ranges[position].key
        return None


def _clone_nodes(record: CaptureRecord) -> dict[int, GraphNode]:
    nodes = {node.node_id: node for node in record.nodes}
    originals = {clone.node_id: clone.original_node_id for clone in record.clones}
    resolved: dict[int, GraphNode] = {}
    for clone in originals:
        chain: list[int] = []
        current = clone
        while current not in nodes and current not in resolved:
            chain.append(current)
            if current not in originals:
                raise ValueError(f"Clone node {clone} resolves to unknown original node id {current}")
            current = originals[current]
        node = nodes[current] if current in nodes else resolved[current]
        for item in chain:
            resolved[item] = node
    return resolved


def _dependency_ends(nodes: dict[int, GraphNode], activities: dict[int, dict[str, Any]]) -> dict[int, float | None]:
    """Memoize latest activity ends through event/empty nodes, without recursion.

    Every activity node of the launch is in ``activities``, so the traversal
    only computes pass-through nodes of the captured (acyclic) graph.
    """
    ends: dict[int, float | None] = {
        node_id: float(event["ts"]) + float(event["dur"])
        for node_id, event in activities.items()
    }
    for root in nodes:
        pending = [(root, False)]
        while pending:
            node_id, finish = pending.pop()
            if node_id in ends:
                continue
            node = nodes[node_id]
            if finish:
                dependencies = [ends[item] for item in node.dependencies if ends[item] is not None]
                ends[node_id] = max(dependencies) if dependencies else None
            else:
                pending.append((node_id, True))
                pending.extend((item, False) for item in node.dependencies)
    return ends


def _validate_kernel(event: dict[str, Any], node: GraphNode) -> None:
    name = event.get("name")
    if name != node.kernel_name and (node.kernel_name_demangled is None or name != node.kernel_name_demangled):
        raise ValueError(f"Kernel name mismatch for node {node.node_id}: trace {name!r}, captured raw {node.kernel_name!r}, demangled {node.kernel_name_demangled!r}")
    for field in (GRID, BLOCK):
        actual = event.get("args", {}).get(field)
        expected = getattr(node, field)
        if not isinstance(actual, (tuple, list)) or tuple(actual) != expected:
            raise ValueError(f"Kernel {field} mismatch for node {node.node_id}: trace {actual!r}, captured {expected!r}")


def attribute_replay_trace(events: Iterable[dict[str, Any]], record: CaptureRecord,
                           container_scopes: Collection[str]) -> ReplayAttribution:
    """Attribute complete graph launches and eager activities, failing ambiguity.

    ``container_scopes`` names the scopes of the integrator's tree that only
    group child scopes; a kernel owned directly by one of them fails (A2).
    """
    step_ranges: dict[tuple[Any, Any], list[_Range]] = defaultdict(list)
    hold_ranges: dict[tuple[Any, Any], list[_Range]] = defaultdict(list)
    scope_ranges: dict[tuple[Any, Any], list[_Range]] = defaultdict(list)
    launches: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    graph_activities: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    eager_activities: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    scopes: set[ScopeKey] = set()
    steps: set[int] = set()
    for event in events:
        category = event.get("cat")
        if category == _ANNOTATION_CATEGORY:
            name = event.get("name")
            if isinstance(name, str) and name.startswith(HOLD_ANNOTATION_PREFIX):
                hold = parse_step_annotation(name.replace(HOLD_ANNOTATION_PREFIX, STEP_ANNOTATION_PREFIX, 1))
                hold_ranges[_thread(event)].append(_range(event, hold))
                continue
            step = parse_step_annotation(name)
            if step is not None:
                steps.add(step)
                step_ranges[_thread(event)].append(_range(event, step))
            else:
                key = parse_annotation_name(name)
                if key is not None:
                    scopes.add(key)
                    scope_ranges[_thread(event)].append(_range(event, key))
        elif category in _LAUNCH_CATEGORIES:
            correlation = event.get("args", {}).get("correlation")
            launches[correlation].append(event)
        elif category in _ACTIVITIES:
            arguments = event.get("args", {})
            node_id = arguments.get("graph node id")
            graph = node_id is not None and int(node_id) > 0
            correlation = arguments.get("correlation")
            if correlation is None:
                raise ValueError(f"Activity {event.get('name')!r} has no launch correlation")
            (graph_activities if graph else eager_activities)[correlation].append(event)

    step_indices = {thread: _RangeIndex(rows, False) for thread, rows in step_ranges.items()}
    for thread, rows in hold_ranges.items():
        for hold in rows:
            enclosing = [step for step in step_ranges[thread]
                         if step.key == hold.key and step.start <= hold.start and hold.end <= step.end]
            if len(enclosing) != 1:
                raise ValueError(f"Hold range {hold.key} must lie inside its step range on the same thread")
    hold_indices = {thread: _RangeIndex(rows, False) for thread, rows in hold_ranges.items()}
    hold_counts = {(thread, index): 0 for thread, rows in hold_ranges.items() for index in range(len(rows))}
    scope_indices = {thread: _RangeIndex(rows, True) for thread, rows in scope_ranges.items()}
    clones = _clone_nodes(record)
    captured_ids = {node.node_id for node in record.nodes}
    captures: dict[int, dict[int, GraphNode]] = defaultdict(dict)
    entries: dict[int, set[ScopeOwner]] = defaultdict(set)
    for node in record.nodes:
        captures[node.capture][node.node_id] = node
    for entry in record.scope_entries:
        if entry.owner is not None:
            entries[entry.capture].add(entry.owner)
    intervals: list[ReplayInterval] = []
    outside: dict[int, int] = {step: 0 for step in steps}
    unattributed: set[int] = set()

    def launch_for(correlation: Any, graph: bool) -> dict[str, Any]:
        candidates = launches.get(correlation, ())
        if graph:
            candidates = [event for event in candidates if event.get("cat") == "cuda_runtime"
                          and str(event.get("name", "")).startswith("cudaGraphLaunch")]
        else:
            candidates = [event for event in candidates if not str(event.get("name", "")).startswith("cudaGraphLaunch")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {'cudaGraphLaunch' if graph else 'eager launch'} for correlation {correlation!r}, found {len(candidates)}; incomplete trace or ambiguous mapping")
        return candidates[0]

    def step_for(launch: dict[str, Any]) -> int | None:
        index = step_indices.get(_thread(launch))
        return index.at(float(launch["ts"])) if index is not None else None

    def append(event: dict[str, Any], step: int, owner: ScopeOwner | None, kind: str,
               launch: dict[str, Any], capture: int | None = None,
               original: int | None = None, gap: float | None = None,
               activity: str | None = None) -> None:
        arguments = event.get("args", {})
        start = float(event["ts"])
        activity = activity or _ACTIVITIES[event["cat"]]
        dimensions = {}
        for field in (GRID, BLOCK):
            value = arguments.get(field)
            if activity in (_KERNEL_ACTIVITY, HOLD_ACTIVITY):
                if (not isinstance(value, (tuple, list)) or len(value) != _LAUNCH_DIMENSIONS
                        or any(type(item) is not int for item in value)):
                    raise ValueError(f"Kernel {event['name']!r} requires {field} with three ints, got {value!r}")
                dimensions[field] = tuple(value)
            else:
                dimensions[field] = None
        intervals.append(ReplayInterval(
            step, owner, kind, activity, event["name"],
            start, start + float(event["dur"]), arguments.get("stream"),
            arguments.get("graph id"), arguments.get("graph node id"), original,
            capture, arguments.get("correlation"), float(launch["ts"]) + float(launch["dur"]), gap,
            dimensions[GRID], dimensions[BLOCK],
        ))
        if owner is not None:
            scopes.add(ScopeKey(step, *owner))

    # Count graph launches without activities too, if they are outside measured steps.
    for candidates in launches.values():
        for launch in candidates:
            if launch.get("cat") == "cuda_runtime" and str(launch.get("name", "")).startswith("cudaGraphLaunch") and step_for(launch) is None:
                unattributed.add(id(launch))

    for correlation, activities in graph_activities.items():
        launch = launch_for(correlation, True)
        step = step_for(launch)
        if step is None:
            unattributed.add(id(launch))
            continue
        matched: dict[int, dict[str, Any]] = {}
        capture: int | None = None
        for event in activities:
            node_id = int(event["args"]["graph node id"])
            if node_id not in clones:
                hint = "is itself a captured node id" if node_id in captured_ids else "is not a captured node id either"
                raise ValueError(f"Graph node id {node_id} is not a clone id; it {hint}")
            node = clones[node_id]
            if capture is not None and node.capture != capture:
                raise ValueError(f"Graph launch correlation {correlation!r} spans multiple captures: {capture} and {node.capture}")
            capture = node.capture
            if node.node_id in matched:
                raise ValueError(f"Graph launch correlation {correlation!r} repeats node {node.node_id}; ambiguous mapping")
            if _ACTIVITIES[event["cat"]] != node.node_type:
                raise ValueError(f"Graph activity type mismatch for node {node.node_id}: trace {event['cat']!r}, captured {node.node_type!r}")
            if node.node_type == "kernel":
                _validate_kernel(event, node)
            matched[node.node_id] = event
        nodes = captures[capture]
        required = {node_id for node_id, node in nodes.items() if node.node_type in _ACTIVITY_NODES}
        if matched.keys() != required:
            raise ValueError(f"Graph launch correlation {correlation!r} has missing nodes {sorted(required - matched.keys())} or unexpected nodes {sorted(matched.keys() - required)}; incomplete trace or ambiguous mapping")
        scopes.update(ScopeKey(step, *owner) for owner in entries[capture])
        ends = _dependency_ends(nodes, matched)
        for node_id, event in matched.items():
            node = nodes[node_id]
            gap = None
            if node.node_type == "kernel":
                dependency_ends = [ends[item] for item in node.dependencies if ends[item] is not None]
                if dependency_ends:
                    gap = float(event["ts"]) - max(dependency_ends)
            append(event, step, node.owner, "graph", launch, capture, node_id, gap)

    for correlation, kernels in eager_activities.items():
        launch = launch_for(correlation, False)
        step = step_for(launch)
        if step is None:
            unattributed.add(id(launch))
            continue
        hold_index = hold_indices.get(_thread(launch))
        hold_step = hold_index.at(float(launch["ts"])) if hold_index is not None else None
        if hold_step is not None:
            if hold_step != step or len(kernels) != 1 or kernels[0]["cat"] != _KERNEL_ACTIVITY:
                raise ValueError(f"Hold range {hold_step} must launch exactly one kernel inside its step")
            rows = hold_ranges[_thread(launch)]
            index = next(index for index, row in enumerate(rows)
                         if row.start <= float(launch["ts"]) < row.end)
            hold_counts[(_thread(launch), index)] += 1
            append(kernels[0], step, None, "eager", launch, activity=HOLD_ACTIVITY)
            continue
        index = scope_indices.get(_thread(launch))
        key = index.at(float(launch["ts"])) if index is not None else None
        if key is not None and key.step_index != step:
            raise ValueError(f"Eager launch step {step} disagrees with scope annotation {key!r}")
        for event in kernels:
            append(event, step, None if key is None else (key.global_layer_id, key.scope), "eager", launch)

    for (thread, index), count in hold_counts.items():
        if count != 1:
            raise ValueError(f"Hold range {hold_ranges[thread][index].key} must launch exactly one kernel, found {count}")

    totals: dict[ScopeKey, float] = {key: 0.0 for key in scopes}
    counts: dict[ScopeKey, int] = {key: 0 for key in scopes}
    windows: dict[tuple[int, int], tuple[float, float]] = {}
    for interval in intervals:
        if interval.activity != "kernel":
            continue
        if interval.owner is None:
            outside[interval.step_index] += 1
            continue
        key = ScopeKey(interval.step_index, *interval.owner)
        if interval.owner[1] in container_scopes:
            raise ValueError(f"A2: kernel {interval.name!r} is owned directly by container scope {key!r}")
        totals[key] += interval.end_us - interval.start_us
        counts[key] += 1
        layer = key.step_index, key.global_layer_id
        previous = windows.get(layer, (interval.start_us, interval.end_us))
        windows[layer] = min(previous[0], interval.start_us), max(previous[1], interval.end_us)
    return ReplayAttribution(
        {key: ScopeKernelTime(totals[key] * _US_TO_MS, counts[key]) for key in totals},
        windows, tuple(sorted(intervals, key=lambda item: item.start_us)), outside, len(unattributed),
    )


__all__ = [
    "ReplayAttribution", "ReplayInterval", "STEP_ANNOTATION_PREFIX", "attribute_replay_trace",
    "HOLD_ANNOTATION_PREFIX", "HOLD_ACTIVITY", "hold_annotation_name",
    "load_trace_events", "parse_step_annotation", "step_annotation_name",
]
