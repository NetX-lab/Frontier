"""Reduce saved V4 replay intervals to four device-gap values on CPU.

The classes are cuda_graph, cuda_graph_piecewise, eager and eager_preparation.
Graph kernels follow captured activity dependencies; graph roots and outside
preparation use R1 (the latest end of any strictly earlier step activity).
Attention-owned eager kernels follow physical stream order and the layer's
fork/join scope frontier, with the layer entry as the initial dependency. In a
graph-mode step they also follow the whole preceding graph launch on their
stream, whose replay nodes the trace reports on other streams.
Copies and memsets provide dependency ends but only preparation prices them.

Gaps stay signed: programmatic dependent launch (PDL) can start a dependent
kernel before its predecessor ends, and the dependent duration contains that
overlap. Clipping would count the overlap twice. Host wait is the portion of a
positive gap before its launch call ended; only zero-host-wait samples price a
class. Diagnostics retain excluded, negative, root and attention-owned samples.
Step dispatch and capture kinds must agree; graph-mode eager outside work may
only precede or follow all graph activity. All input identities must match.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from frontier.execution_time_predictor.dsv4_composition import attention_scope_predecessors
from frontier.profiling.attention.dsa_plan import ENGINE_POLICY_GRAPH_REPLAY
from frontier.profiling.attention.dsa_vllm_runner import (
    ACTIVITY, CAPTURE_KIND, CASE_ID, END_US, ENGINE_INDEX, GLOBAL_LAYER_ID,
    GRAPH_GAP_US, INTERVAL_RECORD, LAUNCH_END_US, LAUNCH_KIND, NAME, RECORD_KIND,
    SCOPE, START_US, STEP_INDEX, STREAM,
)
from frontier.profiling.attention.dsv4_graph_replay_runner import (
    CONTAINER_SCOPES, CUDAGRAPH_DISPATCH_KEY, STEP_HOLD_US, HOLD_CYCLES_PER_US,
)
from frontier.profiling.attention.graph_replay_trace import HOLD_ACTIVITY
from frontier.profiling.kernel_gap.schema import (
    CUDA_GRAPH, CUDA_GRAPH_PIECEWISE, DSV4_EXECUTION_MODES, EAGER, EAGER_PREPARATION,
    EXECUTION_MODE_COLUMN, KERNEL_GAP_US_COLUMN, TENSOR_PARALLEL_COLUMN,
)
from frontier.profiling.utils import build_profiling_output_path

ENGINE_RECORD, STEP_RECORD, CALL_RECORD = "engine", "step", "call"
GRAPH_LAUNCH, EAGER_LAUNCH = "graph", "eager"
KERNEL, MEMCPY, MEMSET = "kernel", "memcpy", "memset"
FULL, PIECEWISE, NONE = "FULL", "PIECEWISE", "NONE"
GRAPH_CAPTURE, BREAKABLE_CAPTURE = "graph", "breakable_segment"
CG_MODE, NUM_TOKENS = "cg_mode", "num_tokens"
LAYERS, VARIANT, COMPRESS_RATIO = "layers", "attention_variant", "compress_ratio"
ENGINE_POLICY, ENGINE_CAPTURE = "engine_policy", "cudagraph_capture"
ENGINE_MODE = "cudagraph_mode"
THRESHOLD = "multi_stream_gemm_token_threshold"
SHORT_CONTEXT_SCOPE = "dsv4_index_short_context_fill"
DEVICE_NAME, DRIVER_VERSION, TARGET_TP = "device_name", "driver_version", "target_tensor_parallel_size"
RUNTIME_VERSIONS, TORCH_VERSION, TORCH_CUDA = "runtime_versions", "torch", "torch_cuda"
NO_DEPENDENCY, NON_KERNEL = "no_dependency_end", "non_kernel_dependency"
HOST_BOUND = "host_bound"
OVERLAPS_HOLD = "overlaps_hold"
HOLD_DEPENDENCY = "hold_dependency"
HOLD_DIAGNOSTICS = "hold_by_step_mode_and_case"
US_TO_MS = 1e-3
IDENTITY_FIELDS = (DEVICE_NAME, DRIVER_VERSION, TARGET_TP)
NUM_GAPS_COLUMN, CHAIN_COLUMN, GPU_NAME_COLUMN = "num_gaps", "chain", "gpu_name"
TORCH_VERSION_COLUMN, CUDA_VERSION_COLUMN = "torch_version", "cuda_version"
CSV_COLUMNS = (EXECUTION_MODE_COLUMN, TENSOR_PARALLEL_COLUMN, KERNEL_GAP_US_COLUMN,
               NUM_GAPS_COLUMN, CHAIN_COLUMN, GPU_NAME_COLUMN, DRIVER_VERSION,
               TORCH_VERSION_COLUMN, CUDA_VERSION_COLUMN)


_HOLD_DURATION = 'duration'
_HOLD_SLACK = 'slack'
_POST_HOLD_ATTENTION = 'attention'
_POST_HOLD_UNOWNED = 'unowned'
_HOLD_STEP_COUNT = 'step_count'
_REALIZED_HOLD_DURATION_US = 'realized_hold_duration_us'
_HOLD_SLACK_US = 'slack_us'
_ATTENTION_POST_HOLD_LAUNCH_COUNT = 'attention_owned_post_hold_launch_count'
_UNOWNED_POST_HOLD_LAUNCH_COUNT = 'unowned_post_hold_launch_count'

@dataclass(frozen=True)
class ReplayRecords:
    source: str
    engine: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...]
    calls: tuple[Mapping[str, Any], ...]
    intervals: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class LayerCall:
    source: str
    engine: Mapping[str, Any]
    step: Mapping[str, Any]
    global_layer_id: int
    variant: str
    calls: tuple[Mapping[str, Any], ...]
    intervals: tuple[Mapping[str, Any], ...]
    step_intervals: tuple[Mapping[str, Any], ...]

    @property
    def short_context(self) -> bool:
        return any(row[SCOPE] == SHORT_CONTEXT_SCOPE for row in self.calls)


@dataclass(frozen=True)
class ActivityGap:
    source: str
    engine_mode: str
    step_mode: str
    interval: Mapping[str, Any]
    segment: str | None
    root: bool
    dependency_end_us: float | None
    gap_us: float | None
    host_wait_us: float | None
    device_gap_us: float | None
    exclusion_reason: str | None

    @property
    def host_ahead(self) -> bool:
        return self.dependency_end_us is not None and self.host_wait_us == 0.0 and self.exclusion_reason != OVERLAPS_HOLD

    @property
    def attention_owned(self) -> bool:
        return self.interval[SCOPE] is not None


@dataclass(frozen=True)
class GapReduction:
    values_us: Mapping[str, float]
    identity: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    reasons: tuple[str, ...]


def _identity(engine: Mapping[str, Any], source: str) -> dict[str, Any]:
    try:
        return {**{field: engine[field] for field in IDENTITY_FIELDS},
                TORCH_VERSION: engine[RUNTIME_VERSIONS][TORCH_VERSION],
                TORCH_CUDA: engine[RUNTIME_VERSIONS][TORCH_CUDA]}
    except KeyError as error:
        raise ValueError(f"Replay engine {source} has no identity field {error.args[0]!r}") from error


def validate_identities(records: Sequence[ReplayRecords]) -> Mapping[str, Any]:
    if not records:
        raise ValueError("V4 replay gaps require at least one raw file")
    expected = _identity(records[0].engine, records[0].source)
    for record in records:
        if record.engine[ENGINE_POLICY] != ENGINE_POLICY_GRAPH_REPLAY:
            raise ValueError(f"Replay engine {record.source} requires engine_policy=graph_replay, got {record.engine[ENGINE_POLICY]!r}")
        for field in (STEP_HOLD_US, HOLD_CYCLES_PER_US):
            if field not in record.engine:
                raise ValueError(f"Replay engine {record.source} has no hold field {field!r}")
        hold_us, cycles_per_us = record.engine[STEP_HOLD_US], record.engine[HOLD_CYCLES_PER_US]
        if type(hold_us) not in (int, float) or not math.isfinite(hold_us) or hold_us < 0:
            raise ValueError(f"Replay engine {record.source} has invalid step_hold_us {hold_us!r}")
        if hold_us > 0:
            if (type(cycles_per_us) not in (int, float) or not math.isfinite(cycles_per_us)
                    or cycles_per_us <= 0):
                raise ValueError(f"Replay engine {record.source} requires positive finite hold_cycles_per_us")
        elif cycles_per_us is not None:
            raise ValueError(f"Replay engine {record.source} requires hold_cycles_per_us=null without a hold")
        actual = _identity(record.engine, record.source)
        for field in expected:
            if actual[field] != expected[field]:
                raise ValueError(f"Replay identity differs in {record.source}: {field}={actual[field]!r}, expected {expected[field]!r}")
    return expected


def read_replay_records(paths: Sequence[Path]) -> tuple[ReplayRecords, ...]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        by_kind = defaultdict(list)
        for row in rows:
            if row[RECORD_KIND] not in (ENGINE_RECORD, STEP_RECORD, CALL_RECORD, INTERVAL_RECORD):
                raise ValueError(f"Replay raw file {path} has unknown record kind {row[RECORD_KIND]!r}")
            by_kind[row[RECORD_KIND]].append(row)
        if len(by_kind[ENGINE_RECORD]) != 1:
            raise ValueError(f"Replay raw file {path} requires one engine record")
        records.append(ReplayRecords(str(path), by_kind[ENGINE_RECORD][0],
                                     tuple(by_kind[STEP_RECORD]), tuple(by_kind[CALL_RECORD]),
                                     tuple(by_kind[INTERVAL_RECORD])))
    validate_identities(records)
    return tuple(records)


def _step_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return row[CASE_ID], row[STEP_INDEX]


def build_layer_calls(records: Sequence[ReplayRecords]) -> tuple[LayerCall, ...]:
    """Join call rows and intervals by file, case, step and engine layer identity."""
    validate_identities(records)
    result = []
    for record in records:
        steps = {_step_key(step): step for step in record.steps}
        if len(steps) != len(record.steps):
            raise ValueError(f"Replay raw file {record.source} repeats a step record")
        layer_rows = {layer[GLOBAL_LAYER_ID]: layer for layer in record.engine[LAYERS]}
        calls, owned, step_intervals = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in (*record.calls, *record.intervals):
            if row[ENGINE_INDEX] != record.engine[ENGINE_INDEX] or _step_key(row) not in steps:
                raise ValueError(f"Replay row in {record.source} has an unrecorded engine or step: {_step_key(row)}")
            key = (*_step_key(row), row[GLOBAL_LAYER_ID])
            if row[RECORD_KIND] == CALL_RECORD:
                if row[SCOPE] not in CONTAINER_SCOPES:
                    calls[key].append(row)
            else:
                if (row[SCOPE] is None) != (row[GLOBAL_LAYER_ID] is None):
                    _invalid(steps[_step_key(row)], row, "owner scope and layer must both be null or both be present")
                step_intervals[_step_key(row)].append(row)
                if row[SCOPE] is not None:
                    if row[SCOPE] in CONTAINER_SCOPES and row[ACTIVITY] == KERNEL:
                        _invalid(steps[_step_key(row)], row, "kernel is owned by a container scope")
                    owned[key].append(row)
        for key in sorted(calls.keys() | owned.keys()):
            case, step_index, layer_id = key
            if layer_id not in layer_rows:
                raise ValueError(f"Replay call {record.source}, case {case}, step {step_index} has unknown layer {layer_id}")
            result.append(LayerCall(record.source, record.engine, steps[(case, step_index)],
                                    layer_id, layer_rows[layer_id][VARIANT], tuple(calls[key]),
                                    tuple(sorted(owned[key], key=lambda row: row[START_US])),
                                    tuple(sorted(step_intervals[(case, step_index)], key=lambda row: row[START_US]))))
    return tuple(result)


def _invalid(step, row, reason):
    activity = "<no activity>" if row is None else f"{row[ACTIVITY]} {row[NAME]!r} at {row[START_US]} us"
    raise ValueError(f"V4 replay case {step[CASE_ID]}, step {step[STEP_INDEX]}, activity {activity}: {reason}")


class _EarlierEnds:
    """R1 prefix maxima with strictly earlier starts, including equal-start ties."""

    def __init__(self, intervals):
        ordered = sorted(intervals, key=lambda row: row[START_US])
        self.starts = [row[START_US] for row in ordered]
        self.ends = []
        latest = -math.inf
        for row in ordered:
            latest = max(latest, row[END_US])
            self.ends.append(latest)

    def before(self, start):
        position = bisect.bisect_left(self.starts, start) - 1
        return self.ends[position] if position >= 0 else None


def activity_gaps(records: Sequence[ReplayRecords]) -> tuple[ActivityGap, ...]:
    """Compute every activity's signed gap and explicit host-wait split."""
    calls = build_layer_calls(records)
    by_call = {(call.source, *_step_key(call.step), call.global_layer_id): call for call in calls}
    result = []
    for record in records:
        by_step = defaultdict(list)
        for row in record.intervals:
            by_step[_step_key(row)].append(row)
        for step in record.steps:
            intervals = sorted(by_step[_step_key(step)], key=lambda row: row[START_US])
            holds = [row for row in intervals if row[ACTIVITY] == HOLD_ACTIVITY]
            expected_holds = int(record.engine[STEP_HOLD_US] > 0)
            if len(holds) != expected_holds:
                _invalid(step, holds[0] if holds else None,
                         f"step requires {expected_holds} hold intervals, found {len(holds)}")
            hold_end = holds[0][END_US] if holds else None
            mode = step[CUDAGRAPH_DISPATCH_KEY][CG_MODE]
            graphs = [row for row in intervals if row[LAUNCH_KIND] == GRAPH_LAUNCH]
            if mode not in (FULL, PIECEWISE, NONE):
                _invalid(step, intervals[0] if intervals else None, f"unknown dispatch mode {mode!r}")
            if mode == NONE and graphs:
                _invalid(step, graphs[0], "NONE step has a graph activity")
            if mode != NONE and not graphs:
                _invalid(step, intervals[0] if intervals else None, f"{mode} step has no graph activity")
            first_graph = min((row[START_US] for row in graphs), default=math.inf)
            last_graph = max((row[END_US] for row in graphs), default=-math.inf)
            earlier = _EarlierEnds(intervals)
            graph_ends = _EarlierEnds(graphs)
            stream_rows = defaultdict(list)
            for row in intervals:
                for field in (START_US, END_US, LAUNCH_END_US):
                    if not math.isfinite(row[field]):
                        _invalid(step, row, f"{field} is not finite")
                if row[END_US] < row[START_US]:
                    _invalid(step, row, "negative activity duration")
                if row[ACTIVITY] not in (KERNEL, MEMCPY, MEMSET, HOLD_ACTIVITY) or row[LAUNCH_KIND] not in (GRAPH_LAUNCH, EAGER_LAUNCH):
                    _invalid(step, row, "unknown activity or launch kind")
                if row[ACTIVITY] == HOLD_ACTIVITY and (row[SCOPE] is not None or row[LAUNCH_KIND] != EAGER_LAUNCH):
                    _invalid(step, row, "hold must be an unowned eager activity")
                stream_rows[row[STREAM]].append(row)
            stream_starts = {stream: [row[START_US] for row in rows] for stream, rows in stream_rows.items()}
            prepared = {}
            for row in intervals:
                segment, root, reason = None, False, None
                overlaps_hold = hold_end is not None and row[START_US] < hold_end
                dep_end = None
                if row[ACTIVITY] == HOLD_ACTIVITY:
                    reason = HOLD_DEPENDENCY
                elif row[LAUNCH_KIND] == GRAPH_LAUNCH:
                    expected = GRAPH_CAPTURE if mode == FULL else BREAKABLE_CAPTURE
                    if row[CAPTURE_KIND] != expected:
                        _invalid(step, row, f"capture kind {row[CAPTURE_KIND]!r} does not match {mode} ({expected})")
                    segment = CUDA_GRAPH if mode == FULL else CUDA_GRAPH_PIECEWISE
                    graph_gap = row[GRAPH_GAP_US]
                    root = row[ACTIVITY] == KERNEL and graph_gap is None
                    if row[ACTIVITY] != KERNEL:
                        # The raw captured-dependency gap is recorded for kernels only.
                        segment, reason = None, NON_KERNEL
                    elif graph_gap is not None:
                        if not math.isfinite(graph_gap):
                            _invalid(step, row, "graph_gap_us is not finite")
                        dep_end = row[START_US] - graph_gap
                    else:
                        dep_end = earlier.before(row[START_US])
                elif row[SCOPE] is not None:
                    segment = EAGER
                    key = (record.source, *_step_key(row), row[GLOBAL_LAYER_ID])
                    call = by_call[key]
                    if key not in prepared:
                        present = {item[SCOPE] for item in call.intervals if item[SCOPE] not in CONTAINER_SCOPES}
                        preds = attention_scope_predecessors(
                            present, variant=call.variant, graph_runtime_mode=mode,
                            rows=step[CUDAGRAPH_DISPATCH_KEY][NUM_TOKENS],
                            multi_stream_gemm_token_threshold=record.engine[THRESHOLD],
                            indexer_short_context=call.short_context,
                        )
                        first = {}
                        scope_ends = defaultdict(dict)
                        for item in call.intervals:
                            first.setdefault((item[SCOPE], item[STREAM]), item[START_US])
                            ends = scope_ends[item[SCOPE]]
                            ends[item[STREAM]] = max(ends.get(item[STREAM], -math.inf), item[END_US])
                        entry = _EarlierEnds([item for item in intervals if item[GLOBAL_LAYER_ID] != call.global_layer_id]).before(call.intervals[0][START_US])
                        prepared[key] = preds, first, scope_ends, entry
                    preds, first, scope_ends, entry = prepared[key]
                    terms = []
                    position = bisect.bisect_left(stream_starts[row[STREAM]], row[START_US]) - 1
                    if position >= 0:
                        terms.append(stream_rows[row[STREAM]][position][END_US])
                    if mode != NONE:
                        graph_end = graph_ends.before(row[START_US])
                        if graph_end is None and not overlaps_hold:
                            _invalid(step, row, "attention-owned eager activity precedes every graph activity")
                        if graph_end is not None:
                            terms.append(graph_end)
                    if row[SCOPE] not in CONTAINER_SCOPES and row[START_US] == first[(row[SCOPE], row[STREAM])]:
                        if preds[row[SCOPE]]:
                            terms.extend(end for name in preds[row[SCOPE]]
                                         for stream, end in scope_ends[name].items() if stream != row[STREAM])
                        elif entry is not None:
                            terms.append(entry)
                    dep_end = max(terms) if terms else None
                else:
                    dep_end = earlier.before(row[START_US])
                    if mode != NONE:
                        if first_graph <= row[START_US] <= last_graph and not overlaps_hold:
                            _invalid(step, row, "unowned eager activity is between the first and last graph activity")
                        segment = EAGER_PREPARATION
                if row[ACTIVITY] != KERNEL and row[SCOPE] is not None and segment != EAGER_PREPARATION:
                    segment, reason = None, NON_KERNEL
                gap = None if dep_end is None else row[START_US] - dep_end
                host_wait = None if gap is None else min(max(row[LAUNCH_END_US] - dep_end, 0.0), max(gap, 0.0))
                if overlaps_hold and row[ACTIVITY] != HOLD_ACTIVITY:
                    reason = OVERLAPS_HOLD
                elif dep_end is None and row[ACTIVITY] != HOLD_ACTIVITY:
                    reason = NO_DEPENDENCY
                elif reason is None and host_wait > 0:
                    reason = HOST_BOUND
                result.append(ActivityGap(record.source, record.engine[ENGINE_CAPTURE][ENGINE_MODE], mode,
                                          row, segment, root, dep_end, gap, host_wait,
                                          None if gap is None else gap - host_wait, reason))
    return tuple(result)


def gap_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Linear percentiles and signed statistics; empty samples serialize as null."""
    if not values:
        return dict.fromkeys(("mean", "median", "p10", "p90", "min", "max"), None) | {"negative_count": 0}
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10, method="linear")),
            "p90": float(np.percentile(array, 90, method="linear")),
            "min": float(np.min(array)), "max": float(np.max(array)),
            "negative_count": int(np.count_nonzero(array < 0))}


def _sample_summary(samples):
    ahead = [sample.gap_us for sample in samples if sample.host_ahead and sample.exclusion_reason is None]
    exclusions = Counter(sample.exclusion_reason for sample in samples if sample.exclusion_reason is not None)
    return {"sample_count": len(samples), "host_ahead_count": len(ahead),
            "host_bound_excluded_count": exclusions[HOST_BOUND],
            "exclusion_counts": dict(sorted(exclusions.items())), **gap_statistics(ahead)}


def _class_summary(samples):
    return {**_sample_summary(samples),
            "roots": _sample_summary([sample for sample in samples if sample.root]),
            "attention_owned": _sample_summary([sample for sample in samples if sample.attention_owned])}


def _hold_diagnostics(records: Sequence[ReplayRecords]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in records:
        intervals = defaultdict(list)
        for row in record.intervals:
            intervals[_step_key(row)].append(row)
        for step in record.steps:
            rows = intervals[_step_key(step)]
            holds = [row for row in rows if row[ACTIVITY] == HOLD_ACTIVITY]
            owned = [row for row in rows if row[SCOPE] is not None]
            end = holds[0][END_US] if holds else None
            grouped[(step[CUDAGRAPH_DISPATCH_KEY][CG_MODE], step[CASE_ID])].append({
                _HOLD_DURATION: holds[0][END_US] - holds[0][START_US] if holds else None,
                _HOLD_SLACK: end - max(row[LAUNCH_END_US] for row in owned) if holds and owned else None,
                _POST_HOLD_ATTENTION: sum(row[LAUNCH_END_US] > end for row in owned) if holds else 0,
                _POST_HOLD_UNOWNED: sum(row[LAUNCH_END_US] > end for row in rows
                               if row[SCOPE] is None and row[ACTIVITY] != HOLD_ACTIVITY) if holds else 0,
            })
    result = {}
    for (mode, case), rows in sorted(grouped.items()):
        duration = gap_statistics([row[_HOLD_DURATION] for row in rows if row[_HOLD_DURATION] is not None])
        slack = gap_statistics([row[_HOLD_SLACK] for row in rows if row[_HOLD_SLACK] is not None])
        result.setdefault(mode, {})[case] = {
            _HOLD_STEP_COUNT: len(rows),
            _REALIZED_HOLD_DURATION_US: {key: duration[key] for key in ("min", "median", "max")},
            _HOLD_SLACK_US: {key: slack[key] for key in ("min", "median")},
            _ATTENTION_POST_HOLD_LAUNCH_COUNT: sum(row[_POST_HOLD_ATTENTION] for row in rows),
            _UNOWNED_POST_HOLD_LAUNCH_COUNT: sum(row[_POST_HOLD_UNOWNED] for row in rows),
        }
    return result


def reduce_dsv4_gaps(records: Sequence[ReplayRecords]) -> GapReduction:
    """Use only host-ahead class samples; expose unavailable values without substitution."""
    identity = validate_identities(records)
    samples = activity_gaps(records)
    classes, by_mode, values, reasons = {}, {}, {}, []
    for segment in DSV4_EXECUTION_MODES:
        selected = [sample for sample in samples if sample.segment == segment]
        summary = _class_summary(selected)
        classes[segment] = summary
        by_mode[segment] = {mode: _class_summary([sample for sample in selected if sample.step_mode == mode])
                            for mode in (FULL, PIECEWISE, NONE) if any(sample.step_mode == mode for sample in selected)}
        mean = summary["mean"]
        if summary["host_ahead_count"] == 0:
            reasons.append(f"{segment}: no host-ahead sample")
        elif not math.isfinite(mean) or mean < 0:
            reasons.append(f"{segment}: host-ahead mean must be finite and non-negative, got {mean}")
        else:
            values[segment] = mean
    non_kernel = defaultdict(float)
    for sample in samples:
        if sample.attention_owned and sample.interval[ACTIVITY] != KERNEL:
            non_kernel[sample.engine_mode] += sample.interval[END_US] - sample.interval[START_US]
    diagnostics = {
        HOLD_DIAGNOSTICS: _hold_diagnostics(records),
        "identity": identity, "values_us": values, "reasons": reasons,
        "classes": classes, "classes_by_step_mode": by_mode,
        "none_unowned": _class_summary([sample for sample in samples if sample.step_mode == NONE and not sample.attention_owned and sample.interval[ACTIVITY] != HOLD_ACTIVITY]),
        "owned_non_kernel_time_us_by_engine_mode": {record.engine[ENGINE_CAPTURE][ENGINE_MODE]: non_kernel[record.engine[ENGINE_CAPTURE][ENGINE_MODE]] for record in records},
    }
    return GapReduction(values, identity, diagnostics, tuple(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_files", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostics_path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reduction = reduce_dsv4_gaps(read_replay_records(args.raw_files))
    except (OSError, ValueError, KeyError, TypeError) as error:
        args.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_path.write_text(json.dumps({"reasons": [str(error)]}, indent=2) + "\n", encoding="utf-8")
        print(f"[dsv4_gaps] {error}")
        return 1
    args.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_path.write_text(json.dumps(reduction.diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if reduction.reasons:
        for reason in reduction.reasons:
            print(f"[dsv4_gaps] {reason}")
        return 1
    identity = reduction.identity
    tp = identity[TARGET_TP]
    rows = [{EXECUTION_MODE_COLUMN: segment, TENSOR_PARALLEL_COLUMN: tp,
             KERNEL_GAP_US_COLUMN: reduction.values_us[segment],
             NUM_GAPS_COLUMN: reduction.diagnostics["classes"][segment]["host_ahead_count"],
             CHAIN_COLUMN: f"dsv4_graph_replay:{args.model}:tp{tp}", GPU_NAME_COLUMN: identity[DEVICE_NAME],
             DRIVER_VERSION: identity[DRIVER_VERSION], TORCH_VERSION_COLUMN: identity[TORCH_VERSION],
             CUDA_VERSION_COLUMN: identity[TORCH_CUDA]} for segment in DSV4_EXECUTION_MODES]
    path = Path(build_profiling_output_path(output_root=str(args.output_dir), profiling_type="compute",
                                            hardware=args.device, model_name=args.model, op_name="kernel_gap"))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)
    print(f"[dsv4_gaps] wrote {path}: {dict(reduction.values_us)}")
    return 0


__all__ = ["ActivityGap", "GapReduction", "LayerCall", "ReplayRecords", "activity_gaps",
           "build_layer_calls", "gap_statistics", "main", "read_replay_records", "reduce_dsv4_gaps"]

if __name__ == "__main__":
    raise SystemExit(main())
