"""Qualify reduced V4 attention launch identities against native TP0 traces.

The pilot disables async scheduling and enables chunked prefill, whereas the
native engine enables async scheduling without chunking. First tails need their
one-token final chunk. References require the same step class, CUDA Graph mode,
padded dispatch rows and eager attention rows. FULL attention uses padded rows;
PIECEWISE and NONE attention use scheduled rows, while a dummy executes all
dispatched rows. Shapes absent from the native traces are NO_REFERENCE and
remain unqualified. The TP1 reduced engine has no sequence parallelism. The MLP
is dense, so suppressing the DP width tensor cannot change that forward. Host
synchronizations and full launch queues are detected by hold slack rather than
excluded by construction. Sampling is inside the pilot step range as unowned
eager preparation and is not included in attention identity. Native dispatch
contains other operators and collectives, so qualification requires count-wise
containment, not multiset equality.

Expansion uses the full-model counts of each variant and planner/reuse role.
Every reduced layer of a role must carry the same kernel multiset in the step.

Run with tests/analysis on PYTHONPATH; gzip TP0 traces are read one at a time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from frontier.profiling.attention import dsv4_attention_main
from frontier.profiling.attention.dsa_plan import PHASE_DECODE, PHASE_DUMMY, PHASE_FIRST_TAIL, PHASE_PREFILL
from frontier.profiling.attention.dsv4_attention_plan import resolve_dsv4_layers
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.kernel_gap.dsv4_replay_gaps import (
    ACTIVITY, CASE_ID, CG_MODE, ENGINE_CAPTURE, ENGINE_INDEX, ENGINE_MODE,
    GLOBAL_LAYER_ID, KERNEL, NAME, NUM_TOKENS, STEP_INDEX, CUDAGRAPH_DISPATCH_KEY,
    ReplayRecords, read_replay_records,
)
from dsv4_composition_check import planner_roles
from dsv4_graph_bridge_check import BridgeConfig, graph_mode_flags, load_config
from v4_runtime_census import dispatch_kernel_multisets, lane_traces

PASS, FAIL, NO_REFERENCE = "PASS", "FAIL", "NO_REFERENCE"
# This per-layer norm dispatches one grid-x block per padded token row.
_ROW_COUNT_KERNEL = "_fused_q_kv_rmsnorm_kernel"
_EAGER_ATTENTION_MODES = frozenset({"PIECEWISE", "NONE"})
GRID, BLOCK = "grid", "block"
FIRST_TAIL_ALONE, FIRST_TAIL_WITH_DECODES = "first_tail_alone", "first_tail_with_decodes"
_DIMENSIONS = 3

_NATIVE_LAYER_COUNT = 'num_hidden_layers'
_COUNT = 'count'
_STATUS = 'status'
_CANDIDATE_COUNT = 'candidate_count'
_EXPECTED_KERNEL_COUNT = 'expected_kernel_count'
_MISSING_KERNEL_COUNT = 'missing_kernel_count'
_MISSING_KERNELS = 'missing_kernels'
_GRID_MISMATCHES = 'grid_mismatches'
_EXPECTED_GRID = 'expected_grid'
_NATIVE_GRIDS = 'native_grids'
_MISSING_COUNT = 'missing_count'
_CLOSEST_FORWARD_COUNT = 'closest_candidate_forward_count'
_RUN = 'run'
_NATIVE_CLASS = 'native_class'
_STEP_COUNT = 'step_count'
_PASS_COUNT = 'pass_count'
_VERDICT = 'verdict'
_LAYER_ROLE_COUNTS = 'layer_role_counts'
_PLANS_TILE_SCHEDULE = 'plans_tile_schedule'
_VARIANT = 'variant'
_STEPS = 'steps'
_CASES = 'cases'
_NATIVE_CANDIDATES = 'native_candidates'
_CLASS = 'class'
_MODE = 'mode'
_FORWARD_COUNT = 'forward_count'
_ROLES = 'roles'
_KERNEL_COUNT = 'kernel_count'
_REASONS = 'reasons'
_SHAPE_KEY = 'shape_key'
_SHAPE_COVERAGE = 'shape_coverage'
_PILOT_CASE_IDS = 'pilot_case_ids'
_STATUS_COUNTS = 'step_status_counts'
_NATIVE_FORWARD_COUNT = 'native_forward_count'
_REQUESTS = 'requests'
_NUM_SCHEDULED_TOKENS = 'num_scheduled_tokens'

KernelIdentity = tuple[str, tuple[int, int, int], tuple[int, int, int]]
ShapeKey = tuple[str, str, int, int | None]


@dataclass(frozen=True)
class NativeCandidate:
    step_class: str
    mode: str
    padded_rows: int
    eager_rows: int | None
    kernels: Counter[KernelIdentity]
    forward_count: int
    roles: tuple[str, ...]

    @property
    def shape_key(self) -> ShapeKey:
        return self.step_class, self.mode, self.padded_rows, self.eager_rows


def _padded_rows(kernels: Counter[KernelIdentity], source: str) -> int:
    grids = {grid for (name, grid, _), count in kernels.items() if name == _ROW_COUNT_KERNEL and count > 0}
    if len(grids) != 1:
        raise ValueError(f"{source}: row-count kernel {_ROW_COUNT_KERNEL!r} has {len(grids)} grids, expected exactly one")
    return next(iter(grids))[0]


def native_candidates(trace_dirs: Sequence[Path]) -> tuple[NativeCandidate, ...]:
    """Deduplicate exact dispatch multisets by the full shape key over TP0 lanes."""
    counts, roles = Counter(), defaultdict(set)
    for directory in trace_dirs:
        for role, _, path in lane_traces(directory):
            for forward in dispatch_kernel_multisets(role, path):
                padded_rows = _padded_rows(forward.kernels, f"{path}: forward {forward.index}")
                eager_rows = None
                if forward.mode in _EAGER_ATTENTION_MODES:
                    eager_rows = padded_rows if forward.step_class == PHASE_DUMMY else forward.scheduled_tokens
                shape = forward.step_class, forward.mode, padded_rows, eager_rows
                signature = tuple(sorted(forward.kernels.items()))
                key = shape, signature
                counts[key] += 1
                roles[key].add(role)
    return tuple(NativeCandidate(*shape, Counter(dict(signature)), count, tuple(sorted(roles[key])))
                 for key, count in counts.items() for shape, signature in (key,))


def _plans(config: BridgeConfig):
    return {mode: dsv4_attention_main.build_plan(dsv4_attention_main.build_parser().parse_args(
        ["plan", *config.plan_args, *graph_mode_flags(mode, config.cudagraph_capture_sizes)]))
        for mode in dict.fromkeys(run.mode for run in config.runs)}


def layer_role_counts(config: BridgeConfig) -> Mapping[tuple[str, bool], int]:
    """Count native decoder layers by variant and SWA-group planner role."""
    mode = config.runs[0].mode
    args = dsv4_attention_main.build_parser().parse_args(
        ["plan", *config.plan_args, *graph_mode_flags(mode, config.cudagraph_capture_sizes)])
    hf_config = json.loads(Path(args.model_config_path).read_text(encoding="utf-8"))
    layers = resolve_dsv4_layers(ModelConfig.from_model_name(args.model_name), hf_config,
                                 hf_config[_NATIVE_LAYER_COUNT])
    return Counter((layer.attention_variant, layer.plans_tile_schedule) for layer in layers)


def expand_attention_kernels(layer_kernels: Mapping[int, Counter[KernelIdentity]],
                             layer_variants: Mapping[int, str],
                             layer_roles: Mapping[int, bool],
                             role_counts: Mapping[tuple[str, bool], int]) -> Counter[KernelIdentity]:
    """Expand equal reduced multisets by the full-model count of each layer role."""
    layers_by_role = defaultdict(list)
    for layer, variant in layer_variants.items():
        layers_by_role[(variant, layer_roles[layer])].append(layer)
    extra = layers_by_role.keys() - role_counts.keys()
    if extra:
        raise ValueError(f"Identity reduced layer roles absent from the full model: {sorted(extra)!r}")
    expected = Counter()
    for role, count in role_counts.items():
        layers = sorted(layers_by_role[role])
        if not layers:
            raise ValueError(f"Identity has no reduced layers for full-model role {role!r}")
        for layer in layers:
            if layer not in layer_kernels:
                raise ValueError(f"Identity role {role!r} has no attention kernels for layer {layer}")
        kernels = layer_kernels[layers[0]]
        if any(layer_kernels[layer] != kernels for layer in layers[1:]):
            raise ValueError(f"Identity role {role!r} has unequal kernel multisets in layers {layers}")
        for identity, occurrences in kernels.items():
            expected[identity] += count * occurrences
    return expected


def native_step_class(phase: str, batch_size: int) -> str:
    if phase == PHASE_FIRST_TAIL:
        return FIRST_TAIL_ALONE if batch_size == 1 else FIRST_TAIL_WITH_DECODES
    if phase in (PHASE_PREFILL, PHASE_DECODE, PHASE_DUMMY):
        return phase
    raise ValueError(f"Identity cannot match pilot phase {phase!r}")


def _identity(row) -> KernelIdentity:
    dimensions = []
    for field in (GRID, BLOCK):
        value = row[field]
        if (not isinstance(value, (list, tuple)) or len(value) != _DIMENSIONS
                or any(type(item) is not int for item in value)):
            raise ValueError(f"Identity kernel {row[NAME]!r} requires {field} with three ints")
        dimensions.append(tuple(value))
    return row[NAME], *dimensions


def _kernel_rows(kernels: Counter[KernelIdentity]) -> list[dict[str, Any]]:
    return [{NAME: name, GRID: list(grid), BLOCK: list(block), _COUNT: count}
            for (name, grid, block), count in sorted(kernels.items())]


def match_kernels(expected: Counter[KernelIdentity], kind: str, mode: str,
                  padded_rows: int, eager_rows: int | None,
                  candidates: Sequence[NativeCandidate]) -> dict[str, Any]:
    shape = kind, mode, padded_rows, eager_rows
    selected = [candidate for candidate in candidates if candidate.shape_key == shape]
    closest = min(selected, key=lambda candidate: sum((expected - candidate.kernels).values()), default=None)
    missing = None if closest is None else expected - closest.kernels
    grid_mismatches = []
    if closest is not None:
        for (name, grid, block), count in sorted(missing.items()):
            other_grids = [list(other_grid) for (other_name, other_grid, other_block) in sorted(closest.kernels)
                           if other_name == name and other_block == block and other_grid != grid]
            if other_grids:
                grid_mismatches.append({NAME: name, BLOCK: list(block), _EXPECTED_GRID: list(grid),
                                        _NATIVE_GRIDS: other_grids, _MISSING_COUNT: count})
    return {_STATUS: NO_REFERENCE if closest is None else FAIL if missing else PASS,
            _CANDIDATE_COUNT: len(selected), _EXPECTED_KERNEL_COUNT: sum(expected.values()),
            _MISSING_KERNEL_COUNT: None if missing is None else sum(missing.values()),
            _MISSING_KERNELS: None if missing is None else _kernel_rows(missing),
            _GRID_MISMATCHES: grid_mismatches,
            _CLOSEST_FORWARD_COUNT: None if closest is None else closest.forward_count}


def _status_counts(steps: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(step[_STATUS] for step in steps)
    return {status: counts[status] for status in (PASS, FAIL, NO_REFERENCE)}


def _shape_coverage(steps: Sequence[Mapping[str, Any]],
                    candidates: Sequence[NativeCandidate]) -> list[dict[str, Any]]:
    by_shape = defaultdict(list)
    native_counts = Counter()
    for step in steps:
        by_shape[tuple(step[_SHAPE_KEY])].append(step)
    for candidate in candidates:
        native_counts[candidate.shape_key] += candidate.forward_count
    return [{_SHAPE_KEY: list(shape), _PILOT_CASE_IDS: sorted({step[CASE_ID] for step in shaped_steps}),
             _STATUS_COUNTS: _status_counts(shaped_steps), _NATIVE_FORWARD_COUNT: native_counts[shape]}
            for shape, shaped_steps in sorted(by_shape.items())]


def check_identity(config: BridgeConfig, records: Sequence[ReplayRecords],
                   candidates: Sequence[NativeCandidate]) -> dict[str, Any]:
    plans, roles = _plans(config), planner_roles(config, records)
    counts = layer_role_counts(config)
    by_run = {(record.engine[ENGINE_CAPTURE][ENGINE_MODE], record.engine[ENGINE_INDEX]): record for record in records}
    steps, cases = [], []
    for run in config.runs:
        plan, record = plans[run.mode], by_run[(run.mode, run.engine_index)]
        case_specs = {case.case_id: case for case in plan.cases_for_engine(run.engine_index)}
        layer_variants = {layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers}
        layer_roles = {layer: roles[(run.mode, run.engine_index, layer)] for layer in layer_variants}
        intervals = defaultdict(lambda: defaultdict(Counter))
        for row in record.intervals:
            if row[ACTIVITY] == KERNEL and row[GLOBAL_LAYER_ID] is not None:
                intervals[(row[CASE_ID], row[STEP_INDEX])][row[GLOBAL_LAYER_ID]][_identity(row)] += 1
        for case_id in run.case_ids:
            spec = case_specs[case_id]
            case_steps = []
            for step in record.steps:
                if step[CASE_ID] != case_id:
                    continue
                kind = native_step_class(spec.phase, spec.batch_size)
                mode = step[CUDAGRAPH_DISPATCH_KEY][CG_MODE]
                expected = expand_attention_kernels(intervals[(case_id, step[STEP_INDEX])], layer_variants, layer_roles, counts)
                padded_rows = step[CUDAGRAPH_DISPATCH_KEY][NUM_TOKENS]
                observed_rows = _padded_rows(expected, f"Pilot {run.key} case {case_id!r} step {step[STEP_INDEX]}")
                if observed_rows != padded_rows:
                    raise ValueError(f"Pilot {run.key} case {case_id!r} step {step[STEP_INDEX]}: "
                                     f"row-count kernel grid x {observed_rows} differs from dispatch num_tokens {padded_rows}")
                eager_rows = None
                if mode in _EAGER_ATTENTION_MODES:
                    eager_rows = (padded_rows if spec.phase == PHASE_DUMMY else
                                  sum(request[_NUM_SCHEDULED_TOKENS] for request in step[_REQUESTS]))
                result = {_RUN: run.key, CASE_ID: case_id, STEP_INDEX: step[STEP_INDEX], _NATIVE_CLASS: kind,
                          CG_MODE: mode, _SHAPE_KEY: [kind, mode, padded_rows, eager_rows],
                          **match_kernels(expected, kind, mode, padded_rows, eager_rows, candidates)}
                steps.append(result)
                case_steps.append(result)
            case_counts = _status_counts(case_steps)
            case_status = (FAIL if case_counts[FAIL] else
                           NO_REFERENCE if case_counts[NO_REFERENCE] else PASS)
            cases.append({_RUN: run.key, CASE_ID: case_id, _STEP_COUNT: len(case_steps),
                          _PASS_COUNT: case_counts[PASS], _STATUS_COUNTS: case_counts, _STATUS: case_status})
    step_counts = _status_counts(steps)
    verdict = FAIL if step_counts[FAIL] else PASS if step_counts[PASS] else NO_REFERENCE
    return {_VERDICT: verdict, _STATUS_COUNTS: step_counts, _SHAPE_COVERAGE: _shape_coverage(steps, candidates),
            _LAYER_ROLE_COUNTS: [{_VARIANT: variant, _PLANS_TILE_SCHEDULE: planner, _COUNT: count}
                                 for (variant, planner), count in sorted(counts.items())], _STEPS: steps, _CASES: cases,
            _NATIVE_CANDIDATES: [{_CLASS: row.step_class, _MODE: row.mode, _SHAPE_KEY: list(row.shape_key),
                                  _FORWARD_COUNT: row.forward_count, _ROLES: list(row.roles),
                                  _KERNEL_COUNT: sum(row.kernels.values())} for row in candidates]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native_trace_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        records = read_replay_records([config.output_root / run.mode / "raw" / f"dsv4_raw_e{run.engine_index}.jsonl"
                                      for run in config.runs])
        report = check_identity(config, records, native_candidates(args.native_trace_dirs))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {_VERDICT: FAIL, _STEPS: [], _CASES: [], _SHAPE_COVERAGE: [],
                  _STATUS_COUNTS: _status_counts([]), _REASONS: [str(error)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    coverage = report[_SHAPE_COVERAGE]
    referenced = [row[_SHAPE_KEY] for row in coverage if row[_NATIVE_FORWARD_COUNT]]
    unreferenced = [row[_SHAPE_KEY] for row in coverage if not row[_NATIVE_FORWARD_COUNT]]
    print(f"[dsv4_identity] {report[_VERDICT]}: {len(report[_STEPS])} steps, {report[_STATUS_COUNTS]}")
    print(f"[dsv4_identity] referenced shapes ({len(referenced)}): {referenced}")
    print(f"[dsv4_identity] unreferenced shapes ({len(unreferenced)}): {unreferenced}")
    for reason in report.get(_REASONS, []):
        print(f"[dsv4_identity] {reason}")
    return int(report[_VERDICT] != PASS)


__all__ = ["NativeCandidate", "check_identity", "expand_attention_kernels", "main", "match_kernels",
           "native_candidates", "native_step_class", "layer_role_counts"]

if __name__ == "__main__":
    raise SystemExit(main())
