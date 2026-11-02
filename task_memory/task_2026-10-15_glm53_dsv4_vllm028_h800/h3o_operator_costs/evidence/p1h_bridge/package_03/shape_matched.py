"""Identity per pilot step against native forwards of the same class, mode and padded token count.

The padded token count is read from the grid of _fused_q_kv_rmsnorm_kernel (grid x = rows).
Steps without a native forward at that count are reported as having no reference.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dsv4_graph_bridge_check import load_config
from dsv4_kernel_identity_check import (
    _identity, expand_attention_kernels, layer_role_counts, native_candidates, native_step_class, _plans,
)
from dsv4_composition_check import planner_roles
from frontier.profiling.kernel_gap.dsv4_replay_gaps import (
    ACTIVITY, CASE_ID, CG_MODE, CUDAGRAPH_DISPATCH_KEY, ENGINE_CAPTURE, ENGINE_INDEX, ENGINE_MODE,
    GLOBAL_LAYER_ID, KERNEL, STEP_INDEX, read_replay_records,
)

INDICATOR = '_fused_q_kv_rmsnorm_kernel'


def rows(kernels):
    grids = {grid for (name, grid, block) in kernels if name == INDICATOR}
    if len(grids) != 1:
        return None
    return next(iter(grids))[0]


config = load_config(Path(sys.argv[1]))
out = Path(sys.argv[3])
records = read_replay_records([config.output_root / run.mode / 'raw' / f'dsv4_raw_e{run.engine_index}.jsonl' for run in config.runs])
candidates = native_candidates([Path(sys.argv[2])])
by_shape = defaultdict(list)
for candidate in candidates:
    by_shape[(candidate.step_class, candidate.mode, rows(candidate.kernels))].append(candidate)
print('native shapes', sorted((k[0], k[1], k[2], sum(c.forward_count for c in v)) for k, v in by_shape.items()))
plans, roles, counts = _plans(config), planner_roles(config, records), layer_role_counts(config)
by_run = {(r.engine[ENGINE_CAPTURE][ENGINE_MODE], r.engine[ENGINE_INDEX]): r for r in records}
report = []
for run in config.runs:
    plan, record = plans[run.mode], by_run[(run.mode, run.engine_index)]
    specs = {case.case_id: case for case in plan.cases_for_engine(run.engine_index)}
    variants = {layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers}
    layer_roles = {layer: roles[(run.mode, run.engine_index, layer)] for layer in variants}
    intervals = defaultdict(lambda: defaultdict(Counter))
    for row in record.intervals:
        if row[ACTIVITY] == KERNEL and row[GLOBAL_LAYER_ID] is not None:
            intervals[(row[CASE_ID], row[STEP_INDEX])][row[GLOBAL_LAYER_ID]][_identity(row)] += 1
    for step in record.steps:
        spec = specs[step[CASE_ID]]
        kind, mode = native_step_class(spec.phase, spec.batch_size), step[CUDAGRAPH_DISPATCH_KEY][CG_MODE]
        expected = expand_attention_kernels(intervals[(step[CASE_ID], step[STEP_INDEX])], variants, layer_roles, counts)
        shape = rows(expected)
        matched = by_shape.get((kind, mode, shape), [])
        best = min(matched, key=lambda c: sum((expected - c.kernels).values()), default=None)
        missing = None if best is None else expected - best.kernels
        extra_names = None if best is None else sorted({name for (name, _, _) in missing})
        report.append({'case_id': step[CASE_ID], 'step': step[STEP_INDEX], 'class': kind, 'mode': mode, 'rows': shape,
                       'dispatch': step[CUDAGRAPH_DISPATCH_KEY], 'native_matches': len(matched),
                       'native_forwards': sum(c.forward_count for c in matched),
                       'status': 'NO_REFERENCE' if best is None else 'PASS' if not missing else 'FAIL',
                       'missing': None if best is None else [[n, list(g), list(b), c] for (n, g, b), c in sorted(missing.items())]})
out.write_text(json.dumps(report, indent=1) + '\n')
summary = defaultdict(Counter)
for row in report:
    summary[(row['case_id'], row['class'], row['mode'], row['rows'])][row['status']] += 1
for key, value in summary.items():
    print(key, dict(value))
print('totals', Counter(row['status'] for row in report))
