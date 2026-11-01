"""Estimate extra-pass cost for the supplement-only workloads; no measurement is authorized here.

Uses the same frozen timing model as a6_repeatability_20261101/reconstruction_costs.py.
"""
import csv
import hashlib
import json
from pathlib import Path

from v4_attention_coverage import plan_from_spec

root = Path(__file__).resolve().parent.parent
repo = Path.cwd()
table = repo / 'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv'
with table.open() as stream:
    required = {row['case_id'] for row in csv.DictReader(stream)}
plans = {name: plan_from_spec(root / name / 'package_plan.json') for name in ('g1h', 'g1h_supplement')}
cases = {name: {case.case_id: case for case in plan.cases if case.case_id in required} for name, plan in plans.items()}
assert set(cases['g1h']) | set(cases['g1h_supplement']) == required
assert not set(cases['g1h']) & set(cases['g1h_supplement'])


def cost(selected, repeats):
    steps = sum(case.estimated_steps for case in selected.values())
    engines = {case.engine_index for case in selected.values()}
    seconds = len(engines) * 486.104 + repeats * (steps * 0.205 + len(selected) * 0.2)
    return {'cases': len(selected), 'engine_indices': sorted(engines), 'reconstructions_per_case': repeats,
            'estimated_engine_steps': repeats * steps, 'estimated_gpu_hours': seconds / 3600}


result = {'classification': 'CONDITIONAL_COST_ESTIMATES_NO_LAUNCH',
          'table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
          'supplement_only': [cost(cases['g1h_supplement'], repeats) for repeats in (1, 2)],
          'original_grid': [cost(cases['g1h'], repeats) for repeats in (1, 2)],
          'all_cases': [cost({**cases['g1h'], **cases['g1h_supplement']}, repeats) for repeats in (1, 2)],
          'model_source': 'same constants as a6_repeatability_20261101/reconstruction_costs.py',
          'limits': 'Planning estimates only, not timeout exposure or platform billing.'}
output = root / 'a6_noise_20261101/supplement_pass_costs.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
