"""Estimate candidate full-grid reconstruction cost; no measurement is authorized here."""
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
plans = [plan_from_spec(root / name / 'package_plan.json') for name in ('g1h', 'g1h_supplement')]
cases = {case.case_id: case for plan in plans for case in plan.cases if case.case_id in required}
assert cases.keys() == required
steps = sum(case.estimated_steps for case in cases.values())
engines = {case.engine_index for case in cases.values()}
options = []
for repeats in (1, 2, 4, 8, 16):
    seconds = len(engines) * 486.104 + repeats * (steps * 0.205 + len(cases) * 0.2)
    options.append({'reconstructions_per_case': repeats, 'measured_steps': repeats * len(cases),
                    'estimated_engine_steps': repeats * steps, 'estimated_gpu_hours': seconds / 3600})
result = {'classification': 'CONDITIONAL_COST_ESTIMATES_NO_LAUNCH',
          'cases': len(cases), 'engine_indices': sorted(engines),
          'one_reconstruction_engine_steps': steps, 'options': options,
          'table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
          'model_source': 'g1h/README.md timing model; same constants as the frozen probe',
          'limits': 'Planning estimates only, not timeout exposure or platform billing; probe results must justify any next step.'}
output = root / 'a6_repeatability_20261101/reconstruction_costs.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
