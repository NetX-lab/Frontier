"""Per dummy case: planner kernels per step, dispatch, and attention kernels per layer."""
import json
import sys
from collections import Counter, defaultdict

NAME = 'smxx::decode::get_mla_metadata_kernel'
path = sys.argv[1]
steps, planners, scopes = {}, Counter(), defaultdict(Counter)
for line in open(path):
    row = json.loads(line)
    if row.get('phase') != 'dummy' and row['kind'] == 'step':
        continue
    key = (row.get('case_id'), row.get('step_index'))
    if row['kind'] == 'step':
        steps[key] = row
    elif row['kind'] == 'interval' and str(row.get('case_id', '')).startswith('e0-dummy'):
        if row['name'].startswith(NAME):
            planners[key] += 1
        scopes[key][row['scope']] += 1
by_case = defaultdict(list)
for key, row in sorted(steps.items()):
    by_case[key[0]].append((key[1], row['cudagraph_dispatch'], planners.get(key, 0), len(scopes[key]), sum(scopes[key].values())))
for case, rows in by_case.items():
    print(case)
    for step, dispatch, number, nscopes, nintervals in rows:
        print('   step', step, dispatch, 'planners', number, 'scopes', nscopes, 'intervals', nintervals)
