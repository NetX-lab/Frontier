"""Compare scopes and attention kernel names of the first step of two dummy cases (one layer each variant)."""
import json
import sys
from collections import Counter, defaultdict

path, cases = sys.argv[1], sys.argv[2:]
first, kernels, steps = {}, defaultdict(Counter), {}
for line in open(path):
    row = json.loads(line)
    if row['kind'] == 'step' and row['case_id'] in cases and row['case_id'] not in steps:
        steps[row['case_id']] = row
for line in open(path):
    row = json.loads(line)
    if row['kind'] == 'interval' and row['case_id'] in steps and row['step_index'] == steps[row['case_id']]['step_index'] \
            and row['global_layer_id'] in (0, 2, 3):
        kernels[(row['case_id'], row['global_layer_id'])][(row['scope'], row['name'][:90])] += 1
for case in cases:
    step = steps[case]
    print(case, step['cudagraph_dispatch'], 'requests', step['requests'])
    print('   csa_attention_metadata', json.dumps(step.get('csa_attention_metadata'))[:400])
for layer in (0, 2, 3):
    a, b = kernels[(cases[0], layer)], kernels[(cases[1], layer)]
    print('layer', layer, 'only in', cases[0], sorted(set(a) - set(b)))
    print('layer', layer, 'only in', cases[1], sorted(set(b) - set(a)))
