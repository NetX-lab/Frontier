"""Count the FlashMLA decode tile-schedule planner kernel per layer in every recorded step.

Expected after B4: every step that runs decode attention launches the planner
exactly once in each of layers 0-3 and in no other layer.
"""
import json
import sys
from collections import Counter, defaultdict

NAME = 'smxx::decode::get_mla_metadata_kernel'
EXPECTED = {0: 1, 1: 1, 2: 1, 3: 1}

failures = 0
for path in sys.argv[1:]:
    steps, counts = {}, defaultdict(Counter)
    for line in open(path):
        row = json.loads(line)
        key = (row.get('case_id'), row.get('step_index'))
        if row['kind'] == 'step':
            steps[key] = (row['phase'], row['cudagraph_dispatch']['cg_mode'])
        elif row['kind'] == 'interval' and row['name'].startswith(NAME):
            counts[key][row['global_layer_id']] += 1
    summary = Counter()
    for key, (phase, mode) in sorted(steps.items()):
        layers = dict(sorted(counts.get(key, Counter()).items()))
        verdict = 'none' if not layers else 'expected' if layers == EXPECTED else 'unexpected'
        summary[(phase, mode, verdict)] += 1
        if verdict == 'unexpected':
            failures += 1
            print('UNEXPECTED', path, key, phase, mode, layers)
    orphan = set(counts) - set(steps)
    if orphan:
        failures += 1
        print('ORPHAN planner intervals outside recorded steps', path, sorted(orphan)[:5])
    print(path)
    for (phase, mode, verdict), number in sorted(summary.items()):
        print(f'  {phase:10s} {mode:10s} {verdict:10s} {number} steps')
print('planner_layers', 'FAIL' if failures else 'PASS', failures)
sys.exit(int(failures > 0))
