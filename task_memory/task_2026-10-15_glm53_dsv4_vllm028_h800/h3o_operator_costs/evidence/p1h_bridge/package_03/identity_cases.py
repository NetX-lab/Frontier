"""Per case: status counts, and the names of missing kernels split into grid-only and other misses."""
import json
import sys
from collections import Counter, defaultdict

report = json.load(open(sys.argv[1]))
for case in report['cases']:
    print(case['case_id'], case['status'], f"{case['pass_count']}/{case['step_count']}", 'margin' if case['margin'] else '')
by_case = defaultdict(list)
for step in report['steps']:
    by_case[(step['run'], step['case_id'])].append(step)
for key, steps in by_case.items():
    failing = [step for step in steps if step['status'] != 'PASS']
    if not failing:
        continue
    step = failing[0]
    grid_names = {row['name'] for row in step['grid_mismatches']}
    names = Counter()
    for row in step['missing_kernels']:
        names[row['name']] += row['count']
    print('==', key, step['native_class'], step['cg_mode'], 'step', step['step_index'], 'expected', step['expected_kernel_count'],
          'missing', step['missing_kernel_count'], 'candidates', step['candidate_count'], 'failing steps', len(failing))
    for name, count in names.most_common():
        print(f"   {count:4d} {'grid ' if name in grid_names else 'NAME '} {name[:150]}")
