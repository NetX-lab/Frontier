"""Write the package inputs for two more measurement passes of the supplement-only workloads.

Every published workload that only G1h-s01 measured is cloned twice. Within each
engine the clones keep the G1h-s01 case order, pass 1 before pass 2. All other
package settings are copied from the G1h-s01 plan; only the job identity, output
location, timeout and run lists change.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPLEMENT = HERE.parent / 'g1h_supplement'
REPO = HERE.parents[3]
TABLE = REPO / 'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv'
PASSES = (1, 2)

with TABLE.open() as stream:
    published = {row['case_id'] for row in csv.DictReader(stream)}
plan = json.loads((SUPPLEMENT / 'package_plan.json').read_text())
pilot = json.loads((SUPPLEMENT / 'pilot_cases.json').read_text())
cases = {case['case_id']: case for case in pilot['cases']}
repeats = set(plan['repeat_case_ids'])
runs, clones, mapping = [], [], {}
for run in plan['runs']:
    selected = [case_id for case_id in run['case_ids'] if case_id in published]
    assert not repeats & set(selected)
    assert set(run['case_ids']) - set(selected) <= repeats
    case_ids = []
    for number in PASSES:
        for case_id in selected:
            clone = f'a6s-p{number}-{case_id}'
            assert cases[case_id]['engine_index'] == run['engine_index']
            assert (cases[case_id]['warmup'], cases[case_id]['measured']) == (0, 1)
            clones.append({**cases[case_id], 'case_id': clone})
            mapping[clone] = {'original': case_id, 'repeat': number}
            case_ids.append(clone)
    runs.append({**run, 'case_ids': case_ids})
originals = {item['original'] for item in mapping.values()}
assert len(originals) == 432 and all(case_id.startswith('g1hs-') for case_id in originals)
assert originals == {case_id for case_id in published if case_id.startswith('g1hs-')}
assert len(mapping) == len(clones) == 2 * len(originals)

package_plan = {key: value for key, value in plan.items() if key not in ('coverage_grid', 'repeat_case_ids')}
package_plan.update(name='A6_SUPPLEMENT_PASSES', task_id='H3O-A6-SUPPLEMENT-PASSES',
                    job_tag='dsv4-a6-supplement-passes-20261101-01',
                    output_dir='/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/'
                               'a6_supplement_passes_20261101_01',
                    timeout_seconds=5400, planning_gpu_hours_cap=1.5, runs=runs)
for name, value in (('package_plan.json', package_plan), ('case_mapping.json', mapping),
                    ('pilot_cases.json', {'engines': pilot['engines'], 'cases': clones})):
    with (HERE / name).open('x') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps({'originals': len(originals), 'clones': len(clones),
                  'runs': [(run['engine_index'], len(run['case_ids'])) for run in runs]}))
