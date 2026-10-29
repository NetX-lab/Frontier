"""Compare exact locked test IDs without hiding unsuccessful raw outcomes."""
import collections
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing')


def digest(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def parse(path):
    tree = ET.parse(path).getroot()
    result = {}
    for case in tree.iter('testcase'):
        key = case.get('classname', '') + '::' + case.get('name', '')
        assert key not in result, (str(path),key)
        status = next((s for s in ('failure','error','skipped') if case.find(s) is not None),'pass')
        node = case.find(status)
        result[key] = {'status':status,'message':'' if node is None else node.get('message',''),
                       'text':'' if node is None else ''.join(node.itertext())}
    return result, [s.attrib for s in tree.iter('testsuite')]


def ids(result,status):
    return {key for key,value in result.items() if value['status'] == status}


def transitions(base,candidate):
    return {'missing':sorted(set(base)-set(candidate)),
            'new_ids':sorted(set(candidate)-set(base)),
            'pass_to_skip':sorted(k for k in ids(base,'pass') & set(candidate) if candidate[k]['status']=='skipped'),
            'pass_to_failure_or_error':sorted(k for k in ids(base,'pass') & set(candidate) if candidate[k]['status'] in ('failure','error'))}


def main():
    raw_path = Path('/data/ycfeng/tmp/d30_unit_baseline01.xml')
    candidate_baseline_path = Path('/data/ycfeng/tmp/d30_unit_candidate01.xml')
    overlay_path = Path('/data/ycfeng/tmp/d30_canonical_count_green01.xml')
    base,_ = parse(raw_path)
    latest,_ = parse(candidate_baseline_path)
    overlay,_ = parse(overlay_path)
    assert set(overlay) <= set(latest)
    latest.update(overlay)
    locked = json.loads((TASK/'repairs/evidence/d29_regression_baseline01.json').read_text())
    for status,field in [('failure','unit_failure_ids'),('error','unit_error_ids')]:
        assert ids(base,status) == ids(latest,status) == set(locked[field])
    full_path = ROOT/'unit/results.xml'
    full,attributes = parse(full_path)
    supplement_path = ROOT/'supplement101/results.xml'
    supplement,_ = parse(supplement_path)
    prior, _ = parse(Path('/data/ycfeng/tmp/d29_unit_parent03.xml'))
    evidence = {'baseline_xml':digest(raw_path), 'latest_baseline_xml':digest(candidate_baseline_path),
                'baseline_overlay':digest(overlay_path), 'locked_ids':digest(TASK/'repairs/evidence/d29_regression_baseline01.json'),
                'candidate':digest(full_path),'supplement':digest(supplement_path)}
    report = {'task_id':'D29-E41-FINAL-REGRESSIONS','source_revision':'e41abc073500d61f224eda4fca1123e74e40544d',
              'raw_status_counts':dict(collections.Counter(v['status'] for v in full.values())),
              'raw_suite':attributes,'raw_candidate_exit_code':json.loads((ROOT/'unit/execution.json').read_text())['executions'][0]['returncode'],
              'evidence':evidence,'raw_baseline_transitions':transitions(base,full),
              'latest_baseline_transitions':transitions(latest,full), 'prior_final03_transitions':transitions(prior,full)}
    reconciliation = []
    for status,field in [('failure','unit_failure_ids'),('error','unit_error_ids')]:
        old = set(locked[field]); new = ids(full,status)
        report[status] = {'baseline':len(old),'candidate':len(new),'new':sorted(new-old),'removed':sorted(old-new)}
        for key in sorted(new-old):
            failure = full[key]
            if status == 'failure':
                covered = key in supplement and supplement[key]['status']=='pass' and "No module named 'aiohttp'" in failure['text']
                reconciliation.append({'id':key,'status':status,'observed_exception':failure['message'],
                                       'covered_same_id_pass':covered,'traceback':failure['text']})
            else:
                module = key.split('::')[-1]
                tested = sorted(k for k in supplement if k.startswith(module+'::'))
                covered = "No module named 'torch'" in failure['text'] and len(tested)==16 and all(supplement[k]['status']=='pass' for k in tested)
                reconciliation.append({'id':key,'status':status,'observed_exception':failure['message'],
                                       'covered_same_module_16_pass':covered,'covered_ids':tested,'traceback':failure['text']})
    report['environment_reconciliation'] = reconciliation
    report['supplement_counts'] = dict(collections.Counter(v['status'] for v in supplement.values()))
    memberships = []
    for suite,old_xml in [('metadata196','d29_prepare_metadata_parent09.xml'),('catalogue53','d29_catalogue_parent08.xml')]:
        data,_ = parse(ROOT/suite/'results.xml')
        old_data,_ = parse(Path('/data/ycfeng/tmp')/old_xml)
        memberships.append({'suite':suite,'cases':len(data),
                            'old_ids_removed':sorted(set(old_data)-set(data)), 'new_ids':sorted(set(data)-set(old_data)),
                            'not_passing_in_focused':sorted(k for k,v in data.items() if v['status']!='pass'),
                            'not_passing_in_full':sorted(k for k in data if k not in full or full[k]['status']!='pass')})
    report['affected_memberships'] = memberships
    passed = report['raw_candidate_exit_code']==1 and all(not report[s]['removed'] for s in ['failure','error'])
    passed = passed and all(r.get('covered_same_id_pass',r.get('covered_same_module_16_pass',False)) for r in reconciliation)
    passed = passed and report['supplement_counts']=={'pass':101}
    passed = passed and all(not report[f][k] for f in ['raw_baseline_transitions','latest_baseline_transitions'] for k in ['missing','pass_to_skip'])
    passed = passed and all(not m[k] for m in memberships for k in ['old_ids_removed','new_ids','not_passing_in_focused','not_passing_in_full'])
    report['status'] = 'NO_NEW_CODE_FAILURES_WITH_EXPLICIT_ENVIRONMENT_SUPPLEMENT' if passed else 'FAIL_REQUIRED_REGRESSION_GATE'
    (ROOT/'unit_final04_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='environment_reconciliation'},indent=2))
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
