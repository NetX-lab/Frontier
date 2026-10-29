"""Verify completed frozen short-suite receipts without touching running suites."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

EVIDENCE = Path('/data/ycfeng/Frontier/.worktrees/d29-post-named-query-regressions/task_memory/task_2026-10-22_step5_position_routing/post_named_query_regressions/evidence')
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08')
EXPECTED = {'supplement101': 101, 'metadata196': 197, 'catalogue53': 53, 'governance52': 52}
OLD_ID = 'tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text'
NEW_ID = 'tests.unit.test_collective_sim_cc_backend::test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published'
checks = {}


def checked(path, expected_hash=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None:
        assert digest == expected_hash, path
    checks[str(path)] = {'sha256': digest, 'bytes': len(raw)}
    return raw


def cases(item):
    raw = checked(item['path'], item['sha256'])
    assert len(raw) == item['bytes']
    result = {}
    for case in ET.fromstring(raw).iter('testcase'):
        name = case.attrib['classname'] + '::' + case.attrib['name']
        assert name not in result, name
        result[name] = next((tag for tag in ('error', 'failure', 'skipped') if case.find(tag) is not None), 'pass')
    return result


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args]).decode().strip()


source = {'head': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
          'status': git('status', '--porcelain'),
          'submodule_head': git('-C', str(SOURCE / 'frontier/cc_backend/backends/collective-sim'), 'rev-parse', 'HEAD'),
          'submodule_status': git('-C', str(SOURCE / 'frontier/cc_backend/backends/collective-sim'), 'status', '--porcelain')}
assert source['head'] == 'dbe19f531ff68d9edafc480418e8927c40eb1c60'
assert source['tree'] == '6c43090097e92c6fc0c33aa7358e69c2cf092705'
assert source['submodule_head'] == '3dac53c544e546574cb6e8078cd14053d0916113'
assert source['status'] == source['submodule_status'] == ''
results = {}
for suite, expected_count in EXPECTED.items():
    execution_raw = checked(EVIDENCE / suite / 'execution.json')
    comparison_raw = checked(EVIDENCE / suite / 'comparison.json')
    execution, comparison = json.loads(execution_raw), json.loads(comparison_raw)
    assert execution['suite'] == comparison['suite'] == suite
    assert execution['source_before'] == execution['source_after'] == source
    assert execution['MemoryMax_bytes'] == 30 * 1024**3
    assert not execution['native_binary_projection'] and not execution['source_binary_present']
    assert all(item['returncode'] == 0 for item in execution['executions'])
    assert comparison['raw_exit'] == 0
    xml = comparison['candidate_xml']
    assert execution['junit'] == xml['path']
    raw_directory = Path(xml['path']).parent
    assert checked(raw_directory / 'execution.json') == execution_raw
    assert checked(raw_directory / 'comparison.json') == comparison_raw
    candidate = cases(xml)
    counts = dict(Counter(candidate.values()))
    assert counts == comparison['actual_counts'] == {'pass': expected_count}
    baselines = ([{'baseline': '46fe', 'xml': comparison['prior46fe_xml']}]
                 if suite == 'supplement101' else comparison['baselines'])
    differences = []
    for baseline in baselines:
        before = cases(baseline['xml'])
        missing = sorted(before.keys() - candidate.keys())
        added = sorted(candidate.keys() - before.keys())
        changed = sorted(name for name in before.keys() & candidate.keys() if before[name] != candidate[name])
        assert not changed
        approved_replacement = suite == 'metadata196' and baseline['baseline'] == '79'
        assert missing == ([OLD_ID] if approved_replacement else [])
        assert added == ([NEW_ID] if approved_replacement else [])
        differences.append({'baseline': baseline['baseline'], 'missing': missing,
                            'new': added, 'changed_statuses': changed,
                            'raw_presence': 'FAIL_APPROVED_OLD_ID_MISSING' if missing else 'PASS'})
    results[suite] = {'counts': counts, 'raw_exit': 0, 'comparisons': differences,
                      'execution_receipt': str(EVIDENCE / suite / 'execution.json'),
                      'versions': execution['versions']}
receipt = {'task_id': 'D29-POST-NAMED-QUERY-PARENT', 'status': 'PASS_SHORT_SUITES_ONLY',
           'source': source, 'independent_suites': results, 'checks': checks,
           'pending_regressions': ['strict77', 'canonical_full_unit', 'final_binding_guards'],
           'limits': ['Four completed suites only; counts are not summed.',
                      'No running-suite outputs were inspected and no test was rerun.',
                      'The approved79 metadata ID replacement remains visible; raw presence is not waived.',
                      'Whole CLI6144/1024/2400s and native/GPU/G3/E2E are separate unfinished gates.']}
output = Path(__file__).with_name('post_named_query_regression_acceptance.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'source': source, 'hash_checks': len(checks),
                  'independent_counts': {name: result['counts'] for name, result in results.items()},
                  'receipt_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))
