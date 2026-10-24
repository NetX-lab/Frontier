"""Verify completed post-key receipts and retain the approved test-ID change."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

RECORDS = Path(__file__).resolve().parent.parent / 'post_key_regressions/evidence'
checks = []


def check(path, expected, size=None):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, path
    assert size is None or len(raw) == size, path
    checks.append({'path': str(path), 'sha256': expected})


for name in ('strict_archive_manifest.json', 'initial_archive_manifest.json'):
    for item in json.loads((RECORDS / name).read_text()):
        for kind in ('source', 'retained'):
            check(Path(item[kind]), item['sha256'], item['bytes'])
strict = json.loads((RECORDS / 'strict77/final06_receipt.json').read_text())
audit = strict['audit']
assert strict['source_revision'] == '46fe83ac55876c10f6dd8ff792705e87efd1905f'
assert strict['verdict'] == 'PASS'
assert audit['cases_compared'] == audit['identical_cases'] == 77
assert not any(audit['comparison_counts'].values())
for arm in ('baseline', 'candidate'):
    info = audit[arm]
    assert info['artifacts_rehashed'] == 451 and info['cache_filename_count'] == 519
    for kind, filename in [('manifest', 'manifest.json'), ('results', 'results.jsonl')]:
        check(Path(info['label_path']) / filename, info[kind + '_sha256'])
check(Path('/data/ycfeng/tmp/d29_workflow_fidelity_20261024_final06/comparison.json'), audit['comparison_sha256'])
short = json.loads((RECORDS / 'short_selections_comparison.json').read_text())
selections = []
for entry in short['selections']:
    values = {}
    for version in ('old_xml', 'new_xml'):
        item = entry[version]
        path = Path(item['path'])
        check(path, item['sha256'], item['bytes'])
        cases = ET.parse(path).findall('.//testcase')
        assert not any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped'))
        values[version] = {case.attrib['classname'] + '::' + case.attrib['name'] for case in cases}
        assert len(values[version]) == len(cases)
    assert sorted(values['old_xml'] - values['new_xml']) == entry['id_transitions']['missing']
    assert sorted(values['new_xml'] - values['old_xml']) == entry['id_transitions']['new_ids']
    assert len(values['new_xml']) == entry['raw_counts']['pass'] and entry['raw_exit'] == 0
    selections.append({'suite': entry['suite'], 'pass': len(values['new_xml']),
        'missing': entry['id_transitions']['missing'], 'new_ids': entry['id_transitions']['new_ids']})
assert [entry['pass'] for entry in selections] == [101, 197, 53]
manifest = json.loads((RECORDS / 'records_manifest.json').read_text())
for item in manifest['files']:
    check(RECORDS.parent / item['path'], item['sha256'], item['bytes'])
final = json.loads((RECORDS / 'final_verification.json').read_text())
assert final['source_revision'] == strict['source_revision']
assert final['raw_exact_status_gate'] == 'FAIL_RETAINED_ID_LOSS_OR_STATUS_CHANGE'
assert final['all_required_raw_predicates_passed'] is False
suite_cases = {}
for suite in final['suites']:
    for item in [suite['junit'], suite['execution'], *suite['logs']]:
        check(Path(item['path']), item['sha256'], item['bytes'])
    cases = {}
    for case in ET.parse(suite['junit']['path']).findall('.//testcase'):
        key = case.get('classname', '') + '::' + case.attrib['name']
        assert key not in cases, key
        cases[key] = next((tag for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None), 'pass')
    assert dict(Counter(cases.values())) == suite['counts']
    suite_cases[suite['suite']] = cases
guard = json.loads((RECORDS / 'source_final_receipt.json').read_text())
for item in guard['input_before_after']:
    assert item['before'] == item['after']
    check(Path(item['after']['path']), item['after']['sha256'], item['after']['bytes'])
old = {}
for case in ET.parse('/data/ycfeng/tmp/d29_unit_parent05.xml').findall('.//testcase'):
    key = case.get('classname', '') + '::' + case.attrib['name']
    assert key not in old, key
    old[key] = next((tag for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None), 'pass')
current = suite_cases['unit']
assert dict(Counter(old.values())) == {'pass': 8018, 'failure': 279, 'error': 32, 'skipped': 281}
missing = sorted(old.keys() - current.keys())
added = sorted(current.keys() - old.keys())
assert missing == final['one_unique_planned_missing_id'] == selections[1]['missing']
assert added == sorted(item['id'] for item in final['new_source_cases'])
assert all(current[key] == 'pass' for key in added)
assert all(old[key] == current[key] for key in old.keys() & current.keys())
for status in ('failure', 'error', 'skipped'):
    assert {key for key, value in old.items() if value == status} == {key for key, value in current.items() if value == status}
for item in final['governance_reconciliation']:
    assert current[item['id']] == 'failure' and suite_cases['governance52'][item['id']] == 'pass'
for key in ('source_guard', 'canonical_comparison', 'canonical_legacy_comparison', 'first_environment_run_preserved_not_canonical'):
    item = final[key]
    check(Path(item['path']), item['sha256'], item['bytes'])
source = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-07')
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == final['source_revision']
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip() == final['source_tree']
assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True) == ''
assert subprocess.check_output(['git', '-C', str(source / 'frontier/cc_backend/backends/collective-sim'), 'rev-parse', 'HEAD'], text=True).strip() == final['submodule_revision']
item = guard['source_inventory']
check(Path(item['path']), item['sha256'], item['bytes'])
inventory = json.loads(Path(item['path']).read_text())
assert len(inventory) == 2629
for item in inventory:
    check(source / item['path'], item['sha256'], item['bytes'])
scripts = json.loads((RECORDS / 'command_bindings.json').read_text())['scripts']
scripts += json.loads((RECORDS / 'environment_reconciliation/canonical_reconciliation_preflight.json').read_text())['script_bindings']
assert len(scripts) == 8
for item in scripts:
    check(Path(item['path']), item['sha256'], item['bytes'])
receipt = {'task_id': 'D29-POST-KEY-PARENT', 'status': 'ACCEPTED_NO_NEW_CODE_FAILURE_WITH_APPROVED_TEST_ID_REPLACEMENT',
    'source_revision': strict['source_revision'], 'source_tree': strict['source_tree'],
    'checks': [item for item in checks if not Path(item['path']).is_relative_to(source)],
    'hash_checks': len(checks), 'source_files_rehashed': len(inventory), 'scripts_rehashed': len(scripts),
    'guarded_inputs_rehashed': len(guard['input_before_after']),
    'record_files_rehashed': len(manifest['files']),
    'strict77': {'identical': 77, 'compared': 77, 'comparison_counts': audit['comparison_counts'],
        'owner_rehashed_artifacts_per_side': 451, 'owner_cache_filenames_per_side': 519,
        'parent_scope': 'Receipt/copy and manifest/results/comparison hashes checked; 451 per-side artifact rehash remains owner-auditor evidence.'},
    'short_selections': selections, 'counts_not_summed': True,
    'unit': {'counts': dict(Counter(current.values())), 'baseline_counts': dict(Counter(old.values())),
        'raw_exit': 1, 'missing_old_ids': missing, 'new_ids_all_pass': added,
        'retained_status_changes': [], 'failure_error_skip_id_differences': []},
    'governance': {'counts': dict(Counter(suite_cases['governance52'].values())),
        'covered_old_torch_failures': len(final['governance_reconciliation'])},
    'raw_exact_status_gate': final['raw_exact_status_gate'], 'raw_exact_ID_gate_waived': False,
    'approved_replacement_admission': 'D29-RUN-INPUT-SNAPSHOT requires the fixed-run input test instead of the former hot-update test; the raw old-ID presence predicate remains FAIL.',
    'limits': ['One removed/new metadata ID is the approved input lifetime replacement, not an alias.',
        'Raw full-unit exit1 and historical279 failures/32 errors remain; this is not all-green.',
        'Full-unit and final guards complete on46fe only; named-model-query successor is outside this receipt.',
        'No full CLI/native qualification.'],
    'checker_correction': 'Initial inline check assumed results.json; directory discovery confirmed results.jsonl. No receipt was written before correction.'}
output = Path(__file__).with_name('post_key_regression_acceptance.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'hash_checks': len(checks),
    'strict77': '77/77', 'short': selections, 'unit': receipt['unit'],
    'governance': receipt['governance'], 'source_files': len(inventory),
    'receipt_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))
