"""Verify completed post-key receipts without touching the running unit suite."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

RECORDS = Path('/data/ycfeng/Frontier/.worktrees/d29-post-key-regressions/task_memory/task_2026-10-22_step5_position_routing/post_key_regressions/evidence')
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
receipt = {'task_id': 'D29-POST-KEY-PARENT', 'status': 'PARTIAL_ACCEPTED_STRICT77_AND_SHORT_ONLY',
    'source_revision': strict['source_revision'], 'source_tree': strict['source_tree'], 'checks': checks,
    'strict77': {'identical': 77, 'compared': 77, 'comparison_counts': audit['comparison_counts'],
        'owner_rehashed_artifacts_per_side': 451, 'owner_cache_filenames_per_side': 519,
        'parent_scope': 'Receipt/copy and manifest/results/comparison hashes checked; 451 per-side artifact rehash remains owner-auditor evidence.'},
    'short_selections': selections, 'counts_not_summed': True,
    'limits': ['One removed/new metadata ID is the approved input lifetime replacement, not an alias.',
        'Full-unit and final guards pending.', 'No full CLI/native qualification.'],
    'checker_correction': 'Initial inline check assumed results.json; directory discovery confirmed results.jsonl. No receipt was written before correction.'}
output = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_key_regression_acceptance.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'hash_checks': len(checks),
    'strict77': '77/77', 'short': selections,
    'receipt_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))
