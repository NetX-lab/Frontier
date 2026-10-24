"""Verify the named-query parent replay and final-source focused regression."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

RAW = Path('/data/ycfeng/tmp/d29_named_query_parent01')
OWNER = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-named-query')
EVIDENCE = OWNER / 'task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/repair_evidence'
checked = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
for name, item in manifest['files'].items():
    path = EVIDENCE / name
    assert digest(path) == item['sha256'] and path.stat().st_size == item['bytes'], name
    checked.append(name)
assert len(checked) == 21
owner = json.loads((EVIDENCE / 'verification.json').read_text())
verifier = EVIDENCE / 'verify.py'
source = verifier.read_text()
assert source.count('\ndef suite(path):') == 1
prefix = source.split('\ndef suite(path):')[0]
namespace = {'__file__': str(verifier), '__name__': 'parent_wave_verification'}
sys.argv = [str(verifier), str(RAW / 'wave')]
# Run every unchanged wave/source/input assertion, before owner-only suite/archive code.
exec(compile(prefix, str(verifier), 'exec'), namespace)
assert namespace['comparisons'] == owner['byte_comparisons']


def suite(path):
    cases = {}
    for case in ET.parse(path).iter('testcase'):
        name = case.get('classname', '') + '::' + case.attrib['name']
        assert name not in cases, name
        cases[name] = next((tag for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None), 'pass')
    return cases


baseline = suite(EVIDENCE / 'focused-baseline.xml')
candidate = suite(EVIDENCE / 'focused-candidate.xml')
parent = suite(RAW / 'pytest.xml')
assert parent == candidate
assert all(parent[name] == status for name, status in baseline.items())
new = sorted(parent.keys() - baseline.keys())
assert len(new) == 8 and all(parent[name] == 'pass' for name in new)
assert dict(Counter(parent.values())) == {'pass': 125, 'skipped': 1}
governance = {label: suite(EVIDENCE / f'governance-{label}.xml') for label in ('baseline', 'candidate')}
assert governance['baseline'] == governance['candidate']
assert dict(Counter(governance['candidate'].values())) == {'pass': 14, 'failure': 1}
reports = namespace['reports']
receipt = {
    'task_id': 'D29-NAMED-MODEL-QUERY-PARENT',
    'status': 'PASS_COMPONENT_REPLAY_AND_FOCUSED_TESTS',
    'source_commit': '1737ce9c4a761d2c57ee617f5f3a78d7bcfd9a58',
    'records_commit': '3d75c786781802ad2753cbdd81d989c9b694660c',
    'sources_at_replay': namespace['frozen']['sources'],
    'owner_artifacts_checked': checked,
    'full_file_comparisons': namespace['comparisons'],
    'diagnostic_headers': namespace['diagnostic_headers'],
    'parent_counts': dict(Counter(parent.values())),
    'retained_baseline_cases': len(baseline),
    'retained_status_changes': [],
    'new_cases_all_pass': new,
    'owner_governance_counts': dict(Counter(governance['candidate'].values())),
    'operation_counts': {label: report['operation_counts'] for label, report in reports.items()},
    'warm_median_s': {label: report['warm_median_s'] for label, report in reports.items()},
    'protected_body_sha256': namespace['frozen']['protected_body_sha256'],
    'reused_verifier': {'path': str(verifier), 'sha256': digest(verifier),
        'scope': 'Unchanged prefix through all wave/source/input checks; owner suite/archive phase is not executed.'},
    'parent_artifacts': {},
    'limits': ['The two replay arms ran concurrently with isolated caches while focused tests were active; timing is observational.',
        'Owner governance baseline/candidate evidence was checked without repeating that suite.',
        'Owner session labels remain in reused scripts; this receipt identifies the parent execution.',
        'No full CLI, strict77, full-unit or native/GPU qualification for this successor.'],
}
for relative in ('pytest.xml', 'pytest.log', 'prepare.log', 'baseline.log', 'candidate.log',
                 'wave/freeze.json', 'wave/run.py', 'wave/fixture_wiring.py',
                 'wave/baseline/wave_result.json', 'wave/candidate/wave_result.json'):
    path = RAW / relative
    receipt['parent_artifacts'][relative] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
output = Path(__file__).with_name('verification.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'parent_counts': receipt['parent_counts'],
    'retained_baseline_cases': len(baseline), 'new_cases': len(new),
    'byte_comparisons': len(namespace['comparisons']), 'warm_median_s': receipt['warm_median_s'],
    'receipt_sha256': digest(output)}, indent=2))
