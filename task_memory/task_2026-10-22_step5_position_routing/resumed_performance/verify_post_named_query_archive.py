"""Verify the merged regression audit without relaunching any suite."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AUDIT = HERE / 'post_named_query_final_parent'
OWNER = '03d6132e97d6bd9dc0b125795b6021cc0d43fec4'
SOURCE = 'dbe19f531ff68d9edafc480418e8927c40eb1c60'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


manifest = (AUDIT / 'artifact_sha256.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest() == 'e36241e8e6cb41872102ca60f1dc89db874ee75dda7ec31a86370cd16631d627'
hashes = {'artifact_sha256.json': hashlib.sha256(manifest).hexdigest()}
for name, binding in json.loads(manifest)['files'].items():
    raw = (AUDIT / name).read_bytes()
    assert len(raw) == binding['bytes']
    assert hashlib.sha256(raw).hexdigest() == binding['sha256'], name
    assert raw == git('show', f'{OWNER}:{(AUDIT / name).relative_to(REPO)}'), name
    hashes[name] = binding['sha256']
receipt = json.loads((AUDIT / 'parent_acceptance.json').read_bytes())
raw = Path(receipt['unit']['junit']['path']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['unit']['junit']['sha256']
statuses = {}
for case in ET.fromstring(raw).iter('testcase'):
    name = case.attrib['classname'] + '::' + case.attrib['name']
    assert name not in statuses
    statuses[name] = next((state for state in ('error', 'failure', 'skipped')
                          if case.find(state) is not None), 'pass')
assert dict(Counter(statuses.values())) == receipt['unit']['counts']
assert receipt['unit']['counts'] == {'pass': 8028, 'failure': 279, 'error': 32, 'skipped': 281}
assert receipt['source_revision'] == SOURCE and receipt['unit']['raw_exit'] == 1
assert receipt['raw_presence_gate_waived'] is False
transition = receipt['transitions']['46fe']
assert not transition['missing'] and not transition['changed_statuses']
assert len(transition['new_ids_all_pass']) == 8
assert all(statuses[name] == 'pass' for name in transition['new_ids_all_pass'])
assert not receipt['violations']
assert not git('diff', '--name-only', SOURCE, 'HEAD', '--', '.', ':(exclude)task_memory/**').strip()
result = {'task_id': 'D29-POST-NAMED-QUERY-ARCHIVE-ACCEPTANCE',
          'status': receipt['status'], 'source_revision': SOURCE,
          'archive_revision': git('rev-parse', 'HEAD').decode().strip(),
          'owner_records_revision': OWNER, 'auditor_artifacts_verified': hashes,
          'raw_unit_counts': receipt['unit']['counts'], 'raw_exit': 1,
          'approved_old_id_presence_gate_waived': False, 'new_cases_passed': 8,
          'source_changes_since_frozen_revision': [], 'scope_totals': receipt['scope_totals'],
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'limits': ['No suite or probe rerun.', 'Detailed fresh inventories and strict77 checks belong to the independent audit.',
                     'No whole CLI, package, native, GPU, G3 or E2E acceptance.']}
(AUDIT / 'archive_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('status', 'raw_unit_counts', 'raw_exit', 'new_cases_passed', 'scope_totals')}, indent=2))
