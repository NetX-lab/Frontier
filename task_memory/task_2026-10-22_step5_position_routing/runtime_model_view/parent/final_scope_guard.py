"""Bind the integrated source to the independently audited named-query union."""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[4]
TASK = REPO / 'task_memory/task_2026-10-22_step5_position_routing'
AUDIT = TASK / 'repairs/evidence/named_query_scope'


def git(*args, cwd=REPO):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


manifest_path = AUDIT / 'artifact_sha256.json'
assert manifest_path.read_bytes() == git('show', f'de5eeab0:{manifest_path.relative_to(REPO)}')
manifest = json.loads(manifest_path.read_text())
for name, item in manifest['files'].items():
    raw = (AUDIT / name).read_bytes()
    assert len(raw) == item['bytes'] and sha(raw) == item['sha256'], name
audit = json.loads((AUDIT / 'named_query_scope_receipt.json').read_text())
assert not audit['violations']
head = git('rev-parse', 'HEAD').decode().strip()
source = audit['commits']['candidate_source']['commit']
assert not git('diff', '--name-only', source, head, '--', '.', ':(exclude)task_memory')
assert not git('diff', 'HEAD', '--name-only', '--', '.', ':(exclude)task_memory')
pin = audit['non_source']['submodule_pin']
module_path = 'frontier/cc_backend/backends/collective-sim'
module = REPO / module_path
assert git('rev-parse', 'HEAD', cwd=module).decode().strip() == pin
assert not git('diff', pin, '--name-only', cwd=module)
native = {item['path']: item for item in audit['non_source']['native_rows']}
bindings = {}
for row in audit['binding_guard']['rows']:
    name = row['binding']
    if name.startswith('native:'):
        item = native[name.removeprefix('native:')]
        assert sha(Path(item['base_path']).read_bytes()) == item['base_sha256']
        value = sha(Path(item['derived_path']).read_bytes())
        assert value == item['derived_sha256']
    elif name.startswith('submodule:'):
        relative = name.removeprefix('submodule:').removeprefix(module_path + '/')
        value = git('rev-parse', f'{pin}:{relative}', cwd=module).decode().strip()
        assert (module / relative).read_bytes() == git('show', f'{pin}:{relative}', cwd=module)
    else:
        value = git('rev-parse', f'{head}:{name}').decode().strip()
        if name != module_path:
            assert (REPO / name).read_bytes() == git('show', f'{head}:{name}')
    bindings[name] = value
owned = {item['path'] for item in audit['step']['paths']}
digest = lambda exclude: sha('\n'.join(f'{name} {value}' for name, value in sorted(bindings.items()) if name not in exclude).encode())
assert len(bindings) == 124 and len(bindings.keys() - owned) == 120
assert digest(set()) == audit['binding_guard']['candidate_all_digest']
assert digest(owned) == audit['binding_guard']['candidate_non_query_digest']
for item in audit['step']['paths']:
    path = item['path']
    added, deleted, _ = git('diff', '--numstat', audit['commits']['original']['commit'], head, '--', path).decode().strip().split('\t')
    assert int(added) + int(deleted) == item['cumulative']['gross'] <= item['cap']
parent = json.loads(Path(__file__).with_name('verification.json').read_text())
assert parent['status'] == 'PASS_COMPONENT_REPLAY_AND_FOCUSED_TESTS'
receipt = {'task_id': 'D29-NAMED-MODEL-QUERY-PARENT', 'status': 'ACCEPTED_COMPONENT_AND_COMPOSED_SCOPE',
    'integrated_head': head, 'integrated_tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
    'source_merge': 'ffb435ca', 'owner_source': source, 'owner_records_archive': '8ecb705e',
    'audit_source_commit': 'de5eeab05a2f312da415a9e6557847ab583b4990', 'audit_archive': 'c6b1f3b8',
    'audit_receipt_sha256': sha((AUDIT / 'named_query_scope_receipt.json').read_bytes()),
    'audited_files_verified': len(manifest['files']), 'binding_count': len(bindings),
    'non_query_binding_count': len(bindings.keys() - owned),
    'all_binding_digest': digest(set()), 'non_query_binding_digest': digest(owned),
    'step': audit['step'], 'totals': audit['totals'], 'allocated_paths': audit['allocated_paths'],
    'changed_paths': audit['changed_paths'], 'violations': [],
    'parent_tests_sha256': sha(Path(__file__).with_name('verification.json').read_bytes()),
    'reviews': {'standards': 'No findings', 'spec': 'No findings', 'record': 'review.md'},
    'limits': ['Source and all124 bindings match the audited component; native accounting is not a new native run.',
        'Owner small-read audit used30GiB RLIMIT_AS after scope denial; this parent guard runs under MemoryMax30G.',
        'Fresh frozen regressions, full6144/1024/2400s CLI and native/GPU/G3/E2E remain separate gates.']}
output = Path(__file__).with_name('acceptance.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'head': head, 'bindings': len(bindings),
    'non_query_digest': digest(owned), 'totals': audit['totals'], 'receipt_sha256': sha(output.read_bytes())}, indent=2))
