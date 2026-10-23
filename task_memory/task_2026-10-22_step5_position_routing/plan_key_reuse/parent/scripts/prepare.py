"""Bind the unchanged accepted wave driver to fresh parent output directories."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OWNER = Path('/data/ycfeng/tmp/d29_plan_key_reuse/wave01')
CANDIDATE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key')
RECORDS = CANDIDATE / 'task_memory/task_2026-10-22_step5_position_routing/plan_key_reuse'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((RECORDS / 'evidence/manifest.json').read_text())
for relative, expected in manifest['files'].items():
    path = RECORDS / relative
    assert path.stat().st_size == expected['bytes'], relative
    assert sha(path) == expected['sha256'], relative

frozen = json.loads((OWNER / 'freeze.json').read_text())
assert sha(OWNER / 'freeze.json') == manifest['files']['evidence/freeze.json']['sha256']
for label, source in frozen['sources'].items():
    path = Path(source['root'])
    for relative, expected in source['hashes'].items():
        assert sha(path / relative) == expected, (label, relative)
    source['head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    source['status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True)
    assert source['status'] == '', (label, source['status'])
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=path / 'frontier/cc_backend/backends/collective-sim', text=True).strip()
    assert pin == source['submodule_pin']
assert frozen['sources']['candidate']['head'] == 'd1e2fb8c0162c58e7e694fab2d3a900638f0d50a'
assert frozen['sources']['baseline']['head'] == '6c7121d819e679054285bb3be5cb117a7e77bc4e'
for name, expected in [('original_setup', 'setup_sha256'), ('protected_body', 'protected_body_sha256')]:
    assert sha(Path(frozen[name])) == frozen[expected]
for path, expected in frozen['input_hashes'].items():
    assert sha(Path(path)) == expected, path
fixture = Path(frozen['fixture'])
metadata = {str(path.relative_to(fixture)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in fixture.rglob('*') if path.is_file()}
assert metadata == frozen['fixture_metadata']
raw = Path(frozen['collective_original']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == frozen['collective_sha256']
for label in frozen['sources']:
    target = ROOT / label / 'collective-fixture/h800/collective_local.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
for name in ('run.py', 'fixture_wiring.py'):
    original = RECORDS / 'evidence/scripts' / name
    assert sha(original) == sha(OWNER / name)
    (ROOT / name).write_bytes(original.read_bytes())
frozen['parent_verification'] = {
    'task_id': 'D29-PLAN-KEY-PARENT',
    'owner_manifest_sha256': sha(RECORDS / 'evidence/manifest.json'),
    'owner_files_verified': len(manifest['files']),
    'driver_and_wiring_unchanged': True,
    'source_heads_refreshed_without_source_changes': True,
}
(ROOT / 'cache').mkdir()
(ROOT / 'freeze.json').write_text(json.dumps(frozen, indent=2) + '\n')
print(json.dumps(frozen['parent_verification'], indent=2))
