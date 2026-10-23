"""Seal task records and verify committed source plus the protected baseline."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CODE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key')
RECORD = CODE / 'task_memory/task_2026-10-22_step5_position_routing/plan_key_reuse'
freeze = json.loads((ROOT / 'freeze.json').read_text())
for source in freeze['sources'].values():
    for relative, expected in source['hashes'].items():
        assert hashlib.sha256((Path(source['root']) / relative).read_bytes()).hexdigest() == expected
baseline = Path(freeze['sources']['baseline']['root'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=baseline, text=True).strip() == freeze['sources']['baseline']['head']
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=baseline, text=True) == ''
subprocess.run(['git', 'diff', '--check'], cwd=CODE, check=True)
files = {str(path.relative_to(RECORD)): {'bytes': path.stat().st_size,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(RECORD.rglob('*')) if path.is_file() and path.name != 'manifest.json'}
manifest = {'task_id': 'D29-PLAN-KEY-REUSE', 'source_commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
    'protected_baseline_clean_and_unchanged': True, 'source_hashes_unchanged': True, 'files': files}
(RECORD / 'evidence/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'source_commit': manifest['source_commit'], 'manifest_files': len(files), 'status': 'PASS'}))
