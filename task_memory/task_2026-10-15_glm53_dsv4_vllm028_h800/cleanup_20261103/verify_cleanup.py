"""Verify the task cleanup against the retained pre-cleanup inputs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

TASK = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800')
OUT = TASK / 'cleanup_20261103'
before = json.loads((OUT / 'protected_before.json').read_text())
manifest = json.loads((OUT / 'deletion_manifest.json').read_text())

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

missing = []
changed = []
for path, digest in before['files'].items():
    if not Path(path).is_file():
        missing.append(path)
    elif sha(path) != digest:
        changed.append(path)
paths = [row['path'] for row in manifest['entries']]
still_present = [path for path in paths if os.path.lexists(path)]
roots_missing = [path for path in before['protected_paths'] if not Path(path).exists()]
worktrees = subprocess.run(['git', 'worktree', 'list', '--porcelain'], cwd=TASK, text=True, capture_output=True, check=True).stdout
current = {line.split(' ', 1)[1] for line in worktrees.splitlines() if line.startswith('worktree ')}
worktrees_removed = sorted(set(before['worktree_paths']) - current)
proc = Path('/proc/81539')
launcher = {'pid': 81539, 'alive': proc.exists()}
if proc.exists():
    launcher.update(cmdline=(proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(), cwd=os.readlink(proc / 'cwd'))
    launcher['identity_unchanged'] = launcher['cmdline'] == before['launcher_cmdline'] and launcher['cwd'] == before['launcher_cwd']
else:
    launcher['identity_unchanged'] = False
raw_df = subprocess.run(['df', '-B1', '/data'], text=True, capture_output=True, check=True).stdout
fields = raw_df.splitlines()[-1].split()
disk = {'filesystem': fields[0], 'size_bytes': int(fields[1]), 'used_bytes': int(fields[2]), 'available_bytes': int(fields[3]), 'use_percent': fields[4], 'raw_df': raw_df}
disk_before = json.loads((OUT / 'disk_before.json').read_text())['initial_delete_observation']
disk['available_increase_bytes'] = disk['available_bytes'] - disk_before['available_bytes']
result = {'utc': datetime.now(timezone.utc).isoformat(), 'protected_file_count': len(before['files']), 'protected_missing': missing, 'protected_changed': changed, 'protected_roots_missing': roots_missing, 'worktrees_removed': worktrees_removed, 'deleted_path_count': len(paths), 'still_present': still_present, 'launcher': launcher, 'disk_before': disk_before, 'disk_after': disk}
result['status'] = 'PASS' if not missing + changed + roots_missing + worktrees_removed + still_present and launcher['identity_unchanged'] else 'FAIL_REVIEW'
(OUT / 'lead_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
