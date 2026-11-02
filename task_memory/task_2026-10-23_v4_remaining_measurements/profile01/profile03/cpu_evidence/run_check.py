"""Record an explicitly CPU-only command without inheriting platform credentials."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
name, python, script = sys.argv[1:]
env = dict(PATH='/usr/bin:/bin', HOME='/home/brainpp', LANG='C.UTF-8',
           TMPDIR=str(root), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
command = [python, '-I', '-B', str(root / script)]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=60)
(root / (name + '.stdout.log')).write_text(result.stdout)
(root / (name + '.stderr.log')).write_text(result.stderr)
receipt = dict(command=command, cwd=str(root), environment=env, started_utc=started,
               returncode=result.returncode, script_sha256=hashlib.sha256((root / script).read_bytes()).hexdigest(),
               python_driver=sys.version, cpu_only=True, submitted=False)
(root / (name + '.command.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
print(result.stdout, end='')
