"""Run the retained auditor in the exact generated cache namespace."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'strict77_audit'
DEST.mkdir(exist_ok=False)
comparison = (ROOT/'strict77_output/comparison.json').read_bytes()
(DEST/'comparison.json').write_bytes(comparison)
runner = runpy.run_path(str(ROOT/'run_suite.py'))
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runner['REPO']), TMPDIR='/data/ycfeng/tmp',
           FRONTIER_TMP_ROOT='/data/ycfeng/tmp', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
command = runner['namespace'](ROOT/'strict77/generated_repo_root', env, readonly=True)
command += ['timeout', '60s', runner['PY310'], '-B', str(ROOT/'audit_final_matrix.py')]
before = runner['state']()
receipt = dict(command=command, source_before=before, configured_memory_max_bytes=30*1024**3,
               measured_cgroup_rss_peak=None, predicates_changed=False,
               comparison_sha256=hashlib.sha256(comparison).hexdigest(),
               candidate_cache_namespace=dict(source=str(ROOT/'strict77/generated_repo_root/cache'),
                                              target=str(runner['REPO']/'cache')),
               baseline_cache_namespace=dict(source='/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-matrix/generated_repo_root/cache',
                                             target='/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key/cache'),
               start_utc=datetime.now(timezone.utc).isoformat())
(DEST/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
started = time.monotonic()
with (DEST/'audit.log').open('x') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
receipt.update(returncode=result.returncode, duration_seconds=time.monotonic()-started,
               end_utc=datetime.now(timezone.utc).isoformat(), source_after=runner['state']())
(DEST/'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(returncode=result.returncode, destination=str(DEST), duration_seconds=receipt['duration_seconds']), indent=2))
raise SystemExit(result.returncode)
