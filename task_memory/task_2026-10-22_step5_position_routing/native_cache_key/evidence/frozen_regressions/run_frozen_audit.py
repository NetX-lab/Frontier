"""Run the retained auditor in the exact cache namespace of each completed run."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key')
old_repo = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08')
view = ROOT/'frozen-matrix/generated_repo_root'
baseline_view = Path('/data/ycfeng/tmp/d29_post_named_query_regressions_20261024/strict77/generated_repo_root')
destination = ROOT/'frozen-audit-03'
destination.mkdir(exist_ok=False)
raw = (ROOT/'frozen-matrix/comparison.json').read_bytes()
(destination/'comparison.json').write_bytes(raw)
script = (ROOT/'audit_frozen_matrix.py').read_text()
old = "audit.ROOT = ROOT/'frozen-matrix'"
assert script.count(old) == 1
script = script.replace(old,"audit.ROOT = ROOT/'frozen-audit-03'")
script = script.replace("audit.CANDIDATE = audit.ROOT/'candidate'", "audit.CANDIDATE = ROOT/'frozen-matrix/candidate'")
(ROOT/'audit_frozen_matrix03.py').write_text(script)
retained = runpy.run_path('/data/ycfeng/tmp/d29_post_named_query_regressions_20261024/run_suite.py')
namespace = retained['namespace']
namespace.__globals__['retained_namespace'].__globals__['REPO'] = REPO
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO),TMPDIR='/data/ycfeng/tmp',
           OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
command = namespace(view,env,readonly=True)
assert command[-1] == '--'
command[-1:-1] = ['--ro-bind',str(baseline_view),str(old_repo)]
command += ['timeout','180s','/data/ycfeng/envs/frontier-py310/bin/python','-B',str(ROOT/'audit_frozen_matrix03.py')]
bindings = dict(command=command,comparison_sha256=hashlib.sha256(raw).hexdigest(),
                candidate_cache_namespace=dict(source=str(view/'cache'),target=str(REPO/'cache')),
                baseline_cache_namespace=dict(source=str(baseline_view/'cache'),target=str(old_repo/'cache')),
                changed_predicates=False,previous_failed_receipt_retained=str(ROOT/'frozen-matrix/postrun_receipt.json'))
(destination/'launch.json').write_text(json.dumps(bindings,indent=2)+'\n')
with (destination/'audit.log').open('x') as log:
    result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=190)
print(json.dumps(dict(returncode=result.returncode,destination=str(destination)),indent=2))
raise SystemExit(result.returncode)
