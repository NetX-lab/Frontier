"""Audit the persisted generated cache in its original source namespace."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05')
view = ROOT/'strict77/generated_repo_root'
command = ['bwrap','--die-with-parent','--ro-bind','/','/','--dev-bind','/dev','/dev',
           '--bind',str(ROOT),str(ROOT),'--ro-bind',str(view),str(REPO)]
for entry in sorted(REPO.iterdir()):
    command.extend(['--ro-bind',str(entry),str(entry)])
command += ['--chdir',str(REPO),'--','/data/ycfeng/envs/frontier-py310/bin/python','-B',str(ROOT/'audit_strict77.py')]
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',TMPDIR='/data/ycfeng/tmp',
           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (ROOT/'strict77_final04/audit.log').open('x') as stream:
    result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
print('strict77 retained audit exit',result.returncode)
raise SystemExit(result.returncode)
