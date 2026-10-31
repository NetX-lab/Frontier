import json, os, resource, subprocess, sys, time
from pathlib import Path
s=Path(__file__).resolve().parent
root=Path(sys.argv[1]); label=sys.argv[2]
cwd=s if label.startswith('regression_base') else s/f'{label}_cwd'
if cwd != s:
 cwd.mkdir(exist_ok=True)
 (cwd/'data').symlink_to(root/'data',target_is_directory=True)
work_tmp=s/'tmp' if cwd==s else cwd/'tmp'
work_tmp.mkdir(exist_ok=True)
targets=json.loads((s/'regression_targets_initial.json').read_text()) if len(sys.argv)==3 else sys.argv[3:]
resource.setrlimit(resource.RLIMIT_AS,(31457280*1024,31457280*1024))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',WANDB_DISABLED='true',VIDUR_DISABLE_WANDB='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_BS_SOURCE_ROOT='/data/ycfeng/tmp/opencode/vllm-v0.28.0',TMPDIR=str(work_tmp),FRONTIER_TMP_ROOT=str(work_tmp),PYTHONPATH=f'{root}:{root}/tests/unit:{root}/tests/comparison/calibration:{root}/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages')
command=['timeout','3600',sys.executable,'-m','pytest',*[str(root/t) for t in targets],'-q','--continue-on-collection-errors','-p','no:cacheprovider',f'--basetemp={s}/tmp/{label}',f'--junitxml={s}/{label}.xml']
(s/f'{label}_command.json').write_text(json.dumps(dict(command=command,cwd=str(cwd),env={k:env[k] for k in ['PYTHONPATH','TMPDIR','FRONTIER_TMP_ROOT','VLLM_BS_SOURCE_ROOT','PYTHONDONTWRITEBYTECODE','WANDB_DISABLED','VIDUR_DISABLE_WANDB','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},memory_limit_kib=31457280),indent=2)+'\n')
started=time.monotonic()
with (s/f'{label}.log').open('w') as log:
 result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3660)
print(label,'exit',result.returncode,'wall_s',time.monotonic()-started,flush=True)
print((s/f'{label}.log').read_text()[-6000:])
