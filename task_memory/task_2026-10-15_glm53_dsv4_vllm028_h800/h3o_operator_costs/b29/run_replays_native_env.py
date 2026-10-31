import concurrent.futures, json, os, resource, subprocess, sys, time
from pathlib import Path
s=Path(__file__).resolve().parent
r=Path('/data/ycfeng/Frontier/.worktrees/b29-ingress-delays')
a=r/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10'
resource.setrlimit(resource.RLIMIT_AS,(62914560*1024,62914560*1024))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',WANDB_DISABLED='true',VIDUR_DISABLE_WANDB='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(s/'tmp'),FRONTIER_TMP_ROOT=str(s/'tmp'),PYTHONPATH=f'{r}:{a}')
def run(pair):
 case,variant=pair
 root=s/f'{case}_{variant}_native_env'
 root.mkdir(exist_ok=True)
 source=Path(f'/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/{case}_cold_g1hs_r2/cache')
 cp=['timeout','120', 'cp','-a',str(source),str(root/'cache')]
 subprocess.run(cp,check=True,timeout=130)
 # Inventory only the private copied cache, never recurse over stage4.
 count=sum(p.is_file() for p in (root/'cache').rglob('*'))
 assert count==1758,(source,count)
 command=['timeout','3600',sys.executable,str(a/'a10_driver.py'),case,str(root),'--trace',str(s/'trace_first8.csv'),'--without-observer']
 if variant=='baseline':command.append('--without-ingress')
 (root/'command.json').write_text(json.dumps(dict(command=command,cwd=str(s),env={k:env[k] for k in ['PYTHONPATH','TMPDIR','FRONTIER_TMP_ROOT','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','WANDB_DISABLED','VIDUR_DISABLE_WANDB']},memory_limit_kib=62914560,cache_copy_command=cp,initial_cache_files=count),indent=2)+'\n')
 started=time.monotonic()
 with (root/'run.log').open('w') as log:
  result=subprocess.run(command,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3660)
 print(case,variant,'exit',result.returncode,'wall_s',time.monotonic()-started,flush=True)
 return dict(case=case,variant=variant,exit=result.returncode,wall_s=time.monotonic()-started)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
 results=list(executor.map(run,[(c,v) for c in ['p2_32','p2_64'] for v in ['baseline','delayed']]))
(s/'replay_status_native_env.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(row['exit']==0 for row in results),results
