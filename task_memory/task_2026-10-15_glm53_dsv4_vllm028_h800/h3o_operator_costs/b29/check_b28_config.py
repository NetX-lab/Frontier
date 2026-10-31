import json,os,resource,subprocess,sys
from pathlib import Path
s=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_AS,(31457280*1024,31457280*1024))
roots={'base':Path('/data/ycfeng/Frontier/.worktrees/b29-base'),'candidate':Path('/data/ycfeng/Frontier/.worktrees/b29-ingress-delays')}
new={'request_ingress_delay_ms','analytical_kv_cache_transfer_config_request_handoff_delay_ms'}
results={}
for case in ['p2_32','p2_64']:
 snapshots={}
 for side,r in roots.items():
  directory=s/f'b28_config_{case}_{side}'
  out=s/f'b28_snapshot_{case}_{side}.json'
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  command=['timeout','60',sys.executable,str(s/'config_snapshot.py'),str(r),case,str(directory),str(out)]
  result=subprocess.run(command,cwd=s,env=env,capture_output=True,text=True,timeout=70)
  assert result.returncode==0,result.stdout+result.stderr
  snapshot=json.loads(out.read_text().replace(str(directory),'<RUN_ROOT>'))
  snapshots[side]=snapshot
 for name in new:
  assert snapshots['candidate']['flat'].pop(name)==0.0
  assert snapshots['candidate']['changes']['added_fields'].pop(name)==0.0
 assert snapshots['candidate']==snapshots['base'],(case,'config or override diff changed')
 results[case]={'status':'PASS','flat_fields_compared':len(snapshots['base']['flat']),'new_fields_at_zero':sorted(new),'normalization':'each run root -> <RUN_ROOT>; same scratch short trace for both sides'}
(s/'b28_config_check.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
