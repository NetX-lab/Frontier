"""Keep the approved probe launcher alive and retain its terminal receipt."""
import datetime, json, os, subprocess, time
from pathlib import Path
repo=Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
rel=Path('task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101')
here=repo/rel
package=Path('/data/ycfeng/Frontier/.worktrees/g1h-supplement-build')/rel/'package_01'
receipt=here/'launch_command.json'
if receipt.exists():raise RuntimeError('Submission receipt exists; inspect the existing job instead of resubmitting')
cmd=['bash',str(package/'launch/launch_glm_producers.sh'),str(package/'jobs/A6_REPEATABILITY.json')]
log=Path('/data/ycfeng/tmp/glm53_dsv4/a6_repeatability_launcher_20261101.log')
r={'command':cmd,'cwd':str(repo),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'log':str(log),'exit_code':None,'status':'LAUNCHING'}
receipt.write_text(json.dumps(r,indent=2)+'\n')
clock=time.monotonic()
with log.open('x') as handle:
 p=subprocess.Popen(cmd,cwd=repo,stdout=handle,stderr=subprocess.STDOUT)
 r['launcher_pid']=p.pid;receipt.write_text(json.dumps(r,indent=2)+'\n')
 print('LAUNCHER_STARTED',p.pid,flush=True)
 code=p.wait()
r.update(exit_code=code,status='TERMINAL',elapsed_seconds=time.monotonic()-clock,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)
raise SystemExit(code)
