"""Final read-only handoff checks and bounded verification-log search."""
import importlib.util, json, os, subprocess, sys
from pathlib import Path
ROOT=Path.cwd();T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800';OUT=Path(__file__).resolve().parent
G=T/'h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2'
D=Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads')
checks=[];notes={}
def load(p):return json.loads(Path(p).read_text())
def check(n,b,d=None):checks.append(dict(name=n,status='PASS' if b else 'FAIL',detail=d))
for case,base in [('p2_32',D/'dsv4-h200-p2-canonical-20261025-01/artifacts'),('generation08',D/'formal_p2_64_20261107_08_ssh/dsv4-h200-p2-64-formal-20261107-08/artifacts')]:
 nodes=list(base.glob('nodes/*/result.json'))
 for p in nodes:
  j=load(p);cmd=j['server_command']
  values={flag:cmd[cmd.index(flag)+1] for flag in ['--max-model-len','--max-num-batched-tokens','--pipeline-parallel-size','--gpu-memory-utilization','--load-format']}
  log=p.parent/'server.log';resolved=[(i,l) for i,l in enumerate(log.read_text(errors='replace').splitlines(),1) if 'max_seq_len=7168' in l]
  check(case+' actual bounds:'+p.parent.name,values['--max-model-len']=='7168' and values['--max-num-batched-tokens']=='8192' and values['--pipeline-parallel-size']=='1' and values['--gpu-memory-utilization']=='0.85' and values['--load-format']=='dummy' and bool(resolved),dict(values=values,server_log=str(log),resolved_lines=[i for i,l in resolved]))
 check(case+' node count',len(nodes)==(4 if case=='p2_32' else 8),len(nodes))
x=load(G/'result.json');check('seven stages and comparison complete',len(x['steps'])==7 and all(s['status']=='PASS' for s in x['steps']) and not x['errors'])
notes['historical_comparison_commands']=x['commands']
for i,c in enumerate(x['commands']):
 check('historical command completion:'+str(i),c['exit_code']==0 and not c['stderr'])
c=x['commands'][0];parsed=json.loads(c['stdout']);check('actual R5 stdout',parsed['status']=='PASS' and parsed['failed']==[],parsed)
notes['environment']={'executable':sys.executable,'python':sys.version,'packages':{n:importlib.util.find_spec(n) is not None for n in ['numpy','pandas','scipy','sklearn','pytest','fasteners','ddsketch','plotly']}}
notes['bounded_heartbeat_log_search']=[]
for base in [T/'h200_mainline/p2_64_formal_retry08',D.parent,Path('/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness')]:
 searched=[];matches=[]
 for directory,dirs,files in os.walk(base):
  depth=len(Path(directory).relative_to(base).parts)
  dirs[:]=[d for d in dirs if d not in {'downloads','download','traces','.git','task_memory','cache','.pi','remaining_measurements'}] if depth<2 else []
  for name in files:
   p=Path(directory)/name
   if p.suffix not in {'.log','.txt','.out'} or p.stat().st_size>2000000:continue
   searched.append(str(p))
   for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
    if any(n in line for n in ['101 passed','18 passed in 5.74','10.01 s','1.01 s']):matches.append(dict(path=str(p),line=i,text=line[:400]))
 notes['bounded_heartbeat_log_search'].append(dict(base=str(base),files_read=len(searched),paths=searched,matches=matches))
for command in [['git','rev-parse','HEAD'],['git','status','--short','--untracked-files=no'],['git','diff','abd1e14198b2b258921ccd669ac87c91f499e6dd..HEAD','--stat','--','frontier']]:
 p=subprocess.run(command,text=True,capture_output=True)
 notes.setdefault('git',[]).append(dict(command=command,rc=p.returncode,stdout=p.stdout,stderr=p.stderr))
 check('read-only '+' '.join(command),p.returncode==0 and (bool(p.stdout) if command[1]=='rev-parse' else not p.stdout))
notes['audit_output_counts']={}
for n in ['metrics_audit.json','evidence_audit.json','qualification_audit.json','runtime_supplement.json']:
 cs=load(OUT/n)['checks'];notes['audit_output_counts'][n]={s:sum(c['status']==s for c in cs) for s in ['PASS','FAIL']}
 check('retained review assertions:'+n,all(c['status']=='PASS' for c in cs),notes['audit_output_counts'][n])
(OUT/'handoff_audit.json').write_text(json.dumps(dict(checks=checks,notes=notes),indent=2)+'\n')
print('CHECKS',len(checks),'PASS',sum(c['status']=='PASS' for c in checks),'FAIL',sum(c['status']=='FAIL' for c in checks))
print('ENVIRONMENT',json.dumps(notes['environment']))
print('HEARTBEAT_SEARCH',json.dumps([{k:v for k,v in s.items() if k!='paths'} for s in notes['bounded_heartbeat_log_search']]))
print('GIT',json.dumps(notes['git']))
