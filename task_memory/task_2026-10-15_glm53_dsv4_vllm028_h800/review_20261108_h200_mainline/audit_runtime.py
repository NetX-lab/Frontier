"""CPU-only joins of native observations and formal runtime evidence; stdlib only."""
import collections, csv, datetime, gzip, hashlib, json, math, statistics, subprocess, sys
from pathlib import Path
ROOT=Path.cwd();T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800';OUT=Path(__file__).resolve().parent
D=Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads');checks=[];summary={}
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(n,b,d=None):checks.append(dict(name=n,status='PASS' if b else 'FAIL',detail=d))
def dt(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def clocks(v):
 if isinstance(v,dict):
  if 'started_at' in v and 'finished_at' in v:return [(dt(v['finished_at'])-dt(v['started_at'])).total_seconds()]
  if 'startedAt' in v and 'finishedAt' in v:return [(dt(v['finishedAt'])-dt(v['startedAt'])).total_seconds()]
  return [s for x in v.values() for s in clocks(x)]
 if isinstance(v,list):return [s for x in v for s in clocks(x)]
 return []
cd=T/'h3o_operator_costs/formal_p2_20261102/gpu_hours_clocks';hrs=[]
for r in csv.DictReader((cd/'gpu_hours_complete.csv').open()):
 p=cd/r['clock_file'] if r['clock_file'] else Path(r['source']);v=load(p)
 if 'jobs' in v:
  v=next(j for j in v['jobs'] if r['job_id'] in json.dumps(j.get('replicas',[])))
 ss=clocks(v['replicas']);h=sum(ss)*(int(r['gpus'])/len(v['replicas']))/3600
 check('all35 clocks:'+r['job_id'],bool(ss) and math.isclose(h,float(r['gpu_hours']),abs_tol=1e-9),{'containers':len(ss),'gpu_hours':h});hrs.append(h)
summary['historical35_gpu_hours']=sum(hrs)
# Source delay values: endpoint joins, rather than trusting printed measurement means.
ref=D/'dsv4-h200-p2-64-canonical-20261025-01/artifacts';wave=D/'dsv4-h200-p2-64-wave-20261030-01/artifacts'
pc={r['request_id']:r for r in rows(ref/'run/proxy_requests.jsonl')}
a=[(pc[r['request_id']]['proxy_arrival_monotonic']-r['dispatch_monotonic'])*1000 for r in rows(ref/'run/client_requests.jsonl') if r['segment']=='formal' and (pc[r['request_id']]['decode_first_byte_monotonic']-pc[r['request_id']]['prefill_response_monotonic'])*1000<=1000]
check('client dispatch to proxy observed term',len(a)==379 and f'{statistics.mean(a):.1f}'=='0.6',{'count':len(a),'mean_ms':statistics.mean(a)})
pw={r['request_id']:r for r in rows(wave/'run/proxy_requests.jsonl')};pf={};dc={};enq={};updates={};api=[];api_enqueue=[]
for p in sorted(wave.glob('nodes/*/scheduler_lifecycle/*.jsonl')):
 rr=rows(p);start=next(r for r in rr if r['event']=='start');role=start['role'];pending={};enqueue={}
 for r in rr:
  rid=r.get('request_id','')
  if r['event']=='add_enqueue' and 'formal' in rid:enqueue[rid]=r['t']
  if r['event']=='add' and 'formal' in rid:
   key=rid.split('cmpl-',1)[1].rsplit('-0-',1)[0]
   if role=='prefill':
    pf[rid]=key;api.append((r['arrival_time']-pw[key]['proxy_arrival_time'])*1000);api_enqueue.append((enqueue[rid]-r['arrival_time'])*1000)
   else:dc[rid]=(key,r['remote_request_id']);enq[rid]=enqueue[rid]
  if role=='prefill' and r['event']=='schedule':
   for new in r.get('new',[]) or []:
    if 'formal' in new[0]:pending[r['step']]=new[0]
  if role=='prefill' and r['event']=='update' and r.get('step') in pending:updates[pending.pop(r['step'])]=r['t']
b=[(pw[key]['prefill_response_time']-updates[rid])*1000 for rid,key in pf.items() if rid in updates]
c=[(enq[rid]-pw[key]['prefill_response_time'])*1000 for rid,(key,remote) in dc.items() if remote in updates]
terms={'proxy_to_api':(api,'3.11',2),'api_to_enqueue':(api_enqueue,'6.59',2),'prefill_update_to_response':(b,'2.3',1),'response_to_decode_enqueue':(c,'9.1',1)}
summary['source_delay_means_ms']={'client_dispatch_to_proxy':statistics.mean(a)}
for name,(v,expected,n) in terms.items():
 value=statistics.mean(v);check('delay endpoints:'+name,len(v)==128 and format(value,f'.{n}f')==expected,{'count':len(v),'mean_ms':value});summary['source_delay_means_ms'][name]=value
# P2-32 native raw completions and recorded topology.
base=D/'dsv4-h200-p2-canonical-20261025-01/artifacts';rr=rows(base/'run/client_requests.jsonl');check('P2-32 native432 unique',len(rr)==432 and len({r['request_id'] for r in rr})==432)
summary['p2_32_native']={}
for i in (1,2,3):
 for seg,n in [('formal',128),('warmup',16)]:
  pop=[r for r in rr if r['repeat']==i and r['segment']==seg]
  check(f'P2-32 native population:{i}/{seg}',len(pop)==n and all(r['status']=='completed' and r['stream_done'] and r['http_status']==200 and r['finish_reason']=='length' and not r['validation_errors'] and not r.get('error') and r['num_prefill_tokens']==6144 and r['num_decode_tokens']==1024 and r['output_token_count']==1024 for r in pop))
formal=T/'h3o_operator_costs/formal_p2_20261102/readings/p2_32_warm_reading.json';v=load(formal)
for src in v['sources']:
 p=Path(src['path'])
 if p.parent.name!='vllm':continue
 i=int(p.stem.rsplit('repeat',1)[1]);csvrows=list(csv.DictReader(p.open()));byid={r['request_id']:r for r in csvrows};res=[]
 for r in rr:
  if r['repeat']!=i or r['segment']!='formal':continue
  q=byid[r['template_request_id']];vals={'arrival_s':r['dispatch_monotonic'],'completion_s':r['last_token_monotonic'],'ttft_ms':1000*(r['first_token_monotonic']-r['dispatch_monotonic']),'tpot_ms':1000*(r['last_token_monotonic']-r['first_token_monotonic'])/1023,'request_e2e_time_ms':1000*(r['last_token_monotonic']-r['dispatch_monotonic'])}
  res.extend(abs(float(q[k])-value) for k,value in vals.items())
 check('P2-32 raw native to row CSV:'+str(i),max(res)<1e-6,{'max_residual':max(res)})
workers=list(base.glob('nodes/*/serving/*.json'));ranks=collections.defaultdict(set);summary['p2_32_native']['workers']=len(workers)
for p in workers:
 j=load(p);inv=j['inventory'];g=inv['groups'];check('P2-32 worker:'+str(p.relative_to(base)),not j['observation_errors'] and g['tp']['world_size']==8 and g['dp']['world_size']==2 and g['ep']['world_size']==16 and 'H200' in inv['device_identity']['name']);ranks[j['role']].add(g['world']['rank'])
check('P2-32 topology census',len(workers)==32 and all(len(r)==16 for r in ranks.values()) and len(ranks)==2)
summary['p2_32_native']['ranks_by_role']={k:len(v) for k,v in ranks.items()}
# Deferred memory qualification: byte identity is checked, not extrapolated to pressure accuracy.
ret=load(T/'h3o_operator_costs/input_qualification_20261101/evidence/retained_checks.json')
for case in ret['measured_capacity_f4']:
 for m in case['metrics']:check('F4 actual artifact hash:'+case['case']+'/'+m['name'],all(Path(p).is_file() and sha(p)==m['sha256'] for p in m['paths']))
summary['formal_forward_ep']={}
for case in ('p2_32','p2_64'):
 p=Path('/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs')/case/'forwards.jsonl.gz';cs=collections.Counter();affected=[]
 for line in gzip.open(p,'rt'):
  r=json.loads(line);equal=len(set(r['num_tokens_across_dp']))==1;cs[(r['role'],r['mode'],equal)]+=1
  if not equal:affected.append(r)
 check('formal unequal EP scope:'+case,all(r['role']=='prefill' and r['mode']=='NONE' for r in affected),dict((str(k),v) for k,v in cs.items()))
 summary['formal_forward_ep'][case]={str(k):v for k,v in cs.items()}
# p50/p95 are retained diagnostics (not gates).
q=T/'h3o_operator_costs/formal_p2_20261102/assessment/diagnostics.csv';ds=list(csv.DictReader(q.open()));check('recorded percentile diagnostics',len(ds)==144);summary['percentile_diagnostics']={'path':str(q),'rows':len(ds),'fields':list(ds[0])}
# Read-only runtime environment and module inventory.
summary['modules_over_2000']={str(p):len(p.read_text().splitlines()) for p in (ROOT/'frontier').rglob('*.py') if 'collective-sim' not in str(p) and len(p.read_text().splitlines())>2000}
check('documented oversized module inventory',set(summary['modules_over_2000'])=={str(ROOT/p) for p in ['frontier/execution_time_predictor/sklearn_execution_time_predictor.py','frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py','frontier/metrics/metrics_store.py','frontier/profiling/attention/main.py']})
(OUT/'runtime_audit.json').write_text(json.dumps({'interpreter':sys.version,'checks':checks,'summaries':summary},indent=2)+'\n')
print('CHECKS',len(checks),'PASS',sum(c['status']=='PASS' for c in checks),'FAIL',sum(c['status']=='FAIL' for c in checks));print(json.dumps(summary,indent=2));print('FAILURES',json.dumps([c for c in checks if c['status']=='FAIL'],indent=2))
