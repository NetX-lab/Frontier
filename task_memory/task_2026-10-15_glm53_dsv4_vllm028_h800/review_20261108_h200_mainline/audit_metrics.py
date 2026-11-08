"""Independent stdlib-only audit. Reads evidence, writes only beside this script."""
import csv, json, math, hashlib, statistics, sys
from pathlib import Path
ROOT=Path.cwd(); T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'; OUT=Path(__file__).resolve().parent
checks=[]
def check(name, ok, detail):
 checks.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=detail))
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def walk(j):
 if isinstance(j,dict):
  if isinstance(j.get('path'),str) and isinstance(j.get('sha256'),str):yield j
  for v in j.values():yield from walk(v)
 elif isinstance(j,list):
  for v in j:yield from walk(v)
def hashrefs(j,label):
 out=[]
 for r in walk(j):
  p=Path(r['path'])
  if not p.is_absolute():p=ROOT/p
  if str(p).startswith('/mnt/'):out.append(dict(path=str(p),status='NOT_RUN_CLOUD'));continue
  status='MISSING' if not p.is_file() else ('MATCH' if sha(p)==r['sha256'] else 'MISMATCH')
  out.append(dict(path=str(p),status=status))
 check(label,all(r['status']=='MATCH' for r in out),out)
 return out
fields=['ttft_ms','tpot_ms','request_e2e_time_ms','token_throughput_tps']
def metrics(rows):
 lat={k:statistics.fmean(float(r[k]) for r in rows if k!='tpot_ms' or int(r['request_num_decode_tokens'])>1) for k in fields[:3]}
 span=max(float(r['completion_s']) for r in rows)-min(float(r['arrival_s']) for r in rows)
 tokens=sum(int(r['request_num_prefill_tokens'])+int(r['request_num_decode_tokens']) for r in rows)
 return dict(**lat,token_throughput_tps=tokens/span,span_s=span,tokens=tokens,count=len(rows))
results={}
for case,wp in [('p2_32',T/'h3o_operator_costs/formal_p2_20261102/readings/p2_32_warm_reading.json'),('p2_64_old',T/'h3o_operator_costs/formal_p2_20261102/readings/p2_64_warm_reading.json'),('p2_64_generation08',T/'h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2/warm_reading.json')]:
 warm=load(wp); refs=hashrefs(warm['sources'],case+'_bound_sources'); sources=[Path(r['path']) for r in warm['sources'] if r['path'].endswith('.csv')]
 if not all(p.is_file() for p in sources):continue
 rows={('vllm/' if p.parent.name=='vllm' else 'frontier/')+p.stem:list(csv.DictReader(p.open())) for p in sources}
 all_m={s:metrics(rr) for s,rr in rows.items()}
 for s,rr in rows.items():
  ids=[r['request_id'] for r in rr]
  check(case+'/'+s+'/population',len(rr)==128 and len(set(ids))==128 and set(ids)=={f'fixed-6144-1024-req128-qps1-formal-r{i:03d}' for i in range(128)} and all(int(r['request_num_prefill_tokens'])==6144 and int(r['request_num_decode_tokens'])==1024 for r in rr),{'count':len(rr),'unique':len(set(ids))})
  residual=max(abs((float(r['completion_s'])-float(r['arrival_s']))*1000-float(r['request_e2e_time_ms'])) for r in rr)
  tpot_res=max(abs((float(r['request_e2e_time_ms'])-float(r['ttft_ms']))/1023-float(r['tpot_ms'])) for r in rr)
  check(case+'/'+s+'/interval_identity',residual<1e-3 and tpot_res<1e-3,{'e2e_max_residual_ms':residual,'tpot_max_residual_ms':tpot_res})
  if s.startswith('frontier/'):
   check(case+'/frontier_fixed_arrivals',[float(r['arrival_s']) for r in sorted(rr,key=lambda r:r['request_id'])]==list(range(128)),{})
 f=next(s for s in rows if s.startswith('frontier/')); natives=[s for s in rows if s.startswith('vllm/')]
 readings={}
 for reading in ['all_request','warm']:
  mm={}; worst=(0,'','')
  for s in natives:
   excluded=warm['openers'][s] if reading=='warm' else []
   pair={name:metrics([r for r in rows[name] if r['request_id'] not in excluded]) for name in [s,f]}; mm[s]=pair
   for claim in warm[reading]['repetitions'][s]['metrics']:
    k=claim['metric']; n=pair[s][k]; v=pair[f][k]; e=abs(v-n)/abs(n) if n else None
    check(case+'/'+reading+'/'+s+'/'+k, e is not None and math.isclose(n,claim['vllm'],abs_tol=1e-8) and math.isclose(v,claim['frontier'],abs_tol=1e-8) and math.isclose(e,claim['relative_error'],abs_tol=1e-10),dict(native=n,frontier=v,error=e,within_10pct=e is not None and e<=.1))
    if e>worst[0]:worst=(e,s,k)
  pooled={}
  for name in ['vllm','frontier']:
   vals=[pair[s if name=='vllm' else f] for s,pair in mm.items()]
   pooled[name]={k:(sum(v[k]*v['count'] for v in vals)/sum(v['count'] for v in vals)) for k in fields[:3]}
   pooled[name]['token_throughput_tps']=sum(v['tokens'] for v in vals)/sum(v['span_s'] for v in vals)
  for claim in warm[reading]['metrics']:
   k=claim['metric'];check(case+'/'+reading+'/pooled/'+k,math.isclose(pooled['vllm'][k],claim['vllm'],abs_tol=1e-8) and math.isclose(pooled['frontier'][k],claim['frontier'],abs_tol=1e-8),{n:pooled[n][k] for n in pooled})
  readings[reading]=dict(pairs=mm,pooled=pooled,worst=worst)
 results[case]=readings
 check(case+'/unchanged_all_request',warm['all_request']==json.loads(warm['all_request_verbatim']),{'verdict':warm['all_request']['verdict'],'exclusions':warm['openers']})
# Direct native streamed endpoints from the retained gate input, independent of normalizer.
G=T/'h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2'
run=G/'gate/p2_64_r4_case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-08/run'
raw=[json.loads(x) for x in (run/'client_requests.jsonl').read_text().splitlines()]; summary=load(run/'replay_summary.json'); reps=summary['canonical_repetitions']; pop={}
for i in [1,2,3]:
 for phase,n in [('warmup',16),('formal',128)]:
  rr=[r for r in raw if r['repeat']==i and r['segment']==phase];pop[f'{i}/{phase}']=len(rr)
  check(f'native08/{i}/{phase}/raw_population',len(rr)==n and all(r['status']=='completed' and r['http_status']==200 and r['stream_done'] and r['finish_reason']=='length' and not r.get('error') and not r['validation_errors'] and r['num_prefill_tokens']==6144 and r['num_decode_tokens']==1024 and r['output_token_count']==1024 for r in rr),{'count':len(rr)})
 rr=sorted([r for r in raw if r['repeat']==i and r['segment']=='formal'],key=lambda r:r['template_request_id'])
 check(f'native08/{i}/scheduled_arrivals',[r['arrived_at'] for r in rr]==list(range(128)),{})
 jitter=[(r['dispatch_monotonic']-rr[0]['dispatch_monotonic'])-r['arrived_at'] for r in rr]
 csvrows=list(csv.DictReader((G/f'gate/rows/vllm/dsv4-h200-p2-64-formal-20261107-08-repeat{i:02d}.csv').open())); byid={r['request_id']:r for r in csvrows}
 diffs=[]
 for r in rr:
  q=byid[r['template_request_id']]; vals=dict(arrival_s=r['dispatch_monotonic'],completion_s=r['last_token_monotonic'],ttft_ms=1000*(r['first_token_monotonic']-r['dispatch_monotonic']),tpot_ms=1000*(r['last_token_monotonic']-r['first_token_monotonic'])/1023,request_e2e_time_ms=1000*(r['last_token_monotonic']-r['dispatch_monotonic']))
  diffs.extend(abs(float(q[k])-v) for k,v in vals.items())
 check(f'native08/{i}/raw_to_csv',max(diffs)<1e-6,{'max_residual':max(diffs),'relative_dispatch_jitter_s_min':min(jitter),'relative_dispatch_jitter_s_max':max(jitter)})
check('native08/432_unique',len(raw)==432 and len({r['request_id'] for r in raw})==432,pop)
check('native08/idle_boundaries',len(reps['idle_boundaries'])==7 and all(b['verified'] for b in reps['idle_boundaries']),reps['idle_boundaries'])
# Promotion errors, hashes, and prior committed existence.
receipt=load(T/'h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json');hashrefs(receipt,'promotion_hashes');errs=[]
for s,r in receipt['binding']['repetitions'].items():
 for m in r['metrics']:
  e=abs(m['frontier']-m['native'])/abs(m['native']) if m['native'] else None;errs.append(e)
  check('promotion/'+s+'/'+m['metric'],e is not None and math.isfinite(e) and e<=.3 and math.isclose(e,m['relative_error'],abs_tol=1e-12),{'recomputed':e})
check('promotion_max',len(errs)==12 and math.isclose(max(errs),receipt['predicate']['maximum_relative_error'],abs_tol=1e-12),{'unique_errors':len(errs),'recorded_errors_including_all_and_warm':receipt['predicate']['checked_errors'],'max':max(errs),'issued':receipt['issued_utc']})
output=dict(interpreter=sys.version,results=results,checks=checks)
(OUT/'metrics_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print('CHECKS',len(checks),'PASS',sum(c['status']=='PASS' for c in checks),'FAIL',sum(c['status']=='FAIL' for c in checks))
for case,res in results.items():
 print(case,{r:{'worst':v['worst'],'pooled':v['pooled']} for r,v in res.items()})
for c in checks:
 if c['status']!='PASS':print('FAIL',c['check'],json.dumps(c['detail'])[:4000])
