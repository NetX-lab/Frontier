"""Read-only checks of sealed provenance, native inventories and actual replay ledgers."""
import collections, csv, datetime, gzip, hashlib, json, math, subprocess, sys
from pathlib import Path
ROOT=Path.cwd(); T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'; OUT=T/'review_20261108_h200_mainline'
P=T/'h200_mainline/p2_64_formal_retry08/package_01'
D=Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh/dsv4-h200-p2-64-formal-20261107-08')
G=T/'h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2'
checks=[]; summaries={}
def load(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def check(name,ok,detail=None):
 checks.append({'name':name,'status':'PASS' if ok else 'FAIL','detail':detail})
 if not ok: print('FAIL',name,detail,flush=True)
def bind(p,expected,name=None): check(name or str(p),p.is_file() and sha(p)==expected,{'path':str(p),'expected':expected,'observed':sha(p) if p.is_file() else 'MISSING'})
inv=load(D/'cloud_inventory.json'); files=inv['files']
for f in files:
 p=D/'artifacts'/f['path'];bind(p,f['sha256']);check('size:'+f['path'],p.stat().st_size==f['bytes'])
check('inventory population',len(files)==130 and sum(f['bytes'] for f in files)==75585300)
check('inventory no extra files',{str(p.relative_to(D/'artifacts')) for p in (D/'artifacts').rglob('*') if p.is_file()}=={f['path'] for f in files})
bind(D/'cloud_inventory.json','14dc2ae8b5f5f4a0fcdfc28a3a5591471b579d6b06d8cbfde8e998baa66c1ddb')
for name,h in load(P/'manifest.json')['files'].items():bind(P/name,h,'sealed:'+name)
bind(P/'manifest.json','1a2bbc13339b7c741cb8151611864c7e7815d8bb180588ffff2700ae888d3508')
lock=load(P/'source_lock.json')
for name,v in lock['committed_files'].items():
 cwd=lock['native_checkout'] if v['commit']==lock['native_commit'] else str(ROOT)
 proc=subprocess.run(['git','show',v['commit']+':'+v['path']],cwd=cwd,capture_output=True)
 check('git blob:'+name,proc.returncode==0 and hashlib.sha256(proc.stdout).hexdigest()==v['sha256'],{'repo':cwd,'commit':v['commit'],'path':v['path'],'exit':proc.returncode})
proc=subprocess.run(['git','cat-file','-t',lock['native_commit']],capture_output=True)
check('native SHA distinct from local commit',proc.returncode!=0)
terminal=load(P/'case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-08/platform_terminal_receipt.json')
secs=[]
for r in terminal['replicas']:
 c=r['containers'][0];s=c['state']['terminated']; delta=(datetime.datetime.fromisoformat(s['finishedAt'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(s['startedAt'].replace('Z','+00:00'))).total_seconds();secs.append(delta)
 check('terminal:'+r['name'],r['phase']=='Succeeded' and s['exitCode']==0 and c['restartCount']==0 and delta==c['duration_seconds'])
check('terminal sum',sum(secs)==10488 and math.isclose(sum(secs)*8/3600,terminal['container_gpu_hours']))
summaries['terminal']={'seconds':secs,'total_seconds':sum(secs),'container_gpu_hours':sum(secs)*8/3600}
ranks=list((D/'artifacts').glob('nodes/*/serving/rank*.json')); summaries['native']={'ranks':len(ranks)}
census=collections.defaultdict(collections.Counter);owners=collections.defaultdict(set)
for p in ranks:
 x=load(p);v=x['inventory']; label=str(p.relative_to(D))
 check('inventory:'+label,x['observation_errors']==[] and x['counters']['dropped_rows']==0 and x['run_generation']=='v4-native-v028-h200-p2-64-formal-20261107-08' and v['device_identity']['name']=='NVIDIA H200' and [v['groups'][k]['world_size'] for k in ['tp','dp','ep']]==[8,4,32])
 owners[x['role']].add(v['groups']['world']['rank'])
 check('routing:'+label,v['moe_routing']['environment']=='balanced_topk' and v['moe_routing']['strategy']=='balanced_routing.BalancedTopKRouting' and len(v['moe_routing']['routers'])==43)
 check('quant:'+label,sum(a['role']=='mxfp4_quantization_inventory' for a in v['initialization']['auxiliary'])==43 and all(a['valid'] for a in v['initialization']['auxiliary']))
 for key,val in [('role',x['role']),('blocks',v['cache']['num_blocks']),('graph_configured',v['effective_graph']['configured_mode']),('graph_dispatcher',v['effective_graph']['dispatcher_mode']),('moe_backend',str(sorted({b['backend'] for b in v['initialization']['backends'].values() if 'experts' in b}))),('max_model_len',str(v['initialization']['resolved'].get('max_model_len'))),('scheduler',v['scheduler_class'])]:census[key][str(val)]+=1
 check('capture sizes:'+label,v['effective_graph']['capture_sizes']==[1,2,4]+list(range(8,257,8)))
 check('captures:'+label,len(v['effective_graph']['captured_descriptors'])>0)
 if p.name=='rank0.json':
  summaries['native'].setdefault('rank0_memory',[]).append({'host':x['host'],'role':x['role'],'blocks':v['cache']['num_blocks'],'memory':v['memory']['native_scalars'],'resolved':v['initialization']['resolved']})
check('64 complete worker inventories',len(ranks)==64 and owners=={'prefill':set(range(32)),'decode':set(range(32))})
summaries['native']['census']={k:dict(v) for k,v in census.items()}
cov=load(G/'coverage.json')
for s in cov['sources']:bind(Path(s['path']),s['sha256'],'coverage source:'+s['path'])
for i,c in enumerate(cov['cohorts']):
 stamps=[]
 for rec in c['tp_records']:
  text=Path(rec['path']).read_text().splitlines()[rec['line']-1]
  check(f'cohort {i}:{rec["line"]}',rec['local_stamp'] in text and 'compatib' in text.lower(),text)
  stamps.append(rec['utc_second'])
 check('cohort8:'+str(i),len(stamps)==8 and c['phase']=='warmup')
summaries['coverage']={'cohorts':len(cov['cohorts']),'records':sum(len(x['tp_records']) for x in cov['cohorts']),'bindings':dict(collections.Counter(c['binding'] for c in cov['cohorts']))}
result=load(G/'result.json')
for name,h in result['formal_replay']['files'].items():bind(Path('/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64')/name,h,'bound formal replay:'+name)
for case in ['p2_32','p2_64']:
 r=Path('/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs')/case;receipt=load(r/'receipt.json')
 for tab in receipt['profiling_tables']:bind(Path(tab['path']),tab['sha256'],case+' profiling:'+tab['path'])
 a=receipt['a1'];check(case+' operator-only config',receipt['status']=='FINISHED' and receipt['source_dirty_paths']==[] and set(a['enable_dummy_mode'].values())=={False} and set(a['cc_backend_types'].values())=={'measured'} and not any(a['step_cost_table_bound_by_predictor'].values()) and a['no_step_cost_file_beside_inputs'])
 costs=load(r/'collectives.json')['rows'];check(case+' exact collective costs',all(v['exact'] and v['lower_bytes']==v['upper_bytes']==v['requested_bytes'] for v in costs))
 check(case+' collective units arithmetic',all(math.isclose(v['predicted_ms'],v['lower_time_us']/1000) and math.isclose(v['predicted_ms']*v['call_count'],v['predicted_total_ms'],rel_tol=1e-10) for v in costs))
 counts=collections.Counter(); real=collections.defaultdict(collections.Counter);scopes=set(); widths=set(); zero=0;bad=[]
 with gzip.open(r/'forwards.jsonl.gz','rt') as f:
  for n,line in enumerate(f,1):
   v=json.loads(line);counts[(v['role'],v['real'],v['step_class'],v['mode'])]+=1; widths.add(tuple(v['num_tokens_across_dp']));scopes.update(v['scope_ms']);
   if v['real']:
    for ident in v['request_ids']:real[v['role']][ident]+=v['local_tokens']/max(1,len(v['request_ids']))
   zero+=v['end_s']<=v['start_s']
   if not math.isclose((v['end_s']-v['start_s'])*1000,v['composed_ms'],rel_tol=1e-9,abs_tol=1e-8):bad.append(n)
 check(case+' actual forward positive duration',zero==0 and not bad,{'nonpositive':zero,'duration_mismatch_lines':bad[:10]})
 summaries[case]={'source':receipt['source_commit'],'forwards':sum(counts.values()),'mode_counts':[{'role':k[0],'real':k[1],'class':k[2],'mode':k[3],'count':v} for k,v in counts.items()],'scopes':sorted(scopes),'collective_points':len(costs),'unequal_collective_points':sum(x['unequal_rows'] is not None for x in costs),'kv_admission':receipt['kv_admission'],'a11':receipt['a11']}
 check(case+' attention/runner/routing reachable',{'dsv4_sparse_attention','dsv4_compressor_projection','dsv4_index_scores','dsv4_index_topk','moe_gating_routing_topk','moe_grouped_gemm','dsv4_ep_dispatch_allgather','dsv4_input_preparation'}<=scopes)
summaries['modules_over_2000']={str(p):len(p.read_text().splitlines()) for p in (ROOT/'frontier').rglob('*.py') if len(p.read_text().splitlines())>2000}
(OUT/'evidence_audit.json').write_text(json.dumps({'interpreter':sys.version,'checks':checks,'summaries':summaries},indent=2)+'\n')
print('CHECKS',len(checks),'PASS',sum(x['status']=='PASS' for x in checks),'FAIL',sum(x['status']=='FAIL' for x in checks))
print(json.dumps(summaries,indent=2))
