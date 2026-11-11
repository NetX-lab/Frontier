"""CPU-only evidence arithmetic and extracted pure composition, no full simulator imports."""
import ast, collections, csv, datetime, gzip, hashlib, json, math, statistics, subprocess, sys, types
from pathlib import Path
import numpy as np
ROOT=Path.cwd();T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800';OUT=T/'review_20261108_h200_mainline';Q=T/'h3o_operator_costs/input_qualification_20261101'
checks=[];summaries={}
def load(p):return json.loads(Path(p).read_text())
def check(n,b,d=None):checks.append({'name':n,'status':'PASS' if b else 'FAIL','detail':d})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def bind(p,h):check('hash:'+str(p),Path(p).is_file() and sha(p)==h)
clockdir=T/'h3o_operator_costs/formal_p2_20261102/gpu_hours_clocks';total=0;clockchecked=0;unknown=[]
for r in rows(clockdir/'gpu_hours_complete.csv'):
 hours=float(r['gpu_hours']);total+=hours
 if r['clock_file']:
  p=clockdir/r['clock_file'];bind(p,r['clock_sha256']);v=load(p)
 else:
  p=Path(r['source']);bind(p,r['source_sha256']);v=load(p)
 if isinstance(v,dict) and 'replicas' in v and isinstance(v['replicas'],list) and all('containers' in a for a in v['replicas']):
  try:
   secs=sum((datetime.datetime.fromisoformat(c['state']['terminated']['finishedAt'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(c['state']['terminated']['startedAt'].replace('Z','+00:00'))).total_seconds() for a in v['replicas'] for c in a['containers'])
   gpu=int(r['gpus'])/len(v['replicas']);observed=secs*gpu/3600
   check('clock arithmetic:'+r['job_id'],math.isclose(observed,hours,abs_tol=1e-9));clockchecked+=1
  except (KeyError,TypeError):unknown.append(r['job_id'])
 else:unknown.append(r['job_id'])
check('35 ledger sum',math.isclose(total,158.27166666666665,abs_tol=1e-9));summaries['historical_gpu_hours']={'rows':35,'sum':total,'clock_receipts_recomputed':clockchecked,'other_basis':unknown}
# Current generation gaps are inspected; do not substitute estimates for exact clocks.
summaries['formal_clock_receipts']=[]
for p in sorted((T/'h200_mainline').glob('p2_64*/**/*terminal*receipt*.json')):
 try:v=load(p)
 except (ValueError,UnicodeError):continue
 if isinstance(v,dict) and 'container_gpu_hours' in v:summaries['formal_clock_receipts'].append({'path':str(p),'job':v.get('job'),'hours':v.get('container_gpu_hours'),'known_hours':v.get('known_container_gpu_hours'),'complete':v.get('container_clocks_complete')})
# Extract only pure grouping and composition bodies to avoid package imports/writes.
source=ROOT/'frontier/attention/deepseek_v4.py';tree=ast.parse(source.read_text());names={'_integer','_approximate_gcd','deepseek_v4_kv_cache_groups','deepseek_v4_tile_schedule_planners'}
ns=dict(UNCOMPRESSED_VARIANT='uncompressed',CSA_VARIANT='csa',HCA_VARIANT='hca',ATTENTION_CACHE='attention',INDEXER_CACHE='indexer',SWA_CACHE='swa',COMPRESSOR_STATE_CACHE='compressor_state',INDEX_COMPRESSOR_STATE_CACHE='index_compressor_state')
ns['DEEPSEEK_V4_CACHES_BY_VARIANT']={'uncompressed':('swa',),'csa':('attention','indexer','swa','compressor_state','index_compressor_state'),'hca':('attention','swa','compressor_state')}
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(source), 'exec'),ns)
cfg=load(T/'h200_mainline/p2_64_formal_retry08/package_01/model/config.json');variants=[{0:'uncompressed',4:'csa',128:'hca'}[a] for a in cfg['compress_ratios'][:cfg['num_hidden_layers']]];weights=collections.Counter(zip(variants,ns['deepseek_v4_tile_schedule_planners'](variants)))
source=ROOT/'frontier/execution_time_predictor/dsv4_composition.py';tree=ast.parse(source.read_text());body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))];prefix=ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass\nfrom types import MappingProxyType\n').body
m=types.ModuleType('review_dsv4_composition');sys.modules[m.__name__]=m;m.__dict__.update(INPUT_SCOPE='dsv4_input_preparation',CUDA_GRAPH='cuda_graph',CUDA_GRAPH_PIECEWISE='cuda_graph_piecewise',EAGER='eager');exec(compile(ast.fix_missing_locations(ast.Module(body=prefix+body,type_ignores=[])),str(source),'exec'),m.__dict__)
audit=load(Q/'evidence/audit.json');att=audit['independent_attention'];original=rows(ROOT/'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv');others={}
key=lambda r:(r['case_id'],r['attention_variant'],str(r['plans_tile_schedule']).lower())
for name,v in att['table_records'].items():
 bind(Path(v['path']),v['sha256']);rr=rows(v['path']);others[name]={key(r):r for r in rr};check(name+' full population',len(rr)==7105 and len(others[name])==7105 and set(others[name])=={key(r) for r in original})
# Production exact lookup averages target medians over feature-identical rows.
# Derive the keys from the source feature builder, rather than using a raw row's scopes.
feature_source=ROOT/'frontier/attention/dsv4_features.py';feature_tree=ast.parse(feature_source.read_text())
fm=types.ModuleType('review_dsv4_features');sys.modules[fm.__name__]=fm
fm.__dict__.update(CSA_VARIANT='csa',HCA_VARIANT='hca',UNCOMPRESSED_VARIANT='uncompressed',DeepSeekV4CoverageError=ValueError,_integer=ns['_integer'])
feature_body=[n for n in feature_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
feature_prefix=ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass\n').body
exec(compile(ast.fix_missing_locations(ast.Module(body=feature_prefix+feature_body,type_ignores=[])),str(feature_source),'exec'),fm.__dict__)
lookup_values=collections.defaultdict(list)
def features_for(r):
 if r['phase']=='dummy':return fm.build_dsv4_dummy_features(int(r['padded_rows']))
 qs,cs,ds=[json.loads(r[c]) for c in ('prefill_seq_lens','prefill_kv_cache_sizes','decode_kv_cache_sizes')]
 return fm.build_dsv4_features(qs+[1]*len(ds),cs+ds,[False]*len(qs)+[True]*len(ds),variant=r['attention_variant'],compress_ratio=int(r['compress_ratio']),sliding_window=cfg['sliding_window'],index_topk=cfg['index_topk'],padded_rows=int(r['padded_rows']))
def scope_key(r,scope):
 features=features_for(r)
 k=(scope,r['attention_variant'],r['graph_runtime_mode'],r['phase'])
 if scope=='dsv4_sparse_attention':k+=(r['plans_tile_schedule'].lower()=='true',)
 return k+tuple(float(features.values[c]) for c in fm.dsv4_feature_columns(scope,r['phase']))
for r in original:
 for scope in json.loads(r['operator_scopes']):
  lookup_values[scope_key(r,scope)].append((float(r['time_stats.'+scope+'.median']),float(r['gap_stats.'+scope+'.median'])))
lookup={k:(statistics.mean(v[0] for v in vs),statistics.mean(v[1] for v in vs)) for k,vs in lookup_values.items()}
identity=att['independent_pass_identity_columns_checked'];inderrors=collections.defaultdict(list)
for omission,record in zip(audit['omissions'],att['workloads']):
 check('omission unit',omission['unit_id']==record['unit_id']);idx=omission['row_indices']+omission['missing_row_indices'];members=[original[i] for i in idx];pred=0;origwin=0;wins=collections.Counter()
 for r in members:
  scopes=json.loads(r['operator_scopes']);costs={n:lookup[scope_key(r,n)][0] for n in scopes};gaps={n:lookup[scope_key(r,n)][1] for n in scopes};planner=r['plans_tile_schedule'].lower()=='true';w=weights[(r['attention_variant'],planner)]
  pred+=w*m.compose_attention_phase(costs,gaps,variant=r['attention_variant'],graph_runtime_mode=r['graph_runtime_mode'],rows=int(float(r['padded_rows'])),multi_stream_gemm_token_threshold=int(float(r['multi_stream_gemm_token_threshold'])),indexer_short_context=r['indexer_short_context'].lower()=='true').elapsed_ms
  origwin+=w*float(r['layer_window_stats.median'])
  for name,table in others.items():
   other=table[key(r)];check('attention shape:'+name+str(key(r)),all(r[c]==other[c] for c in identity));wins[name]+=w*float(other['layer_window_stats.median'])
 check('attention original composition:'+str(record['unit_id']),math.isclose(pred*1000,record['original_exact_composition_us'],rel_tol=1e-10))
 check('attention original window:'+str(record['unit_id']),math.isclose(origwin*1000,record['original_measured_window_us'],rel_tol=1e-10))
 for name,win in wins.items():
  err=abs(pred-win)/abs(win);inderrors[(name,tuple(record['class']))].append(err)
  check('attention independent value:'+name+str(record['unit_id']),math.isclose(win*1000,record['independent_pass_windows_us'][name],rel_tol=1e-10) and math.isclose((pred-win)/win,record['signed_errors'][name],rel_tol=1e-10,abs_tol=1e-12))


summaries['attention']={'layer_counts':{str(k):v for k,v in weights.items()},'lookup_keys':len(lookup),'aggregated_lookup_keys':sum(len(v)>1 for v in lookup_values.values())}
for k,v in inderrors.items():
 med,p90=np.quantile(v,[0.5,0.9]);check('attention per-pass class limits:'+str(k),med<=0.05 and p90<=0.10);summaries['attention'][str(k)]={'count':len(v),'median_error':float(med),'p90_error':float(p90),'maximum':max(v)}
summaries['attention']['maximum']=max(e for v in inderrors.values() for e in v)
summaries['attention']['comparisons']=sum(len(v) for v in inderrors.values())
# Native NIXL full-batch critical medians: same-index slowest reader per trial.
rawpath=Path('/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/C_nixl/nixl_reads.jsonl');raw=[json.loads(l) for l in rawpath.read_text().splitlines()];bind(rawpath,load(Q/'nixl_timing_q2/results.json')['integrity']['raw_sha256'])
observations=[];all_coordinate=[];trials=0;notifs=0
for r in raw:
 readers=[a for a in r['readers'] if a.get('reader')];n=len(readers[0]['timings']);critical=[max(a['timings'][i]['complete_us'] for a in readers)/1000 for i in range(n)]
 check('NIXL data:'+r['case']+str(r['pass']),all(a['data_verified'] for a in readers) and r['notifications_received']==r['notifications_expected'] and all(t>0 and math.isfinite(t) for t in critical));trials+=sum(len(a['timings']) for a in readers);notifs+=r['notifications_received']
 observations.append((r['pass'],r['case'],r['bytes_per_reader'],statistics.median(critical)))
 if r['prompt_tokens']==6144 and r['active_pairs']==8 and r['requests_in_flight']==1:all_coordinate+=critical
check('NIXL population',len(raw)==90 and len({r['case'] for r in raw})==30 and trials==12150 and notifs==80352)
xs=np.asarray([v[2]/1e6 for v in observations]);ys=np.asarray([v[3] for v in observations]);ab=np.linalg.lstsq(np.column_stack((np.ones(len(xs)),xs)),ys,rcond=None)[0];proposed=load(Q/'nixl_timing_q2/proposed_input.json')
check('NIXL affine coefficients',math.isclose(float(ab[0]),proposed['model']['intercept_ms'],rel_tol=1e-10) and math.isclose(float(ab[1]),proposed['model']['slope_ms_per_MB'],rel_tol=1e-10))
errs=[]
for pas in range(3):
 train=[i for i,v in enumerate(observations) if v[0]!=pas];a,b=np.linalg.lstsq(np.column_stack((np.ones(len(train)),xs[train])),ys[train],rcond=None)[0];e=[abs(a+b*v[2]/1e6-v[3])/v[3] for v in observations if v[0]==pas];errs+=e;check('NIXL heldout physical and median:'+str(pas),a>0 and b>0 and max(e)<=0.05)
check('NIXL exact coordinate tails',len(all_coordinate)==90 and max(all_coordinate)<18.724965)
summaries['nixl']={'records':len(raw),'reader_trials':trials,'notifications':notifs,'intercept_ms':float(ab[0]),'bandwidth_gbps':float(8/ab[1]),'maximum_heldout_median_error':max(errs),'p2_coordinate_max_ms':max(all_coordinate)}
# Prelaunch chronology: receipt bytes are embedded as a hash in the actual native launch manifest.
receipt=T/'h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json';issued=load(receipt)['issued_utc'];package=T/'h200_mainline/p2_64_formal_retry08/package_01';manifest=load(package/'case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-08/run_manifest.json');config=load(package/'config.json')
checksummaries={}
for k,v in manifest.items():
 if 'promotion' in k or 'created' in k or 'started' in k or 'time' in k: checksummaries[k]=v
check('promotion exact bytes prelaunch binding',sha(receipt) in json.dumps(config) and sha(receipt) in json.dumps(manifest))
submission=load(T/'h200_mainline/p2_64_formal_retry08/lead_submission_acceptance.json')['job_creation_timestamp'];check('promotion precedes generation08',datetime.datetime.fromisoformat(issued)<datetime.datetime.fromisoformat(submission.replace('Z','+00:00')))
# Corrected 2026-11-11 (Codex review F003): task_memory is gitignored but this receipt was force-added, so Git does carry its creation timestamp.
gitlog=subprocess.run(['git','log','--follow','--diff-filter=A','--format=%H %cI','--',str(receipt.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True).stdout.split()
commit,committed_utc=gitlog[0],gitlog[1];blob=subprocess.run(['git','show',commit+':'+str(receipt.relative_to(ROOT))],cwd=ROOT,capture_output=True).stdout
check('promotion receipt committed unchanged before submission',hashlib.sha256(blob).hexdigest()==sha(receipt) and datetime.datetime.fromisoformat(committed_utc)<datetime.datetime.fromisoformat(submission.replace('Z','+00:00')))
summaries['promotion']={'issued':issued,'generation08_submission_utc':submission,'receipt_sha256':sha(receipt),'actual_manifest_fields':checksummaries,'git_history':{'adding_commit':commit,'committed_utc':committed_utc,'committed_bytes_identical':hashlib.sha256(blob).hexdigest()==sha(receipt)}}
# Historical regression evidence remains detailed after approved cleanup.
fid=Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4');summaries['fidelity']={}
for stage in ['fidelity','fidelity_b24','fidelity_b26','fidelity_b26b','fidelity_b28','fidelity_b29']:
 p=fid/stage/'comparison.json';v=load(p);mism=v['mismatched_cases'];unexpected=[d for c in mism for d in c['differences'] if (d['artifact'],d['detail']) not in {('config.json','/request_ingress_delay_ms: missing in baseline'),('config.json','/kv_cache_transfer_config/request_handoff_delay_ms: missing in baseline')}]
 check('fidelity:'+stage,v['cases_compared']==77 and not unexpected and not v['candidate_only_failures'] and not v['predictor_cache_files_only_in_baseline'] and not v['predictor_cache_files_only_in_candidate']);summaries['fidelity'][stage]={'identical':len(v['identical_cases']),'mismatched':len(mism),'unexpected_differences':len(unexpected),'candidate_only_failures':v['candidate_only_failures']}
logs=[Path('/data/ycfeng/tmp/glm53_dsv4/h200/p2_dp_recovery_harness_tests.log')]
summaries['historical_logs']={str(p):p.read_text()[-300:] if p.exists() else 'MISSING' for p in logs}
(OUT/'qualification_audit.json').write_text(json.dumps({'interpreter':sys.version,'checks':checks,'summaries':summaries},indent=2)+'\n');print('CHECKS',len(checks),'PASS',sum(v['status']=='PASS' for v in checks),'FAIL',sum(v['status']=='FAIL' for v in checks));print(json.dumps(summaries,indent=2));print('FAILURES',json.dumps([v for v in checks if v['status']=='FAIL'],indent=2))
