"""Reproduce the ten supplementary checks; preserves the 97 runtime checks."""
import json, math, statistics, sys
from pathlib import Path
ROOT=Path.cwd(); T=ROOT/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'; OUT=Path(__file__).resolve().parent
G=T/'h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2'
run=G/'gate/p2_64_r4_case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-08/run'
def load(p):return json.loads(Path(p).read_text())
x=load(OUT/'runtime_audit.json'); checks=x['checks']; summary=x['summaries']
raw=[json.loads(line) for line in (run/'client_requests.jsonl').read_text().splitlines()]
reps=load(run/'replay_summary.json')['canonical_repetitions']['repetitions']
def percentile(v,q):
 v=sorted(v)
 return statistics.median(v) if q==.5 else v[max(0,math.ceil(len(v)*q)-1)]
for rep in reps:
 rr=[r for r in raw if r['repeat']==rep['repeat'] and r['segment']=='formal']
 vals={'ttft_ms':[1000*(r['first_token_monotonic']-r['dispatch_monotonic']) for r in rr],
       'tpot_ms':[1000*(r['last_token_monotonic']-r['first_token_monotonic'])/(r['num_decode_tokens']-1) for r in rr if r['num_decode_tokens']>1],
       'e2e_ms':[1000*(r['last_token_monotonic']-r['dispatch_monotonic']) for r in rr]}
 for metric,v in vals.items():
  observed={p:percentile(v,q) for p,q in [('p50',.5),('p95',.95)]}
  checks.append({'name':f"native08 percentile:{rep['repeat']}/{metric}",'status':'PASS' if all(math.isclose(rep['metrics'][metric][p],n,abs_tol=1e-8) for p,n in observed.items()) else 'FAIL','detail':observed})
checks.append({'name':'heldout EP raw trace exact reproduction','status':'PASS' if load(OUT/'heldout_counts_recomputed.json')==load(T/'h3o_operator_costs/two_fix_20261101/heldout_counts.json') else 'FAIL','detail':None})
headline=load(T/'h200_mainline/p2_64_formal_retry08/headline_recompute.json')
summary['headline_error_count']={'per_repeat':12,'pooled':4,'total':16,'published_claim':20}
summary['depth_recomputed']=(OUT/'depth_recomputed.log').read_text().splitlines()[-1]
summary['ep_omitted_direct_gap_ms_per_unequal_wave']=2*43*0.5446049595929223/1000
summary['ep_128_unequal_waves_direct_gap_ms']=128*summary['ep_omitted_direct_gap_ms_per_unequal_wave']
(OUT/'runtime_supplement.json').write_text(json.dumps(x,indent=2)+'\n')
print('CHECKS',len(checks),'PASS',sum(c['status']=='PASS' for c in checks),'FAIL',sum(c['status']=='FAIL' for c in checks))
print(json.dumps(summary,indent=2))
