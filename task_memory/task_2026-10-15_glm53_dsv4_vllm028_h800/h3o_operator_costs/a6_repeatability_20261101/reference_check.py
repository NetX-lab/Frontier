import csv,datetime,json,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(30*1024**3,30*1024**3))
repo=Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
sys.path[:0]=[str(repo),str(repo/'tests/analysis')]
from v4_attention_repeatability import read_selected,build_dsv4_replay_rows,plan_from_spec
here=repo/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101'
plan=plan_from_spec(here/'package_plan.json');mapping=json.loads((here/'case_mapping.json').read_text());selected={m['original'] for m in mapping.values()}
paths=[Path(p) for p in json.loads((here/'geometry_check.json').read_text())['raw_inputs']]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic();rows=[];inputs=[]
for p in paths:
 record,receipt=read_selected(p,selected);inputs.append(receipt);rows.extend(build_dsv4_replay_rows(plan,[record]))
with (repo/'data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv').open() as stream:
 table={(r['case_id'],r['attention_variant'],r['plans_tile_schedule']):r for r in csv.DictReader(stream) if r['case_id'] in selected}
count=0
for r in rows:
 t=table[(r['case_id'],r['attention_variant'],str(r['plans_tile_schedule']))]
 for k,v in r.items():
  if k.startswith('time_stats.') and k.endswith('.median'):
   assert abs(float(t[k])-v)<1e-12,(r['case_id'],k);count+=1
output={'status':'PASS','started_utc':started,'elapsed_seconds':time.monotonic()-clock,'reference_rows':len(rows),'published_medians_verified':count,'inputs':inputs}
(here/'reference_check.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
