import importlib.util,json,resource,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(30*1024**3,30*1024**3))
p=Path('tests/analysis/v4_f_variant_comparison.py')
s=importlib.util.spec_from_file_location('f_comparison',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
t=Path('task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800').resolve();st=Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4')
e=m.entries(t,st); b=m.baselines(e); results=[]
for variant,case,run,out in e:
 if variant!='F2':
  r=m.verify_reading(variant,case,run,out,b[case]);print(variant,case,r['verdict'],len(r['metrics']));results.append(r)
print('unavailable',[(r['variant'],r['status']) for r in m.unavailable(t)])
out=t/'h3o_operator_costs/f_comparisons_20261101/comparison_precheck.json'
with out.open('x') as stream:json.dump({'status':'PASS_EXISTING_EVIDENCE','python':sys.version,'results':results,'f2_status':'RUNNING_NO_FINAL_READING'},stream,indent=2)
