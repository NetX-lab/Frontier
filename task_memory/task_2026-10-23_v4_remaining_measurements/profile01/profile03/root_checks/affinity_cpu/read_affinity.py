import json,runpy,sys
sys.path.insert(0,'/data/ycfeng/steptron')
sys.argv=['sdk_control','/data/ycfeng/Frontier/.worktrees/v4-native-profile03-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_03'];s=runpy.run_path('/data/ycfeng/tmp/glm53_dsv4/profile03_root_sdk_46joquue/input.py');r=s['r']
from brainpp.rjob.struct import Affinity,Resources,Task,Template
rows=[]
for name,cfg in r:
 a=Affinity(positive_tags=cfg.positive_tags,negative_tags=cfg.negative_tags,task_affinities=cfg.task_affinities,task_anti_affinities=cfg.task_anti_affinities);t=Task(template=Template(image=cfg.image,resources=Resources(cpu=cfg.cpu,gpu=cfg.gpu,memory_in_mb=cfg.mem_gb*1000,custom_resources=cfg.custom_resources),environments={},affinity=a),replicas=cfg.replica);k=t.to_k8s_template(name,'cpu-no-submission',Affinity(),{});rows.append(dict(name=name,leaf_anti_affinities=cfg.task_anti_affinities,serialized_affinity=k['template']['spec']['affinity']))
print(json.dumps(dict(status='OBSERVED_CPU_SERIALIZATION',platform_operations=0,root_anti_affinities=r.task_anti_affinities,rows=rows)))
