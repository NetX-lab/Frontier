import datetime, json
from pathlib import Path
from brainpp.rjob import RJobClient
cd=Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
c=RJobClient(access_key=(cd/'accesskey_id').read_text().strip(),secret_key=(cd/'accesskey_secret').read_text().strip())
r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'operation':'read only','personal_live_jobs':[],'pages':0}
continuation=''
while True:
 d=c.api.list_namespaced_custom_object(group=c.group,version=c.version,namespace=c.namespace,plural=c.plural,label_selector='kubebrain.brainpp.cn/creator=i-fengyicheng',limit=100,_continue=continuation,_request_timeout=20)
 r['pages']+=1
 for j in d['items']:
  phase=(j.get('status') or {}).get('phase')
  if phase in ['Succeeded','Failed','Stopped']:continue
  labels=j['metadata'].get('labels',{});spec=j['spec'];gpus=0
  for name,task in spec['taskSpecs'].items():
   pod=task['template']['spec'];each=sum(float(x.get('resources',{}).get('limits',{}).get('nvidia.com/gpu',0)) for x in pod['containers']);gpus+=each*int(task.get('replicas',task.get('replica',1)))
  r['personal_live_jobs'].append({'name':j['metadata']['name'],'phase':phase,'gpu_reservations':gpus,'resource_labels':{k:v for k,v in labels.items() if any(w in k.lower() for w in ['preempt','quota','gpu'])},'preemptible_spec':spec.get('preemptible')})
 continuation=d.get('metadata',{}).get('continue','')
 if not continuation:break
r['personal_list_continues']=False
try:
 q=c.api.get_cluster_custom_object(group='quotagroup.brainpp.cn',version='v1',plural='quotagroups',name='step_main',_request_timeout=20)
 r['quota']={'spec':{k:v for k,v in q.get('spec',{}).items() if 'quota' in k.lower()},'status':q.get('status',{})}
except Exception as e:r['quota_error']={'type':type(e).__name__,'status':getattr(e,'status',None)}
p=Path('task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_three_run_20261101/quota.json');assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='quota'},indent=2));print('quota_status_keys',list(r.get('quota',{}).get('status',{})))
