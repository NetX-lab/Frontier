from pathlib import Path
import datetime,json
from brainpp.rjob import RJobClient
A=Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng');c=RJobClient(access_key=(A/'accesskey_id').read_text().strip(),secret_key=(A/'accesskey_secret').read_text().strip());rows=[]
for job in c.list(['exp-1024-141341-669621']):
 for task in job.spec.tasks.values():
  for r in task.replicaStatus or []:rows.append(r.to_dict())
nodes=[r.get('node_name') for r in rows if r.get('node_name')];print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job='exp-1024-141341-669621',replicas=rows,assigned_nodes=nodes,distinct_nodes=len(set(nodes)),placement_verified=len(nodes)==4 and len(set(nodes))==4,scope='Single actual-placement snapshot prompted by SDK-hard versus platform-preferred affinity; no completion loop or controls.')))
