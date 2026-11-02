import ast,json,sys
from pathlib import Path
from steptron.exp.base_exp import ResourceConfig
p=Path(sys.argv[1]);config=json.loads((p/'config.json').read_text());tree=ast.parse((p/'submit_native_diagnostic.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ResourceConfig'];assert len(calls)==1
r=eval(compile(ast.Expression(calls[0]),str(p/'submit_native_diagnostic.py'),'eval'),{'ResourceConfig':ResourceConfig,'config':config})
fields=('gpu','replica','cpu','mem_gb','preemptible','positive_tags','image','mounts','custom_resources','task_anti_affinities','restart_on');values={k:getattr(r,k) for k in fields}
assert (r.gpu,r.replica,r.cpu,r.mem_gb)==(8,4,64,512) and r.preemptible is False and r.restart_on is None
assert r.positive_tags==['H800'] and r.image==config['image'] and r.mounts==[config['cloud_mount']]
assert r.custom_resources==['mellanox.com/mlnx_rdma:1','rdma/mlnx_shared:8']
assert r.task_anti_affinities==['hard/default'] and r.envs['DISTRIBUTED_JOB']=='true'
assert not any(k.startswith('NCCL_') for k in r.envs)
assert all(r.envs[k]=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','VLLM_ENABLE_CUDA_COMPATIBILITY'))
print(json.dumps(dict(status='PASS_ACTUAL_SDK_RESOURCE_CONSTRUCTOR',python=sys.version,resources=values,platform_operations=0,scope='Actual frozen ResourceConfig expression; no backend submission, topology or GPU proof.')))
