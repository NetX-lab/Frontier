"""Capture the existing launcher's actual ResourceConfig without platform calls."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import steptron.utils.stepmind as stepmind
repo=Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
relative=Path('task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101')
build=Path('/data/ycfeng/Frontier/.worktrees/g1h-supplement-build')
package=build/relative/'package_01'
receipt={}
def capture(resource, **kwargs):
    receipt.update({k:getattr(resource,k) for k in ('cpu','gpu','mem_gb','replica','image','positive_tags','extra_requirements','mounts','envs','command','task_specs','preemptible','custom_resources')})
    receipt.update(kwargs)
    receipt['submission_cwd']=os.getcwd()
    assert receipt['gpu']==1 and receipt['replica']==1 and receipt['cpu']==16 and receipt['mem_gb']==128
    assert receipt['preemptible'] is False and receipt['charged_group']=='step_main'
    assert receipt['positive_tags']==['H200']
    assert receipt['code_mount_point']==receipt['submission_cwd']==str(build)
    assert 'timeout 5400 python3' in receipt['command']
    assert receipt['extra_requirements']==[] and receipt['use_image'] is True
    return SimpleNamespace(rjob=SimpleNamespace(meta=SimpleNamespace(name='DRY_RUN_NO_SUBMISSION')),poll=lambda:'NOT_SUBMITTED')
stepmind.spawn_tasks=capture
os.environ['STEPMIND_BACKEND']='rjob'
spec=importlib.util.spec_from_file_location('sealed_a6_launcher',package/'launch/glm_producer_jobs.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.submit(package/'jobs/A6_REPEATABILITY.json')
receipt['status']='PASS_DRY_RUN'
receipt['platform_operations']=0
(repo/relative/'payload_check.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('envs','command')},indent=2))
