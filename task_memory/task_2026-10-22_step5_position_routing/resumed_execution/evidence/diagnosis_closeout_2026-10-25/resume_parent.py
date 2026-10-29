from pathlib import Path
from hashlib import sha256
import json, subprocess, sys, time
from datetime import datetime, timezone
root=Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
owner=Path('/data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish')
task='task_memory/task_2026-10-22_step5_position_routing'
source='2ca5e57113b0bc99ab84e926e5bb492b3670072e'
frozen='1461573de388b72450189bc0c1b88b4ac32491f0'
commit='6335f581e0129ddb273836f058f4e84fcbb35c22'
started=time.perf_counter()
def git(*args, cwd=root):
    return subprocess.check_output(['git','--no-optional-locks',*args],cwd=cwd)
e=json.loads((owner/task/'init_preparation_reuse/evidence.json').read_text())
assert git('rev-parse','HEAD',cwd=owner).decode().strip()==commit
assert not git('status','--porcelain',cwd=owner).strip()
assert git('show','-s','--format=%P',commit).decode().strip()==e['records_baseline']
init_paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines()
assert len(init_paths)==6 and all(p.startswith(task+'/init_preparation_reuse/') for p in init_paths)
for item in e['code_bindings']:
    path=item['path']
    assert sha256((root/path).read_bytes()).hexdigest()==item['sha256']
    for ref in (e['records_baseline'],frozen,source,'HEAD'):
        assert sha256(git('show',ref+':'+path)).hexdigest()==item['sha256']
for item in e['authority_and_retained_evidence']:
    content=git('show',item['git_commit']+':'+item['git_path'],cwd=Path(item['git_root'])) if 'git_commit' in item else Path(item['path']).read_bytes()
    assert len(content)==item['bytes'] and sha256(content).hexdigest()==item['sha256']
assert e['consumed_controls']['new_total']==e['consumed_controls']['recovered_total']==0
assert e['verdict']=='ALLOC' and e['prospective_paths_not_an_allocation']['allocated_lines']==0
assert e['inferred_contribution']['modeled_removed_work_s']==2*5*(9.232+3.426)
assert e['inferred_contribution']['accepted_loader_savings_recounted_s']==0
copy=json.loads((root/task/'runtime_snapshot_copy/evidence.json').read_text())
for item in copy['source_bindings']:
    path=item['path']
    assert git('rev-parse','HEAD:'+path).decode().strip()==item['target_blob']
    assert git('hash-object',str(root/path)).decode().strip()==item['target_blob']
copy_receipt=json.loads((root/task/'runtime_snapshot_copy/receipt.json').read_text())
copy_commit='b061f88458ef9b64c164d551038643eb9ca77fd5'
copy_paths=git('diff-tree','--no-commit-id','--name-only','-r',copy_commit).decode().splitlines()
assert sorted(copy_paths)==sorted(copy_receipt['exact_changed_paths'])
for path in copy_paths:
    assert (root/path).read_bytes()==git('show',copy_commit+':'+path)
assert copy_receipt['disposition']=='NO_ALLOC' and copy_receipt['control_usage']['used']==0
assert copy_receipt['control_usage']['launch_attempts']==0
assert not git('diff','--name-only',source,'HEAD','--','.',':!task_memory').strip()
assert not git('diff','--name-only',frozen,'HEAD','--','.',':!task_memory').strip()
result={'status':'PASS_RECORDS_AND_SOURCE_BINDINGS','source_commit':source,'current_head':git('rev-parse','HEAD').decode().strip(),'tested_candidate':frozen,'init_owner_commit':commit,'init_owner_parent':e['records_baseline'],'init_record_paths':init_paths,'init_code_bindings_verified':len(e['code_bindings']),'init_retained_bindings_verified':len(e['authority_and_retained_evidence']),'init_disposition':'ALLOC_RECOMMENDATION_ONLY','init_controls_new':0,'init_controls_recovered':0,'init_source_allocation':0,'init_modeled_removed_work_s':126.58,'init_current_startup_measured':False,'copy_owner_commit':copy_commit,'copy_record_paths':copy_paths,'copy_bindings_verified':len(copy['source_bindings']),'copy_disposition':'NO_ALLOC_STATIC_DIAGNOSIS','copy_controls_used':0,'copy_control_ceiling':1,'copy_launch_attempts':0,'non_task_source_equal_to_frozen_and_production':True,'python':sys.version,'executable':sys.executable,'wall_s':time.perf_counter()-started,'verified_utc':datetime.now(timezone.utc).isoformat(),'configured_memory_max_bytes':32212254720,'measured_cgroup_peak_bytes':None,'limitations':['Metadata/source/retained-receipt verification only; no runtime controls or suites','ALLOC is a recommendation, not a source allocation or measured startup saving','Required-shape usage remains6/6; no pricing/control allowance extended']}
output=Path('/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/diagnosis_integration/init_parent_before.json')
with output.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
print(json.dumps({k:v for k,v in result.items() if not k.endswith('_paths')},indent=2))
