"""Archive the single completed two-call joint execution without rerunning it."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

repo = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
task = repo / 'task_memory/task_2026-10-22_step5_position_routing'
source = Path('/data/ycfeng/tmp/d29_current_joint_20261025_01')
capture = Path('/data/ycfeng/tmp/d29_current_joint_capture_20261025_01')
destination = task / 'resumed_execution/evidence/joint_timing_2026-10-25'
started = time.monotonic()

def digest(path, compressed=False):
    result = hashlib.sha256()
    size = 0
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
            size += len(chunk)
    return size, result.hexdigest()

business = json.loads((source/'execution.json').read_bytes())
outer = json.loads((capture/'execution.json').read_bytes())
assert business['status'] == outer['business_status'] == 'PASS_TWO_COHESIVE_CPU_FIXTURE_WAVES'
assert business['pricing_attempts'] == outer['pricing_attempts'] == 2
assert outer['exit_code'] == 0 and outer['automatic_retries'] == outer['required_shape_calls'] == 0
assert outer['records_head'] == outer['records_head_after'] == business['source']['records_head']
assert [s['label'] for s in business['samples']] == ['cold', 'warm']
for sample in business['samples']:
    validation = sample['validation']
    assert validation['status'] == 'PASS_COMPLETE_OUTPUT_PARITY'
    assert [validation[key] for key in ('ledger_rows','operator_rows','compute_proofs')] == [16,208,144]
    assert not sample['validation_in_timed_interval']
    assert sample['trace_buffer_after_close'] == 0 and sample['ledger_count'] == 16
    for filename, key in [('frontier_ep_wave_lane_ledger.jsonl','ledger'),('op_traces.jsonl','trace')]:
        assert digest(source/sample['label']/filename) == (validation[key+'_bytes'], validation[key+'_sha256'])
for label in ['stdout','stderr']:
    assert digest(capture/label) == (outer['artifacts'][label]['bytes'],outer['artifacts'][label]['sha256'])
preparation = task/'resumed_execution/evidence/execution_preparation_2026-10-25'
bindings = json.loads((preparation/'joint_bindings.json').read_bytes())
reference = Path(bindings['references']['pricing']['path'])
for sample in business['samples']:
    assert digest(source/sample['label']/'pricing_payload.json') == digest(reference)
    assert [row['ep_id'] for row in sample['lane_identities']] == list(range(16))
for ref in [business['source']['source_commit'],'1461573de388b72450189bc0c1b88b4ac32491f0']:
    assert not subprocess.check_output(['git','diff','--name-only',ref,'HEAD','--','.',':(exclude)task_memory'],cwd=repo,text=True).strip()
names=['step5-preview','step5-d29-ledger-stream','d29-ledger-final-1f6b4314','d29-snapshot-copy-finish','d29-init-preparation-finish']
worktrees=[]
for name in names:
    path=repo.parent/name
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True).strip()
    assert subprocess.run(['git','merge-base','--is-ancestor',head,'HEAD'],cwd=repo).returncode == 0
    if name != 'step5-preview':
        assert not status
    worktrees.append({'name':name,'path':str(path),'head':head,'status':status,'ancestor_of_primary':True})
destination.mkdir(parents=True, exist_ok=False)
records=[]
selected=[(source/'execution.json',Path('business_execution.json')),
          (capture/'execution.json',Path('command_execution.json')),
          (capture/'stdout',Path('command.stdout')),(capture/'stderr',Path('command.stderr')),
          (Path(__file__),Path('archive_joint.py'))]
for original, relative in selected:
    target=destination/relative
    shutil.copyfile(original,target)
    assert digest(original) == digest(target)
    size,sha=digest(original)
    records.append({'source':str(original),'archive':str(relative),'bytes':size,'sha256':sha,'compression':'none'})
for sample in business['samples']:
    for filename in ['pricing_payload.json','frontier_ep_wave_lane_ledger.jsonl','op_traces.jsonl']:
        original=source/sample['label']/filename
        relative=Path('outputs')/sample['label']/(filename+'.gz')
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with original.open('rb') as stream, target.open('xb') as output:
            with gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0) as compressed:
                shutil.copyfileobj(stream,compressed)
        assert digest(target,True) == digest(original)
        size,sha=digest(original)
        records.append({'source':str(original),'archive':str(relative),'bytes':size,'sha256':sha,
                        'compression':'gzip; exact full original bytes verified after decompression',
                        'archive_bytes':target.stat().st_size,'archive_sha256':digest(target)[1]})
receipt={'task_id':'D29-CURRENT-JOINT-TIMING-ARCHIVE','status':'PASS_EXACT_COMPLETED_EXECUTION_ARCHIVE',
         'collected_utc':datetime.now(timezone.utc).isoformat(),'source':business['source'],
         'pricing_calls_used_and_ceiling':[2,2],'required_shape_used_and_ceiling':[6,6],
         'new_runtime_or_platform_calls':0,'source_budget_audit_repeated':False,
         'full_output_revalidated_by_execution':True,'archive':str(destination),
         'files':records,'worktrees':worktrees,'wall_s':time.monotonic()-started,
         'compressed_full_output_original_bytes':sum(r['bytes'] for r in records if r['compression']!='none'),
         'compressed_full_output_archive_bytes':sum(r.get('archive_bytes',0) for r in records)}
(destination/'archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:receipt[key] for key in ['status','pricing_calls_used_and_ceiling','new_runtime_or_platform_calls','compressed_full_output_original_bytes','compressed_full_output_archive_bytes','wall_s']}))
