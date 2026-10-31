"""Check completed F1a evidence without rerunning normalization or the gate."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
run=Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f1a_p2_64_codex_20261031a')
base=Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29')
receipt=json.loads((run/'receipt.json').read_text())
baseline=json.loads((base/'receipt.json').read_text())
reading=json.loads((HERE/'f1a_p2_64/reading.json').read_text())
watch=json.loads((HERE/'watcher.json').read_text())['cases']['f1a_p2_64']
assert receipt['status']=='FINISHED' and receipt['source_dirty_paths']==[]
assert receipt['source_commit']=='9f5f8b614f1e278b02ad39015f0230de562c82ba'
assert receipt['a1']['enable_dummy_mode']=={'PREFILL':False,'DECODE':False}
assert receipt['a1']['step_cost_table_bound_by_predictor']=={'PREFILL':False,'DECODE':False}
assert receipt['a1']['cc_backend_types']=={'PREFILL':'astra_sim_analytical','DECODE':'astra_sim_analytical'}
assert receipt['a11']['stale_stage_end_events']==0
assert (run.with_suffix('.log')).read_text().splitlines()[-1]=='EXIT 0'
assert watch['exit_code']==0 and watch['status']=='READING_COMPLETE'
assert reading['formal_request_count']==128 and len(reading['repetitions'])==3
assert reading['request_id_gaps']==reading['token_count_mismatches']=={}
assert reading['client_metric_gaps']==[] and reading['blocked_status'] is None
assert reading['verdict']=='FAIL'
rows=list(csv.DictReader((HERE/'f1a_p2_64/rows/frontier/a10.csv').open()))
assert len(rows)==len({r['request_id'] for r in rows})==128
assert all(int(r['request_num_prefill_tokens'])==6144 and int(r['request_num_decode_tokens'])==1024 for r in rows)
reference_hashes={}
for p in (HERE/'f1a_p2_64/rows/vllm').glob('*.csv'):
    earlier=Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/codex_resume_20261031/p2_64_warm_k19b29/rows/vllm')/p.name
    assert p.read_bytes()==earlier.read_bytes()
    reference_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert len(reference_hashes)==3
compute={Path(r['path']).name:r['sha256'] for r in receipt['profiling_tables'] if '/compute/' in r['path']}
base_compute={Path(r['path']).name:r['sha256'] for r in baseline['profiling_tables'] if '/compute/' in r['path']}
assert compute==base_compute and len(compute)==6
for records in [reading['metrics']]+[r['metrics'] for r in reading['repetitions'].values()]:
    assert len(records)==4
    for m in records:
        actual=abs(m['frontier']-m['vllm'])/abs(m['vllm'])
        assert math.isclose(actual,m['relative_error'],rel_tol=1e-12,abs_tol=1e-12)
        assert m['status']==('PASS' if actual<=0.1 else 'FAIL')
result={'status':'PASS_EVIDENCE_FAIL_ACCURACY','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit':receipt['source_commit'],'source_dirty_paths':receipt['source_dirty_paths'],
        'launcher_session':50972,'launcher_exit_code':0,'business_exit_marker':'EXIT 0',
        'init_wall_s':receipt['init_wall_s'],'run_wall_s':receipt['run_wall_s'],
        'formal_rows':128,'formal_tokens':917504,'reference_csv_sha256':reference_hashes,
        'compute_table_sha256_equal_to_b29':compute,'gate':watch,'reading':str(HERE/'f1a_p2_64/reading.json'),
        'limitations':'Diagnostic ASTRA result; no promotion. Formal qualification remains open. Gate is consumed once, not regenerated.'}
with (HERE/'f1a_p2_64_verification.json').open('x') as out:json.dump(result,out,indent=2)
with (HERE/'f1a_p2_64_receipt.json').open('x') as out:json.dump(receipt,out,indent=2)
print(result['status'], 'rows',len(rows),'reference_matches',len(reference_hashes),'compute_matches',len(compute))
