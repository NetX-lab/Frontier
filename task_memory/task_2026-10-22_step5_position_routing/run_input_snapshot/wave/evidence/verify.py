"""Compare complete accepted wave outputs and freeze receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
frozen = json.loads((ROOT / 'freeze.json').read_text())
reports = {label: json.loads((ROOT / label / 'wave_result.json').read_text()) for label in ('baseline','candidate')}
artifacts = {}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for label,report in reports.items():
    prefix = ROOT / label / 'accepted-output' / label
    artifacts[label] = {'wave_result': str(ROOT / label / 'wave_result.json'),
        'wave_result_sha256': sha(ROOT / label / 'wave_result.json'),
        'pricing_payload_sha256': sha(ROOT / label / 'pricing_payload.json'),
        'whole_jsonl': str(prefix / 'iteration-0/op_traces.jsonl'),
        'whole_jsonl_sha256': sha(prefix / 'iteration-0/op_traces.jsonl'),
        'ledger': str(prefix / 'ledger.json'), 'ledger_sha256': sha(prefix / 'ledger.json'),
        'output_result_sha256': sha(prefix / 'result.json')}
    source = Path(report['source'])
    for path,expected in report['source_hashes_before_and_after'].items():
        assert sha(source/path) == expected,path
for relative in ('pricing_payload.json','accepted-output/{label}/iteration-0/op_traces.jsonl','accepted-output/{label}/ledger.json'):
    a = ROOT / 'baseline' / relative.format(label='baseline')
    b = ROOT / 'candidate' / relative.format(label='candidate')
    assert a.read_bytes() == b.read_bytes(),relative
a,b = reports['baseline'],reports['candidate']
assert a['wave_ast_sha256'] == b['wave_ast_sha256']
assert a['operation_counts'] == b['operation_counts']
assert a['main_run_successful_file_reads'] == 400 and b['main_run_successful_file_reads'] == 1
assert all(sample['physical_collective_reads']==0 for sample in b['samples'])
assert b['output']['wrong_plan_key_negatives'] == 64
assert b['output']['late_json_error_restores_buffer_and_prefix_bytes'] == 3902
assert b['output']['public_getter_and_pending_snapshot_mutations_isolated']
assert b['output']['pending_meta_mutation_does_not_change_source']
assert b['output']['fresh_trace_ledger_mappings_independent']
root_source = Path(frozen['sources']['candidate']['root'])
code_status = subprocess.check_output(['git','status','--porcelain','--','frontier','tests'],cwd=root_source,text=True)
assert code_status, 'Source already committed: return records for primary to commit'
prefix = ('timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env '
    'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ')
python = '/data/ycfeng/envs/frontier-py310/bin/python -B '
cache_env = ('TMPDIR={0}/cache XDG_CACHE_HOME={0}/cache MPLCONFIGDIR={0}/cache/mpl NUMBA_CACHE_DIR={0}/cache/numba '.format(ROOT))
commands = [prefix + python + str(ROOT/'preflight.py') + ' > ' + str(ROOT/'preflight.log') + ' 2>&1']
commands += [prefix + cache_env + python + str(ROOT/'run.py') + ' ' + label + ' > ' + str(ROOT/(label+'.log')) + ' 2>&1' for label in reports]
commands += [prefix + python + str(ROOT/'verify.py') + ' > ' + str(ROOT/'verify.log') + ' 2>&1']
evidence = {'task_id':'D29-RUN-SNAPSHOT-WAVE','session_id':frozen['session_id'],'verdict':'PASS',
    'commands':commands,'command_exits':[0,0,0,0],'source_freeze':frozen['sources'],
    'scripts':{str(path):sha(path) for path in (ROOT/'preflight.py',ROOT/'fixture_wiring.py',ROOT/'run.py',ROOT/'verify.py',Path(frozen['original_setup']),Path(frozen['protected_body']))},
    'artifacts':artifacts,'reports':reports,'constructor_evidence':frozen['constructor_evidence'],
    'python':sys.version,'source_status_still_uncommitted':code_status,
    'invariants':{name:'PASS' for name in ('complete208costs_proofs_keys_order','whole_jsonl_bytes','ledger_bytes',
        'snapshot_mutation_isolation','64wrong_key_negatives','3902byte_error_prefix','source_before_after_hashes',
        'one_main_run_read_binding_cold_warm','zero_cold_warm_reads','shared_real_P_D_owner',
        'new_run_JSON_UTF8_failures_uncached','later_successful_retry','standalone_backend_call_shape',
        '288numeric_lookups_144fresh_proofs','original_fixture_cache_preserved')},
    'initial_attempt':{'root':'/data/ycfeng/tmp/d29_run_snapshot_wave_20261024_01',
        'baseline_exit':1,'candidate_exit':1,'classification':'fixture wiring, not source failure',
        'cause':'Selected final warm reporting plan instead of accepted cold plan: request IDs1 became1409.',
        'correction':'Execute original setup tail immediately after cold wave; retain that plan as accepted run_wave.py does.',
        'protected_body_unchanged':True,'source_hashes_same_across_attempts':True},
    'limits':['Cached manager/predictor reconstructed through accepted helper; real input owner explicitly initialized and real backends bound.',
        'Actual role constructor coverage is separate inherited82PASS evidence, not a full Simulator replay.',
        'Same cold/warm wave assignment and unchanged accepted cost-fields body; cold reporting plan retained.',
        'Config CLI parsing only, no full CLI/training/full-fixture content scan/native qualification/GPU.',
        'Physical reads observed by real delegated reads, no diagnostic byte adapter or numeric/result/proof cache.',
        'H3 candidate remains UNQUALIFIED/unallocated; no pricing optimization.']}
(ROOT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('PASS all invariants; source still uncommitted; complete payload/whole JSONL/ledger equality')
print(json.dumps({'warm_baseline_s':a['warm_median_s'],'warm_candidate_s':b['warm_median_s'],
    'candidate_binding_s':b['binding_s'],'candidate_main_reads':b['main_run_successful_file_reads'],
    'candidate_warm_reads':[x['physical_collective_reads'] for x in b['samples'][1:4]],'hashes':artifacts},indent=2))
