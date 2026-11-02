"""Execute final small Q4 audits after both replays finish; writes only assessment."""
import ast
import csv
import datetime
import json
import os
import platform
import math
from collections import Counter
import subprocess
import sys
from pathlib import Path
from assess_inputs import HERE, FORMAL, REPO, H3O, Q1, load, record, save, sha

ENV={'PYTHONPATH':f'{REPO}:{REPO}/tests/analysis','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':str(HERE/'scratch')}
PY='/data/ycfeng/envs/frontier-py310/bin/python'
steps=[('assess_inputs.py',120),('compare_formal.py',300),('read_formal.py',120),('audit_runtime.py',180),('account_gpu.py',120),('promotion_assessment.py',120),('render_report.py',120)]


EXPECTED={
 'assess_inputs.py':'27 dispositions, row26 coverage PASS; seven tables, exact population/config/source/helper/cache hashes bound for each FINISHED run',
 'compare_formal.py':'Both formal/two-fix comparisons PASS: zero complete forward differences, exact metrics/collectives/gate rows and both reading documents; all provenance deltas recorded',
 'read_formal.py':'Exact official-harness reproduction; 24 per-repeat warm metric rows and 144 percentile diagnostics; original all-request failures preserved',
 'audit_runtime.py':'Non-dummy measured timing, correct topology/flags/limits/native copies, 128 exact completions each; Q2 domain and no crossings',
 'account_gpu.py':'35 distinct relevant H200 jobs with exact allocation clocks separated from estimates/unknowns; total actual hours labelled unknown',
 'promotion_assessment.py':'Exactly three repeats/four finite errors each <=0.30 plus completed input/reference/runtime checks; draft only, original64 chronology and costed options',
 'render_report.py':'All eight results, 27 input rows and 16 ticket checkboxes, case-specific verdicts, unknown spend and promotion limitations; no task/evidence edits'
}


def protected():
    tablepaths=[x['path'] for x in load(Q1/'ledger.json')['inputs'] if x.get('path')]
    paths=[Q1/'ledger.json',Q1/'ledger.md',Q1/'decisions.md',Q1/'nixl_timing_q2/proposed_input.json',*map(Path,tablepaths)]
    return [record(p) for p in paths]


def snapshot():
    save('scratch/protected_before.json',protected())
    print('protected input snapshot saved; no replay polling/mutation')


def validate_outputs():
    inputs=load(HERE/'input_status.json')
    assert len(inputs['inputs'])==27 and inputs['current_counts']==dict(Counter(x['status'] for x in inputs['inputs']))
    assert inputs['coverage_verdict']=='PASS' and inputs['inputs'][25]['status']=='QUALIFIED'
    binding=load(HERE/'binding.json')
    comparison=load(HERE/'formal_vs_two_fix.json')['cases']
    runtime=load(HERE/'runtime.json')
    readings=load(HERE/'reading_details.json')
    verdict=load(HERE/'verdict.json')
    with (HERE/'gate_readings.csv').open() as stream: rows=list(csv.DictReader(stream))
    names={'ttft_ms','tpot_ms','request_e2e_time_ms','token_throughput_tps'}
    expected={(case,repeat,metric) for case in ['p2_32','p2_64'] for repeat in ['repeat01','repeat02','repeat03'] for metric in names}
    assert len(rows)==24 and {(r['case'],r['repeat'],r['metric']) for r in rows}==expected
    for row in rows:
        assert float(row['limit'])==.10
        for scope in ['warm','all']:
            assert all(math.isfinite(float(row[scope+'_'+key])) for key in ['native','frontier','relative_error'])
            assert float(row[scope+'_native'])!=0
            assert row[scope+'_status']==('PASS' if float(row[scope+'_relative_error'])<=.10 else 'FAIL')
        assert int(row['original_count'])==128 and int(row['original_tokens'])==917504
        assert int(row['retained_tokens'])==int(row['retained_count'])*7168
        assert int(row['retained_count'])+len(json.loads(row['excluded_ids']))==128
    for case in ['p2_32','p2_64']:
        numerical=all(r['warm_status']=='PASS' for r in rows if r['case']==case)
        assert verdict[case]['numerical_verdict']==readings[case]['warm_verdict']==('PASS' if numerical else 'FAIL')
        assert binding[case]['terminal_status']=='FINISHED' and binding[case]['checks_pass']
        assert sum(x['kind']=='profiling_table' for x in binding[case]['inputs'])==7
        assert binding[case]['generated_cache_count']>0 and all(x['status']=='PASS' for x in binding[case]['inputs'])
        assert comparison[case]['status']=='PASS' and comparison[case]['full_forward_rows']['mismatches']==0
        assert runtime[case]['status']=='PASS' and runtime[case]['completion_count']==128
    with (HERE/'diagnostics.csv').open() as stream: diagnostic=list(csv.DictReader(stream))
    assert len(diagnostic)==144 and all(r['statistic'] in ['p50','p95'] and r['gating']=='False' and math.isfinite(float(r['value'])) for r in diagnostic)
    milestones=load(HERE/'milestones.json')['items']
    assert Counter(x['section'] for x in milestones)=={'frontier_reference_run':3,'comparison_and_calibration':4,'promotion_32_to_64':4,'final_acceptance':5}
    assert all(set(x['status'])=={'p2_32','p2_64'} and set(x['status'].values())<={'PASS','FAIL','NOT_APPLICABLE','BLOCKED_ON_DECISION'} for x in milestones)
    promotion=load(HERE/'promotion.json')
    assert promotion['draft_only'] and not promotion['issued_receipt'] and promotion['receipt_timestamp'] is None
    assert promotion['promotion_predicate']==(promotion['numerical_predicate_30pct'] and promotion['qualified_input_predicate'] and promotion['formal32_runtime_coverage_verified'])
    assert promotion['numerical_predicate_30pct']==all(float(r['warm_relative_error'])<=.30 for r in rows if r['case']=='p2_32')
    if not promotion['chronology']['prelaunch_promotion_receipt_found'] and verdict['p2_64']['numerical_verdict']=='PASS':
        assert verdict['p2_64']['verdict']=='BLOCKED_ON_DECISION'
    gpu=load(HERE/'scratch/gpu_hours_detail.json')
    with (HERE/'gpu_hours.csv').open() as stream: jobs=list(csv.DictReader(stream))
    assert len(jobs)==gpu['job_count']==len({r['job_id'] for r in jobs})
    assert sum(r['gpu_hours']=='UNKNOWN' for r in jobs)==len(gpu['unknown_jobs'])
    assert abs(sum(float(r['gpu_hours']) for r in jobs if r['gpu_hours']!='UNKNOWN')-gpu['known_container_gpu_hours'])<1e-9
    assert gpu['total_actual_gpu_hours'].startswith('UNKNOWN') if gpu['unknown_jobs'] else True
    assert sum(x['job_count'] for x in gpu['scope_subtotals'].values())==gpu['job_count']
    print('PASS cross-artifact consistency:27 inputs,24 repeat metrics,144 percentile diagnostics,16 milestones,draft promotion and deduplicated GPU accounting')


def main():
    receipts=[]
    for c in ['p2_32','p2_64']:
        command=load(FORMAL/f'{c}_command.json');assert command['status']=='FINISHED',(c,command['status'])
        assert command['receipt_sha256']==sha(Path(command['run_root'])/'receipt.json')
        assert command['gate']['reading_sha256']==sha(Path(command['gate_root'])/'reading.json')
        for kind in ['warm','empty']:
            assert command['readings'][kind]['output_sha256']==sha(FORMAL/f'readings/{c}_{kind}_reading.json')
    for name,limit in steps:
        command=['timeout',f'{limit}s',PY,str(HERE/name)];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        log=HERE/'scratch'/f'final_{name[:-3]}.log'
        with log.open('w') as stream:done=subprocess.run(command,cwd=REPO,env={**os.environ,**ENV},stdout=stream,stderr=subprocess.STDOUT)
        r=dict(command=command,cwd=str(REPO),environment=ENV,python_version=platform.python_version(),started_utc=start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),expected=EXPECTED[name],exit_code=done.returncode,status='PASS' if done.returncode==0 else 'FAIL',log=record(log),observed=log.read_text())
        receipts.append(r);save('command_receipts.json',dict(commands=receipts))
        print(name,r['status'],r['observed'].strip(),flush=True)
        if done.returncode:raise SystemExit(done.returncode)
    validate_outputs()
    before=load(HERE/'scratch/protected_before.json');after=protected();assert before==after,'Protected evidence changed during assessment; investigate lead/external writer, do not overwrite.'
    files=['input_status.json','binding.json','formal_vs_two_fix.json','gate_readings.csv','milestones.json','promotion.json','gpu_hours.csv','verdict.json','report.md']
    assert all((HERE/name).exists() for name in files)
    assert len(load(HERE/'input_status.json')['inputs'])==27
    assert len(load(HERE/'milestones.json')['items'])==16
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    doc=dict(commands=receipts,protected_evidence_before=before,protected_evidence_after=after,protected_hash_check='PASS',replay_commands={c:record(FORMAL/f'{c}_command.json') for c in ['p2_32','p2_64']},artifact_checks='PASS',output_records=[record(HERE/name) for name in files])
    save('command_receipts.json',doc)
    with (HERE/'changed_files.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','sha256','bytes']);writer.writeheader()
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and path.name!='changed_files.csv':writer.writerow(record(path))
    print('PASS command/input/output integrity,27 rows,16 milestones and final artifacts; changed_files.csv lists every assessment file except itself')

if __name__=='__main__':snapshot() if '--snapshot' in sys.argv else main()
