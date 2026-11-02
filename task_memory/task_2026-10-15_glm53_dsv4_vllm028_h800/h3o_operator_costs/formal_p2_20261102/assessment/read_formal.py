"""Extract formal readings with harness arithmetic and source/event exclusions."""
import argparse
import csv
import json
import math
from pathlib import Path
from assess_inputs import HERE, FORMAL, H3O, TASK, load, record, save
from tests.analysis.e2e_warm_reading import collect
from tests.comparison.e2e_gate.gate import read_rows, metric_values


def window(frame):
    # Exact endpoint definition used by client_reading; no new throughput reducer.
    return dict(first_submission_s=float(frame['arrival_s'].min()),last_token_s=float(frame['completion_s'].max()),
                total_tokens=int(frame[['request_num_prefill_tokens','request_num_decode_tokens']].to_numpy().sum()))


def main(control=False):
    base=H3O/'two_fix_20261101' if control else FORMAL
    rows,diagnostics,details=[],[],{}
    events=load(H3O/'warm_target_20261101/openers.json')
    for case in ['p2_32','p2_64']:
        cmd=load(base/f'{case}_command.json');assert cmd['status']=='FINISHED', (case,cmd['status'])
        gate=Path(cmd['gate_root']);reading_path=base/f'readings/{case}_warm_reading.json'
        reading=load(reading_path);original=load(gate/'reading.json')
        reproduced=collect(gate/'rows',gate/'reading.json',H3O/f'warm_target_20261101/{case}_repeat_openers.json')
        assert reproduced==reading, 'Existing harness did not exactly reproduce formal warm reading'
        sample,sides=read_rows(gate/'rows');member=next(s for s in sides if s.startswith('frontier/'))
        assert reading['all_request']==original and reading['all_request_verbatim']==(gate/'reading.json').read_text()
        event_run=events['runs']['canonical_'+case]
        opener_records=[e for e in event_run['events'] if e['phase']=='formal']
        assert {e['bound_request_id'].rsplit('-repeat',1)[0] for e in opener_records} == set(sum(reading['openers'].values(),[]))
        case_windows={}
        for side,r in reading['warm']['repetitions'].items():
            ids=reading['openers'][side];repeat=side.rsplit('-',1)[-1]
            retained={s:sides[s].loc[~sides[s].index.isin(ids)] for s in [side,member]}
            original_windows={s:window(sides[s]) for s in [side,member]}
            retained_windows={s:window(retained[s]) for s in [side,member]}
            assert all(original_windows[s]==original['formal_windows'][s] for s in [side,member])
            for metric in r['metrics']:
                allm=next(x for x in original['repetitions'][side]['metrics'] if x['metric']==metric['metric'])
                if ids:
                    for s,key in [(side,'vllm'),(member,'frontier')]:
                        n,d,_=metric_values(retained[s])[metric['metric']]
                        assert abs(n/d-metric[key])<1e-10
                assert math.isfinite(metric['relative_error']) and metric['status']==('PASS' if metric['relative_error']<=.1 else 'FAIL')
                row=dict(case=case,repeat=repeat,metric=metric['metric'],unit='tokens/s' if metric['metric']=='token_throughput_tps' else 'ms',
                         warm_native=metric['vllm'],warm_frontier=metric['frontier'],warm_relative_error=metric['relative_error'],warm_status=metric['status'],limit=.10,
                         all_native=allm['vllm'],all_frontier=allm['frontier'],all_relative_error=allm['relative_error'],all_status=allm['status'],original_count=128,retained_count=r['retained_request_count'],excluded_ids=json.dumps(ids),
                         original_native_start_s=original_windows[side]['first_submission_s'],original_native_end_s=original_windows[side]['last_token_s'],
                         retained_native_start_s=retained_windows[side]['first_submission_s'],retained_native_end_s=retained_windows[side]['last_token_s'],
                         original_frontier_start_s=original_windows[member]['first_submission_s'],original_frontier_end_s=original_windows[member]['last_token_s'],
                         retained_frontier_start_s=retained_windows[member]['first_submission_s'],retained_frontier_end_s=retained_windows[member]['last_token_s'],
                         original_tokens=original_windows[side]['total_tokens'],retained_tokens=retained_windows[side]['total_tokens'],reading=str(reading_path.resolve()),reading_sha256=record(reading_path)['sha256'],
                         exclusion_evidence=str(H3O/'warm_target_20261101/openers.json')+'#/runs/canonical_'+case+'/events')
                rows.append(row)
            case_windows[repeat]=dict(original=original_windows,retained=retained_windows,exclusions=ids)
            for scope,frames in [('all_request',{s:sides[s] for s in [side,member]}),('warm',retained)]:
                for s,frame in frames.items():
                    for m in ['ttft_ms','tpot_ms','request_e2e_time_ms']:
                        for percentile,q in [('p50',.5),('p95',.95)]:
                            diagnostics.append(dict(case=case,repeat=repeat,scope=scope,side=s,metric=m,statistic=percentile,value=float(frame[m].quantile(q,interpolation='linear')),unit='ms',gating=False,method='pandas Series.quantile; linear interpolation'))
        details[case]=dict(reading=record(reading_path),original_gate=record(gate/'reading.json'),harness_exact_reproduction='PASS',
                           pooled_diagnostics=reading['warm']['metrics'],windows=case_windows,excluded_request_events=opener_records,
                           evidence=record(H3O/'warm_target_20261101/openers.json'),all_request_verdict=original['verdict'],warm_verdict=reading['warm']['verdict'],
                           repetition_count=len(reading['warm']['repetitions']),original_request_repeat_count=384,retained_request_repeat_count=reading['warm']['retained_request_repeat_count'])
    prefix='scratch/control_' if control else ''
    for name,data in [('gate_readings.csv',rows),('diagnostics.csv',diagnostics)]:
        with (HERE/(prefix+name)).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(prefix+'reading_details.json',details)
    print('PASS exact harness reproduction; metric rows',len(rows),'warm PASS',sum(r['warm_status']=='PASS' for r in rows),'all-request FAIL',sum(r['all_status']=='FAIL' for r in rows),'percentile diagnostics',len(diagnostics))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',action='store_true');main(p.parse_args().control)
