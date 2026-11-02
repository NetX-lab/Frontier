"""Read-only Q4 comparisons; reuse B29 comparator, stream full forward rows."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

from assess_inputs import HERE, H3O, FORMAL, load, record, save, sha, changes
spec = importlib.util.spec_from_file_location('b29', H3O / 'two_fix_20261101/compare_b29.py')
b29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b29)


def full_rows(a, b):
    count = mismatches = 0
    first = None
    old_hash,new_hash=hashlib.sha256(),hashlib.sha256()
    with gzip.open(a, 'rb') as old, gzip.open(b, 'rb') as new:
        for i, pair in enumerate(itertools.zip_longest(old, new)):
            count += 1
            if pair[0] is not None:old_hash.update(pair[0])
            if pair[1] is not None:new_hash.update(pair[1])
            x, y = (json.loads(v) if v is not None else None for v in pair)
            if x != y:
                mismatches += 1
                if first is None:
                    first = {'index': i, 'changes': changes(x, y)}
    return dict(rows=count, mismatches=mismatches, first_difference=first,old_decoded_bytes_sha256=old_hash.hexdigest(),new_decoded_bytes_sha256=new_hash.hexdigest(),status='PASS' if not mismatches and old_hash.digest()==new_hash.digest() else 'FAIL')


def reading_comparison(oldpath,newpath):
    old,new=load(oldpath),load(newpath)
    semantic=changes({k:v for k,v in old.items() if k!='sources'},{k:v for k,v in new.items() if k!='sources'})
    source_hashes_equal=[x['sha256'] for x in old['sources']]==[x['sha256'] for x in new['sources']]
    return dict(old=record(oldpath),new=record(newpath),semantic_differences=semantic,source_hashes_equal=source_hashes_equal,
                source_provenance_differences=changes(old['sources'],new['sources']),status='PASS' if not semantic and source_hashes_equal else 'FAIL')


def compare(case, control=False):
    newcmd = load((H3O / 'two_fix_20261101' if control else FORMAL) / f'{case}_command.json')
    assert newcmd['status'] == 'FINISHED', (case, newcmd['status'])
    if control:
        oldroot = Path(b29.B29_RUN.format(case=case))
        oldgate = Path(str(b29.B29_GATE).format(case=case)).parent
        oldreading = Path(str(b29.B29_READING).format(case=case, kind='warm'))
        newreading = H3O / f'two_fix_20261101/readings/{case}_warm_reading.json'
    else:
        oldcmd = load(H3O / f'two_fix_20261101/{case}_command.json')
        oldroot, oldgate = Path(oldcmd['run_root']), Path(oldcmd['gate_root'])
        oldreading = H3O / f'two_fix_20261101/readings/{case}_warm_reading.json'
        newreading = FORMAL / f'readings/{case}_warm_reading.json'
    newroot, newgate = Path(newcmd['run_root']), Path(newcmd['gate_root'])
    oldrec, newrec = load(oldroot / 'receipt.json'), load(newroot / 'receipt.json')
    oldmetrics, newmetrics = Path(oldrec['metrics_dir']), Path(newrec['metrics_dir'])
    identitydiff = changes(b29.input_identity(oldrec), b29.input_identity(newrec))
    sequences = b29.forward_comparison(b29.forwards(oldroot / 'forwards.jsonl.gz'), b29.forwards(newroot / 'forwards.jsonl.gz'))
    full = full_rows(oldroot / 'forwards.jsonl.gz', newroot / 'forwards.jsonl.gz')
    kv = b29.ledger_comparison(b29.ledger(oldmetrics / 'frontier_transfer_ledger.jsonl'), b29.ledger(newmetrics / 'frontier_transfer_ledger.jsonl'))
    requests = b29.request_comparison(b29.request_metrics(oldmetrics / 'request_metrics.csv'), b29.request_metrics(newmetrics / 'request_metrics.csv'))
    files = []
    for name in ['request_metrics.csv', 'frontier_transfer_ledger.jsonl', 'op_precision_metadata.csv', 'metrics_ground_truth.jsonl', 'system_metrics.json']:
        a, b = oldmetrics / name, newmetrics / name
        equal = sha(a) == sha(b)
        files.append(dict(file=name, old=record(a), new=record(b), status='PASS' if equal else 'FAIL'))
    files.append(dict(file='collectives.json', old=record(oldroot / 'collectives.json'), new=record(newroot / 'collectives.json'),
                      status='PASS' if sha(oldroot / 'collectives.json') == sha(newroot / 'collectives.json') else 'FAIL'))
    gate_rows = []
    new_gate_files = sorted((newgate / 'rows').rglob('*.csv'))
    assert len(new_gate_files) == 4, ('Expected native request rows for three repeats and Frontier rows', case, new_gate_files)
    for p in new_gate_files:
        a = oldgate / 'rows' / p.relative_to(newgate / 'rows')
        if control and not a.exists():
            a = Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/codex_resume_20261031') / f'{case}_warm_k19b29/rows' / p.relative_to(newgate / 'rows')
        gate_rows.append(dict(file=str(p.relative_to(newgate)), old=record(a), new=record(p), status='PASS' if sha(a) == sha(p) else 'FAIL'))
    ga, gb = load(oldgate / 'reading.json'), load(newgate / 'reading.json')
    ra, rb = load(oldreading), load(newreading)
    metricrows = b29.metric_rows(case, 'all_request', ga, gb) + b29.metric_rows(case, 'warm', ra['warm'], rb['warm'])
    readingdiff = changes(ga, gb) + changes(ra['warm'], rb['warm'], '/warm')
    assert rb['all_request_verbatim'] == (newgate / 'reading.json').read_text()
    assert rb['empty_set_reproduction'] == 'PASS_EXACT'
    reading_files=[]
    for kind in ['warm','empty']:
        oldpath=Path(str(b29.B29_READING).format(case=case,kind=kind)) if control else H3O/f'two_fix_20261101/readings/{case}_{kind}_reading.json'
        newpath=(H3O/'two_fix_20261101' if control else FORMAL)/f'readings/{case}_{kind}_reading.json'
        reading_files.append(dict(kind=kind,**reading_comparison(oldpath,newpath)))
    # Wire end is internal to engine_step; the public transfer ledger exposes scheduler completion.
    from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
    from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import AnalyticalKVCacheTransferPredictor
    from frontier.types import ClusterType
    prices=[]
    for r in [oldrec,newrec]:
        cfg=load(Path(r['metrics_dir'])/'config.json')['kv_cache_transfer_config']
        c=AnalyticalKVCacheTransferConfig(network_bandwidth_gbps=cfg['network_bandwidth_gbps'],network_latency_ms=cfg['network_latency_ms'],kv_connector='nixl',nixl_read_completion='engine_step',enable_compression=False)
        prices.append(AnalyticalKVCacheTransferPredictor(c).get_transfer_time(ClusterType.PREFILL,ClusterType.DECODE,None,39521664 if control and r is oldrec else 46103040))
    wire = dict(observed_field=False, source='Production analytical predictor on both resolved pairs, plus Q2 event proof; transfer ledger has no wire-end timestamp',
                old_price_ms=prices[0],new_price_ms=prices[1],expected_wire_end_delta_ms=prices[1]-prices[0],
                minimum_possible_next_step_gap_ms=18.724965326 if case == 'p2_32' else 24.380075963)
    semantic_pass = (full['status'] == 'PASS' and all(x['status'] == 'PASS' for x in files + gate_rows)
                     and not readingdiff and all(x['status']=='PASS' for x in reading_files) and all(v['zero'] == v['count'] for v in requests.values())
                     and all(kv[k]['zero'] == kv[k]['count'] for k in ['start_delta_ms','end_delta_ms','duration_delta_ms']))
    result = dict(case=case, mode='two_fix_vs_B29_control' if control else 'formal_vs_two_fix',
                  input_identity_changes=identitydiff, forwards=sequences, full_forward_rows=full,
                  scheduler_kv_transfers=kv, request_metrics=requests, files=files, gate_rows=gate_rows,
                  gate_and_warm_reading_differences=readingdiff,reading_files=reading_files,receipt_differences=changes(oldrec,newrec),wire_end=wire,
                  source_receipts={'old': record(oldroot / 'receipt.json'), 'new': record(newroot / 'receipt.json')},
                  metadata_note='Run roots, wall times, timestamps, receipt/source hashes and gzip header time are administrative provenance; compare decoded full forward rows and exact metric/gate-row bytes.',
                  status='PASS' if semantic_pass else 'FAIL')
    return result, metricrows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--control', action='store_true'); args = p.parse_args()
    results, rows = {}, []
    for case in ['p2_32', 'p2_64']:
        results[case], r = compare(case, args.control); rows += r
        print(case, results[case]['status'], 'full forward differences', results[case]['full_forward_rows']['mismatches'],
              'file findings', [(x['file'],x['status']) for x in results[case]['files'] if x['status'] != 'PASS'])
    target = HERE / 'scratch/full_control.json' if args.control else HERE / 'formal_vs_two_fix.json'
    target.write_text(json.dumps(dict(cases=results, existing_comparator=str(H3O / 'two_fix_20261101/compare_b29.py')), indent=2) + '\n')
    csvpath = HERE / ('scratch/full_control_metrics.csv' if args.control else 'scratch/formal_comparison_metrics.csv')
    with csvpath.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert all(x['status'] == 'PASS' for x in results.values()), 'Unexpected differences; inspect JSON'

if __name__ == '__main__': main()
