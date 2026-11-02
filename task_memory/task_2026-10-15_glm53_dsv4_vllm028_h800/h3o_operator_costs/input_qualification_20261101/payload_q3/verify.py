"""Independent arithmetic and completeness audit of the Q3 evidence artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PYTHON = '/data/ycfeng/envs/frontier-py310/bin/python'
RELATIVE = str(HERE.relative_to(REPO))
ENV = ('env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 '
       'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
       f'TMPDIR="$PWD/{RELATIVE}/scratch" timeout 120s {PYTHON}')


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = json.loads((HERE/'results.json').read_text())
    audits = []
    for path, expected in results['evidence_hashes'].items():
        p = Path(path)
        assert p.stat().st_size == expected['bytes'] and digest(p) == expected['sha256'], path
    audits.append(dict(name='retained source and input immutability', status='PASS',
                       checked_files=len(results['evidence_hashes'])))

    groups = rows(HERE/'per_group_bound.csv')
    expected_counts = []
    for g in groups:
        bs, win = int(g['logical_block_size']), int(g['window_tokens'])
        skipped = max(0, 6144-win+1-int(g['extra_retained_tokens']))//bs if win else 0
        count = math.ceil(6144/bs)-skipped
        assert count == int(g['maximum_read_pages']) == int(g['d_non_null_pages'])
        assert int(g['registered_stride_bytes']) == 1002240
        assert int(g['bound_bytes']) == count*1002240
        assert int(g['structural_bytes']) == count*int(g['component_bytes_per_page'])
        expected_counts.append(count)
    assert expected_counts == [24, 2, 2, 2, 16]
    assert sum(int(g['bound_bytes']) for g in groups) == results['check1']['B_max'] == 46103040
    assert sum(int(g['structural_bytes']) for g in groups) == 39521664
    audits.append(dict(name='independent per-group closed-form arithmetic', status='PASS',
                       group_counts=expected_counts, B_max=46103040, structural_bytes=39521664))

    from frontier.config import AnalyticalKVCacheTransferConfig
    from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import AnalyticalKVCacheTransferPredictor
    from frontier.types import ClusterType
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(
        network_bandwidth_gbps=390.39349928254796, network_latency_ms=0.0236521842818856,
        kv_connector='nixl', enable_compression=False))
    actual = predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE, None, 46103040)
    expected = 0.0236521842818856 + 46103040*8/(390.39349928254796*1e6)
    assert actual == expected == results['check2']['production_price_ms']
    original = rows(HERE.parent/'nixl_timing_q2/possible_post_windows.csv')
    calculated = rows(HERE/'model_windows.csv')
    assert len(original) == len(calculated) == 2034
    keys = lambda r: (r['run'], r['transfer_id'], r['lane'], r['possible_post_s'])
    assert len({keys(r) for r in original}) == 2034
    assert {keys(r) for r in original} == {keys(r) for r in calculated}
    assert all(r['status']=='PASS' and float(r['price_ms'])==actual and
               float(r['margin_ms'])>0 for r in calculated)
    audits.append(dict(name='production formula and every possible post', status='PASS',
                       price_ms=actual, posting_windows=2034, crossings=0))

    raw_path = next(Path(p) for p in results['evidence_hashes'] if p.endswith('/C_nixl/nixl_reads.jsonl'))
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    qualifying = [r for r in raw if r['active_pairs']==8 and r['bytes_per_reader']>=46103040]
    coordinate = min(r['bytes_per_reader'] for r in qualifying)
    assert coordinate == 46103040
    expected_trials = {}
    for record in qualifying:
        if record['bytes_per_reader'] != coordinate:
            continue
        for reader in record['readers']:
            for i, timing in enumerate(reader['timings']):
                key = (record['case'], record['pass'], reader['rank'], i)
                assert key not in expected_trials
                expected_trials[key] = timing
    exported = rows(HERE/'selected_reader_trials.csv')
    observed = {}
    for trial in exported:
        key = (trial['case'], int(trial['pass_index']), int(trial['rank']), int(trial['trial_index']))
        assert key not in observed
        timing = expected_trials[key]
        for field in ('prep', 'post', 'complete'):
            assert float(trial[field+'_ms']) == timing[field+'_us']/1000
        assert trial['below_p2_32'] == trial['below_p2_64'] == 'True'
        assert float(trial['complete_ms']) < float(trial['gap_p2_32_ms'])
        assert float(trial['complete_ms']) < float(trial['gap_p2_64_ms'])
        observed[key] = float(trial['complete_ms'])
    assert set(expected_trials) == set(observed) and len(observed) == 1440
    assert max(observed.values()) == results['check3']['max_ms'] == 12.554511427879333
    slowest = [dict(case=k[0], pass_index=k[1], rank=k[2], trial_index=k[3], complete_ms=v)
               for k,v in observed.items() if v==max(observed.values())]
    audits.append(dict(name='all selected raw trials exported without exclusions', status='PASS',
                       selected_trials=1440, missing_trials=0, slowest_trials=slowest))

    assert [results[f'check{i}']['status'] for i in range(1,5)] == ['PASS']*4
    assert len(results['check4']['configurations'])==4
    assert all(not c['op_level_tracing'] and c['zero_duration_decode_forwards']==0 and
               c['explicit_capacity'] and c['peak_overlapping_transfers']==1
               for c in results['check4']['configurations'])
    test_log = (HERE/'lifecycle_tests.log').read_text()
    assert '9 passed' in test_log and 'failed' not in test_log.lower()
    failed_log = (HERE/'scratch/analysis_initial_failed.log').read_text()
    assert "KeyError: 'byte_count'" in failed_log
    commands = [
        dict(command=f'{ENV} {RELATIVE}/analyze.py > {RELATIVE}/analysis.log 2>&1',
             status='PASS', exit_code=0, expected='four checks PASS',
             actual='four checks PASS; B_max=46103040; 2034 windows; 1440 selected trials',
             log='analysis.log'),
        dict(command=f'{ENV} -m pytest tests/unit/test_nixl_read_completion.py -q -p no:cacheprovider '
                     f'--basetemp "$PWD/{RELATIVE}/scratch/pytest" > {RELATIVE}/lifecycle_tests.log 2>&1',
             status='PASS', exit_code=0, expected='READ lifecycle regressions pass',
             actual='9 passed in 0.78s', log='lifecycle_tests.log'),
        dict(command=f'{ENV} {RELATIVE}/verify.py > {RELATIVE}/verification.log 2>&1',
             status='PASS', exit_code=0, expected='immutable inputs; independent arithmetic; complete exports',
             actual=audits, log='verification.log')]
    verification = dict(status='PASS', python='3.10.6', cwd=str(REPO), commands=commands,
        initial_failure=dict(status='FAIL', exit_code=1, log='scratch/analysis_initial_failed.log',
                             error="KeyError: 'byte_count'", cause="Q3 script used byte_count; retained ledger uses bytes",
                             resolution='corrected Q3 lookup to bytes; rerun all four checks PASS; no retained source edit'),
        audits=audits, not_run=['native serving/GPU run', 'P2 replay', 'route B observer', 'H800 qualification'])
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',audit_count=len(audits),audits=audits)))


if __name__=='__main__':
    main()
