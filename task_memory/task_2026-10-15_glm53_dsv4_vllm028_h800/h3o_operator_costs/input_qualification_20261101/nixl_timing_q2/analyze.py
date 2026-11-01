"""Rebuild H200 NIXL observations, candidate rules and P2 concurrency/window proofs.

Read-only inputs; outputs are restricted to this evidence directory. No E2E
metric outcome is read. Quantiles use the producer's floor(q * count) index.
"""
from __future__ import annotations

import bisect
import collections
import csv
import datetime
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
from pathlib import Path

import numpy as np

from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVRegionLayout, read_cases, summarize_case,
)

HERE = Path(__file__).resolve().parent
QUALIFICATION = HERE.parent
TASK = HERE.parents[2]
REPO = HERE.parents[4]
DOWNLOAD = Path('/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download')
RUNS = {
    f'{case}_{version}': base / case_name
    for case in ('p2_32', 'p2_64')
    for version, base, case_name in (
        ('b29', Path('/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10'), f'{case}_warm_k19b29'),
        ('two_fix', Path('/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/runs'), case),
    )
}
INPUTS: dict[str, dict] = {}


def digest(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def track(path):
    path = Path(path)
    INPUTS[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return path


def load(path):
    return json.loads(track(path).read_text())


def p90(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * len(values)))]


def stats(values):
    return dict(median_ms=statistics.median(values), p90_ms=p90(values),
                min_ms=min(values), max_ms=max(values), count=len(values))


def relative(prediction, measurement):
    return abs(prediction - measurement) / measurement * 100


def affine(xs, ys):
    """OLS in ms and decimal MB. Reject nonphysical coefficients, never clamp."""
    a, b = np.linalg.lstsq(np.column_stack((np.ones(len(xs)), xs)), ys, rcond=None)[0]
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError(f'Nonphysical unconstrained fit: intercept={a}, slope={b}')
    return dict(intercept_ms=float(a), slope_ms_per_MB=float(b), bandwidth_gbps=float(8 / b))


def predict(model, byte_count):
    return model['intercept_ms'] + model['slope_ms_per_MB'] * byte_count / 1e6


def sweep(rows):
    """Peak counts on half-open [start,end) windows; process ends before starts."""
    events = []
    for r in rows:
        if r['end_ts_s'] <= r['start_ts_s']:
            raise ValueError('Nonpositive window')
        lane = (r['target_replica_id'], r['target_replica_local_id'])
        events.extend(((r['start_ts_s'], 1, lane), (r['end_ts_s'], -1, lane)))
    counts, peaks = collections.Counter(), collections.Counter()
    peak_total = peak_lanes = 0
    witness = None
    for t, delta, lane in sorted(events):
        counts[lane] += delta
        assert counts[lane] >= 0
        peaks[lane] = max(peaks[lane], counts[lane])
        total = sum(counts.values())
        peak_lanes = max(peak_lanes, sum(v > 0 for v in counts.values()))
        if total > peak_total:
            peak_total = total
            witness = {'time_s': t, 'counts': {str(k): v for k, v in counts.items() if v}}
    assert not any(counts.values())
    return dict(peak_requests=peak_total, peak_active_lanes=peak_lanes,
                peak_per_lane={str(k): v for k, v in peaks.items()}, witness=witness)


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurement():
    verification = load(QUALIFICATION / 'nixl6144_run/verification.json')
    assert verification['status'] == 'PASS' and len(verification['checks']) == 11
    assert all(c['pass'] for c in verification['checks'])
    manifest = load(DOWNLOAD / 'download_manifest.json')
    for entry in manifest:
        path = track(DOWNLOAD / entry['path'])
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    assert len(manifest) == 23
    leaf = load(QUALIFICATION / 'packages/nixl6144/nixl_leaf.json')
    stage = load(QUALIFICATION / 'packages/nixl6144/stage_nixl6144.json')
    plan = load(DOWNLOAD / 'C_nixl/nixl_plan.json')
    platform_terminal = load(QUALIFICATION / 'nixl6144_run/platform_terminal.json')
    raw_path = track(DOWNLOAD / 'C_nixl/nixl_reads.jsonl')
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    layout = KVRegionLayout.from_config(leaf['kv_cache'])
    cases = read_cases(leaf['grid'], layout, leaf['tensor_parallel_size'])
    expected = {(c.name, p) for c in cases for p in range(leaf['passes'])}
    counts = collections.Counter((r['case'], r['pass']) for r in raw)
    assert set(counts) == expected and set(counts.values()) == {1}
    original = list(csv.DictReader(track(DOWNLOAD / 'C_nixl/nixl_summary.csv').open()))
    assert len(original) == len(raw) == 90
    assert all(c[k] == str(v) for r, c in zip(raw, original) for k, v in summarize_case(r).items())
    retained = list(csv.DictReader(track(QUALIFICATION / 'nixl6144_run/case_summary.csv').open()))
    retained = {(r['case'], int(r['pass'])): r for r in retained}
    records, readers, trials = [], [], []
    for r in sorted(raw, key=lambda r: (r['prompt_tokens'], r['active_pairs'], r['requests_in_flight'], r['pass'])):
        c = next(c for c in cases if c.name == r['case'])
        assert r['blocks_per_request'] == c.blocks_per_request
        assert r['bytes_per_reader'] == c.blocks_per_request * c.requests_in_flight * layout.block_bytes
        assert r['descriptors_per_request'] == c.blocks_per_request * layout.num_regions
        assert [rd['rank'] for rd in r['readers']] == list(range(8, 8 + r['active_pairs']))
        assert r['notifications_received'] == r['notifications_expected'] == r['active_pairs'] * r['requests_in_flight'] * 32
        reader_values = []
        for rd in r['readers']:
            assert rd['data_verified'] is True and len(rd['timings']) == 30
            values = [t['complete_us'] / 1000 for t in rd['timings']]
            assert all(math.isfinite(v) and v > 0 for v in values)
            assert all(t['prep_us'] >= 0 and t['post_us'] >= 0 and
                       t['complete_us'] >= t['prep_us'] + t['post_us'] for t in rd['timings'])
            reader_values.extend(values)
            readers.append(dict(case=r['case'], pass_index=r['pass'], rank=rd['rank'], **stats(values)))
            for i, t in enumerate(rd['timings']):
                trials.append(dict(case=r['case'], pass_index=r['pass'], rank=rd['rank'], trial_index=i,
                                   **{k.replace('_us', '_ms'): v / 1000 for k, v in t.items()}))
        critical = [max(t['complete_us'] for t in repeat) / 1000
                    for repeat in zip(*[rd['timings'] for rd in r['readers']])]
        s, rs = stats(critical), stats(reader_values)
        prior = retained[(r['case'], r['pass'])]
        for col, val in [('complete_median_ms', s['median_ms']), ('complete_p90_ms', s['p90_ms']),
                         ('complete_max_ms', s['max_ms']), ('reader_complete_median_ms', rs['median_ms']),
                         ('reader_complete_p90_ms', rs['p90_ms']), ('reader_complete_max_ms', rs['max_ms'])]:
            assert abs(float(prior[col]) - val) <= 5e-7
        records.append(dict(case=r['case'], pass_index=r['pass'], prompt_tokens=r['prompt_tokens'],
                            pairs=r['active_pairs'], inflight=r['requests_in_flight'],
                            bytes_per_request=c.blocks_per_request * layout.block_bytes,
                            bytes_per_reader=r['bytes_per_reader'], critical=s, reader=rs,
                            critical_trials_ms=critical))
    assert len(trials) == 12150
    write_csv('reader_summary.csv', readers)
    write_csv('scratch/reader_trials.csv', trials)
    pooled, spreads = {}, []
    for case in cases:
        rr = [r for r in records if r['case'] == case.name]
        values = [v for r in rr for v in r['critical_trials_ms']]
        m = statistics.median(values)
        pooled[case.name] = m
        medians = [r['critical']['median_ms'] for r in rr]
        deviations = [relative(m, v) for v in medians]
        spreads.append(dict(case=case.name, prompt_tokens=case.prompt_tokens, pairs=case.active_pairs,
                            inflight=case.requests_in_flight, pass0_median_ms=medians[0],
                            pass1_median_ms=medians[1], pass2_median_ms=medians[2], pooled_median_ms=m,
                            spread_pct=(max(medians) - min(medians)) / statistics.median(medians) * 100,
                            max_deviation_from_pooled_pct=max(abs(v / m - 1) * 100 for v in medians)))
    write_csv('repeatability.csv', spreads)
    # Equal-byte control by matched pass, retaining all repetitions and passes.
    paired = []
    for p in (1, 8):
        for n in (1, 2, 4, 8, 16):
            for pass_index in (0, 1, 2):
                a, b = [next(r for r in records if r['prompt_tokens'] == t and r['pairs'] == p
                             and r['inflight'] == n and r['pass_index'] == pass_index) for t in (6143, 6144)]
                assert a['bytes_per_reader'] == b['bytes_per_reader']
                paired.append(dict(pairs=p, inflight=n, pass_index=pass_index,
                                   median_6143_ms=a['critical']['median_ms'], median_6144_ms=b['critical']['median_ms'],
                                   delta_ms=b['critical']['median_ms'] - a['critical']['median_ms'],
                                   median_difference_pct=relative(b['critical']['median_ms'], a['critical']['median_ms']),
                                   p90_6143_ms=a['critical']['p90_ms'], p90_6144_ms=b['critical']['p90_ms'],
                                   max_6143_ms=a['critical']['max_ms'], max_6144_ms=b['critical']['max_ms']))
    write_csv('equal_byte_passes.csv', paired)
    # Preserve earlier unphysical fit as evidence, without trying to repair it.
    old = load(TASK / 'h3o_operator_costs/b30b/f3_nixl02_20261101/fit_diagnosis.json')
    return records, pooled, dict(records=90, cases=30, passes=[0, 1, 2], reader_trials=len(trials),
                               critical_path_trials=2700, notifications=sum(r['notifications_received'] for r in raw),
                               max_pass_spread_pct=max(r['spread_pct'] for r in spreads),
                               median_pass_spread_pct=statistics.median(r['spread_pct'] for r in spreads),
                               max_equal_byte_pass_difference_pct=max(r['median_difference_pct'] for r in paired),
                               raw_sha256=digest(raw_path), prior_fit_diagnosis=old)


def candidates(records, pooled):
    xs = [r['bytes_per_reader'] / 1e6 for r in records]
    ys = [r['critical']['median_ms'] for r in records]
    model = affine(xs, ys)
    models = {p: affine([x for x, r in zip(xs, records) if r['pass_index'] != p],
                        [y for y, r in zip(ys, records) if r['pass_index'] != p]) for p in (0, 1, 2)}
    results = []
    for r in records:
        same_coordinate = [other for other in records
                           if (other['bytes_per_request'], other['pairs'], other['inflight']) ==
                              (r['bytes_per_request'], r['pairs'], r['inflight'])]
        peers = [v for other in same_coordinate if other['pass_index'] != r['pass_index']
                 for v in other['critical_trials_ms']]
        point = statistics.median([v for other in same_coordinate for v in other['critical_trials_ms']])
        point_cv = statistics.median(peers)
        estimate = predict(model, r['bytes_per_reader'])
        estimate_cv = predict(models[r['pass_index']], r['bytes_per_reader'])
        point_errors = [relative(point_cv, v) for v in r['critical_trials_ms']]
        fit_errors = [relative(estimate_cv, v) for v in r['critical_trials_ms']]
        results.append(dict(case=r['case'], pass_index=r['pass_index'], prompt_tokens=r['prompt_tokens'],
                            pairs=r['pairs'], inflight=r['inflight'], bytes_per_reader=r['bytes_per_reader'],
                            critical_median_ms=r['critical']['median_ms'], critical_p90_ms=r['critical']['p90_ms'],
                            critical_max_ms=r['critical']['max_ms'], reader_median_ms=r['reader']['median_ms'],
                            reader_p90_ms=r['reader']['p90_ms'], reader_max_ms=r['reader']['max_ms'],
                            bound32_tail_pass=r['critical']['max_ms'] < 18.72496532622958,
                            bound64_tail_pass=r['critical']['max_ms'] < 24.380075962758763,
                            p2_coordinate=r['pairs'] == 8 and r['inflight'] == 1 and r['prompt_tokens'] == 6144,
                            point_pooled_ms=point, point_other_two_passes_ms=point_cv,
                            point_holdout_median_error_pct=relative(point_cv, r['critical']['median_ms']),
                            point_holdout_trial_median_error_pct=statistics.median(point_errors),
                            point_holdout_trial_p90_error_pct=p90(point_errors),
                            point_median_rule_pass=relative(point_cv, r['critical']['median_ms']) <= 5,
                            point_trial_rule_pass=statistics.median(point_errors) <= 5 and p90(point_errors) <= 10,
                            affine_ms=estimate, affine_other_two_passes_ms=estimate_cv,
                            affine_in_sample_median_error_pct=relative(estimate, r['critical']['median_ms']),
                            affine_holdout_median_error_pct=relative(estimate_cv, r['critical']['median_ms']),
                            affine_holdout_trial_median_error_pct=statistics.median(fit_errors),
                            affine_holdout_trial_p90_error_pct=p90(fit_errors),
                            affine_median_rule_pass=relative(estimate_cv, r['critical']['median_ms']) <= 5,
                            affine_trial_rule_pass=statistics.median(fit_errors) <= 5 and p90(fit_errors) <= 10))
    write_csv('case_pass_results.csv', results)
    spec = dict(
        classification='PROPOSED_ONLY_YC_CHOICE_REQUIRED',
        recommendation='AFFINE_MEDIAN_P2_SINGLE_READ',
        model=model, leave_one_pass_out_models=models,
        formula='batch_complete_ms = intercept_ms + slope_ms_per_MB * (requests_in_flight * bytes_per_request / 1e6)',
        fit_target='all 90 case-pass critical-path medians; equal row weight; no removed trials or passes',
        proposed_median_rule='Every leave-one-pass-out prediction must have <=5% relative error against that held-out case-pass median; all fits require positive intercept and slope. Retain all trial tails. This is median qualification, not per-trial p90 fidelity.',
        proposed_tail_check='For P2 exact bytes/pairs/inflight only, every observed completion must be positive and below each subsequent step-start gap. Finite samples are not a universal serving tail bound.',
        domain=dict(hardware='recorded two-node H200/UCX4 probe', packed_stride_bytes=1002240,
                    measured_bytes_per_request=[46103040, 51114240], pairs=[1, 8], inflight=[1, 2, 4, 8, 16]),
        adoption_domain=dict(prompt_tokens=6144, bytes_per_request=46103040, active_pairs=8, inflight=1,
                             cases=['P2-32', 'P2-64'], request_count=128, arrival_trace='unchanged retained QPS1 trace',
                             completion='engine_step', outside_domain='fail qualification; no cross-hardware substitution or extrapolation'),
        proposed_existing_configuration_fields={
            'analytical_kv_cache_transfer_config_network_bandwidth_gbps': model['bandwidth_gbps'],
            'analytical_kv_cache_transfer_config_network_latency_ms': model['intercept_ms'],
        },
        runtime_limit='Existing predictor is bytes-only and cannot enforce concurrency/domain. External preflight/event census must reject out-of-domain cases; generalized table/batch use requires an interface change.',
        proposed_trial_rule='Separately evaluate median<=5%, p90<=10% of absolute relative trial errors per held-out case/pass; NOT_ADOPTED; retain FAILs.',
        candidate_definitions={
            'empirical_bound': dict(definition='Positive observed batch completions strictly below every later polling start; use sample maximum without dividing by inflight.',
                                   rule='Every case/pass/trial must be below the scale-specific earliest step. P2 use admits exact bytes=46103040,pairs=8,inflight=1 only. Full-grid diagnostics remain reported.',
                                   domain='Retained P2 windows and standalone H200 sample only; no guaranteed future serving-tail coverage',
                                   interface_change=False, cost='0 new GPUh; CPU window scan; additional serving-tail evidence needed for universal guarantee'),
            'concurrency_point': dict(definition='Pooled critical-path median for each exact (bytes/request,pairs,inflight); 6143/6144 combine at the same byte coordinate; every original prompt/pass remains separately validated.',
                                     rule='Leave each pass out; train point on other two passes; <=5% error against each held-out case-pass median. Separately report unadopted per-trial median5/p90 10% rule.',
                                     domain='Exact measured coordinates only; fail on every other value, no interpolation. n>1 values measure wait-all batch completion, not individual request completion.',
                                     interface_change=True, cost='0 new GPUh; new predictor context/table interface requires separate source approval. Fresh replays if binding changes or any completion can move'),
            'physical_affine': dict(definition='OLS all90 full-batch medians vs total bytes/reader; positive intercept and slope, no clamp. P2 config uses n=1 only.',
                                   rule='Every two-pass training fit physical; every held-out case-pass median <=5% error; every observed P2-coordinate trial below every possible-post gap. Trial median5/p90 10% separately FAIL.',
                                   domain='Approval restricted to retained P2 exact byte/pair/n=1 coordinates; external census rejects other workloads. No inferred hardware/fabric/unseen-geometry fidelity.',
                                   interface_change=False, cost='0 new GPUh; analysis and external domain check; no fresh replay under proven no-motion. General n>1 use needs interface/source work'),
        },
        all_case_results=results)
    return spec


def replay_windows(name, root, spec, records):
    receipt = load(root / 'receipt.json')
    metrics = Path(receipt['metrics_dir'])
    config = load(metrics / 'config.json')
    assert config['kv_cache_transfer_config']['nixl_read_completion'] == 'engine_step'
    assert config['kv_cache_transfer_config']['enable_compression'] is False
    cl = config['cluster_config']
    dp, tp = cl['decode_replica_config']['attn_dp'], cl['decode_replica_config']['attn_tensor_parallel_size']
    assert tp == 8 and dp == (2 if name.startswith('p2_32') else 4)
    assert cl['decode_replica_config']['device'] == 'h200'
    assert cl['decode_replica_config']['num_pipeline_stages'] == 1
    assert cl['replica_scheduler_config']['max_forwards_in_flight'] == 19
    ledger = [json.loads(line) for line in track(metrics / 'frontier_transfer_ledger.jsonl').read_text().splitlines()]
    transfers = [r for r in ledger if r['transfer_kind'] == 'kv_cache']
    assert len(transfers) == 128 and all(r['status'] == 'completed' for r in transfers)
    assert all(len(r['request_ids']) == 1 and r['target_bound'] for r in transfers)
    # Only arrival rows are inspected; no completion or E2E metric result is used.
    arrivals = []
    with track(metrics / 'metrics_ground_truth.jsonl').open() as stream:
        for line in stream:
            if '"request_arrival"' in line:
                arrivals.append(json.loads(line))
    assert len(arrivals) == 128 and {r['num_prefill_tokens'] for r in arrivals} == {6144}
    occupancy = sweep(transfers)
    assert occupancy['peak_requests'] == occupancy['peak_active_lanes'] == 1
    starts = collections.defaultdict(set)
    formed_rows, zero_device = 0, 0
    # Read only time/role/identity/zero-duration fields; no scope price or E2E metric drives the choice.
    with gzip.open(track(root / 'forwards.jsonl.gz'), 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['role'] == 'decode':
                starts[(row['replica'], row['dp'])].add(row['formed_s'])
                formed_rows += 1
                zero_device += row['end_s'] <= row['start_s']
    starts = {k: sorted(v) for k, v in starts.items()}
    point = next(r for r in spec['all_case_results'] if r['p2_coordinate'])['point_pooled_ms']
    proposed = predict(spec['model'], 46103040)
    input_max = max(r['critical']['max_ms'] for r in records if r['pairs'] == 8 and r['inflight'] == 1 and r['prompt_tokens'] == 6144)
    # Strengthen the empirical check with every n=1 control (both bytes, both pair counts).
    control_max = max(r['critical']['max_ms'] for r in records if r['inflight'] == 1)
    rows, proofs = [], []
    link = config['kv_cache_transfer_config']
    current_bytes = 39521664 if name.endswith('b29') else 46103040
    current_ms = link['network_latency_ms'] + current_bytes * 8 / (link['network_bandwidth_gbps'] * 1e6)
    for transfer in transfers:
        lane = starts[(transfer['target_replica_id'], transfer['target_replica_local_id'])]
        first = bisect.bisect_left(lane, transfer['start_ts_s'])
        last = bisect.bisect_right(lane, transfer['end_ts_s'])
        assert last > first
        for s in lane[first:last]:
            j = bisect.bisect_right(lane, s)
            assert j < len(lane)
            gap = (lane[j] - s) * 1000
            # Every possible post in the envelope, not a claim of direct post observation.
            proofs.append(dict(run=name, transfer_id=transfer['transfer_id'], lane=str((transfer['target_replica_id'], transfer['target_replica_local_id'])),
                               possible_post_s=s, next_start_s=lane[j], next_gap_ms=gap,
                               current_price_ms=current_ms, proposed_price_ms=proposed,
                               exact_point_ms=point, observed_p2_tail_max_ms=input_max,
                               all_n1_control_tail_max_ms=control_max,
                               deterministic_invariance=current_ms < gap and proposed < gap and point < gap,
                               empirical_p2_tail_pass=input_max < gap,
                               empirical_all_n1_tail_pass=control_max < gap))
        rows.append(dict(run=name, transfer_id=transfer['transfer_id'], request_id=transfer['request_ids'][0],
                         lane=str((transfer['target_replica_id'], transfer['target_replica_local_id'])),
                         start_s=transfer['start_ts_s'], end_s=transfer['end_ts_s'], duration_ms=transfer['duration_ms'],
                         first_possible_post_s=lane[first], possible_posts=last - first))
    assert all(p['deterministic_invariance'] and p['empirical_all_n1_tail_pass'] for p in proofs)
    assert zero_device == 0, 'Idle zero-cost wake would invalidate the no-motion proof'
    ordered = sorted(transfers, key=lambda r: r['start_ts_s'])
    separation = min((b['start_ts_s'] - a['end_ts_s']) * 1000 for a, b in zip(ordered, ordered[1:]))
    return dict(name=name, run_root=str(root), source_commit=receipt['source_commit'],
                tp=tp, dp=dp, ep=cl['decode_replica_config']['moe_expert_parallel_size'],
                decode_forward_rows=formed_rows, zero_duration_decode_forwards=zero_device,
                kv_transfers=len(transfers), transfers_per_lane=dict(collections.Counter(str(r['target_replica_local_id']) for r in transfers)),
                full_transfer_occupancy=occupancy,
                physical_reads_per_active_request=tp, max_reads_per_TP_pair=1,
                max_simultaneous_physical_READs=tp, max_simultaneously_reading_pairs=tp,
                concurrency_inference='Each physical READ is a positive subinterval of its transfer-ledger window. Windows do not overlap even across DP lanes; TP8 is implicit, eight paired READs per logical request.',
                measured_grid_covers_replay=True, matched_case='t6144_pairs8_inflight1',
                min_separation_between_transfer_windows_ms=separation,
                max_transfer_window_ms=max(r['duration_ms'] for r in transfers),
                possible_posts_checked=len(proofs), min_subsequent_step_gap_ms=min(p['next_gap_ms'] for p in proofs),
                current_price_ms=current_ms, proposed_price_ms=proposed, exact_point_ms=point,
                observed_p2_tail_max_ms=input_max, all_n1_control_tail_max_ms=control_max,
                deterministic_crossings=0, empirical_p2_crossings=0, empirical_n1_control_crossings=0,
                proof='All prices are strictly positive and below the first later step for every possible post. The same step finds each READ; the same output reports it. No zero-duration decode forward can introduce an idle wire-end wake. By induction through the deterministic DES, changing only the wire price cannot move a READ release or later event.',
                per_transfer_rows=rows, per_possible_post_rows=proofs)


def main():
    # Bind the source and decision artifacts without editing their contents.
    for path in [REPO / 'AGENTS.md', REPO.parent.parent / 'AGENTS.md',
                 TASK / 'requirements.md', TASK / 'plan.md', TASK / 'progress.md', TASK / 'summary.md',
                 QUALIFICATION / 'decisions.md', QUALIFICATION / 'ledger.md',
                 QUALIFICATION / 'packages/nixl6144/README.md',
                 QUALIFICATION / 'nixl6144_run/observation_notes.md',
                 QUALIFICATION / 'nixl6144_run/case_pass_spread.csv',
                 QUALIFICATION / 'nixl6144_run/paired_6143_6144.csv',
                 TASK / 'test_report_2026-11-01_nixl6144_h200.md',
                 TASK / 'test_report_2026-11-01_f3_h200_nixl.md',
                 TASK / 'h3o_operator_costs/two_fix_20261101/nixl_window_check.py',
                 TASK / 'h3o_operator_costs/two_fix_20261101/nixl_window_check.json',
                 TASK / 'h3o_operator_costs/two_fix_20261101/run_replay.py',
                 TASK / 'h3o_operator_costs/a10/a10_driver.py',
                 TASK / 'h3o_operator_costs/a10/a10_gate.py',
                 REPO / 'frontier/profiling/collectives/nixl_kv_read_benchmark.py',
                 REPO / 'frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py',
                 REPO / 'frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py',
                 REPO / 'frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py']:
        track(path)
    for name in ['05-dsv4-flash-family.md','09-predictor-bundle.md',
                 '11-h200-regression-delivery.md','19-comparison-and-acceptance.md']:
        track(TASK / 'tickets' / name)
    records, pooled, integrity = measurement()
    spec = candidates(records, pooled)
    (HERE / 'proposed_input.json').write_text(json.dumps(spec, indent=2) + '\n')
    windows = [replay_windows(name, root, spec, records) for name, root in RUNS.items()]
    write_csv('transfer_windows.csv', [row for w in windows for row in w.pop('per_transfer_rows')])
    write_csv('possible_post_windows.csv', [row for w in windows for row in w.pop('per_possible_post_rows')])
    result = dict(classification='Q2_PROPOSAL_NOT_FORMAL_ADOPTION', generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  integrity=integrity, candidates=spec, replay_windows=windows,
                  environment=dict(python_executable=os.sys.executable, python_version=platform.python_version(), numpy_version=np.__version__,
                                   cwd=str(Path.cwd()), variables={k: os.environ.get(k) for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TMPDIR']}),
                  evidence_hashes=INPUTS,
                  checks=dict(measurement_integrity='PASS', all_cases_passes_reported='PASS', concurrency_census='PASS',
                              leave_one_pass_out_positive_affine='PASS', deterministic_event_proof='PASS',
                              universal_serving_tail_bound='NOT_ESTABLISHED', fresh_CPU_replays='NOT_RUN_NO_COMPLETION_CAN_MOVE'))
    (HERE / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Integrity:', {k:v for k,v in integrity.items() if k != 'prior_fit_diagnosis'})
    print('Affine:', spec['model'], 'point6144:', predict(spec['model'], 46103040))
    for key in ('point_median_rule_pass','point_trial_rule_pass','affine_median_rule_pass','affine_trial_rule_pass','bound32_tail_pass','bound64_tail_pass'):
        rr = spec['all_case_results']
        print(key, sum(r[key] for r in rr), '/', len(rr), 'failures:',[(r['case'],r['pass_index']) for r in rr if not r[key]])
    for w in windows:
        print(w['name'], 'requests/pairs peaks',w['full_transfer_occupancy']['peak_requests'],w['max_simultaneously_reading_pairs'],
              'checked',w['possible_posts_checked'],'min_gap_ms',w['min_subsequent_step_gap_ms'], 'crossings',w['deterministic_crossings'])


if __name__ == '__main__':
    main()
