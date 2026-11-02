"""Post-run pair coverage, always a soft gate (process exit zero).

coverage.py --run ARTIFACTS --output NEW_JSON --openers NEW_JSON
An unverified event derivation never produces an empty exclusion map. Legacy
controls lack intended-pair fields and explicitly report that missing evidence.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent / 'warm_target_20261101/identify_openers.py'
R4 = HERE.parents[2] / 'task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ownership(run, method):
    owners, paths, lanes = {}, [], defaultdict(set)
    for path in sorted(run.glob('nodes/*/serving/rank0.json')):
        record = json.loads(path.read_text())
        role, lane = record['role'], record['inventory']['request_owner']
        require(role in ('prefill', 'decode') and lane not in lanes[role]
                and record['counters']['dropped_rows'] == 0, f'Invalid serving owner: {path}')
        lanes[role].add(lane)
        paths.append(path)
        for group in record['counters']['request_ids']:
            for engine_request in group['request_ids']:
                key = (role, method.key(engine_request))
                require(key not in owners, f'Duplicate serving owner: {key}')
                owners[key] = lane
    require(set(lanes) == {'prefill', 'decode'} and all(values == set(range(len(values)))
                                                    for values in lanes.values()), 'Missing/nonconsecutive DP lanes')
    return owners, lanes, paths


def idle_evidence(run, phases, client, lanes):
    """Reinspect samples and order, including service-bound cold-start counters."""
    # R4 inherits this exact R2 predicate. Reuse it without changing its code.
    checker = load_module('coverage_r4', R4)
    counts = defaultdict(dict)
    for repeat in phases['repetitions']:
        for phase in ('warmup', 'formal'):
            counts[phase][repeat['repeat']] = sum(r['repeat'] == repeat['repeat'] and r['segment'] == phase
                                                for r in client.values())
    require(all(len(set(values.values())) == 1 for values in counts.values()), 'Phase counts differ between repeats')
    workload = dict(repetitions=dict(repeat_indices=[r['repeat'] for r in phases['repetitions']]),
                    selection=dict(num_formal_requests=next(iter(counts['formal'].values()))),
                    warmup=dict(num_requests_per_repetition=next(iter(counts['warmup'].values()))))
    cold = phases['idle_boundaries'][0]
    dedicated = None
    if cold.get('proof') == 'dedicated_services_ready_client_zero_native_counters_zero':
        # Validate service receipts against the run manifest, not just the stored label.
        manifest = json.loads((run / 'run_manifest.json').read_text())
        worker = load_module('coverage_worker', Path(manifest['harness_worktree']) /
                             'tests/comparison/calibration/pd_replay.py')
        services = cold['fresh_services']['services']
        urls = {role: services[role]['url'] for role in ('prefill', 'decode')}
        dedicated = worker.dedicated_services(run, urls, len(lanes['decode']))
    require(lanes['prefill'] == lanes['decode'], 'R4 idle contract requires equal role DP sizes')
    passed, evidence = checker.native_idle_evidence(phases, list(client.values()), workload,
                                                   len(lanes['decode']), dedicated)
    return dict(verified=passed, evidence=evidence,
                boundaries=[dict(boundary=b['boundary'], verified=b['verified']) for b in phases['idle_boundaries']])


def analyze(run):
    run = Path(run).resolve()
    method = load_module('retained_openers_method', METHOD)
    derivation = method.identify(run)
    client_rows, proxy_rows = read_rows(run / 'run/client_requests.jsonl'), read_rows(run / 'run/proxy_requests.jsonl')
    client = {r['request_id']: r for r in client_rows}
    proxy = {r['request_id']: r for r in proxy_rows}
    require(len(client) == len(client_rows) and len(proxy) == len(proxy_rows), 'Duplicate client/proxy identities')
    owners, lanes, owner_paths = ownership(run, method)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    require(manifest['resources']['replica'] == len(owner_paths),
            'Serving lane inventories do not cover every configured P2 node (one local DP lane per node)')
    require(all({rid for role, rid in owners if role == name} == set(client)
                for name in ('prefill', 'decode')), 'Incomplete P or D request ownership')
    actual, remotes = {}, {}
    for rid, record in proxy.items():
        p, d = owners['prefill', rid], owners['decode', rid]
        remote = record['kv_transfer_params']['remote_engine_id']
        require(int(remote.rsplit('_dp', 1)[1]) == p, f'Remote engine/P inventory mismatch: {rid}')
        require(remotes.setdefault(p, remote) == remote, 'A P lane changed remote engine identity')
        actual[rid] = dict(prefill=p, decode=d)
    require(len(set(remotes.values())) == len(lanes['prefill']), 'Remote engines do not identify all P lanes')
    phases = json.loads((run / 'run/canonical_repetitions.json').read_text())
    idle = idle_evidence(run, phases, client, lanes)
    expected = {(p, d) for p in lanes['prefill'] for d in lanes['decode']}
    first = {(actual[r['request_id']]['prefill'], r['decode_lane']): r['request_id']
             for r in derivation['first_contact_pairs']}
    require(set(first) == expected, 'First-contact domain incomplete')
    # A log does not print the remote engine. Keep warmup cohorts' candidate
    # lists; require a full matching to first-contact requests in that window.
    # We report ambiguity rather than fabricate a unique warmup event binding.
    event_matches = []
    for event in derivation['events']:
        possible = [rid for rid in event['candidate_request_ids'] if rid in first.values()]
        event_matches.append(dict(decode_lane=event['decode_lane'], phase=event['phase'], repeat=event['repeat'],
                                  tp_records=event['tp_records'], candidate_first_contacts=possible))
    for lane in lanes['decode']:
        events = [e for e in event_matches if e['decode_lane'] == lane]
        require(len(events) == len(lanes['prefill']),
                f'Missing TP compatibility cohorts on D{lane}: {len(events)} for {len(lanes["prefill"])} pairs')
        matches = []
        def visit(index, chosen):
            if len(matches) >= 2:
                return
            if index == len(events):
                matches.append(chosen.copy())
                return
            for rid in events[index]['candidate_first_contacts']:
                if rid not in chosen:
                    visit(index + 1, chosen + [rid])
        visit(0, [])
        require(matches, f'No complete TP-cohort/first-contact matching on D{lane}')
        for index, event in enumerate(events):
            event['binding'] = 'UNIQUE_MATCHING' if len(matches) == 1 else 'AMBIGUOUS_WITHIN_WARMUP_WINDOW'
            event['bound_request_id'] = matches[0][index] if len(matches) == 1 else None
        # All candidate matchings remain within the event's recorded phase.
        require(all(client[rid]['segment'] == event['phase'] and client[rid]['repeat'] == event['repeat']
                    for event in events for rid in event['candidate_first_contacts']), 'Cross-phase cohort candidates')
    contact_records = []
    for pair, rid in sorted(first.items()):
        row = client[rid]
        windows = [w for w in derivation['windows'] if w['repeat'] == row['repeat'] and w['phase'] == row['segment']]
        [window] = windows
        require(window['origin_wall'] <= proxy[rid]['prefill_response_time'] < proxy[rid]['decode_first_byte_time']
                <= window['end_wall'], f'First contact outside its phase: {rid}')
        contact_records.append(dict(pair=list(pair), request_id=rid, repeat=row['repeat'], phase=row['segment'],
                                    window=window, remote_engine_id=proxy[rid]['kv_transfer_params']['remote_engine_id']))
    repeats, openers = {}, {}
    tag = manifest['run_tag']
    for repeat in phases['repetitions']:
        index = repeat['repeat']
        warmups = [r for r in client.values() if r['repeat'] == index and r['segment'] == 'warmup']
        formal = [r for r in client.values() if r['repeat'] == index and r['segment'] == 'formal']
        reached = {(actual[r['request_id']]['prefill'], actual[r['request_id']]['decode']) for r in warmups}
        target_records = []
        for row in warmups:
            rid = row['request_id']
            target_records.append(dict(request_id=rid, intended_pair=row.get('intended_pair'),
                                       proxy_intended_pair=proxy[rid].get('intended_pair'), observed_pair=actual[rid],
                                       matches=('intended_pair' in row and row['intended_pair'] ==
                                                proxy[rid].get('intended_pair') == actual[rid])))
        intended = all(row['matches'] for row in target_records)
        no_formal_target = all('intended_pair' not in r and 'intended_pair' not in proxy[r['request_id']] for r in formal)
        first_before = [r for r in contact_records if r['phase'] == 'warmup' and
                        r['window']['end_wall'] < next(w['origin_wall'] for w in derivation['windows']
                                                      if w['repeat'] == index and w['phase'] == 'formal')]
        formal_ids = [rid for rid in derivation['formal_openers'] if client[rid]['repeat'] == index]
        covered = (len(first_before) == len(expected) and not formal_ids and intended and reached == expected
                   and no_formal_target and idle['verified'])
        repeats[f'repeat{index:02d}'] = dict(status='COVERED' if covered else 'NOT_COVERED',
            first_contacts_warmed_before_formal=[r['pair'] for r in first_before],
            pairs_reached_in_this_warmup=[list(pair) for pair in sorted(reached)],
            missing_warmup_pairs=[list(pair) for pair in sorted(expected - reached)],
            new_contacts_in_this_warmup=[r for r in contact_records if r['repeat'] == index and r['phase'] == 'warmup'],
            formal_openers=formal_ids, intended_pairs_verified=intended, warmup_requests=target_records,
            formal_requests_untargeted=no_formal_target, idle_verified=idle['verified'])
        openers[f'vllm/{tag}-repeat{index:02d}'] = [client[rid]['template_request_id'] for rid in formal_ids]
    return dict(status='COVERED' if all(r['status'] == 'COVERED' for r in repeats.values()) else 'NOT_COVERED',
                evidence_verified=True, soft_gate=True, run=str(run), expected_pairs=[list(p) for p in sorted(expected)],
                repetitions=repeats, first_contacts=contact_records, idle=idle, cohorts=event_matches,
                event_derivation=derivation, openers=openers,
                limitations=['Compatibility logs have one-second resolution; ambiguous warmup events retain candidates.',
                             'Later repetitions reuse warmed connections; no new handshake is required in each repeat.',
                             'Legacy untargeted controls cannot prove intended-pair matching.'],
                sources=derivation['sources'] + [dict(path=str(p), sha256=method.hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in [METHOD, Path(__file__), *owner_paths]])


def safe_analyze(run):
    try:
        return analyze(run)
    except (Exception, SystemExit) as error:
        return dict(status='NOT_COVERED', evidence_verified=False, soft_gate=True,
                    run=str(run), error=f'{type(error).__name__}: {error}', openers=None)


def save(path, document):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + '\n')


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--openers', type=Path, required=True)
        args = parser.parse_args()
        result = safe_analyze(args.run)
        save(args.output, result)
        if result['openers'] is not None:
            save(args.openers, result['openers'])
        print(result['status'], 'evidence_verified=' + str(result['evidence_verified']))
    except (Exception, SystemExit) as error:
        print(f'PAIR_COVERAGE_UNVERIFIED {type(error).__name__}: {error}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
