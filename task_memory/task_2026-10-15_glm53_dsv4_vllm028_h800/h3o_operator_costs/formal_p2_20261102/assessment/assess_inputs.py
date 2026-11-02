"""Read-only input audit for Q4. Writes only beside this file; no simulator run."""
import csv
import datetime
import hashlib
import json
from git_objects import object_data
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
H3O = FORMAL.parent
TASK = H3O.parent
REPO = TASK.parents[1]
Q1 = H3O / 'input_qualification_20261101'
CASES = ('p2_32', 'p2_64')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ref(value, base=Q1):
    path, marker, pointer = value.partition('#')
    return str((base / path).resolve()) + (marker + pointer if marker else '')


def current_status():
    ledger = load(Q1 / 'ledger.json')
    out = []
    for i, entry in enumerate(ledger['inputs'], 1):
        row = dict(ledger_row=i, **entry)
        row['frozen_status'] = entry['status']
        row['frozen_missing_evidence'] = entry['missing_evidence']
        row['evidence'] = [ref(x) for x in entry['evidence']]
        row['decision'] = 'Frozen Q1 disposition under the recorded criterion; unchanged.'
        row['required_for_final_pass'] = {c: entry['status'] == 'QUALIFIED' for c in CASES}
        row['closure_proposal'] = None
        if i == 1:
            row.update(status='QUALIFIED', decision='D-P2-Q1-CHOICES attention A and original table A; exact-point supplement for all 48 omissions, both independent passes separately.',
                       current_result='All 96 comparisons finite and <=3.663151%; 240 exact layer queries, zero interpolation; original KFold5/seed1234 NOT_EVALUABLE retained.')
            row['evidence'] += [str(TASK / 'requirements.md') + ':170', str(Q1 / 'evidence/audit.json') + '#/independent_attention']
            row['required_for_final_pass'] = {c: True for c in CASES}
        elif i == 17:
            row.update(status='QUALIFIED', decision='D-P2-PAYLOAD-C route C, all four checks accepted.',
                       current_result='Source-derived, metric-invariant: 46 pages x 1,002,240 B = 46,103,040 B per TP pair. Not observed native descriptor arrays.')
            row['evidence'] += [str(TASK / 'requirements.md') + ':192', str(TASK / 'test_report_2026-11-02_q3_payload_bound.md'), str(Q1 / 'payload_q3/results.json')]
            row['required_for_final_pass'] = {c: True for c in CASES}
        elif i == 18:
            row.update(status='OPEN', decision='D-P2-Q2-NIXL-TIMING supersedes this H800 cost as a formal P2 input; it does not qualify the H800 pair on H200.',
                       current_result='Not consumed by the formal runs. Effective H200 replacement is separately QUALIFIED at t6144_pairs8_inflight1.')
            row['required_for_final_pass'] = {c: False for c in CASES}
            row['final_pass_reason'] = 'Ticket19:134-140 requires own-hardware costs, not qualification of a retired cross-hardware candidate. The ledger limitation prohibits using warm numerical PASS to waive the rule. The formal overrides remove this cost.'
            row['closure_proposal'] = {'proposal': 'Retain OPEN and mark retired for H200; no exception or new H800 qualification needed for these cases.', 'cost': '0 GPUh; records only.'}
            row['evidence'] += [str(TASK / 'requirements.md') + ':181']
        elif i == 19:
            row.update(status='FAIL', decision='The NIXL02 affine candidate failed the physical-intercept criterion. D-P2-Q2-NIXL-TIMING adopts the different nixl6144 H200 fit and closes the prospective timing choice.',
                       current_result='Retain NIXL02 -46.6769198402 us intercept and p90 residual 29.175907% FAIL; raw measurement integrity is qualified. This candidate is not consumed.')
            row['required_for_final_pass'] = {c: False for c in CASES}
            row['evidence'] += [str(TASK / 'requirements.md') + ':181', str(TASK / 'test_report_2026-11-01_q2_h200_nixl_timing.md')]
        elif i == 20:
            row['current_result'] = 'D-P2-Q2-NIXL-TIMING extends the retained event proof to the adopted H200 price 0.9684023419505275 ms; no arbitrary timing independence claim.'
            row['evidence'] += [str(TASK / 'test_report_2026-11-01_q2_h200_nixl_timing.md')]
        elif i in (23, 24):
            row['required_for_final_pass'] = {c: False for c in CASES}
            row['decision'] = 'Spec10.4 measured-capacity isolation and the Q1 limitation defer this unverified autonomous/pressure-memory condition; no new criterion.'
            row['final_pass_reason'] = 'Ticket19:140 requires effective limits and actual memory budget. Rows21/22 provide case-bound measured pools and native startup budget; it does not require autonomous planner equality or independent disjoint phase peaks. The row limitation expressly excludes this from measured-capacity isolation.'
            row['closure_proposal'] = {'proposal': 'Keep OPEN. A separately authorized memory-pressure/phase-instrumentation study would be needed to close the recorded autonomous claim.', 'cost': '0 GPUh for this assessment; future design and GPU budget unknown, no ready package.'}
        elif i == 26:
            row['status'] = 'QUALIFIED'
            row['decision'] = 'Reassess the same coverage criterion after D-P2-Q1-CHOICES, D-P2-Q2-NIXL-TIMING, and D-P2-PAYLOAD-C; no criterion change.'
            row['current_result'] = 'PASS for retained P2-32/P2-64 coordinates: attention omissions have the adopted independent-pass disposition; seven same-H200 tables are pinned; measured exact collective coverage passes; H200 timing and payload are qualified. Formal runtime receipt/sequence binding is checked separately.'
            row['required_for_final_pass'] = {c: True for c in CASES}
            row['evidence'] += [str(TASK / 'requirements.md') + ':170', str(TASK / 'requirements.md') + ':181', str(TASK / 'requirements.md') + ':192']
        elif i == 27:
            row['required_for_final_pass'] = {'p2_32': False, 'p2_64': True}
            row['final_pass_reason'] = 'Ticket19:149-165 and Acceptance:184-186 explicitly require a valid pre-launch 30% receipt for formal64. The ledger limitation preserves original clean64/basic authorization but rejects retrospective promotion. Not a final32 prerequisite.'
            row['closure_proposal'] = {'proposal': 'YC may waive the launch-order condition for the existing reference, or Claude can issue the qualified32 receipt before a new formal native64 campaign.', 'cost': 'Waiver: 0 GPUh; new campaign: historical 24.322222 container GPUh proxy, 64 wholly preemptible H200 step_main, 64 GPUh timeout ceiling; queue/observer/coverage differences unknown.'}
        if i in (1,17,26): row['missing_evidence'] = ''
        out.append(row)
    status = dict(schema_version=1, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  frozen_ledger=record(Q1 / 'ledger.json'), frozen_counts=dict(Counter(x['frozen_status'] for x in out)),
                  current_counts=dict(Counter(x['status'] for x in out)), inputs=out,
                  coverage_verdict='PASS', coverage_scope='Retained P2 exact coordinates only; formal consumed-input and terminal coverage audit required separately.',
                  h200_replacement=dict(status='QUALIFIED', decision='D-P2-Q2-NIXL-TIMING', input=record(Q1 / 'nixl_timing_q2/proposed_input.json'),
                                        bandwidth_gbps=390.39349928254796, latency_ms=0.0236521842818856,
                                        domain='t6144_pairs8_inflight1, 46,103,040 B per TP pair, engine_step, compression off, P2 H200 configs'))
    save('input_status.json', status)
    return status


def tree_entries(oid):
    raw = object_data(oid)
    pos, result = 0, {}
    while pos < len(raw):
        end = raw.index(b'\0', pos)
        mode, name = raw[pos:end].split(b' ', 1)
        result[name.decode()] = (mode.decode(), raw[end + 1:end + 21].hex())
        pos = end + 21
    return result


def blob_at(commit, path):
    oid = object_data(commit).splitlines()[0].split()[1].decode()
    for part in Path(path).parts:
        oid = tree_entries(oid)[part][1]
    return oid


def changes(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append({'path': path + '/' + key, 'old': a.get(key), 'new': b.get(key)})
            else:
                result += changes(a[key], b[key], path + '/' + key)
        return result
    return [] if a == b else [{'path': path, 'old': a, 'new': b}]


def normalized_config(config, root):
    if isinstance(config, dict):
        return {k: normalized_config(v, root) for k, v in config.items()}
    if isinstance(config, list):
        return [normalized_config(v, root) for v in config]
    if isinstance(config, str):
        return config.replace(str(root), '<run_root>')
    return config


def bindings():
    ledger = load(Q1 / 'ledger.json')
    audit = load(Q1 / 'evidence/audit.json')
    lookup = {x.get('path'): (i, x) for i, x in enumerate(ledger['inputs'], 1) if x.get('path')}
    result = {}
    for case in CASES:
        command = load(FORMAL / (case + '_command.json'))
        root = Path(command['run_root'])
        receipt = load(root / 'receipt.json')
        old = load(H3O / f'two_fix_20261101/{case}_command.json')
        old_root = Path(old['run_root'])
        old_receipt = load(old_root / 'receipt.json')
        frozen = next(x for x in audit['runs'] if case in x['root'])
        items, findings = [], []
        assert len(receipt['profiling_tables'])==7, (case,'Consumed table count',receipt['profiling_tables'])
        assert receipt['source_commit']==command['source_commit'], (case,'Driver and supervisor source commits differ')
        for consumed in receipt['profiling_tables']:
            actual = record(consumed['path'])
            matched = lookup.get(actual['path'])
            match = bool(matched and actual['sha256'] == matched[1]['sha256'] == consumed['sha256'] and consumed['rows'] == matched[1]['rows'])
            items.append(dict(kind='profiling_table', ledger_row=matched[0] if matched else None, receipt=consumed, actual=actual, status='PASS' if match else 'FAIL'))
            if not match:
                findings.append(dict(kind='CONSUMED_TABLE_UNBOUND', path=actual['path']))
        for kind, path, pinned, row in [('flat_config', receipt['input'], frozen['input_sha256'], 10),
                                       ('trace', receipt['trace']['path'], frozen['trace']['sha256'], 8)]:
            actual = record(path)
            match = actual['sha256'] == pinned == (receipt['input_sha256'] if kind == 'flat_config' else receipt['trace']['sha256'])
            items.append(dict(kind=kind, ledger_row=row, actual=actual, ledger_sha256=pinned, evidence=str(Q1 / 'evidence/audit.json') + '#/runs', status='PASS' if match else 'FAIL'))
        trace_root = Path(receipt['trace']['path']).parent
        native_manifest_path = TASK / 'h200_mainline/diagnostic/inputs' / ('p2_32r' if case == 'p2_32' else 'p2_64r') / 'run_manifest.json'
        native_manifest = load(native_manifest_path)
        population = []
        for name in ['trace.csv', 'manifest.json', 'request_ids.json', 'warmup_pool.csv']:
            actual = record(trace_root / name)
            pin = native_manifest['inputs_sha256']['inputs/trace/' + name]
            population.append(dict(**actual, native_manifest_sha256=pin, matches=actual['sha256'] == pin))
        with Path(receipt['trace']['path']).open(newline='') as stream:
            trace_rows = list(csv.DictReader(stream))
        ids = load(trace_root / 'request_ids.json')['formal_request_ids']
        manifest = load(trace_root / 'manifest.json')
        exact_population = len(trace_rows) == len(ids) == len(set(ids)) == 128 and all(
            int(r['num_prefill_tokens']) == 6144 and int(r['num_decode_tokens']) == 1024
            and float(r['arrived_at']) == i and ids[i] == load(trace_root / 'request_ids.json')['rows'][i]['request_id']
            for i, r in enumerate(trace_rows))
        items.append(dict(kind='request_population', ledger_row=8, files=population, native_manifest=record(native_manifest_path),
                          rows=len(trace_rows), columns=list(trace_rows[0]), first_row=trace_rows[0], last_row=trace_rows[-1],
                          exact_lengths_ids_arrivals=exact_population, token_totals=manifest['lengths'],
                          status='PASS' if exact_population and all(x['matches'] for x in population) else 'FAIL'))
        if receipt.get('status') == 'FINISHED':
            for cached in receipt.get('cache_files', []):
                if isinstance(cached, dict) and cached.get('sha256'):
                    cache_path=Path(cached['path']) if cached.get('path') else Path(receipt['cache_dir'])/cached['name']
                    actual=record(cache_path)
                    items.append(dict(kind='generated_cache',ledger_row=15,actual=actual,receipt_sha256=cached['sha256'],status='PASS' if actual['sha256']==cached['sha256'] else 'FAIL'))
        config_path = Path(receipt['metrics_dir']) / 'config.json'
        config = load(config_path)
        previous_config = load(Path(old_receipt['metrics_dir']) / 'config.json')
        delta = changes(normalized_config(previous_config, old_root), normalized_config(config, root))
        allowed = {'/kv_cache_transfer_config/network_bandwidth_gbps', '/kv_cache_transfer_config/network_latency_ms'}
        unexpected = [x for x in delta if x['path'] not in allowed]
        items.append(dict(kind='resolved_config', ledger_rows=[10,11,12,13,14,15,16,20,21], actual=record(config_path), frozen_config_sha256=frozen['config_sha256'],
                          normalized_changes=delta, status='PASS' if not unexpected and {x['path'] for x in delta} == allowed else 'FAIL'))
        findings.append(dict(kind='EXPECTED_RESOLVED_CONFIG_HASH_CHANGE', blocking=False, evidence='Frozen Q1 config includes old root and H800 pair. New root and exactly two adopted H200 timing fields explain the new byte hash.', actual=record(config_path), frozen_sha256=frozen['config_sha256']))
        timing = record(command['timing_input'])
        items.append(dict(kind='h200_timing', ledger_successor_of=[18,19], actual=timing, command_sha256=command['timing_input_sha256'],
                          qualified_by=str(TASK / 'requirements.md') + ':181', value=config['kv_cache_transfer_config'],
                          status='PASS' if timing['sha256'] == command['timing_input_sha256'] and config['kv_cache_transfer_config']['network_bandwidth_gbps'] == 390.39349928254796 and config['kv_cache_transfer_config']['network_latency_ms'] == 0.0236521842818856 else 'FAIL'))
        findings.append(dict(kind='EXPECTED_INPUT_ADDED_AFTER_FROZEN_LEDGER', blocking=False, input=timing, evidence='D-P2-Q2-NIXL-TIMING explicitly qualifies this successor. Frozen ledger was not edited.'))
        source_paths=[H3O/name for name in ['a10/a10_driver.py','a10/a10_common.py','a10/a10_observer.py','a10/a10_gate.py']]
        auxiliary_paths=[TASK/'h4a_p0_diagnostic/h4a4_driver.py',TASK/'h4a_p0_diagnostic/h4a4_gate.py',REPO/'tests/fixtures/dsv4_native_dp_sync_rows.json',REPO/'tests/analysis/e2e_warm_reading.py',REPO/'tests/comparison/e2e_gate/run_gate.py',REPO/'tests/comparison/e2e_gate/gate.py']
        for path in source_paths+auxiliary_paths:
            actual = record(path)
            blobs = {label: blob_at(commit, path.relative_to(REPO)) for label, commit in [('formal',command['source_commit']),('two_fix',old['source_commit'])]}
            raw = path.read_bytes()
            actual_oid = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if path==H3O/'a10/a10_driver.py':
                assert actual['sha256']==command['driver_sha256'],(case,'Supervisor driver hash mismatch')
            items.append(dict(kind='driver_or_dependency', ledger_rows=[11,14,15], actual=actual, source_blob_oids=blobs, source_content_sha256={'formal': actual['sha256'], 'two_fix': actual['sha256']} if actual_oid == blobs['formal'] == blobs['two_fix'] else {}, status='PASS' if actual_oid == blobs['formal'] == blobs['two_fix'] else 'FAIL'))
        findings.append(dict(kind='AUXILIARY_INPUTS_BOUND_BY_LEDGER_OWNER',blocking=False, paths=[str(p) for p in auxiliary_paths],evidence='These consumed helper/fixture paths have no separate path/hash row in the frozen ledger. Rows11/14/15 bind runtime/predictor/source generation; each actual SHA256 and formal/two-fix source blob is checked. No cost is added.'))
        items.append(dict(kind='formal_runner', ledger_successor_of=[14,15], actual=record(FORMAL / 'run_formal_replay.py'), source_blob_oid=blob_at(command['source_commit'], (FORMAL / 'run_formal_replay.py').relative_to(REPO)), decision='Assignment selects runner committed at abd1e141; not in frozen ledger.'))
        raw = (FORMAL / 'run_formal_replay.py').read_bytes()
        items[-1]['status'] = 'PASS' if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == items[-1]['source_blob_oid'] else 'FAIL'
        assert Path(receipt['cache_dir']).resolve().is_relative_to(root.resolve()), (case,'Cache is outside the fresh run root')
        items[-1]['fresh_cache_proof']={'runner_guard':str(FORMAL/'run_formal_replay.py')+':92','rule':'Matching committed runner rejects any existing run/gate root before writing the command record or starting the driver. The recorded cache directory is below that new run root.','cache_dir':str(Path(receipt['cache_dir']).resolve()),'status':'PASS'}
        findings.append(dict(kind='EXPECTED_RUNNER_ADDED_AFTER_FROZEN_LEDGER', blocking=items[-1]['status'] != 'PASS', input=items[-1]['actual'], evidence='Assignment explicitly selects the formal runner committed at abd1e141; the source blob binds its exact contents. It is an authorized driver successor, absent from frozen Q1.'))
        source_trees = {}
        for label, commit in [('formal', command['source_commit']), ('two_fix', old['source_commit'])]:
            tree = tree_entries(object_data(commit).splitlines()[0].split()[1].decode())
            source_trees[label] = {'commit': commit, 'commit_content_sha256': hashlib.sha256(object_data(commit)).hexdigest(), 'frontier_tree': tree['frontier'][1], 'data_tree': tree['data'][1]}
        source_same = all(source_trees['formal'][k] == source_trees['two_fix'][k] for k in ['frontier_tree','data_tree'])
        items.append(dict(kind='source_generation', ledger_rows=[14,15], evidence=str(Q1 / 'evidence/audit.json') + '#/runs', trees=source_trees, clean_at_launch=command['source_dirty_paths'] == receipt['source_dirty_paths'] == [], status='PASS' if source_same and command['source_dirty_paths'] == receipt['source_dirty_paths'] == [] else 'FAIL'))
        findings.append(dict(kind='EXPECTED_SOURCE_COMMIT_CHANGE', blocking=not source_same, evidence='Assignment explicitly selects abd1e141; formal and qualified two-fix frontier/data trees are identical. Driver/dependency SHA256s also match.', generations=source_trees))
        for item in items:
            if item['status'] != 'PASS':
                findings.append(dict(kind='INPUT_BINDING_FAILURE', blocking=True, input_kind=item['kind'], evidence=item))
        inactive = sorted(set(sum(receipt['a1']['resolved_inputs'].values(), [])) - {x['path'] for x in receipt['profiling_tables']})
        result[case] = dict(command_record=record(FORMAL / (case + '_command.json')), terminal_status=command['status'], receipt=record(root / 'receipt.json'),
                            inputs=items, findings=findings, unexpected_config_changes=unexpected, a1=receipt['a1'],
                            configured_inactive_paths=inactive, generated_cache_count=sum(x['kind']=='generated_cache' for x in items),cache_lock_entries=[x for x in receipt.get('cache_files',[]) if not x.get('sha256')],cache_lock_note='Synchronization artifacts are not consumed model inputs and have no hash in the driver receipt.', inactive_reason='Ledger14 explicitly qualifies host zero/disabled alternatives; production family resolution only consumes the seven listed tables; PP1 and measured CC make legacy PP/all_reduce/send_recv inactive.',
                            kv_admission=receipt.get('kv_admission'), checks_pass=all(x['status'] == 'PASS' for x in items) and not unexpected,
                            resolved_config=config)
    save('binding.json', result)
    return result


if __name__ == '__main__':
    status = current_status()
    binding = bindings()
    print('current_counts', status['current_counts'], 'coverage', status['coverage_verdict'])
    for case, data in binding.items():
        print(case, 'binding', data['checks_pass'], 'status', data['terminal_status'], 'findings', len(data['findings']))

    assert all(data['checks_pass'] for data in binding.values()), 'Consumed-input binding failures; inspect binding.json before reporting verification PASS.'
