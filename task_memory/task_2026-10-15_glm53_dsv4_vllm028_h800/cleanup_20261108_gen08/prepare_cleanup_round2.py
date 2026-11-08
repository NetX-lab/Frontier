#!/usr/bin/env python3
"""Round-two preparation using the preserved first-run inventory helpers."""
import ast
import json
from pathlib import Path

from prepare_cleanup import (RECORD, TASK, ROOT, PREVIOUS, SNAPSHOTS, HOST_SHA,
                             now, require, write, load, metadata, hash_row, walk, totals)

ASSIGNMENT_SHA = 'a2c5eeec8073e8939f9a594dfa253939a0dbeb182b1c9ad465ad95f59ee15e60'
EXPECTED = {'files': 4, 'bytes': 300753869, 'allocated_bytes': 300761088}
TABLE = {
    'protected_hash_before.json': (75188389, 75190272, '8cb0e44aabf312785b149e59e305f05f8c4c21b8560d20c056b7db7d2faccda8'),
    'protected_hash_after.json': (75188388, 75190272, '55c8d5d65e499451186a30af078a2301077b18735e35b2425baa1c73b091c1c3'),
    'protected_hash_before_final.json': (75188546, 75190272, 'cd50532f06fd0d027b4bee62459e7f5bf0d5b0d806c6733ed191783ba5915b7b'),
    'protected_hash_after_final.json': (75188546, 75190272, '9cf0b5b8c4cf1afa8094e26b4d3f7da758e45994bad8b1eef8dfabc756c7f265'),
}
NEW_SCRIPTS = {'prepare_cleanup_round2.py', 'execute_cleanup_round2.py', 'verify_cleanup_round2.py'}
HOST_LIMIT = ('Lead-supplied 2026-11-08 host scan, 289 processes; no executor process scan. '
              '127 entries had unreadable links; lead reports 14/15 sampled entries had exited, '
              'the other was pid 1 /sbin/init. Raw roots name before/after but omit the two *_final '
              'snapshots. assignment_2.md explicitly accepts this scan for all four because '
              'their hashes have not changed since 2026-11-07. The follow-up sample and '
              'acceptance of omitted roots are lead-supplied, not findings of the raw scan.')


def host_check():
    path = RECORD / 'cleanup_scope_processes.json'
    require(hash_row(path)['sha256'] == HOST_SHA, 'Supplied process evidence changed')
    obj = load(path)
    require(obj['clear_for_deletion'] is True and obj['processes_scanned'] == 289
            and not obj['referencing_pids'] and not obj['referencing_processes'], 'Host evidence is not clear')
    return {'source': str(path), 'sha256': HOST_SHA, 'observed_utc': obj['observed_utc'],
            'processes_scanned': 289, 'clear_for_deletion': True, 'limitation': HOST_LIMIT}


def main():
    started = now()
    require(hash_row(RECORD/'assignment_2.md')['sha256'] == ASSIGNMENT_SHA, 'Round-two assignment changed')
    require(not any((RECORD/n).exists() for n in ('scoped_inventory_before_round2.json',
            'deletion_manifest_round2.json', 'deletion_journal_round2.jsonl',
            'deletion_receipt_round2.json', 'verification_round2.json')), 'Round two already started')
    for name in NEW_SCRIPTS:
        ast.parse((RECORD/name).read_text(), filename=str(RECORD/name))
    process = host_check()
    initial = load(RECORD/'deletion_manifest_final.json')
    old = load(RECORD/'scoped_inventory_before_final.json')
    require(hash_row(RECORD/'scoped_inventory_before_final.json')['sha256'] == initial['before_inventory_sha256'],
            'First-run baseline changed')
    receipt = load(RECORD/'deletion_receipt.json')
    require(hash_row(RECORD/'deletion_manifest_final.json')['sha256'] == receipt['manifest_sha256'],
            'First-run manifest changed')
    require(hash_row(RECORD/'deletion_journal.jsonl')['sha256'] == receipt['journal_sha256'],
            'First-run journal changed')
    require(receipt['status'] == 'PARTIAL' and receipt['declined'] == [{
        'group': 'previous_snapshots', 'paths': list(map(str, SNAPSHOTS)),
        'reasons': ['Observed counts/allocation differ from assignment table']}], 'Unexpected first-run decline')
    original = {e['path']: e for e in old['entries']
                if Path(e['path']).parent == PREVIOUS or Path(e['path']) == PREVIOUS}
    paths = sorted(walk(PREVIOUS))
    require(set(map(str, paths)) == set(original) and len(paths) == 30,
            'Previous-cleanup parent membership differs from 29 first-run files')
    rows = [hash_row(q) for q in paths]
    bypath = {e['path']: e for e in rows}
    targets = set(map(str, SNAPSHOTS))
    survivors = [e for e in rows if e['kind'] == 'file' and e['path'] not in targets]
    require(len(survivors) == 25 and all(e == original[e['path']] for e in survivors),
            'One or more of 25 surviving files differs from its first-run hash/metadata')
    lead = load(PREVIOUS/'lead_acceptance.json')
    require(lead['status'] == 'ACCEPTED', 'Surviving acceptance evidence is not accepted')
    errors = []
    entries = [dict(bypath[str(q)], group='previous_snapshots') for q in SNAPSHOTS]
    for e in entries:
        name = Path(e['path']).name
        expected = TABLE[name]
        saved = lead['snapshot_files_not_committed'][name]
        if (e['bytes'], e['allocated_bytes'], e['sha256']) != expected:
            errors.append({'path': e['path'], 'reason': 'Differs from corrected table',
                           'expected': {'bytes': expected[0], 'allocated_bytes': expected[1], 'sha256': expected[2]},
                           'actual': {k: e[k] for k in ('bytes', 'allocated_bytes', 'sha256')}})
        if e['bytes'] != saved['bytes'] or e['sha256'] != saved['sha256']:
            errors.append({'path': e['path'], 'reason': 'Differs from surviving lead_acceptance.json'})
        if e['signature']['nlink'] != 1 or {k:v for k,v in e.items() if k != 'group'} != original[e['path']]:
            errors.append({'path': e['path'], 'reason': 'File identity/metadata differs from first run or has hard links'})
    observed = totals(entries)
    if observed != EXPECTED:
        errors.append({'reason': 'Observed total differs from corrected assignment', 'expected': EXPECTED, 'actual': observed})
    group = {'id': 'previous_snapshots', 'paths': list(map(str, SNAPSHOTS)), 'expected': EXPECTED,
             'observed': observed, 'eligible': not errors, 'errors': errors, 'entries': entries,
             'reason': 'Uncommitted bulk inventories; exact size/hash retained in unchanged lead_acceptance.json'}
    inputs = [hash_row(q) for q in sorted(RECORD.iterdir()) if q.is_file()]
    inputmap = {e['path']: e for e in inputs}
    historical_records = load(RECORD/'record_files.json')['entries']
    require(all(inputmap[e['path']]['bytes'] == e['bytes'] and inputmap[e['path']]['sha256'] == e['sha256']
                for e in historical_records), 'First-run record differs from its saved hash')
    before = {'started_utc': started, 'finished_utc': now(), 'roots': [str(PREVIOUS)],
              'individual_files': [], 'entries': rows, 'summary': totals(rows),
              'record_inputs': inputs, 'first_run_record_hash_checks': len(historical_records),
              'pre_existing_record_files': [e['path'] for e in inputs if Path(e['path']).name not in NEW_SCRIPTS],
              'note': 'Only four target files and their parent are inventoried; record inputs hashed for immutability.'}
    write('scoped_inventory_before_round2.json', before)
    previous_run = {'manifest': hash_row(RECORD/'deletion_manifest_final.json'),
                    'journal': hash_row(RECORD/'deletion_journal.jsonl'),
                    'receipt': hash_row(RECORD/'deletion_receipt.json'),
                    'record_index': hash_row(RECORD/'record_files.json'),
                    'baseline': hash_row(RECORD/'scoped_inventory_before_final.json'),
                    'journal_policy': 'Round-two journal continues the first run through its SHA256 reference; first-run bytes remain unchanged.'}
    write('deletion_manifest_round2.json', {'schema_version': 1, 'round': 2, 'created_utc': now(),
          'working_root': str(ROOT), 'groups': [group], 'entries': entries if group['eligible'] else [],
          'planned': observed if group['eligible'] else totals([]), 'authorized_empty_directories': [],
          'process_prerequisite': process, 'assignment_sha256': ASSIGNMENT_SHA,
          'before_inventory_sha256': hash_row(RECORD/'scoped_inventory_before_round2.json')['sha256'],
          'protected_evidence': [bypath[str(PREVIOUS/'lead_acceptance.json')]], 'previous_run': previous_run,
          'corrected_table': TABLE, 'first_run_parent_entries': list(original.values()),
          'withdrawn_expected_allocated_bytes': 300759552, 'corrected_expected_allocated_bytes': 300761088,
          'limitations': [HOST_LIMIT, 'No Git commands: committed status is lead-supplied; all 25 surviving files are protected.',
                         'Allocated bytes are st_blocks * 512 for single-link files; no global free-space attribution.',
                         'Runtime directories are host-owned and untouched. Task records outside this round are excluded.']})
    print(json.dumps({'status': 'PASS_PREPARED' if not errors else 'PARTIAL_CLASSIFICATION',
                      'expected': EXPECTED, 'observed': observed, 'errors': errors,
                      'survivors_matching_first_run': len(survivors),
                      'pre_existing_record_files': len(before['pre_existing_record_files'])}, indent=2), flush=True)
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
