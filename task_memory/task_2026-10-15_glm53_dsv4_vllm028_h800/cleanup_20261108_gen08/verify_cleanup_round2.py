#!/usr/bin/env python3
"""Independent round-two audit adapted from verify_cleanup_final.py.

Reuse the first-run hashing, journal reconciliation and save-once format.
Only the snapshot parent is traversed; no executor helpers are imported.
Reruns print a fresh verdict without changing the saved verification record.
"""
import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

R = Path(__file__).resolve().parent
T = R.parent
PREVIOUS = T / 'cleanup_20261107_gen05_07'
SNAPSHOTS = {str(PREVIOUS / ('protected_hash_' + s + '.json'))
             for s in ('before', 'after', 'before_final', 'after_final')}
HOST_SHA = '20c553d72918382e08f7b636c9ab4d1896cd7f155c449fda5132534164340296'
ASSIGNMENT_SHA = 'a2c5eeec8073e8939f9a594dfa253939a0dbeb182b1c9ad465ad95f59ee15e60'
EXPECTED = {'files': 4, 'bytes': 300753869, 'allocated_bytes': 300761088}
TABLE = {
    'protected_hash_before.json': (75188389, 75190272, '8cb0e44aabf312785b149e59e305f05f8c4c21b8560d20c056b7db7d2faccda8'),
    'protected_hash_after.json': (75188388, 75190272, '55c8d5d65e499451186a30af078a2301077b18735e35b2425baa1c73b091c1c3'),
    'protected_hash_before_final.json': (75188546, 75190272, 'cd50532f06fd0d027b4bee62459e7f5bf0d5b0d806c6733ed191783ba5915b7b'),
    'protected_hash_after_final.json': (75188546, 75190272, '9cf0b5b8c4cf1afa8094e26b4d3f7da758e45994bad8b1eef8dfabc756c7f265'),
}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(name):
    return json.loads((R / name).read_text())


def sig(s):
    return {k: getattr(s, 'st_' + k) for k in ('dev', 'ino', 'mode', 'nlink', 'uid', 'gid',
                                              'size', 'blocks', 'mtime_ns', 'ctime_ns')}


def audit(path):
    path = Path(path)
    if path.resolve() != path:
        raise RuntimeError('Noncanonical audit path: ' + str(path))
    s = path.lstat()
    kind = 'file' if stat.S_ISREG(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'other'
    if kind == 'other':
        raise RuntimeError('Unsupported scoped entry: ' + str(path))
    e = {'path':str(path),'kind':kind,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'signature':sig(s)}
    if kind == 'file':
        h = hashlib.sha256()
        with path.open('rb') as f:
            if sig(os.fstat(f.fileno())) != sig(s):
                raise RuntimeError('Changed before audit read: ' + str(path))
            for b in iter(lambda:f.read(8*1024*1024), b''):
                h.update(b)
            if sig(os.fstat(f.fileno())) != sig(s):
                raise RuntimeError('Changed during audit read: ' + str(path))
        if sig(path.lstat()) != sig(s):
            raise RuntimeError('Changed during audit hashing: ' + str(path))
        e['sha256'] = h.hexdigest()
    return e


def enumerate_root(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError('Missing/non-directory audit root: ' + str(root))
    out = [root]
    def raise_error(e):
        raise e
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=raise_error):
        out.extend(Path(parent)/s for s in dirs+files)
    return out


def total(rows):
    rows = [e for e in rows if e['kind']=='file']
    return {'files':len(rows),'bytes':sum(e['bytes'] for e in rows),
            'allocated_bytes':sum(e['allocated_bytes'] for e in rows)}


def authorized_file(s):
    return s in SNAPSHOTS


def save_once(name, obj):
    with (R/name).open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    started = stamp()
    errors = []
    def check(ok, reason):
        if not ok:
            errors.append(reason)
    m = load('deletion_manifest_round2.json')
    receipt = load('deletion_receipt_round2.json')
    before = load('scoped_inventory_before_round2.json')
    check(audit(R/'deletion_manifest_round2.json')['sha256'] == receipt['manifest_sha256'], 'Frozen manifest hash differs')
    check(audit(R/'scoped_inventory_before_round2.json')['sha256'] == receipt['before_inventory_sha256']
          == m['before_inventory_sha256'], 'Frozen baseline hash differs')
    check(audit(R/'assignment_2.md')['sha256'] == receipt['assignment_sha256'] == ASSIGNMENT_SHA,
          'Round-two assignment hash differs')
    check(audit(R/'cleanup_scope_processes.json')['sha256'] == HOST_SHA, 'Supplied process evidence differs')
    process = load('cleanup_scope_processes.json')
    check(process['clear_for_deletion'] is True and process['processes_scanned'] == 289
          and not process['referencing_pids'] and not process['referencing_processes'], 'Supplied host evidence is not clear')
    check(receipt['process_prerequisite'] == m['process_prerequisite'], 'Process prerequisite differs')
    check(receipt['process_prerequisite']['sha256'] == HOST_SHA, 'Process prerequisite digest differs')
    check(audit(R/'deletion_journal_round2.jsonl')['sha256'] == receipt['journal_sha256'], 'Round-two journal hash differs')
    check(receipt['previous_run'] == m['previous_run'], 'Journal continuation references differ')
    for key in ('manifest', 'journal', 'receipt', 'record_index', 'baseline'):
        row = m['previous_run'][key]
        check(audit(row['path']) == row, 'First-run continuation input changed: ' + row['path'])
    original = {e['path']: e for e in load('scoped_inventory_before_final.json')['entries']
                if Path(e['path']).parent == PREVIOUS or Path(e['path']) == PREVIOUS}
    check({e['path']: e for e in m['first_run_parent_entries']} == original,
          'Embedded first-run parent baseline differs from preserved original')
    bmap = {e['path']: e for e in before['entries']}
    planned = {e['path']: e for e in m['entries']}
    deleted = {e['path']: e for e in receipt['deleted']}
    check(len(bmap) == len(before['entries']) == 30 and len(planned) == len(m['entries']) == 4
          and len(deleted) == len(receipt['deleted']) == 4, 'Duplicate or incorrect inventory/deletion count')
    check(set(bmap) == set(original), 'Round-two parent membership differs from first run')
    check(set(planned) == set(deleted) == SNAPSHOTS, 'Deleted paths differ from the four authorized targets')
    check(not m['authorized_empty_directories'] and not receipt['removed_directories']
          and receipt['removed_directory_count'] == 0, 'Directory removal is forbidden in this round')
    check(receipt['status'] == 'PASS_EXECUTED' and not receipt['errors'] and not receipt['declined'],
          'Execution has errors, declined files or a nonpassing status')
    check(len(m['groups']) == 1 and m['groups'][0]['id'] == 'previous_snapshots'
          and m['groups'][0]['eligible'] and not m['groups'][0]['errors']
          and m['groups'][0]['expected'] == m['groups'][0]['observed'] == EXPECTED
          and m['groups'][0]['entries'] == m['entries'], 'Manifest classification differs from corrected assignment')
    acceptance = json.loads((PREVIOUS/'lead_acceptance.json').read_text())
    check(acceptance['status'] == 'ACCEPTED', 'Surviving lead acceptance status differs')
    snapshot_checks = []
    for s, e in deleted.items():
        check(authorized_file(s) and e == planned.get(s) and e['signature']['nlink'] == 1
              and re.fullmatch('[0-9a-f]{64}', e['sha256']) is not None, 'Unsafe/incomplete deleted-file record: ' + s)
        check(not os.path.lexists(s), 'Deleted target still present: ' + s)
        check(dict(bmap[s], group=e['group']) == e and bmap[s] == original[s],
              'Deleted-file pre-deletion size/hash/metadata differs from first run: ' + s)
        expected = TABLE[Path(s).name]
        saved = acceptance['snapshot_files_not_committed'][Path(s).name]
        match = (e['bytes'], e['allocated_bytes'], e['sha256']) == expected
        match_saved = e['bytes'] == saved['bytes'] and e['sha256'] == saved['sha256']
        snapshot_checks.append({'path': s, 'pre_deletion_bytes': e['bytes'],
              'pre_deletion_allocated_bytes': e['allocated_bytes'], 'pre_deletion_sha256': e['sha256'],
              'matches_corrected_table': match, 'matches_surviving_acceptance_record': match_saved,
              'absent': not os.path.lexists(s)})
        check(match and match_saved, 'Deleted-file accounting differs from corrected table/surviving evidence: ' + s)
    events = [json.loads(line) for line in (R/'deletion_journal_round2.jsonl').read_text().splitlines()]
    by_event = collections.defaultdict(list)
    for e in events:
        by_event[e['event']].append(e)
    for name in ('will_delete_file', 'deleted_file'):
        check(len(by_event[name]) == 4 and {e['path'] for e in by_event[name]} == SNAPSHOTS,
              'Journal count/path mismatch: ' + name)
        for e in by_event[name]:
            check({k:v for k,v in e.items() if k not in ('event', 'utc')} == deleted.get(e['path']),
                  'Journal data differs: ' + name)
    check(set(by_event) == {'will_delete_file', 'deleted_file'} and len(events) == 8,
          'Unexpected round-two journal event or count')
    for i in range(0, len(events), 2):
        pair = events[i:i+2]
        check(len(pair) == 2 and pair[0]['path'] == pair[1]['path']
              and (pair[0]['event'], pair[1]['event']) == ('will_delete_file', 'deleted_file'),
              'Journal intent/completion order mismatch')
    check(before['roots'] == [str(PREVIOUS)] and not before['individual_files'], 'Expanded verification scope')
    paths = sorted(enumerate_root(PREVIOUS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        after_rows = list(pool.map(audit, paths))
    amap = {e['path']: e for e in after_rows}
    expected_after = set(bmap) - SNAPSHOTS
    missing, added = sorted(expected_after-set(amap)), sorted(set(amap)-expected_after)
    changed_files, changed_dirs = [], []
    for s in sorted(expected_after & set(amap)):
        a, b = amap[s], bmap[s]
        if a['kind'] != b['kind'] or a['kind'] == 'file' and a != b:
            changed_files.append(s)
        elif a['kind'] == 'directory':
            fields = ('dev', 'ino', 'mode', 'nlink', 'uid', 'gid') if s == str(PREVIOUS) else tuple(b['signature'])
            if any(a['signature'][k] != b['signature'][k] for k in fields):
                changed_dirs.append(s)
    check(not missing and not added and not changed_files and not changed_dirs,
          {'unexpected_missing': missing, 'unexpected_added': added,
           'changed_files': changed_files, 'changed_directories': changed_dirs})
    first_run_checks = []
    for s, b in sorted(original.items()):
        if s in SNAPSHOTS or b['kind'] != 'file':
            continue
        a = amap.get(s)
        match = a is not None and a == b
        first_run_checks.append({'path': s, 'expected_bytes': b['bytes'],
              'actual_bytes': a.get('bytes') if a else None, 'expected_sha256': b['sha256'],
              'actual_sha256': a.get('sha256') if a else None, 'matches_first_run_hash_and_metadata': match})
        check(match, 'Surviving file differs from first-run hash/metadata: ' + s)
    matched = sum(e['matches_first_run_hash_and_metadata'] for e in first_run_checks)
    check(len(first_run_checks) == matched == 25 and total(after_rows)['files'] == 25,
          'Expected exactly 25 unchanged surviving files')
    record_checks = []
    for b in before['record_inputs']:
        a = audit(b['path'])
        match = a == b
        record_checks.append({'path': b['path'], 'expected_sha256': b['sha256'],
                              'actual_sha256': a['sha256'], 'matches_preparation': match})
        check(match, 'Preserved first-run record or round-two script changed: ' + b['path'])
    saved_records = load('record_files.json')['entries']
    record_map = {e['path']: e for e in record_checks}
    check(len(saved_records) == before['first_run_record_hash_checks'] == 19
          and all(record_map[e['path']]['actual_sha256'] == e['sha256'] for e in saved_records),
          'One or more first-run records differs from its original saved hash')
    summary = total(receipt['deleted'])
    check(summary == receipt['deleted_summary'] == m['planned'] == EXPECTED,
          'Deleted totals do not reconcile to corrected assignment')
    check(m['withdrawn_expected_allocated_bytes'] == 300759552
          and m['corrected_expected_allocated_bytes'] == 300761088, 'Allocation correction differs')
    lead_path = str(PREVIOUS/'lead_acceptance.json')
    lead = next(e for e in first_run_checks if e['path'] == lead_path)
    check(m['protected_evidence'] == [original[lead_path]], 'Protected acceptance baseline differs')
    out = {'schema_version': 1, 'round': 2, 'started_utc': started, 'finished_utc': stamp(),
           'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
           'interpreter': sys.executable, 'python_version': sys.version,
           'expected_deleted': EXPECTED, 'actual_deleted': summary,
           'allocation_difference_from_corrected_table': summary['allocated_bytes'] - 300761088,
           'removed_directory_count': len(receipt['removed_directories']),
           'journal_event_counts': {k:len(v) for k,v in by_event.items()},
           'scoped_protection': {'root': str(PREVIOUS), 'expected_surviving_files': 25,
               'actual_surviving_files': total(after_rows)['files'], 'hash_matched_files': matched,
               'expected_remaining_paths': len(expected_after), 'actual_remaining_paths': len(amap),
               'missing': missing, 'added': added, 'changed_files': changed_files, 'changed_directories': changed_dirs,
               'allowed_parent_directory_metadata_changes': ['size', 'blocks', 'mtime_ns', 'ctime_ns'],
               'parent_preserved': str(PREVIOUS) in amap and not changed_dirs},
           'surviving_files_first_run_checks': first_run_checks,
           'lead_acceptance': lead, 'snapshot_recorded_hash_checks': snapshot_checks,
           'record_input_checks': record_checks,
           'pre_existing_record_files_preserved': len(before['pre_existing_record_files']),
           'first_run_saved_record_hashes_matched': len(saved_records),
           'process_prerequisite': receipt['process_prerequisite'], 'previous_run': receipt['previous_run'],
           'limitations': m['limitations'], 'after_inventory': after_rows,
           'assignment_all_targets_absent': all(not os.path.lexists(s) for s in SNAPSHOTS),
           'assignment_outcome': 'implemented' if not errors else 'partial'}
    if not (R/'verification_round2.json').exists():
        save_once('verification_round2.json', out)
    print(json.dumps({k:out[k] for k in ('status', 'errors', 'expected_deleted', 'actual_deleted',
          'allocation_difference_from_corrected_table', 'removed_directory_count', 'journal_event_counts',
          'scoped_protection', 'lead_acceptance', 'pre_existing_record_files_preserved',
          'first_run_saved_record_hashes_matched')}, indent=2), flush=True)
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
