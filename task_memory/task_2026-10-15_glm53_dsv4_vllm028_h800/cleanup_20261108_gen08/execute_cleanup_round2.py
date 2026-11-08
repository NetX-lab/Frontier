#!/usr/bin/env python3
"""Delete only frozen candidates, journaling intent and completion durably."""
import json
import os
from pathlib import Path
import sys

import prepare_cleanup_round2 as p


def allowed(path):
    path = Path(path)
    return path in p.SNAPSHOTS


def durable_parent(path):
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    started = p.now()
    p.require(not any((p.RECORD / name).exists() for name in ('deletion_journal_round2.jsonl', 'deletion_receipt_round2.json')),
              'Execution already started; inspect the durable journal instead of rerunning deletion')
    manifest = p.load(p.RECORD / 'deletion_manifest_round2.json')
    manifest_sha = p.hash_row(p.RECORD / 'deletion_manifest_round2.json')['sha256']
    before = p.load(p.RECORD / 'scoped_inventory_before_round2.json')
    p.require(p.hash_row(p.RECORD / 'scoped_inventory_before_round2.json')['sha256'] == manifest['before_inventory_sha256'],
              'Frozen baseline changed')
    p.require(manifest['assignment_sha256'] == p.hash_row(p.RECORD / 'assignment_2.md')['sha256'] == p.ASSIGNMENT_SHA,
              'Assignment differs')
    p.host_check()
    p.require(p.hash_row(p.RECORD / 'cleanup_scope_processes.json')['sha256'] == p.HOST_SHA, 'Process evidence copy changed')
    expected = {e['path']: e for e in before['entries']}
    actual_paths = {str(q) for root in before['roots'] for q in p.walk(root)} | set(before['individual_files'])
    p.require(actual_paths == set(expected), 'Scoped inventory paths changed before deletion')
    for row in before['entries']:
        p.require(p.metadata(row['path'])['signature'] == row['signature'], 'Scoped entry changed before deletion: ' + row['path'])
    candidates = manifest['entries']
    p.require(len({e['path'] for e in candidates}) == len(candidates), 'Duplicate candidate')
    p.require(all(allowed(e['path']) and e['kind']=='file' and e['signature']['nlink']==1
                  and e == dict(expected[e['path']], group=e['group']) for e in candidates), 'Unsafe/fabricated candidate')
    directories = manifest['authorized_empty_directories']
    p.require(not directories and before['roots'] == [str(p.PREVIOUS)] and not before['individual_files'],
              'Round-two directory removal or expanded scope is forbidden')
    for row in before['record_inputs']:
        p.require(p.hash_row(row['path']) == row, 'Frozen record input changed: ' + row['path'])
    p.require(manifest['planned'] == p.totals(candidates) == p.EXPECTED
              and {e['path'] for e in candidates} == set(map(str, p.SNAPSHOTS))
              and len(manifest['groups']) == 1 and manifest['groups'][0]['id'] == 'previous_snapshots'
              and manifest['groups'][0]['eligible'] and manifest['groups'][0]['entries'] == candidates,
              'Round-two plan differs')
    lead = p.load(p.PREVIOUS / 'lead_acceptance.json')
    for row in candidates:
        name = Path(row['path']).name
        p.require((row['bytes'], row['allocated_bytes'], row['sha256']) == p.TABLE[name]
                  and lead['snapshot_files_not_committed'][name] ==
                      {'bytes': row['bytes'], 'sha256': row['sha256']},
                  'Candidate differs from corrected table or surviving evidence: ' + row['path'])
    validation = {'status':'PASS', 'observed_utc':p.now(), 'manifest_sha256':manifest_sha,
                  'candidate_summary':p.totals(candidates), 'metadata_checked':len(before['entries']),
                  'candidate_policy':'Each file hashed at preparation; frozen full signature rechecked immediately before unlink',
                  'process_prerequisite':manifest['process_prerequisite']}
    p.write('validation_pre_delete_round2.json', validation)
    deleted, removed, declined, errors = [], [], [], []
    declined.extend({'group':g['id'], 'paths':g['paths'], 'reasons':g['errors']} for g in manifest['groups'] if not g['eligible'])
    with (p.RECORD / 'deletion_journal_round2.jsonl').open('x') as journal:
        def event(action, row):
            journal.write(json.dumps({'event':action,'utc':p.now(), **row},sort_keys=True)+'\n')
            journal.flush()
            os.fsync(journal.fileno())
        try:
            for group in manifest['groups']:
                if not group['eligible']:
                    continue
                group_rows = group['entries']
                changed = [e['path'] for e in group_rows if p.metadata(e['path'])['signature'] != e['signature']]
                if changed:
                    declined.append({'group':group['id'],'paths':changed,'reasons':['Changed since hashing; entire group retained']})
                    continue
                for row in group_rows:
                    p.require(allowed(row['path']), 'Outside deletion authority')
                    p.require(p.metadata(row['path'])['signature']==row['signature'], 'Candidate changed before unlink: '+row['path'])
                    event('will_delete_file', row)
                    Path(row['path']).unlink()
                    durable_parent(row['path'])
                    event('deleted_file', row)
                    deleted.append(row)
        except Exception as e:
            errors.append(str(e))
            event('execution_error', {'error':str(e)})
    receipt = {'schema_version':1,'started_utc':started,'finished_utc':p.now(),
               'status':'PASS_EXECUTED' if not errors and not declined else 'PARTIAL',
               'interpreter':sys.executable,'python_version':sys.version,'manifest_sha256':manifest_sha,
               'before_inventory_sha256':manifest['before_inventory_sha256'],'assignment_sha256':p.ASSIGNMENT_SHA,
               'process_prerequisite':manifest['process_prerequisite'],'previous_run':manifest['previous_run'],'round':2,'deleted':deleted,'deleted_summary':p.totals(deleted),
               'removed_directories':removed,'removed_directory_count':len(removed),
               'directory_allocated_bytes_at_removal':sum(e['allocated_bytes'] for e in removed),
               'directory_allocated_bytes_before':sum(e['allocated_bytes'] for e in directories if e['path'] in {r['path'] for r in removed}),
               'declined':declined,'errors':errors,'journal_sha256':p.hash_row(p.RECORD/'deletion_journal_round2.jsonl')['sha256'],
               'reclaimed_bytes_basis':'Sum of st_blocks * 512 for deleted single-link regular files; directory blocks separate. No global free-space attribution.'}
    p.write('deletion_receipt_round2.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','deleted_summary','removed_directory_count','declined','errors')},indent=2),flush=True)
    raise SystemExit(bool(errors or declined))


if __name__ == '__main__':
    main()
