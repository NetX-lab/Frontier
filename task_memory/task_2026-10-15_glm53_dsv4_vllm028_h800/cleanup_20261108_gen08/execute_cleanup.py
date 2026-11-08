#!/usr/bin/env python3
"""Delete only frozen candidates, journaling intent and completion durably."""
import json
import os
from pathlib import Path
import sys

import prepare_cleanup as p


def allowed(path):
    path = Path(path)
    return path == p.LOG or path in p.SNAPSHOTS or (path.is_relative_to(p.GEN) and path != p.GEN)


def durable_parent(path):
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    started = p.now()
    p.require(not any((p.RECORD / name).exists() for name in ('deletion_journal.jsonl', 'deletion_receipt.json')),
              'Execution already started; inspect the durable journal instead of rerunning deletion')
    manifest = p.load(p.RECORD / 'deletion_manifest_final.json')
    manifest_sha = p.hash_row(p.RECORD / 'deletion_manifest_final.json')['sha256']
    before = p.load(p.RECORD / 'scoped_inventory_before_final.json')
    p.require(p.hash_row(p.RECORD / 'scoped_inventory_before_final.json')['sha256'] == manifest['before_inventory_sha256'],
              'Frozen baseline changed')
    p.require(manifest['assignment_sha256'] == p.hash_row(p.RECORD / 'assignment.md')['sha256'] == p.ASSIGNMENT_SHA,
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
    p.require({e['path'] for e in directories} == {e['path'] for e in before['entries']
              if e['kind']=='directory' and Path(e['path']).is_relative_to(p.GEN)}
              if any(g['id']=='generation08' and g['eligible'] for g in manifest['groups']) else not directories,
              'Unauthorized directory scope')
    validation = {'status':'PASS', 'observed_utc':p.now(), 'manifest_sha256':manifest_sha,
                  'candidate_summary':p.totals(candidates), 'metadata_checked':len(before['entries']),
                  'log_policy':'SHA256 read once at preparation; unchanged dev/inode/size/allocation/mtime/ctime rechecked before unlink',
                  'process_prerequisite':manifest['process_prerequisite']}
    p.write('validation_pre_delete.json', validation)
    deleted, removed, declined, errors = [], [], [], []
    declined.extend({'group':g['id'], 'paths':g['paths'], 'reasons':g['errors']} for g in manifest['groups'] if not g['eligible'])
    with (p.RECORD / 'deletion_journal.jsonl').open('x') as journal:
        def event(action, row):
            journal.write(json.dumps({'event':action,'utc':p.now(), **row},sort_keys=True)+'\n')
            journal.flush()
            os.fsync(journal.fileno())
        try:
            # Revalidate the log immediately before starting; any observed growth stops the run.
            log_expected = expected[str(p.LOG)]
            p.require(p.metadata(p.LOG)['signature']==log_expected['signature'], 'Launcher log changed: stop all deletion')
            for group in manifest['groups']:
                if not group['eligible']:
                    continue
                group_rows = group['entries']
                changed = [e['path'] for e in group_rows if p.metadata(e['path'])['signature'] != e['signature']]
                if changed:
                    if group['id']=='launcher_log':
                        raise RuntimeError('Launcher log changed: stop all deletion')
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
                if group['id']=='generation08':
                    for row in sorted(directories,key=lambda e:len(Path(e['path']).parts),reverse=True):
                        q=Path(row['path'])
                        p.require(q.is_relative_to(p.GEN) and not q.is_symlink(), 'Unauthorized empty-directory removal')
                        s=p.metadata(q)
                        p.require(all(s['signature'][k]==row['signature'][k] for k in ('dev','ino','mode','uid','gid')),
                                  'Directory identity changed: '+str(q))
                        p.require(not list(q.iterdir()), 'Directory has unexpected remaining contents: '+str(q))
                        event('will_remove_empty_directory',s)
                        q.rmdir()
                        durable_parent(q)
                        event('removed_empty_directory',s)
                        removed.append(s)
        except Exception as e:
            errors.append(str(e))
            event('execution_error', {'error':str(e)})
    receipt = {'schema_version':1,'started_utc':started,'finished_utc':p.now(),
               'status':'PASS_EXECUTED' if not errors and not declined else 'PARTIAL',
               'interpreter':sys.executable,'python_version':sys.version,'manifest_sha256':manifest_sha,
               'before_inventory_sha256':manifest['before_inventory_sha256'],'assignment_sha256':p.ASSIGNMENT_SHA,
               'process_prerequisite':manifest['process_prerequisite'],'deleted':deleted,'deleted_summary':p.totals(deleted),
               'removed_directories':removed,'removed_directory_count':len(removed),
               'directory_allocated_bytes_at_removal':sum(e['allocated_bytes'] for e in removed),
               'directory_allocated_bytes_before':sum(e['allocated_bytes'] for e in directories if e['path'] in {r['path'] for r in removed}),
               'declined':declined,'errors':errors,'journal_sha256':p.hash_row(p.RECORD/'deletion_journal.jsonl')['sha256'],
               'reclaimed_bytes_basis':'Sum of st_blocks * 512 for deleted single-link regular files; directory blocks separate. No global free-space attribution.'}
    p.write('deletion_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','deleted_summary','removed_directory_count','declined','errors')},indent=2),flush=True)
    raise SystemExit(bool(errors or declined))


if __name__ == '__main__':
    main()
