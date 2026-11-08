#!/usr/bin/env python3
"""Freeze a scoped inventory and classify only the lead's three cleanup groups."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

RECORD = Path(__file__).resolve().parent
TASK = RECORD.parent
ROOT = TASK.parent.parent
PREVIOUS = TASK / 'cleanup_20261107_gen05_07'
FORMAL = TASK / 'h3o_operator_costs/formal_p2_64_native_20261102'
LAUNCH = TASK / 'native_track_20261020/g0d_launch'
GEN = FORMAL / 'generation08'
LOG = LAUNCH / 'launcher.log'
SSH = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh')
HOST = Path('/data/ycfeng/tmp/ds4_runs/cleanup-scan-1108b/scratch/cleanup_scope_processes.json')
SNAPSHOTS = [PREVIOUS / ('protected_hash_' + x + '.json')
             for x in ('before', 'after', 'before_final', 'after_final')]
HOST_SHA = '20c553d72918382e08f7b636c9ab4d1896cd7f155c449fda5132534164340296'
ASSIGNMENT_SHA = '83fa01bd9d9374555bf791176a531437ac93767a67f21e6a838ed9a9d4ee7d16'
HOST_LIMIT = ('Lead-supplied host scan; no executor process scan. 127/289 entries had unreadable links; '
              'lead reports 14/15 sampled entries had exited and the other was pid 1 /sbin/init. '
              'Raw scope_roots names before/after snapshots but omits before_final/after_final; '
              'assignment explicitly accepts this file as the prerequisite for all three groups. '
              'The sample follow-up is supplied in the assignment, not in the raw scan JSON.')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(name, value):
    path = RECORD / name
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def load(path):
    return json.loads(Path(path).read_text())


def signature(s):
    return {k: getattr(s, 'st_' + k) for k in ('dev', 'ino', 'mode', 'nlink', 'uid', 'gid',
                                              'size', 'blocks', 'mtime_ns', 'ctime_ns')}


def metadata(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Noncanonical path: ' + str(path))
    s = path.lstat()
    kind = 'file' if stat.S_ISREG(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'other'
    require(kind != 'other', 'Unsupported scoped entry: ' + str(path))
    return {'path': str(path), 'kind': kind, 'bytes': s.st_size,
            'allocated_bytes': s.st_blocks * 512, 'signature': signature(s)}


def hash_row(path):
    row = metadata(path)
    if row['kind'] == 'file':
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            require(signature(os.fstat(f.fileno())) == row['signature'], 'File changed before open: ' + str(path))
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
            require(signature(os.fstat(f.fileno())) == row['signature'], 'File changed during read: ' + str(path))
        require(metadata(path)['signature'] == row['signature'], 'File changed during hashing: ' + str(path))
        row['sha256'] = h.hexdigest()
    return row


def walk(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Missing/noncanonical inventory root: ' + str(root))
    result = [root]
    for base, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        for name in sorted(dirs + files):
            p = Path(base) / name
            metadata(p)
            result.append(p)
    return result


def totals(rows):
    files = [e for e in rows if e['kind'] == 'file']
    return {'files': len(files), 'bytes': sum(e['bytes'] for e in files),
            'allocated_bytes': sum(e['allocated_bytes'] for e in files)}


def host_check():
    row = hash_row(HOST)
    require(row['sha256'] == HOST_SHA, 'Supplied host evidence changed')
    obj = load(HOST)
    require(obj['clear_for_deletion'] is True and obj['processes_scanned'] == 289
            and obj['referencing_pids'] == [] and obj['referencing_processes'] == [], 'Host evidence is not clear')
    return {'source': str(HOST), 'copy': str(RECORD / 'cleanup_scope_processes.json'),
            'sha256': HOST_SHA, 'observed_utc': obj['observed_utc'], 'processes_scanned': 289,
            'clear_for_deletion': True, 'limitation': HOST_LIMIT}


def main():
    started = now()
    require(hash_row(RECORD / 'assignment.md')['sha256'] == ASSIGNMENT_SHA, 'Assignment changed')
    require(sorted(p.name for p in RECORD.iterdir() if p.is_file()) ==
            ['assignment.md', 'execute_cleanup.py', 'prepare_cleanup.py', 'verify_cleanup_final.py'],
            'Unexpected pre-existing run records; preserve and inspect before preparation')
    runtime_directories = [p for p in RECORD.iterdir() if p.is_dir()]
    require(all(p.name == 'codex-implement-909a1443-fd16-4aa5-aa44-58537cab480c'
                and not p.is_symlink() for p in runtime_directories), 'Unexpected record-directory child')
    process = host_check()
    with (RECORD / 'cleanup_scope_processes.json').open('xb') as f:
        f.write(HOST.read_bytes())
        f.flush()
        os.fsync(f.fileno())
    require(hash_row(process['copy'])['sha256'] == HOST_SHA, 'Host evidence copy mismatch')
    lead = load(PREVIOUS / 'lead_acceptance.json')
    require(lead['status'] == 'ACCEPTED', 'Previous cleanup is not accepted')
    log_evidence = load(LAUNCH / 'launcher_log_evidence.json')
    failed = load(GEN / 'result.json')
    completed = load(FORMAL / 'generation08_run2/result.json')
    immutable = [RECORD / p for p in ('assignment.md', 'prepare_cleanup.py', 'execute_cleanup.py',
                                    'verify_cleanup_final.py', 'cleanup_scope_processes.json')]
    immutable += [TASK / p for p in ('requirements.md', 'plan.md', 'progress.md', 'issues.md',
                                     'tickets/18-native-ground-truth-h200.md', 'tickets/19-comparison-and-acceptance.md',
                                     'test_report_2026-11-07_p2_64_generation08_formal.md')]
    immutable += [ROOT / 'AGENTS.md', HOST]
    roots = [FORMAL, LAUNCH, PREVIOUS, SSH]
    paths = sorted({p for root in roots for p in walk(root)} | set(immutable))
    # Candidates and protected files are hashed together once; notably the sparse log is read only once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(hash_row, paths))
    by_path = {e['path']: e for e in rows}
    require(metadata(LOG)['signature'] == by_path[str(LOG)]['signature'], 'Launcher log changed: stop all deletion')
    generation_rows = [e for e in rows if Path(e['path']).is_relative_to(GEN)]
    groups = [
        {'id': 'generation08', 'paths': [str(GEN)], 'expected': {'files': 173, 'allocated_bytes': 78544896},
         'reason': "UNVERIFIED comparison stopped with ModuleNotFoundError: No module named 'tests'; superseded by generation08_run2",
         'rows': generation_rows, 'errors': []},
        {'id': 'launcher_log', 'paths': [str(LOG)], 'expected': {'files': 1, 'allocated_bytes': 2114797568},
         'reason': 'Stopped stale poll log; earlier truncation destroyed submission output; identity retained in launcher_log_evidence.json',
         'rows': [by_path[str(LOG)]], 'errors': []},
        {'id': 'previous_snapshots', 'paths': list(map(str, SNAPSHOTS)),
         'expected': {'files': 4, 'allocated_bytes': 300759552},
         'reason': 'Uncommitted bulk inventories; each size/hash retained in accepted lead_acceptance.json',
         'rows': [by_path[str(p)] for p in SNAPSHOTS], 'errors': []},
    ]
    gen_error = next((e.get('error', '') for e in failed['steps'] if e['name'] == 'comparison'), '')
    if not (failed['status'] == 'UNVERIFIED' and "ModuleNotFoundError: No module named 'tests'" in gen_error
            and completed['status'] == 'TOOLS_FINISHED' and completed.get('comparison_complete') is True
            and len(completed['steps']) == 7 and all(e['status'] == 'PASS' for e in completed['steps'])):
        groups[0]['errors'].append('Failed/superseding comparison state does not match assignment')
    log_row = by_path[str(LOG)]
    if not (log_row['bytes'] == log_evidence['logical_bytes'] == 4332828936
            and log_row['allocated_bytes'] == log_evidence['allocated_bytes']
            and log_row['sha256'] == log_evidence['sha256']
            and (datetime.datetime.fromtimestamp(log_row['signature']['mtime_ns']//1000000000,
                    datetime.timezone.utc) + datetime.timedelta(microseconds=(log_row['signature']['mtime_ns']%1000000000)//1000)).isoformat() == log_evidence['mtime_utc']):
        groups[1]['errors'].append('Launcher size/allocation/hash/mtime differs from preserved evidence')
    for p in SNAPSHOTS:
        e, expected = by_path[str(p)], lead['snapshot_files_not_committed'][p.name]
        if e['bytes'] != expected['bytes'] or e['sha256'] != expected['sha256']:
            groups[2]['errors'].append('Snapshot differs from recorded size/hash: ' + str(p))
    for group in groups:
        group['observed'] = totals(group['rows'])
        if any(group['observed'][k] != v for k, v in group['expected'].items()):
            group['errors'].append('Observed counts/allocation differ from assignment table')
        if any(e['signature']['nlink'] != 1 for e in group['rows'] if e['kind'] == 'file'):
            group['errors'].append('Candidate has multiple hard links')
        group['eligible'] = not group['errors']
        group['entries'] = [dict(e, group=group['id']) for e in group.pop('rows') if e['kind'] == 'file']
    candidates = [e for g in groups if g['eligible'] for e in g['entries']]
    baseline = {'started_utc': started, 'finished_utc': now(), 'roots': list(map(str, roots)),
                'individual_files': list(map(str, immutable)), 'entries': rows,
                'host_runtime_directories_preserved': list(map(str, runtime_directories)),
                'summary': totals(rows), 'note': 'Scoped recursive parents plus SSH neighbour; no repository-wide snapshot; log hashed once.'}
    write('scoped_inventory_before.json', baseline)
    write('deletion_manifest.json', {'schema_version': 1, 'created_utc': now(), 'working_root': str(ROOT),
          'groups': groups, 'entries': candidates, 'planned': totals(candidates),
          'authorized_empty_directories': [e for e in generation_rows if e['kind'] == 'directory'] if groups[0]['eligible'] else [],
          'process_prerequisite': process, 'assignment_sha256': ASSIGNMENT_SHA,
          'before_inventory_sha256': hash_row(RECORD / 'scoped_inventory_before.json')['sha256'],
          'protected_evidence': [by_path[str(p)] for p in (LAUNCH / 'launcher_log_evidence.json',
                          PREVIOUS / 'lead_acceptance.json', FORMAL / 'generation08_run2/result.json')],
          'limitations': [HOST_LIMIT, 'No Git commands: protect all 25 other previous-cleanup files, exceeding the committed-file scope.',
                         'SSH copy inventoried read-only; duplicate platform-route copy outside this run is not traversed or changed.']})
    print(json.dumps({'status': 'PASS_PREPARED' if all(g['eligible'] for g in groups) else 'PARTIAL_CLASSIFICATION',
                      'groups': [{k:g[k] for k in ('id','expected','observed','eligible','errors')} for g in groups],
                      'inventory': baseline['summary'], 'directories': sum(e['kind']=='directory' for e in rows)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
