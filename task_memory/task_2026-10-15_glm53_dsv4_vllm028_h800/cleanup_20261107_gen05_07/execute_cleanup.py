#!/usr/bin/env python3
"""Execute the frozen 449-file cleanup under assignment_2.md, with durable evidence.

The original preparation scripts, manifest and receipts remain immutable. Only
explicit manifest files and their now-empty scratch ancestors may be removed.
"""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import prepare_cleanup as prep

R, T, D, G = prep.RECORD, prep.TASK, prep.DS4, prep.GLM
CUTOFF = datetime.datetime(2026, 11, 7, 8, 5, tzinfo=datetime.timezone.utc)
EXCEPTIONS = [
    G / 'native_capture_scope_root_3ov6bjro',
    G / 'native_capture_scope_root_cctbt1_m',
    G / 'profile04_terminal_ybbq4f2m',
    G / 'graph_payload_root_image_rieb91rz/diagnostics/fault.log',
    G / 'graph_payload_root_image_rieb91rz/diagnostics/stages.jsonl',
    T / 'native_track_20261020/g0d_launch/launcher.log',
]
MONITORS = [('05', 'p64-formal5-monitor-1107f'),
            ('06', 'p64-formal6-monitor-1107k'),
            ('07', 'p64-formal7-submit-1107m')]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'Noncanonical path: ' + str(p))
    return p


def stable_stat(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size,
            s.st_mtime_ns, s.st_ctime_ns, s.st_blocks)


def immutable_inputs():
    p = R / 'record_inputs_before_final.json'
    if not p.exists():
        rows = [{'path': str(x), 'sha256': prep.sha(x), 'bytes': x.stat().st_size}
                for x in sorted(R.iterdir()) if x.is_file() and x != Path(__file__)]
        prep.write(p.name, {'observed_utc': prep.now(), 'entries': rows,
                           'authorized_update': str(R / 'report.md')})
        require(not (R / 'report_before_execution.md').exists(), 'Report backup already exists')
        (R / 'report_before_execution.md').write_bytes((R / 'report.md').read_bytes())
    return prep.load(p.name)


def manifest():
    m = prep.load('deletion_manifest.json')
    inputs = prep.load('record_inputs_before_final.json')
    baseline = next(e for e in inputs['entries'] if e['path'] == str(R / 'deletion_manifest.json'))
    require(prep.sha(baseline['path']) == baseline['sha256'], 'Frozen manifest changed')
    require(len(m['entries']) == 449 and len(m['kept']) == 32, 'Unexpected frozen counts')
    require(len(m['scope']['scratch_roots']) == 13, 'Unexpected label count')
    require(not m['archive_errors'] and not m['inventory_errors'], 'Prepared inventory has errors')
    paths = [e['path'] for e in m['entries'] + m['kept']]
    require(len(set(paths)) == 481, 'Duplicate manifest path')
    labels = [canonical(p).parent for p in m['scope']['scratch_roots']]
    require(labels == prep.labels(), 'Current labels differ from frozen labels')
    prompts = set(map(str, prep.PROMPTS))
    require(set(m['scope']['prompt_files']) == prompts, 'Unexpected prompt scope')
    counts = {}
    for row in m['entries'] + m['kept']:
        p = Path(row['path'])
        require(p.is_absolute() and os.path.normpath(str(p)) == str(p), 'Unsafe manifest pathname')
        c = row['category']
        counts[c] = counts.get(c, 0) + 1
        if c == 'superseded_prompt':
            require(str(p) in prompts and row['label'] == 'prompts', 'Unauthorized prompt')
        else:
            root = D / row['label'] / 'scratch'
            require(str(root) in m['scope']['scratch_roots'] and p.is_relative_to(root)
                    and p != root, 'File outside frozen scratch scope')
            if c == 'node_compile_cache':
                require('node-compile-cache' in p.relative_to(root).parts[:-1], 'Not a Node cache')
            else:
                require(c in ('archived_duplicate', 'uncovered_scratch'), 'Unexpected category')
        require(row['kind'] == 'file' and row['nlink'] == 1, 'Nonregular or hardlinked candidate')
        require(row['intended_action'] == ('keep' if c == 'uncovered_scratch' else 'delete'),
                'Unexpected intended action')
    require(counts == {'archived_duplicate': 87, 'node_compile_cache': 360,
                      'superseded_prompt': 2, 'uncovered_scratch': 32}, 'Unexpected category totals')
    return m


def snapshot(name):
    require(not (R / name).exists(), 'Snapshot already exists: ' + name)
    m = manifest()
    started = prep.now()
    clock = time.monotonic()
    excluded = [R, *prep.PROMPTS, *map(Path, m['scope']['scratch_roots']), *EXCEPTIONS]
    errors = []
    todo = [(p, s) for root in (T, D, G) for p, s in prep.walk(root, excluded, errors)]
    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, row in enumerate(pool.map(lambda pair: prep.info(*pair), todo), 1):
            entries.append(row)
            if 'error' in row:
                errors.append({'path': row['path'], 'error': row['error']})
            if n % 10000 == 0:
                print(name, 'inventoried', n, 'elapsed_seconds', round(time.monotonic()-clock), flush=True)
    entries.sort(key=lambda e: e['path'])
    content = [{k: e[k] for k in ('path', 'kind', 'sha256', 'link_text_sha256', 'error') if k in e}
               for e in entries]
    exception_rows = []
    for p in EXCEPTIONS:
        s = p.lstat()  # Metadata only: never read the known growing 4 GB log.
        exception_rows.append({'path': str(p), 'bytes': s.st_size, 'dev': s.st_dev, 'ino': s.st_ino,
                               'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
                               'uid': s.st_uid, 'hash_status': 'ACCEPTED_EXCEPTION_NOT_READ',
                               'reason': 'Known growing log' if p == EXCEPTIONS[-1]
                                         else 'Known unreadable path; descendants unverified'})
    out = {'started_utc': started, 'finished_utc': prep.now(), 'runtime_seconds': time.monotonic()-clock,
           'roots': list(map(str, (T, D, G))), 'excluded': list(map(str, excluded)),
           'entries': entries, 'errors': errors, 'accepted_exceptions': exception_rows,
           'content_inventory_sha256': prep.digest(prep.encode(content)),
           'regular_files': sum(e['kind'] == 'file' for e in entries),
           'regular_bytes': sum(e['bytes'] for e in entries if e['kind'] == 'file'),
           'note': 'Fresh byte hashes for all readable protected regular files; symlinks not followed. '
                   'The six expressly accepted paths are only lstat-ed, never opened or traversed.'}
    prep.write(name, out)
    print(json.dumps({k: out[k] for k in ('runtime_seconds', 'regular_files', 'regular_bytes',
                                         'content_inventory_sha256', 'errors')}), flush=True)
    require(not errors, 'Unexpected protected snapshot errors')


def check_file(row):
    p = canonical(row['path'])
    s = p.lstat()
    require(stat.S_ISREG(s.st_mode), 'File is not regular: ' + str(p))
    require((s.st_dev, s.st_ino, s.st_size, s.st_nlink, s.st_mtime_ns, s.st_ctime_ns) ==
            (row['dev'], row['ino'], row['bytes'], row['nlink'], row['mtime_ns'], row['ctime_ns']),
            'File metadata changed since preparation: ' + str(p))
    observed = prep.sha(p)
    require(observed == row['sha256'], 'File hash changed: ' + str(p))
    require(stable_stat(s) == stable_stat(p.lstat()), 'File changed during validation: ' + str(p))
    return {'path': str(p), 'sha256': observed, 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
            'dev': s.st_dev, 'ino': s.st_ino, 'nlink': s.st_nlink,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def cover_check(row):
    c = row['archive_cover']
    require(c['manifest'] in list(map(str, prep.ARCHIVES)), 'Unauthorized covering manifest')
    p = canonical(c['manifest'])
    raw = p.read_bytes()
    require(prep.digest(raw) == c['manifest_sha256'], 'Covering manifest changed')
    entry = json.loads(raw)['files'][c['entry_index']]
    require(all(entry[k] == c[k] for k in ('source', 'destination', 'bytes', 'sha256')),
            'Covering entry changed')
    dest = canonical(p.parent / entry['destination'])
    require(dest.is_relative_to(p.parent) and str(dest) == c['archive_path'], 'Unsafe archive cover')
    observed = prep.sha(dest)
    require(observed == entry['sha256'] == row['sha256'] and dest.stat().st_size == row['bytes'],
            'Current archive cover hash/size mismatch')
    return {'manifest': str(p), 'manifest_sha256': prep.digest(raw), 'entry_index': c['entry_index'],
            'archive_path': str(dest), 'sha256': observed, 'checked_utc': prep.now()}


def scope_inventory(m):
    files, directories, errors = set(), set(), []
    for root in m['scope']['scratch_roots']:
        if not os.path.lexists(root):
            continue
        for p, s in prep.walk(root, [], errors):
            if stat.S_ISDIR(s.st_mode):
                directories.add(str(p))
            else:
                require(stat.S_ISREG(s.st_mode), 'Nonregular entry in scratch: ' + str(p))
                files.add(str(p))
    files.update(str(p) for p in prep.PROMPTS if os.path.lexists(p))
    require(not errors, 'Scratch inventory error: ' + repr(errors))
    return files, directories


def preflight():
    m = manifest()
    files, directories = scope_inventory(m)
    require(files == {e['path'] for e in m['entries'] + m['kept']}, 'Scoped file inventory changed')
    require(directories == set(m['scratch_directories_before']), 'Scratch directory inventory changed')
    rows = []
    for e in m['entries'] + m['kept']:
        r = check_file(e)
        r.update(category=e['category'], label=e['label'])
        if e['category'] == 'archived_duplicate':
            r['archive_check'] = cover_check(e)
        if e['category'] == 'superseded_prompt':
            require(prep.sha(e['replacement_path']) == e['replacement_sha256'], 'Replacement prompt changed')
        rows.append(r)
    for e in m['archive_entries']:
        require(prep.sha(e['archive_path']) == e['sha256'] and
                Path(e['archive_path']).stat().st_size == e['bytes'], 'Archive entry changed')
        require(prep.sha(e['manifest']) == e['manifest_sha256'], 'Archive manifest changed')
    before = prep.load('protected_hash_before_final.json')
    require(not before['errors'], 'Protected before snapshot failed')
    out = {'observed_utc': prep.now(), 'status': 'PASS', 'manifest_sha256': prep.sha(R/'deletion_manifest.json'),
           'files': rows, 'candidate_count': 449, 'kept_count': 32,
           'scratch_directory_count': len(directories), 'archive_entry_count': len(m['archive_entries']),
           'errors': []}
    prep.write('validation_pre_delete_final.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}), flush=True)
    return m


def completion(m):
    started = prep.now()
    cutoff_ns = int(CUTOFF.timestamp()) * 1000000000
    labels = []
    for root in m['scope']['scratch_roots']:
        label = canonical(Path(root).parent)
        exit_path = label / 'exit'
        expected = next(e for e in m['label_exit_evidence'] if e['label'] == str(label))
        require(prep.sha(exit_path) == expected['exit_sha256'], 'Pi exit missing/changed: ' + str(label))
        errors = []
        files = [(p, s) for p, s in prep.walk(label, [], errors) if stat.S_ISREG(s.st_mode)]
        require(not errors and files, 'Cannot inspect label: ' + str(label))
        p, s = max(files, key=lambda pair: pair[1].st_mtime_ns)
        latest_ctime = max(x.st_ctime_ns for _, x in files)
        require(s.st_mtime_ns <= cutoff_ns and latest_ctime <= cutoff_ns,
                'Label received a write/metadata update after 2026-11-07T08:05:00Z: ' + str(label))
        labels.append({'label': label.name, 'exit_path': str(exit_path), 'exit_sha256': expected['exit_sha256'],
                       'exit_text': exit_path.read_text().strip(), 'file_count': len(files),
                       'newest_file': str(p), 'newest_file_mtime_utc': datetime.datetime.fromtimestamp(
                           s.st_mtime, datetime.timezone.utc).isoformat(),
                       'newest_file_mtime_ns': s.st_mtime_ns, 'latest_file_ctime_ns': latest_ctime,
                       'age_seconds': time.time() - s.st_mtime})
    launchers, monitors = [], []
    for gen, name in MONITORS:
        launcher = T / 'h200_mainline' / ('p2_64_formal_retry' + gen) / 'package_01/launcher_exit.json'
        h = prep.sha(launcher)
        expected = next(e for e in m['completion_evidence'] if e.get('generation') == gen)
        require(h == expected['sha256'], 'Launcher completion changed')
        data = json.loads(launcher.read_text())
        require(datetime.datetime.fromisoformat(data['end_utc']) <= CUTOFF, 'Launcher finished after cutoff')
        launchers.append({'generation': gen, 'path': str(launcher), 'sha256': h,
                          'exit_code': data['exit_code'], 'end_utc': data['end_utc']})
        final = D / name / 'scratch/monitor/final.json'
        row = next(e for e in m['entries'] + m['kept'] if e['path'] == str(final))
        observed = check_file(row)
        data = json.loads(final.read_text())
        require(not data['errors'] and data['job']['phase'] == 'Failed' and data['job']['finishTime'],
                'Monitor completion absent/invalid')
        monitors.append({'generation': gen, **observed, 'phase': data['job']['phase'],
                         'finish_time': data['job']['finishTime'], 'finished_utc': data['finished_utc']})
    out = {'started_utc': started, 'finished_utc': prep.now(), 'status': 'PASS_LEAD_ACCEPTED_COMPLETION_EVIDENCE',
           'cutoff_utc': CUTOFF.isoformat(), 'labels': labels, 'launchers': launchers, 'monitors': monitors,
           'host_process_scan': 'NOT_RUN: isolated PID namespace; assignment_2.md explicitly accepts completion evidence',
           'observation': 'p64-formal5-prepare-1104b last wrote on 2026-11-03; all other labels last wrote '
                          'on 2026-11-07, 05:39:28Z–08:00:22Z. All are before the stop cutoff.'}
    prep.write('completion_check_immediate_final.json', out)
    return out


def append_event(stream, event):
    stream.write(json.dumps(event, sort_keys=True) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def unlink_checked(row, journal):
    p = canonical(row['path'])
    parent = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Nonregular/hardlinked deletion input')
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (row['dev'], row['ino'], row['bytes'], row['mtime_ns'], row['ctime_ns']), 'Deletion input changed')
        archive = cover_check(row) if row['category'] == 'archived_duplicate' else None
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
        observed = h.hexdigest()
        require(observed == row['sha256'], 'Hash changed immediately before unlink: ' + str(p))
        require(stable_stat(before) == stable_stat(os.fstat(fd)) ==
                stable_stat(os.stat(p.name, dir_fd=parent, follow_symlinks=False)), 'Deletion input is unstable')
        event = {'event': 'unlink_intent', 'checked_utc': prep.now(), 'path': str(p),
                 'category': row['category'], 'label': row['label'], 'kind': 'file',
                 'bytes': before.st_size, 'allocated_bytes': before.st_blocks * 512,
                 'dev': before.st_dev, 'ino': before.st_ino, 'nlink': before.st_nlink,
                 'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns,
                 'manifest_sha256': row['sha256'], 'observed_sha256_at_deletion': observed,
                 'archive_check': archive, 'operation': 'os.unlink(name, dir_fd=parent_fd)'}
        append_event(journal, event)
        # Recheck identity/stability after the durable intent and immediately before unlink.
        require(stable_stat(before) == stable_stat(os.fstat(fd)) ==
                stable_stat(os.stat(p.name, dir_fd=parent, follow_symlinks=False)), 'Input changed after intent')
        os.unlink(p.name, dir_fd=parent)
        require(not os.path.lexists(p), 'File remains after unlink')
        event.update(event='deleted', deleted_utc=prep.now(), absent_after=True)
        append_event(journal, event)
        return event
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def execute():
    require(not (R / 'deletion_receipt_final.json').exists() and
            not (R / 'deletion_journal_final.jsonl').exists(), 'Existing execution evidence; do not replay deletions')
    m = preflight()
    done = completion(m)  # Final gate: no unrelated work between this and the unlink loop.
    receipt = {'schema_version': 2, 'started_utc': prep.now(), 'status': 'IN_PROGRESS', 'outcome': 'partial',
               'manifest_path': str(R / 'deletion_manifest.json'), 'manifest_sha256': prep.sha(R/'deletion_manifest.json'),
               'assignment_path': str(R / 'assignment_2.md'), 'assignment_sha256': prep.sha(R/'assignment_2.md'),
               'deleted': [], 'kept': [], 'removed_directories': [], 'errors': [], 'completion_check': done,
               'accepted_exception_paths': list(map(str, EXCEPTIONS))}
    prep.write('deletion_receipt_final.json', receipt)
    with (R / 'deletion_journal_final.jsonl').open('x') as journal:
        try:
            for row in m['entries']:
                receipt['deleted'].append(unlink_checked(row, journal))
            affected = set()
            for row in receipt['deleted']:
                if row['category'] == 'superseded_prompt':
                    continue
                root = D / row['label'] / 'scratch'
                p = Path(row['path']).parent
                while p.is_relative_to(root):
                    affected.add(str(p))
                    if p == root:
                        break
                    p = p.parent
            for name in sorted(affected, key=lambda s: (-len(Path(s).parts), s)):
                p = canonical(name)
                require(name in m['scratch_directories_before'], 'Directory was not in frozen scope')
                s = p.lstat()
                require(stat.S_ISDIR(s.st_mode), 'Unsafe directory')
                if list(p.iterdir()):
                    continue
                event = {'event': 'rmdir_intent', 'path': name, 'label': p.relative_to(D).parts[0],
                         'dev': s.st_dev, 'ino': s.st_ino, 'bytes': s.st_size,
                         'allocated_bytes': s.st_blocks * 512, 'checked_utc': prep.now(),
                         'reason': 'Scratch ancestor made empty by the listed file deletions'}
                append_event(journal, event)
                p.rmdir()
                event.update(event='directory_removed', removed_utc=prep.now(), absent_after=not os.path.lexists(p))
                append_event(journal, event)
                receipt['removed_directories'].append(event)
            for row in m['kept']:
                receipt['kept'].append({**row, 'outcome': 'kept', 'kept_reason': row['reason'],
                                        'observed_after': check_file(row), 'kept_file_unchanged': True})
            files, directories = scope_inventory(m)
            require(files == {e['path'] for e in m['kept']}, 'Unexpected remaining scratch files/prompts')
            require(directories == set(m['scratch_directories_before']) -
                    {e['path'] for e in receipt['removed_directories']}, 'Unexpected directory removal')
            receipt.update(status='DELETED_PENDING_PROTECTED_AFTER_CHECK', outcome='partial')
        except Exception as error:
            receipt['errors'].append({'error': str(error), 'observed_utc': prep.now()})
            receipt['status'] = 'PARTIAL_EXECUTION_ERROR'
            append_event(journal, {'event': 'execution_error', **receipt['errors'][-1]})
            raise
        finally:
            receipt['finished_utc'] = prep.now()
            receipt['deleted_file_count'] = len(receipt['deleted'])
            receipt['deleted_bytes'] = sum(e['bytes'] for e in receipt['deleted'])
            receipt['deleted_allocated_bytes'] = sum(e['allocated_bytes'] for e in receipt['deleted'])
            receipt['deleted_summary'] = prep.summarize(receipt['deleted'])
            receipt['kept_summary'] = prep.summarize(receipt['kept'])
            receipt['kept_count'] = len(receipt['kept'])
            receipt['kept_bytes'] = sum(e['bytes'] for e in receipt['kept'])
            receipt['kept_allocated_bytes'] = sum(e['allocated_bytes'] for e in receipt['kept'])
            receipt['removed_directory_count'] = len(receipt['removed_directories'])
            receipt['removed_directory_summary'] = {}
            for entry in receipt['removed_directories']:
                counts = receipt['removed_directory_summary'].setdefault(entry['label'],
                    {'directories': 0, 'bytes': 0, 'allocated_bytes': 0})
                counts['directories'] += 1
                counts['bytes'] += entry['bytes']
                counts['allocated_bytes'] += entry['allocated_bytes']
            receipt['directory_bytes_at_removal'] = sum(e['bytes'] for e in receipt['removed_directories'])
            receipt['directory_allocated_bytes_at_removal'] = sum(e['allocated_bytes'] for e in receipt['removed_directories'])
            prep.write('deletion_receipt_final.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'deleted_file_count', 'deleted_bytes',
                                             'deleted_allocated_bytes', 'removed_directory_count')}), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'initialize':
        immutable_inputs()
        print('PASS: original record hashes saved; report backed up')
    elif cmd == 'snapshot-before':
        snapshot('protected_hash_before_final.json')
    elif cmd == 'snapshot-after':
        snapshot('protected_hash_after_final.json')
    elif cmd == 'preflight':
        preflight()
    elif cmd == 'execute':
        execute()
    else:
        raise SystemExit('Unknown command: ' + cmd)
