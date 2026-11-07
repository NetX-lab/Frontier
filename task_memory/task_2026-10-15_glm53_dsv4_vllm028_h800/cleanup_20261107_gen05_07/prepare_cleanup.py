#!/usr/bin/env python3
"""Read-only cleanup preparation; ds4_runs and protected roots are never mutated.

All output is written beside this script. Deletion is withheld because this
executor has no host-process view, as documented in progress.md:611. No deletion
entry points are provided: Claude must satisfy the prerequisites before removal.
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

RECORD = Path(__file__).resolve().parent
TASK = RECORD.parent
DS4 = Path('/data/ycfeng/tmp/ds4_runs')
GLM = Path('/data/ycfeng/tmp/glm53_dsv4')
PROMPTS = [DS4 / 'prompts' / ('p64-formal7-post-1107n.' + ext) for ext in ('sh', 'md')]
ARCHIVES = [TASK / 'h200_mainline' / ('p2_64_formal_retry' + gen) /
            'terminal_20261107/archive_manifest.json' for gen in ('05', '06', '07')]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def write(name, value):
    (RECORD / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load(name):
    return json.loads((RECORD / name).read_text())


def under(path, root):
    return path == root or path.startswith(root + '/')


def sha(path):
    # Reject symlinks and unstable inputs rather than silently hashing a target.
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Not a regular file: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    h = hashlib.sha256()
    with os.fdopen(fd, 'rb') as f:
        opened = os.fstat(f.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError('Input identity changed: ' + str(path))
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        after = os.fstat(f.fileno())
    observed = os.lstat(path)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(observed):
        raise ValueError('Input changed while hashing: ' + str(path))
    return h.hexdigest()


def walk(root, excluded, errors):
    """Walk actual directory entries without following any symlink."""
    root = Path(root)
    if any(under(str(root), str(p)) for p in excluded):
        return
    try:
        s = root.lstat()
    except OSError as e:
        errors.append({'path': str(root), 'operation': 'lstat', 'error': str(e)})
        return
    yield root, s
    if stat.S_ISDIR(s.st_mode):
        try:
            with os.scandir(root) as entries:
                children = sorted((Path(e.path) for e in entries), key=str)
        except OSError as e:
            errors.append({'path': str(root), 'operation': 'scandir', 'error': str(e)})
            return
        for child in children:
            yield from walk(child, excluded, errors)


def info(path, s):
    out = {'path': str(path), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
           'dev': s.st_dev, 'ino': s.st_ino, 'nlink': s.st_nlink,
           'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
           'mode': stat.S_IMODE(s.st_mode)}
    if stat.S_ISREG(s.st_mode):
        out['kind'] = 'file'
        try:
            out['sha256'] = sha(path)
        except (OSError, ValueError) as e:
            out['error'] = str(e)
    elif stat.S_ISDIR(s.st_mode):
        out['kind'] = 'directory'
    elif stat.S_ISLNK(s.st_mode):
        out['kind'] = 'symlink'
        out['target'] = os.readlink(path)
        out['link_text_sha256'] = digest(os.fsencode(out['target']))
    else:
        out['kind'] = 'special'
    return out


def labels():
    return sorted((p for gen in ('5', '6', '7') for p in DS4.glob('p64-formal' + gen + '-*')), key=str)


def summarize(rows):
    categories = {}
    label_stats = {}
    for row in rows:
        c = row['category']
        for stats, key in ((categories, c), (label_stats, row['label'])):
            counts = stats.setdefault(key, {'files': 0, 'bytes': 0, 'allocated_bytes': 0})
            counts['files'] += row['kind'] == 'file'
            counts['bytes'] += row['bytes'] if row['kind'] == 'file' else 0
            counts['allocated_bytes'] += row['allocated_bytes'] if row['kind'] == 'file' else 0
    return {'categories': categories, 'labels': label_stats}


def prepare():
    if (RECORD / 'deletion_manifest.json').exists():
        raise RuntimeError('Preparation already exists; preserve the original plan')
    archive_entries = []
    index = {}
    archive_errors = []
    for manifest in ARCHIVES:
        raw = manifest.read_bytes()
        data = json.loads(raw)
        for number, entry in enumerate(data['files']):
            row = {'manifest': str(manifest), 'manifest_sha256': digest(raw),
                   'entry_index': number, **entry}
            dest = manifest.parent / entry['destination']
            row['archive_path'] = str(dest)
            # Evidence destinations must be canonical regular files within this archive.
            if (Path(entry['destination']).is_absolute() or
                    not under(str(dest.resolve()), str(manifest.parent.resolve())) or
                    str(dest.resolve()) != str(dest)):
                row['error'] = 'Noncanonical or external archive destination'
            else:
                try:
                    observed_sha = sha(dest)
                    observed_bytes = dest.stat().st_size
                    row.update(observed_sha256=observed_sha, observed_bytes=observed_bytes)
                    if observed_sha != entry['sha256'] or observed_bytes != entry['bytes']:
                        row['error'] = 'Archive file does not match manifest hash/size'
                except (OSError, ValueError) as e:
                    row['error'] = str(e)
            archive_entries.append(row)
            if 'error' in row:
                archive_errors.append(row)
            else:
                index.setdefault((row['sha256'], row['bytes']), []).append(row)
    all_rows = []
    directories = []
    label_evidence = []
    inventory_errors = []
    for label in labels():
        if label.is_symlink() or not label.is_dir() or str(label.resolve()) != str(label):
            raise RuntimeError('Unsafe label path: ' + str(label))
        exit_file = label / 'exit'
        label_evidence.append({'label': str(label), 'exit_path': str(exit_file),
                               'exit_sha256': sha(exit_file), 'exit_text': exit_file.read_text()})
        for p, s in walk(label / 'scratch', [], inventory_errors):
            row = info(p, s)
            if row['kind'] == 'directory':
                directories.append(str(p))
                continue
            row['label'] = label.name
            row['scratch_relative_path'] = str(p.relative_to(label / 'scratch'))
            covers = index.get((row.get('sha256'), row['bytes']), [])
            if row['kind'] != 'file' or 'error' in row:
                row.update(category='unclassified', intended_action='keep',
                           reason=row.get('error', 'Nonregular entry; no authorized classification'))
            elif 'node-compile-cache' in p.relative_to(label / 'scratch').parts[:-1]:
                row.update(category='node_compile_cache', intended_action='delete',
                           reason='Regenerable Node cache within the scratch of a finished scoped Pi label')
            elif covers:
                row.update(category='archived_duplicate', intended_action='delete',
                           reason='Current content equals a manifest hash and the archived destination hash',
                           archive_cover=covers[0], archive_cover_count=len(covers))
                row['recorded_source_path_equal'] = row['path'] == covers[0]['source']
            else:
                row.update(category='uncovered_scratch', intended_action='keep',
                           reason='Current SHA256 is absent from all valid archive entries; outside node-compile-cache')
            all_rows.append(row)
    for prompt in PROMPTS:
        row = info(prompt, prompt.lstat())
        if row['kind'] != 'file' or 'error' in row:
            raise RuntimeError('Unsafe prompt candidate: ' + str(prompt))
        replacement = prompt.with_name(prompt.name.replace('formal7-post-1107n', 'formal8-post-1107r'))
        row.update(label='prompts', category='superseded_prompt', intended_action='delete',
                   reason='Explicitly authorized never-launched generation07 post-run prompt, replaced by generation08',
                   replacement_path=str(replacement), replacement_sha256=sha(replacement))
        all_rows.append(row)
    completion = []
    for label in labels():
        monitor = label / 'scratch/monitor/monitor.jsonl'
        if monitor.is_file():
            lines = monitor.read_text().splitlines()
            last = json.loads(lines[-1])
            completion.append({'label': label.name, 'path': str(monitor), 'sha256': sha(monitor),
                               'row_count': len(lines), 'last_row': {k: last.get(k) for k in
                               ('observed_utc', 'final', 'phase', 'launcher_alive', 'finishTime')}})
    for gen in ('05', '06', '07'):
        p = TASK / 'h200_mainline' / ('p2_64_formal_retry' + gen) / 'package_01/launcher_exit.json'
        data = json.loads(p.read_text())
        completion.append({'generation': gen, 'path': str(p), 'sha256': sha(p),
                           'exit_code': data['exit_code'], 'end_utc': data['end_utc'],
                           'launcher_pid': data['launcher_pid']})
    manifest = {'schema_version': 1, 'prepared_utc': now(), 'status': 'WITHHELD_PREREQUISITES',
                'working_root': '/data/ycfeng', 'record_directory': str(RECORD),
                'assignment_sha256': sha(RECORD / 'assignment.md'),
                'scope': {'scratch_roots': [str(p / 'scratch') for p in labels()],
                          'prompt_files': list(map(str, PROMPTS))},
                'archive_entries': archive_entries, 'archive_errors': archive_errors,
                'inventory_errors': inventory_errors, 'label_exit_evidence': label_evidence,
                'completion_evidence': completion,
                'entries': [r for r in all_rows if r['intended_action'] == 'delete'],
                'kept': [r for r in all_rows if r['intended_action'] == 'keep'],
                'scratch_directories_before': directories,
                'planned_summary': summarize([r for r in all_rows if r['intended_action'] == 'delete']),
                'kept_summary': summarize([r for r in all_rows if r['intended_action'] == 'keep']),
                'note': 'This is a proposed deletion manifest. No listed file has been deleted.'}
    write('deletion_manifest.json', manifest)
    print(json.dumps({'archive_entries': len(archive_entries), 'archive_errors': len(archive_errors),
                      'candidate_files': len(manifest['entries']), 'kept_files': len(manifest['kept']),
                      'planned_summary': manifest['planned_summary'], 'inventory_errors': inventory_errors}))


def snapshot(name):
    t = time.monotonic()
    errors = []
    entries = []
    # Exclude only authorized scratch subtrees and the two prompt files.
    excluded = [RECORD, *PROMPTS, *(p / 'scratch' for p in labels())]
    todo = [(p, s) for root in (TASK, DS4, GLM) for p, s in walk(root, excluded, errors)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for number, row in enumerate(pool.map(lambda pair: info(*pair), todo), 1):
            entries.append(row)
            if 'error' in row:
                errors.append({'path': row['path'], 'operation': 'sha256', 'error': row['error']})
            if number % 10000 == 0:
                print(name, 'hashed/inventoried', number, 'elapsed_seconds', round(time.monotonic()-t), flush=True)
    entries.sort(key=lambda r: r['path'])
    # Directories have no content bytes; use an inventory of pathname/kind. Hash
    # regular bytes and link text; preserve separate metadata for review.
    content = [{k: r[k] for k in ('path', 'kind', 'sha256', 'link_text_sha256', 'error') if k in r}
               for r in entries]
    value = {'started_utc': datetime.datetime.fromtimestamp(time.time()-(time.monotonic()-t),
                                                          datetime.timezone.utc).isoformat(),
             'finished_utc': now(), 'runtime_seconds': time.monotonic()-t,
             'roots': list(map(str, (TASK, DS4, GLM))), 'excluded': list(map(str, excluded)),
             'entries': entries, 'errors': errors, 'content_inventory_sha256': digest(encode(content)),
             'regular_files': sum(r['kind'] == 'file' for r in entries),
             'regular_bytes': sum(r['bytes'] for r in entries if r['kind'] == 'file'),
             'note': 'Symlinks are not followed; link text is hashed. Unreadable directories are inventoried '
                     'by lstat only; their unknown descendants are not claimed hash-verified.'}
    write(name, value)
    print(json.dumps({k: value[k] for k in ('runtime_seconds', 'regular_files', 'regular_bytes',
                                           'content_inventory_sha256', 'errors')}), flush=True)


def process_scan():
    m = load('deletion_manifest.json')
    targets = [e['path'] for e in m['entries']]
    scratch = m['scope']['scratch_roots']
    hits = []
    errors = []
    races = 0
    scanned = 0
    own = os.getpid()
    def inspect_path(pid, kind, value):
        value = value.removesuffix(' (deleted)')
        for target in targets:
            if under(value, target):
                hits.append({'pid': pid, 'kind': kind, 'target': target})
        for target in scratch:
            if under(value, target):
                hits.append({'pid': pid, 'kind': kind, 'scratch_root_reference': target})
    def operation(pid, path, fn):
        nonlocal races
        try:
            return fn(path)
        except (FileNotFoundError, ProcessLookupError):
            races += 1
        except OSError as e:
            errors.append({'pid': pid, 'proc_path': str(path), 'error': str(e)})
        return None
    for entry in sorted(Path('/proc').iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        pid = int(entry.name)
        scanned += 1
        for kind in ('cwd', 'root', 'exe'):
            value = operation(pid, entry / kind, os.readlink)
            if value:
                inspect_path(pid, kind, value)
        fds = operation(pid, entry / 'fd', lambda p: list(p.iterdir())) or []
        for fd in fds:
            value = operation(pid, fd, os.readlink)
            if value:
                inspect_path(pid, 'fd', value)
        maps = operation(pid, entry / 'maps', lambda p: p.read_text()) or ''
        for line in maps.splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6:
                inspect_path(pid, 'maps', parts[5])
        cmd = operation(pid, entry / 'cmdline', lambda p: p.read_bytes()) or b''
        # Do not store command arguments, environment, or credentials.
        for token in cmd.split(b'\0'):
            argument = os.fsdecode(token)
            for target in targets + scratch:
                start = argument.find(target)
                if start != -1:
                    tail = argument[start + len(target):]
                    if not tail or tail[0] in '/ \t\n"\'=:,;':
                        hits.append({'pid': pid, 'kind': 'cmdline', 'target': target})
    ns = os.readlink('/proc/self/ns/pid')
    evidence = {'observed_utc': now(), 'own_pid': own, 'pid_namespace': ns,
                'pid1_comm': Path('/proc/1/comm').read_text().strip(),
                'visible_processes_scanned': scanned, 'hits': hits, 'read_errors': errors,
                'process_exit_races': races,
                'status': 'BLOCKED_NO_HOST_PROCESS_VISIBILITY',
                'local_zero_references': not hits and not errors,
                'host_check': 'NOT_RUN: /proc is the isolated bwrap PID namespace; '
                              'no host execution tool was provided in the assignment.',
                'source_evidence': str(TASK / 'progress.md') + ':611',
                'note': 'Local absence plus historical completion records does not establish '
                        'the required current host-wide no-reference prerequisite.'}
    write('process_check.json', evidence)
    print(json.dumps(evidence), flush=True)


def finalize():
    m = load('deletion_manifest.json')
    before = load('protected_hash_before.json')
    after = load('protected_hash_after.json')
    proc = load('process_check.json')
    protected_before = {r['path']: r for r in before['entries']}
    protected_after = {r['path']: r for r in after['entries']}
    fields = ('kind', 'sha256', 'link_text_sha256', 'error')
    missing = sorted(protected_before.keys() - protected_after.keys())
    added = sorted(protected_after.keys() - protected_before.keys())
    changed = [{'path': p, 'before': {k: protected_before[p].get(k) for k in fields},
                'after': {k: protected_after[p].get(k) for k in fields}}
               for p in sorted(protected_before.keys() & protected_after.keys())
               if any(protected_before[p].get(k) != protected_after[p].get(k) for k in fields)]
    results = []
    failures = []
    for row in m['entries'] + m['kept']:
        copy = dict(row)
        copy.update(outcome='kept', actual_action='none')
        if row['intended_action'] == 'delete':
            copy['kept_reason'] = ('Deletion withheld: required host cwd/open-file/argument scan unavailable; '
                                   'full protected-path SHA256 coverage also incomplete (unreadable or unstable protected inputs).')
        else:
            copy['kept_reason'] = row['reason']
        try:
            current = info(Path(row['path']), os.lstat(row['path']))
            unchanged = (row['kind'], row.get('sha256'), row.get('link_text_sha256'), row['dev'], row['ino']) == (
                current['kind'], current.get('sha256'), current.get('link_text_sha256'), current['dev'], current['ino'])
            copy['kept_file_unchanged'] = unchanged
            if not unchanged or 'error' in current:
                failures.append({'path': row['path'], 'error': 'Retained entry changed or unreadable', 'current': current})
        except OSError as e:
            failures.append({'path': row['path'], 'error': str(e)})
            copy['kept_file_unchanged'] = False
        results.append(copy)
    archived_failures = []
    for entry in m['archive_entries']:
        try:
            if sha(entry['archive_path']) != entry['sha256']:
                archived_failures.append({'path': entry['archive_path'], 'error': 'SHA256 differs from manifest'})
            if sha(entry['manifest']) != entry['manifest_sha256']:
                archived_failures.append({'path': entry['manifest'], 'error': 'Manifest changed'})
        except (OSError, ValueError) as e:
            archived_failures.append({'path': entry['archive_path'], 'error': str(e)})
    receipt = {'schema_version': 1, 'finished_utc': now(), 'outcome': 'blocked',
               'status': 'NO_DELETIONS_PREREQUISITES_UNAVAILABLE',
               'manifest_path': str(RECORD / 'deletion_manifest.json'),
               'manifest_sha256': sha(RECORD / 'deletion_manifest.json'),
               'deleted': [], 'deleted_file_count': 0, 'deleted_bytes': 0,
               'deleted_allocated_bytes': 0, 'removed_directories': [],
               'kept': results, 'kept_count': len(results), 'planned_summary': m['planned_summary'],
               'kept_file_failures': failures, 'process_check': proc,
               'protected_check': {'status': 'PARTIAL_UNREADABLE_OR_UNSTABLE_INPUTS',
                                   'metadata_changed': [{'path': p,
                                       'before': {k: protected_before[p][k] for k in ('bytes', 'dev', 'ino', 'mtime_ns', 'ctime_ns')},
                                       'after': {k: protected_after[p][k] for k in ('bytes', 'dev', 'ino', 'mtime_ns', 'ctime_ns')}}
                                       for p in sorted(protected_before.keys() & protected_after.keys())
                                       if protected_before[p]['kind'] == 'file' and any(
                                           protected_before[p][k] != protected_after[p][k]
                                           for k in ('bytes', 'dev', 'ino', 'mtime_ns', 'ctime_ns'))],
                                   'before_inventory_sha256': before['content_inventory_sha256'],
                                   'after_inventory_sha256': after['content_inventory_sha256'],
                                   'hash_matched_regular_files': sum(protected_before[p]['kind'] == 'file' and
                                       protected_before[p].get('sha256') == protected_after[p].get('sha256') and
                                       'sha256' in protected_before[p] for p in protected_before.keys() & protected_after.keys()),
                                   'before_errors': before['errors'], 'after_errors': after['errors'],
                                   'missing': missing, 'added': added, 'changed': changed},
               'archive_check': {'entries': len(m['archive_entries']), 'failures': archived_failures},
               'blocking_prerequisites': ['Current read-only host PID namespace scan of candidate files '
                                         'and ancestor directories for cwd/fd/maps/cmdline references',
                                         'Read access for all denied protected files/directories and a stable '
                                         'baseline for the changing protected launcher.log, or an explicit '
                                         'narrower verification requirement'],
               'note': 'All candidates and all uncovered scratch files remain. No unlink/rmdir/mv/Git/GPU/'
                       'platform/network-write operation was executed.'}
    write('deletion_receipt.json', receipt)
    print(json.dumps({'outcome': receipt['outcome'], 'kept_count': len(results), 'deleted_file_count': 0,
                      'kept_file_failures': failures, 'archive_failures': archived_failures,
                      'protected_check': receipt['protected_check']}), flush=True)


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'prepare':
        prepare()
    elif command in ('protected_hash_before.json', 'protected_hash_after.json'):
        snapshot(command)
    elif command == 'process':
        process_scan()
    elif command == 'finalize':
        finalize()
    else:
        raise SystemExit('Unknown command: ' + command)
