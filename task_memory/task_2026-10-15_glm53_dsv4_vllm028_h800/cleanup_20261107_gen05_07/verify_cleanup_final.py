#!/usr/bin/env python3
"""Independent, read-only scope/archive audit plus final receipt validation.

Only final receipt/verification records in this directory are updated. This
command cannot remove anything and is safe for Claude to rerun.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

R = Path(__file__).resolve().parent
T = R.parent
D = Path('/data/ycfeng/tmp/ds4_runs')


def sha(p):
    before = os.lstat(p)
    if not stat.S_ISREG(before.st_mode) or str(Path(p).resolve()) != str(p):
        raise RuntimeError('Noncanonical/nonregular audit file: ' + str(p))
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = os.lstat(p)
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after):
        raise RuntimeError('Audit file changed during hashing: ' + str(p))
    return h.hexdigest()


def load(name):
    return json.loads((R / name).read_text())


def write(name, data):
    (R / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def totals(rows):
    groups = {'categories': {}, 'labels': {}}
    for row in rows:
        for field, dimension in [('category', 'categories'), ('label', 'labels')]:
            target = groups[dimension].setdefault(row[field], {'files': 0, 'bytes': 0, 'allocated_bytes': 0})
            target['files'] += 1
            target['bytes'] += row['bytes']
            target['allocated_bytes'] += row['allocated_bytes']
    return groups


def main():
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    m = load('deletion_manifest.json')
    receipt = load('deletion_receipt_final.json')
    inputs = load('record_inputs_before_final.json')
    before, after = load('protected_hash_before_final.json'), load('protected_hash_after_final.json')
    expected_deleted = {e['path']: e for e in m['entries']}
    expected_kept = {e['path']: e for e in m['kept']}
    deleted = {e['path']: e for e in receipt['deleted']}
    kept = {e['path']: e for e in receipt['kept']}
    check(len(receipt['deleted']) == len(deleted) == 449 and set(deleted) == set(expected_deleted),
          'Receipt deletion list differs from the exact 449 manifest candidates')
    check(len(receipt['kept']) == len(kept) == 32 and set(kept) == set(expected_kept),
          'Receipt keep list differs from the exact 32 uncovered files')
    check(not receipt['errors'], 'Execution errors recorded')
    check(sha(R/'deletion_manifest.json') == receipt['manifest_sha256'], 'Manifest differs from final receipt')
    check(sha(R/'assignment_2.md') == receipt['assignment_sha256'], 'Correction assignment changed')
    immutable_checked = []
    for row in inputs['entries']:
        if row['path'] == str(R/'report.md'):
            check(sha(R/'report_before_execution.md') == row['sha256'], 'Prior report backup differs')
            continue
        check(sha(row['path']) == row['sha256'], 'Pre-existing record changed: ' + row['path'])
        immutable_checked.append(row['path'])

    actual_files, actual_dirs = set(), set()
    for root in m['scope']['scratch_roots']:
        if not os.path.lexists(root):
            continue
        check(not Path(root).is_symlink(), 'Scratch root became a symlink')
        for parent, dirs, files in os.walk(root, followlinks=False):
            actual_dirs.add(parent)
            for name in dirs:
                check(not (Path(parent)/name).is_symlink(), 'Symlink in retained scratch')
            for name in files:
                actual_files.add(str(Path(parent)/name))
    actual_files.update(str(p) for p in map(Path, m['scope']['prompt_files']) if os.path.lexists(p))
    check(actual_files == set(expected_kept), 'Remaining scope is not exactly the 32 retained files')
    for p, row in deleted.items():
        check(not os.path.lexists(p), 'Deleted pathname remains: ' + p)
        check(row['observed_sha256_at_deletion'] == row['manifest_sha256'] == expected_deleted[p]['sha256'],
              'Recorded deletion hash mismatch: ' + p)
        check(row['absent_after'], 'Deletion did not record absence: ' + p)
        check(row['bytes'] == expected_deleted[p]['bytes'] and row['nlink'] == 1,
              'Recorded deletion size/link count mismatch: ' + p)
        if row['category'] == 'archived_duplicate':
            cover = expected_deleted[p]['archive_cover']
            observed = row['archive_check']
            check(observed['archive_path'] == cover['archive_path'] and
                  observed['sha256'] == expected_deleted[p]['sha256'] and
                  observed['manifest_sha256'] == cover['manifest_sha256'] and
                  observed['entry_index'] == cover['entry_index'], 'Deletion cover evidence mismatch: ' + p)
    for p, row in expected_kept.items():
        check(sha(p) == row['sha256'], 'Kept file hash changed: ' + p)
        s = os.lstat(p)
        check((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) ==
              (row['dev'], row['ino'], row['bytes'], row['mtime_ns'], row['ctime_ns']),
              'Kept file identity/metadata changed: ' + p)
        check(kept[p]['kept_file_unchanged'] and kept[p]['kept_reason'] == row['reason'],
              'Missing keep reason or preservation check: ' + p)

    archive_checks = []
    for entry in m['archive_entries']:
        archive_path = Path(entry['archive_path'])
        raw = Path(entry['manifest']).read_bytes()
        source = json.loads(raw)['files'][entry['entry_index']]
        h = sha(archive_path)
        check(hashlib.sha256(raw).hexdigest() == entry['manifest_sha256'], 'Archive manifest changed')
        check(h == source['sha256'] == entry['sha256'] and archive_path.stat().st_size == source['bytes'],
              'Archive preservation failed: ' + str(archive_path))
        check(str(Path(entry['manifest']).parent/source['destination']) == str(archive_path), 'Archive path differs')
        archive_checks.append({'path': str(archive_path), 'sha256': h})

    events = [json.loads(line) for line in (R/'deletion_journal_final.jsonl').read_text().splitlines()]
    event_groups = {}
    for event in events:
        event_groups.setdefault(event['event'], []).append(event)
    for kind in ('unlink_intent', 'deleted'):
        rows = event_groups.get(kind, [])
        check(len(rows) == 449 and {e['path'] for e in rows} == set(expected_deleted),
              'Journal is missing exact deletion events: ' + kind)
        for event in rows:
            check(event['observed_sha256_at_deletion'] == expected_deleted[event['path']]['sha256'],
                  'Journal deletion hash mismatch')
    check(not event_groups.get('execution_error'), 'Execution error in journal')
    removed = {e['path'] for e in receipt['removed_directories']}
    check(len(removed) == len(receipt['removed_directories']), 'Duplicate removed directory')
    check(removed <= set(m['scratch_directories_before']), 'Removed directory outside original scratch inventory')
    expected_empty = {p for p in m['scratch_directories_before']
                      if any(Path(e['path']).is_relative_to(p) for e in m['entries'])
                      and not any(Path(e['path']).is_relative_to(p) for e in m['kept'])}
    check(removed == expected_empty, 'Did not remove exactly the scratch ancestors made empty by deletions')
    check(actual_dirs == set(m['scratch_directories_before']) - removed, 'Directory disappearance differs from receipt')
    for p in removed:
        check(not os.path.lexists(p), 'Removed directory still exists')
        check(any(Path(row['path']).is_relative_to(p) for row in receipt['deleted']
                  if row['category'] != 'superseded_prompt'), 'Removed directory not affected by deletions')
    for kind in ('rmdir_intent', 'directory_removed'):
        check(len(event_groups.get(kind, [])) == len(removed) and
              {e['path'] for e in event_groups.get(kind, [])} == removed, 'Directory journal mismatch: ' + kind)

    # Recompute snapshot content inventories, then compare every included path/hash.
    maps = []
    for snapshot, name in [(before, 'before'), (after, 'after')]:
        check(not snapshot['errors'], 'Unexpected protected snapshot errors: ' + name)
        entries = snapshot['entries']
        content = [{k: e[k] for k in ('path', 'kind', 'sha256', 'link_text_sha256', 'error') if k in e}
                   for e in entries]
        raw = (json.dumps(content, sort_keys=True, separators=(',', ':'))+'\n').encode()
        check(hashlib.sha256(raw).hexdigest() == snapshot['content_inventory_sha256'], 'Snapshot digest differs')
        check(len({e['path'] for e in entries}) == len(entries), 'Duplicate protected path')
        check(all(e.get('sha256') for e in entries if e['kind'] == 'file'), 'Missing readable-file hash')
        maps.append({e['path']: e for e in entries})
    b, a = maps
    missing, added = sorted(set(b)-set(a)), sorted(set(a)-set(b))
    check(not missing and not added, {'protected_missing': missing, 'protected_added': added})
    changed = []
    for p in sorted(set(a) & set(b)):
        if any(b[p].get(k) != a[p].get(k) for k in ('kind', 'sha256', 'link_text_sha256')):
            changed.append(p)
    check(not changed, {'protected_content_changed': changed})
    check(before['content_inventory_sha256'] == after['content_inventory_sha256'], 'Protected content inventories differ')
    matched = sum(e['kind'] == 'file' and e['sha256'] == a.get(p, {}).get('sha256') for p, e in b.items())
    label_roots = [str(Path(root).parent) for root in m['scope']['scratch_roots']]
    non_scratch_records = [p for p, e in b.items() if e['kind'] == 'file' and
                           any(p.startswith(root+'/') for root in label_roots)]
    prompt_paths = sorted(str(D/'prompts'/('p64-formal8-'+stem+'.'+ext))
                          for stem in ('submit-1107q', 'post-1107r') for ext in ('md', 'sh'))
    prompt_checks = []
    for p in prompt_paths:
        h = sha(p)
        check(p in a and p in b and h == a[p]['sha256'] == b[p]['sha256'], 'Generation08 prompt changed: ' + p)
        prompt_checks.append({'path': p, 'sha256': h})
    exception_before = {e['path']: e for e in before['accepted_exceptions']}
    exception_after = {e['path']: e for e in after['accepted_exceptions']}
    check(set(exception_before) == set(exception_after) == set(receipt['accepted_exception_paths']) and
          len(exception_before) == 6, 'Known exception list differs')
    for p in exception_before:
        check(exception_before[p]['dev'] == exception_after[p]['dev'] and
              exception_before[p]['ino'] == exception_after[p]['ino'], 'Accepted exception identity changed: ' + p)
    comp = receipt['completion_check']
    check(comp['status'] == 'PASS_LEAD_ACCEPTED_COMPLETION_EVIDENCE' and len(comp['labels']) == 13 and
          len(comp['launchers']) == len(comp['monitors']) == 3, 'Missing accepted completion evidence')
    cutoff = int(datetime.datetime(2026,11,7,8,5,tzinfo=datetime.timezone.utc).timestamp()) * 1000000000
    check(all(e['newest_file_mtime_ns'] <= cutoff and e['latest_file_ctime_ns'] <= cutoff
              for e in comp['labels']), 'Completion evidence exceeds cutoff')
    check(comp['finished_utc'] <= receipt['started_utc'] <= receipt['deleted'][0]['checked_utc'],
          'Completion gate was not immediately before deletion')
    check(totals(receipt['deleted']) == receipt['deleted_summary'] == m['planned_summary'], 'Deletion totals differ')
    check(totals(receipt['kept']) == receipt['kept_summary'] == m['kept_summary'], 'Kept totals differ')
    check(receipt['kept_count'] == 32 and receipt['kept_bytes'] == 268875 and
          receipt['kept_allocated_bytes'] == 344064, 'Unexpected kept counts/bytes')
    check(receipt['removed_directory_count'] == len(removed) and
          receipt['directory_bytes_at_removal'] == sum(e['bytes'] for e in receipt['removed_directories']) and
          receipt['directory_allocated_bytes_at_removal'] == sum(e['allocated_bytes'] for e in receipt['removed_directories']),
          'Directory allocation totals differ')
    check(receipt['deleted_file_count'] == 449 and receipt['deleted_bytes'] == 32483199 and
          receipt['deleted_allocated_bytes'] == 33460224, 'Unexpected deleted counts/bytes')

    protection = {'status': 'PASS_WITH_SIX_ACCEPTED_EXCEPTIONS' if not (missing or added or changed) else 'FAIL',
                  'before_path': str(R/'protected_hash_before_final.json'),
                  'after_path': str(R/'protected_hash_after_final.json'),
                  'before_inventory_sha256': before['content_inventory_sha256'],
                  'after_inventory_sha256': after['content_inventory_sha256'],
                  'hash_matched_regular_files': matched, 'non_scratch_label_regular_files_matched': len(non_scratch_records),
                  'symlinks_matched': sum(e['kind'] == 'symlink' for e in b.values()),
                  'missing': missing, 'added': added, 'changed': changed,
                  'accepted_exceptions_before': list(exception_before.values()),
                  'accepted_exceptions_after': list(exception_after.values()),
                  'generation08_prompt_checks': prompt_checks,
                  'limitation': 'Five unreadable paths/unknown descendants and the active log excluded by express decision; '
                                'no host namespace scan. Directory metadata may change when scratch roots/prompts are removed.'}
    out = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
           'interpreter': sys.executable, 'python_version': sys.version,
           'deleted_file_count': len(deleted), 'kept_file_count': len(kept),
           'removed_directory_count': len(removed), 'remaining_scratch_directory_count': len(actual_dirs),
           'archive_entry_count': len(archive_checks), 'protected_check': protection,
           'immutable_original_records_checked': immutable_checked,
           'journal_event_counts': {k: len(v) for k, v in event_groups.items()},
           'deleted_summary': totals(receipt['deleted']), 'kept_summary': totals(receipt['kept'])}
    write('verification_final.json', out)
    receipt.update(protected_check=protection, archive_check={'entries': len(archive_checks), 'status': 'PASS' if not errors else 'FAIL'},
                   final_verification_path=str(R/'verification_final.json'),
                   verified_utc=out['observed_utc'], outcome='implemented' if not errors else 'partial',
                   status='PASS_EXECUTED_AND_VERIFIED' if not errors else 'PARTIAL_VERIFICATION_ERROR')
    write('deletion_receipt_final.json', receipt)
    print(json.dumps({k: out[k] for k in ('status','errors','deleted_file_count','kept_file_count',
                                         'removed_directory_count','remaining_scratch_directory_count','archive_entry_count')}, indent=2))
    print('Protected hash matches:', matched, 'non-scratch scoped label records:', len(non_scratch_records))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
