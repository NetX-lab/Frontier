#!/usr/bin/env python3
"""Audit the proposed cleanup and all retained scoped files; never delete."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat

R = Path(__file__).resolve().parent
D = Path('/data/ycfeng/tmp/ds4_runs')
T = R.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((R / 'deletion_manifest.json').read_text())
    receipt = json.loads((R / 'deletion_receipt.json').read_text())
    errors = []
    counts = {'archived_duplicate': 0, 'node_compile_cache': 0, 'superseded_prompt': 0,
              'uncovered_scratch': 0, 'unclassified': 0}
    labels = sorted([p for gen in '567' for p in D.glob('p64-formal' + gen + '-*')])
    rows = m['entries'] + m['kept']
    listed = [row['path'] for row in rows]
    if len(set(listed)) != len(listed):
        errors.append('Duplicate scoped paths in manifest')
    actual = set()
    actual_directories = set()
    for label in labels:
        if not (label / 'exit').is_file():
            errors.append('Scoped label missing exit: ' + str(label))
        for parent, dirs, files in os.walk(label / 'scratch', followlinks=False):
            actual_directories.add(str(parent))
            actual.update(str(Path(parent) / f) for f in files)
            actual.update(str(Path(parent) / d) for d in dirs if (Path(parent) / d).is_symlink())
    prompts = {str(D / 'prompts' / ('p64-formal7-post-1107n.' + ext)) for ext in ('sh', 'md')}
    actual |= prompts
    if actual_directories != set(m['scratch_directories_before']):
        errors.append('Scoped scratch directory inventory changed')
    if actual != set(listed):
        errors.append({'scope_missing_from_manifest': sorted(actual - set(listed)),
                       'manifest_outside_scope': sorted(set(listed) - actual)})
    for row in rows:
        p = Path(row['path'])
        category = row['category']
        counts[category] += 1
        if not p.is_file() or p.is_symlink() or str(p.resolve()) != str(p):
            errors.append('Entry no longer a canonical regular file: ' + str(p))
            continue
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            errors.append('Scoped file content changed: ' + str(p))
        if category == 'archived_duplicate':
            cover = row['archive_cover']
            manifest = Path(cover['manifest'])
            entry = json.loads(manifest.read_text())['files'][cover['entry_index']]
            archived = manifest.parent / entry['destination']
            if (entry['sha256'] != row['sha256'] or sha(archived) != row['sha256'] or
                    entry['bytes'] != row['bytes'] or sha(manifest) != cover['manifest_sha256']):
                errors.append('Triple SHA256 coverage failed: ' + str(p))
        elif category == 'node_compile_cache':
            label = next((l for l in labels if p.is_relative_to(l / 'scratch')), None)
            if label is None or 'node-compile-cache' not in p.relative_to(label / 'scratch').parts[:-1]:
                errors.append('Cache outside authorized scope: ' + str(p))
        elif category == 'superseded_prompt':
            if str(p) not in prompts:
                errors.append('Unauthorized prompt: ' + str(p))
            if sha(row['replacement_path']) != row['replacement_sha256']:
                errors.append('Replacement changed: ' + row['replacement_path'])
        else:
            if row['intended_action'] != 'keep':
                errors.append('Uncovered entry selected for deletion: ' + str(p))
            if category == 'uncovered_scratch' and any(
                    a['sha256'] == row['sha256'] and a['bytes'] == row['bytes']
                    for a in m['archive_entries'] if 'error' not in a):
                errors.append('Uncovered file actually matches an archive: ' + str(p))
    archive_failures = []
    for archive in m['archive_entries']:
        if sha(archive['archive_path']) != archive['sha256']:
            archive_failures.append(archive['archive_path'])
    if archive_failures:
        errors.append({'archive_failures': archive_failures})
    if sha(R / 'assignment.md') != m['assignment_sha256']:
        errors.append('Pre-existing assignment changed')
    if receipt['deleted_file_count'] or receipt['deleted'] or receipt['removed_directories']:
        errors.append('Unexpected deletion recorded')
    if len(receipt['kept']) != len(rows) or any(not r['kept_file_unchanged'] for r in receipt['kept']):
        errors.append('Receipt does not verify all retained files')
    if sha(R / 'deletion_manifest.json') != receipt['manifest_sha256']:
        errors.append('Manifest hash differs from receipt')
    # Validate full content inventory comparisons independently of receipt summaries.
    before = json.loads((R / 'protected_hash_before.json').read_text())
    after = json.loads((R / 'protected_hash_after.json').read_text())
    key = lambda row: (row['kind'], row.get('sha256'), row.get('link_text_sha256'), row.get('error'))
    b = {r['path']: key(r) for r in before['entries']}
    a = {r['path']: key(r) for r in after['entries']}
    protected_equal = b == a
    verified_before = {row['path']: row['sha256'] for row in before['entries'] if 'sha256' in row}
    verified_after = {row['path']: row['sha256'] for row in after['entries'] if 'sha256' in row}
    proven_hash_changes = sorted(p for p, h in verified_before.items()
                                if p in verified_after and h != verified_after[p])
    protected_missing_or_added = sorted(set(b) ^ set(a))
    if proven_hash_changes or protected_missing_or_added:
        errors.append({'protected_proven_hash_changes': proven_hash_changes,
                       'protected_missing_or_added': protected_missing_or_added})
    out = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'status': 'PASS_CLASSIFICATION_AND_RETENTION' if not errors else 'FAIL',
           'acceptance_outcome': 'BLOCKED', 'errors': errors, 'counts': counts,
           'scoped_labels': len(labels), 'scoped_files': len(rows),
           'covered_archive_entries': len(m['archive_entries']),
           'protected_content_inventory_equal': protected_equal,
           'protected_proven_hash_changes': proven_hash_changes,
           'protected_missing_or_added': protected_missing_or_added,
           'protected_unverified_file_paths': sorted(
               {row['path'] for row in before['entries'] + after['entries'] if row['kind'] == 'file'}
               - (set(verified_before) & set(verified_after))),
           'protected_sha256_match_count': sum(verified_after.get(p) == h for p, h in verified_before.items()),
           'protected_complete_sha256_check': 'NOT_COMPLETE: unreadable directory descendants/files and unstable input',
           'protected_before_error_count': len(before['errors']),
           'protected_after_error_count': len(after['errors']),
           'host_process_check': 'NOT_RUN: isolated PID namespace',
           'deletion_execution': 'NOT_RUN: prerequisites unavailable'}
    (R / 'verification.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
