"""Lead acceptance of cleanup request 86673e2f.

Checks that every difference reported in verification.json comes from lead work that ran
concurrently with the cleanup, and that no deletion entry touches the P2-64 mainline inputs.
Writes lead_acceptance.json beside this script; it reads everything else.
"""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

C = Path(__file__).resolve().parent
T = C.parent
W = T.parents[1]
H = T / 'h200_mainline'
BUILDER = H / 'build_package.py'
TERMINAL06 = H / 'p2_64_formal_retry06' / 'terminal_20261107'
LABEL06 = Path('/data/ycfeng/tmp/ds4_runs/p64-formal6-submit-1107j')
MAINLINE_MARKERS = ('formal_p2_20261102', 'p2_64_formal', '/downloads/', 'launch_p2_64')
REUSE_RUN = Path('/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64')
COMMIT = '3d69016f'


def utc(text):
    return datetime.datetime.fromisoformat(text).timestamp()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_sha256(commit, path):
    blob = subprocess.run(['git', '-C', str(W), 'show', f'{commit}:{path.relative_to(W)}'],
                          check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def files_created_between(lo, hi):
    """Regular files outside the cleanup record whose ctime lies in [lo, hi]."""
    found = []
    for root, dirs, files in os.walk(W):
        if root == str(W):
            dirs[:] = [d for d in dirs if d != '.git']
        if root.startswith(str(C)):
            dirs[:] = []
            continue
        for name in files:
            path = Path(root) / name
            try:
                st = path.lstat()
            except FileNotFoundError:
                continue
            if path.is_file() and not path.is_symlink() and lo <= st.st_ctime <= hi:
                found.append(path)
    return found


verification = json.loads((C / 'verification.json').read_text())
inventory = json.loads((C / 'inventory.json').read_text())
hash_before = json.loads((C / 'protected_hash_before.json').read_text())['observed_utc']
hash_after = json.loads((C / 'protected_hash_after.json').read_text())['observed_utc']
manifest = json.loads((C / 'deletion_manifest.json').read_text())

changed_files = verification['protected_files_changed']
tree_changes = {row['path']: row for row in verification['protected_tree_count_changes']}
identity_changes = {row['path'] for row in verification['unlisted_identity_changes']}

created = files_created_between(utc(hash_before), utc(hash_after))
new_files = [p for p in created if p != BUILDER]
worktree_row = tree_changes[str(W)]
retry06_row = tree_changes[str(H / 'p2_64_formal_retry06')]
terminal06_files = sum(len(files) for _, _, files in os.walk(TERMINAL06))
label06_turn_mtime = (LABEL06 / 'turn').stat().st_mtime
deleted_paths = [entry['path'] for entry in manifest['entries']]

checks = {
    'receipt_pass_and_no_missing_paths': verification['receipt_status'] == 'PASS'
        and verification['deleted_entries'] == verification['manifest_entries']
        and not verification['unlisted_missing_paths']
        and not verification['snapshot_errors'] and not verification['errors'],
    'only_builder_hash_changed': [row['path'] for row in changed_files] == [str(BUILDER)],
    'builder_after_equals_commit': changed_files[0]['after']['sha256'] == sha256(BUILDER)
        == committed_sha256(COMMIT, BUILDER),
    'only_two_tree_counts_changed': set(tree_changes) == {str(W), str(H / 'p2_64_formal_retry06')},
    'worktree_growth_equals_lead_new_files': worktree_row['after']['regular_file_count']
        - worktree_row['before']['regular_file_count'] == len(new_files),
    'lead_new_files_are_retry06_terminal_or_retry07': all(
        str(p).startswith((str(TERMINAL06), str(H / 'p2_64_formal_retry07'))) for p in new_files),
    'retry06_growth_equals_terminal_archive': retry06_row['after']['regular_file_count']
        - retry06_row['before']['regular_file_count'] == terminal06_files,
    'identity_changes_are_label06_turn2': identity_changes
        == {str(LABEL06 / 'execution.json'), str(LABEL06 / 'exit')}
        and (LABEL06 / 'turn').read_text().strip() == '2'
        and label06_turn_mtime > utc(inventory['finished_utc']),
    'no_deletion_touches_mainline': not [p for p in deleted_paths
                                         if any(m in p for m in MAINLINE_MARKERS)],
    'post_run_reuse_run_present': all((REUSE_RUN / name).exists()
                                      for name in ('collectives.json', 'forwards.jsonl.gz', 'receipt.json')),
    'seals_unchanged': verification['seals']['generation05_manifest_sha256']
        == verification['seals']['generation05_manifest_expected']
        and verification['seals']['promotion_receipt_sha256']
        == verification['seals']['promotion_receipt_expected'],
}

result = {
    'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'request': '86673e2f-3b87-4663-a948-1a9ec9eb03a4',
    'status': 'ACCEPTED_DIFFERENCES_FROM_CONCURRENT_LEAD_WORK' if all(checks.values()) else 'FAIL',
    'checks': checks,
    'evidence': {
        'inventory_window': [inventory['started_utc'], inventory['finished_utc']],
        'protected_hash_window': [hash_before, hash_after],
        'builder_sha256': {'before': changed_files[0]['before']['sha256'],
                           'after': changed_files[0]['after']['sha256'],
                           'commit': COMMIT},
        'lead_new_files_in_window': len(new_files),
        'retry06_terminal_files': terminal06_files,
        'label06_turn2_utc': datetime.datetime.fromtimestamp(
            label06_turn_mtime, datetime.timezone.utc).isoformat(),
        'deleted_entries': len(deleted_paths),
        'uncertain_retained': len(manifest['uncertain_retained']),
    },
}
(C / 'lead_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
