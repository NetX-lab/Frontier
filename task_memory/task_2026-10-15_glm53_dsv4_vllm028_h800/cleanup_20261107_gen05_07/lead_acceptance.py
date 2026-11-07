"""Lead acceptance of the generation05-07 scratch cleanup (codex requests 890bf5f2 and 15216d7e)."""
from pathlib import Path
import hashlib
import json
import os
import sys

TASK = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800')
REC = TASK / 'cleanup_20261107_gen05_07'
PROMPTS = Path('/data/ycfeng/tmp/ds4_runs/prompts')
GEN08_PROMPTS = {  # sha256 recorded by the lead before the first request
    'p64-formal8-post-1107r.md': 'ba59b8a61527586f6744', 'p64-formal8-post-1107r.sh': 'a731c8a82712c045923d',
    'p64-formal8-submit-1107q.md': 'da151135c698b0ee5b45', 'p64-formal8-submit-1107q.sh': 'b28471d276592018ef7d'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


receipt = json.loads((REC / 'deletion_receipt_final.json').read_text())
verification = json.loads((REC / 'verification_final.json').read_text())
checks = {}
checks['codex_receipt'] = receipt['outcome'] == 'implemented' and receipt['errors'] == [] and verification['status'] == 'PASS'
checks['counts'] = (receipt['deleted_file_count'], len(receipt['deleted']), receipt['kept_count']) == (449, 449, 32)
checks['deleted_absent'] = all(not os.path.lexists(e['path']) for e in receipt['deleted'])
checks['deleted_in_scope'] = all(
    (e['path'].startswith('/data/ycfeng/tmp/ds4_runs/p64-formal') and '/scratch/' in e['path']
     and e['path'].split('/')[5][:12] in ('p64-formal5-', 'p64-formal6-', 'p64-formal7-'))
    or e['path'] in (str(PROMPTS / 'p64-formal7-post-1107n.sh'), str(PROMPTS / 'p64-formal7-post-1107n.md'))
    for e in receipt['deleted'])
checks['kept_unchanged'] = all(os.path.exists(e['path']) and sha(e['path']) == e['sha256'] for e in receipt['kept'])
archive_entries = 0
archives_intact = True
for generation in ('05', '06', '07'):
    base = TASK / f'h200_mainline/p2_64_formal_retry{generation}/terminal_20261107'
    for entry in json.loads((base / 'archive_manifest.json').read_text())['files']:
        archive_entries += 1
        archives_intact &= sha(base / entry['destination']) == entry['sha256']
checks['archives_intact'] = archives_intact and archive_entries == 130
checks['gen08_prompts_unchanged'] = all(sha(PROMPTS / name).startswith(prefix) for name, prefix in GEN08_PROMPTS.items())
checks['launcher_exits_present'] = all(
    (TASK / f'h200_mainline/p2_64_formal_retry{g}/package_01/launcher_exit.json').exists() for g in ('05', '06', '07'))
snapshots = {name: dict(bytes=(REC / name).stat().st_size, sha256=sha(REC / name)) for name in (
    'protected_hash_before.json', 'protected_hash_after.json',
    'protected_hash_before_final.json', 'protected_hash_after_final.json')}
status = 'ACCEPTED' if all(checks.values()) else 'REJECTED'
(REC / 'lead_acceptance.json').write_text(json.dumps(dict(
    status=status, python=sys.version.split()[0], command='python3 -B cleanup_20261107_gen05_07/lead_acceptance.py',
    codex_requests=['890bf5f2-a7eb-473d-ac18-51c83fd2de58 (blocked, classification only)',
                    '15216d7e-7bc2-49ed-bad0-f1aa572dfb9a (implemented)'],
    deleted_files=receipt['deleted_file_count'], deleted_bytes=receipt['deleted_bytes'],
    deleted_allocated_bytes=receipt['deleted_allocated_bytes'], kept_files=receipt['kept_count'],
    archive_entries=archive_entries, checks=checks,
    lead_decisions=['Pi exit files, label quiescence, launcher exits and monitor finals replace the host process scan.',
                    'Five unreadable paths under /data/ycfeng/tmp/glm53_dsv4 and the growing old G0-D launcher log are '
                    'unrelated to the scope and accepted as known exceptions.'],
    snapshot_files_not_committed=snapshots,
), indent=2) + '\n')
print(status, json.dumps(checks))
sys.exit(0 if status == 'ACCEPTED' else 1)
