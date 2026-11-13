"""Verify the newly archived receipts and corrected records without runtime calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

repo = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
task = repo/'task_memory/task_2026-10-22_step5_position_routing'
archive = task/'resumed_execution/evidence/worker_readiness_2026-10-25'
def require(condition, message):
    if not condition:
        raise AssertionError(message)
def read_json(path):
    return json.loads(path.read_bytes())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
parent = read_json(archive/'parent_receipt.json')
require(head == parent['source']['records_head'], 'Records HEAD moved before the archive commit')
require(sys.version.split()[0] == '3.10.6' and sys.dont_write_bytecode, 'Wrong verification interpreter')
require(not subprocess.check_output(['git', 'diff', '--name-only', parent['source']['source_commit'],
    '--', '.', ':(exclude)task_memory'], cwd=repo, text=True).strip(), 'Production source changed')
for record in parent['artifacts']:
    path = repo/record['archive']
    require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], 'Archived business receipt differs')
    require(digest(Path(record['source'])) == record['sha256'], 'Original selected receipt differs')
capture = read_json(archive/'parent_capture_manifest.json')
for record in capture['artifacts']:
    path = repo/record['archive']
    require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], 'Parent capture differs')
execution = read_json(archive/'parent_execution.json')
require(execution['exit_code'] == 0 and digest(archive/'parent_verification.py') == execution['script_sha256'],
        'Parent command or script identity differs')
prepared = task/'resumed_execution/evidence/execution_preparation_2026-10-25'
review = read_json(prepared/'preparation_review_followup.json')
for filename, expected in review['script_hashes'].items():
    require(digest(prepared/filename) == expected, 'A reviewed prepared script changed')
require(digest(prepared/'worker_plan.json') == parent['worker_plan_sha256'], 'Reviewed worker plan changed')
original_worktrees = json.loads(subprocess.check_output(['git', 'show', 'HEAD:'+str((task/'resumed_execution/worktrees.json').relative_to(repo))], cwd=repo))
current_worktrees = read_json(task/'resumed_execution/worktrees.json')
checkpoint = current_worktrees.pop('worker_readiness_checkpoint')
require(current_worktrees == original_worktrees and len(original_worktrees['rows']) == 89,
        'Historical worktree census or checkpoint changed')
require(checkpoint['selected_active_worktrees'] == [{**row, 'role': entry['role']}
    for row, entry in zip(parent['worktrees'], checkpoint['selected_active_worktrees'])], 'Current worktree checkpoint differs')
require(all(row['ancestor_of_primary'] and (row['name'] == 'step5-preview' or not row['status'])
    for row in checkpoint['selected_active_worktrees']), 'Current satellite state differs')
markdown = [task/name for name in ('requirements.md', 'plan.md', 'progress.md', 'summary.md', 'handoff_prompt.md')]
markdown += [task/'resumed_execution'/name for name in ('execution_plan_2026-10-25.md',
    'check_progress_2026-10-25_worker_readiness.md', 'test_report_2026-10-25_worker_readiness.md')]
for path in markdown:
    text = path.read_text()
    require('Modification History' in text and '2026-10-24 UTC / 2026-10-25 HKT' in text, 'Missing document maintenance date')
    require(parent['job_name'] in text, 'Missing single-job identity')
    require('2400' in text and ('6/6' in text or '6/6' in text.replace(' ', '')), 'Missing unchanged execution limits')
requirements = (task/'requirements.md').read_text()
require('## [Execution Proposal / Not Approved] D29-CURRENT-JOINT-TIMING-TWO-CALLS' in requirements,
        'Unapproved finite pricing proposal changed')
require('PASS 22｜BLOCKED/FAILED/SKIPPED 4｜TODO 0' in markdown[-2].read_text(), 'Checkpoint counts differ')
secrets = [(Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')/name).read_bytes().strip()
           for name in ('accesskey_id', 'accesskey_secret')]
files = markdown + [path for path in archive.rglob('*') if path.is_file()]
for path in files:
    contents = path.read_bytes()
    require(not any(secret and secret in contents for secret in secrets), 'Known personal key in selected record')
    require(not re.search(rb'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/\"\x27]+:[^\s/\"\x27]+@', contents), 'URL credentials in selected record')
result = {'status': 'PASS_WORKER_ARCHIVE_AND_CURRENT_RECORDS_ONLY',
    'collected_utc': datetime.now(timezone.utc).isoformat(), 'records_head_before_commit': head,
    'python': sys.version, 'executable': sys.executable, 'bytecode_disabled': sys.dont_write_bytecode,
    'business_archive_artifacts': len(parent['artifacts']), 'business_archive_bytes': parent['archive_bytes'],
    'parent_capture_artifacts': len(capture['artifacts']), 'parent_capture_bytes': capture['selected_bytes'],
    'reviewed_scripts_unchanged': len(review['script_hashes']), 'historical_worktree_rows_unchanged': 89,
    'current_worktrees_verified': len(parent['worktrees']), 'maintained_markdown_documents': len(markdown),
    'screened_record_files': len(files), 'authored_diff_check': 'PASS_PREVIOUS_DIRECT_GIT_CHECK',
    'production_source_changed': False, 'new_runtime_or_platform_calls': 0, 'new_pricing_calls': 0,
    'configured_memory_max_bytes': 32212254720, 'effective_cgroup_peak': 'not measured'}
output = archive/'records_verification.json'
require(not output.exists(), 'Records verification receipt already exists')
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
