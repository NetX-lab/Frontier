"""Read-only final PROFILE04 preparation receipt; never seal or submit."""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run')
RECORDS = REPO / 'task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04'
EVIDENCE = RECORDS / 'cpu_evidence'
PACKAGE = REPO / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_04'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


config = json.loads((PACKAGE / 'config.json').read_text())
lock = json.loads((PACKAGE / 'source_lock.json').read_text())
files = {str(p.relative_to(PACKAGE)): dict(sha256=sha(p), bytes=p.stat().st_size,
    lines=len(p.read_bytes().splitlines())) for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
integrity = json.loads((EVIDENCE / 'integrity.receipt.json').read_text())
assert len(files) == 45 and sum(row['bytes'] for row in files.values()) == 6983749
assert {name: row['sha256'] for name, row in files.items()} == integrity['package_hashes']
prospective = json.loads((EVIDENCE / 'prospective_manifest.json').read_text())
assert prospective['files'] == integrity['package_hashes'] and not prospective['submitted']
assert sha(EVIDENCE / 'prospective_manifest.json') == integrity['prospective_manifest_sha256']
syntax = []
for name in files:
    if name.endswith('.py'):
        compile((PACKAGE / name).read_bytes(), str(PACKAGE / name), 'exec')
        syntax.append(name)
assert not (PACKAGE / 'manifest.json').exists() and not (PACKAGE / 'case/runs').exists()
assert not list(PACKAGE.rglob('__pycache__'))
prior = json.loads((RECORDS / 'prior_inventory.json').read_text())
original = Path(prior['package'])
assert {str(p.relative_to(original)): dict(sha256=sha(p), bytes=p.stat().st_size)
    for p in sorted(original.rglob('*')) if p.is_file()} == prior['files']
for path, expected in prior['plans'].items():
    assert sha(REPO / path) == expected
old_manifest = json.loads((original / 'manifest.json').read_text())
assert len(old_manifest['files']) == 45
for name, expected in old_manifest['files'].items():
    assert sha(original / name) == expected
same = [name for name, row in files.items() if sha(original / name) == row['sha256']]
changed = [name for name in files if name not in same]
assert set(changed) == {'case/inputs/engine.json', 'case/inputs/native_config.json',
    'case/manifest.yaml', 'config.json', 'engine.json', 'evidence_readiness.json',
    'overlay.json', 'profile_plan.json', 'source_lock.json'}
matrix = json.loads((RECORDS / 'matrix_binding.json').read_text())
assert sha(Path(matrix['comparison_path'])) == matrix['comparison_sha256']
for name, expected in matrix['unchanged_domain_trees'].items():
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', f'HEAD:{name}'],
        text=True).strip() == expected
source = lock['harness_commit']
assert source == '8cbf7ab9dc0690b537bc95f07e30ec545e40779c'
paths = ['task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py',
    'task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/profile_plan.json',
    'tests/unit/test_native_pd_worker.py']
commands, hashes = {}, {}
for name, flag in [('source_changes.diff', None), ('source_changes.numstat', '--numstat')]:
    command = ['git', '-C', str(REPO), 'diff'] + ([flag] if flag else []) + ['53398d06', source, '--', *paths]
    assert subprocess.check_output(command) == (RECORDS / name).read_bytes()
    commands[name], hashes[name] = command, sha(RECORDS / name)
write(RECORDS / 'source_budget_commit.receipt.json', dict(task_id='W128-PROFILE04-PACKAGE',
    base='53398d06', source_commit=source, commands=commands, sha256=hashes,
    source_and_plan_gross=52, test_gross=21, total_gross=73,
    status='PASS_ACTUAL_COMMIT_DIFF_MATCHES_PRECOMMIT_RECEIPT'))
witness = json.loads((EVIDENCE / 'root_actual_image_cpu.receipt.json').read_text())['witness']
assert witness['cuda_initialized'] is False
assert witness['native_wrapper_sha256'] == config['expected_source_hashes']['vllm/profiler/wrapper.py']
assert [row['actual_activities'] for row in witness['results']] == [['ProfilerActivity.CPU', 'ProfilerActivity.CUDA']] * 2
assert [(row['export_calls'], row['payload_annotations'], row['statistics_calls'], row['lazy_events_calls'])
    for row in witness['results']] == [(1, 8, 1, 1), (1, 8, 0, 0)]
virtual = json.loads((EVIDENCE / 'native_virtual_duration_owner.receipt.json').read_text())
assert virtual['timeout_seconds'] == 300
assert [(row['case'], row['verdict']) for row in virtual['results']] == [
    ('cuda_default_true', 'RED'), ('cuda_false', 'GREEN'), ('export_too_slow_false', 'RED'),
    ('cpu_only_false', 'RED'), ('cuda_false_fifo', 'GREEN')]
write(RECORDS / 'hash_inventory.json', dict(package=str(PACKAGE), files=files,
    generated_files=len(files), generated_bytes=sum(row['bytes'] for row in files.values()),
    generated_lines=sum(row['lines'] for row in files.values()),
    runtime_modules={name: lock['committed_files'][target] for name, target in config['harness_files'].items()},
    native_source_hashes=config['expected_source_hashes'],
    binary_sha256=json.loads((PACKAGE / 'overlay.json').read_text())['binary_sha256'],
    unchanged_from_profile03=same, changed_from_profile03=changed))
receipt = dict(task_id='W128-PROFILE04-PACKAGE', status='PASS_PREPARE_ONLY_PACKAGE',
    source_commit=source, source_and_plan_gross=52, test_gross=21, total_gross=73,
    generation_count=1, generated_files=45, generated_bytes=6983749,
    generated_lines=sum(row['lines'] for row in files.values()),
    prospective_manifest_path=str(EVIDENCE / 'prospective_manifest.json'),
    prospective_manifest_sha256=sha(EVIDENCE / 'prospective_manifest.json'),
    runtime_modules_unchanged=13, native_source_pins=87, binary_pins_unchanged=20,
    syntax_checks=len(syntax), unchanged_package_files_from03=len(same), changed_package_files_from03=changed,
    original03_files_unchanged=len(prior['files']), original03_sealed_inputs_unchanged=45,
    worker_tests=41, manifest_tests=14, tests_passed=55, tests_failed=0,
    matrix_cases_bound=77, matrix_new_capture=False, root_cpu_controls='BOUND_UNCHANGED_NOT_RERUN',
    sealed=False, submitted=False, runtime_qualified=False, source_writer='STOP_AFTER_RECORD_ARCHIVE_COMMIT',
    limits='Root owns dual review, actual-image/SDK controls, final native-duration acceptance, seal and RJobs. '
        'Synchronous export can still exceed300s; no automatic retry. Captured ownership and serving/operator cost remain unverified.')
write(RECORDS / 'final_verification.json', receipt)
print(json.dumps(receipt))
