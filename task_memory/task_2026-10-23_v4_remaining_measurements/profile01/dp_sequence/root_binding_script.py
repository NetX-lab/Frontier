"""Verify frozen diagnostic source and retain the baseline failure evidence."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import subprocess

ROOT = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
OWNER = Path('/data/ycfeng/Frontier/.worktrees/v4-dp-sequence-record')
REL = Path('task_memory/task_2026-10-23_v4_remaining_measurements/profile01/dp_sequence')
SOURCE, OUT = OWNER / REL, ROOT / REL
RAW = Path('/data/ycfeng/tmp/glm53_dsv4/dp_sequence_root_w4peji6l')
BASELINE = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-root-failure')
sha = lambda b: hashlib.sha256(b).hexdigest()
load = lambda p: json.loads(p.read_text())
git = lambda cwd, *args: subprocess.check_output(['git', *args], cwd=cwd)
s, manifest = load(SOURCE / 'source_binding.json'), load(SOURCE / 'records_manifest.json')
f, matrix = load(SOURCE / 'frozen_native_binding.json'), load(SOURCE / 'matrix_binding.json')
checks = []

def check(value, label):
    assert value, label
    checks.append(label)

for row in s['files']:
    raw = (OWNER / row['path']).read_bytes()
    check(sha(raw) == row['candidate_sha256'] and raw == git(OWNER, 'show', s['source_revision'] + ':' + row['path']), row['path'])
    check(sha(git(OWNER, 'show', s['base_revision'] + ':' + row['path'])) == row['base_sha256'], 'base:' + row['path'])
for name, digest in manifest['files'].items():
    check(sha((OWNER / name).read_bytes()) == digest, name)
for row in f['frozen']:
    check(sha((OWNER / row['path']).read_bytes()) == row['sha256'], row['path'])
native = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
for row in f['native']:
    raw = (native / row['path']).read_bytes()
    check(sha(raw) == row['sha256'] and raw == git(native, 'show', f['native_revision'] + ':' + row['path']), row['path'])
comparison = load(Path(matrix['comparison_path']))
check(sha(Path(matrix['comparison_path']).read_bytes()) == matrix['comparison_sha256'], 'comparison hash')
check(comparison['complete_comparison'] and comparison['cases_compared'] == 77 and len(comparison['identical_cases']) == 77, '77 complete cases')
for revision in (s['base_revision'], s['source_revision'], matrix['inherited_source_revision']):
    check(git(OWNER, 'diff', '--name-only', matrix['comparison_source'], revision, '--', *matrix['unchanged_domains']) == b'', 'matrix domain:' + revision)
path = 'tests/comparison/calibration/native_full_structure.py'
before, after = ast.parse(git(OWNER, 'show', s['base_revision'] + ':' + path)), ast.parse((OWNER / path).read_text())
def methods(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeServingInventory')
    return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
a, b = methods(before), methods(after)
check({k for k in a if a[k] != b[k]} == {'install_v2_dispatch'}, 'unchanged base count/emit methods')
hooks = lambda tree: sorted(ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == 'hooks')
check(hooks(before) == hooks(after), 'unchanged hook statements')
run = load(RAW / 'receipt.json')
check(run['status'] == 'FAIL' and run['source_revision'] == s['source_revision'] and run['source_binding_sha256'] == sha((SOURCE / 'source_binding.json').read_bytes()), 'original root receipt binding')
counts = [r['counts'] for r in run['results']]
check(sum(x['tests'] - x['failures'] - x['errors'] - x['skipped'] for x in counts) == 214 and sum(x['failures'] for x in counts) == 1 and sum(x['errors'] + x['skipped'] for x in counts) == 0, '214PASS/1FAIL/0SKIP')
diag = load(BASELINE / 'diagnosis.json')
for name, row in diag['receipts'].items():
    check(row['counts'] == dict(tests=1, failures=1, errors=0, skipped=0) and row['keyerror_rows'] == 32, name + ' original failure')
    check(sha(Path(row['log']).read_bytes()) == row['log_sha256'], name + ' log hash')
    for item in [*row['input_hashes'].values(), row['legacy']]:
        check(sha(Path(item['path']).read_bytes()) == item['sha256'], name + item['path'])
rows = {name: {str(p.relative_to(Path(root))): load(p) for p in Path(root).glob('nodes/*/serving/rank*.json')} for name, root in diag['artifact_roots'].items()}
check(len(rows['baseline']) == len(rows['candidate']) == 32, '32 rank rows')
for name, row in rows['baseline'].items():
    check(row['counters'] == rows['candidate'][name]['counters'] and row['hooks'] == rows['candidate'][name]['hooks'], 'rank equality:' + name)
for kind, revision in [('baseline', s['base_revision']), ('candidate', s['source_revision'])]:
    for name, digest in diag['source_sha256'][kind].items():
        check(sha(git(OWNER, 'show', revision + ':' + name)) == digest, kind + name)
copies = [(RAW / 'receipt.json', 'root_test_receipt.json'), (BASELINE / 'diagnosis.json', 'baseline_diagnosis.json'), (BASELINE / 'diagnosis.md', 'baseline_diagnosis.txt'), (Path(__file__), 'root_binding_script.py')]
copies += [(p, 'root_logs/' + p.name) for pattern in ('*.xml', '*.log') for p in RAW.glob(pattern)]
copies += [(BASELINE / name, 'baseline_control/' + name) for name in ('baseline.json', 'candidate.json', 'baseline.xml', 'candidate.xml', 'baseline.log', 'candidate.log')]
files = []
for old, name in copies:
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = old.read_bytes()
    with dest.open('xb') as stream:
        stream.write(raw)
    files.append(dict(path=str(dest.relative_to(ROOT)), source=str(old), bytes=len(raw), sha256=sha(raw)))
receipt = dict(status='PASS_BOUND_SOURCE_AND_BASELINE_CLASSIFICATION', observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_revision=s['source_revision'], owner_revision=git(OWNER, 'rev-parse', 'HEAD').decode().strip(), checks=checks, root_suite_counts=dict(passed=214, failed=1, errors=0, skipped=0), baseline_failure='Existing V2 groundtruth fixture omits dp_utils and uses old hook checker', rank_counter_and_hook_equality=32, matrix_cases=77, new_matrix_capture=False, native_pairing='UNVERIFIED', prior_auxiliary_error="Root checker initially addressed row['observer']['hooks']; actual schema has top-level row['hooks']. Corrected only this read; no source/tests/fixtures changed.", files=files)
with (OUT / 'root_verification.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status=receipt['status'], checks=len(checks), retained_files=len(files), output=str(OUT))))
