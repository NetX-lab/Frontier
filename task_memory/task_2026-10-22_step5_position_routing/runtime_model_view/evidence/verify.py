"""Verify and retain compact evidence; leave raw traces in scratch storage."""
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
DEST = REPO / 'task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/evidence'
frozen = json.loads((ROOT / 'freeze.json').read_text())
source = Path(frozen['sources']['candidate']['root'])
base = json.loads((ROOT / 'baseline/wave_result.json').read_text())
candidate = json.loads((ROOT / 'candidate/wave_result.json').read_text())
cases = json.loads((ROOT / 'contract_cases.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=source):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


assert git('rev-parse', 'HEAD') == frozen['sources']['candidate']['head']
assert git('status', '--porcelain') == ''
assert not git('diff', '--name-only', 'c4e06a6159ee3b1e63c50abaa346066ae8b6da31',
               '46fe83ac', '--', '.', ':(exclude)task_memory')
assert sha(Path(frozen['protected_body'])) == frozen['protected_body_sha256']
bindings = dict(frozen['sources']['candidate']['hashes'])
registry_path = 'frontier/execution_time_predictor/prediction_model_registry.py'
committed_registry = subprocess.check_output(['git', 'show', f'HEAD:{registry_path}'], cwd=source)
bindings[registry_path] = hashlib.sha256(committed_registry).hexdigest()
for relative, expected in bindings.items():
    assert sha(source / relative) == expected == sha(REPO / relative), relative
fixture = Path(frozen['fixture'])
metadata = {str(path.relative_to(fixture)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in fixture.rglob('*') if path.is_file()}
assert metadata == frozen['fixture_metadata']
comparisons = []


def same(left, right, label):
    assert left.read_bytes() == right.read_bytes(), label
    comparisons.append({'path': label, 'bytes': left.stat().st_size, 'sha256': sha(left)})


same(ROOT / 'baseline/pricing_payload.json', ROOT / 'candidate/pricing_payload.json', 'pricing_payload.json')
left, right = ROOT / 'baseline/accepted-output/baseline', ROOT / 'candidate/accepted-output/candidate'
same(left / 'ledger.json', right / 'ledger.json', 'ledger.json')
diagnostic_headers = {}
for path in sorted(left.rglob('op_traces.jsonl')):
    relative = str(path.relative_to(left))
    if relative in ('late-json-failure/op_traces.jsonl', 'profile/op_traces.jsonl'):
        header_left, prefix_left = path.read_bytes().split(b'\n', 1)
        header_right, prefix_right = (right / relative).read_bytes().split(b'\n', 1)
        expected_bytes = 3902 if relative.startswith('late-json-failure/') else 0
        assert prefix_left == prefix_right and len(prefix_left) == expected_bytes
        headers = [json.loads(header_left), json.loads(header_right)]
        times = [header['meta'].pop('timestamp') for header in headers]
        assert headers[0] == headers[1]
        diagnostic_headers[relative] = {'metadata_header_timestamps': times,
            'other_header_values_equal': True, 'prefix_bytes': len(prefix_left),
            'prefix_sha256': hashlib.sha256(prefix_left).hexdigest()}
    else:
        same(path, right / path.relative_to(left), relative)
for value in (base, candidate):
    counts = value['operation_counts']
    assert [counts[key] for key in ('_get_on_demand_prediction', '_predict_step5_ffn_operator',
        'predict_collective_plan', '_step5_position_queries', 'cache_key')] == [288, 144, 64, 16, 192]
    assert value['output']['events'] == 208 and value['output']['wrong_plan_key_negatives'] == 64
    assert value['output']['late_json_error_restores_buffer_and_prefix_bytes'] == 3902
    assert value['main_run_successful_file_reads'] == 1
    assert value['source_unchanged'] and value['fixture_metadata_unchanged']
assert base['operation_counts']['_resolve_cluster_model_contract'] == 384
assert candidate['operation_counts']['_resolve_cluster_model_contract'] == 0
assert len(cases['cases']) == 11
assert sum(case.get('equivalent') is False for case in cases['cases']) == 1

diffs = []
for relative, class_name, method, candidate_file in (
    (registry_path, 'PredictionModelRegistry', '_models_view_for_family', 'candidate_view.py'),
    ('frontier/execution_time_predictor/moe_routing_workload.py', 'MoeRoutingWorkload', '_step5_position_queries', 'candidate_point.py'),
):
    content = (source / relative).read_text()
    cls = next(node for node in ast.parse(content).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    old = ROOT / ('baseline_' + candidate_file)
    old.write_text(textwrap.dedent(''.join(content.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])))
    new = ROOT / 'candidate' / candidate_file
    diff = subprocess.run(['git', 'diff', '--no-index', '--numstat', str(old), str(new)], capture_output=True, text=True)
    assert diff.returncode == 1, diff.stderr
    added, deleted, _ = diff.stdout.split(maxsplit=2)
    diffs.append({'path': relative, 'added': int(added), 'deleted': int(deleted),
                  'gross': int(added) + int(deleted), 'net': int(added) - int(deleted)})

receipt = {'task_id': 'D29-RUNTIME-MODEL-VIEW', 'status': 'QUALIFIED_CONTROL_PENDING_ERROR_POLICY_DECISION',
    'source_head': git('rev-parse', 'HEAD'), 'accepted_union': '46fe83ac', 'source_bindings': bindings,
    'source_binding_count': len(bindings), 'fixture_file_count': len(metadata),
    'protected_body_sha256': frozen['protected_body_sha256'], 'byte_comparisons': comparisons,
    'diagnostic_metadata_headers': diagnostic_headers,
    'environment': {'python': sys.version, 'executable': sys.executable,
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')}},
    'warm_seconds': {'baseline': [sample['pricing_s'] for sample in base['samples'] if sample['phase'] == 'warm'],
        'candidate': [sample['pricing_s'] for sample in candidate['samples'] if sample['phase'] == 'warm'],
        'baseline_median': base['warm_median_s'], 'candidate_median': candidate['warm_median_s'],
        'change_percent': 100 * (candidate['warm_median_s'] / base['warm_median_s'] - 1)},
    'operation_counts': {'baseline': base['operation_counts'], 'candidate': candidate['operation_counts']},
    'contract_case_groups': len(cases['cases']), 'non_equivalent_case_groups': 1,
    'scratch_method_diff': diffs,
    'limits': 'Two in-memory method replacements; old task/limit strings inside the unchanged protected body are historical, not authority for this control. No implementation allocation, full Simulator/CLI, native qualification or final interface acceptance.'}
(ROOT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
DEST.mkdir(parents=True, exist_ok=True)
paths = ['prepare.py', 'run.py', 'fixture_wiring.py', 'narrow_view.py', 'contract_cases.py', 'verify.py',
         'freeze.json', 'prepare.log', 'baseline.log', 'candidate.log', 'contract_cases.log',
         'contract_cases.json', 'verification.json', 'baseline/wave_result.json', 'candidate/wave_result.json',
         'candidate/candidate_view.py', 'candidate/candidate_point.py',
         'baseline_candidate_view.py', 'baseline_candidate_point.py', 'verify.log', 'verify02.log', 'verify03.log']
manifest = {}
for relative in paths:
    path, target = ROOT / relative, DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    shutil.copyfile(path, target)
    assert sha(path) == sha(target)
    manifest[relative] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
(DEST / 'manifest.json').write_text(json.dumps({'scratch_root': str(ROOT), 'files': manifest}, indent=2) + '\n')
print(json.dumps({'verification_sha256': sha(ROOT / 'verification.json'), 'artifact_count': len(manifest),
    'byte_comparisons': len(comparisons), 'scratch_method_diff': diffs,
    'status': receipt['status']}, indent=2))
