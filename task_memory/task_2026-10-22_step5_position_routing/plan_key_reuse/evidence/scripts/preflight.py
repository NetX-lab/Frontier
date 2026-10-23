"""Freeze scoped source and accepted replay inputs without a training scan."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
(ROOT / 'cache').mkdir(exist_ok=True)
CANDIDATE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key')
BASELINE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot')
TASK = Path('task_memory/task_2026-10-22_step5_position_routing')
paths = json.loads((CANDIDATE / TASK / 'repairs/run_input_snapshot_scope_amendment.json').read_text())['repair_paths']
paths += ['frontier/cc_backend/collective_plan.py', 'frontier/metrics/op_trace_utils.py',
    'frontier/metrics/ep_wave_metrics.py', 'frontier/metrics/trace_store.py',
    'frontier/metrics/metrics_store.py', 'frontier/execution_time_predictor/moe_operator_times.py',
    'frontier/execution_time_predictor/sklearn_execution_time_predictor.py',
    'frontier/scheduler/utils/expert_parallel.py', 'frontier/scheduler/utils/ep_wave.py',
    'tests/unit/test_step5_measured_ffn_replay.py',
    'data/config/models/stepfun-ai__Step-5-Preview.json', 'tests/unit/test_step5_moe_pricing.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = {}
for label, source in (('baseline', BASELINE), ('candidate', CANDIDATE)):
    sources[label] = {'root': str(source),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=source, text=True).strip(),
        'hashes': {path: sha(source / path) for path in paths if (source / path).exists()},
        'status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True)}
assert sources['candidate']['head'] == '087db1bfcb3dbc2377d4032f650b29c48fb104fd'
assert sources['baseline']['head'] == '6c7121d819e679054285bb3be5cb117a7e77bc4e'
for path in paths:
    if path not in ('frontier/cc_backend/collective_plan.py', 'tests/unit/test_step5_moe_pricing.py'):
        assert sources['candidate']['hashes'][path] == sources['baseline']['hashes'][path],path
for label,source in (('baseline', BASELINE), ('candidate', CANDIDATE)):
    repo = source/'frontier/cc_backend/backends/collective-sim'
    pin = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert pin == '3dac53c544e546574cb6e8078cd14053d0916113'
    sources[label]['submodule_pin'] = pin
setup = Path('/data/ycfeng/tmp/d29_immutable_cost_fields_repair/current_setup_baseline.py')
body = Path('/data/ycfeng/tmp/d29_immutable_cost_fields_repair/probe_body.py')
assert sha(setup) == 'c93b9d286342fc9b854ccb6692630542bf306a856ab4254cd84feddfaf7eb0d7'
assert sha(body) == '1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555'
fixture = Path('/data/ycfeng/tmp/d29-parent-position-fixture03')
collectives = list((fixture / 'fixture_collectives').rglob('collective_local.json'))
assert len(collectives) == 1
for label in ('baseline', 'candidate'):
    target = ROOT / label / 'collective-fixture' / collectives[0].parent.name / 'collective_local.json'
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(collectives[0].read_bytes())
fixture_metadata = {str(path.relative_to(fixture)): [path.stat().st_size, path.stat().st_mtime_ns]
                    for path in fixture.rglob('*') if path.is_file()}
xml = Path('/data/ycfeng/tmp/d29_snapshot_green01.xml')
suite = ET.parse(xml).getroot().find('testsuite')
assert suite.attrib['tests'] == '82' and all(suite.attrib[name] == '0' for name in ('errors', 'failures', 'skipped'))
constructor = next(case.attrib for case in suite.findall('testcase') if case.attrib['name'] == 'test_collective_input_binds_both_role_constructors_and_refreshes_next_run')
frozen = {'task_id': 'D29-PLAN-KEY-REUSE', 'session_id': 'ses_e963098ecffebqCOtdysKGY6Xe',
    'sources': sources, 'original_setup': str(setup), 'setup_sha256': sha(setup),
    'protected_body': str(body), 'protected_body_sha256': sha(body),
    'fixture': str(fixture), 'fixture_metadata': fixture_metadata,
    'collective_original': str(collectives[0]), 'collective_sha256': sha(collectives[0]),
    'constructor_evidence': {'xml': str(xml), 'sha256': sha(xml), 'suite': suite.attrib, 'test': constructor},
    'input_hashes': {str(path): sha(path) for path in (
        Path('/data/ycfeng/tmp/d29-parent-position-main08/offline/command.json'),
        Path('/data/ycfeng/tmp/d29_main08_residual_repro/parent-layer60-source.json'),
        fixture / 'fixture_kernel_gap.csv')},
    'limits': 'No full-fixture content/training scan: retain fixture size/mtime and protected setup pickle SHA checks.'}
(ROOT / 'freeze.json').write_text(json.dumps(frozen, indent=2) + '\n')
print('PASS freeze', {label: len(value['hashes']) for label,value in sources.items()}, '82 existing tests; accepted protected body')
