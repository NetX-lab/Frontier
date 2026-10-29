"""Consolidate only observed real-wave evidence; assert the one-variable control."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes')
arms = ('current-a3', 'key-a', 'current-b', 'key-b')
results = {arm: json.loads((ROOT / arm / 'result.json').read_text()) for arm in arms}
red = json.loads((ROOT / 'red/result.json').read_text())
live = {arm: json.loads((ROOT / arm / 'result.json').read_text()) for arm in ('live-current4', 'live-key')}
assert live['live-current4']['checks'] == live['live-key']['checks']
assert red['timings'][-1]['key_repetitions'] == 184
assert len(live['live-key']['checks']) == 10
for pair in (arms[:2], arms[2:]):
    current, key = (results[arm] for arm in pair)
    assert (ROOT / pair[0] / 'pricing_payload.json').read_bytes() == (ROOT / pair[1] / 'pricing_payload.json').read_bytes()
    assert (ROOT / pair[0] / 'ledger.json').read_bytes() == (ROOT / pair[1] / 'ledger.json').read_bytes()
    assert (ROOT / pair[0] / 'output-equality/op_traces.jsonl').read_bytes() == (ROOT / pair[1] / 'output-equality/op_traces.jsonl').read_bytes()
    def count(result, name):
        return sum(row['calls'] for row in result['profile'] if row['function'] == name)
    for name in ('_predict_step5_ffn_operator', '_get_on_demand_prediction',
                 '_step5_position_queries', '_step5_model_points', '_step5_plan_candidates',
                 '_native_measurement_binding', '_measurement_identity',
                 'step5_operator_binding', 'step5_layer_metadata', 'predict_collective_plan'):
        assert count(current, name) == count(key, name), name
    assert count(current, '_get_on_demand_prediction') == 288
    assert count(current, 'cache_key') == 192 and count(key, 'cache_key') == 0
    assert count(key, 'serialized_key_once') == 192
    assert current['constructors_warm'] == key['constructors_warm'] == {
        'CollectivePlan': 0, 'CollectiveTensor': 0, 'CollectivePlanCost': 64}
    assert all(row['fixed_byte_requests'] == 80 for arm in pair for row in results[arm]['timings'])
    assert all(results[arm]['fixture_unchanged'] for arm in pair)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == '56529fcd19e7e209e577c5aac3c581b4c5df8b6a'
assert not subprocess.check_output(['git', 'diff', '--name-only', '2f2d9339', '56529fcd', '--', 'frontier', 'tests'], cwd=SOURCE).strip()

commands = []
base = 'timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B '
for script, labels in (('probe.py', ('red',)), ('compare.py', arms), ('live_checks.py', ('live-current4', 'live-key'))):
    for label in labels:
        log = ROOT / ('red-executed.log' if label == 'red' else f'{label}.log')
        commands.append({'command': base + str(ROOT / script) + ' ' + label + ' > ' + str(log) + ' 2>&1',
                         'exit_code': 1 if label == 'red' else 0,
                         'verdict': 'EXPECTED_RED_REPEATED_KEY_WORK' if label == 'red' else 'PASS',
                         'log': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})

result = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION', 'session_id': 'ses_e963098ecffebqCOtdysKGY6Xe',
          'revision': '56529fcd19e7e209e577c5aac3c581b4c5df8b6a', 'code_revision': '2f2d9339d5bce600fbb0a3f78aebbcf6f71be696',
          'source': str(SOURCE), 'python_executable': sys.executable, 'python_version': sys.version,
          'platform': platform.platform(), 'conda': 'No conda invocation; existing frontier-py310 executable',
          'adapter': 'ONE SINGLE-ROLE DIAGNOSTIC FIXED-FILE-BYTE ADAPTER; 80 requests, zero warm physical reads; not run-wide P/D snapshot',
          'commands': commands, 'red': {key: value for key, value in red.items() if key != 'profile'},
          'verification_command': base + str(ROOT / 'verify.py') + ' > ' + str(ROOT / 'verify.log') + ' 2>&1',
          'controls': {arm: {key: value for key, value in value.items() if key != 'profile'} for arm, value in results.items()},
          'source_or_numeric_call_counts': {name: sum(row['calls'] for row in results['current-a3']['profile'] if row['function'] == name)
              for name in ('_predict_step5_ffn_operator', '_get_on_demand_prediction', '_native_measurement_binding',
                           '_measurement_identity', 'step5_operator_binding', 'step5_layer_metadata',
                           '_step5_position_queries', '_step5_model_points', '_step5_plan_candidates', 'predict_collective_plan')},
          'pairs': [{'current': a, 'key': b, 'saved_s': results[a]['warm_median_s'] - results[b]['warm_median_s'],
                     'saved_pct': 100 * (1 - results[b]['warm_median_s'] / results[a]['warm_median_s'])}
                    for a, b in (arms[:2], arms[2:])],
          'live_checks': live['live-key'], 'all_complete_bytes_equal': True,
          'source_clean': True, 'fixture_bytes_and_metadata_unchanged': True,
          'source_hashes': {path: hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() for path in (
              'frontier/cc_backend/collective_plan.py', 'frontier/cc_backend/backends/collective_sim_cc_backend.py',
              'frontier/execution_time_predictor/moe_routing_workload.py', 'frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py',
              'frontier/execution_time_predictor/moe_operator_times.py', 'frontier/entities/time_components.py')},
          'input_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
              Path('/data/ycfeng/tmp/d29_artifact_bytes_repair/verified03/candidate_setup.py'),
              Path('/data/ycfeng/tmp/d29-parent-position-main08/offline/command.json'),
              Path('/data/ycfeng/tmp/d29_main08_residual_repro/parent-layer60-source.json'),
              Path('/data/ycfeng/tmp/d29-parent-position-fixture03/fixture_collectives/h800/collective_local.json'))},
          'artifact_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
              *(ROOT / name for name in ('probe.py', 'compare.py', 'live_checks.py', 'verify.py')),
              *(ROOT / arm / name for arm in arms for name in ('result.json', 'warm.prof', 'pricing_payload.json', 'ledger.json')))},
          'harness_failures_not_correctness_red': [
              'red.log: user systemd scope unavailable; red-system.log: interactive auth required; sudo -n scope succeeded',
              'current-a.log: public communication getter deepcopies plans, so public identity assertion was invalid; compare retained plans instead',
              'current-a2.log: output came from later waves with newly generated IDs; compare identical first-wave IDs',
              'live-current.log: reconstructed manager still selected last-loaded KERNEL_ONLY family; registry replacement requires explicit CUDA_EVENT family',
              'live-current2/live-current3:6148 and7000 are measured source points, not missing-point negatives; compare fresh full routing source instead'],
          'verification_limits': 'CPU fixture collective branch and existing loaded exact tables, not native qualification/network evaluation; no CLI/training/GPU work; no snapshot constructor wiring; no production/test edits; no allocation or full-run scaling assumed'}
(ROOT / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'verdict': 'PASS', 'paired_gain': result['pairs'], 'numeric_lookups_per_wave': 288,
                  'live_checks_equal': 10, 'all_complete_bytes_equal': True, 'python': sys.version}, indent=2))
