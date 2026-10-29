"""Retain measured attribution and the incompletely qualified single pair."""
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PAIR = ROOT.parent / 'h3-control01'
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes')
attribution = json.loads((ROOT / 'attribution.json').read_text())
prefix = ('timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G env '
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
          'PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B ')
commands = [
    {'command': prefix + str(ROOT / 'analyze.py') + ' > ' + str(ROOT / 'analyze-final.log') + ' 2>&1', 'exit': 0},
    {'command': prefix + str(PAIR / 'run.py') + ' baseline > ' + str(PAIR / 'baseline.log') + ' 2>&1',
     'exit': 'tool timeout at65000ms; process stopped', 'pricing_subphase': 'PASS',
     'post_pricing_real_replacement_checks': 'UNVERIFIED; no completion receipt'},
    {'command': prefix + str(PAIR / 'run_bounded.py') + ' default-contract > ' + str(PAIR / 'default-contract.log') + ' 2>&1',
     'exit': 124, 'pricing_subphase': 'PASS',
     'post_pricing_real_replacement_checks': 'UNVERIFIED; no completion receipt'},
    {'command': prefix + str(ROOT / 'verify_guards.py') + ' > ' + str(ROOT / 'verify_guards.log') + ' 2>&1', 'exit': 0},
    {'command': prefix + str(ROOT / 'collect.py') + ' > ' + str(ROOT / 'collect.log') + ' 2>&1', 'exit': 0},
]
pair = {}
for arm in ('baseline', 'default-contract'):
    result_path = PAIR / arm / 'result.json'
    result = json.loads(result_path.read_text())
    stats = pstats.Stats(str(PAIR / arm / 'warm.prof'))
    def count(name):
        return sum(values[1] for (_, _, function), values in stats.stats.items() if function == name)
    counts = {name: count(name) for name in ('loads', 'dumps', '_get_on_demand_prediction',
               '_predict_step5_ffn_operator', '_measurement_identity', '_step5_position_queries',
               '_step5_plan_candidates', '_native_measurement_binding', 'predict_collective_plan',
               '_resolve_moe_layer_ids', 'is_moe_layer', 'get_moe_layer_ids',
               '_registered_operator_matches', 'profiling_name', '_models_view_for_family')}
    assert counts['loads'] == 1136 and counts['dumps'] == 800
    assert counts['_get_on_demand_prediction'] == 288 and counts['_predict_step5_ffn_operator'] == 144
    assert counts['_measurement_identity'] == 240 and counts['_step5_position_queries'] == 16
    assert counts['predict_collective_plan'] == 64 and counts['_resolve_moe_layer_ids'] == 144
    payload_path = PAIR / arm / 'pricing_payload.json'
    payload = payload_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
    rows = json.loads(payload)
    assert sum(len(row['proofs']) for row in rows) == 144
    pair[arm] = {'result': result, 'counts': counts, 'profiled_exclusive_total_s': stats.total_tt,
                 'paths': {'result': str(result_path), 'profile': str(PAIR / arm / 'warm.prof')},
                 'hashes': {name: hashlib.sha256((PAIR / arm / name).read_bytes()).hexdigest()
                            for name in ('result.json', 'warm.prof', 'pricing_payload.json', 'ledger.json', 'contract_control.py')},
                 'payload_proofs': 144, 'retained_event_ledger_entries': 208}
for name in ('pricing_payload.json', 'ledger.json'):
    assert (PAIR / 'baseline' / name).read_bytes() == (PAIR / 'default-contract' / name).read_bytes()
assert pair['baseline']['result']['event_sha256'] == pair['default-contract']['result']['event_sha256']
base = pair['baseline']['result']['warm_median_s']
control = pair['default-contract']['result']['warm_median_s']
gain = {'absolute_s': base - control, 'relative_percent': 100 * (base - control) / base}
source_paths = ('frontier/model_architectures.py', 'frontier/config/model_config.py',
    'frontier/operators/binding.py', 'frontier/operators/spec.py',
    'frontier/execution_time_predictor/prediction_model_registry.py',
    'frontier/execution_time_predictor/prediction_model_identity.py',
    'frontier/execution_time_predictor/moe_routing_workload.py',
    'frontier/execution_time_predictor/moe_operator_times.py',
    'frontier/execution_time_predictor/sklearn_execution_time_predictor.py',
    'frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py',
    'frontier/cc_backend/collective_plan.py',
    'frontier/cc_backend/backends/collective_sim_cc_backend.py')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
data = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION-H3', 'session_id': 'ses_e963098ecffebqCOtdysKGY6Xe',
        'source': str(SOURCE), 'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        'source_hashes': {path: hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() for path in source_paths},
        'python': sys.version, 'commands': commands, 'old_profiles_first': attribution,
        'exclusive_attribution_method': 'Sum self time only; builtins self time partitioned by immediate caller; all buckets disjoint and sum to profile total',
        'pair': pair, 'preliminary_gain': gain, 'guard_verification': json.loads((ROOT / 'guard_verification.json').read_text()),
        'control_verdict': 'UNQUALIFIED candidate: pricing/count/equality PASS, config-only guards PASS, real estimator/row/identity/fresh-layer completion UNVERIFIED due60s harness deadlines',
        'fixture_fidelity': 'Original comparison writes result.json only after unchanged-input state assertions; appended harnesses lack final receipts',
        'check_progress': {'attribution': 'completed', 'single_pair_pricing': 'completed',
            'config_guards': 'completed', 'actual_replacement_acceptance': 'unfinished/deferred',
            'route': 'stop further runs; records only; no implementation allocation; verify after snapshot delivery'},
        'limits': ['Existing CPU fixture/cache-table reconstructed manager; not native branch/full constructor/fullCLI',
            'Same diagnostic single-role fixed-byte adapter and prior key-only control in both arms; not P/D/run binding',
            'Cold/setup/profiling/equality/guard work excluded from unprofiled pricing timing',
            'One useful single-variable pair only; two appended validation harnesses timed out',
            'No new identity/numeric/result/proof memo; no production/test/native-key/trace-encoding edits']}
(ROOT / 'evidence.json').write_text(json.dumps(data, indent=2) + '\n')
print('PASS existing profile attribution totals/caller counts; pair288 numeric/208 complete-output equality;7 guards')
print(json.dumps(gain))
print(data['control_verdict'])
for key in ('_registered_operator_matches', 'profiling_name', '_is_mixed_model'):
    print(key, [row for row in attribution['key-a']['dominant_callers'].values() if row['function'] == key])
