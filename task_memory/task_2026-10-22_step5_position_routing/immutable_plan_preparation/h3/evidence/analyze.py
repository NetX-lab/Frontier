"""Exclusive attribution of the two existing derived-key warm profiles."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pstats

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SOURCE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes')

def reference(key):
    path, line, name = key
    return {'file': path, 'line': line, 'function': name}

def category(path, name):
    if '/json/' in path or name in ('<built-in method _json.scanstring>', '<built-in method _json.encode_basestring_ascii>'):
        return 'JSON codec'
    if path.endswith('/copy.py'):
        return 'Snapshot copy'
    if path.endswith('sklearn_execution_time_predictor.py') and name == '_get_on_demand_prediction':
        return 'Current duration/count numeric lookup'
    if path.endswith('prediction_model_registry.py') or path.endswith('sklearn_execution_time_predictor.py'):
        return 'Model registry/admission/lookup'
    if path.endswith('moe_routing_workload.py'):
        return 'Actual-point/plan selection'
    if '/cc_backend/' in path:
        return 'Collective pricing/validation'
    if '/entities/' in path:
        return 'Entity construction/validation'
    if path.endswith('model_config.py') or path.endswith('model_architectures.py'):
        return 'Model configuration/layer consistency'
    if '/operators/' in path or path.endswith('prediction_model_identity.py'):
        return 'Operator family/typed-contract derivation'
    if path.endswith('moe_operator_times.py'):
        return 'Actual point/family/operand admission'
    if path.endswith('moe_load_imbalance.py') or '/profiling/moe/' in path:
        return 'Actual source receipt/geometry'
    if path.endswith('sklearn_moe_execution_time_predictor.py') or path.endswith('step5_cost.py'):
        return 'Measured FFN/fresh proof work'
    if '/scheduler/' in path:
        return 'Wave/lane construction'
    if path.startswith(str(SOURCE)):
        return 'Other Frontier domain work'
    if path.endswith('/pathlib.py') or path.startswith(str(PARENT)):
        return 'Diagnostic adapter/filesystem/harness'
    return 'Other Python/builtins'

results = {}
for arm in ('key-a', 'key-b'):
    path = PARENT / arm / 'warm.prof'
    stats = pstats.Stats(str(path))
    groups = defaultdict(float)
    entries = []
    for key, (cc, nc, tt, ct, callers) in stats.stats.items():
        if key[0] == '~' and callers:
            assert abs(sum(value[2] for value in callers.values()) - tt) < 1e-9
            for caller, value in callers.items():
                groups[category(caller[0], caller[2])] += value[2]
        else:
            groups[category(key[0], key[2])] += tt
        entries.append({**reference(key), 'primitive_calls': cc, 'calls': nc, 'exclusive_s': tt,
                        'cumulative_s_not_additive': ct, 'category': category(key[0], key[2])})
    entries.sort(key=lambda row: row['exclusive_s'], reverse=True)
    observed = {}
    for function in ('loads', 'dumps', 'canonical_json'):
        matches = [(key, value) for key, value in stats.stats.items() if key[2] == function and
                   ('/json/' in key[0] if function != 'canonical_json' else key[0].endswith('runtime_identity.py'))]
        assert len(matches) == 1, matches
        key, (cc, nc, tt, ct, callers) = matches[0]
        caller_rows = []
        for caller, values in callers.items():
            ccc, cnc, ctt, cct = values
            caller_rows.append({**reference(caller), 'calls': cnc, 'exclusive_s_inside_target': ctt,
                                'target_subtree_s_not_additive': cct})
        caller_rows.sort(key=lambda row: row['calls'], reverse=True)
        assert sum(row['calls'] for row in caller_rows) == nc
        observed[function] = {'calls': nc, 'exclusive_s': tt,
                              'subtree_s_not_additive': ct, 'callers': caller_rows}
    dominant_callers = {}
    for key, (cc, nc, tt, ct, callers) in stats.stats.items():
        if key[2] in ('_resolve_moe_layer_ids', '_is_mixed_model', 'is_moe_layer',
                      'get_moe_layer_ids', '_registered_operator_matches', 'profiling_name',
                      '_resolve_profile_typed_family_for_query', '_models_view_for_family',
                      '_serialize_selected_layer_cache_identity', '_measurement_identity',
                      '_get_on_demand_prediction', '_native_measurement_binding'):
            dominant_callers[str(key)] = {**reference(key), 'calls': nc, 'exclusive_s': tt,
                'callers': [{**reference(caller), 'calls': values[1], 'exclusive_s_inside_target': values[2]}
                            for caller, values in callers.items()]}
    assert abs(sum(groups.values()) - stats.total_tt) < 1e-9
    results[arm] = {'input_profile': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'profiled_exclusive_total_s': stats.total_tt,
                    'exclusive_categories_s': dict(sorted(groups.items(), key=lambda item: item[1], reverse=True)),
                    'json_callers': observed, 'exclusive_functions': entries}
    results[arm]['dominant_callers'] = dominant_callers
(ROOT / 'attribution.json').write_text(json.dumps(results, indent=2) + '\n')
for arm, result in results.items():
    print(arm, 'TOTAL EXCLUSIVE', result['profiled_exclusive_total_s'])
    print('CATEGORIES', result['exclusive_categories_s'])
    for name, entry in result['json_callers'].items():
        print(name, entry['calls'], [(row['file'].split('/')[-1], row['line'], row['function'], row['calls']) for row in entry['callers']])
    print('TOP EXCLUSIVE', [(row['file'].split('/')[-1], row['line'], row['function'], row['calls'], row['exclusive_s']) for row in result['exclusive_functions'][:22]])
