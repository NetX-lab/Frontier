"""One-variable default model-contract proof control; no result/identity memo."""
import ast
from copy import deepcopy
from pathlib import Path
import sys

requested_label = sys.argv[1]
assert requested_label in ('baseline', 'default-contract')
PARENT = Path('/data/ycfeng/tmp/d29_immutable_plan_preparation')
COMPARE = PARENT / 'compare.py'
nodes = ast.parse(COMPARE.read_text()).body
cut = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign) and
           any(isinstance(target, ast.Attribute) and target.attr == 'cache_key' for target in node.targets))
setup_nodes = deepcopy(nodes[:cut])
for node in setup_nodes:
    if isinstance(node, ast.Assert) and ast.unparse(node.test).startswith('label in '):
        node.test = ast.parse("label in ('baseline', 'default-contract')", mode='eval').body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'driver' for target in node.targets):
        node.value = ast.parse("Path('/data/ycfeng/tmp/d29_immutable_plan_preparation/probe.py')", mode='eval').body
exec(compile(ast.fix_missing_locations(ast.Module(body=setup_nodes, type_ignores=[])), str(COMPARE), 'exec'), globals())

from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import ModelArchitectureProfile
import frontier.model_architectures as architecture_module

original_resolver = ModelArchitectureProfile._resolve_moe_layer_ids.__func__
default_getter = BaseModelConfig.get_moe_layer_ids
default_predicate = BaseModelConfig.is_moe_layer
source_tree = ast.parse(Path(architecture_module.__file__).read_text())
profile_class = next(node for node in source_tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelArchitectureProfile')
method = deepcopy(next(node for node in profile_class.body if isinstance(node, ast.FunctionDef) and node.name == '_resolve_moe_layer_ids'))
method.decorator_list = []
condition = next(node for node in ast.walk(method) if isinstance(node, ast.If) and node.lineno == 1062)
assert ast.unparse(condition.test) == 'callable(predicate)'
condition.test = ast.parse("callable(predicate) and not (getattr(getter, '__func__', None) is D29_DEFAULT_GETTER and getattr(predicate, '__func__', None) is D29_DEFAULT_PREDICATE)", mode='eval').body
module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
control_source = OUT / 'contract_control.py'
control_source.write_text(ast.unparse(module) + '\n')
control_globals = {**architecture_module.__dict__, 'D29_DEFAULT_GETTER': default_getter,
                   'D29_DEFAULT_PREDICATE': default_predicate}
exec(compile(module, str(control_source), 'exec'), control_globals)
if requested_label == 'default-contract':
    ModelArchitectureProfile._resolve_moe_layer_ids = classmethod(control_globals['_resolve_moe_layer_ids'])

# Both arms use the SAME prior derived-key-only control and fixed-byte adapter.
CollectivePlan.cache_key = serialized_key_once
exec(compile(ast.Module(body=nodes[cut + 1:], type_ignores=[]), str(COMPARE), 'exec'), globals())
pricing = result
pricing_out = OUT

# Reuse the already executed ten actual-wave replacement checks, without a new
# setup, model lifetime, or change to the single-variable pricing control.
live_nodes = ast.parse((PARENT / 'live_checks.py').read_text()).body
start = next(index for index, node in enumerate(live_nodes) if isinstance(node, ast.Assign) and
             any(isinstance(target, ast.Name) and target.id == 'checks' for target in node.targets))
OUT = ROOT / (requested_label + '-live')
OUT.mkdir(exist_ok=False)
Path.read_bytes = fixed_file_byte_adapter
exec(compile(ast.Module(body=live_nodes[start:], type_ignores=[]), str(PARENT / 'live_checks.py'), 'exec'), globals())
live = result

# Default data replacement errors and custom method replacement remain live.
from types import MethodType, SimpleNamespace
cfg = namespace['predictor']._model_config
active_resolver = ModelArchitectureProfile._resolve_moe_layer_ids
guards = []

def observation(function):
    try:
        value = function()
    except (ValueError, TypeError) as error:
        return {'error_type': type(error).__name__, 'error': str(error)}
    return {'value': list(value)}

def check_guard(name, config):
    expected = observation(lambda: original_resolver(ModelArchitectureProfile, config))
    actual = observation(lambda: active_resolver(config))
    assert actual == expected, (name, actual, expected)
    guards.append({'name': name, 'observation': actual})

raw_ids = cfg._moe_layer_ids_cache
for name, ids in (('default duplicate IDs', [0, 0]), ('default out-of-range ID', [cfg.num_layers]),
                  ('default noninteger ID', [True]), ('default valid changed IDs', [3])):
    cfg._moe_layer_ids_cache = ids
    check_guard(name, cfg)
cfg._moe_layer_ids_cache = raw_ids
custom_calls = []
prior_instance_predicate = cfg.__dict__.get('is_moe_layer')
def inconsistent_predicate(self, layer):
    custom_calls.append(layer)
    return False
cfg.is_moe_layer = MethodType(inconsistent_predicate, cfg)
check_guard('replaced predicate mismatch', cfg)
assert len(custom_calls) == 2 * cfg.num_layers
if prior_instance_predicate is None:
    del cfg.__dict__['is_moe_layer']
else:
    cfg.is_moe_layer = prior_instance_predicate
custom = SimpleNamespace(num_layers=4, get_moe_layer_ids=lambda: [1], is_moe_layer=lambda layer: layer == 2)
check_guard('custom getter/predicate mismatch', custom)

stats = pstats.Stats(str(pricing_out / 'warm.prof'))
def calls(name):
    return sum(values[1] for (_, _, function), values in stats.stats.items() if function == name)
assert calls('_get_on_demand_prediction') == 288
assert calls('_predict_step5_ffn_operator') == 144
assert calls('loads') == 1136 and calls('dumps') == 800
assert calls('_measurement_identity') == 240
assert calls('_resolve_moe_layer_ids') == 144
assert calls('is_moe_layer') == (16 if requested_label == 'default-contract' else 13264)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip()
assert before == state(fixture)
final = {'task_id': 'D29-IMMUTABLE-PLAN-PREPARATION-H3', 'session_id': 'ses_e963098ecffebqCOtdysKGY6Xe',
         'label': requested_label, 'pricing': pricing, 'live_checks': live,
         'guard_checks': guards, 'custom_predicate_scan_calls': len(custom_calls),
         'warm_counts': {name: calls(name) for name in ('loads', 'dumps', '_get_on_demand_prediction',
             '_predict_step5_ffn_operator', '_measurement_identity', '_resolve_moe_layer_ids',
             'is_moe_layer', 'get_moe_layer_ids', 'profiling_name', 'bind_operator_query')},
         'profiled_exclusive_total_s': stats.total_tt, 'no_new_cache': True,
         'source_clean': True, 'fixture_unchanged': True}
(ROOT / (requested_label + '-result.json')).write_text(json.dumps(final, indent=2) + '\n')
print(json.dumps({key: value for key, value in final.items() if key not in ('pricing', 'live_checks', 'guard_checks')}, indent=2))
