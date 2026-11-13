"""Small config-only differential verification; no new pricing arm or cache."""
from pathlib import Path
import json
import os
import sys
from types import MethodType, SimpleNamespace

SOURCE = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes')
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT.parent / 'h3-control01/default-contract/cwd')
sys.path.insert(0, str(SOURCE))
from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import ModelArchitectureProfile
import frontier.model_architectures as architecture_module

cfg = BaseModelConfig.create_from_name('stepfun-ai/Step-5-Preview')
original = ModelArchitectureProfile._resolve_moe_layer_ids.__func__
ns = {**architecture_module.__dict__, 'D29_DEFAULT_GETTER': BaseModelConfig.get_moe_layer_ids,
      'D29_DEFAULT_PREDICATE': BaseModelConfig.is_moe_layer}
path = ROOT.parent / 'h3-control01/default-contract/contract_control.py'
exec(compile(path.read_text(), str(path), 'exec'), ns)
control = ns['_resolve_moe_layer_ids']
checks = []

def observe(function, obj):
    try:
        return {'value': list(function(ModelArchitectureProfile, obj))}
    except (ValueError, TypeError) as error:
        return {'error_type': type(error).__name__, 'error': str(error)}

def check(name, obj):
    before, after = observe(original, obj), observe(control, obj)
    assert before == after, (name, before, after)
    checks.append({'name': name, 'observation': after})

check('official config', cfg)
raw_ids = cfg._moe_layer_ids_cache
for name, ids in (('duplicate IDs', [0, 0]), ('out-of-range ID', [cfg.num_layers]),
                  ('noninteger ID', [True]), ('valid changed IDs', [3])):
    cfg._moe_layer_ids_cache = ids
    check(name, cfg)
cfg._moe_layer_ids_cache = raw_ids
calls = []
def changed_predicate(self, layer):
    calls.append(layer)
    return False
cfg.is_moe_layer = MethodType(changed_predicate, cfg)
check('replaced predicate mismatch', cfg)
assert len(calls) == 2 * cfg.num_layers
del cfg.__dict__['is_moe_layer']
check('custom getter/predicate mismatch', SimpleNamespace(num_layers=4,
    get_moe_layer_ids=lambda: [1], is_moe_layer=lambda layer: layer == 2))
data = {'verdict': 'PASS', 'checks': checks, 'custom_predicate_calls': len(calls),
        'limits': 'Config-only guard verification; does not substitute for unfinished estimator/row/identity wave checks'}
(ROOT / 'guard_verification.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data, indent=2))
